from contextlib import contextmanager
import inspect
import torch
from torch import nn
from torch.func import functional_call


class ParameterGradientScale:

    def with_gradient_scale(self, inputs, scale):
        params = {
            name: p.detach() + (p - p.detach()) * scale
            for name, p in self.named_parameters()
        }
        return functional_call(self, params, (inputs,))


class GroupedMemoryCompressor(ParameterGradientScale, nn.Module):

    def __init__(self, hidden_size, memory_length, feature_size=64, group_size=4):
        super().__init__()
        if type(group_size) is not int or group_size < 1 or memory_length % group_size:
            raise ValueError("group_size must divide memory_length")
        self.memory_length, self.group_size = (memory_length, group_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.features = nn.Sequential(nn.Linear(hidden_size, feature_size), nn.Tanh())
        self.heads = nn.ModuleList(
            (
                nn.Linear(feature_size * group_size, group_size)
                for _ in range(memory_length // group_size)
            )
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, inputs):
        if inputs.size(-2) != self.memory_length:
            raise ValueError("Unexpected memory length")
        batch, length, hidden = inputs.shape
        features = self.features(self.norm(inputs)).reshape(batch, len(self.heads), -1)
        scores = torch.stack(
            [head(features[:, i]) for i, head in enumerate(self.heads)], 1
        )
        weights = scores.float().softmax(-1).to(inputs.dtype)
        values = inputs.reshape(batch, len(self.heads), self.group_size, hidden)
        return ((values * weights.unsqueeze(-1)).sum(-2), weights)


def validate_compressor_checkpoint(saved, config):
    if not saved.get("compress_latent_memory", False):
        return
    if not getattr(config, "compress_latent_memory", False):
        raise ValueError("A compressed checkpoint requires compress_latent_memory=True")
    for key, default in [("compressor_group_size", 4), ("compressor_hidden_size", 64)]:
        if saved.get(key, default) != getattr(config, key, default):
            raise ValueError(
                f"{key} mismatch: checkpoint={saved.get(key, default)}, model={getattr(config, key, default)}"
            )


def hidden_state_forward_kwargs(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    kwargs = {"use_cache": False}
    parameters = inspect.signature(base.forward).parameters
    for key in ("logits_to_keep", "num_logits_to_keep"):
        if key in parameters:
            kwargs[key] = 1
            break
    return kwargs


def finish_joint_adapter(model):
    flags = [(p, p.requires_grad) for p in model.parameters()]
    model.base_model.disable_adapter_layers()
    for parameter, enabled in flags:
        parameter.requires_grad_(enabled)


@contextmanager
def component_adapter(component):
    model = component.model
    if not getattr(component, "isolate_adapters", True):
        model.set_adapter(component.adapter_name)
        yield
        return
    flags = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.set_adapter(component.adapter_name)
        model.base_model.enable_adapter_layers()
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
        yield
    finally:
        model.base_model.disable_adapter_layers()
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)


@contextmanager
def deterministic_memory(model):
    states = [
        (module, module.training) for module in model.modules() if module is not model
    ]
    try:
        for module, _ in states:
            module.training = False
        yield
    finally:
        for module, training in states:
            module.training = training
