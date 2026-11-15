from .memory_compressor import component_adapter, hidden_state_forward_kwargs
from peft import LoraConfig, PeftModel
import random
import re
import logging
import torch
import torch.nn as nn


class MemGenWeaver(nn.Module):
    adapter_name = "weaver"

    def __init__(
        self,
        model: PeftModel,
        prompt_latents_len: int,
        inference_latents_len: int,
        memory_layer: int | str = -1,
        random_memory_layers: list[int | str] | None = None,
    ):
        super().__init__()
        self.model = model
        self.memory_layer = memory_layer
        self.random_memory_layers = random_memory_layers
        if random_memory_layers is not None:
            logging.info(
                f"MemGenWeaver: random_memory_layers enabled, candidates={random_memory_layers}. Will randomly sample one layer per forward during training."
            )
        self.prompt_query_latents = nn.Parameter(
            torch.randn(prompt_latents_len, model.base_model.config.hidden_size),
            requires_grad=True,
        )
        self.inference_query_latents = nn.Parameter(
            torch.randn(inference_latents_len, model.base_model.config.hidden_size),
            requires_grad=True,
        )

    @property
    def prompt_latents_num(self) -> int:
        return self.prompt_query_latents.size(0)

    @property
    def inference_latents_num(self) -> int:
        return self.inference_query_latents.size(0)

    @property
    def device(self):
        assert self.prompt_query_latents.device == self.inference_query_latents.device
        return self.prompt_query_latents.device

    def _augment(
        self,
        latents: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        memory_layer: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = attention_mask.shape[0]
        latents_num = latents.size(0)
        latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)
        inputs_embeds = torch.cat([inputs_embeds, latents], dim=1)
        latents_mask = torch.ones(
            latents.shape[:-1], dtype=attention_mask.dtype, device=attention_mask.device
        )
        attention_mask = torch.cat([attention_mask, latents_mask], dim=1)
        last_position_ids = position_ids.max(dim=1)[0]
        latents_relative_positions = torch.arange(
            latents_num, device=attention_mask.device
        )
        latents_position_ids = (
            last_position_ids.unsqueeze(1) + latents_relative_positions + 1
        )
        position_ids = torch.cat(
            [position_ids.long(), latents_position_ids.long()], dim=1
        )
        assert inputs_embeds.shape[:2] == attention_mask.shape == position_ids.shape
        with component_adapter(self):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                **(
                    hidden_state_forward_kwargs(self.model)
                    if getattr(self, "compress_latent_memory", False)
                    else {}
                ),
            )
        if memory_layer is not None:
            effective_layer = memory_layer
        else:
            effective_layer = self._get_effective_memory_layer(self.memory_layer)
        hidden_states = outputs.hidden_states[
            self._resolve_memory_layer(outputs.hidden_states, effective_layer)
        ]
        latents_hidden_states = hidden_states[:, -latents_num:, :]
        return (latents_hidden_states, latents_mask, latents_position_ids)

    def _get_effective_memory_layer(self, default_layer: int | str) -> int | str:
        if (
            self.training
            and self.random_memory_layers is not None
            and (len(self.random_memory_layers) > 0)
        ):
            chosen = random.choice(self.random_memory_layers)
            return chosen
        return default_layer

    @staticmethod
    def _resolve_memory_layer(
        hidden_states: tuple[torch.Tensor, ...], memory_layer: int | str = -1
    ) -> int:
        hs_len = len(hidden_states)
        if hs_len == 0:
            raise ValueError(
                "outputs.hidden_states is empty (output_hidden_states=True expected)."
            )
        if isinstance(memory_layer, int):
            idx = memory_layer
            if not -hs_len <= idx < hs_len:
                raise IndexError(
                    f"memory_layer={memory_layer} out of range for hidden_states (len={hs_len})."
                )
            return idx
        if isinstance(memory_layer, str):
            lower = memory_layer.lower().strip()
            if lower == "middle":
                if hs_len == 1:
                    return 0
                num_layers = hs_len - 1
                middle_layer_num = num_layers // 2
                return 1 + middle_layer_num
            m = re.match("^l/(\\d+)(?:\\*(\\d+))?$", lower)
            if m:
                denom = int(m.group(1))
                if denom <= 0:
                    raise ValueError(
                        f"Invalid memory_layer: denominator must be positive, got {denom} in {memory_layer!r}"
                    )
                numer = int(m.group(2)) if m.group(2) else 1
                if numer < 0:
                    raise ValueError(
                        f"Invalid memory_layer: numerator must be non-negative, got {numer} in {memory_layer!r}"
                    )
                if hs_len == 1:
                    return 0
                num_layers = hs_len - 1
                fraction = numer / denom
                block_idx = int(num_layers * fraction)
                block_idx = max(0, min(block_idx, num_layers - 1))
                return 1 + block_idx
            raise ValueError(
                f"Unsupported memory_layer string: {memory_layer!r}. Use an int, 'middle', or 'l/<denom>' / 'l/<denom>*<numer>' (e.g. 'l/4', 'l/4*3', 'l/2')."
            )
        raise TypeError(
            f"memory_layer must be int or str, got: {type(memory_layer).__name__}"
        )

    def augment_prompt(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        memory_layer: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.prompt_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            memory_layer=memory_layer,
        )

    def augment_inference(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        memory_layer: int | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._augment(
            latents=self.inference_query_latents,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            memory_layer=memory_layer,
        )
