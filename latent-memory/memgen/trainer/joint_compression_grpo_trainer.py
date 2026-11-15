import torch
from transformers import TrainerCallback
from .trigger_grpo_trainer import TriggerGRPOTrainer
from memgen.model.memory_compressor import deterministic_memory


class CompressorGradientCallback(TrainerCallback):

    def __init__(self, trainer):
        self.trainer = trainer
        self._handles = []
        self._gradients = {}

    def on_train_begin(self, args, state, control, **kwargs):
        model = self.trainer.accelerator.unwrap_model(self.trainer.model)
        self.on_train_end(args, state, control)
        for role in ("prompt", "inference"):
            module = getattr(model, role + "_compressor", None)
            if module is None:
                continue
            self._gradients[role] = {}
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue

                def hook(gradient, role=role, name=name):
                    bank = self._gradients[role]
                    detached = gradient.detach().float()
                    bank[name] = (
                        bank[name] + detached if name in bank else detached.clone()
                    )

                self._handles.append(p.register_hook(hook))

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        for role, bank in self._gradients.items():
            norm = (
                torch.stack([g.square().sum() for g in bank.values()])
                .sum()
                .sqrt()
                .item()
                if bank
                else 0.0
            )
            self.trainer._metrics["train"][
                f"compressor/{role}_local_backward_grad_norm"
            ].append(norm)
            self.trainer._metrics["train"][f"compressor/{role}_grad_tensors"].append(
                float(len(bank))
            )
            bank.clear()

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._gradients.clear()


class JointCompressionGRPOTrainer(TriggerGRPOTrainer):

    def __init__(self, *args, **kwargs):
        model = kwargs.get("model", args[0] if args else None)
        config = kwargs.get("args")
        if config is not None:
            for flag in ("use_vllm", "use_liger_loss", "gradient_checkpointing"):
                if getattr(config, flag, False):
                    raise ValueError(f"Joint compression does not support {flag}")
        model.configure_joint_trainables()
        super().__init__(*args, **kwargs)
        self.generation_config.top_k = 0
        self.generation_config.top_p = 1.0
        if self.is_fsdp_enabled:
            raise ValueError("Joint compression does not support FSDP")
        plugin = self.accelerator.state.deepspeed_plugin
        if plugin and plugin.zero_stage == 3:
            raise ValueError("Joint compression does not support ZeRO-3")
        self.add_callback(CompressorGradientCallback(self))

    def _get_per_token_logps(self, model, input_ids, attention_mask, augmentation_mask):
        component = getattr(model, "trigger", model)
        prompt_length = input_ids.size(1) - augmentation_mask.size(1)
        positions = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        with deterministic_memory(component):
            logits = component(input_ids, attention_mask, positions)[
                :, prompt_length - 1 : -1
            ]
        valid = augmentation_mask.ne(-100)
        logps = (
            logits.float() / float(self.generation_config.temperature or 1.0)
        ).log_softmax(-1)
        return (
            logps.gather(-1, augmentation_mask.clamp_min(0)[..., None]).squeeze(-1)
            * valid
        )

    def _generate_and_score_completions(self, inputs):
        model = self.accelerator.unwrap_model(self.model)
        model._compression_rollout_records = []
        try:
            with deterministic_memory(model):
                batch = super()._generate_and_score_completions(inputs)
            records = model._compression_rollout_records
            if len(records) != batch["prompt_ids"].size(0):
                raise RuntimeError("Rollout record count does not match training batch")
            size = batch["completion_ids"].size(1)
            for index, key in [(0, "old_per_token_logps"), (1, "old_answer_logps")]:
                batch[key] = torch.cat(
                    [
                        torch.nn.functional.pad(
                            record[index], (0, size - record[index].size(1))
                        )
                        for record in records
                    ]
                )
            return batch
        finally:
            model._compression_rollout_records = None

    def _compute_loss(self, model, inputs):
        ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
        mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
        out = model(
            input_ids=ids,
            attention_mask=mask,
            joint_compression_batch=inputs,
            temperature=self.generation_config.temperature,
        )
        advantages = inputs["advantages"][:, None]

        def clipped(current, old):
            if old is None:
                old = current.detach()
            ratio = (current - old).exp()
            clipped_ratio = ratio.clamp(1 - self.epsilon_low, 1 + self.epsilon_high)
            if getattr(self.args, "delta", None) is not None:
                ratio = ratio.clamp(max=self.args.delta)
            return -torch.minimum(ratio * advantages, clipped_ratio * advantages)

        trigger_loss = clipped(out["trigger_logps"], inputs.get("old_per_token_logps"))
        if self.beta:
            delta = inputs["ref_per_token_logps"] - out["trigger_logps"]
            trigger_loss = trigger_loss + self.beta * (delta.exp() - delta - 1)
        valid = inputs["augmentation_mask"].ne(-100) * inputs["completion_mask"]
        if self.loss_type == "bnpo":
            loss = (trigger_loss * valid).sum() / valid.sum().clamp_min(1)
        elif self.loss_type == "grpo":
            loss = ((trigger_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
        elif self.loss_type == "dr_grpo":
            loss = (trigger_loss * valid).sum() / (
                valid.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        answer_loss = clipped(out["answer_logps"], inputs["old_answer_logps"])
        am = out["answer_mask"]
        loss = loss + ((answer_loss * am).sum(-1) / am.sum(-1).clamp_min(1)).mean()
        mode = "train" if model.training else "eval"
        statistics = (
            torch.stack(
                [
                    out["counts"][:, 0].float().mean(),
                    out["counts"][:, 1].float().mean(),
                    out["entropy"][:, 0].mean(),
                    out["entropy"][:, 1].mean(),
                    out["replayed"].float().mean(),
                    inputs["advantages"].eq(0).float().mean(),
                ]
            )
            .detach()
            .cpu()
            .tolist()
        )
        for name, value in zip(
            (
                "prompt_calls",
                "inference_calls",
                "prompt_weight_entropy",
                "inference_weight_entropy",
                "replay_fraction",
                "zero_advantage_fraction",
            ),
            statistics,
        ):
            self._metrics[mode]["compressor/" + name].append(value)
        return loss
