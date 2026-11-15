from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.models import unwrap_model_for_generation, create_reference_model
from trl.trainer.utils import selective_log_softmax
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback
from peft import PeftConfig
from typing import Union, Callable, Optional, Any
from contextlib import nullcontext
from collections import Counter
import logging
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, Sampler
from accelerate.utils import gather_object
from trl.trainer.grpo_trainer import RepeatSampler
from interactions.base_interaction import InteractionDataProto
from interactions.tensor_utils import TensorHelper, TensorConfig
from memgen.trainer.utils import nanstd, nanmax, nanmin, generate_position_ids
from memgen.model.modeling_memgen import MemGenModel

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class TriggerGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        model: MemGenModel,
        processing_class: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        args: Optional[GRPOConfig] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional[PeftConfig] = None,
    ):
        if args is not None and args.use_vllm:
            raise ValueError("Latent memory requires native model generation")
        self.model_accepts_loss_kwargs = False
        reference_beta = getattr(args, "beta", 0.0)
        if args is not None:
            args.beta = 0.0
        super().__init__(
            model=model,
            args=args,
            reward_funcs=reward_funcs,
            reward_processing_classes=reward_processing_classes,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )
        self.beta = reference_beta
        if args is not None:
            args.beta = reference_beta
        if self.beta:
            ref_model = create_reference_model(model.trigger)
            self.ref_model = self.accelerator.prepare_model(
                ref_model, evaluation_mode=True
            )
        else:
            self.ref_model = None
        self.tensor_fn = TensorHelper(
            TensorConfig(
                pad_token_id=self.processing_class.pad_token_id,
                max_prompt_length=self.max_prompt_length,
                max_obs_length=None,
                max_start_length=None,
            )
        )

    def _set_signature_columns_if_needed(self):
        pass

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.args.per_device_eval_batch_size,
            seed=self.args.seed,
        )

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        augmentation_mask: torch.LongTensor,
    ) -> torch.Tensor:
        prompt_len = attention_mask.size(1) - augmentation_mask.size(1)
        assert input_ids.shape == attention_mask.shape
        position_ids = generate_position_ids(attention_mask)
        forward_kwargs = {}
        if isinstance(self.accelerator.unwrap_model(model), MemGenModel):
            forward_kwargs["trigger_only"] = True
        augmentation_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **forward_kwargs,
        )
        clipped_logits = augmentation_logits[:, prompt_len - 1 : -1]
        assert clipped_logits.shape[:-1] == augmentation_mask.shape
        temp_mask = augmentation_mask.clone()
        augmentation_valid_mask = (temp_mask == -100).clone()
        temp_mask[augmentation_valid_mask] = 0
        logps = selective_log_softmax(clipped_logits, temp_mask)
        logps[augmentation_valid_mask] = 0
        return logps

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        prompts = [x["prompt"] for x in inputs]
        invalid_augmentation_id = -100
        batch_gen_keys = []
        if "prompt" in inputs[0]:
            batch_gen_keys.append("prompt")
        if "tools_kwargs" in inputs[0]:
            batch_gen_keys.append("tools_kwargs")
        if "interaction_kwargs" in inputs[0]:
            batch_gen_keys.append("interaction_kwargs")
        if "agent_name" in inputs[0]:
            batch_gen_keys.append("agent_name")
        if "env" in inputs[0]:
            batch_gen_keys.append("env")
        gen_batch = InteractionDataProto()
        for key in batch_gen_keys:
            gen_batch.no_tensor_batch[key] = [x[key] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        gen_batch.batch["input_ids"] = prompt_ids.to(device)
        gen_batch.batch["attention_mask"] = prompt_mask.to(device)
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as unwrapped_model:
            with (
                FSDP.summon_full_params(self.model_wrapped, recurse=False)
                if self.is_fsdp_enabled
                else nullcontext()
            ):
                prompt_ids = gen_batch.batch["input_ids"]
                prompt_mask = gen_batch.batch["attention_mask"]
                generate_kwargs = {}
                use_sub_batch = getattr(self, "generate_sub_batch", False)
                total_samples = prompt_ids.size(0)
                prompt_length = prompt_ids.size(1)
                if use_sub_batch and total_samples > self.num_generations:
                    gen_sub_batch = self.num_generations
                    sub_completion_list = []
                    sub_aug_mask_list = []
                    for sb_start in range(0, total_samples, gen_sub_batch):
                        sb_end = min(sb_start + gen_sub_batch, total_samples)
                        sb_kwargs = dict(generate_kwargs)
                        sb_ids, sb_aug = unwrapped_model.generate(
                            prompt_ids[sb_start:sb_end],
                            prompt_mask[sb_start:sb_end],
                            generation_config=self.generation_config,
                            return_augmentation_mask=True,
                            **sb_kwargs,
                        )
                        sub_completion_list.append(sb_ids[:, prompt_length:])
                        sub_aug_mask_list.append(sb_aug)
                    max_comp_len = max((t.size(1) for t in sub_completion_list))
                    pad_id = self.processing_class.pad_token_id
                    for i in range(len(sub_completion_list)):
                        pad_n = max_comp_len - sub_completion_list[i].size(1)
                        if pad_n > 0:
                            sub_completion_list[i] = torch.nn.functional.pad(
                                sub_completion_list[i], (0, pad_n), value=pad_id
                            )
                            sub_aug_mask_list[i] = torch.nn.functional.pad(
                                sub_aug_mask_list[i], (0, pad_n), value=-100
                            )
                    completion_ids = torch.cat(sub_completion_list, dim=0)
                    augmentation_mask = torch.cat(sub_aug_mask_list, dim=0)
                    prompt_completion_ids = torch.cat(
                        [prompt_ids, completion_ids], dim=1
                    )
                    prompt_ids = prompt_completion_ids[:, :prompt_length]
                else:
                    prompt_completion_ids, augmentation_mask = unwrapped_model.generate(
                        prompt_ids,
                        prompt_mask,
                        generation_config=self.generation_config,
                        return_augmentation_mask=True,
                        **generate_kwargs,
                    )
                    prompt_ids = prompt_completion_ids[:, :prompt_length]
                    completion_ids = prompt_completion_ids[:, prompt_length:]
                assert completion_ids.shape == augmentation_mask.shape
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        completion_ids = torch.where(
            completion_mask.bool(),
            completion_ids,
            torch.full_like(completion_ids, self.processing_class.eos_token_id),
        )
        augmentation_valid_mask = completion_mask * (
            augmentation_mask != invalid_augmentation_id
        )
        augmentation_mask = torch.where(
            augmentation_valid_mask.bool(),
            augmentation_mask,
            torch.full_like(augmentation_mask, invalid_augmentation_id),
        )
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        with torch.no_grad():
            if (
                self.num_iterations > 1
                or self.args.steps_per_generation
                > self.args.gradient_accumulation_steps
            ):
                old_per_token_logps = self._get_per_token_logps(
                    self.model.trigger,
                    prompt_completion_ids,
                    attention_mask,
                    augmentation_mask,
                )
            else:
                old_per_token_logps = None
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        augmentation_mask,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model.trigger,
                            prompt_completion_ids,
                            attention_mask,
                            augmentation_mask,
                        )
            else:
                ref_per_token_logps = None
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        completions = completions_text
        for i in range(len(inputs)):
            inputs[i]["augmentation_mask"] = augmentation_mask[i]
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        effective_num_gen = (
            self.num_generations
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )
        if rewards.numel() % effective_num_gen:
            raise ValueError("Incomplete reward group")
        mean_grouped_rewards = rewards.view(-1, effective_num_gen).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, effective_num_gen).std(dim=1)
        std_grouped_rewards = torch.nan_to_num(std_grouped_rewards, nan=0.0)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            effective_num_gen, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            effective_num_gen, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 0.0001)
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        completion_lengths = completion_mask.sum(1)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )
        augmentation_lengths = (augmentation_mask == 1).sum(dim=1)
        agg_augmentation_lengths = self.accelerator.gather(augmentation_lengths)
        self._metrics[mode]["augmentations/mean_length"].append(
            agg_augmentation_lengths.float().mean().item()
        )
        self._metrics[mode]["augmentations/min_length"].append(
            agg_augmentation_lengths.float().min().item()
        )
        self._metrics[mode]["augmentations/max_length"].append(
            agg_augmentation_lengths.float().max().item()
        )
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
            agg_completion_lengths
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_lengths) == 0:
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "augmentation_mask": augmentation_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def _compute_loss(self, model, inputs):
        prompt_ids, prompt_mask = (inputs["prompt_ids"], inputs["prompt_mask"])
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        augmentation_mask = inputs["augmentation_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, augmentation_mask
        )
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach()
            if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        augmentation_valid_mask = augmentation_mask != -100
        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * augmentation_valid_mask).sum(-1)
                / augmentation_valid_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * augmentation_valid_mask
            ).sum() / augmentation_valid_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * augmentation_valid_mask).sum() / (
                augmentation_valid_mask.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            mean_kl = (
                per_token_kl * augmentation_valid_mask
            ).sum() / augmentation_valid_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = (
            is_low_clipped * augmentation_valid_mask
        ).sum() / augmentation_valid_mask.sum()
        high_clip = (
            is_high_clipped * augmentation_valid_mask
        ).sum() / augmentation_valid_mask.sum()
        clip_ratio = (
            is_region_clipped * augmentation_valid_mask
        ).sum() / augmentation_valid_mask.sum()
        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()
        )
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()
        )
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()
        )
        return loss
