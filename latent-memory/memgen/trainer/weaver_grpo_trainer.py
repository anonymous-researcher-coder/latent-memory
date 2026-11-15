import copy
from contextlib import nullcontext
from typing import Any, Callable, Optional, Union
import torch
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
    set_seed,
)
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import (
    is_datasets_available,
    is_flash_attn_2_available,
    is_peft_available,
    is_rich_available,
)
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import selective_log_softmax
from trl.data_utils import maybe_apply_chat_template, is_conversational
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    prepare_fsdp,
    unwrap_model_for_generation,
)

if is_peft_available():
    from peft import PeftConfig, get_peft_model
if is_wandb_available():
    import wandb
from interactions.base_interaction import InteractionManager, InteractionDataProto
from data.base_env import StaticEnv
from .utils import nanstd, nanmax, nanmin
from ..model.modeling_memgen import MemGenModel

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class WeaverGRPOTrainer(GRPOTrainer):

    def __init__(
        self,
        model: MemGenModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, ProcessorMixin]
        ] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        env_class=None,
        env_main_config=None,
        generation_manager: InteractionManager = None,
    ):
        if args is not None and args.use_vllm:
            raise ValueError("Latent memory requires native model generation")
        super().__init__(
            model,
            reward_funcs,
            args,
            train_dataset,
            eval_dataset,
            processing_class,
            reward_processing_classes,
            callbacks,
            optimizers,
            peft_config,
        )
        self.env_class = env_class
        self.env_main_config = env_main_config
        self.generation_manager = generation_manager
        assert self.max_prompt_length == generation_manager.config.max_start_length
        assert (
            self.max_completion_length == generation_manager.config.max_response_length
        )
        assert self.temperature == generation_manager.config.temperature

    def _get_per_token_logps(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        logits_to_keep: int,
        batch_size: int = None,
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        supervise_masks = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]
            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "labels": labels,
            }
            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1
            outputs = model(**model_inputs)
            logits = outputs.logits
            labels = outputs.supervised_labels
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)
            labels = labels[:, -logits_to_keep:]
            mask = (labels != -100).long()
            supervise_masks.append(mask)
        logps = torch.cat(all_logps, dim=0)
        masks = torch.cat(supervise_masks, dim=0)
        return (logps, masks)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_gen_keys = []
        if "prompt" in inputs[0]:
            batch_gen_keys.append("prompt")
        if "tools_kwargs" in inputs[0]:
            batch_gen_keys.append("tools_kwargs")
        if "interaction_kwargs" in inputs[0]:
            batch_gen_keys.append("interaction_kwargs")
        if "agent_name" in inputs[0]:
            batch_gen_keys.append("agent_name")
        gen_batch = InteractionDataProto()
        for key in batch_gen_keys:
            gen_batch.no_tensor_batch[key] = [x[key] for x in inputs]
        if issubclass(self.env_class, StaticEnv):
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
            prompts, prompt_mask = (
                prompt_inputs["input_ids"].to(device),
                prompt_inputs["attention_mask"].to(device),
            )
            if self.max_prompt_length is not None:
                prompts = prompts[:, -self.max_prompt_length :]
                prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            gen_batch.batch["input_ids"] = prompts
            gen_batch.batch["attention_mask"] = prompt_mask
        else:
            raise ValueError("Unsupported environment type")
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
                self.generation_manager.actor_rollout_wg = unwrapped_model
                final_gen_batch_output = self.generation_manager.run_agent_loop(
                    gen_batch=gen_batch
                )
        prompts = final_gen_batch_output.batch["prompts"].to(device)
        completion_ids = final_gen_batch_output.batch["responses"].to(device)
        prompt_completion_ids = final_gen_batch_output.batch["input_ids"].to(device)
        attention_mask = final_gen_batch_output.batch["attention_mask"].to(device)
        prompt_mask = attention_mask[:, : prompts.size(1)]
        completion_mask = final_gen_batch_output.batch["info_mask"][
            :, prompts.size(1) :
        ].to(device)
        is_eos = completion_ids == self.eos_token_id
        assert completion_ids.shape == completion_mask.shape
        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        completion_ids_list = [
            [id.item() for id, m in zip(row, mask_row) if m]
            for row, mask_row in zip(completion_ids, completion_mask)
        ]
        completion_lengths = completion_mask.sum(1)
        logits_to_keep = completion_mask.size(1)
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )
        with torch.no_grad():
            if (
                self.num_iterations > 1
                or self.args.steps_per_generation
                > self.args.gradient_accumulation_steps
            ):
                old_per_token_logps, old_supervise_mask = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    labels,
                    logits_to_keep,
                )
            else:
                old_per_token_logps, old_supervise_mask = (None, None)
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, ref_supervise_mask = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        labels,
                        logits_to_keep,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, ref_supervise_mask = (
                            self._get_per_token_logps(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                labels,
                                logits_to_keep,
                            )
                        )
            else:
                ref_per_token_logps, ref_supervise_mask = (None, None)
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        completions = completions_text
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
        )
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
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
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())
        return {
            "prompt_ids": prompts,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "old_supervise_mask": old_supervise_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "ref_supervise_mask": ref_supervise_mask,
        }

    def _compute_loss(self, model, inputs):
        device = self.accelerator.device
        prompt_ids, prompt_mask = (inputs["prompt_ids"], inputs["prompt_mask"])
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        old_supervise_mask, ref_supervise_mask = (
            inputs["old_supervise_mask"],
            inputs["ref_supervise_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        prompt_labels = torch.full(prompt_mask.shape, -100, device=device)
        completion_labels = torch.where(completion_mask == 1, completion_ids, -100)
        labels = torch.cat([prompt_labels, completion_labels], dim=1)
        logits_to_keep = completion_labels.size(1)
        assert prompt_ids.shape == prompt_mask.shape
        assert completion_ids.shape == completion_mask.shape
        assert input_ids.shape == attention_mask.shape == labels.shape
        per_token_logps, supervise_mask = self._get_per_token_logps(
            model, input_ids, attention_mask, labels, logits_to_keep
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
        if old_supervise_mask is None:
            old_supervise_mask = supervise_mask
        if ref_supervise_mask is None:
            ref_supervise_mask = supervise_mask
        assert (
            torch.all(supervise_mask <= completion_mask)
            and torch.all(old_supervise_mask <= completion_mask)
            and torch.all(ref_supervise_mask <= completion_mask)
        )
        supervised_mask = (
            completion_mask * supervise_mask * old_supervise_mask * ref_supervise_mask
        )
        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * supervised_mask).sum(-1)
                / supervised_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * supervised_mask
            ).sum() / supervised_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * supervised_mask).sum() / (
                supervised_mask.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * supervised_mask).sum() / supervised_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped
        low_clip = (is_low_clipped * supervised_mask).sum() / supervised_mask.sum()
        high_clip = (is_high_clipped * supervised_mask).sum() / supervised_mask.sum()
        clip_ratio = (is_region_clipped * supervised_mask).sum() / supervised_mask.sum()
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
