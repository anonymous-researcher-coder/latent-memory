import logging
from typing import Union, Optional
import os
import json
from peft import LoraConfig
from safetensors import safe_open
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    DynamicCache,
)
from transformers.modeling_utils import PreTrainedModel
from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_utils import (
    MemGenOutputWithPast,
    MemGenLoraSwitchMixin,
    MemGenGenerationMixin,
)
from memgen.model.compressed_policy import CompressedPolicyMixin
from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    load_state_dict_from_safetensor,
)


def _resolve_entropy_threshold_from_percentile(
    threshold_data: dict, percentile: str
) -> float:
    if not isinstance(percentile, str):
        raise TypeError(
            f"entropy_threshold_percentile must be a string, got {type(percentile)}"
        )
    key = percentile.strip()
    if key.lower() == "recommended":
        if "recommended_threshold" not in threshold_data:
            raise KeyError("threshold.json does not contain 'recommended_threshold'.")
        return float(threshold_data["recommended_threshold"])
    pct = threshold_data.get("percentile_thresholds", {}) or {}
    if key in pct and isinstance(pct[key], dict) and ("threshold" in pct[key]):
        return float(pct[key]["threshold"])
    msd = threshold_data.get("mean_std_thresholds", {}) or {}
    if key in msd and isinstance(msd[key], dict) and ("threshold" in msd[key]):
        return float(msd[key]["threshold"])
    available = list(pct.keys()) + list(msd.keys()) + ["recommended"]
    raise KeyError(
        f"entropy_threshold_percentile='{percentile}' not found in threshold.json. Available keys: {available}"
    )


class MemGenModel(
    CompressedPolicyMixin, PreTrainedModel, MemGenLoraSwitchMixin, MemGenGenerationMixin
):
    config_class = MemGenConfig

    def __init__(
        self,
        config: MemGenConfig,
        base_tokenizer,
        base_model: PreTrainedModel,
        weaver_model: Optional[PreTrainedModel] = None,
        trigger_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__(config)
        self.config = config
        if weaver_model is None:
            weaver_model = base_model
        if trigger_model is None:
            trigger_model = base_model
        fix_model_parameters(base_model)
        fix_model_parameters(weaver_model)
        fix_model_parameters(trigger_model)
        weaver_model, trigger_model = self._insert_lora_adapters(
            weaver_model,
            config.weaver_lora_config,
            trigger_model,
            config.trigger_lora_config,
        )
        self.weaver = MemGenWeaver(
            weaver_model,
            config.prompt_latents_len,
            config.inference_latents_len,
            memory_layer=getattr(config, "weaver_memory_layer", -1),
            random_memory_layers=getattr(config, "weaver_random_memory_layers", None),
        )
        self.trigger = MemGenTrigger(
            trigger_model,
            config.trigger_active,
            num_actions=getattr(config, "trigger_num_actions", 2),
        )
        for component in (self.weaver, self.trigger):
            component.isolate_adapters = (
                config.isolate_adapters or config.compress_latent_memory
            )
            component.compress_latent_memory = config.compress_latent_memory
        if config.isolate_adapters or config.compress_latent_memory:
            from .memory_compressor import finish_joint_adapter

            for backbone in {id(m): m for m in (weaver_model, trigger_model)}.values():
                finish_joint_adapter(backbone)
        self.reasoner = base_model
        self.tokenizer = base_tokenizer
        reasoner_hidden_size = self.config.hidden_size
        weaver_hidden_size = weaver_model.base_model.config.hidden_size
        self.reasoner_to_weaver = nn.Linear(reasoner_hidden_size, weaver_hidden_size)
        self.weaver_to_reasoner = nn.Linear(weaver_hidden_size, reasoner_hidden_size)
        self.delimiters: list[str] = [",", ".", "\n"]
        self._initialize_compressors()
        import re

        seen, aliases = (set(), [])
        for name, parameter in self.named_parameters(remove_duplicate=False):
            if id(parameter) in seen:
                aliases.append(re.escape(name) + "$")
            seen.add(id(parameter))
        self._tied_weights_keys = aliases
        self._postprocess_models()

    def _postprocess_models(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
            logging.info(
                f"Tokenizer has no pad token. Using EOS token ({self.tokenizer.eos_token}) as pad token."
            )
        self.tokenizer.chat_template = CONVERSATION_TEMPLATE

    @property
    def device(self):
        return self.reasoner.device

    def _get_trigger_layer_for_action(self, action: int) -> int | str | None:
        num_actions = self.trigger.num_actions
        if num_actions == 2:
            return None
        elif num_actions == 3:
            candidates = self.weaver.random_memory_layers
            if candidates is None or len(candidates) < 2:
                raise ValueError(
                    f"trigger_num_actions=3 requires random_memory_layers with at least 2 candidates, got {candidates}"
                )
            if action == 1:
                return candidates[0]
            elif action == 2:
                return candidates[1]
            else:
                return None
        else:
            raise ValueError(f"Unsupported num_actions={num_actions}")

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        assert input_ids.shape == attention_mask.shape == labels.shape
        max_augment_num = self.config.max_inference_aug_num
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        delimiters = self.delimiters
        device = self.device
        embeds_dtype = reasoner.get_input_embeddings().weight.dtype
        B, _ = input_ids.shape
        hidden_size = self.config.hidden_size
        entropy_mode = getattr(self.config, "entropy_trigger_mode", "none")
        if entropy_mode == "entropy_position":
            entropy_threshold = getattr(self.config, "entropy_threshold_value", 3.0)
            entropy_threshold_upper = getattr(
                self.config, "entropy_threshold_upper_value", None
            )
            entropy_min_distance = int(
                getattr(self.config, "entropy_insert_min_distance", 0) or 0
            )
            augmentation_indices = self._select_augment_points_by_entropy(
                input_ids,
                labels,
                tokenizer,
                max_augment_num,
                entropy_threshold,
                entropy_threshold_upper=entropy_threshold_upper,
                min_distance=entropy_min_distance,
            )
        else:
            augmentation_indices = self._select_augment_points_after_delimiter(
                input_ids, labels, delimiters, tokenizer, max_augment_num
            )
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        current_start_idx = 0
        current_inputs_embeds = torch.empty(
            (B, 0, hidden_size), device=device, dtype=embeds_dtype
        )
        current_attention_mask = torch.empty(
            (B, 0), device=device, dtype=attention_mask.dtype
        )
        current_latents_mask = torch.empty((B, 0), device=device, dtype=torch.bool)
        for aug_point_idx in augmentation_indices:
            segment_inputs_embeds = inputs_embeds[:, current_start_idx:aug_point_idx]
            segment_attention_mask = attention_mask[:, current_start_idx:aug_point_idx]
            segment_latents_mask = torch.zeros(
                (B, segment_inputs_embeds.size(1)), device=device, dtype=torch.bool
            )
            current_inputs_embeds = torch.cat(
                [current_inputs_embeds, segment_inputs_embeds], dim=1
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, segment_attention_mask], dim=1
            )
            current_position_ids = self._generate_position_ids(current_attention_mask)
            current_latents_mask = torch.cat(
                [current_latents_mask, segment_latents_mask], dim=1
            )
            weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)
            is_prompt_end_aug = (labels[:, aug_point_idx] != -100).all() and (
                labels[:, aug_point_idx - 1] == -100
            ).all().item()
            if is_prompt_end_aug:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            else:
                weaver_hidden_states, attn_mask, pos_ids = weaver.augment_inference(
                    weaver_inputs_embeds, current_attention_mask, current_position_ids
                )
            latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
            current_inputs_embeds = torch.cat(
                [current_inputs_embeds, latent_inputs_embeds], dim=1
            )
            current_attention_mask = torch.cat(
                [current_attention_mask, attn_mask], dim=1
            )
            current_start_idx = aug_point_idx
            latent_mask = torch.ones(
                (B, latent_inputs_embeds.size(1)), device=device, dtype=torch.bool
            )
            current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)
        remaining_inputs_embeds = inputs_embeds[:, current_start_idx:]
        remaining_attention_mask = attention_mask[:, current_start_idx:]
        latent_mask = torch.zeros(
            (B, remaining_attention_mask.size(1)), device=device, dtype=torch.bool
        )
        current_inputs_embeds = torch.cat(
            [current_inputs_embeds, remaining_inputs_embeds], dim=1
        )
        current_attention_mask = torch.cat(
            [current_attention_mask, remaining_attention_mask], dim=1
        )
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_latents_mask = torch.cat([current_latents_mask, latent_mask], dim=1)
        reasoner_outputs = reasoner(
            inputs_embeds=current_inputs_embeds,
            attention_mask=current_attention_mask,
            position_ids=current_position_ids,
        )
        logits = reasoner_outputs.logits
        shifted = torch.zeros_like(current_latents_mask)
        shifted[:, :-1] = current_latents_mask[:, 1:]
        valid_mask = ~shifted
        valid_logits = logits[valid_mask].view(logits.size(0), -1, logits.size(2))
        return valid_logits

    def _instructional_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        logits = self._forward(input_ids, attention_mask, labels, **kwargs)
        return (logits, labels)

    def _conversational_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.FloatTensor, torch.LongTensor]:
        assert (
            input_ids.shape[0] == 1
        ), "Conversational SFT currently only supports batch_size = 1"
        seq_len = input_ids.shape[1]
        vocab_size = self.config.vocab_size
        device = input_ids.device
        label_row = labels[0]
        should_supervise = label_row != -100
        if not should_supervise.any():
            raise ValueError("At least one completion segment is required")
        valid_mask = should_supervise.int()
        diff = torch.diff(torch.cat([torch.tensor([0], device=device), valid_mask]))
        valid_starts = (diff == 1).nonzero(as_tuple=True)[0].tolist()
        ends = (diff == -1).nonzero(as_tuple=True)[0].tolist()
        if len(ends) < len(valid_starts):
            ends.append(seq_len)
        assert len(valid_starts) == len(ends)
        triplets = []
        start = 0
        for s, e in zip(valid_starts, ends):
            triplets.append((start, s, e))
            start = e
        if len(triplets) <= self.config.max_prompt_aug_num:
            select_turns = [1] * len(triplets)
        else:
            triplets_num = len(triplets)
            selected_indices = set(
                random.sample(range(triplets_num), self.config.max_prompt_aug_num)
            )
            select_turns = [
                1 if i in selected_indices else 0 for i in range(triplets_num)
            ]
        all_logits = torch.zeros(1, seq_len, vocab_size, device=device)
        all_labels = torch.full((1, seq_len), -100, device=device)
        for triplet, should_supervise in zip(triplets, select_turns):
            start, valid_start, end = triplet
            if should_supervise:
                cur_input_ids = input_ids[0, :end].unsqueeze(0)
                cur_attention = attention_mask[0, :end].unsqueeze(0)
                cur_labels = labels[0, :end].clone().unsqueeze(0)
                cur_labels[0, :valid_start] = -100
                logits = self._forward(
                    cur_input_ids, cur_attention, cur_labels, **kwargs
                )
                all_logits[0, start:end, :] = logits[0, start:end, :]
                all_labels[0, start:end] = labels[0, start:end]
        return (all_logits, all_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
        trigger_only: bool = False,
        **kwargs,
    ) -> Union[MemGenOutputWithPast, torch.Tensor]:
        if trigger_only:
            return self.trigger(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=kwargs["position_ids"],
            )
        if "joint_compression_batch" in kwargs:
            return self.joint_policy_forward(
                kwargs["joint_compression_batch"], kwargs.get("temperature", 1.0)
            )
        tokenizer = self.tokenizer
        assert (
            labels is not None
        ), "Reasoning Processor requires input labels for training"
        forward_func = self._instructional_forward
        if self._is_conversation(input_ids, tokenizer):
            labels = self._postprocess_assistant_labels(input_ids, labels, tokenizer)
            forward_func = self._conversational_forward
        batch_size = 1
        iter_num = input_ids.size(0) // batch_size
        logits, supervised_labels = ([], [])
        for i in range(iter_num):
            batch_input_ids = input_ids[i * batch_size : (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size : (i + 1) * batch_size]
            batch_labels = labels[i * batch_size : (i + 1) * batch_size]
            forward_kwargs = dict(kwargs)
            batch_logits, batch_supervised_labels = forward_func(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                **forward_kwargs,
            )
            logits.append(batch_logits)
            supervised_labels.append(batch_supervised_labels)
        all_logits = torch.concat(logits, dim=0)
        all_labels = torch.concat(supervised_labels, dim=0)
        shift_logits = all_logits[..., :-1, :].contiguous()
        shift_labels = all_labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        outputs = MemGenOutputWithPast(loss=loss, logits=all_logits)
        outputs.supervised_labels = all_labels
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generation_config: GenerationConfig = None,
        return_augmentation_mask: bool = False,
        **kwargs,
    ) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]:
        if self.config.compress_latent_memory:
            return self.generate_compressed(
                input_ids,
                attention_mask,
                generation_config,
                return_augmentation_mask,
                **kwargs,
            )
        tokenizer = self.tokenizer
        reasoner = self.reasoner
        weaver = self.weaver
        invalid_token_id = -100
        per_sample_max_augment_num = kwargs.pop("per_sample_max_augment_num", None)
        if per_sample_max_augment_num is not None:
            max_augment_num = per_sample_max_augment_num
        else:
            max_augment_num = self.config.max_inference_aug_num
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        max_new_tokens = generation_config.max_new_tokens
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        prompt_len = input_ids.size(1)
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        B, _, hidden_size = inputs_embeds.shape
        device = inputs_embeds.device
        current_inputs_embeds = inputs_embeds
        current_attention_mask = attention_mask
        current_position_ids = self._generate_position_ids(current_attention_mask)
        current_input_ids = input_ids
        current_text_mask = attention_mask.clone()
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        current_cache: DynamicCache = None
        last_real_tensor_idx = torch.full(
            (B,), input_ids.size(1) - 1, dtype=torch.long, device=device
        )
        sentence_augment_count = torch.zeros(B, dtype=torch.int, device=device)
        last_step_entropy = None
        entropy_min_distance = int(
            getattr(self.config, "entropy_insert_min_distance", 0) or 0
        )
        if entropy_min_distance > 0:
            last_augment_step = torch.full(
                (B,), -(10**9), dtype=torch.long, device=device
            )
        else:
            last_augment_step = None
        augmentation_pos = torch.full(
            (B, max_new_tokens), fill_value=invalid_token_id, device=device
        )
        num_actions = self.trigger.num_actions
        for i in range(max_new_tokens):
            assert (
                current_inputs_embeds.shape[:2]
                == current_attention_mask.shape
                == current_position_ids.shape
            )
            cooldown_mask = None
            if last_augment_step is not None and i > 0:
                cooldown_mask = i - last_augment_step < entropy_min_distance
            augment_decision = self._should_augment(
                current_input_ids,
                sentence_augment_count=sentence_augment_count,
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                is_prompt=i == 0,
                max_augment_num=max_augment_num,
                step_entropy=last_step_entropy,
                cooldown_mask=(
                    finished if cooldown_mask is None else cooldown_mask | finished
                ),
                input_attention_mask=current_text_mask,
            )
            augment_decision = augment_decision.masked_fill(finished, invalid_token_id)
            augmentation_pos[:, i] = augment_decision
            if last_augment_step is not None and i > 0:
                inserted = (augment_decision != 0) & (augment_decision != -100)
                last_augment_step = torch.where(
                    inserted, torch.full_like(last_augment_step, i), last_augment_step
                )
            if num_actions == 2:
                augment_indices = torch.where(augment_decision == 1)[0]
                if len(augment_indices) > 0:
                    if i != 0:
                        sentence_augment_count[augment_indices] += 1
                    pad_len = (
                        weaver.prompt_latents_num
                        if i == 0
                        else weaver.inference_latents_num
                    )
                    new_len = current_inputs_embeds.size(1) + pad_len
                    merged_inputs_embeds = torch.zeros(
                        (B, new_len, hidden_size),
                        device=device,
                        dtype=current_inputs_embeds.dtype,
                    )
                    merged_attention_mask = torch.zeros(
                        (B, new_len), device=device, dtype=current_attention_mask.dtype
                    )
                    candidate_inputs_embeds = current_inputs_embeds[augment_indices]
                    candidate_attention_mask = current_attention_mask[augment_indices]
                    candidate_position_ids = current_position_ids[augment_indices]
                    weaver_inputs_embeds = self.reasoner_to_weaver(
                        candidate_inputs_embeds
                    )
                    if i == 0:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                            weaver_inputs_embeds,
                            candidate_attention_mask,
                            candidate_position_ids,
                        )
                    else:
                        weaver_hidden_states, attn_mask, _ = weaver.augment_inference(
                            weaver_inputs_embeds,
                            candidate_attention_mask,
                            candidate_position_ids,
                        )
                    latent_inputs_embeds = self.weaver_to_reasoner(weaver_hidden_states)
                    candidate_inputs_embeds = torch.cat(
                        [candidate_inputs_embeds, latent_inputs_embeds], dim=1
                    )
                    candidate_attention_mask = torch.cat(
                        [candidate_attention_mask, attn_mask], dim=1
                    )
                    merged_inputs_embeds[augment_indices] = candidate_inputs_embeds
                    merged_attention_mask[augment_indices] = candidate_attention_mask
                    non_augment_indices = torch.where(augment_decision != 1)[0]
                    if len(non_augment_indices) > 0:
                        non_aug_inputs_embeds = current_inputs_embeds[
                            non_augment_indices
                        ]
                        non_aug_attention_mask = current_attention_mask[
                            non_augment_indices
                        ]
                        non_aug_inputs_embeds, non_aug_attention_mask, _ = (
                            self._right_pad(
                                non_aug_inputs_embeds,
                                non_aug_attention_mask,
                                None,
                                pad_len,
                            )
                        )
                        merged_inputs_embeds[non_augment_indices] = (
                            non_aug_inputs_embeds
                        )
                        merged_attention_mask[non_augment_indices] = (
                            non_aug_attention_mask
                        )
                    current_inputs_embeds = merged_inputs_embeds
                    current_attention_mask = merged_attention_mask
                    current_position_ids = self._generate_position_ids(
                        current_attention_mask
                    )
                    last_real_tensor_idx = last_real_tensor_idx.clone()
                    last_real_tensor_idx[augment_indices] += pad_len
            elif num_actions == 3:
                augment_indices_1 = torch.where(augment_decision == 1)[0]
                augment_indices_2 = torch.where(augment_decision == 2)[0]
                augment_indices = torch.where(
                    (augment_decision == 1) | (augment_decision == 2)
                )[0]
                if len(augment_indices) > 0:
                    if i != 0:
                        sentence_augment_count[augment_indices] += 1
                    pad_len = (
                        weaver.prompt_latents_num
                        if i == 0
                        else weaver.inference_latents_num
                    )
                    new_len = current_inputs_embeds.size(1) + pad_len
                    merged_inputs_embeds = torch.zeros(
                        (B, new_len, hidden_size),
                        device=device,
                        dtype=current_inputs_embeds.dtype,
                    )
                    merged_attention_mask = torch.zeros(
                        (B, new_len), device=device, dtype=current_attention_mask.dtype
                    )
                    if augment_indices_1.numel() > 0:
                        layer_1 = self._get_trigger_layer_for_action(1)
                        candidate_inputs_embeds = current_inputs_embeds[
                            augment_indices_1
                        ]
                        candidate_attention_mask = current_attention_mask[
                            augment_indices_1
                        ]
                        candidate_position_ids = current_position_ids[augment_indices_1]
                        weaver_inputs_embeds = self.reasoner_to_weaver(
                            candidate_inputs_embeds
                        )
                        if i == 0:
                            weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                                weaver_inputs_embeds,
                                candidate_attention_mask,
                                candidate_position_ids,
                                memory_layer=layer_1,
                            )
                        else:
                            weaver_hidden_states, attn_mask, _ = (
                                weaver.augment_inference(
                                    weaver_inputs_embeds,
                                    candidate_attention_mask,
                                    candidate_position_ids,
                                    memory_layer=layer_1,
                                )
                            )
                        latent_inputs_embeds = self.weaver_to_reasoner(
                            weaver_hidden_states
                        )
                        candidate_inputs_embeds = torch.cat(
                            [candidate_inputs_embeds, latent_inputs_embeds], dim=1
                        )
                        candidate_attention_mask = torch.cat(
                            [candidate_attention_mask, attn_mask], dim=1
                        )
                        merged_inputs_embeds[augment_indices_1] = (
                            candidate_inputs_embeds
                        )
                        merged_attention_mask[augment_indices_1] = (
                            candidate_attention_mask
                        )
                    if augment_indices_2.numel() > 0:
                        layer_2 = self._get_trigger_layer_for_action(2)
                        candidate_inputs_embeds = current_inputs_embeds[
                            augment_indices_2
                        ]
                        candidate_attention_mask = current_attention_mask[
                            augment_indices_2
                        ]
                        candidate_position_ids = current_position_ids[augment_indices_2]
                        weaver_inputs_embeds = self.reasoner_to_weaver(
                            candidate_inputs_embeds
                        )
                        if i == 0:
                            weaver_hidden_states, attn_mask, _ = weaver.augment_prompt(
                                weaver_inputs_embeds,
                                candidate_attention_mask,
                                candidate_position_ids,
                                memory_layer=layer_2,
                            )
                        else:
                            weaver_hidden_states, attn_mask, _ = (
                                weaver.augment_inference(
                                    weaver_inputs_embeds,
                                    candidate_attention_mask,
                                    candidate_position_ids,
                                    memory_layer=layer_2,
                                )
                            )
                        latent_inputs_embeds = self.weaver_to_reasoner(
                            weaver_hidden_states
                        )
                        candidate_inputs_embeds = torch.cat(
                            [candidate_inputs_embeds, latent_inputs_embeds], dim=1
                        )
                        candidate_attention_mask = torch.cat(
                            [candidate_attention_mask, attn_mask], dim=1
                        )
                        merged_inputs_embeds[augment_indices_2] = (
                            candidate_inputs_embeds
                        )
                        merged_attention_mask[augment_indices_2] = (
                            candidate_attention_mask
                        )
                    non_augment_indices = torch.where(
                        (augment_decision != 1) & (augment_decision != 2)
                    )[0]
                    if len(non_augment_indices) > 0:
                        non_aug_inputs_embeds = current_inputs_embeds[
                            non_augment_indices
                        ]
                        non_aug_attention_mask = current_attention_mask[
                            non_augment_indices
                        ]
                        non_aug_inputs_embeds, non_aug_attention_mask, _ = (
                            self._right_pad(
                                non_aug_inputs_embeds,
                                non_aug_attention_mask,
                                None,
                                pad_len,
                            )
                        )
                        merged_inputs_embeds[non_augment_indices] = (
                            non_aug_inputs_embeds
                        )
                        merged_attention_mask[non_augment_indices] = (
                            non_aug_attention_mask
                        )
                    current_inputs_embeds = merged_inputs_embeds
                    current_attention_mask = merged_attention_mask
                    current_position_ids = self._generate_position_ids(
                        current_attention_mask
                    )
                    last_real_tensor_idx = last_real_tensor_idx.clone()
                    last_real_tensor_idx[augment_indices] += pad_len
            full_seq_len = current_inputs_embeds.size(1)
            if current_cache is not None:
                cached_len = current_cache.get_seq_length()
                new_count = full_seq_len - cached_len
                reasoner_inputs_embeds = current_inputs_embeds[:, -new_count:]
                reasoner_position_ids = current_position_ids[:, -new_count:]
            else:
                new_count = full_seq_len
                reasoner_inputs_embeds = current_inputs_embeds
                reasoner_position_ids = current_position_ids
            outputs = reasoner(
                inputs_embeds=reasoner_inputs_embeds,
                attention_mask=current_attention_mask,
                position_ids=reasoner_position_ids,
                output_hidden_states=False,
                use_cache=True,
                past_key_values=current_cache,
            )
            entropy_mode = getattr(self.config, "entropy_trigger_mode", "none")
            if entropy_mode == "entropy_position":
                logit_idx_in_output = last_real_tensor_idx - (full_seq_len - new_count)
                step_logits = outputs.logits[
                    torch.arange(B, device=device), logit_idx_in_output
                ]
                probs = F.softmax(step_logits, dim=-1)
                last_step_entropy = -(probs * torch.log(probs + 1e-08)).sum(dim=-1)
            (
                current_inputs_embeds,
                current_attention_mask,
                current_position_ids,
                current_input_ids,
            ) = self._append_one_step(
                outputs,
                current_inputs_embeds,
                current_attention_mask,
                current_position_ids,
                current_input_ids,
                do_sample=(
                    generation_config.do_sample
                    if generation_config.do_sample is not None
                    else False
                ),
                temperature=(
                    generation_config.temperature
                    if generation_config.temperature is not None
                    else 0.0
                ),
                repetition_penalty=(
                    generation_config.repetition_penalty
                    if generation_config.repetition_penalty is not None
                    else 1.0
                ),
                last_real_tensor_idx=last_real_tensor_idx,
                new_tokens_count=new_count,
            )
            current_input_ids[:, -1] = current_input_ids[:, -1].masked_fill(
                finished, pad_token_id
            )
            current_attention_mask[:, -1] = (~finished).to(current_attention_mask.dtype)
            current_text_mask = torch.cat(
                [current_text_mask, (~finished).to(current_text_mask.dtype)[:, None]],
                dim=1,
            )
            finished |= current_input_ids[:, -1].eq(eos_token_id)
            current_cache = outputs.past_key_values
            last_real_tensor_idx = torch.full(
                (B,), current_inputs_embeds.size(1) - 1, dtype=torch.long, device=device
            )
            if finished.all():
                break
            del outputs
        new_generated_len = current_input_ids.size(1) - prompt_len
        augmentation_pos = augmentation_pos[:, :new_generated_len]
        self._check_generate(current_input_ids[:, prompt_len:], augmentation_pos)
        if return_augmentation_mask:
            return (current_input_ids, augmentation_pos)
        else:
            return current_input_ids

    @classmethod
    def from_pretrained(
        cls,
        load_directory,
        *,
        config,
        base_tokenizer,
        base_model,
        weaver_model=None,
        trigger_model=None,
    ):
        from .checkpoint_loading import load_memgen_checkpoint

        model = cls(config, base_tokenizer, base_model, weaver_model, trigger_model)
        load_memgen_checkpoint(model, load_directory)
        return model

    @classmethod
    def from_config(cls, config_dict: dict):
        model_name = config_dict.get("model_name")
        max_prompt_aug_num = config_dict.get("max_prompt_aug_num", 1)
        max_inference_aug_num = config_dict.get("max_inference_aug_num", 5)
        weaver_config = config_dict.get("weaver", {})
        prompt_latents_len = weaver_config.get("prompt_latents_len", 8)
        inference_latents_len = weaver_config.get("inference_latents_len", 8)
        weaver_memory_layer = weaver_config.get("memory_layer", -1)
        weaver_lora_config_dict = weaver_config.get("lora_config", None)
        weaver_model_name = weaver_config.get("model_name", None)
        raw_random_layers = weaver_config.get("random_memory_layers", None)
        weaver_random_memory_layers = None
        if raw_random_layers is not None:
            if isinstance(raw_random_layers, str):
                if raw_random_layers.strip().lower() in ("null", "none", ""):
                    weaver_random_memory_layers = None
                else:
                    candidates = []
                    for item in raw_random_layers.split(","):
                        item = item.strip()
                        if not item:
                            continue
                        try:
                            candidates.append(int(item))
                        except ValueError:
                            candidates.append(item)
                    weaver_random_memory_layers = (
                        candidates if len(candidates) > 0 else None
                    )
            elif isinstance(raw_random_layers, list):
                weaver_random_memory_layers = raw_random_layers
        trigger_config = config_dict.get("trigger", {})
        trigger_active = trigger_config.get("active", False)
        trigger_lora_config_dict = trigger_config.get("lora_config", None)
        trigger_model_name = trigger_config.get("model_name", None)
        trigger_epsilon_greedy = trigger_config.get("epsilon_greedy", 0.0)
        trigger_force_action = trigger_config.get("force_action", None)
        if isinstance(
            trigger_force_action, str
        ) and trigger_force_action.strip().lower() in ("null", "none", ""):
            trigger_force_action = None
        trigger_num_actions = config_dict.get("trigger_num_actions", None)
        if trigger_num_actions is None:
            if (
                weaver_random_memory_layers is not None
                and len(weaver_random_memory_layers) >= 2
            ):
                trigger_num_actions = 3
            else:
                trigger_num_actions = 2
        trigger_num_actions = int(trigger_num_actions)
        entropy_config = config_dict.get("entropy_trigger", {})
        entropy_trigger_mode = entropy_config.get("mode") or "none"
        if entropy_trigger_mode not in ("none", "entropy_position"):
            raise ValueError("entropy_trigger_mode must be none or entropy_position")
        entropy_threshold_value = entropy_config.get("threshold_value", None)
        if isinstance(
            entropy_threshold_value, str
        ) and entropy_threshold_value.strip().lower() in ("null", "none", ""):
            entropy_threshold_value = None
        entropy_threshold_percentile = entropy_config.get("threshold_percentile", None)
        if isinstance(
            entropy_threshold_percentile, str
        ) and entropy_threshold_percentile.strip().lower() in ("null", "none", ""):
            entropy_threshold_percentile = None
        entropy_threshold_file = entropy_config.get("threshold_file", None)
        if isinstance(
            entropy_threshold_file, str
        ) and entropy_threshold_file.strip().lower() in ("null", "none", ""):
            entropy_threshold_file = None
        entropy_threshold_upper_value = entropy_config.get(
            "threshold_upper_value", None
        )
        if isinstance(
            entropy_threshold_upper_value, str
        ) and entropy_threshold_upper_value.strip().lower() in ("null", "none", ""):
            entropy_threshold_upper_value = None
        entropy_threshold_upper_percentile = entropy_config.get(
            "threshold_upper_percentile", None
        )
        if isinstance(
            entropy_threshold_upper_percentile, str
        ) and entropy_threshold_upper_percentile.strip().lower() in (
            "null",
            "none",
            "",
        ):
            entropy_threshold_upper_percentile = None
        entropy_insert_min_distance = entropy_config.get("insert_min_distance", 0)
        if isinstance(
            entropy_insert_min_distance, str
        ) and entropy_insert_min_distance.strip().lower() in ("null", "none", ""):
            entropy_insert_min_distance = 0
        entropy_insert_min_distance = int(entropy_insert_min_distance or 0)
        if (
            entropy_trigger_mode == "entropy_position"
            and entropy_threshold_value is None
            and entropy_threshold_percentile is not None
        ):
            if entropy_threshold_file is None:
                raise ValueError(
                    "threshold_file is required when resolving a threshold percentile"
                )
            if not os.path.exists(entropy_threshold_file):
                raise FileNotFoundError(
                    f"entropy_threshold_percentile='{entropy_threshold_percentile}' is set, but threshold file not found: {entropy_threshold_file}. Provide an existing threshold file, or set model.entropy_trigger.threshold_file explicitly."
                )
            with open(entropy_threshold_file, "r", encoding="utf-8") as f:
                threshold_data = json.load(f)
            entropy_threshold_value = _resolve_entropy_threshold_from_percentile(
                threshold_data, entropy_threshold_percentile
            )
            _log_msg = f"[Entropy Threshold] mode={entropy_trigger_mode} | file={entropy_threshold_file} | percentile={entropy_threshold_percentile} | resolved_value={entropy_threshold_value:.6g}"
            print(_log_msg)
            logging.getLogger(__name__).info(_log_msg)
        elif (
            entropy_threshold_value is not None
            and entropy_threshold_percentile is not None
        ):
            _log_msg = f"[Entropy Threshold] mode={entropy_trigger_mode} | explicit_value={float(entropy_threshold_value):.6g} | percentile_hint={entropy_threshold_percentile} (ignored, value takes precedence)"
            print(_log_msg)
            logging.getLogger(__name__).info(_log_msg)
        elif entropy_threshold_value is not None and entropy_trigger_mode != "none":
            _log_msg = f"[Entropy Threshold] mode={entropy_trigger_mode} | explicit_value={float(entropy_threshold_value):.6g}"
            print(_log_msg)
            logging.getLogger(__name__).info(_log_msg)
        if (
            entropy_trigger_mode == "entropy_position"
            and entropy_threshold_upper_value is None
            and entropy_threshold_upper_percentile is not None
        ):
            if entropy_threshold_file is None:
                raise ValueError(
                    "threshold_file is required when resolving an upper threshold percentile"
                )
            if not os.path.isfile(entropy_threshold_file):
                raise FileNotFoundError("Upper entropy threshold file not found")
            with open(entropy_threshold_file, "r", encoding="utf-8") as f:
                _upper_threshold_data = json.load(f)
            entropy_threshold_upper_value = _resolve_entropy_threshold_from_percentile(
                _upper_threshold_data, entropy_threshold_upper_percentile
            )
            _log_msg_upper = f"[Entropy Threshold Upper] mode={entropy_trigger_mode} | file={entropy_threshold_file} | percentile={entropy_threshold_upper_percentile} | resolved_value={entropy_threshold_upper_value:.6g}"
            print(_log_msg_upper)
            logging.getLogger(__name__).info(_log_msg_upper)
        elif (
            entropy_threshold_upper_value is not None
            and entropy_threshold_upper_percentile is not None
        ):
            _log_msg_upper = f"[Entropy Threshold Upper] mode={entropy_trigger_mode} | explicit_value={float(entropy_threshold_upper_value):.6g} | percentile_hint={entropy_threshold_upper_percentile} (ignored, value takes precedence)"
            print(_log_msg_upper)
            logging.getLogger(__name__).info(_log_msg_upper)
        elif (
            entropy_threshold_upper_value is not None and entropy_trigger_mode != "none"
        ):
            _log_msg_upper = f"[Entropy Threshold Upper] mode={entropy_trigger_mode} | explicit_value={float(entropy_threshold_upper_value):.6g}"
            print(_log_msg_upper)
            logging.getLogger(__name__).info(_log_msg_upper)
        if entropy_insert_min_distance > 0 and entropy_trigger_mode != "none":
            _log_msg_dist = f"[Entropy Insert Min-Distance] mode={entropy_trigger_mode} | min_distance={entropy_insert_min_distance} tokens"
            print(_log_msg_dist)
            logging.getLogger(__name__).info(_log_msg_dist)
        from transformers import AutoConfig

        memgen_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        memgen_config = MemGenConfig.from_pretrained(
            model_name,
            max_prompt_aug_num=max_prompt_aug_num,
            max_inference_aug_num=max_inference_aug_num,
            prompt_latents_len=prompt_latents_len,
            inference_latents_len=inference_latents_len,
            weaver_memory_layer=weaver_memory_layer,
            weaver_random_memory_layers=weaver_random_memory_layers,
            isolate_adapters=config_dict.get("isolate_adapters", True),
            compress_latent_memory=config_dict.get("compress_latent_memory", False),
            compressor_group_size=config_dict.get("compressor", {}).get(
                "group_size", 4
            ),
            compressor_hidden_size=config_dict.get("compressor", {}).get(
                "hidden_size", 64
            ),
            compressor_checkpoint_replay=config_dict.get("compressor", {}).get(
                "checkpoint_replay", True
            ),
            compressor_rollout_batch_size=config_dict.get("compressor", {}).get(
                "rollout_batch_size", 4
            ),
            prompt_compressor_loss_weight=config_dict.get("compressor", {}).get(
                "prompt_loss_weight", 1.0
            ),
            inference_compressor_loss_weight=config_dict.get("compressor", {}).get(
                "inference_loss_weight", 1.0
            ),
            weaver_lora_config=weaver_lora_config_dict,
            trigger_active=trigger_active,
            trigger_lora_config=trigger_lora_config_dict,
            trigger_num_actions=trigger_num_actions,
            trigger_epsilon_greedy=trigger_epsilon_greedy,
            trigger_force_action=trigger_force_action,
            entropy_trigger_mode=entropy_trigger_mode,
            entropy_threshold_value=entropy_threshold_value,
            entropy_threshold_percentile=entropy_threshold_percentile,
            entropy_threshold_file=entropy_threshold_file,
            entropy_threshold_upper_value=entropy_threshold_upper_value,
            entropy_threshold_upper_percentile=entropy_threshold_upper_percentile,
            entropy_insert_min_distance=entropy_insert_min_distance,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        base_tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if weaver_model_name != model_name:
            weaver_model = AutoModelForCausalLM.from_pretrained(
                weaver_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        else:
            weaver_model = base_model
        if trigger_model_name != model_name:
            trigger_model = AutoModelForCausalLM.from_pretrained(
                trigger_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        else:
            trigger_model = base_model
        load_model_path = config_dict.get("load_model_path", None)
        if isinstance(load_model_path, str) and load_model_path.strip().lower() in (
            "null",
            "none",
            "",
        ):
            load_model_path = None
        if not load_model_path:
            model = cls(
                config=memgen_config,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                weaver_model=weaver_model,
                trigger_model=trigger_model,
            )
        else:
            model = cls.from_pretrained(
                load_model_path,
                config=memgen_config,
                base_model=base_model,
                base_tokenizer=base_tokenizer,
                weaver_model=weaver_model,
                trigger_model=trigger_model,
            )
        return model
