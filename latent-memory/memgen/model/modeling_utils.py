from dataclasses import dataclass
from typing import Optional, Literal
from peft import PeftModel, LoraConfig
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from memgen.model.trigger import MemGenTrigger
from memgen.model.weaver import MemGenWeaver
from memgen.utils import (
    CONVERSATION_TEMPLATE,
    fix_model_parameters,
    open_model_parameters,
)


@dataclass
class MemGenOutputWithPast(CausalLMOutputWithPast):
    supervised_labels: Optional[torch.LongTensor] = None


class MemGenLoraSwitchMixin:

    def _insert_lora_adapters(
        self,
        weaver_model: PreTrainedModel,
        weaver_lora_config: dict,
        trigger_model: PreTrainedModel,
        trigger_lora_config: dict,
    ) -> tuple[PeftModel, PeftModel, PeftModel]:
        same_model = weaver_model is trigger_model
        weaver_lora_config = LoraConfig(**weaver_lora_config)
        trigger_lora_config = LoraConfig(**trigger_lora_config)
        if same_model:
            peft_model = PeftModel(
                weaver_model, weaver_lora_config, adapter_name=MemGenWeaver.adapter_name
            )
            peft_model.add_adapter(
                peft_config=trigger_lora_config, adapter_name=MemGenTrigger.adapter_name
            )
            return (peft_model, peft_model)
        else:
            weaver_model_with_lora = PeftModel(
                weaver_model, weaver_lora_config, adapter_name=MemGenWeaver.adapter_name
            )
            trigger_model_with_lora = PeftModel(
                trigger_model,
                trigger_lora_config,
                adapter_name=MemGenTrigger.adapter_name,
            )
            return (weaver_model_with_lora, trigger_model_with_lora)

    def fix_component(self, name: Literal["weaver", "trigger"]):
        component = getattr(self, name)
        fix_model_parameters(component)
        if name == "weaver":
            fix_model_parameters(self.weaver_to_reasoner)
            fix_model_parameters(self.reasoner_to_weaver)

    def open_component(self, name: Literal["weaver", "trigger"]):
        component = getattr(self, name)
        open_model_parameters(component)
        if name == "weaver":
            open_model_parameters(self.weaver_to_reasoner)
            open_model_parameters(self.reasoner_to_weaver)
        fix_model_parameters(component.model.base_model)
        for p in component.model.named_parameters():
            if f"lora_A.{name}" in p[0] or f"lora_B.{name}" in p[0]:
                p[1].requires_grad = True


class MemGenGenerationMixin(GenerationMixin):

    def _get_next_token(
        self,
        next_token_logits: torch.Tensor,
        do_sample: bool,
        temperature: Optional[float] = 0.0,
    ) -> torch.Tensor:
        if len(next_token_logits.shape) != 2:
            raise ValueError(
                "Input logits must be a 2D tensor [batch_size, vocab_size]"
            )
        if do_sample and temperature != 0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(next_token_logits, dim=-1, keepdim=True)

    def _generate_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _is_conversation(self, input_ids: torch.Tensor, tokenizer) -> bool:
        if len(input_ids.shape) != 2:
            raise ValueError(
                "input_ids must be a 2D tensor of shape (batch_size, seq_len)"
            )
        seq = input_ids[0].tolist()
        im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        has_start = any(
            (
                seq[i : i + len(im_start_ids)] == im_start_ids
                for i in range(len(seq) - len(im_start_ids) + 1)
            )
        )
        has_end = any(
            (
                seq[i : i + len(im_end_ids)] == im_end_ids
                for i in range(len(seq) - len(im_end_ids) + 1)
            )
        )
        return has_start and has_end

    def _postprocess_assistant_labels(
        self, input_ids: torch.Tensor, labels: torch.Tensor, tokenizer
    ) -> torch.Tensor:
        if tokenizer.chat_template != CONVERSATION_TEMPLATE:
            raise ValueError(
                f"Invalid tokenizer.chat_template detected.\nExpected:\n{CONVERSATION_TEMPLATE}\n\nGot:\n{tokenizer.chat_template}\n\nPlease ensure that you are using the correct conversation template."
            )
        pattern_ids: list[int] = tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        batch_size, seq_len = input_ids.shape
        new_labels = labels.clone()
        for b in range(batch_size):
            seq = input_ids[b].tolist()
            for i in range(len(seq) - len(pattern_ids) + 1):
                if seq[i : i + len(pattern_ids)] == pattern_ids:
                    new_labels[b, i : i + len(pattern_ids)] = -100
        return new_labels

    def _check_ends_with_delimiter(
        self, input_ids: torch.Tensor, tokenizer, delimiters: list[str]
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        augmentation_decisions = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=input_ids.device
        )
        decoded_inputs = tokenizer.batch_decode(input_ids)
        for i in range(batch_size):
            ends_with_augment_str = False
            for aug_str in delimiters:
                if decoded_inputs[i].endswith(aug_str):
                    ends_with_augment_str = True
                    break
            augmentation_decisions[i] = ends_with_augment_str
        return augmentation_decisions

    def _select_augment_points_after_delimiter(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        delimiters: list[str],
        tokenizer: PreTrainedTokenizerBase,
        max_num: int = 10,
    ) -> list[int]:
        assert input_ids.shape == labels.shape
        B, seq_len = (input_ids.size(0), input_ids.size(1))
        prompt_augment_idx = []
        inference_augment_idx = []
        for i in range(1, seq_len):
            if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
                prompt_augment_idx.append(i)
            elif (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
                batch_tokens_before_i = input_ids[:, :i]
                if any(
                    self._check_ends_with_delimiter(
                        batch_tokens_before_i, tokenizer, delimiters
                    )
                ):
                    inference_augment_idx.append(i)
        if len(prompt_augment_idx) != 1:
            raise ValueError(
                "Single-turn forward must have exactly one prompt augment index"
            )
        final_points = (
            prompt_augment_idx[:1] if self.config.max_prompt_aug_num > 0 else []
        )
        if len(inference_augment_idx) > max_num:
            inference_augment_idx = inference_augment_idx[:max_num]
        final_points.extend(inference_augment_idx)
        final_points.sort()
        return final_points

    def _select_augment_points_by_entropy(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        max_num: int = 10,
        entropy_threshold: float = 3.0,
        entropy_threshold_upper: Optional[float] = None,
        min_distance: int = 0,
    ) -> list[int]:
        assert input_ids.shape == labels.shape
        B, seq_len = (input_ids.size(0), input_ids.size(1))
        assert (
            B == 1
        ), f"_select_augment_points_by_entropy requires batch_size=1, got {B}. Each sample must select augmentation positions independently using its own entropy."
        prompt_augment_idx = []
        for i in range(1, seq_len):
            if (labels[:, i] != -100).all() and (labels[:, i - 1] == -100).all():
                prompt_augment_idx.append(i)
        if len(prompt_augment_idx) != 1:
            raise ValueError(
                "Single-turn forward must have exactly one prompt augment index"
            )
        reasoner = self.reasoner
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = reasoner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-08)).sum(dim=-1)
            del outputs
        candidate_idx = []
        for i in range(1, seq_len):
            if (labels[:, i] != -100).all() and (labels[:, i - 1] != -100).all():
                pos_entropy = entropy[0, i - 1].item()
                if pos_entropy < entropy_threshold:
                    continue
                if (
                    entropy_threshold_upper is not None
                    and pos_entropy >= entropy_threshold_upper
                ):
                    continue
                candidate_idx.append(i)
        inference_augment_idx = []
        last_selected = -(10**9)
        stride = max(int(min_distance), 1)
        for pos in candidate_idx:
            if pos - last_selected >= stride:
                inference_augment_idx.append(pos)
                last_selected = pos
        if len(inference_augment_idx) > max_num:
            inference_augment_idx = inference_augment_idx[:max_num]
        final_points = (
            prompt_augment_idx[:1] if self.config.max_prompt_aug_num > 0 else []
        )
        final_points.extend(inference_augment_idx)
        final_points.sort()
        return final_points

    @torch.no_grad()
    def _should_augment(
        self,
        input_ids: torch.LongTensor,
        sentence_augment_count: torch.LongTensor,
        do_sample: bool,
        temperature: float,
        is_prompt: bool = False,
        max_augment_num: int | torch.Tensor | None = None,
        step_entropy: torch.FloatTensor | None = None,
        cooldown_mask: torch.BoolTensor | None = None,
        input_attention_mask: torch.Tensor | None = None,
    ) -> torch.LongTensor:
        tokenizer = self.tokenizer
        delimiters = self.delimiters
        trigger = self.trigger
        if max_augment_num is None:
            max_augment_num = self.config.max_inference_aug_num
        entropy_mode = getattr(self.config, "entropy_trigger_mode", "none")
        entropy_threshold = getattr(self.config, "entropy_threshold_value", None)
        entropy_threshold_upper = getattr(
            self.config, "entropy_threshold_upper_value", None
        )
        batch_size = input_ids.size(0)
        if is_prompt and self.config.max_prompt_aug_num == 0:
            return torch.full(
                (batch_size,), -100, dtype=torch.long, device=input_ids.device
            )
        if is_prompt:
            attention_mask = (
                (input_ids != tokenizer.pad_token_id).long()
                if input_attention_mask is None
                else input_attention_mask
            )
            position_ids = self._generate_position_ids(attention_mask)
            aug_vector = torch.zeros(
                (batch_size,), dtype=torch.long, device=input_ids.device
            )
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]
        else:
            attention_mask = (
                (input_ids != tokenizer.pad_token_id).long()
                if input_attention_mask is None
                else input_attention_mask
            )
            position_ids = self._generate_position_ids(attention_mask)
            aug_vector = torch.full(
                (batch_size,), -100, dtype=torch.long, device=input_ids.device
            )
            if entropy_mode == "none":
                ends_with_delimiters = self._check_ends_with_delimiter(
                    input_ids, tokenizer, delimiters
                ).squeeze(1)
                aug_vector[ends_with_delimiters] = 0
            elif entropy_mode == "entropy_position":
                if step_entropy is not None and entropy_threshold is not None:
                    in_range = step_entropy >= entropy_threshold
                    if entropy_threshold_upper is not None:
                        in_range = in_range & (step_entropy < entropy_threshold_upper)
                    aug_vector[in_range] = 0
            if cooldown_mask is not None:
                aug_vector[cooldown_mask] = -100
            over_limit = sentence_augment_count >= max_augment_num
            aug_vector[over_limit] = -100
            trigger_indices = (aug_vector != -100).nonzero(as_tuple=True)[0]
        if trigger_indices.numel() > 0:
            trigger_logits = trigger(
                input_ids=input_ids[trigger_indices],
                attention_mask=attention_mask[trigger_indices],
                position_ids=position_ids[trigger_indices],
            )
            last_token_logits = trigger_logits[:, -1]
            force_action = getattr(self.config, "trigger_force_action", None)
            if force_action is not None:
                next_tokens = torch.full(
                    (last_token_logits.size(0),),
                    int(force_action),
                    dtype=torch.long,
                    device=last_token_logits.device,
                )
            else:
                eps = float(getattr(self.config, "trigger_epsilon_greedy", 0.0) or 0.0)
                if eps > 0.0:
                    num_actions = last_token_logits.size(-1)
                    greedy_actions = torch.argmax(last_token_logits, dim=-1)
                    if num_actions <= 1:
                        next_tokens = greedy_actions.view(-1)
                    else:
                        r = torch.randint(
                            0,
                            num_actions - 1,
                            greedy_actions.shape,
                            device=greedy_actions.device,
                        )
                        rand_actions = r + (r >= greedy_actions).long()
                        explore = (
                            torch.rand(
                                greedy_actions.shape, device=greedy_actions.device
                            )
                            < eps
                        )
                        next_tokens = torch.where(
                            explore, rand_actions, greedy_actions
                        ).view(-1)
                else:
                    next_tokens = self._get_next_token(
                        last_token_logits, do_sample=do_sample, temperature=temperature
                    ).view(-1)
            aug_vector[trigger_indices] = next_tokens
        return aug_vector

    @torch.no_grad()
    def _apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        if penalty == 1.0:
            return logits
        score = torch.gather(logits, 1, input_ids)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits = logits.scatter_(1, input_ids, score)
        return logits

    @torch.no_grad()
    def _append_one_step(
        self,
        reasoner_outputs,
        current_inputs_embeds: torch.Tensor,
        current_attention_mask: torch.Tensor,
        current_position_ids: torch.Tensor,
        current_input_ids: torch.Tensor,
        do_sample: bool,
        temperature: float,
        repetition_penalty: float = 1.0,
        last_real_tensor_idx: torch.LongTensor | None = None,
        new_tokens_count: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = current_inputs_embeds.size(0)
        device = current_inputs_embeds.device
        if last_real_tensor_idx is not None and new_tokens_count is not None:
            full_seq_len = current_inputs_embeds.size(1)
            logit_idx_in_output = last_real_tensor_idx - (
                full_seq_len - new_tokens_count
            )
            next_token_logits = reasoner_outputs.logits[
                torch.arange(B, device=device), logit_idx_in_output
            ]
        else:
            next_token_logits = reasoner_outputs.logits[:, -1]
        next_token_logits = self._apply_repetition_penalty(
            next_token_logits, current_input_ids, repetition_penalty
        )
        next_token_ids = self._get_next_token(
            next_token_logits, do_sample=do_sample, temperature=temperature
        )
        current_input_ids = torch.cat([current_input_ids, next_token_ids], dim=1)
        next_token_embeds = self.reasoner.get_input_embeddings()(next_token_ids)
        current_inputs_embeds = torch.cat(
            [current_inputs_embeds, next_token_embeds], dim=1
        )
        attn_mask = torch.ones(
            (B, 1),
            dtype=current_attention_mask.dtype,
            device=current_attention_mask.device,
        )
        current_attention_mask = torch.cat([current_attention_mask, attn_mask], dim=1)
        if last_real_tensor_idx is not None:
            next_position_id = (
                current_position_ids.gather(1, last_real_tensor_idx.unsqueeze(1)) + 1
            )
        else:
            next_position_id = current_position_ids[:, -1:] + 1
        current_position_ids = torch.cat(
            [current_position_ids, next_position_id], dim=1
        )
        return (
            current_inputs_embeds,
            current_attention_mask,
            current_position_ids,
            current_input_ids,
        )

    @torch.no_grad()
    def _left_pad(
        self,
        input_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_num: int,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        if input_embeds is not None:
            B, L, D = input_embeds.shape
            pad_embeds = torch.zeros(
                (B, pad_num, D), dtype=input_embeds.dtype, device=input_embeds.device
            )
            input_embeds = torch.cat([pad_embeds, input_embeds], dim=1)
        if attention_mask is not None:
            B = attention_mask.size(0)
            pad_mask = torch.zeros(
                (B, pad_num), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([pad_mask, attention_mask], dim=1)
        if position_ids is not None:
            B = position_ids.size(0)
            pad_pos = torch.zeros(
                (B, pad_num), dtype=position_ids.dtype, device=position_ids.device
            )
            position_ids = torch.cat([pad_pos, position_ids], dim=1)
        return (input_embeds, attention_mask, position_ids)

    @torch.no_grad()
    def _right_pad(
        self,
        input_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        pad_num: int,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        if input_embeds is not None:
            B, L, D = input_embeds.shape
            pad_embeds = torch.zeros(
                (B, pad_num, D), dtype=input_embeds.dtype, device=input_embeds.device
            )
            input_embeds = torch.cat([input_embeds, pad_embeds], dim=1)
        if attention_mask is not None:
            B = attention_mask.size(0)
            pad_mask = torch.zeros(
                (B, pad_num), dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)
        if position_ids is not None:
            B = position_ids.size(0)
            pad_pos = torch.zeros(
                (B, pad_num), dtype=position_ids.dtype, device=position_ids.device
            )
            position_ids = torch.cat([position_ids, pad_pos], dim=1)
        return (input_embeds, attention_mask, position_ids)

    @torch.no_grad()
    def _left_clip_pad_tokens(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
    ) -> tuple[torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        B, L, D = inputs_embeds.shape
        first_nonpad_idx = []
        for b in range(B):
            nonzero = (attention_mask[b] != 0).nonzero(as_tuple=True)[0]
            if len(nonzero) == 0:
                first_nonpad_idx.append(L)
            else:
                first_nonpad_idx.append(nonzero[0].item())
        min_pad = min(first_nonpad_idx)
        if min_pad == 0:
            return (inputs_embeds, attention_mask, position_ids)
        inputs_embeds = inputs_embeds[:, min_pad:, :]
        attention_mask = attention_mask[:, min_pad:]
        position_ids = position_ids[:, min_pad:]
        return (inputs_embeds, attention_mask, position_ids)

    @torch.no_grad()
    def _check_generate(
        self, input_ids: torch.LongTensor, augmentation_pos: torch.LongTensor
    ):
        entropy_mode = getattr(self.config, "entropy_trigger_mode", "none")
        if entropy_mode == "entropy_position":
            return
        delimiters = self.delimiters
        tokenizer = self.tokenizer
        num_actions = self.trigger.num_actions
        valid_actions = set(range(num_actions))
        B, L = input_ids.shape
        assert augmentation_pos.shape == input_ids.shape
        for b in range(B):
            for i in range(1, L):
                is_augment_point = augmentation_pos[b, i].item()
                if is_augment_point == -100:
                    continue
                if is_augment_point in valid_actions:
                    prefix_input_ids = input_ids[b, :i].unsqueeze(0)
                    ends_with_delimiter = self._check_ends_with_delimiter(
                        prefix_input_ids, tokenizer, delimiters
                    ).item()
                    if not ends_with_delimiter:
                        decoded_prefix = tokenizer.decode(
                            prefix_input_ids.squeeze(0), skip_special_tokens=False
                        )
                        raise ValueError(
                            f"Augmentation position error at batch {b}, index {i}. augmentation_pos is {is_augment_point}, but the prefix does NOT end with a delimiter.\nPrefix: '...{decoded_prefix[-50:]}'\nDelimiters: {delimiters}"
                        )
                else:
                    raise ValueError(
                        f"Invalid value in augmentation_pos at batch {b}, index {i}: {is_augment_point}. Expected one of {valid_actions} or -100."
                    )
