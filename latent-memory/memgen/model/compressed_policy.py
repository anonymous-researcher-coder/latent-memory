import inspect
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .memory_compressor import GroupedMemoryCompressor, deterministic_memory


class CompressedPolicyMixin:

    def _initialize_compressors(self):
        if not self.config.compress_latent_memory:
            return
        cfg = self.config
        if cfg.trigger_force_action is not None or cfg.trigger_epsilon_greedy:
            raise ValueError(
                "Compression policy training requires unforced temperature sampling"
            )
        self.prompt_compressor = GroupedMemoryCompressor(
            cfg.hidden_size,
            cfg.prompt_latents_len,
            cfg.compressor_hidden_size,
            cfg.compressor_group_size,
        )
        self.inference_compressor = GroupedMemoryCompressor(
            cfg.hidden_size,
            cfg.inference_latents_len,
            cfg.compressor_hidden_size,
            cfg.compressor_group_size,
        )
        self._compression_rollout_records = None
        self._compression_recording = False

    def named_compressors(self):
        if self.config.compress_latent_memory:
            yield ("prompt", self.prompt_compressor)
            yield ("inference", self.inference_compressor)

    def configure_joint_trainables(self):
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.trigger.named_parameters():
            if (
                ".lora_A.trigger." in name
                or ".lora_B.trigger." in name
                or name.startswith("output_layer.")
            ):
                parameter.requires_grad_(True)
        for _, compressor in self.named_compressors():
            compressor.requires_grad_(True)

    def compress_memory(self, memory, is_prompt, gradient_scale=1.0):
        module = self.prompt_compressor if is_prompt else self.inference_compressor
        return module.with_gradient_scale(memory, gradient_scale)

    def _get_trigger_layer_for_action(self, action):
        candidates = self.config.weaver_random_memory_layers
        if self.trigger.num_actions > 2 and candidates:
            return candidates[action - 1]
        return self.config.weaver_memory_layer

    def _replay_compressed_one(
        self,
        prompt_ids,
        completion_ids,
        actions,
        temperature,
        prompt_scale,
        inference_scale,
        frozen_memory_cache=None,
    ):
        with deterministic_memory(self):
            embeds = self.reasoner.get_input_embeddings()(prompt_ids)
            completion_embeds = self.reasoner.get_input_embeddings()(completion_ids)
            action_list = actions.tolist()
            positions, entropies = ([], [[], []])
            last = 0
            for t in [i for i, action in enumerate(action_list) if action > 0] + [
                len(action_list)
            ]:
                if t > last:
                    positions.extend(
                        range(embeds.size(1) - 1, embeds.size(1) - 1 + t - last)
                    )
                    embeds = torch.cat([embeds, completion_embeds[:, last:t]], 1)
                if t == len(action_list):
                    break
                is_prompt = t == 0
                layer = self._get_trigger_layer_for_action(action_list[t])
                cache_key = None
                if frozen_memory_cache is not None and (not embeds.requires_grad):
                    prefix = torch.cat([prompt_ids, completion_ids[:, :t]], 1)
                    cache_key = (tuple(prefix[0].tolist()), is_prompt, str(layer))
                raw = (
                    frozen_memory_cache.get(cache_key)
                    if cache_key is not None
                    else None
                )
                if raw is None:
                    mask = torch.ones(
                        embeds.shape[:2], dtype=torch.long, device=embeds.device
                    )
                    augment = (
                        self.weaver.augment_prompt
                        if is_prompt
                        else self.weaver.augment_inference
                    )
                    hidden, _, _ = augment(
                        self.reasoner_to_weaver(embeds),
                        mask,
                        self._generate_position_ids(mask),
                        memory_layer=layer,
                    )
                    raw = self.weaver_to_reasoner(hidden)
                    if cache_key is not None and (not raw.requires_grad):
                        raw = raw.clone()
                        frozen_memory_cache[cache_key] = raw
                memory, weights = self.compress_memory(
                    raw, is_prompt, prompt_scale if is_prompt else inference_scale
                )
                entropy = (
                    -(weights.float() * weights.float().clamp_min(1e-12).log())
                    .sum(-1)
                    .mean()
                )
                entropies[0 if is_prompt else 1].append(entropy)
                embeds = torch.cat([embeds, memory], 1)
                last = t
            mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
            indices = torch.tensor(positions, dtype=torch.long, device=embeds.device)
            extra = {}
            signature = inspect.signature(self.reasoner.forward).parameters
            for key in ("logits_to_keep", "num_logits_to_keep"):
                if key in signature:
                    extra[key] = indices
                    break
            logits = self.reasoner(
                inputs_embeds=embeds,
                attention_mask=mask,
                position_ids=self._generate_position_ids(mask),
                use_cache=False,
                **extra
            ).logits
            logits = logits[0] if extra else logits[0, indices]
            values = []
            for chunk, target in zip(logits.split(32), completion_ids[0].split(32)):
                values.append(
                    (chunk.float() / temperature)
                    .log_softmax(-1)
                    .gather(-1, target[:, None])
                    .squeeze(-1)
                )
            entropy = torch.stack(
                [
                    torch.stack(role).mean() if role else embeds.new_zeros(())
                    for role in entropies
                ]
            )
            return (torch.cat(values), entropy)

    def joint_policy_forward(self, data, temperature=1.0, gradient_normalization=True):
        temperature = float(temperature or 1.0)
        if temperature <= 0:
            raise ValueError("Policy temperature must be positive")
        prompt, p_mask = (data["prompt_ids"], data["prompt_mask"])
        completion, c_mask = (data["completion_ids"], data["completion_mask"])
        actions = data["augmentation_mask"]
        valid = actions.ne(-100) & c_mask.bool()
        invoke = actions.gt(0) & c_mask.bool()
        counts = torch.stack([invoke[:, 0].long(), invoke[:, 1:].sum(-1)], -1)
        active_count = counts.gt(0).sum(0).clamp_min(1)
        with deterministic_memory(self):
            ids = torch.cat([prompt, completion], 1)
            mask = torch.cat([p_mask, c_mask], 1)
            logits = self.trigger(ids, mask, self._generate_position_ids(mask))[
                :, prompt.size(1) - 1 : -1
            ]
            trigger_logps = (
                (logits.float() / temperature)
                .log_softmax(-1)
                .gather(-1, actions.clamp_min(0)[..., None])
                .squeeze(-1)
            )
            trigger_logps = trigger_logps * valid
        answer_mask = invoke.cumsum(-1).gt(0) & c_mask.bool()
        answers, entropy, replayed = ([], [], [])
        frozen_memory_cache = {}
        advantages = data.get("advantages")
        for b in range(prompt.size(0)):
            n = int(c_mask[b].sum())
            needed = (
                bool(counts[b].sum())
                and n > 0
                and (advantages is None or bool(advantages[b] != 0))
            )
            replayed.append(needed)
            if not needed:
                answers.append(
                    torch.zeros(completion.size(1), device=completion.device)
                )
                entropy.append(torch.zeros(2, device=completion.device))
                continue
            scales = [
                float(getattr(self.config, role + "_compressor_loss_weight"))
                for role in ("prompt", "inference")
            ]
            if gradient_normalization:
                scales = [
                    value
                    * prompt.size(0)
                    / float(active_count[r] * counts[b, r].clamp_min(1))
                    for r, value in enumerate(scales)
                ]
            p = prompt[b : b + 1, p_mask[b].bool()]
            c, a = (completion[b : b + 1, :n], actions[b, :n])

            def replay(p=p, c=c, a=a, scales=scales):
                return self._replay_compressed_one(
                    p,
                    c,
                    a,
                    temperature,
                    *scales,
                    frozen_memory_cache=frozen_memory_cache
                )

            if self.config.compressor_checkpoint_replay and torch.is_grad_enabled():
                lp, ent = checkpoint(replay, use_reentrant=False)
            else:
                lp, ent = replay()
            answers.append(F.pad(lp, (0, completion.size(1) - n)))
            entropy.append(ent)
        return {
            "trigger_logps": trigger_logps,
            "answer_logps": torch.stack(answers),
            "answer_mask": answer_mask,
            "counts": counts,
            "entropy": torch.stack(entropy),
            "replayed": torch.tensor(replayed, device=completion.device),
        }

    @torch.no_grad()
    def generate_compressed(
        self,
        input_ids,
        attention_mask,
        generation_config=None,
        return_augmentation_mask=False,
        **kwargs
    ):
        from transformers import GenerationConfig

        config = generation_config or GenerationConfig(
            max_new_tokens=128, do_sample=False
        )
        if getattr(config, "repetition_penalty", 1.0) != 1.0:
            raise ValueError(
                "Compressed policy requires repetition_penalty=1 for replay equivalence"
            )
        batch_size = int(self.config.compressor_rollout_batch_size)
        recurrent = "linear_attention" in getattr(
            self.reasoner.config, "layer_types", []
        )
        if recurrent:
            batch_size = 1
        self._compression_effective_batch_size = batch_size
        records = self._compression_rollout_records
        self._compression_recording = records is not None
        outputs, decisions = ([], [])
        start = time.perf_counter()
        try:
            with deterministic_memory(self):
                for offset in range(0, input_ids.size(0), batch_size):
                    budget = kwargs.get("per_sample_max_augment_num")
                    if torch.is_tensor(budget):
                        budget = budget[offset : offset + batch_size]
                    out, acts, record = self._compressed_generate_batch(
                        input_ids[offset : offset + batch_size],
                        attention_mask[offset : offset + batch_size],
                        config,
                        budget,
                    )
                    outputs.append(out)
                    decisions.append(acts)
                    if records is not None:
                        records.extend(record)
        finally:
            self._compression_recording = False
        self._compression_rollout_seconds = time.perf_counter() - start
        length = max((out.size(1) for out in outputs))
        output = torch.cat(
            [
                F.pad(out, (0, length - out.size(1)), value=self.tokenizer.pad_token_id)
                for out in outputs
            ]
        )
        action_length = length - input_ids.size(1)
        actions = torch.cat(
            [F.pad(a, (0, action_length - a.size(1)), value=-100) for a in decisions]
        )
        return (output, actions) if return_augmentation_mask else output

    def _compressed_generate_batch(self, ids, mask, config, budget):
        batch, prompt_length = ids.shape
        device = ids.device
        temperature = float(config.temperature or 1.0)
        text_mask = mask.clone()
        embeds = self.reasoner.get_input_embeddings()(ids)
        positions = self._generate_position_ids(mask)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        counts = torch.zeros(batch, dtype=torch.long, device=device)
        last_insert = torch.full_like(counts, -(10**9))
        entropy = None
        cache = None
        actions, trigger_lps, answer_lps = ([], [], [])
        lengths = torch.zeros_like(counts)
        captured = []
        handle = self.trigger.register_forward_hook(
            lambda module, args, output: captured.append(output[:, -1].detach())
        )
        try:
            for step in range(config.max_new_tokens):
                captured.clear()
                decision = self._should_augment(
                    ids,
                    sentence_augment_count=counts,
                    do_sample=config.do_sample,
                    temperature=temperature,
                    is_prompt=step == 0,
                    max_augment_num=budget,
                    step_entropy=entropy,
                    input_attention_mask=text_mask,
                    cooldown_mask=(
                        step - last_insert < self.config.entropy_insert_min_distance
                        if step
                        else None
                    ),
                )
                candidates = decision.ne(-100)
                tlp = torch.zeros(batch, device=device)
                if captured:
                    values = (captured[-1].float() / temperature).log_softmax(-1)
                    if values.size(0) != int(candidates.sum()):
                        raise RuntimeError("Trigger candidate recording mismatch")
                    tlp[candidates] = values.gather(
                        -1, decision[candidates, None]
                    ).squeeze(-1)
                decision = decision.masked_fill(finished, -100)
                tlp.masked_fill_(finished, 0)
                inserted = decision.gt(0)
                if inserted.any():
                    rows = [embeds[b, mask[b].bool()] for b in range(batch)]
                    for action in decision[inserted].unique().tolist():
                        selected = (decision == action).nonzero().flatten()
                        candidate = embeds[selected]
                        candidate_mask = mask[selected]
                        augment = (
                            self.weaver.augment_prompt
                            if step == 0
                            else self.weaver.augment_inference
                        )
                        hidden, _, _ = augment(
                            self.reasoner_to_weaver(candidate),
                            candidate_mask,
                            self._generate_position_ids(candidate_mask),
                            memory_layer=self._get_trigger_layer_for_action(action),
                        )
                        memory, _ = self.compress_memory(
                            self.weaver_to_reasoner(hidden), step == 0
                        )
                        for j, b in enumerate(selected.tolist()):
                            rows[b] = torch.cat([rows[b], memory[j]], 0)
                    width = max((row.size(0) for row in rows))
                    embeds = torch.stack(
                        [F.pad(row, (0, 0, width - row.size(0), 0)) for row in rows]
                    )
                    mask = torch.stack(
                        [
                            F.pad(
                                torch.ones(
                                    row.size(0), device=device, dtype=torch.long
                                ),
                                (width - row.size(0), 0),
                            )
                            for row in rows
                        ]
                    )
                    positions = self._generate_position_ids(mask)
                    cache = None
                    if step:
                        counts += inserted
                        last_insert = torch.where(inserted, step, last_insert)
                cache_length = 0 if cache is None else cache.get_seq_length()
                extra = {}
                for key in ("logits_to_keep", "num_logits_to_keep"):
                    if key in inspect.signature(self.reasoner.forward).parameters:
                        extra[key] = 1
                        break
                output = self.reasoner(
                    inputs_embeds=embeds[:, cache_length:],
                    attention_mask=mask,
                    position_ids=positions[:, cache_length:],
                    past_key_values=cache,
                    use_cache=True,
                    output_hidden_states=False,
                    **extra
                )
                cache = output.past_key_values
                scores = output.logits[:, -1].float()
                probabilities = scores.softmax(-1)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
                    -1
                )
                embeds, mask, positions, ids = self._append_one_step(
                    output, embeds, mask, positions, ids, config.do_sample, temperature
                )
                tokens = ids[:, -1]
                alp = (
                    (scores / temperature)
                    .log_softmax(-1)
                    .gather(-1, tokens[:, None])
                    .squeeze(-1)
                )
                ids[:, -1] = tokens.masked_fill(finished, self.tokenizer.pad_token_id)
                mask[:, -1] = (~finished).long()
                text_mask = torch.cat([text_mask, (~finished).long()[:, None]], 1)
                actions.append(decision)
                trigger_lps.append(tlp)
                answer_lps.append(alp)
                lengths += (~finished).long()
                eos = self.tokenizer.eos_token_id
                finished |= tokens.eq(eos)
                if finished.all():
                    break
        finally:
            handle.remove()
        action_tensor = torch.stack(actions, 1)
        tlps, alps = (torch.stack(trigger_lps, 1), torch.stack(answer_lps, 1))
        records = [
            (tlps[b : b + 1, : int(n)], alps[b : b + 1, : int(n)])
            for b, n in enumerate(lengths)
        ]
        return (ids, action_tensor, records)
