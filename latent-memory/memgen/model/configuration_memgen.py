import math
from transformers import PretrainedConfig
from typing import Optional


class MemGenConfig(PretrainedConfig):
    model_type = "memgen"

    def __init__(
        self,
        weaver_lora_config: Optional[dict] = None,
        prompt_latents_len: int = 0,
        inference_latents_len: int = 0,
        weaver_memory_layer: int | str = -1,
        weaver_random_memory_layers: Optional[list] = None,
        trigger_active: bool = False,
        trigger_lora_config: Optional[dict] = None,
        trigger_num_actions: int = 2,
        trigger_epsilon_greedy: float = 0.0,
        trigger_force_action: Optional[int] = None,
        max_prompt_aug_num: int = 1,
        max_inference_aug_num: int = 5,
        entropy_trigger_mode: str = "none",
        entropy_threshold_value: Optional[float] = None,
        entropy_threshold_percentile: Optional[str] = None,
        entropy_threshold_file: Optional[str] = None,
        entropy_threshold_upper_value: Optional[float] = None,
        entropy_threshold_upper_percentile: Optional[str] = None,
        entropy_insert_min_distance: int = 0,
        isolate_adapters: bool = True,
        compress_latent_memory: bool = False,
        compressor_group_size: int = 4,
        compressor_hidden_size: int = 64,
        compressor_checkpoint_replay: bool = True,
        compressor_rollout_batch_size: int = 4,
        prompt_compressor_loss_weight: float = 1.0,
        inference_compressor_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.isolate_adapters = isolate_adapters
        self.compress_latent_memory = compress_latent_memory
        self.compressor_group_size = compressor_group_size
        self.compressor_hidden_size = compressor_hidden_size
        self.compressor_checkpoint_replay = compressor_checkpoint_replay
        self.compressor_rollout_batch_size = compressor_rollout_batch_size
        self.prompt_compressor_loss_weight = prompt_compressor_loss_weight
        self.inference_compressor_loss_weight = inference_compressor_loss_weight
        if compress_latent_memory:
            if (
                type(compressor_group_size) is not int
                or compressor_group_size < 1
                or prompt_latents_len < 1
                or (inference_latents_len < 1)
                or prompt_latents_len % compressor_group_size
                or inference_latents_len % compressor_group_size
            ):
                raise ValueError(
                    "compressor_group_size must divide both positive memory lengths"
                )
            if (
                type(compressor_rollout_batch_size) is not int
                or compressor_rollout_batch_size < 1
            ):
                raise ValueError("compressor_rollout_batch_size must be positive")
        self.weaver_lora_config = weaver_lora_config
        self.prompt_latents_len = prompt_latents_len
        self.inference_latents_len = inference_latents_len
        self.weaver_memory_layer = weaver_memory_layer
        self.weaver_random_memory_layers = weaver_random_memory_layers
        self.trigger_active = trigger_active
        self.trigger_lora_config = trigger_lora_config
        self.trigger_num_actions = trigger_num_actions
        if not 0.0 <= float(trigger_epsilon_greedy) <= 1.0:
            raise ValueError(
                f"trigger_epsilon_greedy must be in [0, 1], got {trigger_epsilon_greedy}"
            )
        self.trigger_epsilon_greedy = float(trigger_epsilon_greedy)
        if trigger_force_action is not None:
            if not isinstance(trigger_force_action, int):
                raise ValueError(
                    f"trigger_force_action must be an int or None, got {type(trigger_force_action)}"
                )
            if trigger_force_action < 0 or trigger_force_action >= int(
                trigger_num_actions
            ):
                raise ValueError(
                    f"trigger_force_action must be in [0, {int(trigger_num_actions) - 1}], got {trigger_force_action}"
                )
            self.trigger_force_action = int(trigger_force_action)
        else:
            self.trigger_force_action = None
        self.max_prompt_aug_num = max_prompt_aug_num
        self.max_inference_aug_num = max_inference_aug_num
        if entropy_trigger_mode is None:
            entropy_trigger_mode = "none"
        valid_entropy_modes = ("none", "entropy_position")
        if entropy_trigger_mode not in valid_entropy_modes:
            raise ValueError(
                f"entropy_trigger_mode must be one of {valid_entropy_modes}, got '{entropy_trigger_mode}'"
            )
        self.entropy_trigger_mode = entropy_trigger_mode
        self.entropy_threshold_value = (
            float(entropy_threshold_value)
            if entropy_threshold_value is not None
            else None
        )
        if self.entropy_threshold_value is not None and not math.isfinite(
            self.entropy_threshold_value
        ):
            raise ValueError("entropy_threshold_value must be finite")
        self.entropy_threshold_percentile = entropy_threshold_percentile
        self.entropy_threshold_file = entropy_threshold_file
        if self.entropy_trigger_mode != "none" and self.entropy_threshold_value is None:
            raise ValueError(
                f"entropy_trigger_mode='{self.entropy_trigger_mode}' requires entropy_threshold_value to be set (or provide entropy_threshold_percentile + entropy_threshold_file to resolve it from threshold.json)."
            )
        self.entropy_threshold_upper_value = (
            float(entropy_threshold_upper_value)
            if entropy_threshold_upper_value is not None
            else None
        )
        if self.entropy_threshold_upper_value is not None and not math.isfinite(
            self.entropy_threshold_upper_value
        ):
            raise ValueError("entropy_threshold_upper_value must be finite")
        self.entropy_threshold_upper_percentile = entropy_threshold_upper_percentile
        if (
            self.entropy_threshold_upper_value is not None
            and self.entropy_threshold_value is not None
        ):
            if self.entropy_threshold_upper_value <= self.entropy_threshold_value:
                raise ValueError(
                    f"entropy_threshold_upper_value ({self.entropy_threshold_upper_value}) must be greater than entropy_threshold_value ({self.entropy_threshold_value})."
                )
        if int(entropy_insert_min_distance) < 0:
            raise ValueError(
                f"entropy_insert_min_distance must be >= 0, got {entropy_insert_min_distance}"
            )
        self.entropy_insert_min_distance = int(entropy_insert_min_distance)
