from .memory_compressor import component_adapter, hidden_state_forward_kwargs
from peft import LoraConfig, PeftModel
import torch
import torch.nn as nn


class MemGenTrigger(nn.Module):
    adapter_name = "trigger"

    def __init__(
        self,
        model: PeftModel,
        active: bool,
        num_actions: int = 2,
        inactive_default_action: int = 1,
    ):
        super().__init__()
        self.active = active
        self.model = model
        if num_actions < 2:
            raise ValueError(f"num_actions must be >= 2, got {num_actions}")
        if not 0 <= inactive_default_action < num_actions:
            raise ValueError(
                f"inactive_default_action must be in [0, {num_actions}), got {inactive_default_action}"
            )
        self.num_actions = num_actions
        self.inactive_default_action = inactive_default_action
        head_input_dim = model.base_model.config.hidden_size
        self.output_layer = nn.Linear(head_input_dim, num_actions)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.Tensor,
    ) -> torch.FloatTensor:
        if self.active:
            with component_adapter(self):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                    **(
                        hidden_state_forward_kwargs(self.model)
                        if getattr(self, "compress_latent_memory", False)
                        else {}
                    ),
                )
            hidden_states = outputs.hidden_states[-1]
            logits = self.output_layer(hidden_states)
        else:
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(
                batch_size, seq_len, self.num_actions, device=input_ids.device
            )
            logits[..., self.inactive_default_action] = 1.0
        return logits
