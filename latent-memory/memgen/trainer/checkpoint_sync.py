import torch.distributed as dist
from transformers import TrainerCallback


class BestCheckpointSyncCallback(TrainerCallback):

    def on_save(self, args, state, control, **kwargs):
        if dist.is_available() and dist.is_initialized():
            checkpoint = [state.best_model_checkpoint]
            dist.broadcast_object_list(checkpoint, src=0)
            state.best_model_checkpoint = checkpoint[0]
        return control
