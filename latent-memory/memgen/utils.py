from dataclasses import dataclass, field
import glob
import json
import logging
import os
import shutil
from typing import Optional, Callable, Dict, List
from safetensors import safe_open
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

CONVERSATION_TEMPLATE = '\n{# ───── main loop ───── #}\n{%- for message in messages -%}\n    {%- set content = message.content if message.content is string else "" -%}\n    {%- if (message.role == "user") or (message.role == "system") -%}\n        {{ "<|im_start|>" + message.role + "\\n"  + content + "<|im_end|>\\n" }}\n    {%- elif message.role == "assistant" -%}\n        {%- generation -%}\n        {{ "<|im_start|>assistant\\n" + content + "<|im_end|>\\n" }}\n        {%- endgeneration -%}\n    {%- elif message.role == "tool" -%}\n    {{ "<|im_start|>" + "user\\n"  + content + "<|im_end|>\\n" }}\n    {%- endif -%}\n{%- endfor -%}\n{# ───── generation prompt ───── #}\n{%- if add_generation_prompt -%}\n    {{ "<|im_start|>assistant\\n" }}\n{%- endif -%}\n'.strip()


def load_state_dict_from_safetensor(model_path) -> Dict:
    model_state_dict = {}
    with safe_open(model_path, framework="pt") as f:
        for key in f.keys():
            model_state_dict[key] = f.get_tensor(key)
    return model_state_dict


def fix_model_parameters(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = False


def open_model_parameters(model: nn.Module):
    for parameter in model.parameters():
        parameter.requires_grad = True


def log_trainable_params(model: nn.Module):
    logging.info("Trainable parameters in the model:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"  {name}: {param.numel()} params, shape={param.shape}")


@dataclass
class StaticEvalRecorder:
    compute_metrics: List[Callable[[str, str, str], float]] = field(
        default_factory=list
    )
    log_file: Optional[str] = None
    writer: Optional[object] = None
    metric_sums: Dict[str, float] = field(init=False)
    metric_counts: Dict[str, int] = field(init=False)

    def __post_init__(self):
        self.metric_sums = {metric.__name__: 0.0 for metric in self.compute_metrics}
        self.metric_counts = {metric.__name__: 0 for metric in self.compute_metrics}
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "w") as f:
                f.write("")

    def record_batch(self, completions: List[str], examples: List[Dict]):
        keys = [key for key in examples[0]]
        reward_kwargs = {key: [example[key] for example in examples] for key in keys}
        reward_kwargs["completions"] = completions
        batched_results = {}
        for metric in self.compute_metrics:
            metric_name = metric.__name__
            batched_scores = metric(**reward_kwargs)
            batched_results[metric_name] = batched_scores
        for i, (completion, example) in enumerate(zip(completions, examples)):
            metrics_result = {
                metric_name: batched_results[metric_name][i]
                for metric_name in batched_results
            }
            for metric_name, score in metrics_result.items():
                self.metric_sums[metric_name] += score
                self.metric_counts[metric_name] += 1
            prompt = example.get("prompt", "")
            solution = example.get("solution", "")
            record = {
                "prompt": prompt,
                "solution": solution,
                "completion": completion,
                "metrics": metrics_result,
            }
            if self.log_file:
                with open(self.log_file, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if self.writer:
                mean_metrics = self.get_mean_metrics()
                for name, value in mean_metrics.items():
                    self.writer.add_scalar(
                        name, value, global_step=self.metric_counts[name]
                    )

    def get_mean_metrics(self) -> Dict[str, float]:
        return {
            name: (
                self.metric_sums[name] / self.metric_counts[name]
                if self.metric_counts[name] > 0
                else 0.0
            )
            for name in self.metric_sums
        }

    def finalize(self):
        mean_metrics = self.get_mean_metrics()
        final_record = {"summary_metrics": mean_metrics}
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_record, ensure_ascii=False) + "\n")
        if self.writer:
            mean_metrics = self.get_mean_metrics()
            for name, value in mean_metrics.items():
                self.writer.add_scalar(
                    name + "_final", value, global_step=self.metric_counts[name]
                )


def create_tensorboard(save_dir: str):
    log_dir = os.path.join(save_dir, "runs")
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def remove_trainer_checkpoints(trainer_output_dir):
    ckpt_paths = glob.glob(os.path.join(trainer_output_dir, "checkpoint-*"))
    for ckpt in ckpt_paths:
        shutil.rmtree(ckpt, ignore_errors=True)
