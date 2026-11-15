import os
import random
import logging
import json
from accelerate import Accelerator
from datasets import Dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig, GRPOConfig
from trl.models import unwrap_model_for_generation
from data import BaseBuilder
from interactions.base_interaction import (
    InteractionConfig,
    InteractionManager,
    InteractionDataProto,
)
from interactions.singleturn_interaction import SingleTurnInteractionManager
from memgen.model.modeling_memgen import MemGenModel
from memgen.trainer.checkpoint_sync import BestCheckpointSyncCallback
from memgen.trainer.weaver_grpo_trainer import WeaverGRPOTrainer
from memgen.trainer.trigger_grpo_trainer import TriggerGRPOTrainer
from memgen.utils import (
    StaticEvalRecorder,
    create_tensorboard,
    remove_trainer_checkpoints,
    log_trainable_params,
)


class MemGenRunner:

    def __init__(
        self,
        model: MemGenModel,
        data_builder: BaseBuilder,
        config: dict,
        working_dir: str,
    ):
        self.config = config
        self.working_dir = working_dir
        self._parse_configs(config.get("run"))
        self.processing_class = model.tokenizer
        self.model = model
        self.dataset_dict = data_builder.get_dataset_dict()
        self.env_cls = data_builder.get_env_cls()
        self.env = self.env_cls(config.get("dataset"))
        self.weaver_train_dataset, self.trigger_train_dataset = (
            self._parse_train_dataset(self.dataset_dict["train"])
        )
        self.weaver_valid_dataset, self.trigger_valid_dataset = (
            self._parse_valid_dataset(self.dataset_dict["valid"])
        )
        self.test_dataset = self.dataset_dict["test"]
        if self.train_weaver:
            self.weaver_train_dataset = self._filter_dataset(self.weaver_train_dataset)
            self.weaver_valid_dataset = self._filter_dataset(self.weaver_valid_dataset)
        if self.train_trigger:
            self.trigger_train_dataset = self._filter_dataset(
                self.trigger_train_dataset
            )
            self.trigger_valid_dataset = self._filter_dataset(
                self.trigger_valid_dataset
            )
        if self.env_cls.ENV_CARD == "STATIC":
            self.inter_cls = SingleTurnInteractionManager
        else:
            raise ValueError("Unsupported environment type.")
        self.generation_manager: InteractionManager = self.inter_cls(
            self.processing_class, self.model, self.interaction_config
        )

    def _parse_train_dataset(self, train_dataset: Dataset) -> tuple[Dataset, Dataset]:
        trigger_trainset_size = min(len(train_dataset) // 2, len(train_dataset))
        rand_indices = random.sample(range(len(train_dataset)), trigger_trainset_size)
        return (train_dataset, train_dataset.select(rand_indices))

    def _parse_valid_dataset(self, valid_dataset: Dataset) -> tuple[Dataset, Dataset]:
        trigger_validset_size = min(len(valid_dataset) // 2, len(valid_dataset))
        rand_indices = random.sample(range(len(valid_dataset)), trigger_validset_size)
        return (valid_dataset, valid_dataset.select(rand_indices))

    def _filter_dataset(self, dataset: Dataset) -> Dataset:
        tokenizer = self.processing_class
        max_len = 1024
        if self.train_weaver and self.train_weaver_method == "sft":
            max_len = self.weaver_sft_training_args.max_length
        elif self.train_weaver and self.train_weaver_method == "grpo":
            max_len = self.weaver_grpo_training_args.max_prompt_length
        elif self.train_trigger and self.train_trigger_method == "grpo":
            max_len = self.trigger_grpo_training_args.max_prompt_length
        else:
            raise ValueError("Wrong training mode.")

        def filter_func(sample):
            if "prompt" in sample and sample["prompt"] is not None:
                prompt = tokenizer.apply_chat_template(sample["prompt"], tokenize=True)
                return len(prompt) < max_len
            elif "messages" in sample and sample["messages"] is not None:
                conversation = tokenizer.apply_chat_template(
                    sample["messages"][:2], tokenize=True
                )
                return len(conversation) < max_len
            return True

        dataset = dataset.filter(filter_func)
        return dataset

    def _create_weaver_trainer(self):
        if self.train_weaver_method == "sft":
            weaver_trainer = SFTTrainer(
                model=self.model,
                args=self.weaver_sft_training_args,
                callbacks=[BestCheckpointSyncCallback()],
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
            )
        elif self.train_weaver_method == "grpo":
            weaver_trainer = WeaverGRPOTrainer(
                model=self.model,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.weaver_grpo_training_args,
                callbacks=[BestCheckpointSyncCallback()],
                train_dataset=self.weaver_train_dataset,
                eval_dataset=self.weaver_valid_dataset,
                processing_class=self.processing_class,
                env_class=self.env_cls,
                env_main_config=self.config.get("dataset"),
                generation_manager=self.generation_manager,
            )
        else:
            raise ValueError("Unsupported weaver training method.")
        return weaver_trainer

    def _train_weaver(self):
        self.model.fix_component("trigger")
        self.model.open_component("weaver")
        log_trainable_params(self.model)
        weaver_trainer = self._create_weaver_trainer()
        weaver_trainer.train()
        weaver_trainer.save_model()
        output_dir = weaver_trainer.args.output_dir
        weaver_trainer.accelerator.wait_for_everyone()
        if weaver_trainer.is_world_process_zero():
            remove_trainer_checkpoints(output_dir)
        weaver_trainer.accelerator.wait_for_everyone()

    def _create_trigger_trainer(self):
        if self.train_trigger_method == "grpo":
            trainer_cls = TriggerGRPOTrainer
            if self.model.config.compress_latent_memory:
                from memgen.trainer.joint_compression_grpo_trainer import (
                    JointCompressionGRPOTrainer,
                )

                trainer_cls = JointCompressionGRPOTrainer
            trigger_trainer = trainer_cls(
                model=self.model,
                processing_class=self.processing_class,
                train_dataset=self.trigger_train_dataset,
                eval_dataset=self.trigger_valid_dataset,
                reward_funcs=[self.env_cls.compute_reward],
                args=self.trigger_grpo_training_args,
                callbacks=[BestCheckpointSyncCallback()],
            )
            trigger_trainer.generate_sub_batch = self._trigger_generate_sub_batch
        else:
            raise ValueError("Unsupported trigger training method.")
        return trigger_trainer

    def _train_trigger(self):
        self.model.fix_component("weaver")
        self.model.open_component("trigger")
        if self.model.config.compress_latent_memory:
            self.model.configure_joint_trainables()
        log_trainable_params(self.model)
        trigger_trainer = self._create_trigger_trainer()
        trigger_trainer.train()
        trigger_trainer.save_model()
        output_dir = trigger_trainer.args.output_dir
        trigger_trainer.accelerator.wait_for_everyone()
        if trigger_trainer.is_world_process_zero():
            remove_trainer_checkpoints(output_dir)
        trigger_trainer.accelerator.wait_for_everyone()

    def train(self):
        if self.train_weaver:
            self._train_weaver()
        if self.train_trigger:
            self._train_trigger()

    def evaluate(self):
        self.model = self.model.to(torch.bfloat16)
        self.model.fix_component("weaver")
        self.model.fix_component("trigger")
        evaluate_func_mapping = {"STATIC": self._static_evaluate}
        evaluate_func = evaluate_func_mapping.get(self.env.ENV_CARD)
        if evaluate_func is None:
            raise ValueError("The env has unrecogonized ENV_CARD attribute")
        return evaluate_func()

    def _static_evaluate(self):
        accelerator = Accelerator()
        output_dir = self.interaction_config.output_dir
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
        if self.eval_repeats <= 1:
            self._set_eval_seed(self.eval_base_seed)
            save_file = os.path.join(output_dir, "answer.json")
            mean_metrics = self._static_evaluate_once(save_file=save_file)
            if not accelerator.is_main_process:
                return
            summary_record = {
                "eval_repeats": 1,
                "eval_seed_stride": self.eval_seed_stride,
                "per_repeat_metrics": [
                    {"repeat": 1, "seed": self.eval_base_seed, "metrics": mean_metrics}
                ],
                "summary_metrics": mean_metrics,
            }
            summary_file = os.path.join(output_dir, "answer_summary.json")
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
            logging.info(
                "Single static evaluation finished: save_file=%s, summary_file=%s, metrics=%s",
                save_file,
                summary_file,
                mean_metrics,
            )
            return
        per_repeat_metrics = []
        for repeat_idx in range(self.eval_repeats):
            repeat_id = repeat_idx + 1
            repeat_seed = self.eval_base_seed + repeat_idx * self.eval_seed_stride
            self._set_eval_seed(repeat_seed)
            save_file = os.path.join(output_dir, f"answer_repeat{repeat_id}.json")
            mean_metrics = self._static_evaluate_once(save_file=save_file)
            per_repeat_metrics.append(
                {"repeat": repeat_id, "seed": repeat_seed, "metrics": mean_metrics}
            )
        if not accelerator.is_main_process:
            return
        metric_names = (
            per_repeat_metrics[0]["metrics"].keys() if per_repeat_metrics else []
        )
        summary_metrics = {}
        for metric_name in metric_names:
            values = [item["metrics"][metric_name] for item in per_repeat_metrics]
            summary_metrics[metric_name] = float(sum(values) / len(values))
        summary_record = {
            "eval_repeats": self.eval_repeats,
            "eval_seed_stride": self.eval_seed_stride,
            "per_repeat_metrics": per_repeat_metrics,
            "summary_metrics": summary_metrics,
        }
        summary_file = os.path.join(output_dir, "answer_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(summary_record, ensure_ascii=False) + "\n")
        answer_file = os.path.join(output_dir, "answer.json")
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps({"summary_metrics": summary_metrics}, ensure_ascii=False)
                + "\n"
            )
        logging.info(
            "Repeated static evaluation finished: repeats=%s, average metrics=%s",
            self.eval_repeats,
            summary_metrics,
        )

    def _set_eval_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _static_evaluate_once(self, save_file: str) -> dict:
        accelerator = Accelerator()
        batch_size = self.interaction_config.batch_size
        total_batches = (len(self.test_dataset) + batch_size - 1) // batch_size
        batch_ids = range(
            accelerator.process_index, total_batches, accelerator.num_processes
        )
        model_wrapped = accelerator.prepare_model(
            model=self.model, evaluation_mode=True
        )
        model_wrapped.eval()
        local_results = []
        for batch_id in tqdm(batch_ids):
            start = batch_id * batch_size
            test_batch = [
                self.test_dataset[i]
                for i in range(start, min(start + batch_size, len(self.test_dataset)))
            ]
            with unwrap_model_for_generation(
                model_wrapped, accelerator
            ) as unwrapped_model:
                prompts = [x["prompt"] for x in test_batch]
                prompt_inputs = self.processing_class.apply_chat_template(
                    prompts,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=True,
                    return_dict=True,
                )
                prompt_ids, prompt_mask = (
                    prompt_inputs["input_ids"],
                    prompt_inputs["attention_mask"],
                )
                gen_batch = InteractionDataProto()
                gen_batch.batch["input_ids"] = prompt_ids.to(accelerator.device)
                gen_batch.batch["attention_mask"] = prompt_mask.to(accelerator.device)
                gen_batch.no_tensor_batch["initial_prompts"] = [
                    x["prompt"] for x in test_batch
                ]
                self.generation_manager.actor_rollout_wg = unwrapped_model
                gen_output = self.generation_manager.run_agent_loop(gen_batch)
                completion_ids = gen_output.batch["responses"]
                completions = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=True
                )
            if len(completions) != len(test_batch):
                raise RuntimeError(
                    "Generation did not return one completion per example"
                )
            local_results.append((batch_id, completions, test_batch))
        if accelerator.num_processes > 1:
            gathered = (
                [None] * accelerator.num_processes
                if accelerator.is_main_process
                else None
            )
            torch.distributed.gather_object(
                local_results, object_gather_list=gathered, dst=0
            )
        else:
            gathered = [local_results]
        if not accelerator.is_main_process:
            return {}
        ordered = sorted(
            (batch for rank in gathered for batch in rank), key=lambda batch: batch[0]
        )
        if [batch[0] for batch in ordered] != list(range(total_batches)):
            raise RuntimeError(
                "Distributed evaluation has missing or duplicate batches"
            )
        writer = create_tensorboard(save_dir=self.working_dir)
        try:
            recorder = StaticEvalRecorder(
                compute_metrics=[self.env_cls.compute_reward],
                writer=writer,
                log_file=save_file,
            )
            for _, completions, examples in ordered:
                recorder.record_batch(completions, examples)
            mean_metrics = recorder.get_mean_metrics()
            recorder.finalize()
            return mean_metrics
        finally:
            writer.close()

    def _parse_configs(self, configs):
        self.train_weaver = configs.get("train_weaver", True)
        self.train_trigger = configs.get("train_trigger", False)
        self.train_weaver_method = configs.get("train_weaver_method", "sft")
        if self.train_weaver:
            if self.train_weaver_method not in ["sft", "grpo"]:
                raise ValueError("Unsupported weaver training method.")
            weaver_config = configs.get("weaver", dict())
            if self.train_weaver_method == "sft":
                weaver_sft_config = weaver_config.get("sft", dict())
                self.weaver_sft_training_args = SFTConfig(**weaver_sft_config)
                self.weaver_sft_training_args.output_dir = os.path.join(
                    self.working_dir, "weaver"
                )
            else:
                self.weaver_sft_training_args = None
            if self.train_weaver_method == "grpo":
                weaver_grpo_config = weaver_config.get("grpo", dict())
                self.weaver_grpo_training_args = GRPOConfig(**weaver_grpo_config)
                self.weaver_grpo_training_args.output_dir = os.path.join(
                    self.working_dir, "weaver"
                )
            else:
                self.weaver_grpo_training_args = None
        else:
            self.weaver_sft_training_args = None
            self.weaver_grpo_training_args = None
        self.train_trigger_method = configs.get("train_trigger_method", "grpo")
        if self.train_trigger:
            if self.train_trigger_method not in ["grpo"]:
                raise ValueError("Unsupported trigger training method.")
            trigger_config = configs.get("trigger", dict())
            if self.train_trigger_method == "grpo":
                trigger_grpo_config = trigger_config.get("grpo", dict())
                self._trigger_generate_sub_batch = trigger_grpo_config.pop(
                    "generate_sub_batch", False
                )
                self.trigger_grpo_training_args = GRPOConfig(**trigger_grpo_config)
                self.trigger_grpo_training_args.output_dir = os.path.join(
                    self.working_dir, "trigger"
                )
            else:
                self.trigger_grpo_training_args = None
        else:
            self.trigger_grpo_training_args = None
        interaction_configs = configs.get("interaction", {})
        interaction_eval_cfg = interaction_configs.get("eval", {})
        temperature = interaction_configs.get("temperature", 1.0)
        if isinstance(temperature, str):
            try:
                temperature = float(temperature)
            except ValueError:
                logging.warning(
                    f"Invalid run.interaction.temperature={temperature!r}; falling back to 1.0. (Check your launch script for unset variables.)"
                )
                temperature = 1.0
        repetition_penalty = interaction_configs.get("repetition_penalty", 1.0)
        if isinstance(repetition_penalty, str):
            try:
                repetition_penalty = float(repetition_penalty)
            except ValueError:
                logging.warning(
                    f"Invalid run.interaction.repetition_penalty={repetition_penalty!r}; falling back to 1.0."
                )
                repetition_penalty = 1.0
        self.interaction_config = InteractionConfig(
            max_turns=interaction_configs.get("max_turns", 30),
            max_start_length=interaction_configs.get("max_start_length", 1024),
            max_prompt_length=interaction_configs.get("max_prompt_length", 4096),
            max_response_length=interaction_configs.get("max_response_length", 512),
            max_obs_length=interaction_configs.get("max_obs_length", 512),
            do_sample=interaction_configs.get("do_sample", False),
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            batch_size=interaction_configs.get("batch_size", 32),
            output_dir=os.path.join(self.working_dir, "evaluate"),
        )
        eval_repeats = interaction_configs.get(
            "eval_repeats", interaction_eval_cfg.get("repeats", 1)
        )
        eval_seed_stride = interaction_configs.get(
            "eval_seed_stride", interaction_eval_cfg.get("seed_stride", 1)
        )
        self.eval_repeats = max(1, int(eval_repeats))
        self.eval_seed_stride = max(1, int(eval_seed_stride))
        self.eval_base_seed = int(configs.get("seed", 42))
