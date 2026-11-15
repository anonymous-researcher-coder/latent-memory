import json
import logging
from omegaconf import OmegaConf


class Config:

    def __init__(self, args):
        self.config = {}
        self.args = args
        user_config = self._build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.cfg_path)
        runner_config = self.build_runner_config(config, **user_config)
        model_config = self.build_model_config(config, **user_config)
        dataset_config = self.build_dataset_config(config, **user_config)
        self.config = OmegaConf.merge(
            runner_config, model_config, dataset_config, user_config
        )
        run_seed = self.config.run.get("seed", 42)
        for component, method in (
            ("weaver", "sft"),
            ("weaver", "grpo"),
            ("trigger", "grpo"),
        ):
            trainer_config = self.config.run.get(component, {}).get(method)
            if trainer_config is not None:
                trainer_config.setdefault("seed", run_seed)
                trainer_config.setdefault("data_seed", trainer_config["seed"])
        compression = self.config.model.get("compress_latent_memory", False)
        if not isinstance(compression, bool):
            raise ValueError("model.compress_latent_memory must be True or False")
        if compression:
            group_size = self.config.model.get("compressor", {}).get("group_size", 4)
            if type(group_size) is not int or group_size < 1:
                raise ValueError(
                    "model.compressor.group_size must be a positive integer"
                )
            if any(
                (
                    self.config.model.weaver[k] % group_size
                    for k in ("prompt_latents_len", "inference_latents_len")
                )
            ):
                raise ValueError(
                    "model.compressor.group_size must divide both Weaver memory lengths"
                )
            if self.config.run.mode == "train" and (
                self.config.run.get("train_weaver", True)
                or not self.config.run.get("train_trigger", False)
            ):
                raise ValueError(
                    "Learned compression is trained only with Trigger GRPO and a frozen SFT Weaver"
                )
            checkpoint = self.config.model.get("load_model_path")
            if not checkpoint or str(checkpoint).lower() in ("null", "none"):
                raise ValueError(
                    "Compression requires an existing SFT Weaver or joint checkpoint"
                )
            import os

            with open(
                os.path.join(str(checkpoint), "config.json"), encoding="utf-8"
            ) as f:
                saved = json.load(f)
            if self.config.run.mode == "evaluate" and (
                not saved.get("compress_latent_memory", False)
            ):
                raise ValueError(
                    "Compressed evaluation requires a trained joint-compression checkpoint"
                )
            if (
                saved.get("compress_latent_memory", False)
                and saved.get("compressor_group_size", 4) != group_size
            ):
                raise ValueError(
                    "Compressor group_size mismatch; start a new ratio from a Weaver SFT checkpoint"
                )
        try:
            if (
                "trigger" in self.config.model
                and "epsilon_greedy" not in self.config.model.trigger
            ):
                self.config.model.trigger.epsilon_greedy = 0.0
        except Exception:
            pass
        try:
            random_layers = None
            if (
                "weaver" in self.config.model
                and "random_memory_layers" in self.config.model.weaver
            ):
                raw = self.config.model.weaver.random_memory_layers
                if isinstance(raw, str) and raw.strip().lower() not in (
                    "null",
                    "none",
                    "",
                ):
                    random_layers = [x.strip() for x in raw.split(",") if x.strip()]
                elif isinstance(raw, (list, tuple)):
                    random_layers = list(raw)
            if random_layers is not None and len(random_layers) >= 2:
                self.config.model.trigger_num_actions = 3
            else:
                self.config.model.trigger_num_actions = 2
        except Exception:
            pass

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)

    @staticmethod
    def build_model_config(config, **kwargs):
        return {"model": OmegaConf.merge({"isolate_adapters": True}, config.model)}

    @staticmethod
    def build_runner_config(config, **kwargs):
        return {"run": config.run}

    @staticmethod
    def build_dataset_config(config, **kwargs):
        dataset = config.get("dataset", None)
        if dataset is None:
            raise KeyError(
                "Expecting 'dataset' as the root key for dataset configuration."
            )
        return dict(dataset=dataset)

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []
        if len(opts) == 0:
            return opts
        result = []
        i = 0
        while i < len(opts):
            if "=" in opts[i]:
                result.append(opts[i])
                i += 1
            elif i + 1 < len(opts):
                result.append(f"{opts[i]}={opts[i + 1]}")
                i += 2
            else:
                result.append(opts[i])
                i += 1
        return result

    def get_config(self):
        return self.config

    @property
    def run_cfg(self):
        return self.config.run

    @property
    def dataset_cfg(self):
        return self.config.dataset

    @property
    def model_cfg(self):
        return self.config.model

    def pretty_print(self):
        logging.info("\n=====  Running Parameters    =====")
        logging.info(self._convert_node_to_json(self.config.run))
        logging.info("\n======  Dataset Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.dataset))
        logging.info(f"\n======  Model Attributes  ======")
        logging.info(self._convert_node_to_json(self.config.model))

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def to_dict(self):
        return OmegaConf.to_container(self.config)
