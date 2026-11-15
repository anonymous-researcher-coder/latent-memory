import argparse
import logging
import os
import random
import numpy as np
import torch
from accelerate import PartialState
from accelerate.utils import broadcast_object_list
from common.config import Config
from common.logger import setup_logger
from common.run_naming import build_run_name
from data import get_data_builder
from memgen.model.modeling_memgen import MemGenModel
from memgen.runner import MemGenRunner


def set_seed(random_seed: int, use_gpu: bool):
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"set seed: {random_seed}")


def parse_args():
    parser = argparse.ArgumentParser(description="Memory Generator")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def build_working_dir(config: Config) -> str:
    mode = config.run_cfg.mode
    dataset_name = config.dataset_cfg.name
    model_name = os.path.basename(config.model_cfg.model_name.rstrip("/"))
    parent_dir = os.path.join("results", mode, dataset_name, model_name)
    working_dir = build_run_name(config.to_dict())
    return os.path.join(parent_dir, working_dir)


def main():
    args = parse_args()
    config = Config(args)
    distributed_state = PartialState()
    set_seed(config.run_cfg.seed, use_gpu=True)
    working_dir = broadcast_object_list(
        [build_working_dir(config) if distributed_state.is_main_process else None]
    )[0]
    config.run_cfg.log_dir = os.path.join(working_dir, "logs")
    setup_logger(output_dir=config.run_cfg.log_dir)
    config.pretty_print()
    config_dict = config.to_dict()
    data_builder = get_data_builder(config_dict.get("dataset"))
    model = MemGenModel.from_config(config_dict.get("model"))
    runner = MemGenRunner(
        model=model,
        data_builder=data_builder,
        config=config_dict,
        working_dir=working_dir,
    )
    if config.run_cfg.mode == "train":
        runner.train()
    elif config.run_cfg.mode == "evaluate":
        runner.evaluate()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("MemGen run failed")
        raise
