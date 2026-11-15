from data.base_builder import BaseBuilder
from data.base_env import BaseEnv, StaticEnv
from data.gsm8k.builder import GSM8KBuilder
from data.gpqa.builder import GPQABuilder
from data.svamp.builder import SVAMPBuilder
from data.strategyqa.builder import StrategyQABuilder

_DATA_BUILDER_MAP = {
    'gsm8k': GSM8KBuilder,
    'gpqa': GPQABuilder,
    'svamp': SVAMPBuilder,
    'strategyqa': StrategyQABuilder,
}


def get_data_builder(dataset_cfg) -> BaseBuilder:
    name = dataset_cfg.get('name')
    if name not in _DATA_BUILDER_MAP:
        raise ValueError(f'Unsupported dataset: {name}')
    return _DATA_BUILDER_MAP[name](dataset_cfg)
