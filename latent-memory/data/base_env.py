from abc import ABC, abstractmethod
from typing import Literal, Dict, Tuple

class BaseEnv(ABC):
    ENV_CARD: Literal['STATIC'] = None

    def __init__(self, config):
        self.config = config

    @classmethod
    @abstractmethod
    def compute_reward(cls, **kwargs):
        ...

class StaticEnv(BaseEnv):
    ENV_CARD = 'STATIC'
