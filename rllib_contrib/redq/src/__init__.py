from ray.tune.registry import register_trainable
from rllib_redq.redq.redq import REDQ
from rllib_redq.redq.redq import REDQConfig

__all__ = ["REDQConfig", "REDQ"]

register_trainable("rllib-contrib-redq", REDQ)
