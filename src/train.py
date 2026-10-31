import contextlib
import torch
from loguru import logger
from src.model_args import DeepSeekV3ModelArgs
from torch.distributed.checkpoint.stateful import Stateful




class Trainer(Stateful):
    job_config: job