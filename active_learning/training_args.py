import json
from pathlib import Path
from typing import Optional

import dataclasses
import torch
from dataclasses import dataclass, field

from configuration.config import data_dir
from utils.file_utils import cache_property


@dataclass
class TrainingArguments:
    output_dir: str = field(default=str(Path(data_dir)/'outputs'), metadata={'help': 'output dir'})

    gradient_accumulation_steps: int = field(default=1, metadata={'help': 'gradient accumulation steps'})
    max_gradient_norm: float = field(default=1.0, metadata={'help': 'max gradient norm'})

    learning_rate: float = field(default=5e-5, metadata={'help': 'learning rate'})
    warmup_steps: int = field(default=0, metadata={'help': 'warmup_steps'})
    weight_decay: float = field(default=0.0, metadata={'help': 'weight_decay'})

    epoch_num: int = field(default=100, metadata={'help': 'epoch number'})
    per_gpu_batch_size: int = field(default=16, metadata={'help': 'batch size per gpu'})

    seed: int = field(default=32, metadata={'help': 'seed'})

    logging_steps: int = field(default=100, metadata={'help': 'Log every X update steps'})
    saving_steps: int = field(default=500, metadata={'help': 'save checkpoint every X update steps'})
    save_total_limit: Optional[int] = field(default=2, metadata={'help': 'limit the total number of checkpoints'})

    threshold: float = field(default=0, metadata={'help': 'inference threshold'})

    server_ip: str = field(default='', metadata={'help': 'can be used for distant debugging'})
    server_port: str = field(default='', metadata={'help': 'can be used for distant debugging'})

    @property
    def batch_size(self):
        return self.per_gpu_batch_size * max(1, self.n_gpu)

    @cache_property
    def _setup_devices(self):
        if torch.cuda.is_available():
            n_gpu = torch.cuda.device_count()
            device = torch.device('cuda')
        else:
            n_gpu = 0
            device = torch.device('cpu')

        return n_gpu, device

    @property
    def n_gpu(self):
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return n_gpu
        # return self._setup_devices[0]

    @property
    def device(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        return device
        # return self._setup_devices[1]

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self):
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}










