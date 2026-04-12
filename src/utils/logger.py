"""
Metrics logger - writes to TensorBoard and a JSONL file.

Usage:
    logger = Logger(logdir="logs/explicit/dmc_cartpole_swingup/seed_0")
    logger.log(step=1000, metrics={"train/reward_loss": 0.5, "eval/return": 120.0})
    logger.close()
"""

import json
import os
import time
from typing import Dict

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


class Logger:
    """
    Dual TensorBoard + JSONL logger.

    Args:
        logdir: root directory for logs (will be created if needed).
        use_tb: whether to write to TensorBoard (requires tensorboard package).
    """

    def __init__(self, logdir: str, use_tb: bool = True):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)
        self._jsonl_path = os.path.join(logdir, 'metrics.jsonl')
        self._jsonl_file = open(self._jsonl_path, 'a')
        self._writer = None
        if use_tb and TB_AVAILABLE:
            self._writer = SummaryWriter(logdir)
        self._start_time = time.time()

    def log(self, step: int, metrics: Dict[str, float]):
        """Write a dict of scalar metrics at a given step."""
        row = {'step': step, 'time': time.time() - self._start_time}
        row.update({k: float(v) for k, v in metrics.items()})
        # JSONL
        self._jsonl_file.write(json.dumps(row) + '\n')
        self._jsonl_file.flush()
        # TensorBoard
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, float(v), global_step=step)

    def print_metrics(self, step: int, metrics: Dict[str, float], prefix: str = ''):
        """Print a concise summary to stdout."""
        parts = [f'step={step}']
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                parts.append(f'{k}={v:.4g}')
            else:
                parts.append(f'{k}={v}')
        print(f'[{prefix or "log"}] ' + ' | '.join(parts))

    def close(self):
        self._jsonl_file.close()
        if self._writer is not None:
            self._writer.close()
