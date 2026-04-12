"""
Checkpoint manager - saves/loads agent + buffer + training state.

Checkpoints are saved every `save_every` steps. The last `keep` checkpoints
are retained; older ones are removed to save disk space.

Usage:
    ckpt = CheckpointManager(logdir="logs/.../seed_0", save_every=50_000)
    ckpt.save(step=50000, agent=agent, buffer=buffer, metrics={...})
    step = ckpt.load_latest(agent, buffer)
"""

import glob
import os
import re
from typing import Optional

import torch


class CheckpointManager:
    """
    Manages periodic checkpointing of agent and replay buffer.

    Args:
        logdir: directory where checkpoints are written.
        save_every: save a checkpoint every this many steps.
        keep: number of recent checkpoints to retain (older ones deleted).
    """

    def __init__(self, logdir: str, save_every: int = 50_000, keep: int = 3):
        self.logdir = logdir
        self.save_every = save_every
        self.keep = keep
        self._ckpt_dir = os.path.join(logdir, 'checkpoints')
        os.makedirs(self._ckpt_dir, exist_ok=True)
        self._last_saved = -1

    def should_save(self, step: int) -> bool:
        return step > 0 and step % self.save_every == 0

    def save(self, step: int, agent, buffer, metrics: dict = None):
        """Serialise agent state_dict, buffer state, and step count to disk."""
        fname = os.path.join(self._ckpt_dir, f'checkpoint_{step:07d}.pt')
        payload = {
            'step': step,
            'agent': agent.state_dict(),
            'buffer': buffer.state_dict(),
            'metrics': metrics or {},
        }
        torch.save(payload, fname)
        self._last_saved = step
        self._prune()
        return fname

    def load(self, path: str, agent, buffer) -> int:
        """Restore agent and buffer from a specific checkpoint file.

        Returns:
            The training step at which the checkpoint was saved.
        """
        payload = torch.load(path, map_location='cpu', weights_only=False)
        agent.load_state_dict(payload['agent'])
        buffer.load_state_dict(payload['buffer'])
        return int(payload['step'])

    def load_latest(self, agent, buffer) -> Optional[int]:
        """
        Find and load the most recent checkpoint.

        Returns:
            The step number if a checkpoint was found, else None.
        """
        path = self.latest()
        if path is None:
            return None
        return self.load(path, agent, buffer)

    def latest(self) -> Optional[str]:
        """Return the path of the most recent checkpoint, or None."""
        pattern = os.path.join(self._ckpt_dir, 'checkpoint_*.pt')
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=self._step_from_path)

    def _step_from_path(self, path: str) -> int:
        m = re.search(r'checkpoint_(\d+)\.pt$', path)
        return int(m.group(1)) if m else 0

    def _prune(self):
        """Keep `keep` checkpoints spread evenly across training steps.

        Rather than discarding the oldest checkpoints (which removes early
        snapshots needed for temporal evolution analysis), we select `keep`
        files whose steps are as evenly spaced as possible from first to last,
        always retaining the most recent checkpoint.
        """
        pattern = os.path.join(self._ckpt_dir, 'checkpoint_*.pt')
        files = sorted(glob.glob(pattern), key=self._step_from_path)
        if len(files) <= self.keep:
            return

        # Choose keep indices spaced evenly from 0 to len-1, always include last
        if self.keep == 1:
            keep_indices = {len(files) - 1}
        else:
            keep_indices = {
                round(i * (len(files) - 1) / (self.keep - 1))
                for i in range(self.keep)
            }
            keep_indices.add(len(files) - 1)  # always keep newest

        for i, f in enumerate(files):
            if i not in keep_indices:
                try:
                    os.remove(f)
                except OSError:
                    pass
