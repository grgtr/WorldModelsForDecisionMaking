"""
Unified episode-based replay buffer supporting both agents.

Stores complete episodes as numpy arrays. Supports two sampling modes:
  - sample_sequences: contiguous windows of fixed length (explicit/RSSM agent)
  - sample_transitions: short trajectory slices of length horizon+1 (implicit agent)

"""

import os
import numpy as np
from typing import Dict, Optional


class EpisodeBuffer:
    """
    Replay buffer that stores complete episodes and samples windows from them.

    Episodes are stored as dictionaries mapping field names to numpy arrays of
    shape [T, ...] where T is the episode length.

    Capacity is enforced by removing oldest episodes (LRU) once the total
    number of stored transitions exceeds `capacity`.
    """

    def __init__(self, capacity: int = 500_000, seed: int = 0):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self._episodes: Dict[int, dict] = {}   # episode_id to field arrays
        self._order: list = []                  # episode ids in insertion order
        self._total: int = 0                    # total transitions stored
        self._next_id: int = 0
        self._current_ep: Optional[dict] = None  # episode being collected

    # Episode collection (one transition at a time)

    def start_episode(self):
        self._current_ep = {
            'obs': [], 'action': [], 'reward': [],
            'done': [], 'next_obs': []
        }

    def add_transition(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool):
        """Add a single transition to the episode being collected."""
        assert self._current_ep is not None, "Call start_episode() first."
        self._current_ep['obs'].append(obs.astype(np.float32))
        self._current_ep['action'].append(action.astype(np.float32))
        self._current_ep['reward'].append(np.float32(reward))
        self._current_ep['next_obs'].append(next_obs.astype(np.float32))
        self._current_ep['done'].append(np.bool_(done))

    def end_episode(self) -> int:
        """Finalise the current episode and add it to the buffer.

        Returns:
            episode_id of the stored episode.
        """
        assert self._current_ep is not None, "Call start_episode() first."
        ep = {k: np.stack(v, axis=0) for k, v in self._current_ep.items()}
        ep_len = len(ep['obs'])
        ep_id = self._next_id
        self._next_id += 1
        self._episodes[ep_id] = ep
        self._order.append(ep_id)
        self._total += ep_len
        self._current_ep = None
        # Evict oldest episodes if over capacity
        self._evict()
        return ep_id

    def _evict(self):
        """Remove oldest episodes until total transitions ≤ capacity."""
        while self._total > self.capacity and len(self._order) > 1:
            oldest_id = self._order.pop(0)
            ep = self._episodes.pop(oldest_id)
            self._total -= len(ep['obs'])

    # Batch-add a full episode (for prefill / loading)

    def add_episode(self, obs: np.ndarray, action: np.ndarray,
                    reward: np.ndarray, done: np.ndarray) -> int:
        """Add a pre-collected episode directly.

        Args:
            obs: [T, obs_dim] or [T+1, obs_dim] (if obs includes final obs)
            action: [T, act_dim]
            reward: [T]
            done: [T]

        Returns:
            episode_id
        """
        T = len(action)
        if len(obs) == T + 1:
            next_obs = obs[1:]
            obs = obs[:-1]
        else:
            next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
        ep = {
            'obs': obs[:T].astype(np.float32),
            'action': action[:T].astype(np.float32),
            'reward': reward[:T].astype(np.float32),
            'next_obs': next_obs[:T].astype(np.float32),
            'done': done[:T].astype(bool),
        }
        ep_id = self._next_id
        self._next_id += 1
        self._episodes[ep_id] = ep
        self._order.append(ep_id)
        self._total += T
        self._evict()
        return ep_id

    # Sampling for Explicit agent: contiguous seq windows

    def sample_sequences(self, batch_size: int, seq_len: int) -> dict:
        """
        Sample `batch_size` contiguous windows of length `seq_len`.

        Windows may start anywhere within an episode as long as the episode
        is at least `seq_len` steps long. Episodes shorter than `seq_len`
        are padded with boundary repetition (obs repeated) and `is_first=True`
        at the start.

        Returns a dict with numpy arrays of shape [batch_size, seq_len, ...]:
          obs, action, reward, done, is_first, next_obs
        """
        valid_ids = [eid for eid in self._order
                     if len(self._episodes[eid]['obs']) >= 2]
        if not valid_ids:
            raise RuntimeError("Buffer has no episodes with length >= 2.")

        batch = {k: [] for k in ('obs', 'action', 'reward', 'done',
                                  'next_obs', 'is_first')}
        for _ in range(batch_size):
            ep_id = self.rng.choice(valid_ids)
            ep = self._episodes[ep_id]
            T = len(ep['obs'])
            if T <= seq_len:
                # Pad at the beginning with the first frame
                pad = seq_len - T
                is_first = np.zeros(seq_len, dtype=bool)
                is_first[pad] = True
                for k in ('obs', 'action', 'reward', 'done', 'next_obs'):
                    arr = ep[k]
                    padded = np.concatenate(
                        [np.repeat(arr[:1], pad, axis=0), arr], axis=0)
                    batch[k].append(padded[:seq_len])
                batch['is_first'].append(is_first)
            else:
                start = self.rng.integers(0, T - seq_len + 1)
                is_first = np.zeros(seq_len, dtype=bool)
                is_first[0] = (start == 0)
                for k in ('obs', 'action', 'reward', 'done', 'next_obs'):
                    batch[k].append(ep[k][start:start + seq_len])
                batch['is_first'].append(is_first)

        return {k: np.stack(v, axis=0) for k, v in batch.items()}

    # Sampling for Implicit agent: short trajectory slices

    def sample_transitions(self, batch_size: int, horizon: int) -> dict:
        """
        Sample `batch_size` trajectory slices of length `horizon+1`.

        Returns a dict with numpy arrays of shape:
          obs:    [batch_size, horizon+1, obs_dim]
          action: [batch_size, horizon,   act_dim]
          reward: [batch_size, horizon]
          done:   [batch_size, horizon]
        """
        valid_ids = [eid for eid in self._order
                     if len(self._episodes[eid]['obs']) >= horizon + 1]
        if not valid_ids:
            # Fall back to any episode (will be padded)
            valid_ids = list(self._order)

        obs_list, act_list, rew_list, done_list = [], [], [], []
        for _ in range(batch_size):
            ep_id = self.rng.choice(valid_ids)
            ep = self._episodes[ep_id]
            T = len(ep['obs'])
            if T < horizon + 1:
                # Repeat last obs / zero action
                pad = horizon + 1 - T
                obs_slice = np.concatenate([ep['obs'],
                                            np.repeat(ep['obs'][-1:], pad, 0)], 0)
                act_slice = np.concatenate([ep['action'],
                                            np.zeros((pad,) + ep['action'].shape[1:])], 0)
                rew_slice = np.concatenate([ep['reward'],
                                            np.zeros(pad)], 0)
                done_slice = np.concatenate([ep['done'],
                                             np.ones(pad, dtype=bool)], 0)
            else:
                start = self.rng.integers(0, T - horizon)
                obs_slice = ep['obs'][start:start + horizon + 1]
                act_slice = ep['action'][start:start + horizon]
                rew_slice = ep['reward'][start:start + horizon]
                done_slice = ep['done'][start:start + horizon]

            obs_list.append(obs_slice)
            act_list.append(act_slice)
            rew_list.append(rew_slice)
            done_list.append(done_slice)

        return {
            'obs': np.stack(obs_list, axis=0),       # [B, H+1, O]
            'action': np.stack(act_list, axis=0),    # [B, H, A]
            'reward': np.stack(rew_list, axis=0),    # [B, H]
            'done': np.stack(done_list, axis=0),     # [B, H]
        }

    @property
    def num_transitions(self) -> int:
        return self._total

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def __len__(self) -> int:
        return self._total

    # Checkpointing

    def state_dict(self) -> dict:
        """Serialise buffer state (all episodes) to a Python dict."""
        return {
            'episodes': dict(self._episodes),
            'order': list(self._order),
            'total': self._total,
            'next_id': self._next_id,
        }

    def load_state_dict(self, state: dict):
        """Restore buffer from a serialised state dict."""
        self._episodes = dict(state['episodes'])
        self._order = list(state['order'])
        self._total = int(state['total'])
        self._next_id = int(state['next_id'])

    def save(self, path: str):
        """Save buffer to a numpy .npz file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        episodes_flat = {}
        for ep_id, ep in self._episodes.items():
            for k, v in ep.items():
                episodes_flat[f'{ep_id}_{k}'] = v
        np.savez_compressed(
            path,
            order=np.array(self._order, dtype=np.int64),
            total=np.array([self._total], dtype=np.int64),
            next_id=np.array([self._next_id], dtype=np.int64),
            ep_ids=np.array(list(self._episodes.keys()), dtype=np.int64),
            **episodes_flat
        )

    def load(self, path: str):
        """Load buffer from a numpy .npz file."""
        data = np.load(path, allow_pickle=False)
        self._order = list(data['order'])
        self._total = int(data['total'][0])
        self._next_id = int(data['next_id'][0])
        ep_ids = list(data['ep_ids'])
        self._episodes = {}
        for ep_id in ep_ids:
            ep = {}
            for k in ('obs', 'action', 'reward', 'next_obs', 'done'):
                key = f'{ep_id}_{k}'
                if key in data:
                    ep[k] = data[key]
            self._episodes[int(ep_id)] = ep
