"""
DeepMind Control Suite wrapper - state observations only.

"""

import numpy as np

try:
    from dm_control import suite
    from dm_control.rl.control import Environment as DMEnv
    DMC_AVAILABLE = True
except ImportError:
    DMC_AVAILABLE = False


class DMControlEnv:
    """
    Wrapper around dm_control that returns concatenated proprioceptive state.

    Args:
        domain: dm_control domain name ('cartpole', 'walker').
        task: task name ('swingup', 'walk').
        action_repeat: number of times to repeat each action.
        seed: random seed.
        time_limit: override default episode length (seconds).
    """

    def __init__(self, domain: str, task: str, action_repeat: int = 2, seed: int = 0, time_limit: float = None):
        if not DMC_AVAILABLE:
            raise ImportError("dm_control is required for DMControl environments.")
        kwargs = {}
        if time_limit is not None:
            kwargs['time_limit'] = time_limit
        self._env: DMEnv = suite.load(
            domain_name=domain,
            task_name=task,
            task_kwargs={'random': seed},
            **kwargs
        )
        self._action_repeat = action_repeat

        # Observation and action spaces
        obs_spec = self._env.observation_spec()
        self.obs_dim = sum(
            int(np.prod(v.shape)) if v.shape else 1
            for v in obs_spec.values()
        )
        act_spec = self._env.action_spec()
        self.act_dim = int(np.prod(act_spec.shape))
        self.act_continuous = True
        self._act_min = act_spec.minimum.astype(np.float32)
        self._act_max = act_spec.maximum.astype(np.float32)

    def reset(self) -> np.ndarray:
        ts = self._env.reset()
        return self._extract_obs(ts)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        # Rescale from [-1, 1] to actual action range
        action = (action + 1.0) / 2.0 * (self._act_max - self._act_min) + self._act_min
        reward = 0.0
        for _ in range(self._action_repeat):
            ts = self._env.step(action)
            reward += ts.reward or 0.0
            if ts.last():
                break
        obs = self._extract_obs(ts)
        done = ts.last()
        info = {}
        return obs, reward, done, info

    def _extract_obs(self, ts) -> np.ndarray:
        obs_dict = ts.observation
        parts = []
        for v in obs_dict.values():
            arr = np.atleast_1d(v).astype(np.float32).flatten()
            parts.append(arr)
        return np.concatenate(parts)

    def random_action(self) -> np.ndarray:
        # Returns action in [-1, 1]
        return np.random.uniform(-1.0, 1.0, size=(self.act_dim,)).astype(np.float32)

    def close(self):
        self._env.close()
