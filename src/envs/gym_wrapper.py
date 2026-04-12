"""
Gymnasium environment wrapper - supports MuJoCo continuous control and CartPole.

Provides a unified interface (same as dmc_wrapper) for gymnasium environments.
Actions are normalised to [-1, 1]; for discrete envs like CartPole-v1 the
normalised range maps to {0, 1}.
"""

import numpy as np

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False


class GymEnv:
    """
    Gymnasium wrapper with action normalisation.

    Args:
        env_id: gymnasium environment id ('HalfCheetah-v4').
        action_repeat: number of action repeats.
        seed: random seed.
    """

    def __init__(self, env_id: str, action_repeat: int = 1, seed: int = 0):
        if not GYM_AVAILABLE:
            raise ImportError("gymnasium (or gym) is required.")
        try:
            import gymnasium as gym
        except ImportError:
            import gym
        self._env = gym.make(env_id)
        self._action_repeat = action_repeat
        self._seed = seed

        obs_space = self._env.observation_space
        self.obs_dim = int(np.prod(obs_space.shape))

        act_space = self._env.action_space
        if hasattr(act_space, 'n'):
            # Discrete
            self.act_dim = 1
            self.act_continuous = False
            self._n_actions = act_space.n
        else:
            self.act_dim = int(np.prod(act_space.shape))
            self.act_continuous = True
            self._act_low = act_space.low.astype(np.float32)
            self._act_high = act_space.high.astype(np.float32)

    def reset(self) -> np.ndarray:
        result = self._env.reset(seed=self._seed)
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        return obs.astype(np.float32).flatten()

    def step(self, action: np.ndarray):
        if self.act_continuous:
            # Rescale from [-1, 1] to env action range
            action = np.clip(action, -1.0, 1.0)
            action = (action + 1.0) / 2.0 * (self._act_high - self._act_low) + self._act_low
        else:
            # Discrete: action is scalar in [-1, 1]
            action = int(np.round((action.flatten()[0] + 1.0) / 2.0 * (self._n_actions - 1)))
            action = np.clip(action, 0, self._n_actions - 1)

        reward = 0.0
        for _ in range(self._action_repeat):
            result = self._env.step(action)
            if len(result) == 5:
                obs, r, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, r, done, info = result
            reward += float(r)
            if done:
                break

        return obs.astype(np.float32).flatten(), reward, done, info

    def random_action(self) -> np.ndarray:
        if self.act_continuous:
            return np.random.uniform(-1.0, 1.0, size=(self.act_dim,)).astype(np.float32)
        else:
            return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)

    def close(self):
        self._env.close()
