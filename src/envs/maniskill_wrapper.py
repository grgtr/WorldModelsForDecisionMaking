"""
ManiSkill2 environment wrapper - state observations only.

Supported tasks (obs_mode="state"):
  PickCube-v1, PushCube-v1, StackCube-v1
  CartPole-v1, Walker2d-v2, HalfCheetah-v2 (ManiSkill reimplementations)

Requires: mani_skill2 (or mani_skill)
"""

import numpy as np

MANISKILL_AVAILABLE = False
try:
    import mani_skill.envs  # noqa: F401
    MANISKILL_AVAILABLE = True
    _MS_MODULE = 'mani_skill'
except ImportError:
    try:
        import mani_skill2.envs  # noqa: F401
        MANISKILL_AVAILABLE = True
        _MS_MODULE = 'mani_skill2'
    except ImportError:
        pass


class ManiSkillEnv:
    """
    ManiSkill2 wrapper returning flat state observations.

    Args:
        env_id: ManiSkill2 environment id ('PickCube-v1').
        action_repeat: number of action repeats.
        seed: random seed.
        obs_mode: observation mode (should be 'state').
        control_mode: robot control mode.
    """

    def __init__(self, env_id: str, action_repeat: int = 2, seed: int = 0,
                 obs_mode: str = 'state', control_mode: str = 'pd_joint_delta_pos'):
        if not MANISKILL_AVAILABLE:
            raise ImportError("mani_skill2 or mani_skill is required.")

        try:
            import gymnasium as gym
        except ImportError:
            import gym

        self._action_repeat = action_repeat
        self._seed = seed

        env_kwargs = dict(obs_mode=obs_mode, render_mode=None)
        if 'Cube' in env_id or 'Pick' in env_id or 'Push' in env_id or 'Stack' in env_id:
            env_kwargs['control_mode'] = control_mode

        self._env = gym.make(env_id, **env_kwargs)

        # Probe observation space
        obs_space = self._env.observation_space
        if hasattr(obs_space, 'shape'):
            self.obs_dim = int(np.prod(obs_space.shape))
        else:
            # Dict observation space — flatten all numeric sub-spaces
            obs = self._env.reset()[0] if hasattr(self._env.reset(), '__len__') \
                  else self._env.reset()
            self.obs_dim = self._flatten_obs(obs).shape[0]

        act_space = self._env.action_space
        self.act_dim = int(np.prod(act_space.shape))
        self.act_continuous = True
        self._act_low = act_space.low.astype(np.float32)
        self._act_high = act_space.high.astype(np.float32)

    def _flatten_obs(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            parts = []
            for v in obs.values():
                if isinstance(v, np.ndarray):
                    parts.append(v.astype(np.float32).flatten())
                elif isinstance(v, dict):
                    parts.append(self._flatten_obs(v))
            return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
        return np.atleast_1d(obs).astype(np.float32).flatten()

    def reset(self) -> np.ndarray:
        result = self._env.reset(seed=self._seed)
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
        return self._flatten_obs(obs)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        # Rescale to env action space
        action = (action + 1.0) / 2.0 * (self._act_high - self._act_low) + self._act_low

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

        return self._flatten_obs(obs), reward, done, info

    def random_action(self) -> np.ndarray:
        return np.random.uniform(-1.0, 1.0, size=(self.act_dim,)).astype(np.float32)

    def close(self):
        self._env.close()
