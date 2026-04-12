"""
Unified environment factory.

Usage:
    env = make_env("dmc_cartpole_swingup", seed=0, action_repeat=2)
    env = make_env("maniskill_PickCube-v1", seed=0)
    env = make_env("gym_HalfCheetah-v4", seed=0)

All returned environments have:
    env.obs_dim          : int - flat observation dimension
    env.act_dim          : int - flat action dimension
    env.act_continuous   : bool - True for continuous, False for discrete
    env.reset()          -> np.ndarray[float32]   (obs_dim,)
    env.step(action)     -> (obs, reward, done, info)
    env.random_action()  -> np.ndarray[float32]   (act_dim,)
    env.close()
"""


# Supported environment IDs

# DMControl: "<domain>_<task>"
DMC_ENVS = {
    'dmc_cartpole_swingup':   ('cartpole', 'swingup'),
    'dmc_cartpole_balance':   ('cartpole', 'balance'),
    'dmc_walker_walk':        ('walker',   'walk'),
    'dmc_walker_run':         ('walker',   'run'),
    'dmc_cheetah_run':        ('cheetah',  'run'),
    'dmc_hopper_hop':         ('hopper',   'hop'),
    'dmc_humanoid_walk':      ('humanoid', 'walk'),
    'dmc_reacher_easy':       ('reacher',  'easy'),
    'dmc_reacher_hard':       ('reacher',  'hard'),
    'dmc_finger_spin':        ('finger',   'spin'),
    'dmc_pendulum_swingup':   ('pendulum', 'swingup'),
    'dmc_acrobot_swingup':    ('acrobot',  'swingup'),
    'dmc_quadruped_walk':     ('quadruped','walk'),
    'dmc_quadruped_run':      ('quadruped','run'),
    'dmc_swimmer_swimmer15':  ('swimmer',  'swimmer15'),
}

# ManiSkill2: "maniskill_<env_id>"  (mani_skill2 legacy API, v0 IDs)
MANISKILL_ENVS = {
    'maniskill_LiftCube-v0':        'LiftCube-v0',
    'maniskill_PickCube-v0':        'PickCube-v0',
    'maniskill_StackCube-v0':       'StackCube-v0',
    'maniskill_PegInsertionSide-v0':'PegInsertionSide-v0',
}

# Gymnasium: "gym_<env_id>"  (standard gymnasium/mujoco envs)
GYM_ENVS = {
    'gym_CartPole-v1':      ('CartPole-v1',      1),   # (env_id, action_repeat)
    'gym_HalfCheetah-v4':   ('HalfCheetah-v4',   2),
    'gym_Walker2d-v4':      ('Walker2d-v4',       2),
    'gym_Hopper-v4':        ('Hopper-v4',         2),
    'gym_Ant-v4':           ('Ant-v4',            2),
    'gym_Humanoid-v4':      ('Humanoid-v4',       2),
    'gym_HalfCheetah-v2':   ('HalfCheetah-v2',   2),
    'gym_Walker2d-v2':      ('Walker2d-v2',       2),
}


def make_env(env_id: str, seed: int = 0, action_repeat: int = None, **kwargs):
    """
    Create and return a standardised environment.

    Args:
        env_id: one of the supported IDs above ('dmc_cartpole_swingup').
        seed: random seed.
        action_repeat: override default action repeat.
        **kwargs: additional arguments forwarded to the wrapper.

    Returns:
        Environment object with standard interface.
    """
    env_id = env_id.strip()

    if env_id in DMC_ENVS:
        from .dmc_wrapper import DMControlEnv
        domain, task = DMC_ENVS[env_id]
        ar = action_repeat if action_repeat is not None else 2
        return DMControlEnv(domain, task, action_repeat=ar, seed=seed, **kwargs)

    elif env_id in MANISKILL_ENVS:
        from .maniskill_wrapper import ManiSkillEnv
        ms_id = MANISKILL_ENVS[env_id]
        ar = action_repeat if action_repeat is not None else 2
        return ManiSkillEnv(ms_id, action_repeat=ar, seed=seed, **kwargs)

    elif env_id in GYM_ENVS:
        from .gym_wrapper import GymEnv
        gym_id, default_ar = GYM_ENVS[env_id]
        ar = action_repeat if action_repeat is not None else default_ar
        return GymEnv(gym_id, action_repeat=ar, seed=seed, **kwargs)

    else:
        if env_id.startswith('dmc_'):
            # Parse as dmc_<domain>_<task>
            parts = env_id[4:].rsplit('_', 1)
            if len(parts) == 2:
                from .dmc_wrapper import DMControlEnv
                ar = action_repeat if action_repeat is not None else 2
                return DMControlEnv(parts[0], parts[1], action_repeat=ar,
                                    seed=seed, **kwargs)
        elif env_id.startswith('maniskill_'):
            from .maniskill_wrapper import ManiSkillEnv
            ms_id = env_id[len('maniskill_'):]
            ar = action_repeat if action_repeat is not None else 2
            return ManiSkillEnv(ms_id, action_repeat=ar, seed=seed, **kwargs)
        elif env_id.startswith('gym_'):
            from .gym_wrapper import GymEnv
            gym_id = env_id[4:]
            ar = action_repeat if action_repeat is not None else 1
            return GymEnv(gym_id, action_repeat=ar, seed=seed, **kwargs)

        raise ValueError(
            f"Unknown environment: '{env_id}'.\n"
            f"Supported prefixes: dmc_, maniskill_, gym_\n"
            f"Known IDs: {sorted(list(DMC_ENVS) + list(MANISKILL_ENVS) + list(GYM_ENVS))}"
        )
