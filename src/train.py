"""
Main training entry point for the World Models Research Suite.

Usage:
    # Explicit (RSSM/DreamerV3) agent
    python src/train.py --agent explicit --env dmc_cartpole_swingup --seed 0
    python src/train.py --agent explicit --env dmc_walker_walk --seed 1 --total_steps 500000

    # Implicit (TD-MPC2) agent
    python src/train.py --agent implicit --env dmc_cartpole_swingup --seed 0
    python src/train.py --agent implicit --env maniskill_PickCube-v1 --seed 0

    # Resume from latest checkpoint
    python src/train.py --agent explicit --env dmc_walker_walk --seed 0 --resume

Environment variables for headless rendering on A100:
    export MUJOCO_GL=egl
    export VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
"""

import argparse
import os
import sys
import time
import types

import numpy as np
import yaml

# Make src/ importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.factory import make_env
from src.buffer.episode_buffer import EpisodeBuffer
from src.utils.logger import Logger
from src.utils.checkpoint import CheckpointManager
from src.utils.seed import set_seed


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(agent: str, overrides: dict) -> types.SimpleNamespace:
    """Load base.yaml + agent-specific YAML + CLI overrides → SimpleNamespace."""
    cfg_dir = os.path.join(os.path.dirname(__file__), 'configs')

    with open(os.path.join(cfg_dir, 'base.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    agent_cfg_path = os.path.join(cfg_dir, f'{agent}.yaml')
    if os.path.exists(agent_cfg_path):
        with open(agent_cfg_path, 'r') as f:
            agent_cfg = yaml.safe_load(f)
        _deep_update(cfg, agent_cfg)

    # CLI overrides
    _deep_update(cfg, overrides)

    return types.SimpleNamespace(**cfg)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='World Models Research Suite — Explicit vs Implicit comparison'
    )
    parser.add_argument('--agent', type=str, required=True,
                        choices=['explicit', 'implicit'],
                        help='Agent type: explicit (RSSM/DreamerV3) or implicit (TD-MPC2)')
    parser.add_argument('--env', type=str, default='dmc_cartpole_swingup',
                        help='Environment ID (e.g. dmc_cartpole_swingup, maniskill_PickCube-v1)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--total_steps', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in logdir')
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--no_mixed_precision', action='store_true')
    parser.add_argument('--action_repeat', type=int, default=None)
    parser.add_argument('--no_mpc', action='store_true',
                        help='Disable MPPI for implicit agent (use policy directly)')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def make_agent(agent_type: str, obs_dim: int, act_dim: int,
               act_continuous: bool, config):
    if agent_type == 'explicit':
        from src.agents.explicit_agent import ExplicitAgent
        return ExplicitAgent(obs_dim, act_dim, act_continuous, config)
    elif agent_type == 'implicit':
        from src.agents.implicit_agent import ImplicitAgent
        return ImplicitAgent(obs_dim, act_dim, act_continuous, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ---------------------------------------------------------------------------
# Latent snapshot collection
# ---------------------------------------------------------------------------

def collect_latent_snapshot(agent, buffer, n: int, snapshot_dir: str, step: int):
    """Sample n transitions from the buffer, encode, and save as NPZ.

    Snapshots enable continuous evolution analysis (20 points over 200k steps
    instead of 3 checkpoint-based data points).

    Saved arrays:
      obs    [n, obs_dim]   — raw observations from replay buffer
      z      [n, latent_dim] — agent's latent encoding of those observations
      reward [n]             — rewards from the transitions
      action [n, act_dim]    — actions taken
      step   scalar          — training step at collection time
    """
    os.makedirs(snapshot_dir, exist_ok=True)
    try:
        batch = buffer.sample_transitions(n, horizon=1)
        obs = batch['obs'][:, 0]        # [n, obs_dim]
        action = batch['action'][:, 0]  # [n, act_dim]
        reward = batch['reward'][:, 0]  # [n]
    except Exception:
        return  # buffer not ready yet

    z = agent.encode(obs)
    path = os.path.join(snapshot_dir, f'snapshot_{step:07d}.npz')
    np.savez_compressed(path, obs=obs.astype(np.float32),
                        z=z.astype(np.float32),
                        reward=reward.astype(np.float32),
                        action=action.astype(np.float32),
                        step=np.array(step))
    print(f'[snapshot] step={step}  saved {path}')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent, eval_env, n_episodes: int) -> dict:
    """Run deterministic evaluation episodes and return metrics."""
    returns = []
    lengths = []
    for ep in range(n_episodes):
        obs = eval_env.reset()
        agent.reset_state()
        ep_return = 0.0
        ep_len = 0
        done = False
        while not done:
            action = agent.act(obs, reset=(ep_len == 0), training=False)
            obs, reward, done, _ = eval_env.step(action)
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)
    return {
        'eval/mean_return': float(np.mean(returns)),
        'eval/std_return': float(np.std(returns)),
        'eval/mean_length': float(np.mean(lengths)),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Warn about headless rendering
    if 'MUJOCO_GL' not in os.environ:
        print('[WARNING] MUJOCO_GL not set. For headless (A100), run:')
        print('  export MUJOCO_GL=egl')
        print('  export VK_ICD_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json')

    # Build config override dict from CLI args
    overrides = {'env': args.env, 'seed': args.seed}
    if args.total_steps is not None:
        overrides['total_steps'] = args.total_steps
    if args.device is not None:
        overrides['device'] = args.device
    if args.logdir is not None:
        overrides['logdir'] = args.logdir
    if args.eval_every is not None:
        overrides['eval_every'] = args.eval_every
    if args.checkpoint_every is not None:
        overrides['checkpoint_every'] = args.checkpoint_every
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.no_mixed_precision:
        overrides['mixed_precision'] = False
    if args.action_repeat is not None:
        overrides['action_repeat'] = args.action_repeat
    if args.no_mpc:
        overrides['mpc'] = False

    config = load_config(args.agent, overrides)

    # Log directory
    logdir = os.path.join(
        config.logdir, args.agent, config.env, f'seed_{config.seed}'
    )
    os.makedirs(logdir, exist_ok=True)
    config.logdir_run = logdir

    # Seeding
    set_seed(config.seed)

    import torch
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = str(device)
    print(f'[train] Device: {device}')
    print(f'[train] Agent: {args.agent}  Env: {config.env}  Seed: {config.seed}')
    print(f'[train] Logdir: {logdir}')

    # Environments (eval env created once and reused — avoid per-eval construction cost)
    action_repeat = getattr(config, 'action_repeat', 2)
    env = make_env(config.env, seed=config.seed, action_repeat=action_repeat)
    eval_env = make_env(config.env, seed=config.seed + 999, action_repeat=action_repeat)
    print(f'[train] obs_dim={env.obs_dim}  act_dim={env.act_dim}  '
          f'continuous={env.act_continuous}')

    # Replay buffer
    buffer = EpisodeBuffer(
        capacity=config.buffer_capacity, seed=config.seed
    )

    # Agent
    agent = make_agent(
        args.agent, env.obs_dim, env.act_dim, env.act_continuous, config
    )
    agent = agent.to(device)
    print(f'[train] Agent parameters: '
          f'{sum(p.numel() for p in agent.parameters()):,}')

    # Logger + checkpoint manager
    logger = Logger(logdir)
    ckpt_manager = CheckpointManager(
        logdir, save_every=config.checkpoint_every, keep=3
    )
    snapshot_dir = os.path.join(logdir, 'latent_snapshots')
    snapshot_every = getattr(config, 'snapshot_every', 10000)

    # Resume from checkpoint if requested
    start_step = 0
    if args.resume:
        step_resumed = ckpt_manager.load_latest(agent, buffer)
        if step_resumed is not None:
            start_step = step_resumed
            print(f'[train] Resumed from step {start_step}')
        else:
            print('[train] No checkpoint found; starting from scratch.')

    # ------------------------------------------------------------------
    # Prefill buffer with random experience
    # ------------------------------------------------------------------
    prefill_needed = max(0, config.prefill_steps - buffer.num_transitions)
    if prefill_needed > 0:
        print(f'[train] Prefilling buffer with {prefill_needed} random steps...')
        obs = env.reset()
        buffer.start_episode()
        prefill_count = 0
        while prefill_count < prefill_needed:
            action = env.random_action()
            next_obs, reward, done, _ = env.step(action)
            buffer.add_transition(obs, action, reward, next_obs, done)
            prefill_count += 1
            obs = next_obs
            if done:
                buffer.end_episode()
                obs = env.reset()
                buffer.start_episode()
    # Close the last in-progress episode only if it has transitions
    if buffer._current_ep is not None and len(buffer._current_ep.get('obs', [])) > 0:
        buffer.end_episode()
        print(f'[train] Buffer has {buffer.num_transitions} transitions after prefill.')

    # ------------------------------------------------------------------
    # Pre-training world model for explicit agent
    # ------------------------------------------------------------------
    if args.agent == 'explicit':
        pretrain = getattr(config, 'pretrain', 100)
        if pretrain > 0 and buffer.num_transitions >= config.batch_size * config.batch_length:
            print(f'[train] Pre-training world model for {pretrain} steps...')
            for _ in range(pretrain):
                batch = buffer.sample_sequences(config.batch_size, config.batch_length)
                agent.update(batch)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    step = start_step
    ep_return = 0.0
    ep_len = 0
    obs = env.reset()
    agent.reset_state()
    buffer.start_episode()
    is_first = True

    # For explicit: track steps since last world model update
    steps_since_update = 0

    # For implicit: seed_steps
    seed_steps = getattr(config, 'seed_steps', 1000)

    print(f'[train] Starting training from step {step}/{config.total_steps}')
    t_start = time.time()
    metrics_accum = {}

    while step < config.total_steps:
        # ------ Environment interaction ------
        if args.agent == 'implicit' and step < seed_steps:
            action = env.random_action()
        else:
            action = agent.act(obs, reset=is_first, training=True)

        next_obs, reward, done, _ = env.step(action)
        buffer.add_transition(obs, action, reward, next_obs, done)
        ep_return += reward
        ep_len += 1
        obs = next_obs
        is_first = False
        step += 1
        steps_since_update += 1

        if done:
            buffer.end_episode()
            logger.log(step, {'train/episode_return': ep_return,
                               'train/episode_length': ep_len})
            ep_return = 0.0
            ep_len = 0
            obs = env.reset()
            agent.reset_state()
            buffer.start_episode()
            is_first = True

        # ------ Agent update ------
        if args.agent == 'explicit':
            # Update every train_ratio steps
            train_ratio = getattr(config, 'train_ratio', 512)
            if (steps_since_update >= train_ratio and
                    buffer.num_transitions >= config.batch_size * config.batch_length):
                batch = buffer.sample_sequences(config.batch_size, config.batch_length)
                metrics = agent.update(batch)
                steps_since_update = 0
                for k, v in metrics.items():
                    if isinstance(v, float) and not np.isfinite(v):
                        continue  # skip inf/nan — don't poison EMA accumulator
                    metrics_accum[k] = metrics_accum.get(k, 0) * 0.9 + v * 0.1

        elif args.agent == 'implicit':
            # Update every update_every steps after seed_steps
            horizon = getattr(config, 'horizon', 5)
            update_every = getattr(config, 'update_every', 2)
            if (step >= seed_steps and step % update_every == 0 and
                    buffer.num_transitions >= config.batch_size * (horizon + 1)):
                batch = buffer.sample_transitions(config.batch_size, horizon)
                metrics = agent.update(batch)
                for k, v in metrics.items():
                    if isinstance(v, float) and not np.isfinite(v):
                        continue  # skip inf/nan — don't poison EMA accumulator
                    metrics_accum[k] = metrics_accum.get(k, 0) * 0.9 + v * 0.1

        # ------ Periodic logging ------
        if step % config.log_every == 0 and metrics_accum:
            fps = step / max(time.time() - t_start, 1.0)
            metrics_accum['train/fps'] = fps
            metrics_accum['train/buffer_size'] = buffer.num_transitions
            logger.log(step, metrics_accum)
            logger.print_metrics(step, metrics_accum, prefix=args.agent)

        # ------ Periodic evaluation ------
        if step % config.eval_every == 0:
            eval_metrics = evaluate(agent, eval_env, n_episodes=config.eval_episodes)
            logger.log(step, eval_metrics)
            print(f'[eval] step={step} '
                  f'return={eval_metrics["eval/mean_return"]:.2f} '
                  f'±{eval_metrics["eval/std_return"]:.2f}')

        # ------ Checkpointing ------
        if ckpt_manager.should_save(step):
            path = ckpt_manager.save(step, agent, buffer, metrics_accum)
            print(f'[ckpt] Saved checkpoint at step {step}: {path}')

        # ------ Latent snapshot ------
        if step % snapshot_every == 0 and step > 0:
            collect_latent_snapshot(agent, buffer, n=512,
                                    snapshot_dir=snapshot_dir, step=step)

    # Final evaluation + checkpoint
    eval_metrics = evaluate(agent, eval_env, n_episodes=config.eval_episodes)
    logger.log(step, eval_metrics)
    print(f'[eval] FINAL step={step} '
          f'return={eval_metrics["eval/mean_return"]:.2f} '
          f'±{eval_metrics["eval/std_return"]:.2f}')

    ckpt_manager.save(step, agent, buffer, {**metrics_accum, **eval_metrics})
    # Close any in-progress episode (only if one is open)
    if buffer._current_ep is not None and len(buffer._current_ep.get('obs', [])) > 0:
        buffer.end_episode()
    env.close()
    eval_env.close()
    logger.close()
    print('[train] Done.')


if __name__ == '__main__':
    main()
