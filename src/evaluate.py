"""
Standalone evaluation script — load a checkpoint and evaluate an agent.

Usage:
    python src/evaluate.py \
        --checkpoint logs/explicit/dmc_cartpole_swingup/seed_0/checkpoints/checkpoint_0500000.pt \
        --episodes 20
"""

import argparse
import os
import sys
import types
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_checkpoint_config(ckpt_path: str) -> dict:
    """Try to load config from run directory alongside checkpoint."""
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))  # up from checkpoints/
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return yaml.safe_load(f)
    return {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--agent', type=str, default=None,
                   help='explicit or implicit (auto-detected from checkpoint path if omitted)')
    p.add_argument('--env', type=str, default=None,
                   help='Environment ID (auto-detected from checkpoint path if omitted)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-detect agent and env from checkpoint path
    path_parts = args.checkpoint.replace('\\', '/').split('/')
    agent_type = args.agent
    env_id = args.env
    for i, part in enumerate(path_parts):
        if part in ('explicit', 'implicit') and agent_type is None:
            agent_type = part
        if part.startswith('dmc_') or part.startswith('maniskill_') or part.startswith('gym_'):
            env_id = part

    if agent_type is None:
        raise ValueError("Could not detect agent type; use --agent explicit|implicit")
    if env_id is None:
        raise ValueError("Could not detect env; use --env <env_id>")

    print(f'Agent: {agent_type}  Env: {env_id}')

    # Load config and build agent
    from src.train import load_config
    config = load_config(agent_type, {'env': env_id, 'seed': args.seed,
                                       'device': args.device, 'mixed_precision': False})
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = str(device)

    from src.envs.factory import make_env
    env = make_env(env_id, seed=args.seed, action_repeat=getattr(config, 'action_repeat', 2))

    from src.train import make_agent
    from src.buffer.episode_buffer import EpisodeBuffer
    agent = make_agent(agent_type, env.obs_dim, env.act_dim, env.act_continuous, config)
    agent = agent.to(device)

    # Load checkpoint
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    agent.load_state_dict(payload['agent'])
    print(f'Loaded checkpoint from step {payload["step"]}')

    # Evaluate
    returns, lengths = [], []
    for ep in range(args.episodes):
        obs = env.reset()
        agent.reset_state()
        ep_return, ep_len, done = 0.0, 0, False
        while not done:
            action = agent.act(obs, reset=(ep_len == 0), training=False)
            obs, reward, done, _ = env.step(action)
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)
        print(f'  Episode {ep+1:3d}: return={ep_return:.2f}  length={ep_len}')

    print(f'\nResults over {args.episodes} episodes:')
    print(f'  Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}')
    print(f'  Mean length: {np.mean(lengths):.1f}')
    env.close()


if __name__ == '__main__':
    main()
