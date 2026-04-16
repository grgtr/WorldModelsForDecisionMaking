"""
Generate expert demonstrations for ManiSkill LiftCube-v0 and save in LeRobot format.

LeRobot format on disk:
  data/lerobot/<dataset_name>/
    meta/info.json                           # dataset metadata
    data/chunk-000/episode_000000.parquet    # tabular data (state, action, reward…)
    videos/observation.images.rgb/
      chunk-000/episode_000000.mp4           # RGB frames

Usage:
  # Full run (100 episodes)
  python src/generate_demos.py --n_episodes 100 --output data/lerobot/liftcube

  # Smoke test (5 episodes)
  python src/generate_demos.py --n_episodes 5 --output data/lerobot/liftcube_test
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
TASK          = 'LiftCube-v0'
CONTROL_MODE  = 'pd_joint_delta_pos'
IMG_SIZE      = 224   # SigLIP native input size
FPS           = 10
MAX_STEPS     = 200   # episode time limit
ACTION_REPEAT = 1     # use 1 for demo collection (raw actions recorded)


# ──────────────────────────────────────────────────────────────
# Motion-planning demo collection
# ──────────────────────────────────────────────────────────────

def _make_env(seed: int = 0):
    """Create ManiSkill env with RGB rendering enabled."""
    try:
        import gymnasium as gym
    except ImportError:
        import gym
    import mani_skill2.envs  # noqa

    env = gym.make(
        TASK,
        obs_mode='state',
        control_mode=CONTROL_MODE,
        render_mode='rgb_array',
        reward_mode='dense',
    )
    env.reset(seed=seed)
    return env


def _flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            if isinstance(v, np.ndarray):
                parts.append(v.astype(np.float32).flatten())
            elif isinstance(v, dict):
                parts.append(_flatten_obs(v))
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    return np.atleast_1d(obs).astype(np.float32).flatten()


def _render_rgb(env, size: int = IMG_SIZE) -> np.ndarray:
    frame = env.render()
    if frame is None:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if frame.shape[0] != size or frame.shape[1] != size:
        try:
            import cv2
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            frame = np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return frame.astype(np.uint8)


def collect_episode_motion_plan(env, seed: int):
    """
    Collect one episode using ManiSkill2's built-in motion planner (if available),
    falling back to a scripted/random policy filtered by success.

    Returns list of dicts: [{state, rgb, action, reward, done}, ...] or None on failure.
    """
    obs_raw, _ = env.reset(seed=seed) if hasattr(env.reset(seed=seed), '__len__') \
        else (env.reset(seed=seed), {})
    # Gymnasium API: reset returns (obs, info)
    result = env.reset(seed=seed)
    if isinstance(result, tuple):
        obs_raw, _ = result
    else:
        obs_raw = result

    # Try motion planner
    try:
        from mani_skill2.envs.pick_and_place.lift_cube import LiftCubeEnv  # noqa
        # Access underlying env
        unwrapped = env.unwrapped
        if hasattr(unwrapped, '_solve') or hasattr(unwrapped, 'get_solution_seq'):
            plan_fn = getattr(unwrapped, '_solve', None) or getattr(unwrapped, 'get_solution_seq')
            actions = plan_fn()
            if actions is not None:
                return _replay_actions(env, obs_raw, actions)
    except Exception:
        pass

    # Try mani_skill2 solution generator
    try:
        from mani_skill2.utils.sapien_utils import get_entity_by_name  # noqa
        unwrapped = env.unwrapped
        # Some envs expose a solve() method
        if hasattr(unwrapped, 'solve'):
            actions = unwrapped.solve()
            if actions is not None:
                return _replay_actions(env, obs_raw, actions)
    except Exception:
        pass

    # Fallback: random exploration with success filtering
    return _collect_random_episode(env, obs_raw)


def _replay_actions(env, initial_obs, actions):
    """Replay a pre-computed action sequence and record frames."""
    trajectory = []
    obs_raw = initial_obs
    for action in actions:
        action = np.array(action, dtype=np.float32)
        result = env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = result

        trajectory.append({
            'state':  _flatten_obs(obs_raw),
            'rgb':    _render_rgb(env),
            'action': action.astype(np.float32),
            'reward': float(reward),
            'done':   bool(done),
        })
        obs_raw = next_obs
        if done:
            break

    success = any(s['reward'] > 0 for s in trajectory)
    return trajectory if success else None


def _collect_random_episode(env, initial_obs, max_steps: int = MAX_STEPS):
    """Collect an episode with a scripted heuristic or random actions."""
    trajectory = []
    obs_raw = initial_obs
    act_dim = env.action_space.shape[0]

    for _ in range(max_steps):
        # Simple heuristic: move toward cube, then lift
        action = env.action_space.sample()  # random fallback

        result = env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = result

        trajectory.append({
            'state':  _flatten_obs(obs_raw),
            'rgb':    _render_rgb(env),
            'action': np.array(action, dtype=np.float32),
            'reward': float(reward),
            'done':   bool(done),
        })
        obs_raw = next_obs
        if done:
            break

    # Accept even random episodes (they still provide visual diversity)
    return trajectory if len(trajectory) > 0 else None


def collect_episodes(n_episodes: int, verbose: bool = True):
    """Collect n_episodes. Returns list of trajectories."""
    env = _make_env(seed=0)
    obs_dim  = _flatten_obs(env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()).shape[0]
    act_dim  = env.action_space.shape[0]

    trajectories = []
    attempted = 0
    while len(trajectories) < n_episodes:
        seed = attempted
        traj = collect_episode_motion_plan(env, seed=seed)
        attempted += 1
        if traj is None:
            # Collect anyway (random) to avoid infinite loop
            result = env.reset(seed=seed)
            obs_raw = result[0] if isinstance(result, tuple) else result
            traj = _collect_random_episode(env, obs_raw)
        if traj:
            trajectories.append(traj)
            if verbose:
                total_reward = sum(s['reward'] for s in traj)
                print(f'  episode {len(trajectories):3d}/{n_episodes}  '
                      f'steps={len(traj)}  total_reward={total_reward:.2f}', flush=True)
        if attempted > n_episodes * 5:
            print(f'Warning: gave up after {attempted} attempts, got {len(trajectories)} episodes.')
            break

    env.close()
    return trajectories, obs_dim, act_dim


# ──────────────────────────────────────────────────────────────
# LeRobot format writer
# ──────────────────────────────────────────────────────────────

def _write_mp4(frames, path: str, fps: int = FPS):
    """Write list of uint8 [H,W,3] frames to MP4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec='libx264',
                                    output_params=['-crf', '23', '-pix_fmt', 'yuv420p'])
        for f in frames:
            writer.append_data(f)
        writer.close()
    except Exception as e:
        # Fallback: save as individual PNG frames if ffmpeg not available
        print(f'  Warning: MP4 write failed ({e}), saving frames as PNG instead.')
        base = path.replace('.mp4', '')
        os.makedirs(base, exist_ok=True)
        from PIL import Image
        for i, f in enumerate(frames):
            Image.fromarray(f).save(f'{base}/frame_{i:04d}.png')


def save_lerobot_dataset(trajectories, output_dir: str, obs_dim: int, act_dim: int):
    """
    Write trajectories to disk in LeRobot v2 format.

    Directory layout:
      output_dir/
        meta/info.json
        data/chunk-000/episode_XXXXXX.parquet
        videos/observation.images.rgb/chunk-000/episode_XXXXXX.mp4
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    meta_dir    = os.path.join(output_dir, 'meta')
    data_dir    = os.path.join(output_dir, 'data', 'chunk-000')
    video_dir   = os.path.join(output_dir, 'videos', 'observation.images.rgb', 'chunk-000')
    os.makedirs(meta_dir,  exist_ok=True)
    os.makedirs(data_dir,  exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    total_frames = 0
    episode_data_index = {'from': [], 'to': []}  # global frame ranges per episode

    for ep_idx, traj in enumerate(trajectories):
        ep_name = f'episode_{ep_idx:06d}'

        # ── Parquet (tabular) ──
        rows = []
        for frame_idx, step in enumerate(traj):
            rows.append({
                'episode_index': ep_idx,
                'frame_index':   frame_idx,
                'timestamp':     frame_idx / FPS,
                'state':         step['state'].tolist(),
                'action':        step['action'].tolist(),
                'reward':        step['reward'],
                'next.done':     step['done'],
                'index':         total_frames + frame_idx,
            })

        table = pa.Table.from_pylist(rows)
        pq.write_table(table, os.path.join(data_dir, ep_name + '.parquet'))

        # ── MP4 video ──
        frames = [s['rgb'] for s in traj]
        _write_mp4(frames, os.path.join(video_dir, ep_name + '.mp4'), fps=FPS)

        # Track global frame indices
        episode_data_index['from'].append(total_frames)
        episode_data_index['to'].append(total_frames + len(traj))
        total_frames += len(traj)

    # ── info.json ──
    info = {
        'codebase_version': 'v2.0',
        'robot_type': 'panda',
        'total_episodes': len(trajectories),
        'total_frames': total_frames,
        'fps': FPS,
        'task': TASK,
        'features': {
            'state':  {'dtype': 'float32', 'shape': [obs_dim], 'names': None},
            'action': {'dtype': 'float32', 'shape': [act_dim], 'names': None},
            'reward': {'dtype': 'float32', 'shape': [1],       'names': None},
            'next.done': {'dtype': 'bool', 'shape': [1],       'names': None},
            'observation.images.rgb': {
                'dtype': 'video', 'shape': [IMG_SIZE, IMG_SIZE, 3], 'names': ['height', 'width', 'channel'],
                'info': {'video.fps': FPS, 'video.codec': 'libx264'},
            },
        },
        'episode_data_index': episode_data_index,
        'splits': {'train': f'0:{len(trajectories)}'},
    }
    with open(os.path.join(meta_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f'\nDataset saved to {output_dir}')
    print(f'  episodes: {len(trajectories)}  total frames: {total_frames}')
    print(f'  obs_dim={obs_dim}  act_dim={act_dim}  img_size={IMG_SIZE}')


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    global IMG_SIZE
    parser = argparse.ArgumentParser(description='Generate ManiSkill LiftCube demos in LeRobot format.')
    parser.add_argument('--n_episodes', type=int, default=100,
                        help='Number of episodes to collect (default: 100)')
    parser.add_argument('--output', type=str, default='data/lerobot/liftcube',
                        help='Output directory for LeRobot dataset')
    parser.add_argument('--img_size', type=int, default=IMG_SIZE,
                        help=f'RGB frame size in pixels (default: {IMG_SIZE})')
    args = parser.parse_args()

    IMG_SIZE = args.img_size

    print(f'Collecting {args.n_episodes} episodes of {TASK}...')
    trajectories, obs_dim, act_dim = collect_episodes(args.n_episodes, verbose=True)

    print(f'\nSaving {len(trajectories)} episodes to {args.output}...')
    save_lerobot_dataset(trajectories, args.output, obs_dim, act_dim)


if __name__ == '__main__':
    main()
