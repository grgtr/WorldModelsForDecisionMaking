"""
Latent Space Analysis Script.

Five analyses:
  1. Linear Probing (R²)     — how well latents decode obs coordinates
  2. CCA Alignment           — shared structure between the two latent spaces
  3. Effective Dimensionality — how many PCA dims are actually used
  4. Reward Predictability   — does z_t predict r_{t+1} beyond obs alone?
  5. Representation Evolution — how all metrics change across training checkpoints

Usage:
    python src/analyze_latents.py \
        --explicit_ckpt logs/explicit/maniskill_LiftCube-v0/seed_1/checkpoints/checkpoint_0200000.pt \
        --implicit_ckpt logs/implicit/maniskill_LiftCube-v0/seed_1/checkpoints/checkpoint_0200000.pt \
        --env maniskill_LiftCube-v0 --output analysis/ --seed 1
"""

import argparse
import glob
import os
import re
import sys
import types
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print('[WARNING] scikit-learn not found; probing and CCA will be skipped.')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print('[WARNING] matplotlib/seaborn not found; plots will be skipped.')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--explicit_ckpt', type=str, required=True,
                   help='Path to final explicit checkpoint (dir is auto-scanned for evolution)')
    p.add_argument('--implicit_ckpt', type=str, required=True,
                   help='Path to final implicit checkpoint (dir is auto-scanned for evolution)')
    p.add_argument('--env',      type=str, required=True)
    p.add_argument('--n_samples', type=int, default=5000,
                   help='Samples for main analyses (per agent).')
    p.add_argument('--n_samples_evo', type=int, default=1000,
                   help='Samples per checkpoint for temporal evolution (smaller = faster).')
    p.add_argument('--output',  type=str, default='analysis/')
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--device',  type=str, default='cuda')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_from_path(path: str) -> int:
    m = re.search(r'checkpoint_(\d+)\.pt$', path)
    return int(m.group(1)) if m else 0


def discover_checkpoints(final_ckpt: str) -> list:
    """Return all checkpoint paths in the same directory, sorted by step."""
    ckpt_dir = os.path.dirname(final_ckpt)
    files = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_*.pt')),
                   key=_step_from_path)
    return files


def load_agent(agent_type: str, ckpt_path: str, env, config, device):
    from src.train import make_agent
    agent = make_agent(agent_type, env.obs_dim, env.act_dim,
                       env.act_continuous, config)
    agent = agent.to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    agent.load_state_dict(payload['agent'])
    agent.eval()
    step = payload.get('step', _step_from_path(ckpt_path))
    print(f'  Loaded {agent_type} from step {step}', flush=True)
    return agent, step


def collect_latents(agent, env, n_samples: int, device) -> tuple:
    """Collect (obs, latent, next_reward) tuples.

    Returns:
        obs_arr:    [N, obs_dim]
        latent_arr: [N, latent_dim]
        reward_arr: [N]   — reward received AFTER stepping from that obs
    """
    obs_list, latent_list, reward_list = [], [], []
    obs = env.reset()
    agent.reset_state()
    ep_step = 0

    while len(obs_list) < n_samples:
        obs_arr = np.array(obs, dtype=np.float32)
        latent = agent.encode(obs_arr[np.newaxis])[0]
        obs_list.append(obs_arr)
        latent_list.append(latent)

        action = agent.act(obs_arr, reset=(ep_step == 0), training=False)
        obs, reward, done, _ = env.step(action)
        reward_list.append(float(reward))
        ep_step += 1

        if done:
            obs = env.reset()
            agent.reset_state()
            ep_step = 0

    N = n_samples
    return (np.stack(obs_list[:N]),
            np.stack(latent_list[:N]),
            np.array(reward_list[:N], dtype=np.float32))


# ---------------------------------------------------------------------------
# Analysis 1: Linear Probing
# ---------------------------------------------------------------------------

def linear_probing(obs: np.ndarray, latents: np.ndarray, agent_name: str) -> dict:
    """Ridge regression: z → obs coordinate, 5-fold CV R²."""
    if not SKLEARN_AVAILABLE:
        return {}

    Z = StandardScaler().fit_transform(latents)
    ridge = Ridge(alpha=1.0)
    r2_scores = []
    results = {}

    for coord in range(obs.shape[1]):
        y = obs[:, coord]
        scores = cross_val_score(ridge, Z, y, cv=5, scoring='r2')
        r2 = float(np.mean(scores))
        results[f'coord_{coord}'] = r2
        r2_scores.append(r2)

    arr = np.array(r2_scores)
    results['mean_r2']         = float(np.mean(arr))
    results['median_r2']       = float(np.median(arr))
    results['mean_r2_clipped'] = float(np.mean(np.clip(arr, 0, 1)))
    print(f'  [{agent_name}] Linear probing  '
          f'median R²={results["median_r2"]:.4f}  '
          f'mean R²(clipped)={results["mean_r2_clipped"]:.4f}', flush=True)
    return results


# ---------------------------------------------------------------------------
# Analysis 2: CCA Alignment
# ---------------------------------------------------------------------------

def cca_alignment(latents_e: np.ndarray, latents_i: np.ndarray,
                  n_components: int = 10) -> dict:
    if not SKLEARN_AVAILABLE:
        return {}

    n_components = min(n_components, latents_e.shape[1],
                       latents_i.shape[1], latents_e.shape[0] // 2)

    Ze = StandardScaler().fit_transform(latents_e)
    Zi = StandardScaler().fit_transform(latents_i)

    pca_dim = min(50, Ze.shape[1], Zi.shape[1])
    Ze_r = PCA(n_components=pca_dim).fit_transform(Ze)
    Zi_r = PCA(n_components=pca_dim).fit_transform(Zi)

    cca = CCA(n_components=n_components, max_iter=500)
    cca.fit(Ze_r, Zi_r)
    Xe, Xf = cca.transform(Ze_r, Zi_r)

    corrs = [float(np.corrcoef(Xe[:, k], Xf[:, k])[0, 1]) for k in range(n_components)]
    mean_corr = float(np.mean(corrs))
    print(f'  CCA mean correlation: {mean_corr:.4f}  top-3: {[round(c,4) for c in corrs[:3]]}',
          flush=True)
    return {'canonical_correlations': corrs, 'mean_correlation': mean_corr}


# ---------------------------------------------------------------------------
# Analysis 3: Effective Dimensionality
# ---------------------------------------------------------------------------

def effective_dimensionality(latents: np.ndarray, agent_name: str) -> dict:
    """How many PCA dimensions are actually used?

    Reports:
      - dims for 90 / 95 / 99 % explained variance
      - participation ratio = (sum eigenvalues)² / sum(eigenvalues²)
        → 1 = one dominant dim, D = all dims equally used
    """
    if not SKLEARN_AVAILABLE:
        return {}

    Z = StandardScaler().fit_transform(latents)
    pca = PCA().fit(Z)
    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)

    def dims_for(threshold):
        hits = np.where(cumvar >= threshold)[0]
        return int(hits[0] + 1) if len(hits) else len(evr)

    lam = pca.explained_variance_
    participation_ratio = float((lam.sum() ** 2) / (lam ** 2).sum())

    result = {
        'dims_90pct': dims_for(0.90),
        'dims_95pct': dims_for(0.95),
        'dims_99pct': dims_for(0.99),
        'total_dims': latents.shape[1],
        'participation_ratio': round(participation_ratio, 2),
        'top5_variance_pct': [round(float(v) * 100, 2) for v in evr[:5]],
    }
    print(f'  [{agent_name}] Effective dims  '
          f'90%={result["dims_90pct"]}  95%={result["dims_95pct"]}  '
          f'99%={result["dims_99pct"]}  '
          f'participation ratio={result["participation_ratio"]:.1f}/{latents.shape[1]}',
          flush=True)
    return result


# ---------------------------------------------------------------------------
# Analysis 4: Reward Predictability
# ---------------------------------------------------------------------------

def reward_probe(obs: np.ndarray, latents: np.ndarray,
                 rewards: np.ndarray, agent_name: str) -> dict:
    """Can z_t predict r_{t+1} better than obs_t alone?

    Trains Ridge regression on three feature sets:
      - obs only (baseline)
      - latent only
      - latent + obs (combined)
    Uses 5-fold CV R².  A higher latent R² over obs R² means the latent
    compresses task-relevant signal not just present in the raw state.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    y = rewards
    ridge = Ridge(alpha=1.0)

    def cv_r2(X):
        Xs = StandardScaler().fit_transform(X)
        return float(np.mean(cross_val_score(ridge, Xs, y, cv=5, scoring='r2')))

    r2_obs  = cv_r2(obs)
    r2_lat  = cv_r2(latents)
    r2_both = cv_r2(np.concatenate([obs, latents], axis=1))

    gain = r2_lat - r2_obs  # positive = latent adds info beyond obs
    result = {
        'r2_obs_baseline': round(r2_obs, 4),
        'r2_latent':       round(r2_lat, 4),
        'r2_latent+obs':   round(r2_both, 4),
        'gain_over_obs':   round(gain, 4),
    }
    print(f'  [{agent_name}] Reward probe  '
          f'obs={r2_obs:.4f}  latent={r2_lat:.4f}  '
          f'latent+obs={r2_both:.4f}  gain={gain:+.4f}', flush=True)
    return result


# ---------------------------------------------------------------------------
# Analysis 5: Representation Evolution across checkpoints
# ---------------------------------------------------------------------------

def representation_evolution(ckpts_e: list, ckpts_i: list,
                              env_e, env_i,
                              exp_config, imp_config,
                              device, n_samples: int,
                              output_dir: str) -> dict:
    """Load each checkpoint, collect latents, compute median R² and eff-dim.

    Returns dict with lists indexed by checkpoint step.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    steps_e, med_r2_e, eff_dim_e = [], [], []
    steps_i, med_r2_i, eff_dim_i = [], [], []

    for ckpt in ckpts_e:
        try:
            agent, step = load_agent('explicit', ckpt, env_e, exp_config, device)
            obs, lat, _ = collect_latents(agent, env_e, n_samples, device)
            r2 = linear_probing(obs, lat, f'explicit@{step}').get('median_r2', float('nan'))
            ed = effective_dimensionality(lat, f'explicit@{step}').get('dims_95pct', float('nan'))
            steps_e.append(step); med_r2_e.append(r2); eff_dim_e.append(ed)
            del agent
        except Exception as ex:
            print(f'  [WARN] explicit ckpt {ckpt} failed: {ex}', flush=True)

    for ckpt in ckpts_i:
        try:
            agent, step = load_agent('implicit', ckpt, env_i, imp_config, device)
            obs, lat, _ = collect_latents(agent, env_i, n_samples, device)
            r2 = linear_probing(obs, lat, f'implicit@{step}').get('median_r2', float('nan'))
            ed = effective_dimensionality(lat, f'implicit@{step}').get('dims_95pct', float('nan'))
            steps_i.append(step); med_r2_i.append(r2); eff_dim_i.append(ed)
            del agent
        except Exception as ex:
            print(f'  [WARN] implicit ckpt {ckpt} failed: {ex}', flush=True)

    result = {
        'explicit_steps': steps_e, 'explicit_median_r2': med_r2_e,
        'explicit_eff_dim_95': eff_dim_e,
        'implicit_steps': steps_i, 'implicit_median_r2': med_r2_i,
        'implicit_eff_dim_95': eff_dim_i,
    }

    if MPL_AVAILABLE and (steps_e or steps_i):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        if steps_e:
            ax.plot(steps_e, med_r2_e, 'o-', label='Explicit (RSSM)', color='steelblue')
        if steps_i:
            ax.plot(steps_i, med_r2_i, 's--', label='Implicit (TD-MPC2)', color='darkorange')
        ax.set_xlabel('Training Step'); ax.set_ylabel('Median R²')
        ax.set_title('Linear Decodability over Training')
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        if steps_e:
            ax.plot(steps_e, eff_dim_e, 'o-', label='Explicit (RSSM)', color='steelblue')
        if steps_i:
            ax.plot(steps_i, eff_dim_i, 's--', label='Implicit (TD-MPC2)', color='darkorange')
        ax.set_xlabel('Training Step'); ax.set_ylabel('Dims for 95% variance')
        ax.set_title('Effective Dimensionality over Training')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('Representation Evolution', fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, 'representation_evolution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved evolution plot: {path}', flush=True)

    return result


# ---------------------------------------------------------------------------
# PCA Visualization
# ---------------------------------------------------------------------------

def pca_visualization(latents: np.ndarray, rewards: np.ndarray,
                      agent_name: str, output_dir: str):
    if not (MPL_AVAILABLE and SKLEARN_AVAILABLE):
        return
    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(latents)
    evr = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=rewards, cmap='viridis', alpha=0.4, s=5)
    plt.colorbar(sc, ax=ax, label='Reward')
    ax.set_xlabel(f'PC1 ({evr[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({evr[1]*100:.1f}%)')
    ax.set_title(f'{agent_name} Latent Space (PCA)')
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'pca_{agent_name.lower()}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved PCA plot: {path}', flush=True)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def save_report(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'latent_analysis_report.txt')
    with open(path, 'w') as f:
        f.write('Latent Space Analysis Report\n')
        f.write('=' * 50 + '\n\n')
        for section, data in results.items():
            f.write(f'--- {section} ---\n')
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, list):
                        short = v[:6]
                        f.write(f'  {k}: {[round(x, 4) if isinstance(x, float) else x for x in short]}'
                                f'{"..." if len(v) > 6 else ""}\n')
                    else:
                        f.write(f'  {k}: {v}\n')
            else:
                f.write(f'  {data}\n')
            f.write('\n')
    print(f'\nReport saved: {path}', flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Starting analysis', flush=True)
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    from src.train import load_config
    exp_config = load_config('explicit', {'env': args.env, 'seed': args.seed,
                                          'device': str(device), 'mixed_precision': False})
    imp_config = load_config('implicit', {'env': args.env, 'seed': args.seed,
                                          'device': str(device), 'mixed_precision': False})

    from src.envs.factory import make_env
    action_repeat = getattr(exp_config, 'action_repeat', 2)
    env_e = make_env(args.env, seed=args.seed, action_repeat=action_repeat)
    env_i = make_env(args.env, seed=args.seed, action_repeat=action_repeat)

    # ---- Auto-discover all checkpoints in both directories ----
    ckpts_e = discover_checkpoints(args.explicit_ckpt)
    ckpts_i = discover_checkpoints(args.implicit_ckpt)
    print(f'Found {len(ckpts_e)} explicit checkpoints: '
          f'{[_step_from_path(c) for c in ckpts_e]}', flush=True)
    print(f'Found {len(ckpts_i)} implicit checkpoints: '
          f'{[_step_from_path(c) for c in ckpts_i]}', flush=True)

    # ---- Load final agents for main analyses ----
    print('\nLoading final agents...', flush=True)
    agent_e, _ = load_agent('explicit', args.explicit_ckpt, env_e, exp_config, device)
    agent_i, _ = load_agent('implicit', args.implicit_ckpt, env_i, imp_config, device)

    print(f'\nCollecting {args.n_samples} samples from each agent...', flush=True)
    obs_e, lat_e, rew_e = collect_latents(agent_e, env_e, args.n_samples, device)
    obs_i, lat_i, rew_i = collect_latents(agent_i, env_i, args.n_samples, device)
    print(f'Explicit latent shape: {lat_e.shape}', flush=True)
    print(f'Implicit latent shape: {lat_i.shape}', flush=True)

    results = {}

    # [1] Linear Probing
    print('\n[1] Linear Probing (R² scores)...', flush=True)
    results['linear_probing_explicit'] = linear_probing(obs_e, lat_e, 'Explicit')
    results['linear_probing_implicit'] = linear_probing(obs_i, lat_i, 'Implicit')
    r2_e = results['linear_probing_explicit'].get('median_r2', float('nan'))
    r2_i = results['linear_probing_implicit'].get('median_r2', float('nan'))
    winner = 'Explicit' if r2_e > r2_i else 'Implicit'
    print(f'  >> {winner} more linearly decodable  '
          f'(Explicit={r2_e:.4f}  Implicit={r2_i:.4f} median R²)', flush=True)

    # [2] CCA Alignment
    print('\n[2] CCA Alignment...', flush=True)
    N = min(len(lat_e), len(lat_i))
    results['cca_alignment'] = cca_alignment(lat_e[:N], lat_i[:N])

    # [3] Effective Dimensionality
    print('\n[3] Effective Dimensionality...', flush=True)
    results['eff_dim_explicit'] = effective_dimensionality(lat_e, 'Explicit')
    results['eff_dim_implicit'] = effective_dimensionality(lat_i, 'Implicit')

    # [4] Reward Predictability
    print('\n[4] Reward Predictability (z_t → r_{t+1})...', flush=True)
    results['reward_probe_explicit'] = reward_probe(obs_e, lat_e, rew_e, 'Explicit')
    results['reward_probe_implicit'] = reward_probe(obs_i, lat_i, rew_i, 'Implicit')

    # [5] PCA Visualization (final checkpoint)
    print('\n[5] PCA Visualization...', flush=True)
    pca_visualization(lat_e, rew_e, 'Explicit (RSSM)', args.output)
    pca_visualization(lat_i, rew_i, 'Implicit (TD-MPC2)', args.output)

    if MPL_AVAILABLE and SKLEARN_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, lat, rew, name in [
            (axes[0], lat_e, rew_e, 'Explicit (RSSM)'),
            (axes[1], lat_i, rew_i, 'Implicit (TD-MPC2)'),
        ]:
            Z2 = PCA(n_components=2).fit_transform(lat)
            sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=rew, cmap='plasma', alpha=0.3, s=4)
            plt.colorbar(sc, ax=ax, label='Reward')
            ax.set_title(name); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        plt.suptitle(f'Latent Topology — {args.env}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output, 'pca_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {os.path.join(args.output, "pca_comparison.png")}', flush=True)

    # [6] Representation Evolution (uses all discovered checkpoints)
    print(f'\n[6] Representation Evolution '
          f'({len(ckpts_e)} explicit, {len(ckpts_i)} implicit checkpoints)...', flush=True)
    # Free final agents first to save GPU memory during multi-ckpt loop
    del agent_e, agent_i
    results['representation_evolution'] = representation_evolution(
        ckpts_e, ckpts_i, env_e, env_i,
        exp_config, imp_config, device,
        args.n_samples_evo, args.output,
    )

    save_report(results, args.output)
    env_e.close()
    env_i.close()


if __name__ == '__main__':
    main()
