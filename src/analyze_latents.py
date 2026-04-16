"""
Latent Space Analysis Script.

Analyses:
  1. Linear Probing (R²)         — how well latents decode obs coordinates
  2. CCA Alignment               — shared structure between the two latent spaces
  3. Effective Dimensionality    — how many PCA dims are actually used
  4. Reward Predictability       — does z_t predict r_{t+1} beyond obs alone?
  5. Representation Evolution    — how all metrics change across training checkpoints
  6. Snapshot Geometry Timeline  — continuous eff-dim + dead-unit curves from snapshot NPZs
  7. Metrics Timeline            — KL utilization, horizon degradation, MPPI planner confidence
  8. Cross-Agent CCA Timeline    — CCA correlation between agents at each shared snapshot step

Usage:
    python src/analyze_latents.py \
        --explicit_ckpt logs/explicit/maniskill_LiftCube-v0/seed_1/checkpoints/checkpoint_0200000.pt \
        --implicit_ckpt logs/implicit/maniskill_LiftCube-v0/seed_1/checkpoints/checkpoint_0200000.pt \
        --env maniskill_LiftCube-v0 --output analysis/ --seed 1
"""

import argparse
import glob
import json
import os
import re
import sys
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
    p.add_argument('--skip_env_collection', action='store_true',
                   help='Skip env-based collection (analyses 1-5); use only snapshots+metrics')
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


def _run_dir_from_ckpt(ckpt_path: str) -> str:
    """Given a checkpoint path, return the run directory (parent of 'checkpoints/')."""
    ckpt_dir = os.path.dirname(ckpt_path)
    run_dir = os.path.dirname(ckpt_dir)
    return run_dir


def discover_snapshots(ckpt_path: str) -> str:
    """Return snapshot directory corresponding to a checkpoint file."""
    return os.path.join(_run_dir_from_ckpt(ckpt_path), 'latent_snapshots')


def discover_metrics(ckpt_path: str) -> str:
    """Return metrics.jsonl path corresponding to a checkpoint file."""
    return os.path.join(_run_dir_from_ckpt(ckpt_path), 'metrics.jsonl')


# ---------------------------------------------------------------------------
# Snapshot / metrics loaders
# ---------------------------------------------------------------------------

def load_snapshots(snapshot_dir: str) -> list:
    """Load all snapshot NPZ files from a directory, sorted by step.

    Returns a list of dicts with keys: step, obs, z, reward, action.
    """
    files = sorted(glob.glob(os.path.join(snapshot_dir, 'snapshot_*.npz')))
    snapshots = []
    for f in files:
        try:
            d = np.load(f)
            snapshots.append({
                'step': int(d['step']),
                'obs': d['obs'],
                'z': d['z'],
                'reward': d['reward'],
                'action': d['action'],
            })
        except Exception as ex:
            print(f'  [WARN] Could not load snapshot {f}: {ex}', flush=True)
    return snapshots


def load_metrics_jsonl(metrics_path: str) -> list:
    """Load metrics.jsonl, replacing JSON-invalid NaN/Inf with Python equivalents."""
    if not os.path.exists(metrics_path):
        return []
    records = []
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Replace JSON-invalid tokens before parsing
            line = line.replace(': Infinity,', ': null,').replace(': Infinity}', ': null}')
            line = line.replace(': NaN,', ': null,').replace(': NaN}', ': null}')
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ---------------------------------------------------------------------------
# Geometry helpers (applied to snapshot latents)
# ---------------------------------------------------------------------------

def _participation_ratio(z: np.ndarray) -> float:
    """Participation ratio = (sum λ)² / sum λ² — equals D if all dims equal."""
    if not SKLEARN_AVAILABLE:
        return float('nan')
    zs = StandardScaler().fit_transform(z)
    pca = PCA().fit(zs)
    lam = pca.explained_variance_
    return float((lam.sum() ** 2) / ((lam ** 2).sum() + 1e-10))


def _dims_for_variance(z: np.ndarray, threshold: float = 0.95) -> int:
    if not SKLEARN_AVAILABLE:
        return -1
    zs = StandardScaler().fit_transform(z)
    pca = PCA().fit(zs)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    hits = np.where(cumvar >= threshold)[0]
    return int(hits[0] + 1) if len(hits) else z.shape[1]


def _dead_unit_frac(z: np.ndarray, threshold: float = 0.01) -> float:
    """Fraction of latent dimensions with std < threshold across the batch."""
    per_dim_std = z.std(axis=0)
    return float((per_dim_std < threshold).mean())


# ---------------------------------------------------------------------------
# Analysis 6: Snapshot Geometry Timeline
# ---------------------------------------------------------------------------

def _snapshot_r2(obs: np.ndarray, z: np.ndarray) -> float:
    """Median R² of Ridge regression (z → obs) — same metric as linear_probing()
    but applied to a single snapshot's (obs, z) arrays without env interaction."""
    if not SKLEARN_AVAILABLE or len(z) < 10:
        return float('nan')
    Z = StandardScaler().fit_transform(z.astype(np.float64))
    ridge = Ridge(alpha=1.0)
    scores = []
    for coord in range(obs.shape[1]):
        y = obs[:, coord].astype(np.float64)
        try:
            cv = cross_val_score(ridge, Z, y, cv=min(5, len(Z)), scoring='r2')
            scores.append(float(np.mean(cv)))
        except Exception:
            pass
    return float(np.median(scores)) if scores else float('nan')


def snapshot_geometry_analysis(snapshots_e: list, snapshots_i: list,
                                output_dir: str) -> dict:
    """From pre-saved snapshots compute continuous geometry + decodability curves.

    Replaces the checkpoint-based representation_evolution analysis with 20 data
    points from tiny NPZ files — no env rollouts or agent re-loading required.

    Metrics per snapshot:
      - participation_ratio   (effective dimensionality)
      - dims_95pct            (dims needed for 95% variance)
      - dead_unit_frac        (fraction of dims with std < 0.01)
      - median_r2             (linear probing R²: z → obs coords)
    """
    def _process(snaps, label):
        steps, pr_vals, dims95, dead, r2_vals = [], [], [], [], []
        for s in snaps:
            z = s['z'].astype(np.float64)
            obs = s['obs'].astype(np.float64)
            steps.append(s['step'])
            pr_vals.append(_participation_ratio(z))
            dims95.append(_dims_for_variance(z, 0.95))
            dead.append(_dead_unit_frac(z))
            r2 = _snapshot_r2(obs, z)
            r2_vals.append(r2)
            print(f'  [{label}] step={s["step"]}  PR={pr_vals[-1]:.2f}  '
                  f'dims95={dims95[-1]}  dead={dead[-1]:.3f}  '
                  f'median_R²={r2:.4f}', flush=True)
        return steps, pr_vals, dims95, dead, r2_vals

    print('\n[Snapshot Geometry] Explicit agent:', flush=True)
    steps_e, pr_e, dims95_e, dead_e, r2_e = _process(snapshots_e, 'explicit')
    print('\n[Snapshot Geometry] Implicit agent:', flush=True)
    steps_i, pr_i, dims95_i, dead_i, r2_i = _process(snapshots_i, 'implicit')

    result = {
        'explicit_steps': steps_e, 'explicit_participation_ratio': pr_e,
        'explicit_dims_95pct': dims95_e, 'explicit_dead_unit_frac': dead_e,
        'explicit_median_r2': r2_e,
        'implicit_steps': steps_i, 'implicit_participation_ratio': pr_i,
        'implicit_dims_95pct': dims95_i, 'implicit_dead_unit_frac': dead_i,
        'implicit_median_r2': r2_i,
    }

    if MPL_AVAILABLE and (steps_e or steps_i):
        os.makedirs(output_dir, exist_ok=True)

        # -- Plot A: geometry (PR, dims95, dead units) --
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, ye, yi, title, ylabel in [
            (axes[0], pr_e, pr_i,
             'Participation Ratio over Training', 'Participation Ratio (eff. dims)'),
            (axes[1], dims95_e, dims95_i,
             'Dims for 95% Variance over Training', 'Dims for 95% Variance'),
            (axes[2], dead_e, dead_i,
             'Dead Unit Fraction over Training', 'Fraction of Dead Dims (std < 0.01)'),
        ]:
            if steps_e:
                ax.plot(steps_e, ye, 'o-', color='steelblue', label='Explicit (RSSM)')
            if steps_i:
                ax.plot(steps_i, yi, 's--', color='darkorange', label='Implicit (TD-MPC2)')
            ax.set_xlabel('Training Step'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        plt.suptitle('Latent Geometry Timeline (from Snapshots)', fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, 'snapshot_geometry_timeline.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}', flush=True)

        # -- Plot B: linear probing R² evolution (20 points vs 3 from checkpoints) --
        fig, ax = plt.subplots(figsize=(9, 5))
        if steps_e and any(not np.isnan(v) for v in r2_e):
            ax.plot(steps_e, r2_e, 'o-', color='steelblue', label='Explicit (RSSM)')
        if steps_i and any(not np.isnan(v) for v in r2_i):
            ax.plot(steps_i, r2_i, 's--', color='darkorange', label='Implicit (TD-MPC2)')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Training Step'); ax.set_ylabel('Median R²')
        ax.set_title('Linear Decodability over Training\n(from snapshot obs+z pairs — 20 points)')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, 'snapshot_r2_evolution.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}', flush=True)

    return result


# ---------------------------------------------------------------------------
# Analysis 7: Metrics Timeline (from metrics.jsonl)
# ---------------------------------------------------------------------------

def metrics_timeline_analysis(metrics_e: list, metrics_i: list,
                               output_dir: str) -> dict:
    """Parse logged metrics and produce timeline plots for key diagnostics.

    Explicit agent:
      - wm/kl_active_vars     (how many of 8 categorical variables are active)
      - wm/kl_min_var         (KL of the least-used variable — near 0 = collapse)
      - latent/z_active_frac  (fraction of dims with std > 0.01)

    Implicit agent:
      - latent/consistency_tH / latent/consistency_t0   (horizon degradation ratio)
      - latent/mppi_elite_value_std                     (planner diversity)
      - latent/mppi_plan_std                            (action distribution breadth)
      - latent/z_active_frac
    """
    def _extract(records, keys):
        out = {k: [] for k in keys + ['step']}
        for r in records:
            if any(r.get(k) is not None for k in keys):
                out['step'].append(r['step'])
                for k in keys:
                    out[k].append(r.get(k))
        return out

    e_keys = ['wm/kl_active_vars', 'wm/kl_min_var', 'latent/z_active_frac',
              'latent/encoder_grad_norm']
    i_keys = ['latent/consistency_t0', 'latent/consistency_tH',
              'latent/mppi_elite_value_std', 'latent/mppi_plan_std',
              'latent/z_active_frac', 'latent/encoder_grad_norm']

    e_data = _extract(metrics_e, e_keys)
    i_data = _extract(metrics_i, i_keys)

    # Compute derived: horizon degradation ratio = consistency_tH / consistency_t0
    degradation = []
    for t0, tH in zip(i_data['latent/consistency_t0'], i_data['latent/consistency_tH']):
        if t0 is not None and t0 > 1e-10 and tH is not None:
            degradation.append(tH / t0)
        else:
            degradation.append(None)

    result = {
        'explicit': {k: e_data[k] for k in e_keys + ['step']},
        'implicit': {k: i_data[k] for k in i_keys + ['step']},
        'implicit_horizon_degradation_ratio': degradation,
    }

    if not MPL_AVAILABLE:
        return result

    os.makedirs(output_dir, exist_ok=True)

    # -- Plot 1: KL utilization (explicit) --
    if e_data['step']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        steps_e = e_data['step']
        for ax, key, title, ylabel, color in [
            (axes[0], 'wm/kl_active_vars',
             'KL Active Variables', 'Variables with KL > 0.1 nats', 'steelblue'),
            (axes[1], 'wm/kl_min_var',
             'KL of Least-Active Variable', 'Min KL (nats) — 0 = collapse', 'steelblue'),
            (axes[2], 'latent/z_active_frac',
             'Active Latent Fraction', 'Frac of dims with std > 0.01', 'steelblue'),
        ]:
            vals = e_data[key]
            valid = [(s, v) for s, v in zip(steps_e, vals) if v is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, '-', color=color)
            ax.set_xlabel('Training Step'); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.grid(True, alpha=0.3)
        plt.suptitle('Explicit Agent — KL Utilization & Latent Health', fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, 'explicit_kl_utilization.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}', flush=True)

    # -- Plot 2: Implicit planner health + horizon degradation --
    if i_data['step']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        steps_i = i_data['step']

        # Horizon degradation
        valid_deg = [(s, v) for s, v in zip(steps_i, degradation) if v is not None]
        if valid_deg:
            xs, ys = zip(*valid_deg)
            axes[0].plot(xs, ys, '-', color='darkorange')
        axes[0].axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='no degradation')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('consistency_tH / consistency_t0')
        axes[0].set_title('Horizon Degradation Ratio (> 1 = multi-step worse)')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # MPPI diversity
        valid_std = [(s, v) for s, v in zip(steps_i, i_data['latent/mppi_elite_value_std'])
                     if v is not None]
        if valid_std:
            xs, ys = zip(*valid_std)
            axes[1].plot(xs, ys, '-', color='darkorange')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Elite value std')
        axes[1].set_title('MPPI Planner Diversity (collapse = 0)')
        axes[1].grid(True, alpha=0.3)

        # Active latent fraction
        valid_af = [(s, v) for s, v in zip(steps_i, i_data['latent/z_active_frac'])
                    if v is not None]
        if valid_af:
            xs, ys = zip(*valid_af)
            axes[2].plot(xs, ys, '-', color='darkorange')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Frac of dims with std > 0.01')
        axes[2].set_title('Active Latent Fraction')
        axes[2].grid(True, alpha=0.3)

        plt.suptitle('Implicit Agent — Planner Health & Horizon Degradation', fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, 'implicit_planner_health.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}', flush=True)

    return result


# ---------------------------------------------------------------------------
# Analysis 8: Cross-Agent CCA Timeline (from snapshots at shared steps)
# ---------------------------------------------------------------------------

def cross_agent_cca_timeline(snapshots_e: list, snapshots_i: list,
                              output_dir: str) -> dict:
    """Compute CCA correlation between explicit and implicit latents at each
    shared snapshot step. Reveals whether the two agents converge or diverge
    in their learned representations over training.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    # Index snapshots by step
    snap_e = {s['step']: s['z'] for s in snapshots_e}
    snap_i = {s['step']: s['z'] for s in snapshots_i}
    shared_steps = sorted(set(snap_e.keys()) & set(snap_i.keys()))

    if not shared_steps:
        print('  [CCA Timeline] No shared snapshot steps found.', flush=True)
        return {}

    print(f'  [CCA Timeline] Shared steps: {shared_steps}', flush=True)

    cca_steps, mean_corrs, top1_corrs = [], [], []
    for step in shared_steps:
        ze = snap_e[step].astype(np.float64)
        zi = snap_i[step].astype(np.float64)
        N = min(ze.shape[0], zi.shape[0])
        ze, zi = ze[:N], zi[:N]

        n_comp = min(10, ze.shape[1], zi.shape[1], N // 2)
        Ze = StandardScaler().fit_transform(ze)
        Zi = StandardScaler().fit_transform(zi)

        # Reduce to at most 50 PCA dims to speed up CCA
        pca_dim = min(50, Ze.shape[1], Zi.shape[1])
        Ze_r = PCA(n_components=pca_dim).fit_transform(Ze)
        Zi_r = PCA(n_components=pca_dim).fit_transform(Zi)

        try:
            cca = CCA(n_components=n_comp, max_iter=500)
            cca.fit(Ze_r, Zi_r)
            Xe, Xi = cca.transform(Ze_r, Zi_r)
            corrs = [float(np.corrcoef(Xe[:, k], Xi[:, k])[0, 1]) for k in range(n_comp)]
            mean_c = float(np.mean(corrs))
            top1_c = float(corrs[0]) if corrs else float('nan')
        except Exception as ex:
            print(f'  [WARN] CCA failed at step {step}: {ex}', flush=True)
            mean_c = float('nan')
            top1_c = float('nan')

        cca_steps.append(step)
        mean_corrs.append(mean_c)
        top1_corrs.append(top1_c)
        print(f'  step={step}  mean_corr={mean_c:.4f}  top1_corr={top1_c:.4f}', flush=True)

    result = {
        'shared_steps': cca_steps,
        'mean_canonical_corr': mean_corrs,
        'top1_canonical_corr': top1_corrs,
    }

    if MPL_AVAILABLE and cca_steps:
        os.makedirs(output_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cca_steps, mean_corrs, 'o-', color='mediumseagreen', label='Mean (top-10 CCs)')
        ax.plot(cca_steps, top1_corrs, 's--', color='darkgreen', label='Top-1 CC')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Canonical Correlation')
        ax.set_title('Cross-Agent CCA Timeline\n(Explicit ↔ Implicit representation alignment)')
        ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        path = os.path.join(output_dir, 'cross_agent_cca_timeline.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {path}', flush=True)

    return result


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

    results = {}

    # ---- Auto-discover artifacts from checkpoint paths ----
    ckpts_e = discover_checkpoints(args.explicit_ckpt)
    ckpts_i = discover_checkpoints(args.implicit_ckpt)
    snap_dir_e = discover_snapshots(args.explicit_ckpt)
    snap_dir_i = discover_snapshots(args.implicit_ckpt)
    metrics_path_e = discover_metrics(args.explicit_ckpt)
    metrics_path_i = discover_metrics(args.implicit_ckpt)

    print(f'Found {len(ckpts_e)} explicit checkpoints: '
          f'{[_step_from_path(c) for c in ckpts_e]}', flush=True)
    print(f'Found {len(ckpts_i)} implicit checkpoints: '
          f'{[_step_from_path(c) for c in ckpts_i]}', flush=True)
    print(f'Explicit snapshot dir: {snap_dir_e}', flush=True)
    print(f'Implicit snapshot dir: {snap_dir_i}', flush=True)

    # ---- Load snapshot NPZ files ----
    print('\nLoading snapshots...', flush=True)
    snapshots_e = load_snapshots(snap_dir_e)
    snapshots_i = load_snapshots(snap_dir_i)
    print(f'  Explicit: {len(snapshots_e)} snapshots at steps '
          f'{[s["step"] for s in snapshots_e]}', flush=True)
    print(f'  Implicit: {len(snapshots_i)} snapshots at steps '
          f'{[s["step"] for s in snapshots_i]}', flush=True)

    # ---- Load metrics.jsonl ----
    print('\nLoading metrics logs...', flush=True)
    metrics_e = load_metrics_jsonl(metrics_path_e)
    metrics_i = load_metrics_jsonl(metrics_path_i)
    print(f'  Explicit: {len(metrics_e)} log entries', flush=True)
    print(f'  Implicit: {len(metrics_i)} log entries', flush=True)

    if not args.skip_env_collection:
        from src.train import load_config
        exp_config = load_config('explicit', {'env': args.env, 'seed': args.seed,
                                              'device': str(device), 'mixed_precision': False})
        imp_config = load_config('implicit', {'env': args.env, 'seed': args.seed,
                                              'device': str(device), 'mixed_precision': False})

        from src.envs.factory import make_env
        action_repeat = getattr(exp_config, 'action_repeat', 2)
        env_e = make_env(args.env, seed=args.seed, action_repeat=action_repeat)
        env_i = make_env(args.env, seed=args.seed, action_repeat=action_repeat)

        # ---- Load final agents for main analyses ----
        print('\nLoading final agents...', flush=True)
        agent_e, _ = load_agent('explicit', args.explicit_ckpt, env_e, exp_config, device)
        agent_i, _ = load_agent('implicit', args.implicit_ckpt, env_i, imp_config, device)

        print(f'\nCollecting {args.n_samples} samples from each agent...', flush=True)
        obs_e, lat_e, rew_e = collect_latents(agent_e, env_e, args.n_samples, device)
        obs_i, lat_i, rew_i = collect_latents(agent_i, env_i, args.n_samples, device)
        print(f'Explicit latent shape: {lat_e.shape}', flush=True)
        print(f'Implicit latent shape: {lat_i.shape}', flush=True)

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
            _, axes = plt.subplots(1, 2, figsize=(14, 6))
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
            plt.savefig(os.path.join(args.output, 'pca_comparison.png'), dpi=150,
                        bbox_inches='tight')
            plt.close()
            print(f'  Saved: {os.path.join(args.output, "pca_comparison.png")}', flush=True)

        # [6] Representation Evolution (uses all discovered checkpoints)
        print(f'\n[6] Representation Evolution '
              f'({len(ckpts_e)} explicit, {len(ckpts_i)} implicit checkpoints)...', flush=True)
        del agent_e, agent_i  # free GPU before multi-ckpt loop
        results['representation_evolution'] = representation_evolution(
            ckpts_e, ckpts_i, env_e, env_i,
            exp_config, imp_config, device,
            args.n_samples_evo, args.output,
        )
        env_e.close()
        env_i.close()

    # [7] Snapshot Geometry Timeline (no agent/env needed — pure NPZ loading)
    if snapshots_e or snapshots_i:
        print('\n[7] Snapshot Geometry Timeline...', flush=True)
        results['snapshot_geometry'] = snapshot_geometry_analysis(
            snapshots_e, snapshots_i, args.output)
    else:
        print('\n[7] Snapshot Geometry Timeline: no snapshots found, skipping.', flush=True)

    # [8] Metrics Timeline (KL utilization, horizon degradation, MPPI planner)
    if metrics_e or metrics_i:
        print('\n[8] Metrics Timeline...', flush=True)
        results['metrics_timeline'] = metrics_timeline_analysis(
            metrics_e, metrics_i, args.output)
    else:
        print('\n[8] Metrics Timeline: no metrics.jsonl found, skipping.', flush=True)

    # [9] Cross-Agent CCA Timeline (from snapshots at shared steps)
    if snapshots_e and snapshots_i:
        print('\n[9] Cross-Agent CCA Timeline...', flush=True)
        results['cross_agent_cca_timeline'] = cross_agent_cca_timeline(
            snapshots_e, snapshots_i, args.output)
    else:
        print('\n[9] Cross-Agent CCA Timeline: need snapshots from both agents, skipping.',
              flush=True)

    save_report(results, args.output)


if __name__ == '__main__':
    main()
