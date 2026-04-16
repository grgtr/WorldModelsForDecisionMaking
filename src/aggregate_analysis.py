"""
Aggregate latent-space and training metrics across seeds.

Reads snapshots + metrics.jsonl directly — does not require the per-seed
analyze_latents.py output files.  Produces:
  {output}/{env}/learning_curves.png
  {output}/{env}/r2_evolution.png
  {output}/{env}/geometry_evolution.png
  {output}/{env}/kl_utilization.png        (explicit only)
  {output}/{env}/planner_health.png        (implicit only)
  {output}/{env}/cca_timeline.png
  {output}/cross_env_summary.png
  {output}/aggregated_report.txt

Usage:
    python src/aggregate_analysis.py \\
        --log_root logs/ \\
        --envs dmc_cartpole_swingup dmc_walker_walk maniskill_LiftCube-v0 \\
        --seeds 1 2 3 \\
        --output analysis_v2/aggregated/
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analyze_latents import (
    load_snapshots, load_metrics_jsonl,
    _participation_ratio, _dims_for_variance, _dead_unit_frac, _snapshot_r2,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False
    print('[WARNING] matplotlib not found; plots will be skipped.')

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cross_decomposition import CCA
    SKLEARN = True
except ImportError:
    SKLEARN = False


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

COLOR_E = 'steelblue'
COLOR_I = 'darkorange'
ALPHA_FILL = 0.20
ENV_LABELS = {
    'dmc_cartpole_swingup': 'DMC Cartpole Swingup',
    'dmc_walker_walk':      'DMC Walker Walk',
    'maniskill_LiftCube-v0': 'ManiSkill LiftCube',
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--log_root', type=str, default='logs/')
    p.add_argument('--envs', nargs='+',
                   default=['dmc_cartpole_swingup', 'dmc_walker_walk',
                            'maniskill_LiftCube-v0'])
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--output', type=str, default='analysis_v2/aggregated/')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _run_dir(log_root, agent, env, seed):
    return os.path.join(log_root, agent, env, f'seed_{seed}')


def load_eval_returns(log_root, agent, env, seed):
    """→ np.ndarray shape [N, 2]: columns = step, mean_return."""
    path = os.path.join(_run_dir(log_root, agent, env, seed), 'metrics.jsonl')
    records = load_metrics_jsonl(path)
    rows = [(r['step'], r['eval/mean_return'])
            for r in records
            if 'eval/mean_return' in r and r['eval/mean_return'] is not None]
    # de-duplicate steps (final eval can be logged twice)
    seen = {}
    for step, val in rows:
        seen[step] = val
    arr = np.array(sorted(seen.items()), dtype=float)
    return arr  # [N, 2]


def load_snapshot_series(log_root, agent, env, seed):
    """→ dict of lists indexed by step.

    Keys: step, pr, dims95, dead, r2
    """
    snap_dir = os.path.join(_run_dir(log_root, agent, env, seed),
                            'latent_snapshots')
    snaps = load_snapshots(snap_dir)
    if not snaps:
        return None
    steps, pr, dims95, dead, r2 = [], [], [], [], []
    for s in snaps:
        z = s['z'].astype(np.float64)
        obs = s['obs'].astype(np.float64)
        steps.append(s['step'])
        pr.append(_participation_ratio(z))
        dims95.append(_dims_for_variance(z, 0.95))
        dead.append(_dead_unit_frac(z))
        r2.append(_snapshot_r2(obs, z))
    return {'step': np.array(steps), 'pr': np.array(pr),
            'dims95': np.array(dims95), 'dead': np.array(dead),
            'r2': np.array(r2)}


def load_metric_series(log_root, agent, env, seed, keys):
    """Extract specific scalar keys from metrics.jsonl → dict step→value arrays.

    Only rows containing at least one of the requested keys are included.
    Returns dict: key → np.ndarray [N, 2] (step, value), NaN where missing.
    """
    path = os.path.join(_run_dir(log_root, agent, env, seed), 'metrics.jsonl')
    records = load_metrics_jsonl(path)
    # Collect rows that have any requested key
    all_steps = sorted({r['step'] for r in records if any(k in r for k in keys)})
    step_to_rec = {}
    for r in records:
        step_to_rec[r['step']] = r  # last record wins for duplicate steps

    out = {k: [] for k in keys}
    out['step'] = []
    for step in all_steps:
        r = step_to_rec.get(step, {})
        out['step'].append(step)
        for k in keys:
            v = r.get(k)
            out[k].append(float(v) if v is not None else np.nan)
    out['step'] = np.array(out['step'])
    for k in keys:
        out[k] = np.array(out[k])
    return out


def load_cca_timeline(log_root, env, seeds):
    """Compute cross-agent CCA at each shared snapshot step, per seed.

    Returns dict: seed → {'step': [...], 'mean_corr': [...], 'top1_corr': [...]}
    """
    if not SKLEARN:
        return {}
    results = {}
    for seed in seeds:
        snaps_e = load_snapshots(
            os.path.join(_run_dir(log_root, 'explicit', env, seed),
                         'latent_snapshots'))
        snaps_i = load_snapshots(
            os.path.join(_run_dir(log_root, 'implicit', env, seed),
                         'latent_snapshots'))
        if not snaps_e or not snaps_i:
            continue
        snap_e = {s['step']: s['z'] for s in snaps_e}
        snap_i = {s['step']: s['z'] for s in snaps_i}
        shared = sorted(set(snap_e) & set(snap_i))
        if not shared:
            continue
        cca_steps, mean_corrs, top1_corrs = [], [], []
        for step in shared:
            ze = snap_e[step].astype(np.float64)
            zi = snap_i[step].astype(np.float64)
            N = min(ze.shape[0], zi.shape[0])
            ze, zi = ze[:N], zi[:N]
            n_comp = min(10, ze.shape[1], zi.shape[1], N // 2)
            Ze = StandardScaler().fit_transform(ze)
            Zi = StandardScaler().fit_transform(zi)
            pca_dim = min(50, Ze.shape[1], Zi.shape[1])
            Ze_r = PCA(n_components=pca_dim).fit_transform(Ze)
            Zi_r = PCA(n_components=pca_dim).fit_transform(Zi)
            try:
                cca = CCA(n_components=n_comp, max_iter=500)
                cca.fit(Ze_r, Zi_r)
                Xe, Xi = cca.transform(Ze_r, Zi_r)
                corrs = [float(np.corrcoef(Xe[:, k], Xi[:, k])[0, 1])
                         for k in range(n_comp)]
                mean_c = float(np.nanmean(corrs))
                top1_c = float(corrs[0]) if corrs else np.nan
            except Exception:
                mean_c = np.nan
                top1_c = np.nan
            cca_steps.append(step)
            mean_corrs.append(mean_c)
            top1_corrs.append(top1_c)
        results[seed] = {
            'step': np.array(cca_steps),
            'mean_corr': np.array(mean_corrs),
            'top1_corr': np.array(top1_corrs),
        }
    return results


# ---------------------------------------------------------------------------
# Aggregation helper
# ---------------------------------------------------------------------------

def aggregate(series_list, key, common_steps=None):
    """Align multiple per-seed series to common steps and compute mean ± std.

    Args:
        series_list: list of dicts with 'step' and key arrays.
        key: the metric key to aggregate.
        common_steps: if None, use intersection of all step arrays.

    Returns:
        steps [M], mean [M], std [M] — NaN-safe.
    """
    valid = [s for s in series_list if s is not None and key in s
             and len(s['step']) > 0]
    if not valid:
        return np.array([]), np.array([]), np.array([])

    if common_steps is None:
        step_sets = [set(s['step'].tolist()) for s in valid]
        common = sorted(set.intersection(*step_sets))
    else:
        common = sorted(common_steps)

    if not common:
        return np.array([]), np.array([]), np.array([])

    common = np.array(common)
    vals = []
    for s in valid:
        step_to_val = dict(zip(s['step'].tolist(), s[key].tolist()))
        row = np.array([step_to_val.get(st, np.nan) for st in common])
        vals.append(row)

    mat = np.array(vals)  # [n_seeds, n_steps]
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return common, mean, std


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _band(ax, steps, mean, std, color, label, ls='-', marker=None):
    """Plot mean line with ±1 std shading."""
    valid = ~np.isnan(mean)
    if not np.any(valid):
        return
    ax.plot(steps[valid], mean[valid], ls, color=color, label=label,
            marker=marker, markersize=4)
    ax.fill_between(steps[valid],
                    (mean - std)[valid], (mean + std)[valid],
                    color=color, alpha=ALPHA_FILL)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}', flush=True)


# ---------------------------------------------------------------------------
# Per-env plots
# ---------------------------------------------------------------------------

def plot_learning_curves(env, seeds, log_root, out_dir):
    """Eval return vs training step, mean ± std across seeds."""
    series_e = [load_eval_returns(log_root, 'explicit', env, s) for s in seeds]
    series_i = [load_eval_returns(log_root, 'implicit', env, s) for s in seeds]

    # Aggregate: use union of steps, NaN-fill missing
    def _agg(series_list):
        valid = [s for s in series_list if s is not None and len(s) > 0]
        if not valid:
            return np.array([]), np.array([]), np.array([])
        all_steps = sorted({int(s[i, 0]) for s in valid for i in range(len(s))})
        mat = []
        for s in valid:
            m = dict(zip(s[:, 0].astype(int), s[:, 1]))
            mat.append([m.get(st, np.nan) for st in all_steps])
        mat = np.array(mat)
        return np.array(all_steps), np.nanmean(mat, 0), np.nanstd(mat, 0)

    steps_e, mean_e, std_e = _agg(series_e)
    steps_i, mean_i, std_i = _agg(series_i)

    if not MPL or (len(steps_e) == 0 and len(steps_i) == 0):
        return {'explicit_final': float(np.nanmean([s[-1, 1] for s in series_e if s is not None and len(s)])),
                'implicit_final': float(np.nanmean([s[-1, 1] for s in series_i if s is not None and len(s)]))}

    fig, ax = plt.subplots(figsize=(8, 5))
    _band(ax, steps_e, mean_e, std_e, COLOR_E, 'Explicit (RSSM)')
    _band(ax, steps_i, mean_i, std_i, COLOR_I, 'Implicit (TD-MPC2)', ls='--')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Return')
    ax.set_title(f'Learning Curves — {ENV_LABELS.get(env, env)}\n'
                 f'(n={len([s for s in series_e if s is not None and len(s)>0])} seeds, mean ± std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, env, 'learning_curves.png'))

    result = {}
    if len(mean_e): result['explicit_final_return'] = float(mean_e[-1])
    if len(mean_i): result['implicit_final_return'] = float(mean_i[-1])
    return result


def plot_r2_evolution(env, seeds, log_root, out_dir, snap_series_e, snap_series_i):
    """Snapshot-based linear probing R² over training steps."""
    steps_e, mean_e, std_e = aggregate(snap_series_e, 'r2')
    steps_i, mean_i, std_i = aggregate(snap_series_i, 'r2')

    if not MPL or (len(steps_e) == 0 and len(steps_i) == 0):
        return {}

    fig, ax = plt.subplots(figsize=(8, 5))
    _band(ax, steps_e, mean_e, std_e, COLOR_E, 'Explicit (RSSM)', marker='o')
    _band(ax, steps_i, mean_i, std_i, COLOR_I, 'Implicit (TD-MPC2)',
          ls='--', marker='s')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Median R²  (z → obs coordinates)')
    ax.set_title(f'Linear Decodability over Training — {ENV_LABELS.get(env, env)}\n'
                 f'(from {512}-sample snapshots, {len(steps_e)} checkpoints)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, env, 'r2_evolution.png'))

    result = {}
    if len(mean_e): result['explicit_final_r2'] = float(mean_e[-1])
    if len(mean_i): result['implicit_final_r2'] = float(mean_i[-1])
    return result


def plot_geometry_evolution(env, seeds, log_root, out_dir, snap_series_e, snap_series_i):
    """Participation ratio and dims-95% over training steps."""
    if not MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, key, ylabel, title in [
        (axes[0], 'pr',     'Participation Ratio', 'Participation Ratio'),
        (axes[1], 'dims95', 'Dims for 95% Variance', 'Effective Dimensionality (95%)'),
    ]:
        steps_e, mean_e, std_e = aggregate(snap_series_e, key)
        steps_i, mean_i, std_i = aggregate(snap_series_i, key)
        _band(ax, steps_e, mean_e, std_e, COLOR_E, 'Explicit (RSSM)', marker='o')
        _band(ax, steps_i, mean_i, std_i, COLOR_I, 'Implicit (TD-MPC2)',
              ls='--', marker='s')
        ax.set_xlabel('Training Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Latent Geometry — {ENV_LABELS.get(env, env)}', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, env, 'geometry_evolution.png'))


def plot_kl_utilization(env, seeds, log_root, out_dir):
    """Explicit-only: KL active variables and min-var KL over training."""
    keys = ['wm/kl_active_vars', 'wm/kl_min_var']
    series = [load_metric_series(log_root, 'explicit', env, s, keys)
              for s in seeds]

    if not MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, key, ylabel, title in [
        (axes[0], 'wm/kl_active_vars',
         'Variables with KL > 0.1 nats (max=8)',
         'KL Active Variables'),
        (axes[1], 'wm/kl_min_var',
         'Min KL across 8 variables (nats)',
         'Least-Active Variable KL'),
    ]:
        steps, mean, std = aggregate(series, key)
        if len(steps):
            _band(ax, steps, mean, std, COLOR_E, 'Explicit (RSSM)')
        ax.set_xlabel('Training Step')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Annotate collapse threshold on active-vars plot
    axes[0].axhline(1.0, color='red', ls='--', alpha=0.4, label='collapse (1 var)')
    axes[0].axhline(8.0, color='green', ls='--', alpha=0.3, label='full use (8 vars)')
    axes[0].legend(fontsize=8)

    plt.suptitle(f'Explicit Agent — KL Utilization — {ENV_LABELS.get(env, env)}',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, env, 'kl_utilization.png'))


def plot_planner_health(env, seeds, log_root, out_dir):
    """Implicit-only: horizon degradation ratio and MPPI elite-value std."""
    keys = ['latent/consistency_t0', 'latent/consistency_tH',
            'latent/mppi_elite_value_std', 'latent/z_active_frac']
    series = [load_metric_series(log_root, 'implicit', env, s, keys)
              for s in seeds]

    # Compute horizon degradation ratio per seed
    ratio_series = []
    for s in series:
        if s is None or len(s['step']) == 0:
            ratio_series.append(None)
            continue
        t0 = s['latent/consistency_t0']
        tH = s['latent/consistency_tH']
        ratio = np.where((t0 > 1e-10) & ~np.isnan(t0) & ~np.isnan(tH),
                         tH / t0, np.nan)
        ratio_series.append({'step': s['step'], 'ratio': ratio})

    mppi_series = [{'step': s['step'], 'mppi_std': s['latent/mppi_elite_value_std']}
                   if s is not None and len(s['step']) > 0 else None
                   for s in series]

    if not MPL:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Horizon degradation
    steps, mean, std = aggregate(ratio_series, 'ratio')
    if len(steps):
        _band(axes[0], steps, mean, std, COLOR_I, 'Implicit (TD-MPC2)')
        axes[0].axhline(1.0, color='gray', ls='--', alpha=0.6,
                        label='no degradation')
    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('consistency_tH / consistency_t0')
    axes[0].set_title('Horizon Degradation Ratio\n(> 1 = multi-step quality worse than single)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # MPPI elite-value std
    steps, mean, std = aggregate(mppi_series, 'mppi_std')
    if len(steps):
        _band(axes[1], steps, mean, std, COLOR_I, 'Implicit (TD-MPC2)')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('MPPI elite value std')
    axes[1].set_title('Planner Diversity\n(collapse → 0 = low exploration)')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Implicit Agent — Planner Health — {ENV_LABELS.get(env, env)}',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, env, 'planner_health.png'))


def plot_cca_timeline(env, seeds, log_root, out_dir):
    """Cross-agent CCA correlation timeline, mean ± std across seeds."""
    print(f'  Computing CCA timeline for {env}...', flush=True)
    seed_cca = load_cca_timeline(log_root, env, seeds)

    if not MPL or not seed_cca:
        return {}

    cca_list_mean = [{'step': v['step'], 'corr': v['mean_corr']}
                     for v in seed_cca.values()]
    cca_list_top1 = [{'step': v['step'], 'corr': v['top1_corr']}
                     for v in seed_cca.values()]

    steps_m, mean_m, std_m = aggregate(cca_list_mean, 'corr')
    steps_t, mean_t, std_t = aggregate(cca_list_top1, 'corr')

    fig, ax = plt.subplots(figsize=(8, 5))
    _band(ax, steps_m, mean_m, std_m, 'mediumseagreen', 'Mean (top-10 CCs)',
          marker='o')
    _band(ax, steps_t, mean_t, std_t, 'darkgreen', 'Top-1 CC',
          ls='--', marker='s')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Canonical Correlation')
    ax.set_title(f'Cross-Agent CCA Timeline — {ENV_LABELS.get(env, env)}\n'
                 f'Explicit ↔ Implicit representation alignment')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, os.path.join(out_dir, env, 'cca_timeline.png'))

    result = {}
    if len(mean_m): result['final_mean_cca'] = float(mean_m[-1])
    if len(mean_t): result['final_top1_cca'] = float(mean_t[-1])
    return result


# ---------------------------------------------------------------------------
# Cross-env summary
# ---------------------------------------------------------------------------

def plot_cross_env_summary(all_results, out_dir):
    """3-env bar chart: final return and final R² for both agents."""
    if not MPL:
        return

    envs = list(all_results.keys())
    env_labels = [ENV_LABELS.get(e, e).replace(' ', '\n') for e in envs]
    x = np.arange(len(envs))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric_e, metric_i, ylabel, title in [
        (axes[0], 'explicit_final_return', 'implicit_final_return',
         'Mean Return (final eval)', 'Final Performance'),
        (axes[1], 'explicit_final_r2', 'implicit_final_r2',
         'Median R²  (z → obs, from snapshots)', 'Final Linear Decodability'),
    ]:
        vals_e = [all_results[e].get(metric_e, np.nan) for e in envs]
        vals_i = [all_results[e].get(metric_i, np.nan) for e in envs]

        bars_e = ax.bar(x - width / 2, vals_e, width, label='Explicit (RSSM)',
                        color=COLOR_E, alpha=0.85)
        bars_i = ax.bar(x + width / 2, vals_i, width, label='Implicit (TD-MPC2)',
                        color=COLOR_I, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(env_labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.2, axis='y')

        # Value labels on bars
        for bar in list(bars_e) + list(bars_i):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * ax.get_ylim()[1],
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Cross-Environment Summary (mean across seeds)', fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, 'cross_env_summary.png'))


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def write_report(all_results, out_dir):
    path = os.path.join(out_dir, 'aggregated_report.txt')
    os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w') as f:
        f.write('Aggregated Analysis Report\n')
        f.write('=' * 60 + '\n\n')
        for env, res in all_results.items():
            f.write(f'--- {ENV_LABELS.get(env, env)} ---\n')
            for k, v in res.items():
                if isinstance(v, float):
                    f.write(f'  {k}: {v:.4f}\n')
                else:
                    f.write(f'  {k}: {v}\n')
            f.write('\n')
    print(f'\nReport saved: {path}', flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    all_results = {}

    for env in args.envs:
        print(f'\n{"="*60}', flush=True)
        print(f'Environment: {ENV_LABELS.get(env, env)}', flush=True)
        print(f'{"="*60}', flush=True)

        env_results = {}
        out_env = os.path.join(args.output, env)
        os.makedirs(out_env, exist_ok=True)

        # ---- Load snapshot series for all seeds ----
        print('\nLoading snapshots...', flush=True)
        snap_series_e, snap_series_i = [], []
        for seed in args.seeds:
            se = load_snapshot_series(args.log_root, 'explicit', env, seed)
            si = load_snapshot_series(args.log_root, 'implicit', env, seed)
            if se:
                print(f'  explicit seed={seed}: {len(se["step"])} snapshots', flush=True)
            else:
                print(f'  explicit seed={seed}: no snapshots', flush=True)
            if si:
                print(f'  implicit seed={seed}: {len(si["step"])} snapshots', flush=True)
            else:
                print(f'  implicit seed={seed}: no snapshots', flush=True)
            snap_series_e.append(se)
            snap_series_i.append(si)

        # [1] Learning curves
        print('\nPlotting learning curves...', flush=True)
        env_results.update(
            plot_learning_curves(env, args.seeds, args.log_root, args.output))

        # [2] R² evolution
        print('\nPlotting R² evolution...', flush=True)
        env_results.update(
            plot_r2_evolution(env, args.seeds, args.log_root, args.output,
                              snap_series_e, snap_series_i))

        # [3] Geometry evolution (PR + dims95)
        print('\nPlotting geometry evolution...', flush=True)
        plot_geometry_evolution(env, args.seeds, args.log_root, args.output,
                                snap_series_e, snap_series_i)

        # [4] KL utilization (explicit)
        print('\nPlotting KL utilization...', flush=True)
        plot_kl_utilization(env, args.seeds, args.log_root, args.output)

        # [5] Planner health (implicit)
        print('\nPlotting planner health...', flush=True)
        plot_planner_health(env, args.seeds, args.log_root, args.output)

        # [6] Cross-agent CCA timeline
        print('\nComputing CCA timeline...', flush=True)
        env_results.update(
            plot_cca_timeline(env, args.seeds, args.log_root, args.output))

        all_results[env] = env_results
        print(f'\nResults for {env}: {env_results}', flush=True)

    # Cross-env summary
    print('\nPlotting cross-env summary...', flush=True)
    plot_cross_env_summary(all_results, args.output)

    write_report(all_results, args.output)
    print('\nDone.', flush=True)


if __name__ == '__main__':
    main()
