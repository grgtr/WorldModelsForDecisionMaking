"""
Statistical result visualisation for the World Models Research Suite.

Reads metrics.jsonl files from multiple seeds and produces:
  1. Learning curves — mean return ± 95% CI over training steps
  2. Sample efficiency — steps to reach 50% of max return (bar chart)
  3. Asymptotic performance — mean return over final 50k steps (bar chart)
  4. Stability — return variance over final 50k steps (box plots)

Usage:
    python src/plot_results.py --logdir logs/ --output figures/ --envs dmc_cartpole_swingup dmc_walker_walk
    python src/plot_results.py --logdir logs/ --output figures/   # auto-discover envs
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
    sns.set_theme(style='whitegrid', font_scale=1.2)
except ImportError:
    MPL_AVAILABLE = False
    print('[ERROR] matplotlib and seaborn are required for plotting.')
    sys.exit(1)


AGENT_COLORS = {
    'explicit': '#2196F3',   # blue
    'implicit': '#FF5722',   # orange-red
}
AGENT_LABELS = {
    'explicit': 'Explicit (RSSM)',
    'implicit': 'Implicit (TD-MPC2)',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--logdir', type=str, default='logs/')
    p.add_argument('--output', type=str, default='figures/')
    p.add_argument('--envs', type=str, nargs='*', default=None,
                   help='Environments to plot. Auto-discovered if omitted.')
    p.add_argument('--window', type=int, default=10,
                   help='Smoothing window for learning curves.')
    p.add_argument('--final_steps', type=int, default=50_000,
                   help='Steps to consider for asymptotic / stability metrics.')
    return p.parse_args()


def load_metrics(logdir: str) -> dict:
    """Load all metrics.jsonl files.

    Returns nested dict:
        data[agent][env][seed] = list of {step, metric: value, ...}
    """
    data = defaultdict(lambda: defaultdict(dict))
    pattern = os.path.join(logdir, '*', '*', 'seed_*', 'metrics.jsonl')
    files = glob.glob(pattern)
    if not files:
        # Also try with different nesting
        pattern2 = os.path.join(logdir, '**', 'metrics.jsonl')
        files = glob.glob(pattern2, recursive=True)

    for fpath in files:
        parts = fpath.replace('\\', '/').split('/')
        # Find agent/env/seed pattern
        seed_idx = next((i for i, p in enumerate(parts) if p.startswith('seed_')), None)
        if seed_idx is None or seed_idx < 2:
            continue
        seed = int(parts[seed_idx].split('_')[1])
        env_name = parts[seed_idx - 1]
        agent = parts[seed_idx - 2]

        records = []
        with open(fpath, 'r') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        data[agent][env_name][seed] = records

    return dict(data)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def extract_series(records: list, key: str = 'eval/mean_return') -> tuple:
    """Extract (steps, values) for a given metric key from records."""
    steps, vals = [], []
    for r in records:
        if key in r and 'step' in r:
            steps.append(r['step'])
            vals.append(r[key])
    return np.array(steps), np.array(vals)


def ci95(arr: np.ndarray) -> float:
    """95% CI half-width using normal approximation."""
    if len(arr) <= 1:
        return 0.0
    return 1.96 * np.std(arr) / np.sqrt(len(arr))


def align_series(all_steps: list, all_vals: list, n_points: int = 200) -> tuple:
    """Interpolate multiple step-value series onto a common step grid."""
    max_step = max(s[-1] for s in all_steps if len(s) > 0)
    min_step = min(s[0] for s in all_steps if len(s) > 0)
    grid = np.linspace(min_step, max_step, n_points)
    interp_vals = []
    for steps, vals in zip(all_steps, all_vals):
        if len(steps) < 2:
            continue
        interp_vals.append(np.interp(grid, steps, vals))
    return grid, np.stack(interp_vals, axis=0)


# ---------------------------------------------------------------------------
# Plot 1: Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(data: dict, envs: list, output_dir: str,
                          window: int = 10):
    for env in envs:
        fig, ax = plt.subplots(figsize=(9, 5))

        for agent in ('explicit', 'implicit'):
            if agent not in data or env not in data[agent]:
                continue
            seeds_data = data[agent][env]

            all_steps, all_vals = [], []
            for seed, records in seeds_data.items():
                steps, vals = extract_series(records, 'eval/mean_return')
                if len(steps) < 2:
                    continue
                if window > 1:
                    vals = smooth(vals, window)
                    steps = steps[:len(vals)]
                all_steps.append(steps)
                all_vals.append(vals)

            if not all_steps:
                continue

            grid, mat = align_series(all_steps, all_vals)
            mean = mat.mean(axis=0)
            ci = np.apply_along_axis(ci95, 0, mat)

            color = AGENT_COLORS[agent]
            label = AGENT_LABELS[agent]
            ax.plot(grid, mean, color=color, linewidth=2, label=label)
            ax.fill_between(grid, mean - ci, mean + ci,
                            color=color, alpha=0.2)

        ax.set_xlabel('Environment Steps')
        ax.set_ylabel('Mean Return')
        ax.set_title(f'Learning Curves — {env}')
        ax.legend(loc='upper left')
        plt.tight_layout()
        path = os.path.join(output_dir, f'learning_curve_{env}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Plot 2: Sample Efficiency (steps to 50% of max return)
# ---------------------------------------------------------------------------

def compute_sample_efficiency(data: dict, envs: list,
                               final_steps: int) -> dict:
    """For each agent-env pair, compute steps to reach 50% of max return."""
    results = defaultdict(lambda: defaultdict(list))
    for env in envs:
        for agent in ('explicit', 'implicit'):
            if agent not in data or env not in data[agent]:
                continue
            for seed, records in data[agent][env].items():
                steps, vals = extract_series(records, 'eval/mean_return')
                if len(vals) == 0:
                    continue
                max_ret = vals.max()
                threshold = 0.5 * max_ret
                idx = np.argmax(vals >= threshold)
                if vals[idx] >= threshold:
                    results[env][agent].append(steps[idx])
    return results


def plot_sample_efficiency(results: dict, envs: list, output_dir: str):
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), squeeze=False)
    for col, env in enumerate(envs):
        ax = axes[0][col]
        agents = ['explicit', 'implicit']
        means, cis, labels, colors = [], [], [], []
        for agent in agents:
            vals = results.get(env, {}).get(agent, [])
            if not vals:
                continue
            means.append(np.mean(vals))
            cis.append(ci95(np.array(vals)))
            labels.append(AGENT_LABELS[agent])
            colors.append(AGENT_COLORS[agent])

        x = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors, alpha=0.8)
        ax.errorbar(x, means, yerr=cis, fmt='none', color='black', capsize=5, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Steps to 50% max return')
        ax.set_title(f'{env}')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v/1e3:.0f}k'))

    plt.suptitle('Sample Efficiency (lower is better)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'sample_efficiency.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Plot 3 & 4: Asymptotic Performance + Stability
# ---------------------------------------------------------------------------

def plot_asymptotic_stability(data: dict, envs: list, output_dir: str,
                               final_steps: int = 50_000):
    asym_means = defaultdict(lambda: defaultdict(list))
    asym_vars = defaultdict(lambda: defaultdict(list))

    for env in envs:
        for agent in ('explicit', 'implicit'):
            if agent not in data or env not in data[agent]:
                continue
            for seed, records in data[agent][env].items():
                steps, vals = extract_series(records, 'eval/mean_return')
                if len(vals) == 0:
                    continue
                max_step = steps.max()
                mask = steps >= max_step - final_steps
                if mask.sum() == 0:
                    mask = np.ones(len(steps), dtype=bool)
                asym_means[env][agent].append(vals[mask].mean())
                asym_vars[env][agent].append(vals[mask].var())

    # ---- Asymptotic performance ----
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), squeeze=False)
    for col, env in enumerate(envs):
        ax = axes[0][col]
        agents = ['explicit', 'implicit']
        means, cis, labels, colors = [], [], [], []
        for agent in agents:
            vals = asym_means.get(env, {}).get(agent, [])
            if not vals:
                continue
            means.append(np.mean(vals))
            cis.append(ci95(np.array(vals)))
            labels.append(AGENT_LABELS[agent])
            colors.append(AGENT_COLORS[agent])

        x = np.arange(len(labels))
        ax.bar(x, means, color=colors, alpha=0.8)
        ax.errorbar(x, means, yerr=cis, fmt='none', color='black', capsize=5, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right')
        ax.set_ylabel('Mean return (last 50k steps)')
        ax.set_title(f'{env}')

    plt.suptitle('Asymptotic Performance (higher is better)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'asymptotic_performance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')

    # ---- Stability (box plots of variance) ----
    fig, axes = plt.subplots(1, len(envs), figsize=(5 * len(envs), 5), squeeze=False)
    for col, env in enumerate(envs):
        ax = axes[0][col]
        box_data = []
        box_labels = []
        box_colors = []
        for agent in ('explicit', 'implicit'):
            vals = asym_vars.get(env, {}).get(agent, [])
            if not vals:
                continue
            box_data.append(vals)
            box_labels.append(AGENT_LABELS[agent])
            box_colors.append(AGENT_COLORS[agent])

        if box_data:
            bplot = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, color in zip(bplot['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        ax.set_ylabel('Return variance (last 50k steps)')
        ax.set_title(f'{env}')

    plt.suptitle('Stability (lower variance is better)', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'stability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print(f'Loading metrics from: {args.logdir}')
    data = load_metrics(args.logdir)

    if not data:
        print('[ERROR] No metrics.jsonl files found. Check --logdir path.')
        return

    # Discover environments
    all_envs = set()
    for agent_data in data.values():
        all_envs.update(agent_data.keys())
    envs = args.envs if args.envs else sorted(all_envs)
    print(f'Agents found: {sorted(data.keys())}')
    print(f'Environments: {envs}')

    print('\nPlotting learning curves...')
    plot_learning_curves(data, envs, args.output, window=args.window)

    print('Computing sample efficiency...')
    eff = compute_sample_efficiency(data, envs, args.final_steps)
    plot_sample_efficiency(eff, envs, args.output)

    print('Plotting asymptotic performance and stability...')
    plot_asymptotic_stability(data, envs, args.output, args.final_steps)

    print(f'\nAll figures saved to: {args.output}')


if __name__ == '__main__':
    main()
