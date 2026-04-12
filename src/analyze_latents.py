"""
Latent Space Analysis Script.

Probes the learned representations of the Explicit (RSSM) and Implicit (TD-MPC2)
agents with three analyses:

1. Linear Probing (R² score):
   Train a Ridge regression on frozen latents to predict ground-truth physical
   state coordinates. High R² = latent space linearly encodes physics.
   Tests the "Linear Representation Hypothesis".

2. Canonical Correlation Analysis (CCA):
   Feed identical observation sequences to both agents. CCA measures how much
   the two latent manifolds align on shared "task-relevant" dimensions.
   A high mean canonical correlation suggests both agents discover similar structure.

3. PCA Visualization:
   Reduce latents to 2D via PCA and scatter-plot them, coloured by reward or
   a physical coordinate. Illustrates whether RSSM's discrete latents cluster
   vs. TD-MPC2's continuous manifold.

Usage:
    python src/analyze_latents.py \
        --explicit_ckpt logs/explicit/dmc_walker_walk/seed_0/checkpoints/checkpoint_0500000.pt \
        --implicit_ckpt logs/implicit/dmc_walker_walk/seed_0/checkpoints/checkpoint_0500000.pt \
        --env dmc_walker_walk \
        --n_samples 5000 \
        --output analysis/
"""

import argparse
import os
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
    print('[WARNING] scikit-learn not found; linear probing and CCA will be skipped.')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print('[WARNING] matplotlib/seaborn not found; visualisation will be skipped.')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--explicit_ckpt', type=str, required=True)
    p.add_argument('--implicit_ckpt', type=str, required=True)
    p.add_argument('--env', type=str, required=True)
    p.add_argument('--n_samples', type=int, default=5000,
                   help='Number of (obs, latent) pairs to collect.')
    p.add_argument('--output', type=str, default='analysis/')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def load_agent(agent_type: str, ckpt_path: str, env, config, device):
    """Load an agent from checkpoint."""
    from src.train import make_agent
    agent = make_agent(agent_type, env.obs_dim, env.act_dim,
                       env.act_continuous, config)
    agent = agent.to(device)
    payload = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(payload['agent'])
    agent.eval()
    print(f'  Loaded {agent_type} from step {payload["step"]}')
    return agent


def collect_latents(agent, env, n_samples: int, device) -> tuple:
    """Collect (obs, latent, reward) tuples by rolling out the agent.

    Returns:
        obs_arr: [N, obs_dim]
        latent_arr: [N, latent_dim]
        reward_arr: [N]
    """
    obs_list, latent_list, reward_list = [], [], []
    obs = env.reset()
    agent.reset_state()
    done = False
    ep_step = 0

    while len(obs_list) < n_samples:
        obs_arr = np.array(obs, dtype=np.float32)
        latent = agent.encode(obs_arr[np.newaxis])[0]  # [latent_dim]
        obs_list.append(obs_arr)
        latent_list.append(latent)

        action = agent.act(obs_arr, reset=(ep_step == 0), training=False)
        obs, reward, done, _ = env.step(action)
        reward_list.append(reward)
        ep_step += 1

        if done:
            obs = env.reset()
            agent.reset_state()
            done = False
            ep_step = 0

    return (np.stack(obs_list[:n_samples]),
            np.stack(latent_list[:n_samples]),
            np.array(reward_list[:n_samples]))


# ---------------------------------------------------------------------------
# Analysis 1: Linear Probing
# ---------------------------------------------------------------------------

def linear_probing(obs: np.ndarray, latents: np.ndarray,
                   agent_name: str) -> dict:
    """Train Ridge regression on frozen latents to predict each obs coordinate.

    Args:
        obs: [N, obs_dim] — ground-truth physical state (proxy for physics).
        latents: [N, latent_dim] — frozen latent vectors.
        agent_name: label for printing.

    Returns:
        dict: {coord_index: r2_score, 'mean_r2': float}
    """
    if not SKLEARN_AVAILABLE:
        return {}

    scaler = StandardScaler()
    Z = scaler.fit_transform(latents)

    results = {}
    r2_scores = []
    ridge = Ridge(alpha=1.0)

    for coord in range(obs.shape[1]):
        y = obs[:, coord]
        # 5-fold cross-validated R²
        scores = cross_val_score(ridge, Z, y, cv=5, scoring='r2')
        r2 = float(np.mean(scores))
        results[f'coord_{coord}'] = r2
        r2_scores.append(r2)

    results['mean_r2'] = float(np.mean(r2_scores))
    print(f'  [{agent_name}] Linear probing — Mean R²: {results["mean_r2"]:.4f}')
    return results


# ---------------------------------------------------------------------------
# Analysis 2: CCA Alignment
# ---------------------------------------------------------------------------

def cca_alignment(latents_explicit: np.ndarray,
                  latents_implicit: np.ndarray,
                  n_components: int = 10) -> dict:
    """Compute CCA canonical correlations between two latent spaces.

    Both agents should have been given the *same* observations in the same order.

    Args:
        latents_explicit: [N, latent_explicit]
        latents_implicit: [N, latent_implicit]
        n_components: number of canonical components to compute.

    Returns:
        dict with 'canonical_correlations' and 'mean_correlation'.
    """
    if not SKLEARN_AVAILABLE:
        return {}

    n_components = min(n_components, latents_explicit.shape[1],
                       latents_implicit.shape[1], latents_explicit.shape[0] // 2)

    scaler_e = StandardScaler()
    scaler_i = StandardScaler()
    Ze = scaler_e.fit_transform(latents_explicit)
    Zi = scaler_i.fit_transform(latents_implicit)

    # PCA reduce to manageable dimension before CCA
    pca_dim = min(50, Ze.shape[1], Zi.shape[1])
    pca_e = PCA(n_components=pca_dim).fit(Ze)
    pca_i = PCA(n_components=pca_dim).fit(Zi)
    Ze_r = pca_e.transform(Ze)
    Zi_r = pca_i.transform(Zi)

    cca = CCA(n_components=n_components, max_iter=500)
    cca.fit(Ze_r, Zi_r)
    Xe, Xf = cca.transform(Ze_r, Zi_r)

    # Canonical correlations
    corrs = [float(np.corrcoef(Xe[:, i], Xf[:, i])[0, 1])
             for i in range(n_components)]
    mean_corr = float(np.mean(corrs))

    print(f'  CCA mean canonical correlation: {mean_corr:.4f}')
    print(f'  Top-3 correlations: {corrs[:3]}')
    return {
        'canonical_correlations': corrs,
        'mean_correlation': mean_corr,
    }


# ---------------------------------------------------------------------------
# Analysis 3: PCA Visualization
# ---------------------------------------------------------------------------

def pca_visualization(latents: np.ndarray, rewards: np.ndarray,
                       agent_name: str, output_dir: str):
    """2D PCA scatter plot coloured by reward."""
    if not MPL_AVAILABLE or not SKLEARN_AVAILABLE:
        return

    pca = PCA(n_components=2)
    Z2 = pca.fit_transform(latents)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=rewards, cmap='viridis',
                    alpha=0.4, s=5)
    plt.colorbar(sc, ax=ax, label='Reward')
    ax.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% var)')
    ax.set_title(f'{agent_name} Latent Space (PCA)')

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'pca_{agent_name.lower()}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved PCA plot: {path}')


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def save_report(results: dict, output_dir: str):
    """Save a text summary of all analysis results."""
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
                        f.write(f'  {k}: [{", ".join(f"{x:.4f}" for x in v[:5])}...]\n')
                    else:
                        f.write(f'  {k}: {v}\n')
            else:
                f.write(f'  {data}\n')
            f.write('\n')
    print(f'\nReport saved: {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build config for both agents
    from src.train import load_config
    exp_config = load_config('explicit', {'env': args.env, 'seed': args.seed,
                                           'device': str(device),
                                           'mixed_precision': False})
    imp_config = load_config('implicit', {'env': args.env, 'seed': args.seed,
                                           'device': str(device),
                                           'mixed_precision': False})

    from src.envs.factory import make_env
    env_explicit = make_env(args.env, seed=args.seed,
                             action_repeat=getattr(exp_config, 'action_repeat', 2))
    env_implicit = make_env(args.env, seed=args.seed,
                             action_repeat=getattr(imp_config, 'action_repeat', 2))

    print('Loading agents...')
    agent_explicit = load_agent('explicit', args.explicit_ckpt,
                                env_explicit, exp_config, device)
    agent_implicit = load_agent('implicit', args.implicit_ckpt,
                                env_implicit, imp_config, device)

    print(f'\nCollecting {args.n_samples} samples from each agent...')
    obs_e, latents_e, rewards_e = collect_latents(
        agent_explicit, env_explicit, args.n_samples, device)
    obs_i, latents_i, rewards_i = collect_latents(
        agent_implicit, env_implicit, args.n_samples, device)

    print(f'Explicit latent shape: {latents_e.shape}')
    print(f'Implicit latent shape: {latents_i.shape}')

    results = {}

    # 1. Linear probing
    print('\n[1] Linear Probing (R² scores)...')
    results['linear_probing_explicit'] = linear_probing(obs_e, latents_e, 'Explicit')
    results['linear_probing_implicit'] = linear_probing(obs_i, latents_i, 'Implicit')

    r2_e = results['linear_probing_explicit'].get('mean_r2', 'N/A')
    r2_i = results['linear_probing_implicit'].get('mean_r2', 'N/A')
    print(f'\n  >> Explicit mean R²: {r2_e}  |  Implicit mean R²: {r2_i}')
    if isinstance(r2_e, float) and isinstance(r2_i, float):
        winner = 'Explicit' if r2_e > r2_i else 'Implicit'
        print(f'  >> {winner} latents are more linearly decodable '
              f'(better physical grounding)')

    # 2. CCA alignment (same observations for both agents)
    print('\n[2] CCA Alignment...')
    # Re-collect with same obs sequence for fair CCA
    min_n = min(len(latents_e), len(latents_i), args.n_samples)
    results['cca_alignment'] = cca_alignment(
        latents_e[:min_n], latents_i[:min_n]
    )

    # 3. PCA visualization
    print('\n[3] PCA Visualization...')
    pca_visualization(latents_e, rewards_e, 'Explicit (RSSM)', args.output)
    pca_visualization(latents_i, rewards_i, 'Implicit (TD-MPC2)', args.output)

    # Side-by-side comparison
    if MPL_AVAILABLE and SKLEARN_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, latents, rewards, name in [
            (axes[0], latents_e, rewards_e, 'Explicit (RSSM)'),
            (axes[1], latents_i, rewards_i, 'Implicit (TD-MPC2)'),
        ]:
            pca = PCA(n_components=2).fit(latents)
            Z2 = pca.transform(latents)
            sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=rewards,
                            cmap='plasma', alpha=0.3, s=4)
            plt.colorbar(sc, ax=ax, label='Reward')
            ax.set_title(name)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
        plt.suptitle(f'Latent Space Topology — {args.env}', fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(args.output, 'pca_comparison.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved comparison plot: {out_path}')

    save_report(results, args.output)
    env_explicit.close()
    env_implicit.close()


if __name__ == '__main__':
    main()
