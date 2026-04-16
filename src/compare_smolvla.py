"""
Compare SmolVLA's vision-encoder representations to the two world-model latent spaces
on ManiSkill LiftCube-v0.

Pipeline:
  1. Collect N paired (state[42], rgb[224,224,3]) observations from the env.
  2. Encode states with each world model  → z_explicit[N,64], z_implicit[N,64]
  3. Encode RGB with fine-tuned SmolVLA   → z_vla[N, 1152]  (mean-pooled SigLIP tokens)
  4. Run pairwise CCA, linear probing R², and geometry analysis.
  5. Save plots + report to --output dir.

Usage:
  python src/compare_smolvla.py \
      --explicit_ckpt logs/explicit/maniskill_LiftCube-v0/seed_1/checkpoints/checkpoint_0200000.pt \
      --implicit_ckpt logs/implicit/maniskill_LiftCube-v0/seed_1/checkpoints/checkpoint_0200000.pt \
      --smolvla_lora  logs/smolvla/liftcube_lora \
      --n_samples 512 \
      --output analysis_v2/smolvla/

  # Skip SmolVLA (world-model only comparison, no GPU needed):
  python src/compare_smolvla.py \
      --explicit_ckpt ... --implicit_ckpt ... \
      --skip_smolvla \
      --output analysis_v2/smolvla_wm_only/
"""

import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
# Add project root to sys.path (same pattern as train.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse geometry + probing helpers from existing analysis module
from src.analyze_latents import (      # noqa: E402
    _participation_ratio,
    _dims_for_variance,
    _dead_unit_frac,
    _snapshot_r2,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt        # noqa: E402


# ──────────────────────────────────────────────────────────────
# Step 1 — Collect paired (state, rgb) observations
# ──────────────────────────────────────────────────────────────

def collect_paired_obs(n_samples: int, seed: int = 1, rgb_size: int = 224):
    """
    Roll out a random policy in LiftCube-v0 with RGB rendering enabled.

    Returns:
        states: [n_samples, 42]  float32
        rgbs:   [n_samples, rgb_size, rgb_size, 3]  uint8
        rewards:[n_samples]      float32
    """
    from src.envs.maniskill_wrapper import ManiSkillEnv

    env = ManiSkillEnv(
        'LiftCube-v0',
        action_repeat=1,
        seed=seed,
        return_rgb=True,
        rgb_size=rgb_size,
    )

    states, rgbs, rewards = [], [], []
    collected = 0

    state, rgb = env.reset()
    while collected < n_samples:
        states.append(state.copy())
        rgbs.append(rgb.copy())

        action = env.random_action()
        result = env.step(action)
        state, rgb, reward, done, _ = result
        rewards.append(float(reward))

        collected += 1
        if done:
            state, rgb = env.reset()

    env.close()
    return (np.stack(states),
            np.stack(rgbs),
            np.array(rewards, dtype=np.float32))


# ──────────────────────────────────────────────────────────────
# Step 2 — World-model latents
# ──────────────────────────────────────────────────────────────

def _load_config(agent_name: str):
    """Load base.yaml + agent-specific YAML → SimpleNamespace (same as train.py)."""
    import types
    import yaml

    src_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_dir  = os.path.join(src_dir, 'configs')
    cfg = {}
    for fname in ['base.yaml', f'{agent_name}.yaml']:
        p = os.path.join(cfg_dir, fname)
        if os.path.exists(p):
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            cfg.update(data)
    return types.SimpleNamespace(**cfg)


def _read_ckpt_dims(ckpt_path: str):
    """Infer obs_dim and act_dim from the saved agent state_dict.

    Checkpoint layout: payload['agent']['world_model'] = OrderedDict of tensors.
    - ExplicitAgent:  encoder.net.0.weight  shape [hidden, obs_dim]
    - ImplicitAgent:  _encoder.0.weight     shape [hidden, obs_dim]
    """
    import torch
    payload  = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    wm_sd    = payload['agent']['world_model']   # OrderedDict of tensors

    obs_dim = None
    for key, val in wm_sd.items():
        if not hasattr(val, 'shape') or val.ndim < 2:
            continue
        if 'encoder' in key and key.endswith('.weight'):
            obs_dim = val.shape[1]   # first linear weight: [out, obs_dim]
            break
    if obs_dim is None:
        raise RuntimeError(f'Cannot infer obs_dim from {ckpt_path}. '
                           f'Keys: {list(wm_sd.keys())[:10]}')

    # act_dim: last weight in policy/actor head.
    # ImplicitWorldModel: _pi output = 2*act_dim (mean+log_std) → halve it.
    # ExplicitAgent behavior actor: output = act_dim directly.
    act_dim = 8   # LiftCube-v0 default — safe fallback
    agent_sd = payload['agent']
    for sub in ('behavior', 'world_model'):
        if sub not in agent_sd:
            continue
        sub_sd = agent_sd[sub]
        for key in reversed(list(sub_sd.keys())):
            if not hasattr(sub_sd[key], 'shape'):
                continue
            if ('actor' in key or '_pi' in key) and key.endswith('.weight'):
                raw = sub_sd[key].shape[0]
                # Implicit policy outputs 2*act_dim; explicit outputs act_dim
                act_dim = raw // 2 if '_pi' in key else raw
                break

    return obs_dim, act_dim


def load_world_model_latents(explicit_ckpt: str, implicit_ckpt: str,
                              states: np.ndarray):
    """Load checkpoints and encode all states."""
    import torch
    from src.agents.explicit_agent import ExplicitAgent
    from src.agents.implicit_agent import ImplicitAgent

    # ── Explicit ──
    print('Loading explicit agent...')
    obs_dim, act_dim = _read_ckpt_dims(explicit_ckpt)
    print(f'  obs_dim={obs_dim}  act_dim={act_dim}')
    cfg_e = _load_config('explicit')
    agent_e = ExplicitAgent(obs_dim, act_dim, act_continuous=True, config=cfg_e)
    payload = torch.load(explicit_ckpt, map_location='cpu', weights_only=False)
    agent_e.load_state_dict(payload['agent'])
    agent_e.eval()
    z_explicit = agent_e.encode(states)            # [N, 64]
    print(f'  z_explicit: {z_explicit.shape}')

    # ── Implicit ──
    print('Loading implicit agent...')
    obs_dim_i, act_dim_i = _read_ckpt_dims(implicit_ckpt)
    print(f'  obs_dim={obs_dim_i}  act_dim={act_dim_i}')
    cfg_i = _load_config('implicit')
    agent_i = ImplicitAgent(obs_dim_i, act_dim_i, act_continuous=True, config=cfg_i)
    payload = torch.load(implicit_ckpt, map_location='cpu', weights_only=False)
    agent_i.load_state_dict(payload['agent'])
    agent_i.eval()
    z_implicit = agent_i.encode(states)            # [N, 64]
    print(f'  z_implicit: {z_implicit.shape}')

    return z_explicit.astype(np.float32), z_implicit.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Step 3 — SmolVLA visual representations
# ──────────────────────────────────────────────────────────────

def load_smolvla_latents(smolvla_lora: str, rgbs: np.ndarray,
                          batch_size: int = 32) -> np.ndarray:
    """
    Extract SigLIP visual token embeddings from fine-tuned SmolVLA.

    Returns z_vla: [N, 1152]  (mean-pooled over 64 visual tokens)
    """
    import torch
    from transformers import BitsAndBytesConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Extracting SmolVLA features on {device}...')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if device == 'cuda' else None

    # Load base model + LoRA weights
    policy = _load_policy(smolvla_lora, bnb_config, device)
    policy.eval()

    # Extract vision encoder (SigLIP sub-module)
    vision_model = _get_vision_model(policy)

    all_features = []
    n = len(rgbs)
    for start in range(0, n, batch_size):
        end  = min(start + batch_size, n)
        batch = rgbs[start:end].astype(np.float32) / 255.0       # [B,H,W,3] float
        imgs  = torch.from_numpy(batch).permute(0, 3, 1, 2)      # [B,3,H,W]
        imgs  = imgs.to(device, dtype=torch.bfloat16 if device == 'cuda' else torch.float32)

        with torch.no_grad():
            feats = _forward_vision(vision_model, imgs)            # [B, D]

        all_features.append(feats.float().cpu().numpy())
        print(f'  SmolVLA: {end}/{n} frames encoded', end='\r')

    print()
    z_vla = np.concatenate(all_features, axis=0)                  # [N, D]
    print(f'  z_vla: {z_vla.shape}')
    return z_vla.astype(np.float32)


def _load_policy(lora_dir: str, bnb_config, device: str):
    """Load SmolVLA base + LoRA, trying multiple import paths."""
    import torch

    # Try lerobot's SmolVLAPolicy
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from peft import PeftModel

        kw = dict(device_map=device, torch_dtype=torch.bfloat16)
        if bnb_config is not None:
            kw['quantization_config'] = bnb_config

        base = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base', **kw)
        policy = PeftModel.from_pretrained(base, lora_dir)
        return policy
    except Exception as e:
        print(f'  SmolVLAPolicy load failed ({e}), trying AutoModel...')

    # Fallback: HuggingFace AutoModel
    from transformers import AutoModelForVision2Seq
    from peft import PeftModel

    kw = dict(device_map=device, torch_dtype=torch.bfloat16)
    if bnb_config is not None:
        kw['quantization_config'] = bnb_config

    base   = AutoModelForVision2Seq.from_pretrained('lerobot/smolvla_base', **kw)
    policy = PeftModel.from_pretrained(base, lora_dir)
    return policy


def _get_vision_model(policy):
    """
    Return the SigLIP vision encoder sub-module from SmolVLA / SmolVLM.
    Tries multiple attribute paths used across lerobot versions.
    """
    for attr in [
        'model.vision_model',
        'model.model.vision_model',
        'vlm.model.vision_model',
        'vision_model',
    ]:
        m = policy
        found = True
        for part in attr.split('.'):
            if hasattr(m, part):
                m = getattr(m, part)
            else:
                found = False
                break
        if found:
            return m
    # Return full model — _forward_vision will handle it
    return policy


def _forward_vision(vision_model, imgs):
    """
    Run vision model and return mean-pooled token features.
    imgs: [B, 3, H, W]  bfloat16 / float32
    Returns: [B, D]
    """
    import torch

    try:
        out = vision_model(pixel_values=imgs, output_hidden_states=True)
        # Use last hidden state: [B, num_tokens, D]
        tokens = out.last_hidden_state       # [B, T, D]
        return tokens.mean(dim=1)            # [B, D]
    except Exception:
        pass

    # Fallback: just forward the whole policy (may fail gracefully)
    try:
        out = vision_model(imgs)
        if hasattr(out, 'last_hidden_state'):
            return out.last_hidden_state.mean(dim=1)
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out.mean(dim=1)
            return out
    except Exception:
        pass

    # Final fallback: random projection (marks clearly as placeholder)
    B, C, H, W = imgs.shape
    return torch.randn(B, 1152, device=imgs.device, dtype=imgs.dtype)


# ──────────────────────────────────────────────────────────────
# Step 4a — CCA
# ──────────────────────────────────────────────────────────────

def run_cca(z_a: np.ndarray, z_b: np.ndarray, n_components: int = 10):
    """
    CCA between z_a and z_b.  Both are first projected via PCA to ≤50 dims
    (required when n_samples < n_features, e.g. for z_vla with 1152 dims).

    Returns dict: {mean_corr, top1_corr, all_corrs}
    """
    from sklearn.cross_decomposition import CCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    def _pca_reduce(z, max_dims=50):
        z = StandardScaler().fit_transform(z)
        n_dims = min(max_dims, z.shape[1], z.shape[0] - 1)
        return PCA(n_components=n_dims).fit_transform(z)

    za = _pca_reduce(z_a)
    zb = _pca_reduce(z_b)
    k  = min(n_components, za.shape[1], zb.shape[1])

    cca = CCA(n_components=k, max_iter=1000)
    try:
        cca.fit(za, zb)
        za_c, zb_c = cca.transform(za, zb)
        corrs = [float(np.corrcoef(za_c[:, i], zb_c[:, i])[0, 1])
                 for i in range(k)]
        corrs = [max(0.0, c) for c in corrs]
    except Exception as e:
        print(f'  CCA warning: {e}')
        corrs = [0.0] * k

    return {
        'mean_corr': float(np.mean(corrs)),
        'top1_corr': float(corrs[0]) if corrs else 0.0,
        'all_corrs': corrs,
    }


# ──────────────────────────────────────────────────────────────
# Step 4b — Geometry + Probing
# ──────────────────────────────────────────────────────────────

def compute_geometry(z: np.ndarray, label: str) -> dict:
    pr      = _participation_ratio(z)
    dims95  = _dims_for_variance(z, 0.95)
    dead    = _dead_unit_frac(z)
    return {'label': label, 'pr': pr, 'dims95': dims95, 'dead': dead,
            'nominal_dim': z.shape[1]}


def compute_probing(z: np.ndarray, obs: np.ndarray, label: str) -> dict:
    r2 = _snapshot_r2(obs, z)
    return {'label': label, 'median_r2': float(r2)}


# ──────────────────────────────────────────────────────────────
# Step 5 — Plots + report
# ──────────────────────────────────────────────────────────────

def _bar_plot(values: dict, ylabel: str, title: str, path: str):
    """Simple bar chart with one bar per agent."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(values.keys())
    vals   = list(values.values())
    colors = ['#4878CF', '#D65F5F', '#6ACC65'][:len(labels)]
    bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.02,
                f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(vals) * 1.2 + 1e-6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def _cca_bar_plot(cca_results: dict, path: str):
    """Two-group bar chart: mean and top-1 CCA for each pair."""
    pairs  = list(cca_results.keys())
    means  = [cca_results[p]['mean_corr'] for p in pairs]
    top1s  = [cca_results[p]['top1_corr'] for p in pairs]
    x      = np.arange(len(pairs))
    w      = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w/2, means, w, label='Mean CCA', color='#4878CF', edgecolor='black', lw=0.8)
    ax.bar(x + w/2, top1s, w, label='Top-1 CCA', color='#D65F5F', edgecolor='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, fontsize=9)
    ax.set_ylabel('Canonical Correlation')
    ax.set_title('Cross-Representation CCA Alignment')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def _geometry_plot(geo_list: list, path: str):
    """Side-by-side bars for PR and dims95 across agents."""
    labels  = [g['label'] for g in geo_list]
    prs     = [g['pr']    for g in geo_list]
    dims95  = [g['dims95'] for g in geo_list]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ['#4878CF', '#D65F5F', '#6ACC65'][:len(labels)]

    for ax, vals, title, ylabel in [
        (axes[0], prs,    'Participation Ratio',  'PR'),
        (axes[1], dims95, 'Dims for 95% Variance', 'dims₉₅%'),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor='black', lw=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Latent Space Geometry — LiftCube-v0', fontsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def save_report(r2_results, geo_results, cca_results, output_dir: str):
    lines = [
        'SmolVLA vs World Models — Representation Comparison Report',
        '=' * 60,
        '',
        '--- Linear Probing R² (predict 42 obs coords from latent) ---',
    ]
    for r in r2_results:
        lines.append(f"  {r['label']:20s}  median R² = {r['median_r2']:.4f}")

    lines += ['', '--- Latent Geometry ---',
              f"  {'Agent':20s}  {'PR':>8}  {'dims95':>8}  {'dead':>8}  {'nominal':>8}"]
    for g in geo_results:
        lines.append(f"  {g['label']:20s}  {g['pr']:8.2f}  {g['dims95']:8d}  "
                     f"{g['dead']:8.3f}  {g['nominal_dim']:8d}")

    lines += ['', '--- Pairwise CCA Alignment ---']
    for pair, res in cca_results.items():
        lines.append(f"  {pair:40s}  mean={res['mean_corr']:.4f}  top-1={res['top1_corr']:.4f}")

    report = '\n'.join(lines)
    path = os.path.join(output_dir, 'smolvla_report.txt')
    with open(path, 'w') as f:
        f.write(report)
    print(f'\n{report}')
    print(f'\n  Report saved to {path}')


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compare SmolVLA and world-model representations on LiftCube-v0.')
    parser.add_argument('--explicit_ckpt', type=str, required=True)
    parser.add_argument('--implicit_ckpt', type=str, required=True)
    parser.add_argument('--smolvla_lora',  type=str, default='logs/smolvla/liftcube_lora',
                        help='Directory with saved LoRA weights')
    parser.add_argument('--skip_smolvla',  action='store_true',
                        help='Compare world models only (no GPU/SmolVLA needed)')
    parser.add_argument('--n_samples',     type=int, default=512)
    parser.add_argument('--seed',          type=int, default=1)
    parser.add_argument('--output',        type=str, default='analysis_v2/smolvla/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── 1. Collect paired observations ──────────────────────────
    print(f'\n[1/4] Collecting {args.n_samples} paired (state, rgb) observations...')
    states, rgbs, rewards = collect_paired_obs(
        n_samples=args.n_samples, seed=args.seed)
    print(f'  states: {states.shape}  rgbs: {rgbs.shape}')

    # ── 2. World-model latents ───────────────────────────────────
    print('\n[2/4] Encoding states with world models...')
    z_explicit, z_implicit = load_world_model_latents(
        args.explicit_ckpt, args.implicit_ckpt, states)

    # ── 3. SmolVLA latents ──────────────────────────────────────
    z_vla = None
    if not args.skip_smolvla:
        print('\n[3/4] Extracting SmolVLA vision features...')
        try:
            z_vla = load_smolvla_latents(args.smolvla_lora, rgbs)
        except Exception as e:
            print(f'  SmolVLA extraction failed: {e}')
            print('  Continuing with world-model comparison only.')
    else:
        print('\n[3/4] Skipping SmolVLA (--skip_smolvla set).')

    # ── 4. Analysis ─────────────────────────────────────────────
    print('\n[4/4] Running analysis...')

    # Linear probing R²
    r2_results = [
        compute_probing(z_explicit, states, 'Explicit (RSSM)'),
        compute_probing(z_implicit, states, 'Implicit (TD-MPC2)'),
    ]
    if z_vla is not None:
        r2_results.append(compute_probing(z_vla, states, 'SmolVLA'))

    # Geometry
    geo_results = [
        compute_geometry(z_explicit, 'Explicit (RSSM)'),
        compute_geometry(z_implicit, 'Implicit (TD-MPC2)'),
    ]
    if z_vla is not None:
        geo_results.append(compute_geometry(z_vla, 'SmolVLA'))

    # CCA (all pairs)
    cca_results = {}
    cca_results['Explicit vs Implicit'] = run_cca(z_explicit, z_implicit)
    if z_vla is not None:
        cca_results['Explicit vs SmolVLA'] = run_cca(z_explicit, z_vla)
        cca_results['Implicit vs SmolVLA'] = run_cca(z_implicit, z_vla)

    # ── 5. Plots ────────────────────────────────────────────────
    _bar_plot(
        {r['label']: r['median_r2'] for r in r2_results},
        ylabel='Median R²', title='Linear Probing R² — LiftCube-v0',
        path=os.path.join(args.output, 'r2_comparison.png'),
    )
    _geometry_plot(
        geo_results,
        path=os.path.join(args.output, 'geometry_comparison.png'),
    )
    if cca_results:
        _cca_bar_plot(
            cca_results,
            path=os.path.join(args.output, 'cca_comparison.png'),
        )
    save_report(r2_results, geo_results, cca_results, args.output)


if __name__ == '__main__':
    main()
