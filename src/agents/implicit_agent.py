"""
Implicit World Model Agent (TD-MPC2 style).

Learns a task-centric latent space via consistency + TD losses.
At decision time, performs MPPI (Model Predictive Path Integral) planning
in latent space to select actions.

Key differences from the Explicit agent:
  - No observation decoder (no generative pressure)
  - Consistency loss: dynamics(z_t, a_t) ≈ encoder(o_{t+1})
  - Planning via MPPI/CEM in latent space (not just policy rollout)
  - Two-Q ensemble with discrete (two-hot) regression

Adapted from:
  tdmpc2/tdmpc2/tdmpc2.py (TDMPC2 class)
  tdmpc2/tdmpc2/common/math.py (loss functions)
"""

import torch
import torch.nn as nn
import numpy as np

from ..models.implicit_wm import ImplicitWorldModel
from ..models.math_utils import (
    soft_ce, two_hot_inv, RunningScale, symlog
)


class ImplicitAgent(nn.Module):
    """TD-MPC2-style implicit world model agent with MPPI planning.

    Args:
        obs_dim: flat observation dimension.
        act_dim: flat action dimension.
        act_continuous: must be True (discrete actions not supported by MPPI).
        config: config namespace.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 act_continuous: bool, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_continuous = act_continuous
        self.config = config

        self.world_model = ImplicitWorldModel(obs_dim, act_dim, config)

        # ------ Optimizers ------
        # Encoder trains at enc_lr_scale × base LR (slower)
        enc_params = list(self.world_model.encoder_parameters())
        enc_ids = set(id(p) for p in enc_params)
        target_ids = set(id(p) for p in self.world_model._target_Qs.parameters())
        pi_ids = set(id(p) for p in self.world_model._pi.parameters())
        other_params = [p for p in self.world_model.parameters()
                        if id(p) not in target_ids and id(p) not in pi_ids]

        enc_lr = config.lr * getattr(config, 'enc_lr_scale', 0.3)
        self._model_opt = torch.optim.Adam([
            {'params': enc_params, 'lr': enc_lr},
            {'params': [p for p in other_params if id(p) not in enc_ids],
             'lr': config.lr},
        ], eps=1e-8)
        self._pi_opt = torch.optim.Adam(
            self.world_model.policy_parameters(),
            lr=config.lr, eps=1e-8
        )

        self._grad_clip = getattr(config, 'grad_clip_norm', 20.0)
        self._use_amp = getattr(config, 'mixed_precision', True)
        self._scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

        # Q-value normalisation (running 5th-95th percentile scale)
        self._q_scale = RunningScale(tau=0.01)

        # MPPI warm-start: carry previous mean to next step
        self._plan_mean = None
        self._last_plan_stats = {}  # populated by _plan(), merged into update() return

        # Config shortcuts
        self.horizon = getattr(config, 'horizon', 5)
        self.num_bins = getattr(config, 'num_bins', 101)
        self.vmin = getattr(config, 'vmin', -10.0)
        self.vmax = getattr(config, 'vmax', 10.0)
        self.discount = self._get_discount()

    def _get_discount(self) -> float:
        """Compute discount factor (TD-MPC2 style task-adaptive)."""
        denom = getattr(self.config, 'discount_denom', 5)
        dmin = getattr(self.config, 'discount_min', 0.95)
        dmax = getattr(self.config, 'discount_max', 0.995)
        return max(dmin, min(dmax, 1.0 - 1.0 / denom))

    # ------------------------------------------------------------------
    # MPPI planning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _plan(self, z: torch.Tensor, t0: bool = False,
              eval_mode: bool = False) -> torch.Tensor:
        """MPPI planning in latent space.

        Args:
            z: [1, latent_dim] current latent state.
            t0: True at the start of each episode (resets warm-start).
            eval_mode: True → no action noise at execution.

        Returns:
            action: [act_dim] action to execute.
        """
        cfg = self.config
        H = self.horizon
        N = getattr(cfg, 'num_samples', 256)
        K = getattr(cfg, 'num_elites', 32)
        n_pi = getattr(cfg, 'num_pi_trajs', 16)
        iterations = getattr(cfg, 'mppi_iterations', 4)
        temp = getattr(cfg, 'temperature', 0.5)
        min_std = getattr(cfg, 'min_std', 0.05)
        max_std = getattr(cfg, 'max_std', 2.0)
        device = z.device

        # --- 1. Sample policy-guided trajectories ---
        pi_actions = torch.zeros(H, n_pi, self.act_dim, device=device)
        _z = z.repeat(n_pi, 1)
        for t in range(H):
            _, pi_actions[t], _ = self.world_model.pi(_z)
            _z = self.world_model.next_z(_z, pi_actions[t])

        # --- 2. Initialise action distribution ---
        if t0 or self._plan_mean is None:
            mean = torch.zeros(H, self.act_dim, device=device)
        else:
            # Warm start: shift previous mean by 1 step
            mean = torch.cat([self._plan_mean[1:],
                               torch.zeros(1, self.act_dim, device=device)], dim=0)
        std = max_std * torch.ones(H, self.act_dim, device=device)

        # --- 3. CEM iterations ---
        for _ in range(iterations):
            # Sample random actions
            n_rand = N - n_pi
            rand_actions = (mean.unsqueeze(1) +
                            std.unsqueeze(1) * torch.randn(H, n_rand, self.act_dim,
                                                           device=device))
            rand_actions = rand_actions.clamp(-1.0, 1.0)
            # Concatenate with policy trajectories
            all_actions = torch.cat([pi_actions, rand_actions], dim=1)  # [H, N, A]

            # Evaluate trajectory values
            values = self._estimate_value(z, all_actions)  # [N]

            # Select elites
            elite_values, elite_idx = values.topk(K)
            elite_actions = all_actions[:, elite_idx]  # [H, K, A]

            # Refit distribution with weighted elites
            score = torch.exp(temp * (elite_values - elite_values.max()))
            score = score / score.sum()
            mean = (score.unsqueeze(0).unsqueeze(-1) * elite_actions).sum(dim=1)
            std = (score.unsqueeze(0).unsqueeze(-1) *
                   (elite_actions - mean.unsqueeze(1)).pow(2)).sum(dim=1).sqrt()
            std = std.clamp(min_std, max_std)

        self._plan_mean = mean.clone()

        # Store MPPI diagnostics (merged into update() return metrics)
        self._last_plan_stats = {
            'latent/mppi_elite_value_mean': elite_values.mean().item(),
            'latent/mppi_elite_value_std': elite_values.std().item()
                if elite_values.numel() > 1 else 0.0,
            'latent/mppi_plan_std': std.mean().item(),
        }

        # --- 4. Execute action ---
        # Sample from elite distribution
        elite_score = torch.exp(temp * (elite_values - elite_values.max()))
        elite_score = elite_score / elite_score.sum()
        idx = torch.multinomial(elite_score, 1).item()
        action = elite_actions[0, idx]  # first step of selected trajectory

        if not eval_mode:
            action = action + std[0] * torch.randn_like(action)
        return action.clamp(-1.0, 1.0)

    @torch.no_grad()
    def _estimate_value(self, z: torch.Tensor,
                        actions: torch.Tensor) -> torch.Tensor:
        """Estimate trajectory return for MPPI evaluation.

        Args:
            z: [1, latent_dim] initial latent.
            actions: [H, N, act_dim] action sequences to evaluate.

        Returns:
            values: [N] estimated returns.
        """
        H, N, _ = actions.shape
        _z = z.repeat(N, 1)  # [N, latent_dim]
        G = torch.zeros(N, device=z.device)
        discount = 1.0

        for t in range(H):
            a = actions[t]  # [N, act_dim]
            # Reward prediction
            r_logits = self.world_model.reward_logits(_z, a)
            r = two_hot_inv(r_logits, self.num_bins, self.vmin, self.vmax).squeeze(-1)
            G += discount * r
            discount *= self.discount
            _z = self.world_model.next_z(_z, a)

        # Bootstrap with min Q-value of policy action
        _, pi, _ = self.world_model.pi(_z)
        q_logits_list = self.world_model.q_values(_z, pi, target=False)
        q_vals = torch.stack([
            two_hot_inv(ql, self.num_bins, self.vmin, self.vmax).squeeze(-1)
            for ql in q_logits_list
        ], dim=0)  # [num_q, N]
        G += discount * q_vals.min(dim=0).values

        return G

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, reset: bool = False,
            training: bool = True) -> np.ndarray:
        """Select an action given the current observation.

        For the implicit agent, obs → encode → (MPPI plan or π sample).

        Args:
            obs: [obs_dim] numpy array.
            reset: True at episode start (resets MPPI warm-start).
            training: True → add exploration noise; False → deterministic.

        Returns:
            action: [act_dim] numpy array.
        """
        device = next(self.world_model.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            z = self.world_model.encode(obs_t)  # [1, latent_dim]

            use_mpc = getattr(self.config, 'mpc', True)
            if use_mpc:
                action = self._plan(z, t0=reset, eval_mode=not training)
            else:
                mu, pi, _ = self.world_model.pi(z.squeeze(0))
                action = pi if training else mu

        if action.dim() == 0:
            action = action.unsqueeze(0)
        return action.cpu().numpy().flatten()

    def reset_state(self):
        """Reset MPPI warm-start (call at episode start)."""
        self._plan_mean = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch: dict) -> dict:
        """One training update on a short trajectory batch.

        Args:
            batch: dict with numpy arrays:
              obs [B, H+1, O], action [B, H, A],
              reward [B, H], done [B, H]

        Returns:
            metrics dict.
        """
        device = next(self.world_model.parameters()).device

        def to_tensor(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)

        obs = to_tensor(batch['obs'])       # [B, H+1, O]
        action = to_tensor(batch['action']) # [B, H, A]
        reward = to_tensor(batch['reward']) # [B, H]
        done = to_tensor(batch['done'])     # [B, H]

        H = action.shape[1]
        B = action.shape[0]
        rho = getattr(self.config, 'rho', 0.5)

        # Geometric temporal weighting
        weights = torch.tensor(
            [rho ** t for t in range(H)], device=device
        )  # [H]

        # ---- World model update ----
        self._model_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            # Encode all observations
            obs_flat = obs.reshape(B * (H + 1), -1)
            z_all = self.world_model.encode(obs_flat).reshape(B, H + 1, -1)
            # z_all: [B, H+1, latent]
            z = z_all[:, :-1]      # current latents [B, H, latent]
            z_next = z_all[:, 1:]  # target next latents [B, H, latent]

            consistency_loss = torch.zeros(1, device=device)
            reward_loss = torch.zeros(1, device=device)
            value_loss = torch.zeros(1, device=device)
            consistency_t0 = None
            consistency_tH = None

            for t in range(H):
                z_t = z[:, t]         # [B, latent]
                a_t = action[:, t]    # [B, act_dim]
                r_t = reward[:, t]    # [B]
                d_t = done[:, t]      # [B]
                z_next_t = z_next[:, t].detach()   # [B, latent] — stop grad for target

                # Consistency: dynamics(z_t, a_t) ≈ encoder(o_{t+1})
                z_pred = self.world_model.next_z(z_t, a_t)
                step_consistency = (z_pred - z_next_t).pow(2).mean()
                consistency_loss += weights[t] * step_consistency
                if t == 0:
                    consistency_t0 = step_consistency.detach().item()
                if t == H - 1:
                    consistency_tH = step_consistency.detach().item()

                # Reward prediction
                r_logits = self.world_model.reward_logits(z_t, a_t)
                reward_loss += weights[t] * soft_ce(
                    r_logits, r_t.unsqueeze(-1), self.num_bins, self.vmin, self.vmax
                ).mean()

                # TD target (min of target Q-values on next step)
                with torch.no_grad():
                    _, pi_next, _ = self.world_model.pi(z_next_t)
                    q_target_list = self.world_model.q_values(
                        z_next_t, pi_next, target=True
                    )
                    q_target = torch.stack([
                        two_hot_inv(ql, self.num_bins, self.vmin, self.vmax).squeeze(-1)
                        for ql in q_target_list
                    ], dim=0).min(dim=0).values  # [B]

                    not_done = (1.0 - d_t.float())
                    td_target = r_t + self.discount * not_done * q_target  # [B]

                # Q-value regression
                for q_net_logits in self.world_model.q_values(z_t, a_t, target=False):
                    value_loss += weights[t] * soft_ce(
                        q_net_logits, td_target.unsqueeze(-1),
                        self.num_bins, self.vmin, self.vmax
                    ).mean()

            consistency_loss /= H
            reward_loss /= H
            value_loss /= (H * self.world_model.num_q)

            c_coef = getattr(self.config, 'consistency_coef', 20.0)
            r_coef = getattr(self.config, 'reward_coef', 0.1)
            v_coef = getattr(self.config, 'value_coef', 0.1)
            total_loss = c_coef * consistency_loss + r_coef * reward_loss + \
                         v_coef * value_loss

        # Z-stats from encoded batch (before backward to keep graph clean)
        with torch.no_grad():
            z_flat = z_all.reshape(-1, z_all.shape[-1]).float()
            z_std_dims = z_flat.std(dim=0)
            z_stats = {
                'latent/z_mean_norm': z_flat.norm(dim=-1).mean().item(),
                'latent/z_std_mean': z_std_dims.mean().item(),
                'latent/z_active_frac': (z_std_dims > 0.01).float().mean().item(),
            }

        self._scaler.scale(total_loss).backward()
        self._scaler.unscale_(self._model_opt)
        wm_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.world_model.parameters()), self._grad_clip
        )
        # Encoder-specific gradient norm
        enc_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.world_model.encoder_parameters() if p.grad is not None
        ) ** 0.5
        self._scaler.step(self._model_opt)
        self._scaler.update()

        # ---- Policy update ----
        self._pi_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            with torch.no_grad():
                # Recompute z without grad for policy update
                z_pg = self.world_model.encode(
                    obs[:, :-1].reshape(B * H, -1)
                ).reshape(B, H, -1).detach()

            pi_loss = torch.zeros(1, device=device)
            for t in range(H):
                z_t = z_pg[:, t]
                _, pi_t, log_pi_t = self.world_model.pi(z_t)
                # Detach Q networks to not propagate through them
                q_list = self.world_model.q_values(z_t, pi_t, target=False)
                q_vals = torch.stack([
                    two_hot_inv(ql, self.num_bins, self.vmin, self.vmax).squeeze(-1)
                    for ql in q_list
                ], dim=0)  # [num_q, B]
                q_mean = q_vals.mean(dim=0)  # [B]

                # Normalise Q-values for stable policy loss
                self._q_scale.update(q_mean)
                q_normed = self._q_scale(q_mean)

                ent_coef = getattr(self.config, 'entropy_coef', 1e-4)
                ent = -log_pi_t.squeeze(-1)  # [B]
                pi_loss += weights[t] * -(ent_coef * ent + q_normed).mean()

            pi_loss /= H

        self._scaler.scale(pi_loss).backward()
        self._scaler.unscale_(self._pi_opt)
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.world_model.policy_parameters()), self._grad_clip
        )
        self._scaler.step(self._pi_opt)
        self._scaler.update()

        # Soft-update target Q-networks
        tau = getattr(self.config, 'tau', 0.01)
        self.world_model.soft_update_target(tau)

        base_metrics = {
            'implicit/consistency_loss': consistency_loss.item(),
            'implicit/reward_loss': reward_loss.item(),
            'implicit/value_loss': value_loss.item(),
            'implicit/total_loss': total_loss.item(),
            'implicit/pi_loss': pi_loss.item(),
            'implicit/wm_grad_norm': float(wm_grad_norm),
            'implicit/pi_grad_norm': float(pi_grad_norm),
            'latent/encoder_grad_norm': enc_grad_norm,
            'latent/consistency_t0': consistency_t0 if consistency_t0 is not None else 0.0,
            'latent/consistency_tH': consistency_tH if consistency_tH is not None else 0.0,
        }
        return {**base_metrics, **z_stats, **self._last_plan_stats}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self):
        return {
            'world_model': self.world_model.state_dict(),
            'model_opt': self._model_opt.state_dict(),
            'pi_opt': self._pi_opt.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.world_model.load_state_dict(state['world_model'])
        self._model_opt.load_state_dict(state['model_opt'])
        self._pi_opt.load_state_dict(state['pi_opt'])

    # ------------------------------------------------------------------
    # Latent access for analysis
    # ------------------------------------------------------------------

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode a batch of observations to latent vectors (for analysis)."""
        device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            z = self.world_model.encode(obs_t)
        return z.cpu().numpy()
