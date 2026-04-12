"""
Implicit World Model (TD-MPC2 style) for the Implicit agent.

Architecture:
  encoder:   symlog(obs) → MLP → SimNorm → z  [256-dim, on probability simplex]
  dynamics:  concat(z, a) → MLP → SimNorm → z_next  [256-dim]
  reward:    concat(z, a) → MLP → 101-bin logits  [two-hot regression]
  Qs (×2):  concat(z, a) → MLP → 101-bin logits  [two-hot Q-value]
  pi:        z → MLP → (mean, log_std)            [Gaussian policy prior]

No observation decoder — the world model is purely task-centric.

Adapted from:
  tdmpc2/tdmpc2/common/world_model.py
  tdmpc2/tdmpc2/common/layers.py
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import SimNorm, build_mlp
from .math_utils import symlog, gaussian_logprob, squash, scale_log_std


class ImplicitWorldModel(nn.Module):
    """Implicit world model: encoder, dynamics, reward, Q-ensemble, policy prior.

    All latent vectors live on the probability simplex (enforced by SimNorm),
    which bounds their norm and stabilises training without gradient clipping
    of activations.

    Args:
        obs_dim: flat observation dimension.
        act_dim: flat action dimension.
        config: config namespace / dataclass with fields:
            latent_dim, mlp_dim, enc_dim, num_enc_layers, simnorm_dim,
            num_q, dropout, log_std_min, log_std_max.
    """

    def __init__(self, obs_dim: int, act_dim: int, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.latent_dim = config.latent_dim
        self.num_q = config.num_q
        self.simnorm_dim = config.simnorm_dim
        self._log_std_min = getattr(config, 'log_std_min', -10.0)
        self._log_std_max = getattr(config, 'log_std_max', 2.0)

        enc_dim = getattr(config, 'enc_dim', config.latent_dim)
        mlp_dim = config.mlp_dim
        latent = config.latent_dim
        act = act_dim
        dropout = getattr(config, 'dropout', 0.01)
        simnorm = SimNorm(config.simnorm_dim)

        # ------ Encoder: obs → z ------
        # obs goes through symlog normalisation inside forward(), then:
        # enc_dim hidden layers followed by SimNorm activation on latent_dim output
        enc_hidden = [enc_dim] * getattr(config, 'num_enc_layers', 2)
        self._encoder = build_mlp(obs_dim, enc_hidden, latent, act=SimNorm(config.simnorm_dim))

        # ------ Dynamics: (z, a) → z_next ------
        self._dynamics = build_mlp(latent + act, [mlp_dim, mlp_dim], latent,
                                   act=SimNorm(config.simnorm_dim))

        # ------ Reward head: (z, a) → 101-bin logits ------
        num_bins = getattr(config, 'num_bins', 101)
        self._reward = build_mlp(latent + act, [mlp_dim, mlp_dim], num_bins)

        # ------ Q-functions: (z, a) → 101-bin logits × num_q ------
        # Separate network instances (no vmap for simplicity; supports num_q=2)
        self._Qs = nn.ModuleList([
            build_mlp(latent + act, [mlp_dim, mlp_dim], num_bins, dropout=dropout)
            for _ in range(self.num_q)
        ])
        # Target Q-networks (Polyak-updated)
        self._target_Qs = copy.deepcopy(self._Qs)
        for p in self._target_Qs.parameters():
            p.requires_grad_(False)

        # ------ Policy prior: z → (mean, log_std) ------
        self._pi = build_mlp(latent, [mlp_dim, mlp_dim], 2 * act)

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation(s) to latent vector z.

        Applies symlog normalisation before the MLP.

        Args:
            obs: [..., obs_dim]

        Returns:
            z: [..., latent_dim] — on the probability simplex.
        """
        return self._encoder(symlog(obs))

    def next_z(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next latent state from current latent and action.

        Args:
            z: [..., latent_dim]
            a: [..., act_dim]

        Returns:
            z_next: [..., latent_dim]
        """
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward_logits(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict reward as 101-bin logits.

        Args:
            z: [..., latent_dim]
            a: [..., act_dim]

        Returns:
            logits: [..., num_bins]
        """
        return self._reward(torch.cat([z, a], dim=-1))

    def q_values(self, z: torch.Tensor, a: torch.Tensor,
                 target: bool = False) -> list:
        """Compute Q-value logits from all Q-networks.

        Args:
            z: [..., latent_dim]
            a: [..., act_dim]
            target: whether to use target networks.

        Returns:
            List of num_q tensors, each [..., num_bins].
        """
        nets = self._target_Qs if target else self._Qs
        za = torch.cat([z, a], dim=-1)
        return [q(za) for q in nets]

    def pi(self, z: torch.Tensor) -> tuple:
        """Sample an action from the policy prior.

        Args:
            z: [..., latent_dim]

        Returns:
            (mean_action, sampled_action, log_prob) all [..., act_dim] or [..., 1].
        """
        out = self._pi(z)
        mu = out[..., :self.act_dim]
        log_std_raw = out[..., self.act_dim:]
        log_std = scale_log_std(log_std_raw, self._log_std_min, self._log_std_max)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        log_pi = gaussian_logprob(eps, log_std)
        pi = mu + eps * std
        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi

    # ------------------------------------------------------------------
    # Target network update
    # ------------------------------------------------------------------

    def soft_update_target(self, tau: float = 0.01):
        """Polyak update of target Q-networks."""
        for online_q, target_q in zip(self._Qs, self._target_Qs):
            for p_online, p_target in zip(online_q.parameters(),
                                          target_q.parameters()):
                p_target.data.lerp_(p_online.data, tau)

    # ------------------------------------------------------------------
    # Convenience: parameters grouped for different learning rates
    # ------------------------------------------------------------------

    def encoder_parameters(self):
        return self._encoder.parameters()

    def non_encoder_parameters(self):
        """All parameters except the encoder (for full LR)."""
        encoder_ids = set(id(p) for p in self._encoder.parameters())
        return [p for p in self.parameters() if id(p) not in encoder_ids
                and p not in self._target_Qs.parameters()]

    def model_parameters(self):
        """World model parameters excluding policy prior and target Qs."""
        policy_ids = set(id(p) for p in self._pi.parameters())
        target_ids = set(id(p) for p in self._target_Qs.parameters())
        return [p for p in self.parameters()
                if id(p) not in target_ids]

    def policy_parameters(self):
        return self._pi.parameters()
