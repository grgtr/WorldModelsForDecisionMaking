"""
Explicit World Model Agent (DreamerV3 style).

Learns a predictive generative model of the environment (RSSM) and trains
an actor-critic via backpropagation through imagined latent trajectories.

Components:
  1. RSSMWorldModel — encoder + RSSM dynamics + reward/cont/decoder heads
  2. ImagBehavior — actor + critic trained entirely inside the latent space

Adapted from:
  dreamerv3-torch/models.py:218-442 (ImagBehavior)
  dreamerv3-torch/dreamer.py:28-134 (Dreamer class)
"""

import torch
import torch.nn as nn
import torch.distributions as torchd
import numpy as np

from ..models.rssm import RSSMWorldModel
from ..models.networks import DreamerMLP, OneHotDist
from ..models.math_utils import lambda_return


# ---------------------------------------------------------------------------
# Reward EMA normaliser (stabilises actor loss scale)
# ---------------------------------------------------------------------------

class RewardEMA:
    """Exponential moving average of reward percentiles for actor loss scaling.

    Adapted from dreamerv3-torch/models.py:424-435
    """

    def __init__(self, alpha: float = 1e-2):
        self.alpha = alpha
        self._lo = None
        self._hi = None

    def __call__(self, x: torch.Tensor) -> tuple:
        lo = torch.quantile(x.detach(), 0.05).item()
        hi = torch.quantile(x.detach(), 0.95).item()
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            self._lo = (1 - self.alpha) * self._lo + self.alpha * lo
            self._hi = (1 - self.alpha) * self._hi + self.alpha * hi
        scale = max(self._hi - self._lo, 1.0)
        return (x - self._lo) / scale, scale


# ---------------------------------------------------------------------------
# Imagination-based Actor-Critic
# ---------------------------------------------------------------------------

class ImagBehavior(nn.Module):
    """Actor-Critic trained on imagined latent trajectories.

    The actor selects actions in latent space (no environment interaction),
    and gradients flow back through the world model's straight-through discrete
    z samples to update the actor's parameters (dynamics-backprop mode).

    Adapted from dreamerv3-torch/models.py:218-442
    """

    def __init__(self, feat_dim: int, act_dim: int, act_continuous: bool,
                 config):
        super().__init__()
        self.config = config
        self.act_continuous = act_continuous

        units = getattr(config, 'dyn_hidden', 256)

        # Actor
        if act_continuous:
            self.actor = DreamerMLP(
                feat_dim, act_dim, layers=3, units=units,
                dist='trunc_normal', outscale=1.0,
                min_std=config.actor_min_std if hasattr(config, 'actor_min_std') else 0.1,
                max_std=config.actor_max_std if hasattr(config, 'actor_max_std') else 1.0,
            )
        else:
            self.actor = DreamerMLP(
                feat_dim, act_dim, layers=3, units=units,
                dist='onehot', outscale=1.0,
            )

        # Critic: 255-bin symlog-discrete value distribution
        self.critic = DreamerMLP(
            feat_dim, 255, layers=3, units=units,
            dist='symlog_disc', outscale=0.0,
        )
        # Slow (EMA) target critic
        self.slow_critic = DreamerMLP(
            feat_dim, 255, layers=3, units=units,
            dist='symlog_disc', outscale=0.0,
        )
        # Initialise slow critic identical to critic
        for p_slow, p_fast in zip(self.slow_critic.parameters(),
                                  self.critic.parameters()):
            p_slow.data.copy_(p_fast.data)
            p_slow.requires_grad_(False)

        self._actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr, eps=1e-5
        )
        self._critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr, eps=1e-5
        )
        self._actor_grad_clip = config.grad_clip
        self._critic_grad_clip = config.grad_clip

        self._reward_ema = RewardEMA()
        self._use_amp = getattr(config, 'mixed_precision', True)
        self._scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

    # ------------------------------------------------------------------
    # Slow critic update (EMA)
    # ------------------------------------------------------------------

    def update_slow_critic(self, fraction: float = 0.02):
        for p_slow, p_fast in zip(self.slow_critic.parameters(),
                                  self.critic.parameters()):
            p_slow.data.lerp_(p_fast.data, fraction)

    # ------------------------------------------------------------------
    # Imagination rollout
    # ------------------------------------------------------------------

    def _imagine(self, start_state: dict, dynamics, horizon: int,
                 act_dim: int) -> tuple:
        """Roll out actor policy for `horizon` steps in latent space.

        Args:
            start_state: RSSM state dict [B, T, ...] — reshaped to [B*T, ...]
            dynamics: the RSSM module
            horizon: number of imagination steps
            act_dim: action dimension

        Returns:
            (feats, states, actions) tensors of shape [horizon, B*T, ...]
        """
        # Flatten batch × time dimensions
        flatten = lambda d: {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}
        state = flatten(start_state)
        BT = state['deter'].shape[0]
        device = state['deter'].device

        feats, states, actions = [], [], []
        for _ in range(horizon):
            feat = dynamics.get_feat(state)
            action_dist = self.actor(feat)
            if self.act_continuous:
                action = action_dist.sample()
            else:
                action = action_dist.sample()  # straight-through one-hot

            next_state = dynamics.img_step(state, action)
            feats.append(feat)
            states.append(state)
            actions.append(action)
            state = next_state

        feats = torch.stack(feats, dim=0)       # [H, BT, feat]
        actions = torch.stack(actions, dim=0)   # [H, BT, act]
        return feats, states, actions

    # ------------------------------------------------------------------
    # Lambda return computation
    # ------------------------------------------------------------------

    def _compute_targets(self, feats: torch.Tensor,
                         states: list, world_model) -> tuple:
        """Compute λ-return targets for actor-critic training.

        Args:
            feats: [H, BT, feat_dim]
            states: list of H state dicts
            world_model: RSSMWorldModel

        Returns:
            (targets, weights, baseline) each [H, BT, 1]
        """
        H, BT, _ = feats.shape
        device = feats.device

        # Predict rewards and continuation along imagination
        rewards = world_model.reward_pred(feats)  # [H, BT, 1]
        conts = world_model.cont_pred(feats)       # [H, BT, 1] probabilities

        # Critic values from slow target (more stable)
        with torch.no_grad():
            values = self.slow_critic(feats).mode()  # [H, BT, 1]

        discount = self.config.discount
        lambda_ = self.config.discount_lambda

        # Bootstrap from final state value
        bootstrap = values[-1]  # [BT, 1]

        # pcont = discount × continuation probability
        pcont = discount * conts  # [H, BT, 1]

        targets = lambda_return(
            rewards.squeeze(-1),
            values.squeeze(-1),
            pcont.squeeze(-1),
            bootstrap.squeeze(-1),
            lambda_,
        ).unsqueeze(-1)  # [H, BT, 1]

        # Cumulative discount weights for value loss
        weights = torch.cumprod(
            torch.cat([torch.ones_like(pcont[:1]), pcont[:-1]], dim=0),
            dim=0
        ).detach()

        return targets, weights, values

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, post_state: dict, world_model,
                   act_dim: int) -> dict:
        """Train actor and critic on imagined rollouts.

        Args:
            post_state: RSSM posterior state dict [B, T, ...].
            world_model: RSSMWorldModel instance.
            act_dim: action space dimension.

        Returns:
            metrics dict.
        """
        horizon = self.config.imag_horizon
        discount = self.config.discount

        # ---- Actor update ----
        self._actor_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            feats, states, actions = self._imagine(
                post_state, world_model.dynamics, horizon, act_dim
            )
            targets, weights, baseline = self._compute_targets(
                feats, states, world_model
            )

            # Advantage normalised by reward EMA
            advantage = (targets - baseline).detach()
            advantage, scale = self._reward_ema(advantage)

            # Actor loss: maximise normalised advantage
            action_dist = self.actor(feats[:-1].detach()
                                     if self.act_continuous else feats[:-1])
            log_prob = action_dist.log_prob(actions[:-1].detach()
                                            if not self.act_continuous else actions[:-1])

            # Dynamics gradient mode: use straight-through advantage
            actor_loss = -(weights[:-1] * advantage[:-1]).mean()

            # Entropy bonus
            if hasattr(action_dist, 'entropy'):
                try:
                    ent = action_dist.entropy().mean()
                    actor_loss = actor_loss - self.config.actor_entropy * ent
                except Exception:
                    pass

        self._scaler.scale(actor_loss).backward()
        self._scaler.unscale_(self._actor_opt)
        actor_norm = torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(), self._actor_grad_clip
        )
        self._scaler.step(self._actor_opt)
        self._scaler.update()

        # ---- Critic update ----
        self._critic_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=self._use_amp):
            # Critic predicts value at each imagined state
            critic_dist = self.critic(feats[:-1].detach())
            critic_loss = -(weights[:-1] * critic_dist.log_prob(
                targets[:-1].detach()
            )).mean()

        self._scaler.scale(critic_loss).backward()
        self._scaler.unscale_(self._critic_opt)
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self._critic_grad_clip
        )
        self._scaler.step(self._critic_opt)
        self._scaler.update()

        # Update slow target
        self.update_slow_critic(self.config.slow_target_fraction)

        return {
            'behavior/actor_loss': actor_loss.item(),
            'behavior/critic_loss': critic_loss.item(),
            'behavior/actor_grad_norm': float(actor_norm),
            'behavior/critic_grad_norm': float(critic_norm),
            'behavior/mean_target': targets.mean().item(),
            'behavior/mean_reward_scale': scale,
        }


# ---------------------------------------------------------------------------
# ExplicitAgent — the top-level agent class
# ---------------------------------------------------------------------------

class ExplicitAgent(nn.Module):
    """DreamerV3-style explicit world model agent.

    Owns a RSSMWorldModel and ImagBehavior. The world model is trained on
    real experience; the actor-critic is trained entirely in imagination.

    Args:
        obs_dim: flat observation dimension.
        act_dim: flat action dimension.
        act_continuous: True for continuous, False for discrete.
        config: config namespace with all hyperparameters.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 act_continuous: bool, config):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_continuous = act_continuous
        self.config = config
        self._device = getattr(config, 'device', 'cuda')

        feat_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter

        self.world_model = RSSMWorldModel(obs_dim, act_dim, config)
        self.behavior = ImagBehavior(feat_dim, act_dim, act_continuous, config)

        # Running latent state (for sequential inference during rollouts)
        self._state = None
        self._prev_action = None

    # ------------------------------------------------------------------
    # Policy inference
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, reset: bool = False,
            training: bool = True) -> np.ndarray:
        """Select an action given the current observation.

        Args:
            obs: [obs_dim] numpy array.
            reset: True at the start of a new episode (resets latent state).
            training: True during training (stochastic), False for evaluation.

        Returns:
            action: [act_dim] numpy array in [-1, 1] (continuous) or int (discrete).
        """
        device = next(self.world_model.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        is_first = torch.tensor([reset], dtype=torch.bool, device=device)

        with torch.no_grad():
            if self._state is None or reset:
                self._state = {k: v.to(device)
                               for k, v in self.world_model.dynamics.initial(1).items()}
                self._prev_action = torch.zeros(
                    1, self.act_dim, device=device, dtype=torch.float32
                )

            embed = self.world_model.encoder(obs_t)
            self._state, _ = self.world_model.dynamics.obs_step(
                self._state, self._prev_action, embed, is_first
            )
            feat = self.world_model.dynamics.get_feat(self._state)
            action_dist = self.behavior.actor(feat)
            if training:
                action = action_dist.sample()
            else:
                action = action_dist.mode()

            self._prev_action = action

        return action.squeeze(0).cpu().numpy()

    def reset_state(self):
        """Reset the recurrent state (call at episode start when using act())."""
        self._state = None
        self._prev_action = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch: dict) -> dict:
        """One training update on a batch of sequences.

        Args:
            batch: dict with numpy arrays:
              obs [B,T,O], action [B,T,A], reward [B,T],
              done [B,T], is_first [B,T]

        Returns:
            metrics dict.
        """
        device = next(self.world_model.parameters()).device

        # Move batch to device
        def to_tensor(x, dtype=torch.float32):
            return torch.tensor(x, dtype=dtype, device=device)

        data = {
            'obs':      to_tensor(batch['obs']),
            'action':   to_tensor(batch['action']),
            'reward':   to_tensor(batch['reward']),
            'done':     to_tensor(batch['done'], torch.bool),
            'is_first': to_tensor(batch['is_first'], torch.bool),
        }

        # World model train step
        post, wm_metrics = self.world_model.train_step(data)

        # Actor-critic train step in imagination
        behavior_metrics = self.behavior.train_step(
            post, self.world_model, self.act_dim
        )

        return {**wm_metrics, **behavior_metrics}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self):
        return {
            'world_model': self.world_model.state_dict(),
            'behavior': self.behavior.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.world_model.load_state_dict(state['world_model'])
        self.behavior.load_state_dict(state['behavior'])

    # ------------------------------------------------------------------
    # Latent access for analysis
    # ------------------------------------------------------------------

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """Encode a batch of observations to latent vectors (for analysis)."""
        device = next(self.parameters()).device
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            # Return just the embedding (not full RSSM state)
        return embed.cpu().numpy()
