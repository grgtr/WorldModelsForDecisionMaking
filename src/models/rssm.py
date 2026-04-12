"""
Recurrent State-Space Model (RSSM) for the Explicit World Model agent.

Architecture:
  h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}))           [deterministic, 512-dim]
  z_t ~ Categorical(MLP(h_t, embed_t))                    [stochastic, 32*8=256-dim]
  z_t_prior ~ Categorical(MLP(h_t))                       [prior, 32*8=256-dim]

Full feature vector: [h_t, z_t_flat] = [512, 256] = 768-dim

Latent dim constraints (from spec):
  - GRU hidden (dyn_deter)
  - Stochastic latent (dyn_stoch * dyn_discrete)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd

from .networks import GRUCell, DreamerMLP, OneHotDist, DiscDist, SymlogDist
from .math_utils import symlog


class RSSM(nn.Module):
    """
    Recurrent State-Space Model.

    Args:
        stoch: number of categorical variables (default 32).
        discrete: bins per categorical variable (default 8).
        deter: GRU hidden dimension.
        hidden: MLP hidden width used in transition layers.
        num_actions: action dimension.
        embed_dim: dimension of the observation embedding from the encoder.
        unimix_ratio: Unimix fraction added to all categorical distributions.
        device: torch device string.
    """

    def __init__(
        self,
        stoch: int = 32,
        discrete: int = 8,
        deter: int = 512,
        hidden: int = 512,
        num_actions: int = None,
        embed_dim: int = None,
        unimix_ratio: float = 0.01,
        device: str = 'cuda'
    ):
        super().__init__()
        self.stoch = stoch
        self.discrete = discrete
        self.deter = deter
        self.hidden = hidden
        self.unimix = unimix_ratio
        self._device = device

        # Prior path: z_t_prior = f(h_t)
        # Input to GRU: concat(stoch_flat, action)
        self._img_in = nn.Sequential(
            nn.Linear(stoch * discrete + num_actions, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3),
            nn.SiLU(),
        )
        # Recurrent cell
        self._cell = GRUCell(hidden, deter, norm=True, update_bias=-1.0)
        # Mapping from h_t to prior stats
        self._img_out = nn.Sequential(
            nn.Linear(deter, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3),
            nn.SiLU(),
        )
        self._imgs_stat = nn.Linear(hidden, stoch * discrete)

        # Posterior path: z_t_post = f(h_t, embed_t)
        self._obs_out = nn.Sequential(
            nn.Linear(deter + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden, eps=1e-3),
            nn.SiLU(),
        )
        self._obs_stat = nn.Linear(hidden, stoch * discrete)

        # Initial state 
        # Learnable initial deter state (helps with first observation)
        self._initial_deter = nn.Parameter(torch.zeros(1, deter))
        nn.init.normal_(self._initial_deter, std=0.01)

    # State helpers

    def initial(self, batch_size: int) -> dict:
        """Return a zero (or learned) initial RSSM state."""
        deter = torch.tanh(self._initial_deter).expand(batch_size, -1)
        stoch_logit = torch.zeros(batch_size, self.stoch, self.discrete,
                                  device=deter.device)
        # Sample uniform initial stochastic state
        stoch = F.one_hot(torch.zeros(batch_size, self.stoch,
                                      dtype=torch.long, device=deter.device),
                          self.discrete).float()
        return {'deter': deter, 'stoch': stoch, 'logit': stoch_logit}

    def get_feat(self, state: dict) -> torch.Tensor:
        """Concatenate flattened stoch [B,..,256] and deter [B,..,512] to [B,..,768]."""
        stoch = state['stoch'].reshape(*state['stoch'].shape[:-2],
                                       self.stoch * self.discrete)
        return torch.cat([stoch, state['deter']], dim=-1)

    def get_dist(self, state: dict) -> torchd.Distribution:
        """Return the categorical distribution for z given `state['logit']`."""
        logit = state['logit']  # [..., stoch, discrete]
        dist = torchd.Independent(OneHotDist(logit, self.unimix), 1)
        return dist

    # Single-step transitions

    def obs_step(self, prev_state: dict, prev_action: torch.Tensor,
                 embed: torch.Tensor, is_first: torch.Tensor) -> tuple:
        """
        Posterior step (observation is available).

        Args:
            prev_state: RSSM state dict from previous step.
            prev_action: [B, act_dim]
            embed: [B, embed_dim] encoded observation.
            is_first: [B] boolean mask - True at episode start.

        Returns:
            (post, prior) - posterior and prior state dicts.
        """
        # Reset state at episode boundaries
        if is_first.any():
            mask = is_first.float().unsqueeze(-1)
            # Zero out recurrent state
            prev_state = {
                'deter': (1.0 - mask) * prev_state['deter'] +
                         mask * torch.tanh(self._initial_deter),
                'stoch': (1.0 - mask.unsqueeze(-1)) * prev_state['stoch'],
                'logit': (1.0 - mask.unsqueeze(-1)) * prev_state['logit'],
            }
            prev_action = (1.0 - mask) * prev_action

        # Prior: compute h_t from h_{t-1}, z_{t-1}, a_{t-1}
        prior = self.img_step(prev_state, prev_action)

        # Posterior: refine z_t using the observation embedding
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self._obs_out(x)
        logit = self._obs_stat(x).view(-1, self.stoch, self.discrete)
        stoch = OneHotDist(logit, self.unimix).sample()
        post = {'deter': prior['deter'], 'stoch': stoch, 'logit': logit}
        return post, prior

    def img_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """
        Prior step (imagination - no observation).

        Args:
            prev_state: RSSM state dict.
            prev_action: [B, act_dim]

        Returns:
            prior state dict: {deter, stoch, logit}
        """
        stoch_flat = prev_state['stoch'].reshape(
            *prev_state['stoch'].shape[:-2], self.stoch * self.discrete)
        inp = torch.cat([stoch_flat, prev_action], dim=-1)
        x = self._img_in(inp)
        deter = self._cell(x, prev_state['deter'])

        # Predict prior z
        h = self._img_out(deter)
        logit = self._imgs_stat(h).view(*deter.shape[:-1], self.stoch, self.discrete)
        stoch = OneHotDist(logit, self.unimix).sample()
        return {'deter': deter, 'stoch': stoch, 'logit': logit}

    # Full-sequence observe (forward pass over T steps)

    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor,
                state: dict = None) -> tuple:
        """
        Compute posterior and prior states for a full sequence.

        Args:
            embed: [B, T, embed_dim]
            action: [B, T, act_dim]
            is_first: [B, T] bool
            state: initial state dict (defaults to initial(B))

        Returns:
            (post, prior) each is a dict of [B, T, ...] tensors.
        """
        B, T, _ = embed.shape
        device = embed.device

        if state is None:
            state = {k: v.to(device) for k, v in self.initial(B).items()}

        posts, priors = [], []
        for t in range(T):
            post, prior = self.obs_step(
                state,
                action[:, t],
                embed[:, t],
                is_first[:, t],
            )
            posts.append(post)
            priors.append(prior)
            state = post

        # Stack into [B, T, ...] tensors
        def stack(lst_of_dicts):
            return {k: torch.stack([d[k] for d in lst_of_dicts], dim=1)
                    for k in lst_of_dicts[0]}

        return stack(posts), stack(priors)

    def kl_loss(self, post: dict, prior: dict,
                free: float = 1.0,
                dyn_scale: float = 0.5,
                rep_scale: float = 0.1) -> tuple:
        """
        Compute KL divergence losses between posterior and prior.

        Two-term KL loss from DreamerV3:
          dyn_loss = KL(sg(post) ‖ prior)  prior learns to predict posterior
          rep_loss = KL(post ‖ sg(prior))  encoder learns to match prior

        Both clipped at `free` nats (free bits) to avoid over-regularisation.

        Args:
            post, prior: state dicts with key 'logit' [B, T, stoch, discrete].
            free: free bits threshold (nats).
            dyn_scale, rep_scale: loss weights.

        Returns:
            (total_loss, kl_value, dyn_loss, rep_loss) all scalars.
        """
        kl_fn = torchd.kl_divergence

        post_logit = post['logit']
        prior_logit = prior['logit']

        # Build distributions (wrap in Independent to sum over stoch dim)
        post_dist = torchd.Independent(
            torchd.Categorical(logits=post_logit), 1)
        prior_dist = torchd.Independent(
            torchd.Categorical(logits=prior_logit), 1)

        # Detached versions for stop-gradient
        post_dist_sg = torchd.Independent(
            torchd.Categorical(logits=post_logit.detach()), 1)
        prior_dist_sg = torchd.Independent(
            torchd.Categorical(logits=prior_logit.detach()), 1)

        # Dynamics loss: prior → match posterior (stop grad through post)
        dyn_loss = torch.clamp(kl_fn(post_dist_sg, prior_dist), min=free)
        # Representation loss: posterior → match prior (stop grad through prior)
        rep_loss = torch.clamp(kl_fn(post_dist, prior_dist_sg), min=free)

        kl_value = kl_fn(post_dist, prior_dist).mean()
        total_loss = (dyn_scale * dyn_loss + rep_scale * rep_loss).mean()
        return total_loss, kl_value.item(), dyn_loss.mean().item(), rep_loss.mean().item()


# Encoder: maps raw state observations to RSSM embedding

class RSSMEncoder(nn.Module):
    """
    MLP encoder for proprioceptive (state) observations.

    Applies symlog to the raw obs before the MLP to handle large magnitudes.
    """

    def __init__(self, obs_dim: int, mlp_units: int = 512, mlp_layers: int = 3):
        super().__init__()
        layers = []
        in_d = obs_dim
        for _ in range(mlp_layers):
            layers += [nn.Linear(in_d, mlp_units, bias=False),
                       nn.LayerNorm(mlp_units, eps=1e-3),
                       nn.SiLU()]
            in_d = mlp_units
        self.net = nn.Sequential(*layers)
        self.outdim = mlp_units

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [B, T, obs_dim] or [B, obs_dim]
        Returns:
            embed: [..., mlp_units]
        """
        return self.net(symlog(obs))


# RSSMWorldModel: encoder + RSSM + prediction heads

class RSSMWorldModel(nn.Module):
    """
    Full RSSM world model: encoder + RSSM + reward / cont / decoder heads.

    Heads:
      decoder:  SymlogDist - reconstructs symlog(obs) from latent (MSE in symlog space)
      reward:   DiscDist   - 255-bin symlog-discrete reward prediction
      cont:     Bernoulli  - episode continuation (1 - terminal)

    """

    def __init__(self, obs_dim: int, act_dim: int, config):
        super().__init__()
        self.config = config
        device = getattr(config, 'device', 'cuda')

        self.encoder = RSSMEncoder(
            obs_dim,
            mlp_units=config.encoder_mlp_units,
            mlp_layers=config.encoder_mlp_layers,
        )
        self.dynamics = RSSM(
            stoch=config.dyn_stoch,
            discrete=config.dyn_discrete,
            deter=config.dyn_deter,
            hidden=config.dyn_hidden,
            num_actions=act_dim,
            embed_dim=self.encoder.outdim,
            unimix_ratio=config.unimix_ratio,
            device=device,
        )

        feat_dim = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        head_units = config.dyn_hidden

        self.heads = nn.ModuleDict({
            'decoder': DreamerMLP(
                feat_dim, obs_dim, layers=2, units=head_units,
                dist='symlog_mse', outscale=1.0,
            ),
            'reward': DreamerMLP(
                feat_dim, 255, layers=2, units=head_units,
                dist='symlog_disc', outscale=0.0,
            ),
            'cont': DreamerMLP(
                feat_dim, 1, layers=2, units=head_units,
                dist='binary', outscale=1.0,
            ),
        })

        # Single optimizer for the whole world model
        self._optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.model_lr,
            eps=config.opt_eps,
        )
        self._grad_clip = config.grad_clip
        self._use_amp = getattr(config, 'mixed_precision', True)
        self._scaler = torch.amp.GradScaler('cuda', enabled=self._use_amp)

    # Training step

    def train_step(self, data: dict) -> tuple:
        """
        World model training step on a batch of sequences.

        Args:
            data: dict with keys:
              obs:       [B, T, obs_dim]  float32
              action:    [B, T, act_dim]  float32
              reward:    [B, T]           float32
              done:      [B, T]           bool
              is_first:  [B, T]           bool

        Returns:
            (post, metrics) where post is the posterior state dict [B,T,...].
        """
        self._optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self._use_amp):
            # 1. Encode observations
            embed = self.encoder(data['obs'])          # [B,T,embed]

            # 2. Compute posterior + prior over sequence
            post, prior = self.dynamics.observe(
                embed, data['action'], data['is_first']
            )

            # 3. KL loss
            kl_loss, kl_val, dyn_loss, rep_loss = self.dynamics.kl_loss(
                post, prior,
                free=self.config.kl_free,
                dyn_scale=self.config.dyn_scale,
                rep_scale=self.config.rep_scale,
            )

            # 4. Head losses
            feat = self.dynamics.get_feat(post)        # [B,T,768]
            head_losses = {}
            head_losses['decoder'] = -self.heads['decoder'](feat).log_prob(
                data['obs']
            ).mean()
            head_losses['reward'] = -self.heads['reward'](feat).log_prob(
                data['reward'].unsqueeze(-1)
            ).mean()
            cont_target = (1.0 - data['done'].float()).unsqueeze(-1)
            head_losses['cont'] = -self.heads['cont'](feat).log_prob(
                cont_target
            ).mean()

            total_loss = kl_loss + sum(head_losses.values())

        # 5. Backward
        self._scaler.scale(total_loss).backward()
        self._scaler.unscale_(self._optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), self._grad_clip
        )
        self._scaler.step(self._optimizer)
        self._scaler.update()

        metrics = {
            'wm/kl_loss': kl_loss.item(),
            'wm/kl_value': kl_val,
            'wm/dyn_loss': dyn_loss,
            'wm/rep_loss': rep_loss,
            'wm/decoder_loss': head_losses['decoder'].item(),
            'wm/reward_loss': head_losses['reward'].item(),
            'wm/cont_loss': head_losses['cont'].item(),
            'wm/total_loss': total_loss.item(),
            'wm/grad_norm': float(grad_norm),
        }
        return post, metrics

    def state_dict_world_model(self):
        return super().state_dict()

    # Inference helpers (used during imagination rollouts)

    def reward_pred(self, feat: torch.Tensor) -> torch.Tensor:
        """Predict scalar reward from feature vector."""
        return self.heads['reward'](feat).mode()

    def cont_pred(self, feat: torch.Tensor) -> torch.Tensor:
        """Predict continuation probability from feature vector."""
        return self.heads['cont'](feat).mean
