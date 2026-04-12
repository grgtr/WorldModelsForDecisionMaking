"""
Explicit agent (RSSM): uses DreamerMLP (SiLU + LayerNorm) and GRUCell.
Implicit agent (TD-MPC2): uses build_mlp / NormedLinear (Mish + LayerNorm) and SimNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd
from .math_utils import symlog, symexp


# Implicit-agent style: SimNorm + NormedLinear + build_mlp
class SimNorm(nn.Module):
    """
    Simplicial Normalization - used as activation+norm in implicit agent.

    Splits the last dimension into groups of size `dim`, applies softmax
    within each group. This keeps latent representations on the probability
    simplex, preventing norm explosion without requiring explicit gradient
    clipping of activations.

    Adapted from https://arxiv.org/abs/2204.00616
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        return F.softmax(x, dim=-1).view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.

    Building block for the implicit agent's MLPs.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0, act: nn.Module = None):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act is not None else nn.Mish(inplace=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(self.ln(x))


def build_mlp(in_dim: int, hidden_dims: list, out_dim: int, act: nn.Module = None, dropout: float = 0.0) -> nn.Sequential:
    """
    Build MLP with LayerNorm, Mish activations, and optionally dropout.

    The final layer is a plain nn.Linear (no norm/activation) unless
    `act` is provided - in that case the final layer uses NormedLinear
    with that activation (SimNorm for dynamics/encoder outputs).

    Args:
        in_dim: input dimension.
        hidden_dims: list of hidden layer sizes.
        out_dim: output dimension.
        act: optional activation for final layer (SimNorm). If None,
             last layer is plain nn.Linear.
        dropout: dropout rate applied only to first hidden layer.
    """
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers = []
    for i in range(len(dims) - 2):
        d_in, d_out = dims[i], dims[i + 1]
        layer_dropout = dropout if i == 0 else 0.0
        layers.append(NormedLinear(d_in, d_out, dropout=layer_dropout))
    # Final layer
    if act is not None:
        layers.append(NormedLinear(dims[-2], dims[-1], act=act))
    else:
        layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# Explicit-agent style: DreamerMLP (SiLU + LayerNorm) + GRUCell

class DreamerMLP(nn.Module):
    """
    Multi-layer perceptron used by the explicit (DreamerV3) agent.

    Architecture: Linear + LayerNorm + SiLU (repeated `layers` times),
    then an output layer whose type depends on `dist`.

    Args:
        inp_dim: input dimension.
        shape: output shape tuple ((act_dim,) for actor, () for scalar,
               (255,) for DiscDist). Use None for a raw linear output.
        layers: number of hidden layers.
        units: hidden layer width.
        act: activation name ('SiLU', 'ReLU', 'ELU').
        norm: whether to apply LayerNorm after each linear.
        dist: output distribution type. Options:
              'mse', 'symlog_mse', 'symlog_disc', 'binary', 'normal',
              'trunc_normal', 'onehot', 'raw' (no distribution wrapper).
        symlog_inputs: apply symlog to inputs before forward pass.
        outscale: weight init scale for the last linear layer.
        min_std, max_std: for normal/trunc_normal distributions.
        absmax: for ContDist clipping.
    """

    def __init__(
        self,
        inp_dim: int,
        shape, layers: int,
        units: int,
        act: str = 'SiLU',
        norm: bool = True,
        dist: str = 'raw',
        symlog_inputs: bool = False,
        outscale: float = 1.0,
        min_std: float = 0.1,
        max_std: float = 1.0,
        absmax: float = None,
        unimix_ratio: float = 0.01,
        temp: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.symlog_inputs = symlog_inputs
        self.dist_type = dist
        self.shape = shape
        self.min_std = min_std
        self.max_std = max_std
        self.absmax = absmax
        self.unimix = unimix_ratio
        self.temp = temp

        act_fn = {'SiLU': nn.SiLU, 'ReLU': nn.ReLU, 'ELU': nn.ELU,
                  'Mish': nn.Mish}[act]

        hidden = []
        in_dim = inp_dim
        for _ in range(layers):
            hidden += [nn.Linear(in_dim, units, bias=False),
                       nn.LayerNorm(units, eps=1e-3),
                       act_fn()]
            in_dim = units
        self.hidden = nn.Sequential(*hidden)

        # Output layers
        if shape is None or dist == 'raw':
            out_dim = 1
            self.mean_layer = nn.Linear(units, out_dim)
        elif isinstance(shape, int):
            self.mean_layer = nn.Linear(units, shape)
        elif isinstance(shape, (tuple, list)):
            if len(shape) == 0:
                # Scalar output
                self.mean_layer = nn.Linear(units, 255 if dist == 'symlog_disc' else 1)
            else:
                self.mean_layer = nn.Linear(units, shape[-1])
        else:
            self.mean_layer = nn.Linear(units, shape)

        # Std head for normal distributions
        if dist in ('normal', 'trunc_normal'):
            self.std_layer = nn.Linear(units, self.mean_layer.out_features)
        else:
            self.std_layer = None

        # Weight init: small outscale for output layers
        nn.init.uniform_(self.mean_layer.weight, -outscale / units**0.5,
                         outscale / units**0.5)
        if self.mean_layer.bias is not None:
            nn.init.zeros_(self.mean_layer.bias)

    def forward(self, x: torch.Tensor):
        if self.symlog_inputs:
            x = symlog(x)
        x = self.hidden(x)
        mean = self.mean_layer(x)

        dist = self.dist_type
        if dist == 'raw':
            return mean
        elif dist == 'mse':
            return MSEDist(mean)
        elif dist == 'symlog_mse':
            return SymlogDist(mean, 'mse')
        elif dist == 'symlog_disc':
            return DiscDist(mean)
        elif dist == 'binary':
            return torchd.Independent(torchd.Bernoulli(logits=mean), len(mean.shape) - 1)
        elif dist == 'onehot':
            return OneHotDist(mean / self.temp, unimix_ratio=self.unimix)
        elif dist == 'normal':
            std = self.std_layer(x)
            std = (self.max_std - self.min_std) * torch.sigmoid(std) + self.min_std
            return ContDist(torchd.Normal(mean, std), self.absmax)
        elif dist == 'trunc_normal':
            std = self.std_layer(x)
            std = (self.max_std - self.min_std) * torch.sigmoid(std + 2.0) + self.min_std
            dist_obj = TruncNormalDist(torch.tanh(mean), std, -1.0, 1.0)
            return dist_obj
        else:
            raise ValueError(f"Unknown dist: {dist}")


class GRUCell(nn.Module):
    """
    GRU cell with optional LayerNorm, used in RSSM.

    Args:
        inp_size: input dimension.
        size: hidden state dimension.
        norm: apply LayerNorm to candidate state.
        update_bias: initialise update gate bias to this value (-1 helps keep
                     memory longer early in training).
    """

    def __init__(self, inp_size: int, size: int, norm: bool = True, update_bias: float = -1.0):
        super().__init__()
        self.size = size
        self.norm = norm
        self._inp_size = inp_size
        # Combined linear: input to [reset_gate, update_gate, candidate_gate]
        self.linear = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self.layer_norm = nn.LayerNorm(3 * size, eps=1e-3)
        if update_bias != 0:
            # Shift update gate bias so the GRU initially passes more state
            with torch.no_grad():
                self.linear.weight.data.zero_()
            self._update_bias = update_bias
        else:
            self._update_bias = None
        # Separate bias parameter for update gate
        self.bias = nn.Parameter(torch.zeros(3 * size))
        if update_bias != 0:
            with torch.no_grad():
                # Set update gate bias (second third) to update_bias
                self.bias.data[size:2 * size] = update_bias

    def forward(self, inputs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, inp_size]
            state: [B, size]

        Returns:
            new_state: [B, size]
        """
        combined = torch.cat([inputs, state], dim=-1)
        parts = self.linear(combined) + self.bias
        if self.norm:
            parts = self.layer_norm(parts)
        reset = torch.sigmoid(parts[..., :self.size])
        update = torch.sigmoid(parts[..., self.size:2 * self.size])
        candidate = torch.tanh(parts[..., 2 * self.size:] * reset)
        return update * candidate + (1.0 - update) * state


# Distribution classes (for explicit agent's heads and RSSM)

class OneHotDist(torchd.OneHotCategorical):
    """
    Straight-through OneHot Categorical with Unimix regularisation.

    Unimix: mixes the network distribution with a small uniform component
    to prevent any category from receiving exactly zero gradient - this
    avoids KL collapse in the discrete RSSM.

    Straight-through gradient: sample returns one-hot in forward, but
    gradients flow through the probabilities (allows BPTT through discrete z).

    """

    def __init__(self, logits: torch.Tensor, unimix_ratio: float = 0.01):
        if unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
        super().__init__(logits=logits)
        self._logits = logits

    def mode(self) -> torch.Tensor:
        _mode = F.one_hot(self._logits.argmax(dim=-1), self._logits.shape[-1]).float()
        return _mode.detach() + self._logits - self._logits.detach()

    def sample(self, sample_shape=()):
        sample = super().sample(sample_shape).detach()
        probs = self.probs
        while len(probs.shape) < len(sample.shape):
            probs = probs.unsqueeze(0)
        return sample + probs - probs.detach()


class DiscDist:
    """
    255-bin discrete regression distribution (DreamerV3 reward/value head).

    Converts scalar targets to symlog-spaced bins and treats the head output
    as a categorical. Allows gradients to flow back through soft bucket
    assignments.

    """

    def __init__(
        self,
        logits: torch.Tensor,
        low: float = -20.0,
        high: float = 20.0,
        transfwd=symlog,
        transbwd=symexp
    ):
        self.logits = logits
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self.num_bins = logits.shape[-1]
        self.bins = torch.linspace(low, high, self.num_bins,
                                   device=logits.device, dtype=logits.dtype)

    def mode(self) -> torch.Tensor:
        probs = F.softmax(self.logits, dim=-1)
        return self.transbwd((probs * self.bins).sum(-1, keepdim=True))

    def mean(self) -> torch.Tensor:
        return self.mode()

    # Inside OneHotCategorical, log_prob is calculated using only max element in targets
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute negative cross-entropy loss given target scalars x."""
        if x.dim() == self.logits.dim() - 1:
            x = x.unsqueeze(-1)
        x = self.transfwd(x).clamp(self.low, self.high)
        bin_size = (self.high - self.low) / (self.num_bins - 1)
        lower = ((x - self.low) / bin_size).floor().long().clamp(0, self.num_bins - 2)
        upper = lower + 1
        lower_frac = 1.0 - (x - self.bins[lower]) / bin_size
        upper_frac = 1.0 - lower_frac
        # Soft cross-entropy
        log_probs = F.log_softmax(self.logits, dim=-1)
        lp = lower_frac * log_probs.gather(-1, lower) + \
             upper_frac * log_probs.gather(-1, upper)
        return lp


class ContDist:
    """
    Wraps a torch distribution, adding .mode() and optional absmax clipping.
    """

    def __init__(self, dist: torchd.Distribution, absmax: float = None):
        self._dist = dist
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self) -> torch.Tensor:
        out = self._dist.mean
        if self.absmax is not None:
            out = torch.clamp(out, -self.absmax, self.absmax)
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out = torch.clamp(out, -self.absmax, self.absmax)
        return out

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x)


class SymlogDist:
    """
    MSE loss in symlog space - used for observation reconstruction head.
    """

    def __init__(self, pred: torch.Tensor, dist: str = 'mse',
                 agg: str = 'sum'):
        self._pred = pred
        self._dist = dist
        self._agg = agg

    def mode(self) -> torch.Tensor:
        return symexp(self._pred)

    def mean(self) -> torch.Tensor:
        return self.mode()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self._dist == 'mse':
            loss = -(symlog(x) - self._pred).pow(2)
        elif self._dist == 'abs':
            loss = -(symlog(x) - self._pred).abs()
        else:
            raise ValueError(self._dist)
        if self._agg == 'sum':
            return loss.sum(dim=-1, keepdim=True)
        return loss.mean(dim=-1, keepdim=True)


class MSEDist:
    """
    Raw MSE regression - simple wrapper returning negative L2 as log_prob.
    """

    def __init__(self, pred: torch.Tensor):
        self._pred = pred

    def mode(self) -> torch.Tensor:
        return self._pred

    def mean(self) -> torch.Tensor:
        return self._pred

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return -(x - self._pred).pow(2).sum(dim=-1, keepdim=True)


class TruncNormalDist:
    """
    Truncated Normal distribution on [-1, 1] for continuous actors.

    Wraps torch.distributions.Normal with clipped sampling and log_prob
    corrected for the truncation interval.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, low: float = -1.0, high: float = 1.0):
        self._mean = mean
        self._std = std
        self._low = low
        self._high = high
        self._dist = torchd.Normal(mean, std)

    def mode(self) -> torch.Tensor:
        return self._mean.clamp(self._low, self._high)

    def sample(self, sample_shape=()):
        s = self._dist.rsample(sample_shape)
        return s.clamp(self._low, self._high)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self._dist.log_prob(x).sum(dim=-1, keepdim=True)

    def entropy(self) -> torch.Tensor:
        return self._dist.entropy().sum(dim=-1, keepdim=True)
