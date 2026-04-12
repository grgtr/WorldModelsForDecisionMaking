import torch
import torch.nn.functional as F

def symlog(x: torch.Tensor) -> torch.Tensor:
    """f(x) = sign(x) * ln(|x| + 1).  Compresses large magnitudes, identity near 0."""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: f(x) = sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

def two_hot(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Converts a batch of scalars scalar to soft two-hot encoded targets for discrete regression.

    Args:
        x: scalar targets.
        num_bins: number of discrete bins.
        vmin, vmax: value range in symlog space.

    Returns:
        Soft two-hot tensor of shape [..., num_bins].
    """
    x = symlog(x).clamp(vmin, vmax)
    bin_size = (vmax - vmin) / (num_bins - 1)
    bin_idx = ((x - vmin) / bin_size).floor().long().clamp(0, num_bins - 2)
    bin_offset = ((x - vmin) / bin_size - bin_idx.float()).unsqueeze(-1)
    soft = torch.zeros(*x.shape, num_bins, device=x.device, dtype=x.dtype)
    soft.scatter_(-1, bin_idx.unsqueeze(-1), 1.0 - bin_offset)
    soft.scatter_(-1, (bin_idx + 1).unsqueeze(-1), bin_offset)
    return soft


def two_hot_inv(logits: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Converts a batch of soft two-hot encoded vectors to scalars.

    Args:
        logits: shape [..., num_bins].

    Returns:
        Scalar estimates of shape [..., 1].
    """
    bins = torch.linspace(vmin, vmax, num_bins, device=logits.device, dtype=logits.dtype)
    val_symlog = (logits.softmax(-1) * bins).sum(-1, keepdim=True)
    return symexp(val_symlog)


def soft_ce(pred_logits: torch.Tensor, target: torch.Tensor,
            num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Computes the cross entropy loss between predictions and soft targets.

    Args:
        pred_logits: [..., num_bins] raw logits.
        target: [...] or [..., 1] scalar targets.

    Returns:
        Per-sample loss of shape [..., 1].
    """
    if target.dim() == pred_logits.dim() - 1:
        target = target.unsqueeze(-1)
    target_th = two_hot(target.squeeze(-1), num_bins, vmin, vmax)
    log_pred = F.log_softmax(pred_logits, dim=-1)
    return -(target_th * log_pred).sum(-1, keepdim=True)

def lambda_return(reward: torch.Tensor,
                  value: torch.Tensor,
                  pcont: torch.Tensor,
                  bootstrap: torch.Tensor,
                  lambda_: float) -> torch.Tensor:
    """Compute lambda-returns for a sequence of imagined transitions.

    Uses a reverse-scan to compute:
        G_t = r_t + pcont_t * [(1-lambda)*V(s_{t+1}) + lambda*G_{t+1}]

    Args:
        reward: [H, B] rewards along imagined trajectory.
        value: [H, B] value estimates (from critic).
        pcont: [H, B] continuation probabilities (1 - terminal).
        bootstrap: [B] bootstrap value at the end of the horizon.
        lambda_: TD-lambda mixing parameter.

    Returns:
        targets: [H, B] lambda-return targets for the actor-critic.
    """
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    # next_values combines value[1:] with bootstrap at the last step
    next_values = torch.cat([value[1:], bootstrap.unsqueeze(0)], dim=0)
    # inputs = r_t + pcont_t * (1-lambda) * V_{t+1}
    inputs = reward + pcont * next_values * (1.0 - lambda_)
    returns = []
    last = bootstrap
    for t in reversed(range(len(inputs))):
        last = inputs[t] + pcont[t] * lambda_ * last
        returns.append(last)
    returns.reverse()
    return torch.stack(returns, dim=0)


# ---------------------------------------------------------------------------
# Policy utilities for implicit agent (Gaussian policy with tanh squashing)
# Adapted from tdmpc2/tdmpc2/common/math.py
# ---------------------------------------------------------------------------

def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute Gaussian log probability.
    Args:
        eps: noise samples, shape [..., act_dim].
        log_std: log standard deviation, shape [..., act_dim].

    Returns:
        Log probability summed over action dimensions, shape [..., 1].
    """
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956  # log(sqrt(2π))
    return log_prob.sum(-1, keepdim=True)


def squash(mu: torch.Tensor, pi: torch.Tensor,
           log_pi: torch.Tensor) -> tuple:
    """Apply tanh squashing to bounded action space with Jacobian correction.

    Args:
        mu: deterministic action (mean), shape [..., act_dim].
        pi: stochastic action sample, shape [..., act_dim].
        log_pi: log probability (pre-squash), shape [..., 1].

    Returns:
        (mu_squashed, pi_squashed, log_pi_corrected)
    """
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


def scale_log_std(x: torch.Tensor, log_std_min: float = -10.0,
                  log_std_max: float = 2.0) -> torch.Tensor:
    """Map raw network output to bounded log std range via tanh."""
    low = log_std_min
    dif = log_std_max - log_std_min
    return low + 0.5 * dif * (torch.tanh(x) + 1.0)


# Running percentile scale for Q-value normalization (implicit agent)

class RunningScale:
    """Tracks the 5th-95th percentile range of Q-values for normalization.

    Prevents the policy loss from being dominated by Q-value magnitude.
    """

    def __init__(self, tau: float = 0.01):
        self.tau = tau
        self._lo = None
        self._hi = None

    def update(self, x: torch.Tensor):
        lo = torch.quantile(x.detach().float(), 0.05).item()
        hi = torch.quantile(x.detach().float(), 0.95).item()
        if self._lo is None:
            self._lo, self._hi = lo, hi
        else:
            self._lo = (1 - self.tau) * self._lo + self.tau * lo
            self._hi = (1 - self.tau) * self._hi + self.tau * hi

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._lo is None:
            return x
        scale = max(self._hi - self._lo, 1.0)
        return x / scale

    @property
    def value(self):
        if self._lo is None:
            return (0.0, 1.0)
        return (self._lo, self._hi)
