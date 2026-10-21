"""Replace-one adjacency, public fixed n, scalar or joint L2 sensitivity."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import binom


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def gaussian_delta(epsilon, mu):
    """Exact GDP conversion, evaluated without exponential overflow."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    positive(mu, "mu")
    a = log_ndtr(-epsilon / mu + mu / 2)
    b = epsilon + log_ndtr(-epsilon / mu - mu / 2)
    return float(np.exp(a) * -np.expm1(min(0., b - a)))


@dataclass(frozen=True)
class Calibration:
    sigma: float
    method: str
    mu: float
    tail_delta: float = 0.
    multiplicity_cap: int | None = None
    sensitivity: float = 0.
    replicates: int = 0
    n: int | None = None

    def delta(self, epsilon):
        """For asymptotic calibration this is an approximation, not a finite bound."""
        return min(1., gaussian_delta(epsilon, self.mu) + self.tail_delta)


def asymptotic_calibration(sensitivity, replicates, mu):
    """Paper Algorithm 1; intentionally explicit opt-in, not finite GDP."""
    positive(sensitivity, "sensitivity")
    integer(replicates, "replicates", 2)
    positive(mu, "mu")
    return Calibration(math.sqrt((2 - 2 / math.e) * replicates) * sensitivity / mu,
                       "paper_asymptotic", mu, sensitivity=sensitivity, replicates=replicates)


def finite_calibration(n, sensitivity, replicates, epsilon, delta):
    """Conservative finite (epsilon,delta) bound via capped multiplicities.

    For the changed row K~Binomial(n,1/n). With probability >=1-B*Pr(K>k),
    all B resamples differ in at most k places. Conditional on this public-
    randomness event, Gaussian composition is k*Delta*sqrt(B)/sigma GDP.
    Charge the complementary event to delta. This is an alternative bound,
    not the paper's tighter Proposition 14 numerical accountant.
    """
    integer(n, "n")
    integer(replicates, "replicates", 2)
    positive(sensitivity, "sensitivity")
    positive(epsilon, "epsilon")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    # Binary search the smallest cap reserving at least half of delta for GDP.
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if replicates * binom.sf(mid, n, 1 / n) <= delta / 2:
            hi = mid
        else:
            lo = mid + 1
    k = lo
    tail = float(replicates * binom.sf(k, n, 1 / n))
    target = delta - tail
    upper = 1.
    while gaussian_delta(epsilon, upper) < target:
        upper *= 2
    mu = brentq(lambda m: gaussian_delta(epsilon, m) - target, 1e-12, upper)
    return Calibration(k * sensitivity * math.sqrt(replicates) / mu,
                       "finite_multiplicity_bound", mu, tail, k, sensitivity, replicates, n)
