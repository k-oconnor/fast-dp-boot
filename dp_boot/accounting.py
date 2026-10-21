"""Proposition 14 Gaussian tradeoff accountant with linear FFT composition.

Upward loss discretization and discarded upper tails give a mathematical upper
bound. Ordinary floating-point evaluation still requires precision/refinement
checks; this is not a certified numerical accountant or the default calibration.
"""
from dataclasses import dataclass
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.special import ndtr
from scipy.stats import binom
from scipy.signal import lfilter
from .privacy import positive, integer


@dataclass(frozen=True)
class PrivacyProfile:
    """Numerical bounds for the compressed dominating tradeoff pair.

    delta_lower brackets its discretization error, not the privacy loss of an
    arbitrary input mechanism. Floating-point error is not certified.
    """
    epsilon: np.ndarray
    delta: np.ndarray
    loss_step: float
    rounding_budget: float
    discarded_tail_bound: float
    fft_points: int
    numerical_mass_error: float
    delta_lower: np.ndarray


def _hockey_stick(composed, offset, step, epsilon):
    """All nonnegative epsilon queries in O(lattice + queries), without overflow.

    Tail expectation is evaluated relative to each lattice point by a stable
    discounted cumulative sum; exp(epsilon) is never formed.
    """
    positive = composed[max(0, offset):]
    survival = np.cumsum(positive[::-1])[::-1]
    discounted = lfilter([1.], [1., -np.exp(-step)], positive[::-1])[::-1]
    indices = np.searchsorted(np.arange(len(positive))*step, epsilon, side="right")
    valid = indices < len(positive)
    result = np.zeros_like(epsilon, dtype=float)
    j = indices[valid]
    result[valid] = survival[j] - np.exp(epsilon[valid]-j*step)*discounted[j]
    return np.maximum(0., result)


def _groups(n, base_mu, group_tail):
    # Collapse tiny binomial tail to the worst group, n. Replacing G_i by G_n
    # weakens the tradeoff, so this compression is conservative, not Poisson.
    cap = min(n, max(1, int(binom.isf(group_tail, n, 1/n))))
    groups = np.arange(1, cap + 1, dtype=float)
    weights = binom.pmf(groups, n, 1/n)
    if cap < n:
        groups = np.r_[groups, float(n)]
        weights = np.r_[weights, binom.sf(cap, n, 1/n)]
    p0 = 0. if n == 1 else float(np.exp(n * np.log1p(-1/n)))
    weights /= weights.sum()
    return p0, groups * base_mu, weights


def _loss_cdf(loss, p0, scales, weights):
    loss = np.asarray(loss)
    # t = log(-C), obtained from q(x) in Proposition 14.
    z = np.abs(loss)
    t = (z + np.log1p(-p0 * np.exp(-z)) - np.log1p(-p0))
    u = t[:, None] / scales
    a_positive = ndtr(-u - scales / 2) @ weights
    a_negative = ndtr(u - scales / 2) @ weights
    cdf = np.where(loss < 0, a_positive,
                   p0 - p0 * a_positive + (1-p0) * a_negative)
    # At zero this chooses the right-continuous CDF, including the q=1 atom.
    return np.clip(cdf, 0., 1.)


def bootstrap_privacy_profile(n, base_mu, replicates, epsilons, *,
                              loss_step=.01, max_loss=8., group_tail=1e-14,
                              max_fft_points=8_000_000):
    """Numerical finite-composition profile of the paper's Gaussian bound.

    base_mu is Delta/sigma for ONE original Gaussian statistic, not aggregate
    mu. PLD is under Q, with L=log(q/p). Round L upward on a lattice, compose
    by one zero-padded FFT power, evaluate E_Q[(1-exp(epsilon-L))_+].
    Losses outside [-max_loss,max_loss] are discarded and both tails are
    charged by a union bound. No circular wraparound is permitted.

    The loss rounding slack is at most B*loss_step on epsilon. Run finer steps
    and wider loss ranges to assess numerical convergence. Check mass error;
    small deltas near floating-point error are not certified.
    """
    integer(n, "n")
    integer(replicates, "replicates")
    positive(base_mu, "base_mu")
    positive(loss_step, "loss_step")
    positive(max_loss, "max_loss")
    integer(max_fft_points, "max_fft_points")
    if not 0 < group_tail < .01:
        raise ValueError("group_tail must lie in (0,.01)")
    eps = np.atleast_1d(np.asarray(epsilons, dtype=float))
    if eps.ndim != 1 or not np.isfinite(eps).all() or np.any(eps < 0):
        raise ValueError("epsilons must be finite nonnegative scalars or a vector")
    j = int(np.ceil(max_loss / loss_step))
    width = 2*j + 1
    length = replicates * (width-1) + 1
    fft_points = next_fast_len(length)
    if fft_points > max_fft_points:
        raise ValueError(f"composition needs {fft_points} FFT points; increase loss_step "
                         "or max_fft_points")
    p0, scales, weights = _groups(n, base_mu, group_tail)
    lattice = np.arange(-j, j+1) * loss_step
    cdf = _loss_cdf(lattice, p0, scales, weights)
    pmf = np.maximum(0., np.diff(cdf, prepend=cdf[0]))
    # Stable Q upper tail, without subtracting a CDF near one.
    z = j * loss_step
    t = z + np.log1p(-p0*np.exp(-z)) - np.log1p(-p0)
    tail = p0 * (ndtr(-t/scales-scales/2) @ weights)
    tail += (1-p0) * (ndtr(-t/scales+scales/2) @ weights)
    tail_bound = min(1., replicates * float(tail + cdf[0]))
    composed = irfft(rfft(pmf, fft_points) ** replicates, fft_points)[:length]
    expected_mass = float(pmf.sum() ** replicates)
    error = float(abs(composed.sum()-expected_mass) + np.abs(composed[composed < 0]).sum())
    composed = np.maximum(composed, 0.)
    offset = replicates*j
    upper = np.minimum(1., _hockey_stick(composed, offset, loss_step, eps)+tail_bound)
    # Subtract B*h from each rounded-up composed loss to obtain a lower bound.
    # No tail charge: omitted contributions are nonnegative.
    lower = _hockey_stick(composed, offset, loss_step, eps+replicates*loss_step)
    return PrivacyProfile(eps, upper, loss_step,
                          replicates*loss_step, tail_bound, fft_points, error, lower)


def profile_calibration(n, sensitivity, replicates, epsilon, delta, *, loss_step=None,
                        max_loss=6., rtol=1e-3, max_fft_points=8_000_000):
    """Smallest sigma, to relative tolerance rtol, whose Proposition 14 upper
    profile gives delta(epsilon) <= delta.

    Same contract as bootstrap_privacy_profile: fixed public n, replace-one
    adjacency, valid global L2 sensitivity, ideal Gaussian noise. Bisection keeps
    the invariant that the returned sigma's upper profile meets delta, so the
    result is never looser than the accountant's bound; floating-point error is
    not certified. The default loss_step keeps the rounding slack B*loss_step at
    a tenth of epsilon, coarsened only as far as max_fft_points requires; finer
    steps (with a larger max_fft_points) give a tighter sigma.
    """
    from .privacy import Calibration, gaussian_delta
    from scipy.optimize import brentq
    integer(n, "n")
    integer(replicates, "replicates", 2)
    positive(sensitivity, "sensitivity")
    positive(epsilon, "epsilon")
    positive(rtol, "rtol")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie in (0,1)")
    if loss_step is None:
        fits = 2.02 * max_loss * replicates / max_fft_points
        loss_step = min(.01, max(epsilon / (10 * replicates), fits))

    def upper(sigma):
        profile = bootstrap_privacy_profile(n, sensitivity / sigma, replicates, [epsilon],
                                            loss_step=loss_step, max_loss=max_loss,
                                            max_fft_points=max_fft_points)
        return float(profile.delta[0])

    # Bracket from the paper's asymptotic rule at the GDP level matching (epsilon, delta).
    top = 1.
    while gaussian_delta(epsilon, top) < delta:
        top *= 2
    mu = brentq(lambda m: gaussian_delta(epsilon, m) - delta, 1e-12, top)
    hi = np.sqrt((2 - 2 / np.e) * replicates) * sensitivity / mu
    for _ in range(60):
        if upper(hi) <= delta:
            break
        hi *= 2
    else:
        raise ValueError("no sigma meets delta; widen max_loss or refine loss_step")
    lo = hi / 2
    for _ in range(60):
        if upper(lo) > delta:
            break
        hi, lo = lo, lo / 2
    while hi / lo - 1 > rtol:
        mid = np.sqrt(lo * hi)
        if upper(mid) <= delta:
            hi = mid
        else:
            lo = mid
    return Calibration(float(hi), "profile_upper_bound", float("nan"), sensitivity=sensitivity,
                       replicates=replicates, n=n, epsilon=float(epsilon),
                       profile_delta=upper(hi))
