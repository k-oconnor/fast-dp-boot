"""Exact Efron bootstrap kernels. Counts never leave the sampling boundary."""
from dataclasses import dataclass
from collections.abc import Callable
import numpy as np
from .privacy import Calibration, integer


@dataclass(frozen=True)
class Release:
    estimates: np.ndarray
    calibration: Calibration

    @property
    def noise_variance(self):
        return self.calibration.sigma ** 2


def _data(data):
    x = np.asarray(data, dtype=np.float64)
    if x.ndim < 1 or len(x) < 1 or not np.all(np.isfinite(x)):
        raise ValueError("data must contain finite rows")
    return x


def _budget(memory_bytes):
    return integer(memory_bytes, "memory_bytes", 64)


def _mean_draws(x, replicates, rng, memory_bytes):
    # Both indices and gathered float64 values occupy 8 bytes per draw.
    capacity = max(1, memory_bytes // 16)
    n = len(x)
    batch = max(1, min(replicates, capacity // n))
    tile = max(1, min(n, capacity // batch))
    out = np.empty(replicates)
    for start in range(0, replicates, batch):
        size = min(batch, replicates - start)
        sums = np.zeros(size)
        for offset in range(0, n, tile):
            count = min(tile, n - offset)
            indices = rng.integers(0, n, size=(size, count))
            sums += x[indices].sum(axis=1)
        out[start:start + size] = sums / n
    return out


def _mean_counts(x, replicates, rng, memory_bytes, auto=False):
    values, counts = np.unique(x, return_counts=True)
    if 16 * len(values) > memory_bytes or (auto and len(values) > max(32, len(x) // 8)):
        return _mean_draws(x, replicates, rng, memory_bytes)
    return _count_draws(values, counts, len(x), replicates, rng, memory_bytes)


def _count_draws(values, counts, n, replicates, rng, memory_bytes):
    probabilities = counts / n
    batch = max(1, min(replicates, memory_bytes // max(16 * len(values), 1)))
    out = np.empty(replicates)
    for start in range(0, replicates, batch):
        size = min(batch, replicates - start)
        weights = rng.multinomial(n, probabilities, size=size)
        out[start:start + size] = (weights @ values) / n
    return out


def bootstrap_mean(data, *, bounds, replicates, calibration, method="auto",
                   memory_bytes=32 * 1024**2, rng=None):
    """Clip to public bounds, exact bootstrap means, then Gaussian perturbation.

    Counts compress identical values losslessly; no binning, Poisson bootstrap,
    or normal approximation. All-unique data generally favor method='draws'.
    Budget covers resampling scratch arrays, excluding input/unique/output.
    Do not release timings or data-dependent support size as private outputs.
    """
    x = _data(data)
    if x.ndim != 1:
        raise ValueError("mean currently expects scalar rows")
    low, high = bounds
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("bounds must be finite and increasing")
    integer(replicates, "replicates", 2)
    _budget(memory_bytes)
    _check_calibration(calibration)
    _check_binding(calibration, len(x), replicates, (high - low) / len(x))
    x = np.clip(x, low, high)
    rng = np.random.default_rng(rng)
    if method in ("counts", "auto"):
        estimates = _mean_counts(x, replicates, rng, memory_bytes, auto=method == "auto")
    elif method == "draws":
        estimates = _mean_draws(x, replicates, rng, memory_bytes)
    else:
        raise ValueError("method must be 'auto', 'counts' or 'draws'")
    estimates += rng.normal(0, calibration.sigma, size=estimates.shape)
    return Release(estimates, calibration)


def bootstrap_mean_from_counts(values, counts, *, bounds, replicates, calibration,
                               memory_bytes=32 * 1024**2, rng=None):
    """Exact mean bootstrap from a private preaggregated frequency table.

    Avoids expanding records. n=sum(counts) must be public and fixed across
    neighbors. The frequency table itself must remain confidential.
    """
    values = _data(values)
    counts = np.asarray(counts)
    if values.ndim != 1 or counts.shape != values.shape or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("counts must be nonnegative integers matching scalar values")
    n = sum(int(c) for c in counts)
    if not 1 <= n <= np.iinfo(np.int64).max:
        raise ValueError("total count must be positive and fit int64")
    low, high = bounds
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError("bounds must be finite and increasing")
    integer(replicates, "replicates", 2)
    _budget(memory_bytes)
    if 16 * len(values) > memory_bytes:
        raise ValueError("frequency table needs at least 16*support_size scratch bytes")
    _check_calibration(calibration)
    _check_binding(calibration, n, replicates, (high-low)/n)
    rng = np.random.default_rng(rng)
    estimates = _count_draws(np.clip(values, low, high), counts, n, replicates, rng, memory_bytes)
    estimates += rng.normal(0, calibration.sigma, size=estimates.shape)
    return Release(estimates, calibration)


def _check_calibration(calibration):
    if not isinstance(calibration, Calibration) or not np.isfinite(calibration.sigma) or calibration.sigma <= 0:
        raise ValueError("provide a valid Calibration with positive finite sigma")


def _check_binding(calibration, n, replicates, sensitivity=None):
    if calibration.replicates != replicates or calibration.n not in (None, n):
        raise ValueError("calibration does not match n/replicates")
    if sensitivity is not None and calibration.sensitivity < sensitivity:
        raise ValueError("calibration sensitivity is smaller than the bounded mean sensitivity")


def bootstrap_statistic(data, statistic: Callable, *, replicates, calibration, rng=None):
    """Generic scalar/vector fallback; one resample in memory at a time.

    Caller must establish the statistic's global joint L2 sensitivity and
    calibrate for this exact n and B. Outputs must have fixed public shape.
    """
    x = _data(data)
    integer(replicates, "replicates", 2)
    _check_calibration(calibration)
    _check_binding(calibration, len(x), replicates)
    rng = np.random.default_rng(rng)
    out = None
    for j in range(replicates):
        value = np.asarray(statistic(x[rng.integers(0, len(x), len(x))]), dtype=float)
        if not np.isfinite(value).all():
            raise ValueError("statistic produced nonfinite values")
        if out is None:
            out = np.empty((replicates,) + value.shape)
        if value.shape != out.shape[1:]:
            raise ValueError("statistic must have fixed output shape")
        out[j] = value + rng.normal(0, calibration.sigma, size=value.shape)
    return Release(out, calibration)
