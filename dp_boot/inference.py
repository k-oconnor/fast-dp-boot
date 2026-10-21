"""Post-processing of already private releases only."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import softmax
from scipy.stats import chi2, norm
from .privacy import integer


def _alpha(alpha):
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")


def asymptotic_interval(release, alpha=.05, omega=None):
    """Paper Algorithm 2, coordinatewise for vector releases (not simultaneous)."""
    _alpha(alpha)
    omega = alpha / 2 if omega is None else omega
    if not 0 < omega < alpha:
        raise ValueError("omega must lie in (0,alpha)")
    y = np.asarray(release.estimates)
    b = len(y)
    if b < 2 or not np.isfinite(y).all():
        raise ValueError("need at least two finite estimates")
    center = y.mean(axis=0)
    variance = y.var(axis=0, ddof=1)
    signal = np.maximum(0., (b - 1) * variance / chi2.ppf(alpha - omega, b - 1)
                        - release.noise_variance)
    upper = signal + (signal + release.noise_variance) / b
    radius = norm.ppf(1 - omega / 2) * np.sqrt(upper)
    return np.stack((center - radius, center + radius), axis=-1)


@dataclass(frozen=True)
class Distribution:
    grid: np.ndarray
    mass: np.ndarray
    converged: bool
    iterations: int
    objective: float

    def quantile(self, probabilities):
        p = np.asarray(probabilities)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("probabilities must lie in [0,1]")
        return np.interp(p, np.cumsum(self.mass), self.grid)

    def interval(self, alpha=.05):
        _alpha(alpha)
        return self.quantile([alpha / 2, 1 - alpha / 2])


def _objective(coefficients, basis, kernel, observed, penalty):
    mass = softmax(basis @ coefficients)
    predicted = np.maximum(fftconvolve(mass, kernel, mode="same"), 1e-300)
    norm_c = np.sqrt(coefficients @ coefficients + 1e-12)
    loss = -observed @ np.log(predicted) + penalty * norm_c
    gradient_mass = -fftconvolve(observed / predicted, kernel[::-1], mode="same")
    gradient = basis.T @ (mass * (gradient_mass - gradient_mass @ mass))
    return loss, gradient + penalty * coefficients / norm_c


def deconvolve_fft(release, *, grid_size=1025, spline_knots=12, penalty=1., maxiter=500):
    """Natural-spline exponential-family deconvolution with FFT likelihood.

    Follows the paper's deconvolveR model family, but is a native binned
    implementation, not a bit-for-bit port of R's basis/grid/optimizer.
    Grid and histogram use only the private estimates. Percentile intervals
    are experimental: numerical convergence does not certify coverage.
    """
    y = np.asarray(release.estimates, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("deconvolution requires a finite scalar release")
    integer(grid_size, "grid_size", 33)
    integer(spline_knots, "spline_knots", 4)
    integer(maxiter, "maxiter")
    if grid_size % 2 != 1 or spline_knots >= grid_size:
        raise ValueError("use an odd grid_size larger than spline_knots")
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("penalty must be positive")
    sigma = release.calibration.sigma
    # Padding suppresses circular/boundary artifacts; FFT convolution is linear.
    spread = max(float(y.std()), sigma, 1e-12)
    center = float(y.mean())
    left = min(float(y.min()), center - 6 * spread) - 4 * sigma
    right = max(float(y.max()), center + 6 * spread) + 4 * sigma
    grid = np.linspace(left, right, grid_size)
    dx = grid[1] - grid[0]
    observed, _ = np.histogram(y, bins=np.r_[grid - dx / 2, grid[-1] + dx / 2])
    offsets = (np.arange(grid_size) - grid_size // 2) * dx
    kernel = np.exp(-.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    knots = np.linspace(-1, 1, spline_knots)
    basis = CubicSpline(knots, np.eye(spline_knots), bc_type="natural")(
        np.linspace(-1, 1, grid_size))
    # Remove the redundant constant softmax direction.
    basis = basis[:, :-1] - basis[:, -1:]
    result = minimize(_objective, np.zeros(spline_knots - 1),
                      args=(basis, kernel, observed, penalty), jac=True,
                      method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-11})
    return Distribution(grid, softmax(basis @ result.x), bool(result.success),
                        result.nit, float(result.fun))
