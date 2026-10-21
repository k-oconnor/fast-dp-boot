"""Normal deconvolution matching the authors' deconvolveR model and settings.

Independent NumPy/SciPy implementation of natural-spline exponential-family
likelihood and low-rank uncertainty propagation; golden tests use R outputs.
"""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import qr
from scipy.optimize import minimize
from scipy.special import ndtr, softmax
from .inference import Distribution, _alpha
from .privacy import integer, positive


def _natural_basis(grid, degrees):
    """Canonical no-intercept natural cubic spline basis, centered and scaled.

    Uniform quantile knots, second-derivative constraints at the boundaries,
    and a full QR nullspace give the same coordinates as splines::ns.
    Matching coordinates matters because the fitted coefficient norm is penalized.
    """
    internal = np.quantile(grid, np.arange(1, degrees) / degrees)
    knots = np.r_[np.repeat(grid[0], 4), internal, np.repeat(grid[-1], 4)]
    spline = BSpline(knots, np.eye(len(knots)-4), 3)
    unconstrained = spline(grid)[:, 1:]
    constraints = spline.derivative(2)([grid[0], grid[-1]])[:, 1:]
    rotation = qr(constraints.T, mode="full")[0][:, 2:]
    basis = unconstrained @ rotation
    basis -= basis.mean(axis=0)
    basis /= np.sqrt((basis*basis).sum(axis=0))
    return basis


def _model(estimates, sigma, grid_size, spline_degree, observation_edges):
    z = estimates / sigma
    q1, q3 = np.quantile(z, [.25, .75])
    if q3 <= q1:
        raise ValueError("private estimates need positive interquartile range")
    grid = np.linspace(q1-3*(q3-q1), q3+3*(q3-q1), grid_size)
    edges = np.linspace(*np.round([z.min(), z.max()], 1), observation_edges)
    if edges[-1] <= edges[0]:
        raise ValueError("rounded observation range is degenerate")
    # R findInterval: [edge_i, edge_(i+1)); observations outside are omitted.
    indices = np.searchsorted(edges, z, side="right") - 1
    valid = (indices >= 0) & (indices < observation_edges-1)
    observed = np.bincount(indices[valid], minlength=observation_edges-1)
    if observed.sum() == 0:
        raise ValueError("no private estimates fall in the reference observation bins")
    lower = edges[:-1, None] - grid
    upper = edges[1:, None] - grid
    # Integrated Normal probabilities; use survival functions in the upper tail.
    likelihood = np.where(lower > 0, ndtr(-lower)-ndtr(-upper), ndtr(upper)-ndtr(lower))
    return grid, _natural_basis(grid, spline_degree), likelihood, observed


def _evaluate(coefficients, basis, likelihood, observed, penalty, hessian=False):
    mass = softmax(basis @ coefficients)
    predicted = np.maximum(likelihood @ mass, np.finfo(float).tiny)
    centered = basis - mass @ basis
    jacobian = (likelihood * mass) @ centered
    norm = max(float(np.linalg.norm(coefficients)), np.finfo(float).tiny)
    weighted = likelihood.T @ (observed/predicted)
    value = float(-observed @ np.log(predicted) + penalty*norm)
    gradient = -(jacobian.T @ (observed/predicted)) + penalty*coefficients/norm
    if not hessian:
        return value, gradient
    curvature = (jacobian.T * (observed/predicted**2)) @ jacobian
    curvature += (centered.T * (mass * (observed.sum()-weighted))) @ centered
    curvature += (penalty/norm) * (np.eye(len(coefficients))
                                  - np.outer(coefficients/norm, coefficients/norm))
    return curvature


@dataclass(frozen=True)
class Deconvolution(Distribution):
    standard_error: np.ndarray
    cdf_standard_error: np.ndarray
    mass_bias: np.ndarray
    coefficients: np.ndarray
    coefficient_covariance: np.ndarray
    information_ratio: float
    _mass_jacobian: np.ndarray
    gradient_norm: float
    retained_observations: int

    @property
    def mass_covariance(self):
        """Materialize the full covariance only on request (O(M^2) output)."""
        return self._mass_jacobian @ self.coefficient_covariance @ self._mass_jacobian.T

    def interval(self, alpha=.05):
        """Same outer grid brackets as the authors' CDF_to_CI, without interpolation."""
        _alpha(alpha)
        cdf = np.cumsum(self.mass)
        indices = [np.searchsorted(cdf, alpha/2, side="right")-1,
                   np.searchsorted(cdf, 1-alpha/2, side="right")]
        return self.grid[np.clip(indices, 0, len(self.grid)-1)]


def deconvolve(release, *, grid_size=1000, spline_degree=5, penalty=.1,
               observation_edges=40, maxiter=200):
    """Fit the authors' Normal deconvolveR model and pointwise uncertainty.

    Matches input/noise scaling, quantile support, natural spline coordinates,
    integrated observation bins, coefficient-norm penalty and CI grid brackets.
    Optimization uses analytic gradient/Hessian rather than R's nlm. Uncertainty
    propagation uses rank-p Jacobians and cumulative sums rather than M-by-M
    covariance operations. Full mass covariance is an optional lazy property.
    Only the already-private release is accessed; CI coverage remains empirical.
    """
    y = np.asarray(release.estimates, dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("deconvolution requires at least two finite scalar estimates")
    sigma = positive(release.calibration.sigma, "sigma")
    integer(grid_size, "grid_size", 16)
    integer(spline_degree, "spline_degree", 2)
    integer(observation_edges, "observation_edges", 3)
    integer(maxiter, "maxiter")
    positive(penalty, "penalty")
    if spline_degree >= grid_size:
        raise ValueError("spline_degree must be smaller than grid_size")
    grid, basis, likelihood, observed = _model(y, sigma, grid_size, spline_degree, observation_edges)
    args = (basis, likelihood, observed, penalty)
    result = minimize(_evaluate, np.ones(spline_degree), args=args, jac=True,
                      hess=lambda a, *args: _evaluate(a, *args, hessian=True),
                      method="trust-exact", options={"maxiter": maxiter, "gtol": 1e-8})
    mass = softmax(basis @ result.x)
    centered = basis - mass @ basis
    derivative = mass[:, None] * centered
    predicted = np.maximum(likelihood @ mass, np.finfo(float).tiny)
    scores = (likelihood @ derivative) / predicted[:, None]
    information = observed.sum() * ((scores.T * predicted) @ scores)
    norm = np.linalg.norm(result.x)
    penalty_gradient = penalty * result.x/norm
    penalty_curvature = penalty/norm * (np.eye(spline_degree)-np.outer(result.x/norm, result.x/norm))
    inverse = np.linalg.inv(information+penalty_curvature)
    covariance = inverse @ information @ inverse.T
    bias = -inverse @ penalty_gradient
    cdf_derivative = np.cumsum(derivative, axis=0)
    se_mass = np.sqrt(np.maximum(0., np.sum((derivative @ covariance)*derivative, axis=1)))
    se_cdf = np.sqrt(np.maximum(0., np.sum((cdf_derivative @ covariance)*cdf_derivative, axis=1)))
    gradient_norm = float(np.linalg.norm(result.jac, ord=np.inf))
    return Deconvolution(grid*sigma, mass, bool(result.success or gradient_norm < 1e-6),
                         result.nit, float(result.fun), se_mass, se_cdf, derivative @ bias,
                         result.x, covariance, float(np.trace(penalty_curvature)/np.trace(information)),
                         derivative, gradient_norm, int(observed.sum()))
