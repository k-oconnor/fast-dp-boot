from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from dp_boot import Release, asymptotic_calibration, deconvolve
from dp_boot.deconvolution import _model, _evaluate

FOLDER = Path(__file__).parent / "fixtures/deconvolver"


def fixture():
    y = np.loadtxt(FOLDER/"release.csv")
    sigma = float(np.loadtxt(FOLDER/"sigma.csv"))
    calibration = asymptotic_calibration(sigma/np.sqrt((2-2/np.e)*len(y)), len(y), 1.)
    stats = np.genfromtxt(FOLDER/"stats.csv", delimiter=",", names=True)
    return Release(y, calibration), stats


def test_reference_model_basis_and_integrated_probabilities():
    r, stats = fixture()
    grid, basis, likelihood, counts = _model(r.estimates, r.calibration.sigma, 101, 5, 40)
    np.testing.assert_allclose(grid, stats["theta"], atol=1e-12)
    np.testing.assert_allclose(basis, np.loadtxt(FOLDER/"basis.csv", delimiter=","), atol=2e-13)
    np.testing.assert_allclose(likelihood, np.loadtxt(FOLDER/"likelihood.csv", delimiter=","), atol=2e-14)
    assert counts.sum() <= len(r.estimates)


def test_reference_fit_and_low_rank_uncertainty():
    r, expected = fixture()
    fit = deconvolve(r, grid_size=101)
    assert fit.converged
    np.testing.assert_allclose(fit.mass, expected["g"], atol=1e-8)
    np.testing.assert_allclose(np.cumsum(fit.mass), expected["G"], atol=1e-8)
    np.testing.assert_allclose(fit.standard_error, expected["SEg"], atol=1e-8)
    # R can take sqrt of a tiny negative variance at the final CDF point.
    valid = np.isfinite(expected["SEG"])
    np.testing.assert_allclose(fit.cdf_standard_error[valid], expected["SEG"][valid], atol=1e-7)
    np.testing.assert_allclose(fit.mass_bias, expected["Biasg"], atol=1e-8)
    np.testing.assert_allclose(fit.mass_covariance,
                               np.loadtxt(FOLDER/"mass_covariance.csv", delimiter=","), atol=1e-9)
    assert abs(fit.objective-float(np.loadtxt(FOLDER/"objective.csv"))) < 1e-9
    # Full covariance is optional; its diagonal agrees with the O(M p^2) route.
    np.testing.assert_allclose(np.diag(fit.mass_covariance), fit.standard_error**2, atol=1e-15)


def test_reference_likelihood_derivatives():
    r, _ = fixture()
    _, basis, likelihood, counts = _model(r.estimates, r.calibration.sigma, 101, 5, 40)
    args = (basis, likelihood, counts, .1)
    a = np.arange(1., 6.)
    _, gradient = _evaluate(a, *args)
    numeric = approx_derivative(lambda a: _evaluate(a, *args)[0], a)
    np.testing.assert_allclose(gradient, numeric.ravel(), rtol=1e-6, atol=1e-7)
    numeric_hessian = approx_derivative(lambda a: _evaluate(a, *args)[1], a)
    np.testing.assert_allclose(_evaluate(a, *args, hessian=True), numeric_hessian,
                               rtol=1e-6, atol=1e-7)
