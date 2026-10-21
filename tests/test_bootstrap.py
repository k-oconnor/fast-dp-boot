import numpy as np
import pytest
from scipy.stats import binom
from dp_boot import (bootstrap_mean, bootstrap_statistic, asymptotic_calibration,
                     finite_calibration, gaussian_delta, asymptotic_interval, deconvolve)
from dp_boot.bootstrap import _mean_counts, _mean_draws
from dp_boot.inference import _objective
from scipy.optimize._numdiff import approx_derivative


@pytest.mark.parametrize("kernel", [_mean_counts, _mean_draws])
def test_exact_bernoulli_bootstrap_distribution(kernel):
    # Full discrete sampling law, not just matching implementation outputs.
    x = np.array([0.] * 7 + [1.] * 3)
    means = kernel(x, 60000, np.random.default_rng(3), 65536)
    actual = np.bincount(np.rint(means * 10).astype(int), minlength=11) / len(means)
    expected = binom.pmf(np.arange(11), 10, .3)
    assert np.max(np.abs(actual - expected)) < .006


@pytest.mark.parametrize("budget", [64, 4096, 1000000])
def test_tiling_moments_and_counts_fallback(budget):
    x = np.linspace(-3, 7, 100)
    for kernel in [_mean_draws, _mean_counts]:
        means = kernel(x, 10000, np.random.default_rng(1), budget)
        assert abs(means.mean() - x.mean()) < .012
        assert means.var() == pytest.approx(x.var() / len(x), rel=.05)


def test_finite_accounting_and_binding():
    c = finite_calibration(1000, .001, 100, epsilon=1., delta=1e-6)
    assert c.delta(1.) == pytest.approx(1e-6, rel=1e-6)
    assert c.tail_delta <= 5e-7
    assert c.multiplicity_cap >= 1
    with pytest.raises(ValueError, match="n/replicates"):
        bootstrap_mean(np.zeros(999), bounds=(0, 1), replicates=100, calibration=c)
    with pytest.raises(ValueError, match="sensitivity"):
        bootstrap_mean(np.zeros(1000), bounds=(0, 2), replicates=100, calibration=c)


def test_gaussian_conversion_extremes():
    from scipy.stats import norm
    assert gaussian_delta(0, 1) == pytest.approx(2 * norm.cdf(.5) - 1)
    assert gaussian_delta(1000, .01) == 0
    assert 0 < gaussian_delta(10, 3) < 1


def test_algorithm_one_noise_and_clipping():
    c = asymptotic_calibration(1 / 20, 50000, 2.)
    r = bootstrap_mean(np.full(20, 9.), bounds=(0, 1), replicates=50000,
                       calibration=c, rng=10)
    assert r.estimates.mean() == pytest.approx(1., abs=.1)
    assert r.estimates.var() == pytest.approx(c.sigma ** 2, rel=.03)
    assert c.method == "paper_asymptotic"


def test_generic_vector_statistic_and_repeatability():
    x = np.arange(20.)
    c = asymptotic_calibration(1., 30, 1.)
    stat = lambda y: np.array([y.mean(), y.mean() * 2])
    a = bootstrap_statistic(x, stat, replicates=30, calibration=c, rng=4)
    b = bootstrap_statistic(x, stat, replicates=30, calibration=c, rng=4)
    assert a.estimates.shape == (30, 2)
    np.testing.assert_array_equal(a.estimates, b.estimates)
    assert asymptotic_interval(a).shape == (2, 2)


def test_algorithm_two_matches_formula():
    from scipy.stats import norm, chi2
    from dp_boot import Release
    c = asymptotic_calibration(.01, 100, 1.)
    y = np.random.default_rng(3).normal(5, .3, 100)
    r = Release(y, c)
    v = max(0., 99 * np.var(y, ddof=1) / chi2.ppf(.025, 99) - c.sigma**2)
    radius = norm.ppf(.9875) * np.sqrt(v + (v + c.sigma**2) / 100)
    np.testing.assert_allclose(asymptotic_interval(r), [y.mean()-radius, y.mean()+radius])


def test_fft_objective_against_dense_and_finite_difference():
    from scipy.linalg import toeplitz
    from scipy.special import softmax
    rng = np.random.default_rng(9)
    basis = rng.normal(size=(65, 5))
    coefficients = rng.normal(size=5) * .1
    kernel = np.exp(-.5 * ((np.arange(65) - 32) / 4)**2)
    kernel /= kernel.sum()
    observed = rng.multinomial(1000, np.full(65, 1/65))
    value, gradient = _objective(coefficients, basis, kernel, observed, 1.)
    col = np.r_[kernel[32:], np.zeros(32)]
    matrix = toeplitz(col)
    dense = -observed @ np.log(matrix @ softmax(basis @ coefficients))
    dense += np.sqrt(coefficients @ coefficients + 1e-12)
    assert value == pytest.approx(dense, rel=1e-12)
    numeric = approx_derivative(lambda a: _objective(a, basis, kernel, observed, 1.)[0], coefficients)
    np.testing.assert_allclose(gradient, numeric.ravel(), rtol=1e-5, atol=1e-5)


def test_deconvolution_recovers_noisy_normal_scale():
    from dp_boot import Release
    rng = np.random.default_rng(9)
    c = asymptotic_calibration(.3 / np.sqrt((2 - 2/np.e)*10000), 10000, 1.)
    y = rng.normal(2., 1., 10000) + rng.normal(0., .3, 10000)
    from dp_boot import deconvolve_fft
    d = deconvolve_fft(Release(y, c), grid_size=513, spline_knots=25)
    assert d.converged
    assert d.mass.sum() == pytest.approx(1.)
    assert np.all(d.mass >= 0)
    mean = d.grid @ d.mass
    variance = (d.grid - mean)**2 @ d.mass
    assert mean == pytest.approx(2, abs=.06)
    assert variance == pytest.approx(1, abs=.13)
    assert d.interval()[0] < 2 < d.interval()[1]


@pytest.mark.parametrize("kwargs", [{"replicates": 1}, {"mu": 0}, {"sensitivity": -1}])
def test_invalid_calibration(kwargs):
    args = dict(sensitivity=1, replicates=10, mu=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        asymptotic_calibration(**args)


def test_preaggregated_matches_expanded_distribution_and_seed():
    from dp_boot import bootstrap_mean_from_counts
    values = np.array([0., .25, 1.])
    counts = np.array([70, 10, 20])
    c = asymptotic_calibration(.01, 1000, 1.)
    a = bootstrap_mean(np.repeat(values, counts), bounds=(0, 1), replicates=1000,
                       calibration=c, method="counts", rng=4)
    b = bootstrap_mean_from_counts(values, counts, bounds=(0, 1), replicates=1000,
                                   calibration=c, rng=4)
    np.testing.assert_array_equal(a.estimates, b.estimates)


def test_draw_scratch_budget():
    class CheckedRng:
        def integers(self, low, high, size):
            assert np.prod(size) * 16 <= 4096
            return np.random.default_rng(1).integers(low, high, size)
    _mean_draws(np.arange(100000.), 5, CheckedRng(), 4096)


def test_auto_uses_draws_on_continuous_data():
    x = np.linspace(0, 1, 100)
    a = _mean_counts(x, 100, np.random.default_rng(8), 65536, auto=True)
    b = _mean_draws(x, 100, np.random.default_rng(8), 65536)
    np.testing.assert_array_equal(a, b)
