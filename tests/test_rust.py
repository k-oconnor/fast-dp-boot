"""Distributional checks for the optional native backend, not RNG parity tests."""
import math
import sys

import numpy as np
import pytest
from scipy.stats import binom

from dp_boot import (
    asymptotic_calibration, bootstrap_mean, bootstrap_mean_from_counts,
    deconvolve, finite_calibration,
)
from dp_boot.bootstrap import _count_draws, _mean_draws


@pytest.fixture
def native():
    return pytest.importorskip("dp_boot_native")


@pytest.mark.parametrize("counts", [False, True])
def test_rust_bernoulli_law(native, counts):
    rng = np.random.default_rng(3)
    if counts:
        out = _count_draws(np.array([0., 1.]), np.array([7, 3]), 10,
                           60000, rng, 64, backend="rust")
    else:
        out = _mean_draws(np.array([0.] * 7 + [1.] * 3), 60000, rng, 64,
                          backend="rust")
    observed = np.bincount(np.rint(out * 10).astype(int), minlength=11) / len(out)
    assert np.max(np.abs(observed - binom.pmf(np.arange(11), 10, .3))) < .006


def test_rust_multinomial_law(native):
    # Enumerate the exact law for a three-category, four-row empirical sample.
    expected = np.zeros(17)
    for a in range(5):
        for b in range(5 - a):
            c = 4 - a - b
            probability = math.factorial(4) / (
                math.factorial(a) * math.factorial(b) * math.factorial(c)
            ) * .25 ** (a + c) * .5 ** b
            expected[b + 4*c] += probability
    out = _count_draws(np.array([0., .25, 1.]), np.array([1, 2, 1]), 4,
                       80000, np.random.default_rng(7), 64, backend="rust")
    observed = np.bincount(np.rint(out * 16).astype(int), minlength=17) / len(out)
    assert np.max(np.abs(observed - expected)) < .004


@pytest.mark.parametrize("budget", [64, 4096])
def test_rust_continuous_moments(native, budget):
    x = np.linspace(-3, 7, 100)
    out = _mean_draws(x, 20000, np.random.default_rng(4), budget, backend="rust")
    assert out.mean() == pytest.approx(x.mean(), abs=.012)
    assert out.var() == pytest.approx(x.var() / len(x), rel=.04)


@pytest.mark.parametrize("method", ["draws", "counts", "auto"])
def test_rust_release_repeatability_and_preaggregation(native, method):
    x = np.repeat([0., .25, 1.], [70, 10, 20])
    calibration = asymptotic_calibration(.01, 1000, 1.)
    args = dict(bounds=(0, 1), replicates=1000, calibration=calibration,
                rng=4, backend="rust")
    a = bootstrap_mean(x, method=method, **args)
    b = bootstrap_mean(x, method=method, **args)
    np.testing.assert_array_equal(a.estimates, b.estimates)
    assert a.calibration is calibration
    if method != "draws":
        preaggregated = bootstrap_mean_from_counts([0., .25, 1.], [70, 10, 20], **args)
        np.testing.assert_array_equal(a.estimates, preaggregated.estimates)
    fit = deconvolve(a)
    assert fit.converged
    assert np.all(np.isfinite(fit.interval()))


@pytest.mark.parametrize("method", ["draws", "counts", "auto"])
def test_rust_clipping_and_noise(native, method):
    calibration = asymptotic_calibration(1/20, 50000, 2.)
    release = bootstrap_mean(np.full(20, 9.), bounds=(0, 1), replicates=50000,
                             calibration=calibration, rng=10, backend="rust", method=method)
    assert release.estimates.mean() == pytest.approx(1., abs=.1)
    assert release.estimates.var() == pytest.approx(calibration.sigma**2, rel=.03)


def test_rust_retains_calibration_checks():
    calibration = finite_calibration(1000, .001, 100, 1., 1e-6)
    with pytest.raises(ValueError, match="n/replicates"):
        bootstrap_mean(np.zeros(999), bounds=(0, 1), replicates=100,
                       calibration=calibration, backend="rust")
    with pytest.raises(ValueError, match="sensitivity"):
        bootstrap_mean(np.zeros(1000), bounds=(0, 2), replicates=100,
                       calibration=calibration, backend="rust")


def test_invalid_backend_and_missing_extension(monkeypatch):
    args = dict(bounds=(0, 1), replicates=10,
                calibration=asymptotic_calibration(.1, 10, 1.))
    with pytest.raises(ValueError, match="backend"):
        bootstrap_mean(np.zeros(10), backend="unknown", **args)
    with pytest.raises(ValueError, match="backend"):
        bootstrap_mean_from_counts([0.], [10], backend="unknown", **args)
    monkeypatch.setitem(sys.modules, "dp_boot_native", None)
    with pytest.raises(ImportError, match="pip install ./rust"):
        bootstrap_mean(np.zeros(10), backend="rust", **args)
    # Importing and using the default backend never requires the extension.
    assert bootstrap_mean(np.zeros(10), **args).estimates.shape == (10,)


def test_native_validation_and_strides(native):
    seed = bytes(32)
    with pytest.raises(ValueError, match="32 bytes"):
        native.mean_draws(np.ones(10), 2, b"bad")
    with pytest.raises(ValueError, match="finite"):
        native.mean_draws(np.array([np.nan]), 2, seed)
    with pytest.raises(ValueError, match="match"):
        native.mean_counts(np.ones(2), np.ones(1, dtype=np.uint64), 2, seed)
    with pytest.raises(ValueError, match="total"):
        native.mean_counts(np.ones(1), np.array([2**53 + 1], dtype=np.uint64), 2, seed)
    out = native.mean_counts(np.array([9., .25, -4.]),
                             np.array([0, 10, 0], dtype=np.uint64), 100, seed)
    np.testing.assert_array_equal(out, .25)
    # Binding copies strided/reversed input in logical order before releasing GIL.
    x = np.arange(20., dtype=float)[::-2]
    np.testing.assert_array_equal(native.mean_draws(x, 10, seed),
                                  native.mean_draws(x.copy(), 10, seed))
