import numpy as np
import pytest
from dp_boot import bootstrap_privacy_profile, gaussian_delta
from dp_boot.accounting import _groups, _loss_cdf
from scipy.stats import norm


def test_single_record_loss_cdf_is_gaussian():
    p0, scales, weights = _groups(1, .5, 1e-14)
    grid = np.linspace(-4, 4, 101)
    np.testing.assert_allclose(_loss_cdf(grid, p0, scales, weights),
                               norm.cdf((grid - .5**2/2)/.5), atol=1e-14)


def test_fft_composition_bounds_known_gaussian_profile():
    b, h = 5, .002
    eps = np.array([0., .5, 1., 3.])
    profile = bootstrap_privacy_profile(1, .3, b, eps, loss_step=h, max_loss=4)
    mu = .3 * np.sqrt(b)
    for epsilon, delta in zip(eps, profile.delta):
        exact = gaussian_delta(epsilon, mu)
        relaxed = gaussian_delta(max(0., epsilon-b*h), mu) if epsilon >= b*h else 1.
        assert exact - 1e-11 <= delta <= relaxed + profile.discarded_tail_bound + 1e-11
    assert np.all(profile.delta_lower <= [gaussian_delta(e,mu)+1e-11 for e in eps])
    assert profile.numerical_mass_error < 1e-11


def test_general_accountant_refines_and_profile_monotone():
    eps = [0., 1., 2., 4.]
    coarse = bootstrap_privacy_profile(10000, .1, 30, eps, loss_step=.02)
    fine = bootstrap_privacy_profile(10000, .1, 30, eps, loss_step=.01)
    assert np.all(np.diff(fine.delta) <= 0)
    assert np.all(fine.delta <= coarse.delta + 1e-10)
    assert fine.discarded_tail_bound < 1e-8
    assert fine.numerical_mass_error < 1e-10


def test_upper_tail_is_charged_and_fft_size_guard():
    a = bootstrap_privacy_profile(1, 2., 3, [1.], max_loss=.2, loss_step=.01)
    assert a.discarded_tail_bound > .1
    assert a.delta[0] >= gaussian_delta(1., 2*np.sqrt(3)) - 1e-10
    with pytest.raises(ValueError, match="FFT points"):
        bootstrap_privacy_profile(1000, .1, 1000000, [1.])


def test_batched_hockey_stick_matches_direct_formula_and_extreme_queries():
    from dp_boot.accounting import _hockey_stick
    rng = np.random.default_rng(4)
    mass = rng.random(1001)
    mass /= mass.sum()
    losses = (np.arange(len(mass))-500)*.01
    eps = np.r_[np.linspace(0.,6.,313), 1000., 1e308]
    actual = _hockey_stick(mass,500,.01,eps)
    expected = [mass[losses>e] @ (-np.expm1(e-losses[losses>e])) for e in eps]
    np.testing.assert_allclose(actual,expected,atol=3e-14)


@pytest.mark.parametrize("n,mu,expected", [
    (5,1/np.sqrt(5),[.2636166523041068,.1258195562708606,.060246973737035144,
                    .014780915369639871,.000979874512766081,6.356941740461143e-5]),
    (50,.3,[.18007967788256235,.05514502080476312,.018616882496457297,
            .0027196859386724825,9.393114812161795e-5,4.321335282895839e-6]),
])
def test_independent_two_release_integral_is_inside_rounding_bracket(n,mu,expected):
    # Golden values: benchmarks/accounting_quadrature.py, continuous integration
    # of exact one-release profiles, including the loss-zero atom. No FFT/lattice.
    p=bootstrap_privacy_profile(n,mu,2,[0.,.5,1.,2.,4.,6.],loss_step=.0005,max_loss=20.)
    assert np.all(p.delta_lower <= np.asarray(expected)+1e-10)
    assert np.all(np.asarray(expected) <= p.delta+1e-10)


def test_profile_calibration_meets_delta_tightly_and_beats_the_finite_bound():
    from dp_boot import bootstrap_mean, finite_calibration, profile_calibration
    n, b, step = 1000, 50, .002
    c = profile_calibration(n, 1/n, b, 1., 1e-6, loss_step=step, rtol=1e-3)
    upper = lambda s: bootstrap_privacy_profile(n, (1/n)/s, b, [1.], loss_step=step, max_loss=6.).delta[0]
    assert upper(c.sigma) <= 1e-6
    assert upper(c.sigma / 1.002) > 1e-6          # tight to the tolerance
    assert c.delta(1.) == pytest.approx(upper(c.sigma))
    assert c.sigma < finite_calibration(n, 1/n, b, epsilon=1., delta=1e-6).sigma
    release = bootstrap_mean(np.zeros(n), bounds=(0, 1), replicates=b, calibration=c,
                             rng=np.random.default_rng(0))
    assert release.estimates.shape == (b,)


def test_profile_calibration_reports_delta_only_at_its_epsilon():
    from dp_boot import profile_calibration
    c = profile_calibration(200, 1/200, 10, 1., 1e-5)
    with pytest.raises(ValueError, match="calibrated epsilon"):
        c.delta(.5)
