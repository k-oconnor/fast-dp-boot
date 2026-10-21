# Fast DP Boot

![Fast DP Boot: a winged, rocket-powered boot](pic/the_boot.jpg)

`fast-dp-boot` is a scalable differentially private bootstrap library. The Python
import is `dp_boot`.

An implementation of Wang, Cheng, and Awan's
[Differentially Private Bootstrap](https://arxiv.org/abs/2210.06140) in Python,
with optional Rust sampling kernels. Validated
against [the authors' source](https://github.com/Zhanyu-Wang/Differentially_Private_Bootstrap)
at commit `fd001561ed338d47143c94b995f4d47478ab9eb0`.

The matched single-worker pipeline runs **28.0x faster than the authors' R
implementation** locally on the clipped-normal workload at n=10,000 and B=2,000.
An interleaved comparison across 20 shared datasets measured:

| Implementation | Median full pipeline | Speedup over R |
| --- | ---: | ---: |
| Authors' R implementation | 1.720 s | 1.0x |
| Python with NumPy sampling | 61.74 ms | 27.9x |
| Python with Rust sampling | 61.46 ms | 28.0x |

Both Python backends are effectively tied on this workload. NumPy remains the
default; Rust implements sampling only, while calibration, Gaussian noise and
inference remain in Python. Timings include bootstrap sampling, Gaussian noise,
deconvolution, uncertainty statistics and eager full covariance output. They
exclude process startup, file I/O, plotting and experiment orchestration.
See the [three-way comparison](benchmarks/RUST_RESULTS.md) for settings,
validation, limitations and [raw results](benchmarks/matched_rust_pipeline.json).

On 100 identical private releases, the deconvolved CI endpoints match R within
5.6e-16. A separate 1,000-trial simulation achieved 89.5% coverage for nominal
90% intervals. Full methods, limits and raw results are in
[the matched comparison report](benchmarks/MATCHED_RESULTS.md).

## Install and use

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

```python
import numpy as np
from dp_boot import (
    bootstrap_mean, finite_calibration, asymptotic_interval, deconvolve,
)

# Synthetic input; n and clipping bounds are public and fixed.
x = np.random.default_rng(42).binomial(1, .4, 10_000)
b = 256
calibration = finite_calibration(
    len(x), sensitivity=1/len(x), replicates=b, epsilon=1., delta=1e-6,
)
release = bootstrap_mean(
    x, bounds=(0, 1), replicates=b, calibration=calibration,
    memory_bytes=8*1024**2,
)
print(asymptotic_interval(release, alpha=.1))
fit = deconvolve(release)
if fit.converged:
    print(fit.interval(alpha=.1))
    print(fit.standard_error, fit.cdf_standard_error)
# Optional O(M^2) output; default pointwise statistics avoid this allocation:
# covariance = fit.mass_covariance
```

`deconvolve` now uses the authors' model and defaults: standardized observations,
1,000-point quantile/IQR support, canonical natural-spline basis of dimension
five, 39 integrated normal observation bins, coefficient-norm penalty .1,
and the same outer grid interval brackets. An analytic-Hessian optimizer fits
the model. Low-rank derivatives produce pointwise standard errors and bias
without the large covariance operations in R. Golden tests cover the fitted
mass, CDF, standard errors, bias and optional full covariance.

The older, statistically different FFT/spline approximation is available as
`deconvolve_fft`, explicitly experimental. Version 0.2 changes the default
estimator; its tuning arguments are `grid_size`, `spline_degree`, `penalty`,
`observation_edges` and `maxiter`. The earlier `spline_knots` argument belongs
to `deconvolve_fft` only.

## Exact bootstrap kernels

* `bootstrap_mean(..., method="auto")` compresses repeated values into exact
  multinomial counts when useful. Otherwise it uses tiled draws. Count sampling
  costs O(BK) after unique-value preprocessing, versus O(Bn) for direct draws.
* `bootstrap_mean_from_counts(values, counts, ...)` consumes a confidential
  frequency table without expanding records or repeating aggregation.
* `bootstrap_statistic(data, statistic, ...)` accepts a scalar/vector statistic
  with caller-established global joint L2 sensitivity. It keeps one resample
  in memory; regression adapters are not supplied automatically.

Sampling remains Efron's bootstrap with replacement: no continuous-value
binning, Poisson approximation or Gaussian bootstrap approximation is used.
For raw means, `memory_bytes` bounds principal sampling arrays, excluding
input, clipping/unique preprocessing, output, small reduction arrays and library
workspaces. Total memory is O(n+B+budget), not strictly the budget. Set
`method="draws"` to skip unique preprocessing for a known continuous workload.

### Optional Rust sampling backend

With a Rust toolchain and a native linker installed, build the separate extension:

```powershell
python -m pip install ./rust
python -m pytest -q
```

Pass `backend="rust"` to `bootstrap_mean` or `bootstrap_mean_from_counts`:

```python
release = bootstrap_mean(
    x, bounds=(0, 1), replicates=b, calibration=calibration, backend="rust",
)
```

The default remains `backend="numpy"`; installing the core package never needs
Rust. The native backend supplies direct row draws and exact multinomial count
sampling via conditional binomials. Python still validates calibration, clips
inputs, adds Gaussian noise, and performs inference/accounting. The generic
`bootstrap_statistic` callback remains a Python implementation.

Rust uses a ChaCha20 stream seeded from 32 bytes of the supplied NumPy generator.
Seeds reproduce within a backend/build, not across backends or dependency
versions. Explicit seeds remain for experiments only. Native bindings copy input
arrays before releasing the GIL; these O(n) or O(K) copies, output, and unique
preprocessing are outside the sampling scratch budget. Direct draws use constant
sampling scratch space; count sampling uses O(K). Rust frequency-table totals
are limited to 2**53 so integer counts remain representable in binary64.

A local 20-repeat comparison found faster continuous-data sampling but slower
small-support count sampling. A separate, interleaved comparison on the original
R workload found both Python backends effectively tied under the same 8 MiB
sampling budget, as shown above. See
[Rust results and scope](benchmarks/RUST_RESULTS.md) for timings and reproduction.
The standalone Rust crate exposes noiseless
sampling kernels; they must stay inside the confidential computation boundary.

## Privacy calibration and accounting

The contract is fixed public n, replace-one adjacency, valid global L2
sensitivity and independent ideal Gaussian randomness. Means clip to supplied
public bounds and check sensitivity >= (upper-lower)/n. Calibrations are bound
to B and, for the finite bound, n. Additional releases consume additional
privacy budget. Statistics, bounds and sampling parameters must not be chosen
from confidential data without appropriate separate accounting.

`finite_calibration` gives a conservative finite (epsilon,delta) alternative
to the paper's numerical accountant. For the changed row K~Binomial(n,1/n),
choose k with B*Pr(K>k) <= delta/2. Condition on all B multiplicities <= k:
each paired statistic differs by at most k*Delta, and independent Gaussian
composition is k*Delta*sqrt(B)/sigma-GDP. This event depends only on sampling
randomness, with the same probability on neighboring datasets. Charge its
complement to delta and solve the exact Gaussian privacy conversion for the
remaining budget. This can require substantially more noise than tight accounting.

The paper's convenient noise rule is explicitly **asymptotic**:

```python
from dp_boot import asymptotic_calibration
calibration = asymptotic_calibration(1/len(x), b, mu=1.)
# sigma = sqrt((2-2/e)*B)*Delta/mu
```

For reproducing the authors' experiment scripts, use their rounded multiplier
1.125 instead of sqrt(2-2/e), as the benchmark drivers do. Neither is a finite
mu-GDP guarantee. Confidence-interval coverage is also an asymptotic/empirical
property; numerical optimizer convergence does not certify coverage.

`bootstrap_privacy_profile` implements the paper's Proposition 14 Gaussian
composition using an analytic privacy-loss CDF and one zero-padded FFT power:

```python
from dp_boot import bootstrap_privacy_profile
profile = bootstrap_privacy_profile(
    len(x), base_mu=(1/len(x))/calibration.sigma, replicates=b,
    epsilons=[.5, 1., 2.], loss_step=.001, max_loss=4.,
)
print(profile.delta_lower, profile.delta)  # discretization bracket
print(profile.discarded_tail_bound, profile.numerical_mass_error)
```

The upper profile rounds losses upward and charges both truncated tails.
Tiny binomial components are grouped conservatively into the worst group n.
The lower/upper bracket describes this dominating pair's discretization, not
the actual privacy cost of an arbitrary mechanism. In exact arithmetic the
rounding slack is at most B*loss_step on epsilon outside charged tails.
Composition costs O(BJ log(BJ)); epsilon queries use cumulative tail recurrences.
An explicit FFT-size guard prevents accidental oversized allocations.

Independent continuous two-release integration checks fall within these
brackets. Segment-exact integration of the upstream interpolated density also
points to numerical reference quadrature/interpolation as the source of its
original .003 discrepancy. See the report for details.
Floating-point errors remain uncertified; mass error is a diagnostic, not a
rigorous error bound. The finite multiplicity calibration remains separate.

This is research software. NumPy randomness and floating-point noise sampling
are not audited as a hardened DP release system. Explicit seeds are for
experiments, not confidential releases with fixed or disclosed randomness.
Do not publish internal support sizes or timings as privacy-protected outputs.
Vector confidence intervals are marginal, not simultaneous confidence regions.

## Validation and scope

```powershell
python -m benchmarks.coverage --trials 1000
python -m benchmarks.matched_pipeline
python -m benchmarks.deconvolution_scaling
python -m benchmarks.matched_rust_pipeline --trials 20
python -m benchmarks.accounting_reference
python -m benchmarks.accounting_quadrature
```

R comparison drivers require the pinned source checkout and a local R package;
setup and provenance are in [the first comparison](benchmarks/UPSTREAM_COMPARISON.md).
They currently assume this Windows machine's R path. Core library use does
not require R; fixtures allow tests without R. Benchmark memory measurement
uses the optional development dependency psutil.

Native default inference scaled to 64,000 latent points in .584 s and 225 MB
peak process working set; a single full covariance at that size would occupy
32.77 GB. This uses low-rank output, and does not claim an executed R comparison
at that size or better statistical performance from increasing the grid.

This implements the Gaussian core, not all paper experiments. Remaining work
includes sensitivity-aware logistic/quantile regression adapters, broader
coverage and throughput sweeps, certified accountant arithmetic, and further
Rust optimization beyond the optional mean-sampling backend. Source boundaries:
`bootstrap.py`, `privacy.py`, `accounting.py`, `deconvolution.py`; the earlier
experimental FFT model and Algorithm 2 remain in `inference.py`.
