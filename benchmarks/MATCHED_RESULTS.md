# Matched implementation, second pass

Reference: [authors' repository](https://github.com/Zhanyu-Wang/Differentially_Private_Bootstrap),
commit `fd001561ed338d47143c94b995f4d47478ab9eb0`.
Environment: Windows, Python 3.11.9, NumPy 2.2.6, SciPy 1.16.1,
R 4.6.0, deconvolveR source release 1.2-2 (R reports 1.2.2).

## Estimator fidelity

`deconvolve` now matches the authors' statistical model: noise-standardized
observations; 1,000-point quantile/IQR support; five centered/scaled canonical
natural-spline columns; 39 integrated Gaussian observation bins; coefficient-
norm penalty .1; starting coefficients of one; and outer CI grid brackets.
Binning preserves R's endpoint conventions, including its omission of
observations outside the rounded range.

Natural constraints and a QR nullspace construct the spline basis with SciPy.
On the saved R fixture it agrees within 1e-14; fitted mass agrees within 3e-10.
The optimizer uses analytic gradient/Hessian rather than R's `nlm`.
Golden tests compare mass, CDF, mass/CDF standard errors, bias and full covariance.

Pointwise uncertainties use D = diag(g)(Q - E_g Q), a rank-five derivative
matrix. CDF derivatives are cumulative row sums of D. Their variances come
from the small coefficient covariance, avoiding dense cumulative matrices and
large covariance products. Full mass covariance D Cov(alpha) D-transpose
remains available as a lazy property. The earlier FFT/spline approximation
is retained as `deconvolve_fft`, not the default.

## Runtime and coverage

| Experiment | Reference R | Python | Evidence |
| --- | ---: | ---: | --- |
| Fit + pointwise statistics, 100 identical private releases | 650 ms median | 9.66 ms median | CI endpoints agree within 5.6e-16; all fits converged |
| Same fits, Python also eagerly constructs full covariance | 650 ms median | 10.61 ms median | 61.3x; full covariance output included |
| Full pipeline, n=10,000, B=2,000, 20 shared datasets | 1.730 s median | .0671 s median | 25.8x; sampling, noise, fit, uncertainty and full covariance |

The pipeline uses identical raw inputs with independent R/NumPy random
streams. It excludes file I/O, process startup, plots and multi-worker
orchestration. Estimator equivalence is separately checked on identical
private releases. Single per-fit measurements are aggregated across inputs;
these are local measurements, not a claim about every CPU or workload.
The R baseline computes its full default statistics even though the paper's
interval only needs the CDF. The full-covariance comparison includes equivalent
Python output rather than timing a reduced result against R's full result.

At n=2000, B=200 and the authors' 1.125 noise multiplier, 1,000 synthetic
trials gave **89.5% coverage for nominal 90% intervals**: 95% binomial interval
87.43%-91.33%, mean width .03095. All fits converged. The earlier approximation
gave 86.2%; this change follows estimator fidelity, not tuning to a coverage
target. One simulation setting does not establish finite coverage. Algorithm 2
remained conservative at 97.6% coverage here.

A low-signal check (n=10,000, B=20, mu=.1, 20 shared releases) matched every
R interval endpoint within 6.7e-16; all fits converged. Twenty trials are used
for parity, not a coverage claim.

Raw results: [fits](matched_deconvolution.json),
[full statistics](matched_statistics.json), [pipeline](matched_pipeline.json),
[coverage](matched_coverage.json), [low-signal parity](matched_low_signal.json).

## Memory scaling

Default inference returns the distribution and pointwise uncertainty without
materializing full covariance. Fresh-process Windows measurements:

| Latent points | Fit time | Peak process working set | One dense covariance alone |
| ---: | ---: | ---: | ---: |
| 1,000 | .014 s | 105 MB | 8 MB |
| 4,000 | .035 s | 113 MB | 128 MB |
| 16,000 | .156 s | 134 MB | 2.05 GB |
| 64,000 | .584 s | 225 MB | 32.77 GB |

Working set includes Python/SciPy. These are native scaling measurements,
not executed R comparisons at larger sizes. Increasing the grid also changes
the reference model's penalty geometry; it need not improve statistical
performance. [Measurements](deconvolution_scaling.json).

## Accounting discrepancy

We separated the authors' interpolated density from their adaptive quadrature.
Integrals of q and q*log(q) are elementary on each linear-density segment.
They give an independent, segment-exact implementation of the interpolated
recursion, applied with direct convolution on a wider epsilon domain.

* Halving its epsilon step from .008 to .004 changed values by only 1.15e-6.
* It still differs from the original quadrature by .003108.
* FFT composition of that same interpolated density agrees with the native
  analytic-density FFT within 3.84e-5 at loss step .0005.
* Upward-rounded native FFT differs from the independently integrated
  recursion by .000288 at that step; upward rounding explains most of this.
* The upstream interpolated q integrates to 1.00000861 rather than one,
  another approximation to distinguish from the underlying theorem.

These results point to reference quadrature/interpolation as the main source
of the original .003 discrepancy, without changing the privacy theorem.

A second independent check uses the exact one-release profile and continuous
loss density for two releases, including the atom at zero. It has no epsilon
interpolation, density interpolation, FFT or PLD lattice. At n=5 and n=50,
all twelve epsilon checks lie inside the native FFT's lower/upper rounding
bracket; quadrature error estimates are about 1e-12 or smaller. The outputs
are also used as regression-test golden values.

The accountant now reports `delta_lower` alongside `delta` (upper) to expose
lattice slack. These describe the compressed dominating pair, not an arbitrary
mechanism's actual privacy loss. All epsilon queries use cumulative tail
recurrences instead of scanning the composed lattice per query. Ordinary
floating-point errors remain uncertified, and the conservative multiplicity
calibration remains separate from numerical-accountant calibration.

Evidence: [density diagnostic](accounting_diagnostic.json),
[segment-exact integration](accounting_reference.json),
[continuous quadrature](accounting_quadrature.json).

## Reproduce

The pinned checkout and local R package setup are described in
[the first comparison](UPSTREAM_COMPARISON.md). From the repository root:

```powershell
python -m pytest -q
python -m benchmarks.upstream_deconvolution --trials 100
# Run before another benchmark overwrites the shared noisy_releases.csv:
python -m benchmarks.matched_statistics
python -m benchmarks.coverage --trials 1000
python -m benchmarks.matched_pipeline
python -m benchmarks.deconvolution_scaling
python -m benchmarks.accounting_reference
python -m benchmarks.accounting_quadrature
```

`accounting_diagnostic` additionally runs a slower controlled variant of the
original quadrature. R drivers currently assume this machine's installed
Windows R path. Library use requires only NumPy and SciPy.
