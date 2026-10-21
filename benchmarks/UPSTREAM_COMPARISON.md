# Comparison against the authors' implementation

This records the first-pass diagnosis. The estimator has since been replaced
by a matched implementation; see [the second-pass results](MATCHED_RESULTS.md)
for current fidelity, end-to-end benchmarks and the accounting investigation.

Reference: [Zhanyu-Wang/Differentially_Private_Bootstrap](https://github.com/Zhanyu-Wang/Differentially_Private_Bootstrap),
commit `fd001561ed338d47143c94b995f4d47478ab9eb0`.
The local sparse checkout is in `reference/upstream`, with its original Apache-2.0 license.

The earlier 600x result compared a synthetic 16-value dataset against a NumPy
loop. It is not a measured improvement over this repository or the paper's
inference pipeline. The earlier dense-versus-FFT convolution benchmark also
does not represent the default `deconvolveR` likelihood used by the authors.

## Bootstrap sampling: shared inputs and actual upstream function

`upstream_bootstrap.R` extracts the unchanged `func_param` and `func_boot`
definitions from `clamp_normal_mean.r` without running its experiment loops.
The Python driver supplies identical input arrays to both implementations.
Both generate exact Efron bootstrap means. Seeds differ across R and NumPy;
matching random streams is not claimed.

| Case | n | B | Authors' R `boot` | Python auto kernel | Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clipped normal, matching their distribution | 10,000 | 2,000 | 1.150 s | 0.0522 s | 22.0x |
| Synthetic 16-value data | 100,000 | 256 | 1.600 s | 0.000427 s | 3,745x |

Median of three in-process runs, one worker. CSV loading, process startup,
noise generation, deconvolution and parallel orchestration are excluded.
The first case uses draws because 3,897 distinct values offer insufficient
compression. Its speedup combines NumPy batching with avoiding R `boot`'s
general-purpose overhead. The second case demonstrates the specialized
multinomial-count algorithm, and is not representative of continuous regression
or the general paper workload. These are component timings, not end-to-end
speedups or an isolation of language effects.

Data, seed, bootstrap means/variances and exact conditional moments are recorded
in [upstream_bootstrap.json](upstream_bootstrap.json). Monte Carlo estimates
need not match exactly; 256 replicates in the synthetic case are not enough
to make a stringent distribution-validation claim. Separate library tests
check the full discrete law with 60,000 replicates.

## Privacy composition: actual upstream recursion

`upstream_accounting.py` executes the authors' unchanged `approx_f_numerical`
and the density-building statements from `dpboot_dual_all`, extracted through
Python's AST to avoid command-line plotting side effects. We use their
calibration `base_mu = 1/sqrt(B)` and their epsilon range [-6,10]. The native
accountant reports the nonnegative portion of that same grid.

This is a comparison of two numerical approximations, not equivalent-accuracy
certification. At n=5, B=5, increasing the upstream density/epsilon grid from
81 to 321 points reduced the maximum absolute delta discrepancy at native
step .001 from .00545 to .00284. The upstream quadrature emitted integration
warnings. Native loss rounding, tail truncation, the upstream interpolation
grid and quadrature error must all be considered. No speedup is promoted as
a validated privacy-accounting improvement while this discrepancy remains.

Full timing, profile arrays and warning messages:
[81 points](upstream_accounting.json), [321 points](upstream_accounting_refined.json).
The finer 1,001-point run is recorded separately in
[upstream_accounting_fine.json](upstream_accounting_fine.json).
Its maximum discrepancy at native step .001 is .00314, so refinement has not
resolved the difference; integration warnings persist. Timing alone is not an
acceptance criterion for this accountant.

## Deconvolution: fidelity gap confirmed

The authors' actual settings in `coverage_check.r`, `clamp_normal_mean.r` and
`Figure7-compute.r` are:

* Divide private estimates by the noise standard deviation.
* Latent grid: 1,000 points from Q1-3*IQR to Q3+3*IQR.
* `deconvolveR::deconv(..., family="Normal", pDegree=5, c0=.1)`.
* Rounded noise multiplier 1.125, rather than the exact sqrt(2-2/e).

With the package's default n=40, the normal-family likelihood matrix has
39 observation bins by 1,000 latent points, as confirmed at runtime. The
package uses integrated Gaussian bin probabilities, scaled natural-spline
columns, and also computes covariance, bias and standard errors. Our current
native implementation uses a different basis/penalty geometry and grid, and
omits those uncertainty calculations. An M-by-M Gaussian convolution benchmark
does not characterize this reference likelihood.

We ran both on ten identical sets of private estimates using the upstream noise
multiplier. Interval endpoints differ; this is expected from the different
models and establishes that the native estimator is not a faithful port.
Timings are retained only as diagnostics because the computations differ.
Ten trials cannot assess coverage. The R dependency used here is deconvolveR
source release 1.2-2, installed locally under R 4.6.0; the authors do not pin
their original package version. Results:
[upstream_deconvolution.json](upstream_deconvolution.json).

The native 1,000-trial undercoverage result remains a limitation of our
implementation, not evidence of a defect in the authors' method.

## Reproduce

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/Zhanyu-Wang/Differentially_Private_Bootstrap.git reference/upstream
git -C reference/upstream checkout fd001561ed338d47143c94b995f4d47478ab9eb0
python -m benchmarks.upstream_bootstrap
python -m benchmarks.upstream_accounting --bins 321 --output benchmarks/upstream_accounting_refined.json
& 'C:/Program Files/R/R-4.6.0/bin/Rscript.exe' --vanilla benchmarks/install_reference_r.R
python -m benchmarks.upstream_deconvolution
```

The R drivers currently use the installed Windows R path; adjust it for another
machine. Dependencies/checkouts and generated shared-input files are isolated
under ignored `reference/`; benchmark scripts and result summaries are retained.
Next acceptance gate: match the reference estimator and establish comparable
numerical/statistical accuracy before reporting an end-to-end improvement.
