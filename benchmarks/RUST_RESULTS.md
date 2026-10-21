# Optional Rust sampling backend

The first Rust implementation covers exact mean-bootstrap sampling, exposed
through `backend="rust"` on the existing Python mean-release APIs. Calibration,
Gaussian noise, deconvolution, and accounting remain in Python. The generic
statistic callback is unchanged. This is not a complete Rust port.

## Direct, interleaved comparison with R

A new three-way run used the original reference workload and measured all three
implementations in the same session, rather than multiplying previous speedups.
Across 20 shared datasets, n=10,000 and B=2,000:

| Implementation | Median release | Median full pipeline | Full-pipeline speedup over R |
| --- | ---: | ---: | ---: |
| Authors' R implementation | 1,120 ms | 1,720 ms | 1.00x |
| Python with NumPy sampling | 48.72 ms | 61.74 ms | 27.86x |
| Python with Rust sampling | 47.70 ms | 61.46 ms | 27.99x |

The measured Rust-backed pipeline is **28.0x faster than R**. Rust and NumPy
are effectively tied here: their full-pipeline median ratio is only 1.005x,
which does not establish a meaningful performance advantage for Rust.

This comparison uses `clip(Normal(.5, 1.0), 0, 1)`, dataset seed 20261022,
`method="auto"`, an 8 MiB memory budget and the authors' 1.125 noise multiplier.
R uses the unchanged `func_boot` from the pinned upstream commit
`fd001561ed338d47143c94b995f4d47478ab9eb0`, R 4.6.0, boot 1.3.32 and
deconvolveR 1.2.2. The Python/native versions are the same as listed below.
The Rust-backed entry still uses Python for Gaussian noise and inference.

Each implementation receives one untimed warmup. R remains loaded in a
persistent process, and trial execution order rotates R/NumPy/Rust,
NumPy/Rust/R, Rust/R/NumPy to reduce order bias. The same dataset and
calibration are used for all three, with independent algorithmic random streams.
Timing includes bootstrap sampling, Gaussian noise, reference-compatible
deconvolution, pointwise statistics and eager full 1,000-by-1,000 mass covariance.
Startup, input generation/loading, CSV I/O, interprocess coordination,
calibration, interval extraction and validation are excluded.

All 40 timed Python fits and 20 additional validation fits converged. On the
20 identical noisy releases produced by R, Python's 90% interval endpoints
agreed with R within **7.78e-16**. Full mass covariance was checked finite and
Python covariance diagonals matched squared mass standard errors. R produced
one nonfinite CDF standard error in each of ten trials; inspection of trial 6
located it at the terminal CDF value of 1, consistent with floating-point
cancellation. Per-trial counts are preserved in the raw result. Its fitted
masses, CDFs, mass standard errors,
biases and mass covariances were finite. The reference computation was unchanged.
This run measures throughput and estimator agreement, not coverage or a
certified privacy guarantee.

The earlier two-way Rust comparison below used a narrower normal distribution
(standard deviation .2), a 32 MiB memory budget, and repeated timing on one
dataset per workload. These differences, especially the NumPy tiling budget,
make its speedup unsuitable for multiplication by the older R/Python ratio.

Raw timings, execution orders, environment, binary/input hashes, endpoint
validation and reference diagnostics are in
[`matched_rust_pipeline.json`](matched_rust_pipeline.json). Previous result
files are preserved. Reproduce with the pinned reference checkout and local R
dependencies available:

```powershell
python -m pip install ./rust
python -m benchmarks.matched_rust_pipeline --trials 20
# Optional: --rscript path/to/Rscript
```

## Local measurements

Windows x86-64, Python 3.11.9, NumPy 2.2.6, SciPy 1.16.1, native package 0.1.0.
The extension was built in release mode by Maturin with rustc 1.99.1
(`58cf8f9c3`, 2026-10-13), MSVC target. Exact crate dependencies are recorded in
[`rust/Cargo.lock`](../rust/Cargo.lock).

Twenty repetitions per workload, one worker, B=2,000. Median elapsed times:

| Workload | NumPy release | Rust release | Release speedup | NumPy pipeline | Rust pipeline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clipped normal, n=10,000 | 111.68 ms | 51.76 ms | 2.16x | 124.02 ms | 65.83 ms |
| 16 values, n=10,000 | 1.56 ms | 2.03 ms | 0.77x | 12.44 ms | 13.33 ms |
| Frequency table, n=1,000,000, K=16 | 1.30 ms | 1.78 ms | 0.73x | 12.90 ms | 13.35 ms |

The continuous-data pipeline speedup was 1.88x. NumPy was faster for both
count-sampling workloads. Backend selection remains explicit, with NumPy the
default. These are measurements of this machine and these workloads, not a
general Rust speedup claim or an update to the separate R comparison.

Release timing includes Python validation, clipping, unique aggregation where
applicable, sampling, input copies across the native boundary, RNG setup and
Gaussian perturbation. The memory budget is the default 32 MiB and the raw-data
method is `auto` for both backends. Pipeline timing starts at the same point and
also includes default deconvolution, pointwise statistics and full 1,000-point
mass covariance output. All timed fits converged.

Inputs are public synthetic data generated once with seed 2026. Both backends
receive the same input for every repetition; they use independent algorithmic
random streams from experiment seeds 1 through 20. Backend order alternates
after a warmup. Calibration, input generation, imports/startup and file I/O
are excluded. The same asymptotic calibration is used for both backends; this
experiment does not establish finite privacy or interval coverage. Timings are
not compared with R and should not be combined with earlier speedup ratios.

Raw individual timings: [`rust_sampling.json`](rust_sampling.json). Reproduce
from the repository root with Rust and native build tools available:

```powershell
python -m pip install ./rust
python -m benchmarks.rust_sampling --repeats 20
```

## Validation and limits

The Python suite passed 42 tests with the native extension installed. New checks
compare both native kernels with the full exact Bernoulli bootstrap law, and
the count kernel with an enumerated three-category multinomial law. Additional
checks cover continuous-data moments, clipping and Gaussian variance, seeded
repeatability, raw/preaggregated equivalence, calibration mismatch rejection,
inference integration, strided inputs and missing-extension behavior. Existing
R golden fixtures still pass because inference is unchanged.

Two standalone Rust unit tests cover repeatability, degenerate frequency tables,
and invalid inputs, including integer overflow. Rust formatting and Clippy checks
also pass. Distribution tests are empirical checks, not a proof or an audit of
the pseudorandom generators. Floating-point sampling remains subject to the
research-software limitations documented in the main README.

Rust samples rows directly using unbiased uniform index generation. Frequency
tables use a sequence of conditional binomial draws, without expanding rows or
using Poisson/normal approximations. Table totals above 2**53 are rejected to
keep integer totals representable in binary64. Sampling returns only means to
the Python wrapper, which adds noise before constructing a release. Direct
calls to the native module or Rust crate return confidential noiseless outputs.

The implementation is sequential and releases the Python GIL during native
work. Input copies are made before releasing it; they count toward total process
memory but are outside `memory_bytes`, as are output and unique preprocessing.
No peak-memory benchmark or parallel scaling claim is made here.
