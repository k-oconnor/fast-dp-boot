"""Run from repository root: python -m benchmarks.scaling --output benchmarks/results.json"""
import argparse
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.signal import fftconvolve
from scipy.linalg import toeplitz
from scipy.fft import rfft, irfft, next_fast_len
from dp_boot.bootstrap import _mean_counts, _mean_draws
from dp_boot.accounting import _groups, _loss_cdf


def timed(fn, repeat=3):
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args()
    b, budget = 256, 8 * 1024**2
    rows = []
    for n in [10000, 100000, 1000000]:
        x = np.random.default_rng(1).integers(0, 16, size=n).astype(float) / 15
        def naive():
            rng = np.random.default_rng(7)
            return np.array([x[rng.integers(0, n, n)].mean() for _ in range(b)])
        baseline = timed(naive)
        counts = timed(lambda: _mean_counts(x, b, np.random.default_rng(7), budget))
        draws = timed(lambda: _mean_draws(x, b, np.random.default_rng(7), budget))
        rows.append(dict(n=n, replicates=b, unique_values=16,
                         naive_seconds=baseline, counts_seconds=counts,
                         tiled_draws_seconds=draws, counts_speedup=baseline/counts,
                         tiled_scratch_budget_bytes=budget,
                         full_matrix_scratch_bytes=16*n*b))
    # Continuous data are a separate regime; counts have no support compression.
    x = np.random.default_rng(2).random(100000)
    continuous = dict(n=len(x), replicates=b,
                      draws_seconds=timed(lambda: _mean_draws(x, b, np.random.default_rng(7), budget)),
                      counts_seconds=timed(lambda: _mean_counts(x, b, np.random.default_rng(7), budget)))
    convolutions = []
    for m in [513, 1025, 4097]:
        kernel = np.exp(-.5 * ((np.arange(m)-m//2)/(m/20))**2)
        kernel /= kernel.sum()
        mass = np.ones(m) / m
        col = np.r_[kernel[m//2:], np.zeros(m//2)]
        dense = toeplitz(col)
        dense_s = timed(lambda: dense @ mass, repeat=10)
        fft_s = timed(lambda: fftconvolve(mass, kernel, mode="same"), repeat=10)
        convolutions.append(dict(grid_size=m, dense_seconds=dense_s, fft_seconds=fft_s,
                                 speedup=dense_s/fft_s, dense_kernel_bytes=dense.nbytes))
    compositions = []
    p0, scales, weights = _groups(10000, .1, 1e-14)
    cdf = _loss_cdf(np.arange(-100, 101)*.02, p0, scales, weights)
    pmf = np.maximum(0., np.diff(cdf, prepend=cdf[0]))
    for count in [32, 128, 512]:
        length = count*(len(pmf)-1)+1
        fft_size = next_fast_len(length)
        def direct():
            out = np.array([1.])
            for _ in range(count):
                out = np.convolve(out, pmf)
            return out
        def fft():
            return irfft(rfft(pmf, fft_size)**count, fft_size)[:length]
        np.testing.assert_allclose(direct(), fft(), atol=1e-12)
        direct_s = timed(direct)
        fft_s = timed(fft)
        compositions.append(dict(replicates=count, bins_per_release=len(pmf),
                                 recursive_direct_seconds=direct_s, fft_seconds=fft_s,
                                 speedup=direct_s/fft_s))
    result = dict(environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, platform=platform.platform(),
                                   processor=platform.processor()),
                  bootstrap=rows, continuous=continuous, convolution=convolutions,
                  privacy_composition=compositions,
                  notes="Median wall time; kernel benchmark includes unique preprocessing. "
                        "Memory numbers are analytical array sizes, not measured RSS. "
                        "Benchmarks exclude Gaussian noise shared by both approaches.")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
