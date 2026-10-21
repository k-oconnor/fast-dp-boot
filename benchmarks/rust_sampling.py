"""Single-worker NumPy/Rust comparison on public synthetic data only.

Includes Python validation, clipping/aggregation, native copies, RNG setup and
Gaussian noise. Pipeline timings also include default deconvolution and full
mass covariance. Excludes calibration, input generation, startup and file I/O.
Random streams differ across backends; this is not an estimator-parity test.
"""
import argparse
from importlib.metadata import version
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from dp_boot import (
    asymptotic_calibration, bootstrap_mean, bootstrap_mean_from_counts, deconvolve,
)


def run(repeats):
    rng = np.random.default_rng(2026)
    workloads = [
        ("clipped_normal", np.clip(rng.normal(.5, .2, 10000), 0, 1), None),
        ("16_values", rng.integers(0, 16, 10000) / 15, None),
        ("preaggregated_million", np.arange(16) / 15, np.full(16, 62500)),
    ]
    results = []
    for name, data, counts in workloads:
        n = len(data) if counts is None else int(counts.sum())
        calibration = asymptotic_calibration(1/n, 2000, 1.)
        timings = {backend: {stage: [] for stage in ("release", "pipeline")}
                   for backend in ("numpy", "rust")}

        def release(backend, seed):
            args = dict(bounds=(0, 1), replicates=2000, calibration=calibration,
                        backend=backend, rng=seed)
            if counts is None:
                return bootstrap_mean(data, **args)
            return bootstrap_mean_from_counts(data, counts, **args)

        # Warm imports/allocators, then alternate order to reduce order bias.
        for backend in timings:
            fit = deconvolve(release(backend, 0))
            if not fit.converged:
                raise RuntimeError("warmup fit did not converge")
            _ = fit.mass_covariance
        for i in range(repeats):
            for backend in (("numpy", "rust") if i % 2 == 0 else ("rust", "numpy")):
                start = perf_counter()
                output = release(backend, i + 1)
                sampled = perf_counter()
                fit = deconvolve(output)
                if not fit.converged:
                    raise RuntimeError(f"{name}/{backend} fit did not converge")
                _ = fit.mass_covariance
                end = perf_counter()
                timings[backend]["release"].append(sampled - start)
                timings[backend]["pipeline"].append(end - start)
        medians = {backend: {stage: float(np.median(samples))
                             for stage, samples in stages.items()}
                   for backend, stages in timings.items()}
        results.append(dict(workload=name, n=n, replicates=2000,
                            median_seconds=medians, samples_seconds=timings,
                            speedup={stage: medians["numpy"][stage] / medians["rust"][stage]
                                     for stage in ("release", "pipeline")}))
    return dict(environment=dict(python=platform.python_version(), platform=platform.platform(),
                                 numpy=version("numpy"), scipy=version("scipy"),
                                 native=version("fast-dp-boot-native")),
                repeats=repeats, workers=1, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/rust_sampling.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run(args.repeats)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in result["results"]:
        print(row["workload"], row["speedup"])
