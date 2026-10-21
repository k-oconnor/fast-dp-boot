"""Interleaved R/NumPy/Rust-backed full-pipeline comparison on shared data.

The Rust entry is the existing Python pipeline with native mean sampling.
Preserves prior benchmarks and saves a separate result artifact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from dp_boot import Release, asymptotic_calibration, bootstrap_mean, deconvolve


REFERENCE_COMMIT = "fd001561ed338d47143c94b995f4d47478ab9eb0"


def run(trials, rscript, folder):
    import dp_boot_native  # Fail before starting R if the extension is absent.

    commit = subprocess.check_output(
        ["git", "-C", "reference/upstream", "rev-parse", "HEAD"], text=True,
    ).strip()
    if commit != REFERENCE_COMMIT:
        raise RuntimeError(f"unexpected upstream reference commit: {commit}")
    folder.mkdir(parents=True, exist_ok=True)
    data = np.clip(np.random.default_rng(20261022).normal(.5, 1., (10000, trials)), 0, 1)
    np.savetxt(folder / "inputs.csv", data, delimiter=",")
    n, b = len(data), 2000
    calibration = asymptotic_calibration(1/n, b, np.sqrt(2-2/np.e) / 1.125)
    memory_bytes = 8 * 1024**2

    def python_pipeline(backend, i, seed):
        start = time.perf_counter()
        release = bootstrap_mean(data[:, i], bounds=(0, 1), replicates=b,
                                 calibration=calibration, memory_bytes=memory_bytes,
                                 rng=seed, backend=backend)
        sampled = time.perf_counter()
        fit = deconvolve(release)
        covariance = fit.mass_covariance
        end = time.perf_counter()
        if not fit.converged:
            raise RuntimeError(f"{backend} trial {i+1} did not converge")
        np.testing.assert_allclose(np.diag(covariance), fit.standard_error**2, atol=1e-14)
        interval = fit.interval(alpha=.1)
        return dict(release_seconds=sampled-start, total_seconds=end-start,
                    lower=float(interval[0]), upper=float(interval[1]), converged=True)

    rows = []
    process = subprocess.Popen(
        [rscript, "--vanilla", "benchmarks/matched_rust_pipeline.R", str(folder)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
    )
    try:
        ready = process.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"R reference failed to initialize: {ready!r}")
        for backend in ("numpy", "rust"):
            python_pipeline(backend, 0, 0)
        for i in range(trials):
            order = ["r", "numpy", "rust"]
            order = order[i % 3:] + order[:i % 3]
            row = dict(trial=i+1, order=order)
            for backend in order:
                if backend == "r":
                    process.stdin.write(f"{i+1}\n")
                    process.stdin.flush()
                    fields = process.stdout.readline().strip().split(",")
                    if len(fields) != 7 or fields[:2] != ["RESULT", str(i+1)]:
                        raise RuntimeError(f"invalid R response: {fields!r}")
                    row[backend] = dict(zip(
                        ("release_seconds", "total_seconds", "lower", "upper"),
                        map(float, fields[2:6]),
                    ))
                    row[backend]["nonfinite_cdf_se"] = int(fields[6])
                else:
                    row[backend] = python_pipeline(backend, i, i+1)
            rows.append(row)
            if (i+1) % 5 == 0 or i+1 == trials:
                print(f"Completed {i+1}/{trials} shared datasets", flush=True)
        process.stdin.write("STOP\n")
        process.stdin.flush()
        process.stdin.close()
        if process.wait(timeout=30) != 0:
            raise RuntimeError("R reference failed while saving releases")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        process.stdout.close()

    # Independent streams need not yield matching endpoints. Compare inference
    # on exactly the same R releases, without including this in pipeline timing.
    r_releases = np.loadtxt(folder / "r_releases.csv", delimiter=",", ndmin=2)
    endpoint_errors = []
    for i, row in enumerate(rows):
        fit = deconvolve(Release(r_releases[:, i], calibration))
        if not fit.converged:
            raise RuntimeError(f"shared-release validation {i+1} did not converge")
        endpoints = fit.interval(alpha=.1)
        error = float(np.max(np.abs(endpoints - [row["r"]["lower"], row["r"]["upper"]])))
        endpoint_errors.append(error)
    if max(endpoint_errors) > 1e-10:
        raise RuntimeError(f"shared-release interval mismatch: {max(endpoint_errors)}")

    medians = {backend: {stage: float(np.median([row[backend][stage] for row in rows]))
                         for stage in ("release_seconds", "total_seconds")}
               for backend in ("r", "numpy", "rust")}
    return dict(
        measured_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         numpy=version("numpy"), scipy=version("scipy"),
                         native=version("fast-dp-boot-native"),
                         native_binary_sha256=hashlib.sha256(Path(dp_boot_native.__file__).read_bytes()).hexdigest(),
                         cargo_lock_sha256=hashlib.sha256(Path("rust/Cargo.lock").read_bytes()).hexdigest(),
                         r=(folder / "r_environment.txt").read_text().splitlines()),
        reference_commit=commit, n=n, replicates=b, trials=trials, workers=1,
        input_distribution="clip(Normal(0.5, 1.0), 0, 1)", input_seed=20261022,
        input_csv_sha256=hashlib.sha256((folder / "inputs.csv").read_bytes()).hexdigest(),
        memory_bytes=memory_bytes, method="auto", noise_multiplier=1.125,
        median_seconds=medians,
        speedup={"r_over_numpy": medians["r"]["total_seconds"] / medians["numpy"]["total_seconds"],
                 "r_over_rust": medians["r"]["total_seconds"] / medians["rust"]["total_seconds"],
                 "numpy_over_rust": medians["numpy"]["total_seconds"] / medians["rust"]["total_seconds"]},
        shared_release_validation=dict(trials=trials,
                                       max_interval_endpoint_error=max(endpoint_errors)),
        reference_diagnostics=dict(
            nonfinite_cdf_standard_errors=sum(row["r"]["nonfinite_cdf_se"] for row in rows),
            affected_trials=[row["trial"] for row in rows if row["r"]["nonfinite_cdf_se"]],
            note="deconvolveR can produce NaN CDF SE at CDF=1 from cancellation. "
                 "All fitted masses, CDFs, mass SEs, biases and full mass covariances "
                 "were checked finite; the reference computation was not modified.",
        ),
        rows=rows,
        notes="Same input arrays and calibration for all three implementations. "
              "Independent R/NumPy/ChaCha20 sampling streams. One untimed warmup per "
              "implementation; cyclic execution order per dataset in persistent processes. "
              "Release includes sampling and Gaussian noise; full pipeline also includes "
              "reference-compatible inference, pointwise uncertainty and eager full mass "
              "covariance. Input loading, process startup, I/O, calibration, interval "
              "extraction and validation excluded. Rust is Python with native sampling, "
              "not an all-Rust pipeline. Local timings; no coverage or privacy certification.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--rscript", default=shutil.which("Rscript") or
                        "C:/Program Files/R/R-4.6.0/bin/Rscript.exe")
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/matched_rust_pipeline.json"))
    parser.add_argument("--inputs", type=Path,
                        default=Path("reference/benchmark_inputs/matched_rust"))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    result = run(args.trials, args.rscript, args.inputs)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("median_seconds", "speedup", "shared_release_validation")}, indent=2))
