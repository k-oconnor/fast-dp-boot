"""Shared-input comparison with the actual authors' boot function in R."""
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from dp_boot.bootstrap import _mean_counts, _mean_draws


def main():
    folder = Path("reference/benchmark_inputs")
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    cases = [dict(case="paper_clipped_normal", x=np.clip(rng.normal(.5, 1, 10000), 0, 1), b=2000),
             dict(case="categorical_16", x=rng.integers(0, 16, 100000)/15, b=256)]
    lines = ["case,file,replicates"]
    for case in cases:
        path = folder / (case["case"]+".csv")
        np.savetxt(path, case["x"], fmt="%.17g")
        lines.append(f'{case["case"]},{path.as_posix()},{case["b"]}')
    (folder/"cases.csv").write_text("\n".join(lines)+"\n")
    subprocess.run(["C:/Program Files/R/R-4.6.0/bin/Rscript.exe", "--vanilla",
                    "benchmarks/upstream_bootstrap.R"], check=True)
    r_results = np.genfromtxt(folder/"r_bootstrap.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    results = []
    for case, r in zip(cases, r_results):
        times = []
        for _ in range(3):
            start = time.perf_counter()
            estimates = _mean_counts(case["x"], case["b"], np.random.default_rng(42), 8*1024**2, auto=True)
            times.append(time.perf_counter()-start)
        elapsed = float(np.median(times))
        results.append(dict(case=case["case"], n=len(case["x"]), replicates=case["b"],
                            unique_values=len(np.unique(case["x"])),
                            r_boot_seconds=float(r["seconds"]), python_seconds=elapsed,
                            speedup=float(r["seconds"])/elapsed,
                            expected_mean=float(case["x"].mean()),
                            expected_variance=float(case["x"].var()/len(case["x"])),
                            r_estimate_mean=float(r["estimate_mean"]),
                            python_estimate_mean=float(estimates.mean()),
                            r_estimate_variance=float(r["estimate_variance"]),
                            python_estimate_variance=float(estimates.var(ddof=1))))
    result = dict(upstream_commit=subprocess.check_output(
        ["git", "-C", "reference/upstream", "rev-parse", "HEAD"], text=True).strip(),
        r_version="4.6.0", cases=results,
        notes="Shared input arrays; median of three in-process kernel timings. "
              "Authors' func_boot/func_param extracted unchanged from clamp_normal_mean.r. "
              "Excludes process startup, CSV I/O, Gaussian perturbation, deconvolution and parallel workers. "
              "Different random generators; distribution agreement, not seed parity.")
    Path("benchmarks/upstream_bootstrap.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
