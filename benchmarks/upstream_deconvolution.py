"""Matched shared-release parity benchmark against the authors' R estimator."""
import json
from pathlib import Path
import subprocess
import time
import argparse
import numpy as np
from dp_boot import bootstrap_mean, asymptotic_calibration, deconvolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/matched_deconvolution.json")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--mu", type=float, default=1.)
    args = parser.parse_args()
    folder = Path("reference/benchmark_inputs")
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261021)
    n, b, trials = args.n, args.replicates, args.trials
    # Match upstream's rounded noise multiplier 1.125 exactly.
    c = asymptotic_calibration(1/n, b, args.mu*np.sqrt(2-2/np.e)/1.125)
    releases = [bootstrap_mean(np.clip(rng.normal(.5, 1., n), 0, 1), bounds=(0, 1),
                               replicates=b, calibration=c, rng=rng) for _ in range(trials)]
    np.savetxt(folder/"noisy_releases.csv", np.column_stack([r.estimates for r in releases]), delimiter=",")
    np.savetxt(folder/"noise_sigma.csv", [c.sigma])
    subprocess.run(["C:/Program Files/R/R-4.6.0/bin/Rscript.exe", "--vanilla",
                    "benchmarks/upstream_deconvolution.R"], check=True)
    r_rows = np.genfromtxt(folder/"r_deconvolution.csv", delimiter=",", names=True)
    rows = []
    for r, upstream in zip(releases, r_rows):
        start = time.perf_counter()
        fit = deconvolve(r)
        elapsed = time.perf_counter()-start
        interval = fit.interval(.1)
        rows.append(dict(trial=int(upstream["trial"]),
                         r_seconds=float(upstream["seconds"]), python_seconds=elapsed,
                         r_interval=[float(upstream["lower"]), float(upstream["upper"])],
                         python_interval=interval.tolist(), python_converged=fit.converged,
                         gradient_norm=fit.gradient_norm))
    summary = dict(trials=trials, n=n, replicates=b, mu=args.mu, seed=20261021,
                   r_version="4.6.0", deconvolveR_version="1.2-2",
                   r_likelihood_shape=[int(r_rows[0]["observed_bins"]), int(r_rows[0]["latent_points"])],
                   rows=rows,
                   summary=dict(r_median_seconds=float(np.median([r['r_seconds'] for r in rows])),
                                python_median_seconds=float(np.median([r['python_seconds'] for r in rows])),
                                max_interval_difference=float(max(np.max(np.abs(np.array(r['r_interval'])-r['python_interval'])) for r in rows)),
                                r_coverage=float(np.mean([r['r_interval'][0] <= .5 <= r['r_interval'][1] for r in rows])),
                                python_coverage=float(np.mean([r['python_interval'][0] <= .5 <= r['python_interval'][1] for r in rows])),
                                converged=sum(r['python_converged'] for r in rows)),
                   notes="Identical private releases, model, spline basis, bins, penalty, and grid brackets. "
                         "Both compute pointwise covariance-derived SEs and bias; Python retains low-rank "
                         "covariance rather than eagerly materializing the full mass covariance matrix. "
                         "Timings include fit and statistics; exclude process startup and CSV I/O. "
                         "Per-fit single timings across multiple inputs, not a production throughput test.")
    summary['summary']['median_speedup'] = summary['summary']['r_median_seconds']/summary['summary']['python_median_seconds']
    Path(args.output).write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
