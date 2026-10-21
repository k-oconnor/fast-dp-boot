"""Synthetic bounded-mean utility experiment; not a coverage guarantee."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from dp_boot import bootstrap_mean, asymptotic_calibration, asymptotic_interval, deconvolve, deconvolve_fft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--output", default="benchmarks/matched_coverage.json")
    parser.add_argument("--model", choices=["reference", "legacy"], default="reference")
    args = parser.parse_args()
    rng = np.random.default_rng(20261021)
    n, b, mu, alpha = 2000, 200, 1., .1
    calibration_mu = mu*np.sqrt(2-2/np.e)/1.125 if args.model == "reference" else mu
    c = asymptotic_calibration(1/n, b, calibration_mu)
    results = {key: [] for key in ["asymptotic", "deconvolution"]}
    converged = 0
    for _ in range(args.trials):
        # Symmetric clipping gives exact population mean .5.
        x = np.clip(rng.normal(.5, 1., n), 0, 1)
        release = bootstrap_mean(x, bounds=(0, 1), replicates=b, calibration=c, rng=rng)
        fit = deconvolve(release) if args.model == "reference" else deconvolve_fft(release, grid_size=513, spline_knots=25)
        converged += fit.converged
        for key, interval in [("asymptotic", asymptotic_interval(release, alpha)),
                              ("deconvolution", fit.interval(alpha))]:
            results[key].append([bool(interval[0] <= .5 <= interval[1]), float(np.diff(interval)[0])])
    summary = dict(trials=args.trials, n=n, replicates=b, mu=mu, nominal_coverage=1-alpha,
                   model=args.model, noise_multiplier=1.125 if args.model == "reference" else float(np.sqrt(2-2/np.e)),
                   seed=20261021, deconvolution_converged=converged,
                   privacy="paper asymptotic calibration; not a finite privacy guarantee")
    for key, values in results.items():
        a = np.asarray(values)
        coverage = a[:, 0].mean()
        ci = binomtest(int(a[:, 0].sum()), args.trials).proportion_ci()
        summary[key] = dict(coverage=float(coverage), mean_width=float(a[:, 1].mean()),
                            coverage_95_percent_interval=[float(ci.low), float(ci.high)])
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
