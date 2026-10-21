"""Time reference-compatible fitting with eager full covariance as well."""
import json
from pathlib import Path
import time
import numpy as np
from dp_boot import Release, asymptotic_calibration, deconvolve


def main():
    original = json.loads(Path('benchmarks/matched_deconvolution.json').read_text())
    y = np.loadtxt('reference/benchmark_inputs/noisy_releases.csv', delimiter=',')
    sigma = float(np.loadtxt('reference/benchmark_inputs/noise_sigma.csv'))
    assert y.shape[1] == original['trials']
    c = asymptotic_calibration(sigma/np.sqrt((2-2/np.e)*len(y)), len(y), 1.)
    rows = []
    for i in range(y.shape[1]):
        start = time.perf_counter()
        fit = deconvolve(Release(y[:,i], c))
        fit_seconds = time.perf_counter()-start
        start = time.perf_counter()
        full = fit.mass_covariance
        covariance_seconds = time.perf_counter()-start
        np.testing.assert_allclose(np.diag(full),fit.standard_error**2,atol=1e-14)
        rows.append(dict(trial=i+1, pointwise_seconds=fit_seconds,
                         full_covariance_extra_seconds=covariance_seconds,
                         total_seconds=fit_seconds+covariance_seconds))
    baseline = original['summary']['r_median_seconds']
    median = float(np.median([r['total_seconds'] for r in rows]))
    result = dict(trials=len(rows), r_median_seconds=baseline,
                   python_median_seconds_including_full_covariance=median,
                   speedup=baseline/median, rows=rows,
                   notes="Reuses baseline and shared releases from matched_deconvolution. "
                         "Python also eagerly materializes the full 1000x1000 mass covariance. "
                         "CDF SEs use cumulative rank-p derivatives instead of dense cumulative covariance.")
    Path('benchmarks/matched_statistics.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})


if __name__ == '__main__':
    main()
