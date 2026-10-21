"""Shared-data end-to-end bootstrap + noise + matched inference comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from dp_boot import bootstrap_mean,asymptotic_calibration,deconvolve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trials',type=int,default=20)
    args=parser.parse_args()
    folder=Path('reference/benchmark_inputs')
    folder.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261022)
    data=np.clip(rng.normal(.5,1.,(10000,args.trials)),0,1)
    np.savetxt(folder/'pipeline_inputs.csv',data,delimiter=',')
    subprocess.run(['C:/Program Files/R/R-4.6.0/bin/Rscript.exe','--vanilla',
                    'benchmarks/matched_pipeline.R'],check=True)
    baseline=np.genfromtxt(folder/'r_pipeline.csv',delimiter=',',names=True)
    n,b=len(data),2000
    calibration=asymptotic_calibration(1/n,b,np.sqrt(2-2/np.e)/1.125)
    rows=[]
    for i in range(args.trials):
        start=time.perf_counter()
        release=bootstrap_mean(data[:,i],bounds=(0,1),replicates=b,
                               calibration=calibration,memory_bytes=8*1024**2,rng=i+1)
        bootstrap_elapsed=time.perf_counter()-start
        fit=deconvolve(release)
        # Equal eager full covariance output, although inference only needs G.
        covariance=fit.mass_covariance
        elapsed=time.perf_counter()-start
        rows.append(dict(trial=i+1,r_total_seconds=float(baseline[i]['total_seconds']),
                         r_bootstrap_seconds=float(baseline[i]['bootstrap_seconds']),
                         python_total_seconds=elapsed,python_bootstrap_seconds=bootstrap_elapsed,
                         converged=fit.converged))
    r_median=float(np.median([r['r_total_seconds'] for r in rows]))
    py_median=float(np.median([r['python_total_seconds'] for r in rows]))
    result=dict(n=n,replicates=b,trials=args.trials,noise_multiplier=1.125,
                 r_median_seconds=r_median,python_median_seconds=py_median,
                 median_speedup=r_median/py_median,rows=rows,
                 notes="Identical clipped-normal input data; independent R/NumPy randomness. "
                       "Includes bootstrap, Gaussian perturbation, inference, pointwise uncertainty, "
                       "and eager full mass covariance. Excludes CSV I/O, process startup, plots "
                       "and experiment orchestration. One worker; medians across datasets. "
                       "Estimator fidelity separately verified on identical private releases.")
    Path('benchmarks/matched_pipeline.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='rows'})


if __name__=='__main__':
    main()
