"""Fresh-process memory and time for reference-compatible low-rank inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import psutil
from dp_boot import Release,asymptotic_calibration,deconvolve


def worker(grid_size):
    folder=Path('tests/fixtures/deconvolver')
    y=np.loadtxt(folder/'release.csv')
    sigma=float(np.loadtxt(folder/'sigma.csv'))
    c=asymptotic_calibration(sigma/np.sqrt((2-2/np.e)*len(y)),len(y),1.)
    start=time.perf_counter()
    fit=deconvolve(Release(y,c),grid_size=grid_size)
    elapsed=time.perf_counter()-start
    info=psutil.Process().memory_info()
    peak=getattr(info,'peak_wset',None)
    return dict(grid_size=grid_size,seconds=elapsed,converged=fit.converged,
                 peak_process_working_set_bytes=peak,
                 hypothetical_single_dense_covariance_bytes=8*grid_size**2,
                 notes='Process peak includes Python/SciPy baseline; no full covariance materialized.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=int)
    args=parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker)))
        return
    rows=[]
    for grid in [1000,4000,16000,64000]:
        output=subprocess.check_output([sys.executable,'-m','benchmarks.deconvolution_scaling',
                                        '--worker',str(grid)],text=True)
        rows.append(json.loads(output))
    Path('benchmarks/deconvolution_scaling.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
