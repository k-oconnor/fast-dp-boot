"""Independent exact segment integration of upstream's interpolated recurrence.

If delta_prev is piecewise linear in epsilon, its integral on every linear-q
segment is determined by integrals of q and q*log(q), both elementary. This
avoids adaptive quadrature while preserving the reference interpolation rule.
Direct convolution (not FFT) applies those segment-integrated weights.
"""
import json
from pathlib import Path
import time
import numpy as np
from benchmarks.upstream_accounting import load_reference
from benchmarks.accounting_diagnostic import interpolated_loss_cdf
from dp_boot import bootstrap_privacy_profile


def segment_recurrence(q, geometry, replicates, epsilon, max_loss=12.):
    step = epsilon[1]-epsilon[0]
    j = int(np.ceil(max_loss/step))
    grid = np.arange(-j,j+1)*step
    cdf, moment, total = interpolated_loss_cdf(grid,q,geometry,moments=True)
    mass = np.diff(cdf)
    fraction = np.clip((np.diff(moment)-grid[:-1]*mass)/step,0.,np.maximum(0.,mass))
    kernel = np.zeros(len(grid))
    kernel[:-1] += np.maximum(0.,mass)-fraction
    kernel[1:] += fraction
    previous = np.maximum(-np.expm1(epsilon),0.)
    for _ in range(replicates):
        padded = np.r_[np.ones(j),previous,np.zeros(j)]
        previous = np.convolve(padded,kernel,mode='valid')
    return previous,total


def main():
    _,factory=load_reference(Path('reference/upstream/dpboot_composition.py'),geometry=True)
    n,b=5,5
    q,_,geometry=factory(n,b,1001)
    stored=json.loads(Path('benchmarks/upstream_accounting_fine.json').read_text())
    target=np.asarray(stored['epsilon'])
    upstream=np.asarray(stored['upstream_delta'])
    cases=[]
    # Wider epsilon domain prevents the reference's hard boundary extrapolation
    # from contaminating the nonnegative comparison range.
    for h in [.016,.008,.004]:
        eps=np.arange(-12.,16.+h/2,h)
        start=time.perf_counter()
        values,total=segment_recurrence(q,geometry,b,eps)
        selected=np.interp(target,eps,values)
        elapsed=time.perf_counter()-start
        cases.append(dict(epsilon_step=h,seconds=elapsed,q_mass=total,delta=selected.tolist(),
                          upstream_quad_max_difference=float(np.max(np.abs(selected-upstream)))))
    fft=bootstrap_privacy_profile(n,1/np.sqrt(b),b,target,loss_step=.0005,max_loss=12.)
    result=dict(n=n,replicates=b,epsilon=target.tolist(),cases=cases,
                 refinement_max_difference=float(np.max(np.abs(np.array(cases[-1]['delta'])-cases[-2]['delta']))),
                 native_upper_fft_vs_segment_reference_max_difference=float(np.max(np.abs(fft.delta-selected))),
                 notes="Independent closed-form q and q*log(q) segment integration; direct convolution. "
                       "Matches upstream interpolated q; no adaptive quadrature or FFT in reference. "
                       "Native FFT rounds losses upward so it intentionally has finite epsilon slack.")
    Path('benchmarks/accounting_reference.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k not in ['cases','epsilon']})
    for c in cases:
        print({k:v for k,v in c.items() if k!='delta'})


if __name__=='__main__':
    main()
