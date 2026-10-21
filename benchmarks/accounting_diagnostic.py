"""Separate upstream density interpolation from recursive integration error."""
import json
from pathlib import Path
import time
import warnings
import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from benchmarks.upstream_accounting import load_reference
from dp_boot import bootstrap_privacy_profile


def interpolated_loss_cdf(loss, q, geometry, moments=False):
    err = geometry['err']
    nodes = np.unique(np.r_[err, geometry['subsampled_alphas'],
                            geometry['subsampled_alphas2'], 1-err])
    nodes = nodes[(nodes >= err) & (nodes <= 1-err)]
    values = np.array([q(float(x)) for x in nodes])
    values[-1] = q(float(np.nextafter(1-err, 0.)))
    threshold = np.exp(loss)
    cdf = np.zeros_like(loss)
    first_moment = np.zeros_like(loss)
    for left, right, a, b in zip(nodes[:-1], nodes[1:], values[:-1], values[1:]):
        low, high = sorted([a, b])
        if high-low < 1e-14:
            cdf += (threshold >= high) * (right-left)*high
            first_moment += (threshold >= high) * (right-left)*high*np.log(high)
        else:
            clipped = np.clip(threshold, low, high)
            cdf += (right-left)*(clipped-low)*(clipped+low)/(2*(high-low))
            primitive = lambda a: a*a*(.5*np.log(a)-.25)
            first_moment += (right-left)*(primitive(clipped)-primitive(low))/(high-low)
    total = float(np.sum(np.diff(nodes)*(values[:-1]+values[1:])/2))
    return (cdf, first_moment, total) if moments else (cdf, total)


def main():
    reference, factory = load_reference(Path('reference/upstream/dpboot_composition.py'), geometry=True)
    n, b, bins = 5, 5, 1001
    q, logratio, geometry = factory(n, b, bins)
    stored = json.loads(Path('benchmarks/upstream_accounting_fine.json').read_text())
    eps = np.asarray(stored['epsilon'])
    baseline = np.asarray(stored['upstream_delta'])
    result = dict(n=n, replicates=b, upstream_bins=bins, cases=[])
    for step in [.001, .0005]:
        start = time.perf_counter()
        j = int(12/step)
        grid = np.arange(-j, j+1)*step
        cdf, total_mass = interpolated_loss_cdf(grid, q, geometry)
        pmf = np.maximum(0., np.diff(cdf, prepend=0.))
        length = b*(len(pmf)-1)+1
        size = next_fast_len(length)
        composed = np.maximum(0., irfft(rfft(pmf,size)**b,size)[:length])
        losses = (np.arange(length)-b*j)*step
        values = np.array([composed[losses>e] @ (-np.expm1(e-losses[losses>e])) for e in eps])
        analytic = bootstrap_privacy_profile(n, 1/np.sqrt(b), b, eps, loss_step=step, max_loss=12.)
        result['cases'].append(dict(step=step, interpolated_q_mass=total_mass,
                                    seconds=time.perf_counter()-start,
                                    analytic_vs_interpolated_fft_max_error=float(np.max(np.abs(analytic.delta-values))),
                                    upstream_recursion_vs_same_density_fft_max_error=float(np.max(np.abs(baseline-values))),
                                    epsilon=eps.tolist(), same_density_fft_delta=values.tolist()))
    # Same original recurrence but explicit domain fixes and larger quadrature
    # limit. Density and epsilon interpolation remain unchanged.
    import scipy.integrate
    original = scipy.integrate.quad
    def controlled(func, left, right):
        return original(func, 1e-10, 1-1e-10, epsabs=1e-9, epsrel=1e-7, limit=500)
    epsilon = np.linspace(-6,10,bins)
    with warnings.catch_warnings(record=True) as caught:
        scipy.integrate.quad = controlled
        try:
            start = time.perf_counter()
            _, _, corrected = reference(q, logratio, b, epsilon, np.array([.5]))
            elapsed = time.perf_counter()-start
        finally:
            scipy.integrate.quad = original
    result['controlled_recursion'] = dict(seconds=elapsed,
        max_error_vs_same_density_fft=float(np.max(np.abs(corrected[epsilon>=0]-values))),
        warnings=[str(w.message) for w in caught], delta=corrected[epsilon>=0].tolist())
    Path('benchmarks/accounting_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='cases' and k!='controlled_recursion'})
    for case in result['cases']:
        print({k:v for k,v in case.items() if k not in ['epsilon','same_density_fft_delta']})
    print({k:v for k,v in result['controlled_recursion'].items() if k not in ['delta','warnings']})


if __name__ == '__main__':
    main()
