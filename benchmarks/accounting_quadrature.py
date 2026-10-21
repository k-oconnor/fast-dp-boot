"""Independent two-release check: exact one-release formula + adaptive integral.

No epsilon interpolation, density interpolation, PLD lattice, or FFT in the
reference. Split the continuous loss integral at zero and include its atom.
"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import binom,norm
from dp_boot import bootstrap_privacy_profile


def reference(n,mu):
    p0=binom.pmf(0,n,1/n)
    p=1-p0
    scales=np.arange(1,n+1)*mu
    weights=binom.pmf(np.arange(1,n+1),n,1/n)/p
    def one(eps):
        r=abs(eps)
        t=r+np.log1p(-p0*np.exp(-r))-np.log(p)
        a=weights @ norm.sf(t/scales+scales/2)
        d=weights @ norm.sf(t/scales-scales/2)
        delta=max(0.,p*d-(np.exp(r)-p0)*a)
        return delta if eps>=0 else -np.expm1(eps)+np.exp(eps)*delta
    atom=p0*(1-2*(weights @ norm.sf(scales/2)))
    def density(loss):
        r=abs(loss)
        t=r+np.log1p(-p0*np.exp(-r))-np.log(p)
        derivative=1/(1-p0*np.exp(-r))
        a=weights @ (norm.pdf(t/scales+scales/2)/scales)
        if loss<0:
            return a*derivative
        d=weights @ (norm.pdf(t/scales-scales/2)/scales)
        return (p0*a+p*d)*derivative
    def two(eps):
        integrand=lambda loss: one(eps-loss)*density(loss)
        left,err1=quad(integrand,-40,0,epsabs=1e-12,epsrel=1e-11,limit=300)
        right,err2=quad(integrand,0,40,epsabs=1e-12,epsrel=1e-11,limit=300)
        return atom*one(eps)+left+right,err1+err2
    return two


def main():
    eps=np.array([0.,.5,1.,2.,4.,6.])
    cases=[]
    for n,mu in [(5,1/np.sqrt(5)),(50,.3)]:
        exact=reference(n,mu)
        values=np.array([exact(e) for e in eps])
        profile=bootstrap_privacy_profile(n,mu,2,eps,loss_step=.0005,max_loss=20)
        assert np.all(profile.delta_lower<=values[:,0]+1e-10)
        assert np.all(values[:,0]<=profile.delta+1e-10)
        cases.append(dict(n=n,base_mu=mu,replicates=2,epsilon=eps.tolist(),
                          independent_delta=values[:,0].tolist(),quadrature_error=values[:,1].tolist(),
                          fft_lower=profile.delta_lower.tolist(),fft_upper=profile.delta.tolist()))
    result=dict(cases=cases,notes="Independent exact one-release formula integrated once for two releases. "
                 "Atom at loss zero included. Finite integration range [-40,40]; Gaussian tails negligible "
                 "for these settings. Quadrature errors are estimates, not interval certificates.")
    Path('benchmarks/accounting_quadrature.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
