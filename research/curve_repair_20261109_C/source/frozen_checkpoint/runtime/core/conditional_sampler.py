"""Exact bounded-energy rejection reference; no premature layered MCMC engine."""
import numpy as np
from latent_reference import real


def sample(query,potential,theta,*,n,seed,max_proposals=100000):
    if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError('Positive integer sample count required')
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:raise ValueError('Nonnegative integer seed required')
    if isinstance(max_proposals,bool) or not isinstance(max_proposals,int) or max_proposals<n:raise ValueError('Explicit proposal budget>=n required')
    theta=real(theta,'theta',1)
    if len(theta)!=potential.parameter_count or potential.rank!=len(query.latent_mean):raise ValueError('Latent potential mismatch')
    rng=np.random.default_rng(seed);r=query.latent_support.factor.shape[1]
    accepted=[];proposals=0
    while len(accepted)<n and proposals<max_proposals:
        count=min(max(n-len(accepted),64),max_proposals-proposals)
        z=query.latent_mean+rng.normal(size=(count,r))@query.latent_support.factor.T
        if potential.kappa==0 or r==0:mask=np.ones(count,dtype=bool)
        else:
            e,_=potential.evaluate(theta,z);mask=np.log(rng.uniform(size=count))<=-e-potential.kappa
        accepted.extend(z[mask]);proposals+=count
    if len(accepted)<n:raise RuntimeError(f'Rejection budget exhausted: accepted{len(accepted)}of{n}; no auto-resample or target change')
    z=np.asarray(accepted[:n]);target=query.decode(z,rng)
    return {'latent':z,'target':target,'targets':query.targets.copy(),'observed':query.observed.copy(),
       'observed_values_unchanged':query.values.copy(),'is_measured_data':False,
       'proposals':proposals,'acceptance_lower_bound':float(np.exp(-2*potential.kappa)),
       'sampler':'bounded rejection; exact target in exact arithmetic, finite budget raises'}
