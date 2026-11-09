"""Small bounded polynomial potential and support-aware Gaussian integration."""
from itertools import product
import numpy as np
from scipy.special import logsumexp
from latent_reference import real,support,nonnegative


class BoundedPotential:
    def __init__(self,rank,kappa):
        if isinstance(rank,bool) or not isinstance(rank,int) or not 1<=rank<=4:raise ValueError('Rank integer1..4 required')
        self.rank=rank;self.kappa=nonnegative(kappa,'amplitude');self.parameter_count=rank*2+rank*(rank-1)//2

    def features(self,z):
        z=real(z,'z',2)
        if z.shape[1]!=self.rank:raise ValueError('Potential latent dimensions mismatch')
        with np.errstate(over='raise',invalid='raise'):
            return np.column_stack([*[z[:,j] for j in range(self.rank)],
                *[(z[:,j]**2-1)/np.sqrt(2) for j in range(self.rank)],
                *[z[:,j]*z[:,k] for j in range(self.rank) for k in range(j+1,self.rank)]])

    def evaluate_features(self,theta,features):
        t=real(theta,'theta',1);f=real(features,'features',2)
        if len(t)!=self.parameter_count or f.shape[1]!=len(t):raise ValueError('Potential coefficient dimensions mismatch')
        if self.kappa==0:return np.zeros(len(f)),np.zeros_like(f)
        v=np.tanh(f@t)
        return self.kappa*v,self.kappa*(1-v*v)[:,None]*f

    def evaluate(self,theta,z):return self.evaluate_features(theta,self.features(z))


class GaussianExpectation:
    def __init__(self,mean,covariance,potential,*,order=32,max_nodes=65536):
        self.mean=real(mean,'posterior mean',1);c=real(covariance,'posterior covariance',2)
        if len(self.mean)!=potential.rank or c.shape!=(len(self.mean),len(self.mean)):raise ValueError('Posterior/potential dimensions mismatch')
        if isinstance(order,bool) or not isinstance(order,int) or not 2<=order<=256:raise ValueError('Quadrature order integer2..256 required')
        if isinstance(max_nodes,bool) or not isinstance(max_nodes,int) or max_nodes<1:raise ValueError('Positive integer node budget required')
        s=support(c,parent_scale=1.);r=s.factor.shape[1]
        if order**r>max_nodes:raise ValueError('Fixed quadrature node budget exceeded; no hidden approximation change')
        if not r:points=self.mean[None,:];logw=np.zeros(1)
        else:
            x,w=np.polynomial.hermite.hermgauss(order);grid=np.array(list(product(range(order),repeat=r)),dtype=int)
            points=self.mean+np.sqrt(2)*x[grid]@s.factor.T;logw=np.log(w[grid]/np.sqrt(np.pi)).sum(1)
        self.features=potential.features(points);self.logweights=logw-logsumexp(logw);self.potential=potential
        self.rank=r;self.order=order;self.nodes=len(points);self.support=s

    def evaluate(self,theta):
        energy,gradient=self.potential.evaluate_features(theta,self.features)
        if self.potential.kappa==0:return 0.,np.zeros(self.potential.parameter_count)
        logz=logsumexp(self.logweights-energy);weights=np.exp(self.logweights-energy-logz)
        return float(logz),-weights@gradient
