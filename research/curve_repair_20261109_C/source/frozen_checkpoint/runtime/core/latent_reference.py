"""Frozen full Gaussian reference and a deterministic low-dimensional projection.

No distribution is fitted/compressed here. Singular conditional residuals are
sampled in their numerical support, never made positive definite by jitter.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import cho_solve


def real(a,name,ndim):
    if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(a,dtype=object).flat):raise ValueError(name+' cannot contain booleans')
    a=np.asarray(a)
    if a.dtype.kind not in 'iuf' or a.ndim!=ndim or not np.isfinite(a).all():raise ValueError(name+' must be finite real with declared shape')
    return np.array(a,dtype=float,copy=True)


def indices(a,n):
    if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(a,dtype=object).flat):raise ValueError('Boolean addresses are not integers')
    x=np.asarray(a)
    if x.ndim!=1 or (x.size and x.dtype.kind not in 'iu'):raise ValueError('Integer address vector required')
    x=x.astype(int)
    if len(set(x.tolist()))!=len(x) or (x<0).any() or (x>=n).any():raise ValueError('Unique in-range addresses required; aliases need provenance canonicalization')
    return x


def nonnegative(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<0:
        raise ValueError(name+' must be finite nonnegative real scalar')
    return float(value)


@dataclass(frozen=True)
class Support:
    factor:np.ndarray
    inverse:np.ndarray
    projector:np.ndarray
    eigenvalues:np.ndarray
    tolerance:float
    discarded_eigenvalue_max_abs:float


def support(covariance,*,parent_scale):
    c=real(covariance,'covariance',2)
    parent_scale=nonnegative(parent_scale,'parent scale')
    if c.shape[0]!=c.shape[1]:raise ValueError('Square covariance required')
    n=len(c)
    if not n:return Support(c.copy(),c.copy(),c.copy(),np.zeros(0),0.,0.)
    tol=128*np.finfo(float).eps*n*max(float(parent_scale),float(np.linalg.norm(c,ord=2)))
    if np.max(abs(c-c.T))>tol:raise ValueError('Asymmetric conditional covariance')
    eig,vec=np.linalg.eigh((c+c.T)/2)
    if eig.min() < -tol:raise ValueError('Genuinely negative conditional covariance; no jitter fallback')
    keep=eig>tol;v=vec[:,keep]
    return Support(v*np.sqrt(eig[keep]),(v/eig[keep])@v.T if keep.any() else np.zeros_like(c),
        v@v.T,eig,tol,float(np.max(abs(eig[~keep]))) if (~keep).any() else 0.)


class GaussianReference:
    def __init__(self,mean,covariance):
        self.mean=real(mean,'mean',1);self.covariance=real(covariance,'covariance',2);self.dimension=len(self.mean)
        if not self.dimension or self.covariance.shape!=(self.dimension,self.dimension):raise ValueError('Reference dimensions mismatch')
        tol=128*np.finfo(float).eps*self.dimension*np.linalg.norm(self.covariance,2)
        if np.max(abs(self.covariance-self.covariance.T))>tol:raise ValueError('Asymmetric reference covariance')
        self.covariance=(self.covariance+self.covariance.T)*.5
        self.root=np.linalg.cholesky(self.covariance)
        for a in (self.mean,self.covariance,self.root):a.flags.writeable=False

    def posterior(self,observed,values):
        o=indices(observed,self.dimension);y=real(values,'values',1)
        if y.shape!=o.shape:raise ValueError('Observed values/address mismatch')
        if not len(o):return self.mean.copy(),self.covariance.copy()
        c=self.covariance;L=np.linalg.cholesky(c[np.ix_(o,o)]);gain=cho_solve((L,True),c[o,:]).T
        mean=self.mean+gain@(y-self.mean[o]);cov=c-gain@c[o,:];cov=(cov+cov.T)*.5
        # Exact stored observation identities, not independent new measurements.
        mean[o]=y;cov[o,:]=0.;cov[:,o]=0.
        return mean,cov

    def logpdf(self,observed,values):
        o=indices(observed,self.dimension);y=real(values,'values',1)
        if y.shape!=o.shape:raise ValueError('Observed values/address mismatch')
        if not len(o):return 0.
        L=np.linalg.cholesky(self.covariance[np.ix_(o,o)]);d=np.linalg.solve(L,y-self.mean[o])
        return float(-.5*(len(o)*np.log(2*np.pi)+2*np.log(np.diag(L)).sum()+d@d))


class LatentProjection:
    def __init__(self,reference,support_indices,directions):
        if not isinstance(reference,GaussianReference):raise ValueError('GaussianReference required')
        self.reference=reference;self.q=indices(support_indices,reference.dimension)
        u=real(directions,'directions',2)
        if u.shape[0]!=len(self.q) or not 1<=u.shape[1]<=len(self.q) or not np.allclose(u.T@u,np.eye(u.shape[1]),rtol=0,atol=1e-10):
            raise ValueError('Orthonormal directions on fixed support required')
        L=np.linalg.cholesky(reference.covariance[np.ix_(self.q,self.q)])
        self.coefficient=np.zeros((u.shape[1],reference.dimension));self.coefficient[:,self.q]=np.linalg.solve(L.T,u).T
        self.rank=u.shape[1];self.z_prior=self.coefficient@reference.covariance@self.coefficient.T
        self.parent_scale=float(np.linalg.norm(self.z_prior,2))
        self.directions=u.copy()
        for a in (self.coefficient,self.z_prior,self.directions,self.q):a.flags.writeable=False

    def posterior(self,observed,values):
        mean,cov=self.reference.posterior(observed,values)
        m=self.coefficient@(mean-self.reference.mean);v=self.coefficient@cov@self.coefficient.T;v=(v+v.T)*.5
        return m,v,support(v,parent_scale=self.parent_scale)

    def query(self,observed,values,targets):
        o=indices(observed,self.reference.dimension);h=indices(targets,self.reference.dimension)
        if np.intersect1d(o,h).size:raise ValueError('Known and missing target addresses must be disjoint')
        mean,cov=self.reference.posterior(o,values);mz,vz,sz=self.posterior(o,values)
        cross=cov[h,:]@self.coefficient.T;gain=cross@sz.inverse
        residual=cov[np.ix_(h,h)]-gain@cross.T;residual=(residual+residual.T)*.5
        sr=support(residual,parent_scale=float(np.linalg.norm(self.reference.covariance,2)))
        return ConditionalQuery(mean[h],cov[np.ix_(h,h)],mz,vz,sz,gain,sr,h,o,real(values,'values',1))


@dataclass(frozen=True)
class ConditionalQuery:
    mean:np.ndarray
    covariance:np.ndarray
    latent_mean:np.ndarray
    latent_covariance:np.ndarray
    latent_support:Support
    decoder_gain:np.ndarray
    residual_support:Support
    targets:np.ndarray
    observed:np.ndarray
    values:np.ndarray

    def decode(self,z,rng):
        z=real(z,'z',2)
        if z.shape[1]!=len(self.latent_mean):raise ValueError('Latent dimension mismatch')
        delta=z-self.latent_mean
        outside=delta-delta@self.latent_support.projector
        tol=1e-10*max(1.,float(np.max(abs(delta))) if delta.size else 1.)
        if outside.size and np.max(abs(outside))>tol:raise ValueError('Latent candidate outside conditioned support')
        noise=rng.normal(size=(len(z),self.residual_support.factor.shape[1]))@self.residual_support.factor.T
        return self.mean+delta@self.decoder_gain.T+noise


def physical_directions(covariance_at_support,coordinate):
    """Fixed level/slope linear functionals, whitened in the reference metric."""
    t=real(coordinate,'coordinate',1);c=real(covariance_at_support,'support covariance',2)
    if len(t)<2 or c.shape!=(len(t),len(t)) or np.any(np.diff(t)<=0):raise ValueError('Ordered distinct physical coordinates required')
    weights=np.empty(len(t));weights[0]=(t[1]-t[0])/2;weights[-1]=(t[-1]-t[-2])/2
    if len(t)>2:weights[1:-1]=(t[2:]-t[:-2])/2
    weights/=weights.sum();a=np.c_[weights,weights*(t-np.sum(weights*t))]
    u,r=np.linalg.qr(np.linalg.cholesky(c).T@a)
    if np.min(abs(np.diag(r)))<=np.finfo(float).eps*np.linalg.norm(r):raise ValueError('Level/slope functionals lack independent support')
    return u*np.where(np.diag(r)>=0,1.,-1.)
