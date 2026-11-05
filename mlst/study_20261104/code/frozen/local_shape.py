"""One fixed finite-scale curvature code; no material responses or fitting.

The GP may be Matern-3/2: finite differences of function values are defined
without assuming a classical second derivative of its random sample paths.
"""
import numpy as np


def _real(a,name,dimension):
    if any(isinstance(v,(bool,np.bool_)) for v in np.asarray(a,dtype=object).flat):
        raise ValueError(name+' cannot contain booleans')
    a=np.asarray(a)
    if a.ndim!=dimension or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise ValueError(name+' must be finite real with the declared dimension')
    return np.array(a,dtype=float,copy=True)


def local_curvature_functionals(coordinate):
    """Return F (2 x q): two predeclared smooth averages of discrete curvature.

    Centres are 1/3 and 2/3 of the fixed support range, halfwidth is range/3.
    Windows are (1-u^2)_+^3. No scan, learned window, or mask-dependent basis.
    """
    t=_real(coordinate,'coordinate',1)
    if len(t)<7 or (np.diff(t)<=0).any():
        raise ValueError('At least seven strictly increasing support coordinates required')
    left,right=np.diff(t)[:-1],np.diff(t)[1:]
    d=np.zeros((len(t)-2,len(t)))
    for i,(a,b) in enumerate(zip(left,right)):
        d[i,i:i+3]=[2/(a*(a+b)),-2/(a*b),2/(b*(a+b))]
    span=t[-1]-t[0];interior=t[1:-1];width=span/3
    centres=t[0]+span*np.array([1/3,2/3]);quadrature=(left+right)/2
    windows=np.maximum(1-((interior[None,:]-centres[:,None])/width)**2,0.)**3
    windows*=quadrature
    if np.any(windows.sum(1)<=0):raise ValueError('Fixed windows unsupported on this grid')
    windows/=windows.sum(1)[:,None]
    f=windows@d
    if np.linalg.matrix_rank(f)!=2:raise ValueError('The two fixed functionals lack independent support')
    return f


def whiten_functionals(covariance,functionals):
    """Prior-only symmetric whitening; frozen once for one frozen reference.

    z=P^-1/2 F(fQ-mQ)=U.T L^-1(fQ-mQ), P=F KQ F.T.
    No observation mask, observation value or hidden response is an input.
    """
    k=_real(covariance,'support covariance',2);f=_real(functionals,'functionals',2)
    if k.shape!=(f.shape[1],f.shape[1]) or f.shape[0]!=2:raise ValueError('q x q covariance and2 x q functionals required')
    tol=128*np.finfo(float).eps*len(k)*np.linalg.norm(k,2)
    if np.max(abs(k-k.T))>tol:raise ValueError('Asymmetric support covariance')
    k=(k+k.T)/2;l=np.linalg.cholesky(k)
    p=f@k@f.T;p=(p+p.T)/2;e,v=np.linalg.eigh(p)
    if e.min()<=128*np.finfo(float).eps*np.linalg.norm(p,2):
        raise ValueError('Functional covariance lacks stable positive rank; no jitter')
    invroot=(v/np.sqrt(e))@v.T;u=l.T@f.T@invroot
    if not np.allclose(u.T@u,np.eye(2),rtol=0,atol=1e-10):raise ValueError('Prior whitening failed')
    return {'directions':u,'functional_covariance':p,'functional_inverse_root':invroot,
            'frozen_functionals':f,'numerical_jitter':0.,'normalization':'fixed prior, never per query'}


def conditional_channel_statistics(vxi,cross,vtarget,directions):
    """Covariance-only diagnostics. J is never an automatic selection rule.

    cross=Cov(Y_H,xi|O). The joint target covariance must be SPD; conditional
    code covariance may have rank zero. No added noise or observed labels.
    """
    vx=_real(vxi,'conditional support covariance',2);c=_real(cross,'cross covariance',2)
    vh=_real(vtarget,'target covariance',2);u=_real(directions,'directions',2)
    h=len(vh);q=len(vx)
    if h<1 or vh.shape!=(h,h) or vx.shape!=(q,q) or c.shape!=(h,q) or u.shape!=(q,2):
        raise ValueError('Covariance dimensions mismatch')
    if not np.allclose(u.T@u,np.eye(2),rtol=0,atol=1e-10):raise ValueError('Orthonormal prior directions required')
    joint=np.block([[vh,c],[c.T,vx]]);scale=float(np.linalg.norm(joint,2))
    tol=256*np.finfo(float).eps*len(joint)*scale
    if np.max(abs(joint-joint.T))>tol:raise ValueError('Asymmetric joint covariance')
    if np.linalg.eigvalsh((joint+joint.T)/2).min() < -tol:raise ValueError('Invalid joint covariance')
    ly=np.linalg.cholesky((vh+vh.T)/2)
    m=u.T@vx@u;m=(m+m.T)/2;b=c@u;e,v=np.linalg.eigh(m)
    rank_tol=128*np.finfo(float).eps*q*float(np.linalg.norm(vx,2))
    if e.min() < -max(tol,rank_tol):raise ValueError('Invalid conditional code covariance')
    keep=e>rank_tol;basis=v[:,keep]
    if np.max(abs(b-b@(basis@basis.T)))>max(tol,rank_tol)*16:
        raise ValueError('Cross covariance outside stochastic code support')
    factor=b@basis/np.sqrt(e[keep]) if keep.any() else np.zeros((h,0))
    explained=factor@factor.T
    remainder=(vh+vh.T)/2-explained
    if np.linalg.eigvalsh(remainder).min() < -max(tol,rank_tol)*16:
        raise ValueError('Explained covariance exceeds target covariance')
    joint_score=float(np.sum(np.linalg.solve(ly,factor)**2)/h)
    bound=min(h,int(keep.sum()))/h
    if joint_score>bound+1e-8:raise ValueError('Normalized joint information bound failed')
    projected=np.linalg.solve(ly,b)
    remaining_code=m-projected.T@projected
    return {'J_joint_mean':joint_score,'J_dimension_upper_bound':bound,
        'rho_marginal':np.diag(explained)/np.diag(vh),
        'conditional_code_eigenvalues':e,'conditional_code_rank':int(keep.sum()),
        'code_covariance_after_targets':(remaining_code+remaining_code.T)/2,
        'explained_target_covariance':explained,'target_residual_covariance':remainder,
        'interpretation':'Pair J with fixed-prior conditional code variance; J can saturate. No accuracy or learnability guarantee.'}
