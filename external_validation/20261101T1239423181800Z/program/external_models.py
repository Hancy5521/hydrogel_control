"""Frozen inference and anchor-only point controls; no reference/correction training."""
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import solve_triangular
from scipy.interpolate import PchipInterpolator

PROBABILISTIC=('curvature_joint_gaussian','reference_gp','conditional_gaussian')
POINT=('pchip_log','powerlaw_6anchors','powerlaw_last2')

def load_runtime(repo):
    root=Path(repo)/'runtime'
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(root/'core'),str(root/'program')]
    from complete_curve import restore_model
    from trial_geometry import projection
    from joint_code_gaussian import CodeReference
    from gaussian_control import context_feature,GaussianDesign,predict_gaussian
    return restore_model,projection,CodeReference,context_feature,GaussianDesign,predict_gaussian

def predict(repo,model,baselines,anchors_gamma,anchors_eta,target_gamma):
    restore,projection,CodeReference,context_feature,GaussianDesign,predict_gaussian=load_runtime(repo)
    base={'reference':model['reference'],'q':model['q'],'heads':{},'generic_seed':20261030,'kappa':1.,'default_head':'gp'}
    gp=restore(base)[0]
    if baselines['reference']!=model['reference']:
        raise ValueError('Saved controls must share the exact reference')
    tc=np.log(anchors_gamma);yc=np.log(anchors_eta);th=np.log(target_gamma)
    coords=np.r_[tc,th];q=np.asarray(model['q']);n=len(coords)
    projected=projection(gp,coords,q,'local',None)
    k=projected.reference.covariance
    l=np.linalg.cholesky(k[np.ix_(projected.q,projected.q)])
    cross=k[:n,projected.q]@solve_triangular(l.T,projected.directions,lower=False)
    prior=CodeReference(projected.reference.mean[:n],k[:n,:n],cross)
    mean,cov=prior.condition(model['theta'],np.arange(6),yc,np.arange(6,n))
    law=gp.condition(tc,yc,th,report=True,new_replicates=False)
    context=gp.joint(tc,report=True)
    h=context_feature(yc,context.mean,context.cov)
    u=2*(th-q[0])/(q[-1]-q[0])-1
    control=predict_gaussian(baselines['heads']['gaussian']['theta'],GaussianDesign.from_reference(law.mean,law.cov,u,h))
    laws={'curvature_joint_gaussian':(mean,cov),'reference_gp':(law.mean,law.cov),
          'conditional_gaussian':(control.mean,control.covariance)}
    answer={}
    for name,(m,v) in laws.items():
        np.linalg.cholesky(v)
        if not np.isfinite(m).all() or not np.isfinite(v).all():raise ValueError('Nonfinite frozen forecast')
        answer[name]={'kind':'Gaussian law for natural-log viscosity','mean':m.tolist(),'covariance':v.tolist()}
    # These estimate point functions using the same six anchors only. They are
    # not probabilistic and receive no invented variance, CRPS or coverage.
    ols=np.linalg.lstsq(np.c_[np.ones(6),tc],yc,rcond=None)[0]
    last_slope=(yc[-1]-yc[-2])/(tc[-1]-tc[-2])
    points={'pchip_log':PchipInterpolator(tc,yc,extrapolate=True)(th),
            'powerlaw_6anchors':ols[0]+ols[1]*th,
            'powerlaw_last2':yc[-1]+last_slope*(th-tc[-1])}
    for name,m in points.items():
        if not np.isfinite(m).all():raise ValueError('Nonfinite anchor-only point forecast')
        answer[name]={'kind':'Point only for natural-log viscosity','mean':m.tolist()}
    return answer
