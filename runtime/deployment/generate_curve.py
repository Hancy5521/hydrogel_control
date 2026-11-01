"""Portable same-code jointGaussian generator. Exactly six real anchors, no fit."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,sys
import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtri
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'core'),str(ROOT/'program')]
sys.dont_write_bytecode=True
from complete_curve import restore_model
from trial_geometry import projection
from joint_code_gaussian import CodeReference,unpack

def parse(data):
    def pairs(pairs):
        d={}
        for k,v in pairs:
            if k in d:raise ValueError('Duplicate JSON key '+k)
            d[k]=v
        return d
    return json.loads(data.decode('utf-8'),object_pairs_hook=pairs,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))
def vector(value,name):
    a=np.asarray(value)
    if a.ndim!=1 or a.dtype.kind not in 'iuf' or not len(a) or not np.isfinite(a).all() or (a<=0).any() or any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat):raise ValueError(name+' must be positive finite real vector')
    return a.astype(float)
def generate(model_payload,input_payload,draws=128,seed=20261031):
    if type(draws) is not int or draws<1 or type(seed) is not int or seed<0:raise ValueError('Positive integerdraws andnonnegative integerseed required')
    if model_payload.get('model_kind')!='same_local_code_jointGaussian_v1':raise ValueError('This generator requires its own joint-code Gaussian MODEL')
    q=np.asarray(model_payload['q'],dtype=float);theta=model_payload['theta'];unpack(theta)
    base={'reference':model_payload['reference'],'q':model_payload['q'],'heads':{},'generic_seed':20261030,'kappa':1.,'default_head':'gp'}
    gp=restore_model(base)[0]
    if set(input_payload)!= {'observed_gamma','observed_eta','target_gamma'}:raise ValueError('Exactly observed_gamma,observed_eta,target_gamma are accepted')
    gamma=vector(input_payload['observed_gamma'],'gamma');eta=vector(input_payload['observed_eta'],'eta');target=vector(input_payload['target_gamma'],'target')
    if gamma.shape!=(6,) or eta.shape!=(6,) or len(np.unique(gamma))!=6 or len(np.unique(target))!=len(target):raise ValueError('Exactly6 unique anchors,matching eta,unique targets')
    limits=np.exp(q[[0,-1]])
    if any((a<limits[0]).any() or (a>limits[1]).any() for a in [gamma,target]):raise ValueError('Coordinates must remain within frozen physical support')
    known={float(g):i for i,g in enumerate(gamma)};mapping=[known.get(float(g)) for g in target];unknown=np.array([j for j,v in enumerate(mapping) if v is None],dtype=int);answer=np.empty((draws,len(target)))
    # Analytic marginal summaries do not depend on the sampling seed or draw count.
    # Preserve the legacy eta_p05/median/p95 fields as empirical sample quantiles.
    analytic=np.empty((3,len(target)))
    if len(unknown):
        tc=np.log(gamma);th=np.log(target[unknown]);coords=np.r_[tc,th]
        if len(np.unique(coords))!=len(coords):raise ValueError('Distinct physicalcoordinates alias in floatingpoint logcoordinates')
        p=projection(gp,coords,q,'local',None);n=len(coords);k=p.reference.covariance;l=np.linalg.cholesky(k[np.ix_(p.q,p.q)]);cross=k[:n,p.q]@solve_triangular(l.T,p.directions,lower=False)
        prior=CodeReference(p.reference.mean[:n],k[:n,:n],cross);mean,cov=prior.condition(theta,np.arange(6),np.log(eta),np.arange(6,n));chol=np.linalg.cholesky(cov)
        with np.errstate(over='raise',under='raise',invalid='raise'):
            analytic[:,unknown]=np.exp(mean[None,:]+ndtri(np.array([.05,.5,.95]))[:,None]*np.sqrt(np.diag(cov))[None,:])
        logs=mean+np.random.default_rng(seed).normal(size=(draws,len(unknown)))@chol.T
        with np.errstate(over='raise',under='raise',invalid='raise'):answer[:,unknown]=np.exp(logs)
    for j,v in enumerate(mapping):
        if v is not None:answer[:,j]=eta[v];analytic[:,j]=eta[v]
    if not np.isfinite(answer).all() or not np.isfinite(analytic).all():raise ValueError('Nonfinite predictions')
    return {'actual_utc':datetime.now(timezone.utc).isoformat(),'model_kind':model_payload['model_kind'],'observed_gamma':gamma.tolist(),'observed_eta':eta.tolist(),'target_gamma':target.tolist(),'eta_draws':answer.tolist(),'eta_p05':np.quantile(answer,.05,axis=0).tolist(),'eta_median':np.quantile(answer,.5,axis=0).tolist(),'eta_p95':np.quantile(answer,.95,axis=0).tolist(),'eta_analytic_p05':analytic[0].tolist(),'eta_analytic_median':analytic[1].tolist(),'eta_analytic_p95':analytic[2].tolist(),'analytic_intervals':'Exact lognormal marginal quantiles of the fixed conditional Gaussian; no MonteCarlo error, not calibration guarantee','target_is_observed':[v is not None for v in mapping],'draws_count':draws,'seed':seed,'is_measured_data':False,'data_role':'Conditional model-generated curves plus exactly preserved known anchors; not new measurements','numerical_jitter':0.,'sampler':'FullconditionalGaussian Cholesky; noGH,no rejection,no head switching','intervals':'Legacy eta_p05/eta_median/eta_p95 are empirical MonteCarlo marginalquantiles; eta_analytic_* are exact marginal model quantiles; neither implies calibration'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=128);p.add_argument('--seed',type=int,default=20261031);a=p.parse_args()
    if a.output.resolve() in [a.model.resolve(),a.input.resolve()]:raise ValueError('Output must differ from inputs')
    m=a.model.read_bytes();i=a.input.read_bytes();result=generate(parse(m),parse(i),a.draws,a.seed);result.update({'model_sha256':hashlib.sha256(m).hexdigest(),'input_sha256':hashlib.sha256(i).hexdigest()})
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('Wrote explicitly generated conditional curves:',a.output)
if __name__=='__main__':main()
