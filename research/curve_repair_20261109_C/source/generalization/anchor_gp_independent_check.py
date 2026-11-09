"""Independently audit saved synthetic GP output; no SUT/sklearn import/fit.

Rebuild the Matérn3/2 observation posterior from saved fitted hyperparameters,
and OLS scale from QR. Only the previously saved fixed synthetic fixture is read.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import numpy as np


HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registration=HERE/'ANCHOR_GP_TOY_REGISTERED.json'
    saved=HERE/'ANCHOR_GP_NUMERIC_GATE.json'
    output=HERE/'ANCHOR_GP_INDEPENDENT_AUDIT.json'
    if output.exists():raise FileExistsError('Retain this once-only independent audit')
    reg=json.loads(registration.read_text(encoding='utf-8'))
    if reg['role']!='TOY_ONLY_NO_MATERIAL' or reg['audit_source_sha256']!=sha(__file__):
        raise PermissionError('Registered independent source required')
    data=json.loads(saved.read_text(encoding='utf-8'))
    if data['status']!='PASS' or data['registration_sha256']!=sha(registration):
        raise ValueError('Exact registered completed synthetic gate required')
    actual=data['actual_sut_prediction']
    xo=np.asarray(data['fixture']['xO']);yo=np.asarray(data['fixture']['yO_synthetic'])
    xt=np.asarray(data['fixture']['xT'])
    design=np.column_stack([np.ones(len(xo)),xo]);target=np.column_stack([np.ones(len(xt)),xt])
    q,r=np.linalg.qr(design,mode='reduced');beta=np.linalg.solve(r,q.T@yo)
    residual=yo-design@beta;sse=float(residual@residual);scale=np.sqrt(sse/4)
    hyper=actual['hyperparameters'];signal=hyper['signal_variance_standardized']
    length=hyper['length_scale_log_gamma'];white=hyper['white_variance_standardized']
    def signal_k(a,b):
        d=np.abs(np.subtract.outer(a,b));u=math.sqrt(3.)*d/length
        return signal*(1+u)*np.exp(-u)
    koo=signal_k(xo,xo)+white*np.eye(6);kto=signal_k(xt,xo)
    target_signal=signal_k(xt,xt)
    mean=target@beta+scale*kto@np.linalg.solve(koo,residual/scale)
    latent=scale**2*(target_signal-kto@np.linalg.solve(koo,kto.T))
    observation=latent+scale**2*white*np.eye(8)
    checks=[]
    def compare(label,a,b):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.allclose(a,b,rtol=1e-10,atol=1e-10):raise AssertionError(label)
        checks.append({'check':label,'elements':int(a.size),'max_absolute_error':float(np.max(np.abs(a-b)))})
    compare('OLS_QR_beta',actual['OLS_beta'],beta)
    compare('SSE',actual['SSE'],sse)
    compare('logSSE',actual['logSSE'],np.log(sse))
    compare('scale_sqrtSSE4',actual['residual_scale'],scale)
    compare('Matern_explicit_signal_plus_white_mean',actual['mean'],mean)
    compare('Matern_explicit_signal_plus_white_covariance',actual['covariance'],observation)
    compare('noise_variance_is_additive_white_diagonal',np.asarray(actual['covariance'])-latent,scale**2*white*np.eye(8))
    if data['valid_synthetic_GP_fits']!=1 or len(data['rejected_cases'])!=3:raise AssertionError('bounded case count')
    if actual['alpha']!=0 or actual['normalize_y'] or actual['n_restarts_optimizer']!=0:raise AssertionError('fixed GP settings')
    for row in data['rejected_cases']:
        if row['optimizer_called'] or not row['exception'].startswith('STOP_NO_FALLBACK'):raise AssertionError('degenerate fit silently repaired')
        detail=row['details']
        if detail['rank_OLS']==2 and not detail['SSE']<=detail['SSE_resolution_threshold']:raise AssertionError('unexplained residual rejection')
    result={'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'PASS',
            'schema':'anchorGP-independent-synthetic-audit-v1','registration_sha256':sha(registration),
            'audited_gate_sha256':sha(saved),'audit_source_sha256':sha(__file__),
            'SUT_imported':False,'sklearn_imported':False,'new_optimizer_calls':0,
            'fixture_is_material':False,'new_material_readings':0,'checks':checks,
            'expected_mean':mean.tolist(),'expected_observation_covariance':observation.tolist(),
            'warnings_retained_from_actual_fit':actual['optimizer_warnings'],
            'boundary_status_retained':actual['boundary_status'],
            'scope':'Implementation audit from saved synthetic output, not material effect or optimizer globaloptimality'}
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print('PASS independent saved-synthetic Mat32+White audit;0 fit/real labels')


if __name__=='__main__':main()
