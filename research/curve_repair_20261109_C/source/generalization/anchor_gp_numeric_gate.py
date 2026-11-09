"""Bounded once-only synthetic numeric gate for the same-six-anchor GP.

Not material evidence. Exactly one authorized nonlinear synthetic GP fit;
three degenerate input cases must reject before kernel optimization. Expected
mean/covariance use the closed Matérn3/2 formula and dense solves, never a second
sklearn model or predict call. No search, restart, real CSV or target Y.
"""
from __future__ import annotations
import ast
import datetime as dt
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE/'anchor_gp.py'
SOURCE_SHA = '0220acc8ff8768d759a3ab87f22f7a2846aa121d628725bfb83480f16c771113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    registration=HERE/'ANCHOR_GP_TOY_REGISTERED.json'
    output=HERE/'ANCHOR_GP_NUMERIC_GATE.json'
    if output.exists():
        raise FileExistsError('Do not overwrite or repeat this once-only synthetic gate')
    reg=json.loads(registration.read_text(encoding='utf-8'))
    if (reg['role']!='TOY_ONLY_NO_MATERIAL' or reg['anchor_gp_source_sha256']!=SOURCE_SHA or
            sha(SOURCE)!=SOURCE_SHA or reg['gate_source_sha256']!=sha(__file__)):
        raise PermissionError('Actual source/hash registration required before gate execution')
    stamp=dt.datetime.fromisoformat(reg['actual_utc_before_gate'].replace('Z','+00:00'))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture UTC registration required')
    raw=SOURCE.read_bytes()
    name='_registered_anchor_gp_toy_source'
    spec=importlib.util.spec_from_file_location(name,SOURCE)
    sut=importlib.util.module_from_spec(spec)
    sys.modules[name]=sut
    exec(compile(raw,str(SOURCE),'exec'),sut.__dict__)
    checks=[]
    def check(label,value):
        if not value:
            raise AssertionError(label)
        checks.append({'check':label,'status':'PASS'})
    def compare(label,a,b):
        a,b=np.asarray(a),np.asarray(b)
        if a.shape!=b.shape or not np.allclose(a,b,atol=1e-10,rtol=1e-10):
            raise AssertionError(label)
        checks.append({'check':label,'status':'PASS','elements':int(a.size),
                       'max_absolute_error':float(np.max(np.abs(a-b)))})
    signature=inspect.signature(sut.fit_predict)
    check('no_target_Y_argument',list(signature.parameters)==['xO','yO','xT'])
    tree=ast.parse(raw)
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and
           isinstance(node.func,ast.Name) and node.func.id=='GaussianProcessRegressor']
    check('one_fixed_GPR_constructor',len(calls)==1)
    kwargs={item.arg:ast.literal_eval(item.value) for item in calls[0].keywords if item.arg!='kernel'}
    # kernel is a variable so parse only literal constructor settings.
    for key,value in [('alpha',0.),('optimizer','fmin_l_bfgs_b'),
                      ('n_restarts_optimizer',0),('normalize_y',False),('random_state',None)]:
        check('fixed_GPR_'+key,kwargs[key]==value)
    xo=np.asarray([-1.7,-1.,-.15,.4,1.65,2.45])
    xt=np.asarray([-1.4,-.6,.05,.9,1.25,1.9,2.2,2.9])
    yo=.5-.25*xo+.12*np.sin(1.3*xo)+.02*np.asarray([1.,-.7,.4,-.6,.9,-.5])
    actual=sut.fit_predict(xo,yo,xt)  # The only valid synthetic optimizer fit.
    beta=np.asarray(actual['OLS_beta'])
    scale=actual['residual_scale']
    xdesign=np.column_stack([np.ones(6),xo])
    tdesign=np.column_stack([np.ones(8),xt])
    q,r=np.linalg.qr(xdesign,mode='reduced')
    expected_beta=np.linalg.solve(r,q.T@yo)
    expected_residual=yo-xdesign@expected_beta
    expected_sse=float(expected_residual@expected_residual)
    compare('OLS_beta_independent_QR',beta,expected_beta)
    compare('own_only_SSE',actual['SSE'],expected_sse)
    compare('unbiased_own_six_residual_scale',scale,np.sqrt(expected_sse/4))
    compare('own_only_standardized_residual',actual['standardized_anchor_residual'],expected_residual/scale)
    hyper=actual['hyperparameters']
    signal,length,white=(hyper['signal_variance_standardized'],hyper['length_scale_log_gamma'],
                         hyper['white_variance_standardized'])
    def matern(a,b):
        distance=np.sqrt(3.)*np.abs(a[:,None]-b[None,:])/length
        return signal*(1+distance)*np.exp(-distance)
    koo=matern(xo,xo)+white*np.eye(6)
    kto=matern(xt,xo)
    ktt_latent=matern(xt,xt)
    expected_mean=tdesign@beta+scale*kto@np.linalg.solve(koo,(yo-xdesign@beta)/scale)
    expected_latent_cov=scale**2*(ktt_latent-kto@np.linalg.solve(koo,kto.T))
    expected_cov=expected_latent_cov+scale**2*white*np.eye(8)
    compare('Mat32_closed_form_Schur_mean',actual['mean'],expected_mean)
    compare('Mat32_closed_form_Schur_observation_covariance',actual['covariance'],expected_cov)
    compare('WhiteKernel_extra_observation_diagonal',np.asarray(actual['covariance'])-expected_latent_cov,
            scale**2*white*np.eye(8))
    check('explicit_white_observational_scope',actual['covariance_includes_observational_white_variance'] is True)
    check('warnings_preserved',isinstance(actual['optimizer_warnings'],list))
    check('bounds_and_no_restarts_preserved',actual['n_restarts_optimizer']==0 and actual['alpha']==0 and
          actual['normalize_y'] is False and actual['boundary_status']['signal']['bounds']==[1e-4,1e4] and
          actual['boundary_status']['length_scale']['bounds']==[1e-2,1e2] and
          actual['boundary_status']['white_noise']['bounds']==[1e-6,1.])
    rejected=[]
    fixtures=[('constant_SSE_zero',xo,np.ones(6),xt),
              ('exact_line_roundoff',xo,1+2*xo,xt),
              ('numerical_rank_failure',1e16+2*np.arange(6.),np.asarray([0.,1.,0.,2.,-1.,3.]),
               1e16+2*np.arange(7.,15.))]
    for label,x,y,t in fixtures:
        try:
            sut.fit_predict(x,y,t)
        except sut.AnchorGPFailure as error:
            rejected.append({'case':label,'exception':str(error),'details':error.details,
                             'optimizer_called':False})
            check('reject_'+label,True)
        else:
            raise AssertionError('A degenerate residual was silently repaired: '+label)
    result={'schema':'anchorGP-synthetic-numeric-gate-v1',
            'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'status':'PASS','role':'TOY_ONLY_NO_MATERIAL','registration_sha256':sha(registration),
            'anchor_gp_source_sha256':SOURCE_SHA,'gate_source_sha256':sha(__file__),
            'fixture':{'xO':xo.tolist(),'yO_synthetic':yo.tolist(),'xT':xt.tolist(),
                       'target_Y_exists':False},'actual_sut_prediction':actual,
            'independent_expected':{'OLS_beta_QR':expected_beta.tolist(),'SSE':expected_sse,
                'Koo_standardized':koo.tolist(),'Kto_standardized':kto.tolist(),
                'Ktt_latent_standardized':ktt_latent.tolist(),'mean':expected_mean.tolist(),
                'covariance_observations':expected_cov.tolist(),
                'covariance_latent':expected_latent_cov.tolist(),
                'white_diagonal_variance_log_eta':scale**2*white},
            'checks':checks,'rejected_cases':rejected,'valid_synthetic_GP_fits':1,
            'repeated_optimization_or_seed_search':False,'new_material_readings':0,
            'not_claimed':['material_effectiveness','independent_source_confirmation','optimizer_hyperparameter_posterior']}
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print('PASS',len(checks),'synthetic checks;1 valid GP optimizer fit;0 material readings')


if __name__=='__main__':
    main()
