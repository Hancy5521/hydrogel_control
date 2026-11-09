"""Saved synthetic-law audit: QR, closed Matérn3/2 and dense Schur only.

No production-model or sklearn imports, no fitting/optimizer, no raw material
data. Checks all saved means/covariances and marginal likelihoods, as well as
the SSE scale and boundary rejection. It does not certify optimizer global
optimality or empirical predictive calibration.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_close(rows, name, observed, expected, atol=3e-10, rtol=3e-9):
    observed, expected = np.asarray(observed), np.asarray(expected)
    rows.append({'check':name,'pass':bool(np.allclose(observed,expected,atol=atol,rtol=rtol)),
                 'values_compared':int(expected.size),
                 'maximum_absolute_error':float(np.max(np.abs(observed-expected)))})


def qr_ols(x, y):
    design=np.column_stack([np.ones(len(x)),x])
    q,r=np.linalg.qr(design,mode='reduced')
    beta=np.linalg.solve(r,q.T@y)
    residual=y-design@beta
    return design,beta,residual


def matern(x, z, signal, length):
    scaled=math.sqrt(3.)*np.abs(x[:,None]-z[None,:])/length
    return signal*(1.+scaled)*np.exp(-scaled)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--gate',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.gate)
    gate=json.loads(path.read_text(encoding='utf-8'))
    if gate['schema']!='curve-reference-numeric-gate-v1' or gate['scope']!='SYNTHETIC_ONLY_NO_MATERIAL_Y':
        raise ValueError('Unexpected saved fixture scope')
    regpath=Path(gate['registration_path'])
    if digest(regpath)!=gate['registration_sha256']:
        raise ValueError('Registration changed')
    reg=json.loads(regpath.read_text(encoding='utf-8'))
    for pin in reg['source_pins']:
        if digest(pin['path'])!=pin['sha256']:
            raise ValueError('Pinned source changed')
    rows=[]
    for case in gate['cases']:
        name=case['case'];law=case['law']
        xo,yo,xt=(np.array(case[k],dtype=float) for k in ('xO','yO','xT'))
        n=len(xo);design,beta,residual=qr_ols(xo,yo)
        sse=float(residual@residual);scale=math.sqrt(sse/(n-2))
        hp=law['hyperparameters'];signal=hp['signal_variance_standardized']
        length=hp['length_scale_x_units'];white=hp['white_variance_standardized']
        koo=matern(xo,xo,signal,length)+white*np.eye(n)
        kto=matern(xt,xo,signal,length)
        ktt=matern(xt,xt,signal,length)+white*np.eye(len(xt))
        chol=np.linalg.cholesky(koo)
        standardized=residual/scale
        solved_z=np.linalg.solve(chol.T,np.linalg.solve(chol,standardized))
        mean=np.column_stack([np.ones(len(xt)),xt])@beta+scale*kto@solved_z
        v=np.linalg.solve(chol,kto.T)
        covariance=scale*scale*(ktt-v.T@v)
        loglik=-0.5*float(standardized@solved_z)-float(np.log(np.diag(chol)).sum())-n/2*math.log(2*math.pi)
        for key,expected in [('OLS_beta',beta),('SSE',sse),('residual_scale',scale),
                             ('standardized_anchor_residual',standardized),('mean',mean),
                             ('covariance',covariance),
                             ('log_marginal_likelihood_standardized',loglik),
                             ('log_marginal_likelihood_y_units',loglik-n*math.log(scale))]:
            check_close(rows,name+'_'+key,law[key],expected)
        # Every target diagonal must include the white observation term.
        latent_covariance=scale*scale*(matern(xt,xt,signal,length)-v.T@v)
        check_close(rows,name+'_white_observation_increment',
                    np.asarray(law['covariance'])-latent_covariance,scale*scale*white*np.eye(len(xt)))
        check_close(rows,name+'_white_variance_y_metadata',hp['white_variance_y_units'],scale*scale*white)
        rows.append({'check':name+'_generic_fixed_reference_metadata','pass':
                     law['n']==n and law['residual_degrees_of_freedom']==n-2 and
                     law['p_OLS']==2 and law['observed_response_values_used']==n and
                     law['n_restarts_optimizer']==0 and law['alpha']==0 and
                     law['normalize_y'] is False and
                     law['covariance_includes_observational_white_variance'] is True and
                     law['target_response_values_accepted'] is False and
                     law['coordinate_transform']=='none; declared by external protocol' and
                     law['response_transform']=='none; declared by external protocol' and
                     law['boundary_status']['signal']['bounds']==[1e-4,1e4] and
                     law['boundary_status']['length_scale']['bounds']==[1e-2,1e2] and
                     law['boundary_status']['white_noise']['bounds']==[1e-6,1.] and
                     isinstance(law['optimizer_warnings'],list)})
    current=gate['cases'][-1]['law'];legacy=gate['legacy_n6_law']
    for key in ('mean','covariance','OLS_beta','SSE','residual_scale','kernel_theta_log'):
        check_close(rows,'n6_legacy_parity_'+key,current[key],legacy[key],atol=2e-11,rtol=0)
    for boundary in gate['boundary_cases']:
        x,y=(np.array(boundary[k],dtype=float) for k in ('xO','yO'))
        design=np.column_stack([np.ones(len(x)),x])
        # SVD determines rank independently of the production least-squares call.
        singular=np.linalg.svd(design,compute_uv=False)
        rank=int((singular>np.finfo(float).eps*max(design.shape)*singular[0]).sum())
        d=boundary['details']
        if rank==2:
            _,beta,residual=qr_ols(x,y)
            threshold=(64*np.finfo(float).eps*(np.linalg.norm(design,2)*np.linalg.norm(beta)+np.linalg.norm(y)))**2
            expected_reject=bool(np.all(y==y[0]) or residual@residual<=threshold)
            check_close(rows,boundary['case']+'_resolution_threshold',d['SSE_resolution_threshold'],threshold,atol=1e-38,rtol=1e-10)
        else:
            expected_reject=True
            rows.append({'check':boundary['case']+'_independent_rank','pass':rank==1 and d['rank_OLS']==1})
        rows.append({'check':boundary['case']+'_rejected_without_fit',
                     'pass':expected_reject and boundary['rejected'] and boundary['optimizer_calls']==0})
    rows.append({'check':'exact_four_synthetic_fits_only','pass':
                 gate['optimizer_calls']==[{'case':'generic_n4','observations':4},
                                          {'case':'generic_n5','observations':5},
                                          {'case':'generic_n6','observations':6},
                                          {'case':'legacy_n6_parity','observations':6}] and
                 gate['material_response_reads']==0 and gate['material_fits']==0})
    result={'schema':'curve-reference-independent-numeric-audit-v1',
            'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'audited_gate_sha256':digest(path),'registration_sha256':digest(regpath),
            'auditor_sha256':digest(__file__),
            'scope':'synthetic saved numerical laws only; no production import/sklearn/fitting',
            'status':'PASS' if all(q['pass'] for q in rows) else 'FAIL',
            'checks':rows,'material_response_reads':0,'optimizer_calls':0,
            'predictive_calibration_certified':False}
    with Path(args.output).open('x',encoding='utf-8') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'checks':len(rows),
                      'values_compared':sum(q.get('values_compared',0) for q in rows),
                      'failures':[q['check'] for q in rows if not q['pass']]}))


if __name__=='__main__':
    main()
