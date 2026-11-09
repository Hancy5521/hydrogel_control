"""ActualUTC-gated synthetic mixtures only; no parent/material file reader."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal,multivariate_t
from pure import build_laws,logpdf,score,marginal_quantiles,sample,METHODS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--authorization',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    auth=json.loads(Path(args.authorization).read_text(encoding='utf-8'))
    assert auth['scope']=='functional_mixture_synthetic_gate' and auth['source_sha256']==sha(__file__)
    stamp=dt.datetime.fromisoformat(auth['actual_utc']);assert stamp.tzinfo and stamp<=dt.datetime.now(dt.timezone.utc)
    for path,digest in auth['source_pins'].items():assert Path(path).suffix=='.py' and sha(path)==digest
    x=np.arange(7);k=.05*np.exp(-.5*np.abs(x[:,None]-x[None,:]))+.01*np.eye(7)
    covariance=block_diag(k,1.7*k);mg=np.linspace(5.,8.,14)
    directions=[np.zeros(14),np.linspace(-.35,.55,14),np.array([1.,-1.]*7)*8.]
    fixtures=[];rng=np.random.default_rng(20261109)
    for name,difference in zip(('identical_endpoints','moderate_disagreement','large_disagreement'),directions):
        mm=mg+difference;half=.5*(mg+mm);total=covariance+.25*np.outer(difference,difference)
        laws=build_laws(mg,mm,covariance);assert np.linalg.matrix_rank(total)==14
        for method in METHODS:
            law=laws[method];mean=np.array(law.get('mean',law.get('location')))
            assert np.array_equal(mean,half)
            observed_cov=np.array(law.get('covariance',law.get('predictive_covariance')))
            assert np.allclose(observed_cov,total,rtol=1e-13,atol=1e-13)
        if np.any(difference):assert np.max(np.abs(total[:7,7:]))>0
        probes=[mg,mm,half,half+.7*np.sqrt(np.diag(covariance)),mg-4*np.sqrt(np.diag(covariance))]
        rows=[];bound_rows=[];quantiles={};traces={}
        for method in METHODS:
            quantiles[method]=marginal_quantiles(laws[method]).tolist()
            assert np.allclose(np.array(quantiles[method])[1],half,rtol=1e-12,atol=1e-12)
            traces[method]=sample(laws[method],16,rng,return_latents=True)
            for i,y in enumerate(probes):
                result=score(laws[method],y)
                assert abs(result['log_rmse']-float(np.sqrt(np.mean((y-half)**2))))<1e-12
                rows.append({'method':method,'probe_index':i,'score':result})
                if method in METHODS[:2]:
                    endpoint=float(multivariate_normal.logpdf(y,mean=mg,cov=covariance)) if method==METHODS[0] else float(multivariate_t.logpdf(y,loc=mg,shape=.5*covariance,df=4))
                    gap=(-logpdf(laws[method],y)+endpoint)/14
                    assert gap<=math.log(2)/14+2e-10
                    bound_rows.append({'method':method,'probe_index':i,'actual_nll_per_dim_gap':gap,
                        'bound_log2_per14':math.log(2)/14,'holds':True})
        # Translation checks a change in modulus log-unit; affine scaling is
        # a separate distribution identity, not a material response transform.
        shift=-math.log(1e6);factor=3.
        translated=build_laws(mg+shift,mm+shift,covariance)
        scaled=build_laws(factor*mg,factor*mm,factor*factor*covariance)
        invariant_rows=[];y=probes[3]
        for method in METHODS:
            base=score(laws[method],y);tr=score(translated[method],y+shift);sc=score(scaled[method],factor*y)
            assert abs(base['joint_nll_per_dim']-tr['joint_nll_per_dim'])<2e-8
            assert abs(sc['joint_nll_per_dim']-base['joint_nll_per_dim']-math.log(factor))<2e-8
            assert np.allclose(sc['pointwise_log_crps'],factor*np.array(base['pointwise_log_crps']),rtol=2e-7,atol=2e-9)
            assert np.allclose(tr['pointwise_log_crps'],base['pointwise_log_crps'],rtol=2e-7,atol=2e-9)
            invariant_rows.append({'method':method,'base':base,'translated':tr,'scaled':sc,
                                   'translation':shift,'factor':factor,'probe_index':3})
        fixtures.append({'name':name,'mG':mg.tolist(),'mM':mm.tolist(),'C':covariance.tolist(),
            'expected_total_covariance':total.tolist(),'laws':laws,'target_vectors':[v.tolist() for v in probes],
            'score_rows':rows,'quantiles05_50_95_by_method':quantiles,'sample_trace':traces,
            'mixture_bound_rows':bound_rows,'unit_and_affine_invariance':invariant_rows,
            'component_crosschannel_covariance_zero':True,
            'total_crosschannel_covariance_nonzero':bool(np.any(difference))})
    output={'schema':'functional-mixture-synthetic-gate-v1','status':'PASS',
        'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'authorization_sha256':sha(args.authorization),'fixtures':fixtures,
        'new_material_numeric_G_reads':0,'parent_targets_read':0,'new_fits':0,'parameter_search':False,
        'point_RMSE_equal_is_structural':True,'mixture_bound_does_not_apply_to_moment_controls':True}
    with Path(args.output).open('x',encoding='utf-8') as f:json.dump(output,f,ensure_ascii=False,allow_nan=False,indent=2)
    print(json.dumps({'status':'PASS','output':args.output,'sha256':sha(args.output)}))


if __name__=='__main__':main()
