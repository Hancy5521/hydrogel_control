"""SOURCE ONLY: fixed-df same-mean/full-covariance Student robust control.

For each saved frozen-GP law N(m,C), use t4(loc=m,shape=C/2). Thus the
complete mean AND predictive covariance remain identical; only tail/scale-
mixture shape changes. This is a classic preset sensitivity control, not a
learned posterior, new algorithm, marginal transport or new confirmation.
Requires separate root UTC/hash registration before reading consumed targets.
No raw files, covariance fitting, degree-of-freedom search, final20 or fallback.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr


HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PROTOCOL=HERE/'MOMENT_MATCHED_STUDENT_PROTOCOL.json'
CAPACITY_SOURCE=HERE/'mean_transport_capacity.py'
CAPACITY_SHA='63ebb4fccc96372f54db76e3c6356f4c46898179ac62d5580b22110b81b8fcce'
STUDENT_SOURCE=ROOT/'baselines'/'pure_student.py'
STUDENT_SHA='567212afb8a48cebcdbf2b69f051f3a1ecfbc7b3d10e5f3bf45181aebb59c892'
METRICS=('joint_nll_per_dimension','crps','log_rmse','coverage90','width90')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('Duplicate JSON key: '+key)
            out[key]=value
        return out
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,
                      parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON token')))


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def load_fixed(path,expected,label):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Fixed '+label+' source changed')
    name='_moment_matched_'+label+'_'+expected
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        exec(compile(raw,str(path),'exec'),module.__dict__)
    return sys.modules[name]


def consumed_adapters():
    module=load_fixed(CAPACITY_SOURCE,CAPACITY_SHA,'saved_input_adapters')
    # Reuse only source/identity readers and declared balancing. No quadratic
    # coefficient, tau minimizer, summary or capacity run function is called.
    return SimpleNamespace(old_dev=module.load_old_dev,expansion=module.load_expansion,
                           xanthan=module.load_xanthan,weights=module.balanced_weights)


def authorize(path):
    reg=read(path)
    if (reg.get('schema')!='moment-matched-student-registration-v1' or
            reg.get('target_scope')!='ALREADY_CONSUMED_DEVELOPMENT_ONLY'):
        raise PermissionError('Separate consumed-development same-moment-control registration required')
    stamp=dt.datetime.fromisoformat(reg['actual_utc'].replace('Z','+00:00'))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture timezone-aware registration required')
    if reg['source_sha256']!=sha(__file__) or reg['protocol_sha256']!=sha(PROTOCOL):
        raise PermissionError('Same-moment source/protocol bytes changed')
    if sha(CAPACITY_SOURCE)!=CAPACITY_SHA or sha(STUDENT_SOURCE)!=STUDENT_SHA:
        raise ValueError('Previously pinned input-reader/Student mathematics source changed')
    if reg['capacity_adapter_source_sha256']!=CAPACITY_SHA or reg['student_math_source_sha256']!=STUDENT_SHA:
        raise PermissionError('Explicit unchanged dependency pins required')
    pins={str(Path(p).resolve()).casefold():h for p,h in reg['input_pins'].items()}
    for section,keys in [('old_dev',('gp_predictions','student_predictions','targets')),
                         ('expansion',('predictions','targets')),('xanthan',('predictions','targets'))]:
        for key in keys:
            path=reg['inputs'][section][key]
            if pins.get(str(Path(path).resolve()).casefold())!=sha(path):
                raise PermissionError('Missing/changed consumed saved input: '+path)
    return reg


def same_moment_law(mean,covariance):
    """N(m,C) -> exact same-moment t4(m,C/2), no scalar fitting."""
    m=np.array(mean,dtype=float,copy=True)
    c=np.array(covariance,dtype=float,copy=True)
    if (m.shape!=(8,) or c.shape!=(8,8) or not np.isfinite(m).all() or not np.isfinite(c).all() or
            not np.allclose(c,c.T,atol=1e-12,rtol=1e-12)):
        raise ValueError('Finite complete8D Gaussian mean and symmetric covariance required')
    np.linalg.cholesky(c)
    shape=.5*c
    np.linalg.cholesky(shape)
    recovered_covariance=2*shape
    if not np.array_equal(recovered_covariance,c):
        raise ArithmeticError('Same covariance not representable exactly; no numerical repair')
    return {'distribution':'multivariate_student_t','df':4,'location':m.tolist(),
            'scale_matrix':shape.tolist(),'predictive_covariance':recovered_covariance.tolist(),
            'moment_matched_to':'complete_frozen_Gaussian','mean_identity_exact':True,
            'full_covariance_identity_exact':True,'df_is_preset_not_fitted':True,
            'posterior_under_global_cross_mask_model_claimed':False}


def gaussian_scores(mean,covariance,y):
    m,c,y=np.asarray(mean),np.asarray(covariance),np.asarray(y)
    if y.shape!=(8,) or not np.isfinite(y).all():raise ValueError('Eight finite consumed targets required')
    lower=np.linalg.cholesky(c);sd=np.sqrt(np.diag(c));error=y-m;z=error/sd
    crps=sd*(z*(2*ndtr(z)-1)+2*np.exp(-.5*z*z)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
    standardized=np.linalg.solve(lower,error)
    joint=(8*math.log(2*math.pi)+2*np.log(np.diag(lower)).sum()+standardized@standardized)/16
    q=1.6448536269514722;inside=np.abs(error)<=q*sd;width=2*q*sd
    points=[{'truth':float(y[i]),'location':float(m[i]),'predictive_sd':float(sd[i]),
             'crps':float(crps[i]),'squared_error':float(error[i]**2),
             'coverage90':bool(inside[i]),'width90':float(width[i])} for i in range(8)]
    return {'joint_nll_per_dimension':float(joint),'crps':float(np.mean(crps)),
            'log_rmse':float(np.sqrt(np.mean(error**2))),
            'coverage90':float(np.mean(inside)),'width90':float(np.mean(width))},points


def summarize(label,rows,weights):
    total=math.fsum(weights[r['context']] for r in rows)
    if total<=0:raise ValueError('Positive nonempty diagnostic scope required')
    means={law:{metric:float(math.fsum(weights[r['context']]*r[law][metric]/total for r in rows))
                    for metric in METRICS}
           for law in ('gaussian','moment_matched_student')}
    return {'scope':label,'contexts':len(rows),'weight_in_global':total,**means,
            'delta_student_minus_gaussian':{metric:means['moment_matched_student'][metric]-means['gaussian'][metric]
                                            for metric in METRICS}}


def run(registration,output):
    reg=authorize(registration)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'STARTED.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
          'role':'FIXED_DF_SAME_MOMENT_CONSUMED_DEVELOPMENT_CONTROL','argv':sys.argv})
    try:
        adapters=consumed_adapters()
        contexts=(adapters.old_dev(reg['inputs']['old_dev'])+
                  adapters.expansion(reg['inputs']['expansion'])+
                  adapters.xanthan(reg['inputs']['xanthan']))
        if len(contexts)!=96:raise ValueError('Only old30+XG66 consumed contexts allowed')
        weights=adapters.weights(contexts)
        student=load_fixed(STUDENT_SOURCE,STUDENT_SHA,'Student_scores')
        results=[];point_terms=[]
        for context in contexts:
            law=same_moment_law(context['gaussian_mean'],context['covariance'])
            gaussian,gpoints=gaussian_scores(context['gaussian_mean'],context['covariance'],context['values'])
            robust,spoints=student.score_saved_law(law['location'],law['scale_matrix'],4,context['values'])
            # Both mean vectors are literally identical. RMSE is a structural
            # identity, not a newly observed material accuracy improvement.
            if not np.array_equal(np.asarray(law['location']),np.asarray(context['gaussian_mean'])):
                raise AssertionError('Unexpected mean change')
            if not math.isclose(robust['log_rmse'],gaussian['log_rmse'],abs_tol=1e-14,rel_tol=1e-14):
                raise AssertionError('Same-mean RMSE identity failed')
            robust['log_rmse']=gaussian['log_rmse']
            identity={k:v for k,v in context.items() if k not in ('gaussian_mean','physical_mean','covariance','values')}
            results.append({**identity,'balanced_global_weight':weights[context['context']],
                'law':law,'gaussian':gaussian,'moment_matched_student':robust,
                'delta_student_minus_gaussian':{m:robust[m]-gaussian[m] for m in METRICS},
                'log_rmse_identity_is_structural':True})
            for kind,points in [('gaussian',gpoints),('moment_matched_student',spoints)]:
                point_terms.extend({**identity,'law_id':kind,'target_offset':i,**point}
                                   for i,point in enumerate(points))
        sources=[];tasks=[];groups=[]
        for source in ('resin','xanthan'):
            selected=[r for r in results if r['source']==source]
            sources.append(summarize(source,selected,weights))
            for window,mask in sorted({(r['window'],r['mask']) for r in selected}):
                tasks.append(summarize(source+'|'+window+'|'+mask,
                    [r for r in selected if (r['window'],r['mask'])==(window,mask)],weights))
            for group in sorted({r['group'] for r in selected}):
                groups.append(summarize(source+'|'+group,[r for r in selected if r['group']==group],weights))
        payload={'schema':'same-full-moment-Student4-control-results-v1','actual_utc':utc(),
            'registration_sha256':sha(registration),'protocol_sha256':sha(PROTOCOL),
            'role':'CONSUMED_DEVELOPMENT_FIXED_ROBUST_CONTROL','df':4,
            'context_results':results,'point_terms':point_terms,
            'global_summary':summarize('equal_sources',results,weights),
            'source_summaries':sources,'task_summaries':tasks,'group_summaries':groups,
            'weighting':'same predeclared equal2sources/task/6recipe-or7salinity/condition/run weights',
            'ledger':{'existing_contexts':96,'law_evaluations':192,'point_score_terms':96*2*8,
                'reused_contextual_target_terms':96*8,'df_candidates':1,'df_search':False,
                'new_OLS_or_kernel_or_scale_fits':0,'new_scalar_minimizations':0,
                'new_raw_measurements':0,'final20_accessed':False,
                'capacity_numeric_coefficient_or_tau_functions_called':False},
            'interpretation':['Classic preset Student robust control, not a learned posterior or new algorithm',
                'Full multivariate t same mean/covariance, not copula with physical Student marginals',
                'Mean/RMSE identity structural; CRPS/NLL/coverage/width require actual scoring',
                'Both data sources already consumed; all comparisons are development diagnostics',
                'Fixed-context laws do not assert a globally Bayes-compatible cross-mask Student process',
                'LogStudent has no finite arithmetic physical-viscosity expectation']}
        write(output/'RESULTS.json',payload)
        write(output/'COMPLETE.json',{'status':'COMPLETE','actual_utc':utc(),
            'payload':'RESULTS.json','payload_sha256':sha(output/'RESULTS.json'),
            'contexts':96,'law_evaluations':192,'new_fits':0})
        print('COMPLETE fixed-df same-moment robust control:96consumed contexts;0fit/dfsearch')
    except BaseException as error:
        write(output/'FAILED.json',{'actual_utc':utc(),'type':type(error).__name__,'message':str(error),
            'decision':'Retain failure; no df/variance/noise/source fallback'})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.registration,args.output)


if __name__=='__main__':main()
