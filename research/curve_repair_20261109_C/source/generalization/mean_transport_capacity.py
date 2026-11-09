"""SOURCE ONLY: exact capacity of a fixed-covariance Gaussian mean segment.

q_tau=N(mG+tau*(mPhysical-mG), C_G), tau in [0,1]. All means/covariances
and outcomes come from already consumed saved predictions/targets. This is a
post-reveal development capacity oracle, not new confirmation or a new theory.
No raw data, final20, kernel/noise/scale update, network, seed or search exists.
Separate actual-UTC/hash registration is required before any targets are read.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE=Path(__file__).resolve().parent
PROTOCOL=HERE/'MEAN_TRANSPORT_CAPACITY_PROTOCOL.json'
SOURCES=('resin','xanthan')
XG_WINDOWS={'overlap_resin_1_to_50','full_reference_support'}
XG_MASKS={'near_tail','prefix_tail','interior_hole'}
NEAREST=[0,1,13,14,15,16]
TAIL=list(range(17,25))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key: '+key)
            result[key]=value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique,
                      parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON token')))


def write(path,payload):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()


def authorize(path):
    reg=read(path)
    if reg.get('schema')!='mean-transport-capacity-registration-v1' or reg.get('target_scope')!='ALREADY_CONSUMED_DEVELOPMENT_ONLY':
        raise PermissionError('Separate already-consumed-development capacity registration required')
    stamp=dt.datetime.fromisoformat(reg['actual_utc'].replace('Z','+00:00'))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture timezone-aware registration UTC required')
    if reg['source_sha256']!=sha(__file__) or reg['protocol_sha256']!=sha(PROTOCOL):
        raise PermissionError('Capacity source/protocol bytes changed')
    required=[]
    for section,fields in [('old_dev',('gp_predictions','student_predictions','targets')),
                           ('expansion',('predictions','targets')),
                           ('xanthan',('predictions','targets'))]:
        required.extend(reg['inputs'][section][name] for name in fields)
    pins={str(Path(name).resolve()).casefold():value for name,value in reg['input_pins'].items()}
    for name in required:
        if pins.get(str(Path(name).resolve()).casefold())!=sha(name):
            raise PermissionError('Missing/changed consumed prediction/target pin: '+name)
    return reg


def _real(value,name,ndim):
    raw=np.asarray(value)
    if (raw.ndim!=ndim or raw.dtype.kind not in 'iuf' or
            any(isinstance(v,(bool,np.bool_)) for v in np.asarray(value,dtype=object).flat)):
        raise ValueError(name+' must contain finite real numbers')
    out=np.array(raw,dtype=float,copy=True)
    if not np.isfinite(out).all():raise ValueError(name+' contains nonfinite values')
    return out


def quadratic_coefficients(gaussian_mean,physical_mean,covariance,values):
    """Return exact floating dense-Gaussian per-dimension NLL coefficients.

NLL(tau)=a*tau²/2+b*tau+c. Cholesky solves preserve the complete C_G;
there is no inverse, low-rank covariance replacement, jitter or noise rescale.
"""
    mg=_real(gaussian_mean,'Gaussian mean',1)
    mp=_real(physical_mean,'physical mean',1)
    k=_real(covariance,'fixed Gaussian covariance',2)
    y=_real(values,'already consumed target values',1)
    d=len(mg)
    if d!=8 or mp.shape!=(d,) or y.shape!=(d,) or k.shape!=(d,d):
        raise ValueError('Each saved context must be the registered complete eight-target law')
    if not np.allclose(k,k.T,atol=1e-12,rtol=1e-12):raise ValueError('Symmetric fixed C_G required')
    lower=np.linalg.cholesky(k)
    delta=mp-mg
    wd=np.linalg.solve(lower,delta)
    wr=np.linalg.solve(lower,mg-y)
    a=float(wd@wd/d)
    b=float(wd@wr/d)
    c=float((d*math.log(2*math.pi)+2*np.log(np.diag(lower)).sum()+wr@wr)/(2*d))
    if not all(math.isfinite(v) for v in (a,b,c)) or a<0:raise ArithmeticError('Invalid Gaussian quadratic coefficients')
    if a==0 and np.any(delta!=0):
        raise ArithmeticError('Nonzero mean direction numerically unresolved; no denominator floor')
    eps=np.finfo(float).eps
    dot_roundoff_bound=float((d*eps/(1-d*eps))*np.sum(np.abs(wd*wr))/d)
    return {'a':a,'b':b,'c':c,'target_dimension':d,
            'delta':delta.tolist(),'delta_l2_norm':float(np.linalg.norm(delta)),
            'b_dotproduct_roundoff_bound':dot_roundoff_bound,
            'b_within_dotproduct_roundoff':bool(abs(b)<=dot_roundoff_bound),
            'covariance_changed':False}


def minimize_quadratic(a,b,c):
    """Classical bounded one-scalar analytic minimizer, not a new optimizer."""
    if not all(math.isfinite(v) for v in (a,b,c)) or a<0:raise ValueError('Finite nonnegative-curvature quadratic required')
    if a==0:
        if b!=0:raise ArithmeticError('Zero direction must have zero slope in this family')
        tau_raw=None;tau=0.;state='IDENTICAL_MEAN_DIRECTION_ZERO_ALL_TAU_EQUIVALENT'
    else:
        tau_raw=-b/a
        if not math.isfinite(tau_raw):raise ArithmeticError('Unresolved unbounded minimizer; no clipping of nonfinite value')
        tau=float(min(1.,max(0.,tau_raw)))
        state=('BOUNDARY_GAUSSIAN_NO_POSITIVE_MEAN_STEP' if tau==0 else
               'BOUNDARY_PHYSICAL_MEAN_FULL_STEP' if tau==1 else 'INTERIOR_MEAN_SEGMENT_OPTIMUM')
    value=lambda t:float(.5*a*t*t+b*t+c)
    return {'tau_unbounded':tau_raw,'tau_star_0_to_1':tau,'state':state,
            'gradient_at_tau0':b,'gradient_at_tau1':a+b,
            'gradient_at_tau0_zero_in_float':bool(b==0),
            'nll_tau0':c,'nll_tau1':value(1.),'nll_tau_star':value(tau),
            'delta_nll_tau_star_minus_tau0':value(tau)-c,
            'positive_feasible_step_improves_in_float':bool(b<0),
            'strictly_convex_in_tau':bool(a>0)}


def _index(rows,key):
    out={}
    for row in rows:
        value=key(row)
        if value in out:raise ValueError('Duplicate saved curve/method identity')
        out[value]=row
    return out


def load_old_dev(spec):
    gp,student,targets=(read(spec[k]) for k in ('gp_predictions','student_predictions','targets'))
    if (student['main_predictions_sha256']!=sha(spec['gp_predictions']) or
            targets['predictions_sha256']!=sha(spec['gp_predictions'])):
        raise ValueError('Old development prediction/Student/target source chain mismatch')
    gindex=_index(gp['forecasts'],lambda r:(r['member'],r['method_id']))
    pindex=_index(student['forecasts'],lambda r:(r['member'],r['method_id']))
    if len(targets['curves'])!=15:raise ValueError('Only original fifteen consumed dev curves required')
    rows=[]
    for truth in targets['curves']:
        member=truth['member']
        g=gindex[member,'GP_nearest_target6']
        p=pindex[member,'power_law__curv_nearest_target6']
        if (g['anchor_indices']!=NEAREST or p['anchor_indices']!=NEAREST or
                g['target_indices']!=TAIL or p['target_indices']!=TAIL or truth['target_indices']!=TAIL or
                g['anchor_values']!=p['anchor_values'] or g['recipe']!=p['recipe'] or g['recipe']!=truth['recipe']):
            raise ValueError('Exact same own-six/target-eight old development payloads required')
        rows.append({'context':'resin_dev|'+member,'source':'resin','cohort':'old_dev',
                     'group':g['recipe'],'condition':g['recipe'],'window':'resin_native_nominal1_to50',
                     'mask':'near_tail','member':member,'run_id':g['run_id'],
                     'gaussian_mean':g['mean'],'physical_mean':p['location'],
                     'covariance':g['covariance'],'values':truth['values']})
    return rows


def load_expansion(spec):
    pred,targets=read(spec['predictions']),read(spec['targets'])
    if targets['predictions_sha256']!=sha(spec['predictions']):raise ValueError('Consumed expansion identity mismatch')
    index=_index(pred['forecasts'],lambda r:(r['member'],r['method_id']))
    if len(targets['curves'])!=15:raise ValueError('Only the consumed fifteen expansion curves are allowed; final20 forbidden')
    rows=[]
    for truth in targets['curves']:
        member=truth['member'];g=index[member,'gaussian_frozen_gp'];p=index[member,'physical_full_student']
        if (g['role']!='future_expansion' or p['role']!='future_expansion' or truth['role']!='future_expansion' or
                g['anchor_indices']!=NEAREST or p['anchor_indices']!=NEAREST or
                g['target_indices']!=TAIL or p['target_indices']!=TAIL or truth['target_indices']!=TAIL or
                g['anchor_values']!=p['anchor_values'] or g['recipe']!=p['recipe'] or g['recipe']!=truth['recipe']):
            raise ValueError('Original consumed future_expansion role/layout/own-six identity required')
        rows.append({'context':'resin_expansion|'+member,'source':'resin','cohort':'consumed_expansion',
                     'group':g['recipe'],'condition':g['recipe'],'window':'resin_native_nominal1_to50',
                     'mask':'near_tail','member':member,'run_id':g['run_id'],
                     'gaussian_mean':g['mean'],'physical_mean':p['location'],
                     'covariance':g['covariance'],'values':truth['values']})
    return rows


def load_xanthan(spec):
    pred,targets=read(spec['predictions']),read(spec['targets'])
    if targets['predictions_sha256']!=sha(spec['predictions']):raise ValueError('Consumed XG source identity mismatch')
    target_map=_index(targets['contexts'],lambda r:r['context'])
    if len(pred['forecasts'])!=66 or len(target_map)!=66:raise ValueError('Exactly consumed XG66 contexts required')
    rows=[]
    for forecast in pred['forecasts']:
        if forecast['window'] not in XG_WINDOWS or forecast['mask'] not in XG_MASKS:
            raise ValueError('Only original frozen two-window/three-mask XG benchmark accepted')
        g,p=forecast['laws']['frozen_gaussian_gp'],forecast['laws']['physical_full_student']
        rows.append({'context':'xanthan|'+forecast['context'],'source':'xanthan','cohort':'consumed_XG',
                     'group':format(float(forecast['salinity_M']),'.17g'),
                     'condition':forecast['preparation_group'],'salinity_M':forecast['salinity_M'],
                     'preparation_order':forecast['preparation_order'],'window':forecast['window'],'mask':forecast['mask'],
                     'member':forecast['member'],'gaussian_mean':g['mean'],'physical_mean':p['location'],
                     'covariance':g['covariance'],'values':target_map[forecast['context']]['values']})
    return rows


def balanced_weights(rows):
    """Equal source -> equal window/mask task -> equal recipe/salt -> conditions/runs."""
    if {r['source'] for r in rows}!=set(SOURCES):raise ValueError('Both fixed consumed sources required')
    weights={}
    for source in SOURCES:
        source_rows=[r for r in rows if r['source']==source]
        tasks=sorted({(r['window'],r['mask']) for r in source_rows})
        groups=sorted({r['group'] for r in source_rows})
        if len(groups)!=(6 if source=='resin' else 7):raise ValueError('Expected six resin recipes or seven XGsalinities')
        if len(tasks)!=(1 if source=='resin' else 6):raise ValueError('Expected one resin task or six XGtasks')
        for task in tasks:
            task_rows=[r for r in source_rows if (r['window'],r['mask'])==task]
            for group in groups:
                selected=[r for r in task_rows if r['group']==group]
                if not selected:raise ValueError('All groups must contribute to every original source/task')
                conditions=sorted({r['condition'] for r in selected})
                for condition in conditions:
                    subset=[r for r in selected if r['condition']==condition]
                    weight=1/(len(SOURCES)*len(tasks)*len(groups)*len(conditions)*len(subset))
                    for row in subset:
                        if row['context'] in weights:raise ValueError('Duplicated source context')
                        weights[row['context']]=weight
    if not math.isclose(sum(weights.values()),1.,abs_tol=1e-12):raise ArithmeticError('Balanced weights must sum to1')
    return weights


def summary(label,rows,weights):
    total=sum(weights[r['context']] for r in rows)
    if total<=0:raise ValueError('Nonempty positive-weight diagnostic scope required')
    coefficients={name:float(math.fsum(weights[r['context']]*r[name]/total for r in rows)) for name in ('a','b','c')}
    return {'scope':label,'contexts':len(rows),'source_weight_in_global':total,
            **coefficients,**minimize_quadratic(**coefficients),
            'normalized_context_weights':{r['context']:weights[r['context']]/total for r in rows}}


def run(registration,output):
    reg=authorize(registration)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'STARTED.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
                                'role':'CONSUMED_DEVELOPMENT_CAPACITY_ORACLE','argv':sys.argv})
    try:
        rows=load_old_dev(reg['inputs']['old_dev'])+load_expansion(reg['inputs']['expansion'])+load_xanthan(reg['inputs']['xanthan'])
        if len(rows)!=96 or len({r['context'] for r in rows})!=96:raise ValueError('Exactly old30+XG66 contexts required')
        if len({r['group'] for r in rows if r['cohort']=='old_dev'})!=3 or len({r['group'] for r in rows if r['cohort']=='consumed_expansion'})!=3:
            raise ValueError('Original disjoint three dev and three expansion recipes required')
        weights=balanced_weights(rows)
        results=[]
        for row in rows:
            coeff=quadratic_coefficients(row['gaussian_mean'],row['physical_mean'],row['covariance'],row['values'])
            identity={k:v for k,v in row.items() if k not in ('gaussian_mean','physical_mean','covariance','values')}
            results.append({**identity,**coeff,**minimize_quadratic(coeff['a'],coeff['b'],coeff['c']),
                            'balanced_global_weight':weights[row['context']]})
        global_summary=summary('all_sources_equal_weight',results,weights)
        source_summaries=[summary(source,[r for r in results if r['source']==source],weights) for source in SOURCES]
        task_summaries=[];group_summaries=[]
        for source in SOURCES:
            selected=[r for r in results if r['source']==source]
            for window,mask in sorted({(r['window'],r['mask']) for r in selected}):
                task_summaries.append(summary(source+'|'+window+'|'+mask,
                    [r for r in selected if (r['window'],r['mask'])==(window,mask)],weights))
            for group in sorted({r['group'] for r in selected}):
                group_summaries.append(summary(source+'|'+group,[r for r in selected if r['group']==group],weights))
        cross_source=[]
        for train in source_summaries:
            for test in source_summaries:
                tau=train['tau_star_0_to_1']
                value=.5*test['a']*tau*tau+test['b']*tau+test['c']
                cross_source.append({'tau_fit_scope':train['scope'],'evaluation_scope':test['scope'],
                    'tau':tau,'joint_nll_per_dimension':value,'delta_vs_unchanged_GP':value-test['c'],
                    'scope_role':'both_sources_consumed_development_not_independent_validation'})
        analyses=1+len(results)+len(source_summaries)+len(task_summaries)+len(group_summaries)
        payload={'schema':'fixed-full-Gaussian-mean-segment-capacity-v1','actual_utc':utc(),
            'registration_sha256':sha(registration),'protocol_sha256':sha(PROTOCOL),
            'role':'CONSUMED_DEVELOPMENT_CAPACITY_ORACLE','context_results':results,
            'global_summary':global_summary,'source_summaries':source_summaries,
            'task_summaries':task_summaries,'group_summaries':group_summaries,
            'cross_source_tau_diagnostics':cross_source,
            'weighting':'equal2sources; equalwindow/mask within source; equal6recipe or7salinity; equalorderconditions then runs within group',
            'ledger':{'existing_contexts':96,'saved_contextual_target_terms_read':96*8,
                      'new_raw_measurements_read':0,'new_final20_values':0,'new_covariance_fits':0,
                      'new_OLS_fits':0,'new_numerical_optimizer_calls':0,
                      'analytic_scalar_minimizations':analyses,
                      'parameters_in_hypothetical_global_family':1,
                      'diagnostic_oracle_tau_not_deployable_per_context':True},
            'scope_limits':['Classical Gaussian quadratic identity, not a new optimizer',
                            'Every tau uses already-consumed outcomes; not independent confirmation',
                            'Covariance/nugget/noise/scale remain exactly unchanged',
                            'Group/context oracles are capacity upper bounds, not new fitted repair models',
                            'No CRPS/calibration guarantee from a jointNLL oracle alone',
                            'Positive pooled improvement alone cannot establish new-source transfer']}
        write(output/'CAPACITY.json',payload)
        write(output/'COMPLETE.json',{'actual_utc':utc(),'status':'COMPLETE','payload':'CAPACITY.json',
                                     'payload_sha256':sha(output/'CAPACITY.json'),
                                     'analytic_scalar_minimizations':analyses,'new_covariance_fits':0})
        print('COMPLETE consumed-development capacity; contexts',len(results),'analytic scalar minimizations',analyses)
    except BaseException as error:
        write(output/'FAILED.json',{'actual_utc':utc(),'type':type(error).__name__,'message':str(error),
                                    'decision':'Preserve failure; no covariance/floor/rawdata fallback'})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.registration,args.output)


if __name__=='__main__':main()
