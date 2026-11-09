"""SOURCE ONLY independent whole-curve mixture audit adapters.

Expected math imports only our pinned independent module. No production import,
parameter fitting, target-driven method selection, or new original-data reader.
Root registration/authorization is required before the eventual CLI is run.
"""
from __future__ import annotations
import hashlib
import importlib.util
import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path
import traceback
import numpy as np
from scipy import linalg,optimize,stats

HERE=Path(__file__).resolve().parent
MATH=HERE/'functional_mixture_independent_math.py'
MATH_SHA='3d03817adcc5395b0f703a605b5f0eb2a4e1392d4b0942d3f551f29a078254d2'
METHODS=('whole_curve_gaussian_density_mixture','whole_curve_student4_density_mixture',
         'moment_matched_gaussian','moment_matched_student4')
KINDS=('whole_curve_equal_gaussian_mixture','whole_curve_equal_student4_mixture',
       'moment_matched_gaussian','moment_matched_student4')
DESC={METHODS[0]:('gaussian','density_mixture'),METHODS[1]:('student4','density_mixture'),
      METHODS[2]:('gaussian','moment_matched'),METHODS[3]:('student4','moment_matched')}
if hashlib.sha256(MATH.read_bytes()).hexdigest()!=MATH_SHA:raise RuntimeError('Independent math pin changed')
_spec=importlib.util.spec_from_file_location('independent_functional_mixture_math',MATH)
im=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(im)


class Checks:
    def __init__(self):self.groups={};self.count=0
    def ok(self,name,condition):
        self.count+=1
        if not condition:raise AssertionError(name)
        group=self.groups.setdefault(name,{'calls':0,'elements':0,'max_absolute_error':0.})
        group['calls']+=1;group['elements']+=1
    def close(self,name,actual,expected,rtol=2e-8,atol=2e-9):
        a=np.asarray(actual,float);b=np.asarray(expected,float);self.count+=1
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError(name+' shape/nonfinite')
        error=float(np.max(np.abs(a-b))) if a.size else 0.
        if not np.all(np.abs(a-b)<=atol+rtol*np.abs(b)):
            raise AssertionError(f'{name}: max_abs={error:.17g}; rtol={rtol}; atol={atol}')
        group=self.groups.setdefault(name,{'calls':0,'elements':0,'max_absolute_error':0.})
        group['calls']+=1;group['elements']+=int(a.size);group['max_absolute_error']=max(group['max_absolute_error'],error)


def laws_check(laws,mg,mm,C,c):
    mg,mm,C=map(lambda v:np.asarray(v,float),(mg,mm,C));mean,total=im.moments(mg,mm,C)
    c.ok('four_fixed_methods',set(laws)==set(METHODS))
    for method,kind in zip(METHODS,KINDS):
        law=laws[method];tail,representation=DESC[method]
        c.ok('distribution_identity',law['distribution']==kind)
        if representation=='density_mixture':
            c.close('fixed_equal_weights',law['weights'],[.5,.5],rtol=0,atol=0)
            c.close('component_locations',law['component_locations'],np.stack([mg,mm]),rtol=2e-13,atol=1e-13)
            c.close('component_C_retained',law['component_covariance'],C,rtol=2e-13,atol=1e-13)
            c.close('mixture_mean',law['mean'],mean,rtol=2e-13,atol=1e-13)
            c.close('mixture_total_covariance',law['predictive_covariance'],total,rtol=2e-13,atol=1e-13)
            c.ok('whole_vector_latent_branch',law['branch_scope']=='one Bernoulli for whole14D curve, never per column')
            if tail=='student4':
                c.ok('component_df4',law['df']==4)
                c.close('component_t4_scale',law['component_scale_matrix'],.5*C,rtol=2e-13,atol=1e-13)
        elif tail=='gaussian':
            c.close('matched_Gaussian_mean',law['mean'],mean,rtol=2e-13,atol=1e-13)
            c.close('matched_Gaussian_total_covariance',law['covariance'],total,rtol=2e-13,atol=1e-13)
        else:
            c.ok('matched_df4',law['df']==4)
            c.close('matched_t4_location',law['location'],mean,rtol=2e-13,atol=1e-13)
            c.close('matched_t4_scale',law['scale_matrix'],.5*total,rtol=2e-13,atol=1e-13)
            c.close('matched_t4_covariance',law['predictive_covariance'],total,rtol=2e-13,atol=1e-13)
    return mean,total


def independent_quantiles(mg,mm,C,tail,representation):
    mg,mm,C=map(lambda v:np.asarray(v,float),(mg,mm,C));d=len(mg);out=np.empty((3,d))
    probs=[.05,.5,.95]
    for i,p in enumerate(probs):
        for j in range(d):
            if representation=='moment_matched':
                out[i,j]=im.matched_marginal_quantile(p,mg[j],mm[j],C[j,j],tail)
            elif p==.5:out[i,j]=(mg[j]+mm[j])/2
            else:
                s=math.sqrt(C[j,j] if tail=='gaussian' else C[j,j]/2)
                z=stats.norm.ppf(p) if tail=='gaussian' else stats.t.ppf(p,4)
                endpoints=sorted([mg[j]+s*z,mm[j]+s*z])
                if endpoints[0]==endpoints[1]:out[i,j]=endpoints[0]
                else:
                    # Quantile lies between the two component quantiles.
                    lower=np.nextafter(endpoints[0],-np.inf);upper=np.nextafter(endpoints[1],np.inf)
                    fun=lambda v:im.marginal_cdf(v,mg[j],mm[j],C[j,j],tail,representation)-p
                    out[i,j]=optimize.brentq(fun,lower,upper,xtol=2e-13,rtol=5e-15)
    return out


def expected_score(y,mg,mm,C,method,c):
    y,mg,mm,C=map(lambda v:np.asarray(v,float),(y,mg,mm,C));tail,rep=DESC[method];D=len(y);mean=(mg+mm)/2
    logpdf=im.mixture_logpdf(y,mg,mm,C,tail) if rep=='density_mixture' else im.moment_matched_logpdf(y,mg,mm,C,tail)
    crps=[];errors=[]
    for j in range(D):
        val,error=im.marginal_crps_cdf(y[j],mg[j],mm[j],C[j,j],tail,rep)
        crps.append(val);errors.append(error)
        if tail=='gaussian' and rep=='density_mixture':
            c.close('Gaussian_CRPS_two_independent_domains',val,
                    im.gaussian_mixture_crps_energy(y[j],mg[j],mm[j],math.sqrt(C[j,j])),rtol=2e-9,atol=2e-10)
    crps=np.array(crps);q=independent_quantiles(mg,mm,C,tail,rep)
    score={'joint_nll_per_dim':-logpdf/D,'log_crps':float(crps.mean()),'pointwise_log_crps':crps,
           'log_rmse':float(np.sqrt(np.mean((y-mean)**2))),'coverage90':float(np.mean((y>=q[0])&(y<=q[2]))),
           'log_width90':float(np.mean(q[2]-q[0]))}
    channels=[]
    if D==14:
        for j,name in enumerate(('Gprime','Gdoubleprime')):
            sl=slice(j*7,(j+1)*7)
            channels.append({'channel':name,'log_rmse':float(np.sqrt(np.mean((y[sl]-mean[sl])**2))),
                             'log_crps':float(crps[sl].mean()),
                             'coverage90':float(np.mean((y[sl]>=q[0,sl])&(y[sl]<=q[2,sl]))),
                             'log_width90':float(np.mean(q[2,sl]-q[0,sl]))})
    if rep=='density_mixture':
        gp=im.component_logpdf(y,mg,C,tail);teacher=im.component_logpdf(y,mm,C,tail)
        c.ok('mixture_NLL_bound_GP_endpoint',-logpdf/D<=-gp/D+math.log(2)/D+2e-10)
        c.ok('mixture_NLL_bound_teacher_endpoint',-logpdf/D<=-teacher/D+math.log(2)/D+2e-10)
    return score,channels,q,max(errors)


def score_check(actual,y,mg,mm,C,method,c):
    expected,channels,q,error=expected_score(y,mg,mm,C,method,c)
    for k,v in expected.items():c.close('saved_'+k,actual[k],v)
    c.ok('score_channel_labels',[r['channel'] for r in actual['channels']]==[r['channel'] for r in channels])
    for a,b in zip(actual['channels'],channels):
        for k in ('log_rmse','log_crps','coverage90','log_width90'):c.close('channel_'+k,a[k],b[k])
    return expected,q,error


def sample_trace_check(trace,mg,mm,C,method,c):
    tail,rep=DESC[method];mg,mm,C=map(lambda v:np.asarray(v,float),(mg,mm,C))
    mean,total=im.moments(mg,mm,C)
    draws=np.asarray(trace['draws']);normal=np.asarray(trace['standard_normal']);n=len(draws)
    c.ok('sample_full_curve_shape',draws.shape==normal.shape==(n,14) and n>0)
    if rep=='density_mixture':
        branch=np.asarray(trace['whole_curve_branch']);c.ok('one_branch_per_curve',branch.shape==(n,) and bool(np.isin(branch,[0,1]).all()))
        locations=np.stack([mg,mm])[branch];cov=C
    else:
        c.ok('no_branch_for_single_law',trace['whole_curve_branch'] is None);locations=np.broadcast_to(mean,draws.shape);cov=total
    residual=normal@np.linalg.cholesky(cov if tail=='gaussian' else cov/2).T
    if tail=='student4':
        chi=np.asarray(trace['chi_square4_per_curve']);c.ok('one_radial_scale_per_curve',chi.shape==(n,) and bool(np.all(chi>0)))
        residual/=np.sqrt(chi[:,None]/4)
    else:c.ok('no_radial_scale_for_Gaussian',trace['chi_square4_per_curve'] is None)
    c.close('whole_curve_sample_reconstruction',draws,locations+residual,rtol=2e-12,atol=2e-12)
    return n


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def pairs(values):
    out={}
    for k,v in values:
        if k in out:raise ValueError('Duplicate JSON key '+k)
        out[k]=v
    return out
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs)
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')


def authorization(reg_path,phase,c):
    reg=read(reg_path);stamp=dt.datetime.fromisoformat(reg['actual_utc'])
    c.ok('actual_nonfuture_root_registration',stamp.tzinfo is not None and stamp<=dt.datetime.now(dt.timezone.utc))
    for path,digest in reg['source_pins'].items():c.ok('registered_source_pin',sha(path)==digest)
    if phase=='gate':c.ok('root_toy_scope',reg['scope']=='functional_mixture_synthetic_gate')
    else:
        c.ok('development_schema',reg['schema']=='functional-mixture-saved18-development-v1')
        c.ok('fixed_family_not_search',reg['methods']==list(METHODS) and reg['mixture_weight']==.5 and reg['student_df']==4 and reg['new_raw_G_accesses']==reg['new_fits']==0)
        c.ok('development_not_confirmation',reg['role']=='POST_REVEAL_DEVELOPMENT_DIAGNOSTIC')
        for name in ('toy_gate','independent_gate'):
            c.ok('prior_gate_PASS',sha(reg[name+'_path'])==reg[name+'_sha256'] and read(reg[name+'_path'])['status']=='PASS')
        post=read(reg['parent_post_audit'])
        c.ok('prior_functional_POST_PASS',post['status']=='PASS' and post['phase']=='post' and
             post['predictions_sha256']==sha(reg['parent_predictions']) and
             post['targets_sha256']==sha(reg['parent_targets']) and post['scores_sha256']==sha(reg['parent_scores']))
    return reg


def toy(reg_path,gate_path,c):
    reg=authorization(reg_path,'gate',c);g=read(gate_path)
    c.ok('gate_identity',g['schema']=='functional-mixture-synthetic-gate-v1' and g['status']=='PASS' and
         g['authorization_sha256']==sha(reg_path) and g['source_sha256']==reg['source_sha256'])
    c.ok('synthetic_only',g['new_material_numeric_G_reads']==g['parent_targets_read']==g['new_fits']==0 and not g['parameter_search'])
    fixtures=g['fixtures'];c.ok('three_prespecified_fixtures',[f['name'] for f in fixtures]==['identical_endpoints','moderate_disagreement','large_disagreement'])
    scalar_terms=0;samples=0;max_integral_error=0.
    for f in fixtures:
        mg,mm,C=map(lambda v:np.asarray(v,float),(f['mG'],f['mM'],f['C']));mean,total=laws_check(f['laws'],mg,mm,C,c)
        c.close('fixture_total_covariance',f['expected_total_covariance'],total,rtol=2e-13,atol=1e-13)
        c.ok('component_crosschannel_zero',bool(np.all(C[:7,7:]==0)) and f['component_crosschannel_covariance_zero'])
        c.ok('mixture_predictive_crosschannel',f['total_crosschannel_covariance_nonzero']==bool(np.any(total[:7,7:]!=0)))
        expected_indices={(m,i) for m in METHODS for i in range(len(f['target_vectors']))}
        c.ok('toy_score_grid',len(f['score_rows'])==20 and {(r['method'],r['probe_index']) for r in f['score_rows']}==expected_indices)
        base={}
        for r in f['score_rows']:
            y=np.array(f['target_vectors'][r['probe_index']]);expected,q,err=score_check(r['score'],y,mg,mm,C,r['method'],c)
            scalar_terms+=14;max_integral_error=max(max_integral_error,err);base[(r['method'],r['probe_index'])]=expected
            if r['probe_index']==0:c.close('toy_saved_quantiles',f['quantiles05_50_95_by_method'][r['method']],q,rtol=2e-9,atol=2e-9)
        for method in METHODS:samples+=sample_trace_check(f['sample_trace'][method],mg,mm,C,method,c)
        for row in f['mixture_bound_rows']:
            method=row['method'];i=row['probe_index'];tail,_=DESC[method];y=np.array(f['target_vectors'][i])
            gap=base[(method,i)]['joint_nll_per_dim']+im.component_logpdf(y,mg,C,tail)/14
            c.close('toy_saved_bound_gap',row['actual_nll_per_dim_gap'],gap)
            c.close('bound_constant',row['bound_log2_per14'],math.log(2)/14,rtol=0,atol=0)
            c.ok('bound_holds',row['holds'] and gap<=math.log(2)/14+2e-10)
        c.ok('toy_bound_cardinality',len(f['mixture_bound_rows'])==10)
        for row in f['unit_and_affine_invariance']:
            method=row['method'];y=np.array(f['target_vectors'][row['probe_index']]);shift=row['translation'];factor=row['factor']
            c.ok('fixed_affine_fixture',shift==-math.log(1e6) and factor==3.)
            expected,q,err=score_check(row['base'],y,mg,mm,C,method,c)
            trans,_,terr=score_check(row['translated'],y+shift,mg+shift,mm+shift,C,method,c)
            scaled,_,serr=score_check(row['scaled'],factor*y,factor*mg,factor*mm,factor*factor*C,method,c)
            c.close('translation_NLL_identity',trans['joint_nll_per_dim'],expected['joint_nll_per_dim'],rtol=2e-9,atol=2e-9)
            c.close('scale_density_Jacobian',scaled['joint_nll_per_dim'],expected['joint_nll_per_dim']+math.log(factor),rtol=2e-9,atol=2e-9)
            c.close('translation_CRPS_identity',trans['pointwise_log_crps'],expected['pointwise_log_crps'])
            c.close('scale_CRPS_identity',scaled['pointwise_log_crps'],factor*expected['pointwise_log_crps'])
            max_integral_error=max(max_integral_error,err,terr,serr);scalar_terms+=42
        # Equal endpoints collapse exactly to their own full-covariance tail.
        if f['name']=='identical_endpoints':
            for i,y in enumerate(f['target_vectors']):
                for a,b in [(METHODS[0],METHODS[2]),(METHODS[1],METHODS[3])]:
                    for key in ('joint_nll_per_dim','pointwise_log_crps','log_rmse','coverage90','log_width90'):
                        c.close('identical_endpoints_distribution_collapse',base[(a,i)][key],base[(b,i)][key],rtol=2e-9,atol=2e-9)
    return {'synthetic_fixtures':3,'toy_scalar_CRPS_terms':scalar_terms,'sample_trace_draws':samples,
            'largest_independent_CDF_quadrature_error_estimate':max_integral_error,
            'new_material_G_reads':0,'parent_target_reads':0,'production_imports':0,
            'notes':'Moment identities are analytic; sample traces validate one whole-vector branch and radial variable. No finite-fourth-moment CLT claim for Student4 empirical covariance.'}


def saved_laws(reg_path,directory,c):
    reg=authorization(reg_path,'laws',c);p=read(directory/'PREDICTIONS.json');old=read(reg['parent_predictions'])
    c.ok('saved_predictions_schema',p['schema']=='functional-mixture-saved-predictions-v1' and p['registration_sha256']==sha(reg_path))
    c.ok('parent_prediction_binding',p['parent_predictions_sha256']==sha(reg['parent_predictions']))
    c.ok('reused_scope_no_fits_no_raw',p['role']=='POST_REVEAL_DEVELOPMENT_DIAGNOSTIC' and p['new_fits']==p['new_raw_G_accesses']==0 and p['other161_temperature_curves_accessed'] is False)
    c.ok('method_grid',p['methods']==list(METHODS) and p['contexts']==len(p['forecasts'])==len(old['forecasts'])==18)
    original={f['context']:f for f in old['forecasts']};c.ok('same_context_set',{f['context'] for f in p['forecasts']}==set(original))
    count=0
    for f in p['forecasts']:
        before=original[f['context']]
        c.ok('same_context_roles',all(f[k]==before[k] for k in ('context','condition_id','state_block','mask')))
        available=before['method_availability']['half_maxwell_mean_gaussian']['available']
        c.ok('parent_availability_preserved',(f['status']=='OK')==available)
        if not available:c.ok('failure_not_deleted',f['laws']=={});continue
        c.close('frozen_GP_endpoint',f['mG'],before['gp_mean'],rtol=0,atol=0)
        c.close('frozen_Maxwell_endpoint',f['mM'],before['maxwell_log_location'],rtol=0,atol=0)
        c.close('frozen_C_endpoint',f['C'],before['full_residual_covariance'],rtol=0,atol=0)
        mean,total=laws_check(f['laws'],f['mG'],f['mM'],f['C'],c)
        c.close('structural_original_half_mean',mean,before['laws']['half_maxwell_mean_gaussian']['mean'],rtol=0,atol=0)
        c.close('common_point_log_location',f['common_point_log_location'],mean,rtol=0,atol=0)
        count+=4
    c.ok('law_count',p['law_count']==count)
    complete=read(directory/'PREDICTION_COMPLETE.json');c.ok('completion_receipt',complete['predictions_sha256']==sha(directory/'PREDICTIONS.json'))
    return {'contexts':18,'laws_audited':count,'new_fits':0,'new_raw_reads':0,'parent_target_reads':0,
            'production_imports':0,'not_new_independent_confirmation':True}


def saved_scores(reg_path,directory,c):
    reg=authorization(reg_path,'scores',c);p=read(directory/'PREDICTIONS.json');s=read(directory/'SCORES.json')
    old=read(reg['parent_scores']);oldrows={(r['context'],r['method']):r for r in old['rows']}
    targets={r['context']:r for r in read(reg['parent_targets'])['contexts']};fs={r['context']:r for r in p['forecasts']}
    c.ok('scores_schema_scope',s['schema']=='functional-mixture-saved-scores-v1' and s['role']=='POST_REVEAL_DEVELOPMENT_DIAGNOSTIC' and s['contexts_denominator']==18)
    c.ok('score_parent_bindings',s['registration_sha256']==sha(reg_path) and s['predictions_sha256']==sha(directory/'PREDICTIONS.json') and s['parent_targets_sha256']==sha(reg['parent_targets']) and s['parent_scores_sha256']==sha(reg['parent_scores']))
    c.ok('no_unread_source_access',not s['other161_temperature_curves_accessed'] and not s['unseen60C_confirmation_executed'] and not s['independent_material_confirmation'] and s['new_fits']==s['new_raw_G_accesses']==0)
    pre=read(directory/'INDEPENDENT_LAWS.json')
    c.ok('derived_laws_frozen_before_score',pre['status']=='PASS' and pre['predictions_sha256']==sha(directory/'PREDICTIONS.json') and s['pre_audit_sha256']==sha(directory/'INDEPENDENT_LAWS.json'))
    rows={};max_integral_error=0.
    for row in s['rows']:
        f=fs[row['context']];y=np.log(np.array(targets[f['context']]['values_Pa']).reshape(14))
        expected,q,error=score_check(row,y,f['mG'],f['mM'],f['C'],row['method'],c)
        rows[(row['context'],row['method'])]={**{k:row[k] for k in ('context','condition_id','state_block','mask','method')},**expected}
        max_integral_error=max(max_integral_error,error)
        c.close('same_point_error_as_original_half',row['log_rmse'],oldrows[(row['context'],'half_maxwell_mean_gaussian')]['log_rmse'],rtol=0,atol=2e-12)
    expected_keys={(f['context'],m) for f in p['forecasts'] if f['status']=='OK' and targets[f['context']]['target_admissible'] for m in METHODS}
    c.ok('score_grid_complete',len(rows)==len(s['rows']) and set(rows)==expected_keys)
    for row in s['macro_by_mask']:
        subset=[r for r in rows.values() if r['mask']==row['mask'] and r['method']==row['method']]
        c.ok('macro_denominator6',row['planned_state_conditions']==6 and row['scored_state_conditions']==len(subset) and row['conditional_subset']==(len(subset)!=6))
        for key in ('joint_nll_per_dim','log_crps','log_rmse','coverage90','log_width90'):
            if subset:c.close('macro_'+key,row[key],np.mean([r[key] for r in subset]))
            else:c.ok('empty_macro_None',row[key] is None)
    c.ok('macro_grid12',len(s['macro_by_mask'])==12)
    for row in s['paired_vs_original']:
        subset=[r for r in rows.values() if r['mask']==row['mask'] and r['method']==row['method'] and (r['context'],row['reference']) in oldrows]
        c.ok('paired_denominator6',row['planned_state_conditions']==6 and row['paired_state_conditions']==len(subset) and row['conditional_subset']==(len(subset)!=6))
        for key in ('joint_nll_per_dim','log_crps','log_rmse','coverage90','log_width90'):
            if subset:c.close('paired_'+key,row['mean_delta_'+key],np.mean([r[key]-oldrows[(r['context'],row['reference'])][key] for r in subset]))
            else:c.ok('empty_pair_None',row['mean_delta_'+key] is None)
    c.ok('paired_grid36',len(s['paired_vs_original'])==36)
    c.ok('RMSE_identity_grid',len(s['RMSE_identities'])==len(rows))
    for r in s['RMSE_identities']:
        value=rows[(r['context'],r['method'])]['log_rmse'];oldvalue=oldrows[(r['context'],'half_maxwell_mean_gaussian')]['log_rmse']
        c.close('saved_RMSE_identity',[r['log_rmse'],r['original_half_log_rmse'],r['difference']],[value,oldvalue,value-oldvalue],rtol=0,atol=2e-12)
    for r in s['mixture_pointwise_bounds']:
        value=rows[(r['context'],r['method'])]['joint_nll_per_dim'];ref='full_anchor_gaussian' if r['method']==METHODS[0] else 'full_anchor_student4'
        endpoint=oldrows[(r['context'],ref)]['joint_nll_per_dim'];gap=value-endpoint
        c.ok('bound_endpoint_identity',r['same_tail_GP_endpoint']==ref)
        c.close('saved_endpoint_NLL',r['endpoint_joint_nll_per_dim'],endpoint)
        c.close('saved_bound_gap',r['actual_gap'],gap)
        c.close('saved_bound_constant',r['bound_log2_per14'],math.log(2)/14,rtol=0,atol=0)
        c.ok('saved_bound_holds',r['holds'] and r['bound_applies_to_mixture_only'] and gap<=math.log(2)/14+2e-10)
    c.ok('bound_grid36',len(s['mixture_pointwise_bounds'])==len(rows)//2)
    return {'contexts_denominator':18,'law_score_rows':len(rows),'scalar_CRPS_terms':14*len(rows),
            'same_point_RMSE_identities':len(rows),'mixture_logscore_bounds':len(s['mixture_pointwise_bounds']),
            'macro_rows':12,'paired_vs_original_rows':36,
            'largest_independent_CDF_quadrature_error_estimate':max_integral_error,
            'new_fits':0,'new_raw_G_reads':0,'production_imports':0,
            'not_new_independent_confirmation':True}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['gate','laws','scores']);parser.add_argument('--registration',required=True)
    parser.add_argument('--gate');parser.add_argument('--directory');parser.add_argument('--report',required=True);args=parser.parse_args()
    report=Path(args.report);stamp=dt.datetime.now(dt.timezone.utc)
    selfreg=report.with_name(report.stem+'_AUDIT_REGISTRATION.json')
    write(selfreg,{'schema':'functional-mixture-independent-audit-registration-v1','actual_utc':stamp.isoformat(),
        'asia_shanghai':stamp.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        'phase':args.phase,'audit_source_sha256':sha(__file__),'independent_math_sha256':sha(MATH),
        'root_registration_sha256':sha(args.registration),'authorization':'Root explicit actual registration and execution instruction',
        'no_SUT_import_no_fit':True,'CDF_quadrature_epsabs':3e-10,'CDF_quadrature_epsrel':3e-10,
        'score_rtol':2e-8,'score_atol':2e-9})
    c=Checks();payload={}
    try:
        if args.phase=='gate':payload=toy(args.registration,Path(args.gate),c)
        elif args.phase=='laws':payload=saved_laws(args.registration,Path(args.directory),c)
        else:payload=saved_scores(args.registration,Path(args.directory),c)
        status='PASS';error=None
    except Exception as exc:
        status='FAIL';error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    result={'schema':'functional-mixture-independent-audit-v1','status':status,'phase':args.phase,'actual_utc':utc(),
        'audit_source_sha256':sha(__file__),'independent_math_sha256':sha(MATH),
        'audit_registration_sha256':sha(selfreg),'root_registration_sha256':sha(args.registration),
        'check_count':c.count,'checks':c.groups,'failure':error,**payload}
    if args.phase=='gate':result['audited_gate_sha256']=sha(args.gate)
    else:
        result['predictions_sha256']=sha(Path(args.directory)/'PREDICTIONS.json')
        if args.phase=='scores':result['scores_sha256']=sha(Path(args.directory)/'SCORES.json')
    write(report,result)
    print(json.dumps({'status':status,'phase':args.phase,'checks':c.count,'report':str(report),'sha256':sha(report),'failure':error['message'] if error else None},ensure_ascii=False))
    if status!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
