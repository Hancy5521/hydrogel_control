"""Independent functional-response audit; no production imports or refitting.

PRE reads saved own12 only, plus input-only metadata and source bytes for SHA.
POST requires a separate explicit root authorization. Raw-source numeric access
is restricted to the selected eighteen contexts' O/T rows and two G columns.
NNLS optimality is checked in independently column-normalized coordinates,
not by trusting small dimensional gradients. No GP hyperparameter optimization.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import itertools
import json
import math
from pathlib import Path
import traceback
import numpy as np
from scipy import integrate, linalg, special, stats

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'sources/generalization_admission_20261109/other_polymer_functions'
INPUT=SOURCE/'darus_functional_gate/INPUT_GATE_RESULT.json'
PLAN=ROOT/'generalization/DARUS_FUNCTIONAL_RESPONSE_PROTOCOL_PREPARED.json'
PROB=('full_anchor_gaussian','half_maxwell_mean_gaussian','maxwell_mean_gaussian',
      'half_maxwell_mean_student4','full_anchor_student4')
POINT=('channel_relative_Maxwell_NNLS_point','raw_Pa_Maxwell_NNLS_point')
ALL=PROB+POINT
INPUT_SHA='10086c3dfd5bab37b252a895cd283966e50e333e526752abe975c59ff9a4a550'


def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def strict_pairs(pairs):
    out={}
    for k,v in pairs:
        if k in out: raise ValueError('Duplicate JSON key '+k)
        out[k]=v
    return out
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'),object_pairs_hook=strict_pairs,
                      parse_constant=lambda token:(_ for _ in ()).throw(ValueError(token)))
def write_new(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


class Checks:
    def __init__(self): self.groups={};self.count=0
    def ok(self,name,condition):
        self.count+=1
        if not condition: raise AssertionError(name)
        entry=self.groups.setdefault(name,{'calls':0,'elements':0,'max_absolute_error':0.0})
        entry['calls']+=1;entry['elements']+=1
    def close(self,name,actual,expected,rtol=2e-8,atol=2e-11):
        a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(name+' nonfinite/shape mismatch')
        error=float(np.max(np.abs(a-b))) if a.size else 0.0
        self.count+=1
        if not np.all(np.abs(a-b)<=atol+rtol*np.abs(b)):
            raise AssertionError(f'{name}: max_abs={error:.17g}; rtol={rtol}; atol={atol}')
        entry=self.groups.setdefault(name,{'calls':0,'elements':0,'max_absolute_error':0.0})
        entry['calls']+=1;entry['elements']+=int(a.size)
        entry['max_absolute_error']=max(entry['max_absolute_error'],error)


def check_binding(reg_path,out,c):
    reg=read(reg_path);p=read(out/'PREPARED.json');f=read(out/'PREDICTIONS.json')
    c.ok('registration_schema',reg['schema']=='darus-functional-mean-transport-registration-v1')
    stamp=dt.datetime.fromisoformat(reg['actual_utc'])
    c.ok('actual_registration_before_forecasts',stamp.tzinfo is not None and
         stamp<=dt.datetime.fromisoformat(f['actual_utc'])<=dt.datetime.now(dt.timezone.utc))
    for name,digest in reg['source_pins'].items(): c.ok('registered_source_pin',sha(name)==digest)
    c.ok('own_source_not_registered_SUT',str(Path(__file__).resolve()) not in reg['source_pins'])
    c.ok('input_gate_pinned',sha(INPUT)==INPUT_SHA)
    c.ok('input_gate_role',read(INPUT)['numeric_Gprime_or_Gdoubleprime_cells_decoded']==0)
    c.ok('same_registration',p['registration_sha256']==f['registration_sha256']==sha(reg_path))
    c.ok('prepared_binding',f['prepared_sha256']==sha(out/'PREPARED.json'))
    c.ok('prepare_source_bindings',p['input_gate_sha256']==sha(INPUT) and p['source_protocol_sha256']==sha(PLAN))
    complete=read(out/'PREDICTION_COMPLETE.json')
    c.ok('prediction_completion_binding',complete['predictions_sha256']==sha(out/'PREDICTIONS.json') and
         complete['anchor_access_sha256']==sha(out/'ANCHOR_ACCESS.json'))
    c.ok('frozen_candidate',reg['main_method']=='half_maxwell_mean_gaussian' and
         reg['mean_transport_fraction']==.5 and reg['methods']==list(PROB) and reg['numeric_moduli_before_registration']==0)
    c.ok('protected_target_scope',f['target_numeric_accesses']==0 and p['numeric_moduli_read']==0 and p['material_fits']==0)
    c.ok('material_not_contexts',f['physical_material_count']==1 and p['independent_specimen_count'] is None)
    return reg,p,f


def maxwell_design(f,tau):
    omega=2*np.pi*np.asarray(f,float);v=omega[:,None]*np.asarray(tau,float)[None,:]
    d=1+v*v
    return np.vstack([np.column_stack([np.ones(len(omega)),v*v/d,np.zeros(len(omega))]),
                      np.column_stack([np.zeros(len(omega)),v/d,omega])])


def nnls_check(result,f,o,t,y,weighted,c):
    tau=np.geomspace(1/(2*np.pi*max(f)),1/(2*np.pi*min(f)),6)
    c.close('Maxwell_fixed_tau',result['fixed_relaxation_times_s'],tau,rtol=2e-14,atol=0)
    if weighted:
        beta=np.array(result['coefficients_Pa_and_Pa_s'])
        sigma=np.linalg.norm(y,axis=1)/math.sqrt(6)
        c.close('weighted_own_RMS',result['channel_RMS_Pa'],sigma,rtol=2e-14,atol=0)
        row_scale=np.repeat(sigma,6)
        c.ok('weighted_not_selected',not result['weights_fitted_or_selected'] and not result['tau_fitted'])
    else:
        b=result['coefficients'];beta=np.array([b['G_infinity_Pa'],*b['G_modes_Pa'],b['eta_solvent_Pa_s']])
        row_scale=np.ones(12)
        c.ok('raw_point_not_probabilistic',not result['probabilistic_scores_supported'])
    A=maxwell_design(np.array(f)[o],tau);At=maxwell_design(np.array(f)[t],tau)
    Aw=A/row_scale[:,None];bw=y.reshape(12)/row_scale
    norms=np.linalg.norm(Aw,axis=0);As=Aw/norms[None,:];gamma=beta*norms
    res=As@gamma-bw;grad=As.T@res
    dual_tol=1e-7*(1+np.linalg.norm(bw)+np.linalg.norm(As,2)*np.linalg.norm(res))
    comp_tol=1e-7*(1+np.linalg.norm(gamma)*max(1,np.linalg.norm(bw)))
    c.ok('NNLS_primal_feasibility',bool(np.all(beta>=0)))
    c.ok('NNLS_dimensionless_dual_feasibility',float(np.min(grad))>=-dual_tol)
    c.ok('NNLS_dimensionless_complementarity',float(np.max(np.abs(gamma*grad)))<=comp_tol)
    pred=At@beta
    c.close('Maxwell_target_prediction',np.r_[result['predicted_Gprime_Pa'],result['predicted_Gdoubleprime_Pa']],pred,rtol=3e-13,atol=1e-7)
    c.ok('Maxwell_target_positive',bool(np.all(pred>0)))
    sse=float(res@res)
    if weighted:
        c.close('weighted_gradient',result['KKT_gradient_weighted_objective'],Aw.T@res,rtol=1e-6,atol=1e-18)
        c.close('weighted_SSE',result['own_weighted_residual_sum_squares'],sse,rtol=2e-10,atol=1e-14)
        c.close('weighted_reported_norm',result['nnls_reported_weighted_residual_norm'],math.sqrt(sse),rtol=1e-10,atol=1e-14)
    else:
        c.close('raw_own_fit',np.r_[result['own_observed_Gprime_fit_Pa'],result['own_observed_Gdoubleprime_fit_Pa']],A@beta,rtol=3e-13,atol=1e-7)
        c.close('raw_SSE',result['own_observed_residual_sum_squares_Pa2'],sse,rtol=1e-10,atol=1e-10)
        c.close('raw_reported_norm',result['nnls_reported_residual_norm_Pa'],math.sqrt(sse),rtol=1e-10,atol=1e-12)
    c.ok('NNLS_no_target_received',result['target_responses_received']==0)
    return np.log(pred),{'smallest_scaled_dual_gradient':float(np.min(grad)),
                         'largest_scaled_complementarity':float(np.max(np.abs(gamma*grad))),
                         'dual_tolerance':float(dual_tol),'complementarity_tolerance':float(comp_tol)}


def gp_check(result,xo,y,xt,c):
    X=np.column_stack([np.ones(6),xo]);Xt=np.column_stack([np.ones(7),xt])
    Q,R=np.linalg.qr(X,mode='reduced');beta=linalg.solve_triangular(R,Q.T@y)
    residual=y-X@beta;sse=float(residual@residual);scale=math.sqrt(sse/4)
    resolution=64*np.finfo(float).eps*(np.linalg.norm(X,2)*np.linalg.norm(beta)+np.linalg.norm(y))
    c.close('GP_OLS_beta_QR',result['OLS_beta'],beta,rtol=1e-8,atol=3e-11)
    c.close('GP_SSE_QR',result['SSE'],sse,rtol=1e-7,atol=1e-18)
    c.close('GP_resolution',result['residual_norm_resolution'],resolution,rtol=2e-10,atol=1e-20)
    c.close('GP_resolution_squared',result['SSE_resolution_threshold'],resolution**2,rtol=1e-9,atol=1e-35)
    c.ok('GP_resolved_SSE',sse>resolution**2 and result['rank_OLS']==2)
    c.close('GP_residual_scale',result['residual_scale'],scale,rtol=5e-8,atol=1e-15)
    u=residual/scale
    c.close('GP_standardized_residual',result['standardized_anchor_residual'],u,rtol=2e-7,atol=2e-9)
    signal,length,noise=np.exp(np.array(result['kernel_theta_log']))
    hyper=result['hyperparameters']
    c.close('GP_saved_hyperparameters',[hyper['signal_variance_standardized'],hyper['length_scale_log_gamma'],hyper['white_variance_standardized']],[signal,length,noise],rtol=1e-12,atol=0)
    c.close('GP_log_white_variance',hyper['white_variance_log_eta'],scale*scale*noise,rtol=1e-7,atol=1e-20)
    c.ok('GP_bounds',1e-4*(1-1e-12)<=signal<=1e4*(1+1e-12) and .01*(1-1e-12)<=length<=100*(1+1e-12) and 1e-6*(1-1e-12)<=noise<=1+1e-12)
    def kernel(a,b):
        d=math.sqrt(3)*np.abs(a[:,None]-b[None,:])/length
        return signal*(1+d)*np.exp(-d)
    K=kernel(xo,xo)+noise*np.eye(6);B=kernel(xt,xo);D=kernel(xt,xt)+noise*np.eye(7)
    L=np.linalg.cholesky(K);sol=linalg.cho_solve((L,True),u)
    mean=Xt@beta+scale*B@sol
    cov=scale*scale*(D-B@linalg.cho_solve((L,True),B.T));cov=(cov+cov.T)/2
    np.linalg.cholesky(cov)
    c.close('GP_given_kernel_mean',result['mean'],mean,rtol=1e-9,atol=1e-8)
    c.close('GP_given_kernel_full_covariance',result['covariance'],cov,rtol=2e-7,atol=2e-13)
    ll=-.5*float(u@sol)-np.log(np.diag(L)).sum()-3*math.log(2*math.pi)
    c.close('GP_given_kernel_log_marginal_likelihood',result['log_marginal_likelihood_standardized'],ll,rtol=2e-8,atol=2e-7)
    c.close('GP_scale_jacobian',result['log_marginal_likelihood_log_eta'],ll-6*math.log(scale),rtol=2e-8,atol=2e-7)
    c.ok('GP_no_rescue_no_target',result['alpha']==0 and not result['normalize_y'] and
         result['n_restarts_optimizer']==0 and not result['target_Y_read'] and
         not result['OLS_mean_uncertainty_integrated'] and not result['hyperparameter_posterior_uncertainty_integrated'])
    return mean,cov


def pre(reg_path,out,c):
    reg,pred_input,p=check_binding(reg_path,out,c)
    gate=read(INPUT);plan=read(PLAN)['first_bounded_stage'];byid={r['condition_id']:r for r in gate['condition_curves']}
    selected=plan['selected_input_conditions'];masks=plan['masks'];contexts=pred_input['contexts']
    c.ok('planned_context_counts',len(contexts)==len(p['forecasts'])==18 and len(selected)==6 and len(masks)==3)
    expected_ids={s['condition_id']+'|'+mask for s in selected for mask in masks}
    c.ok('exact_context_set',len({r['context'] for r in contexts})==18 and {r['context'] for r in contexts}==expected_ids=={r['context'] for r in p['forecasts']})
    # Recheck input-only condition selection, not response-dependent case selection.
    for s in selected:
        options=[r for r in gate['condition_curves'] if r['state_block']==s['state_block']]
        chosen=min(options,key=lambda r:(abs(np.median([n['temperature_C'] for n in r['native_rows']])-30),r['result_no'],r['interval_no']))
        c.ok('input_only_temperature_selection',s['condition_id']==chosen['condition_id'])
    prepared={r['context']:r for r in contexts};ledger=read(out/'ANCHOR_ACCESS.json')
    expected_access=[];diagnostics=[];states=collections.Counter();law_count=0;gp_count=0
    for f in p['forecasts']:
        q=prepared[f['context']];r=byid[q['condition_id']];o=q['O'];t=q['T'];freq=np.array(q['frequency_Hz'])
        c.ok('prepared_original_identity',q['native_rows']==r['native_rows'] and q['source_sha256']==r['source_sha256'] and q['file']==r['file'] and q['state_block']==r['state_block'])
        c.ok('original_source_bytes',sha(SOURCE/'darus_smp_originals'/q['file'])==q['source_sha256'])
        c.ok('same_context_identity',all(f[k]==q[k] for k in ['context','condition_id','state_block','file','mask','O','T']))
        c.ok('fixed_mask',o==masks[q['mask']]['O'] and t==masks[q['mask']]['T'] and len(o)==6 and len(t)==7 and not set(o)&set(t) and set(o)|set(t)==set(range(13)))
        c.close('native_frequency_input',[n['frequency_Hz'] for n in q['native_rows']],freq,rtol=0,atol=0)
        c.ok('frequency_order_positive',len(freq)==13 and bool(np.all(np.diff(freq)>0)) and freq[0]>0)
        c.ok('response_units',q['source_modulus_units']=='MPa' and q['working_modulus_units']=='Pa')
        for i in o:
            for channel,col in ((0,6),(1,7)):
                expected_access.append((f['context'],'own_observed6_frequency_12_scalar',q['file'],q['native_rows'][i]['source_line_1based'],col,channel,'MPa','Pa',1000000))
        y=np.array(f['observed_moduli_Pa'],dtype=float)
        c.ok('only_own12_payload',y.shape==(2,6))
        components=f['components'];gp=components['gp'];weighted=components['weighted_maxwell'];raw=components['raw_Pa_maxwell']
        gp_ok=gp['status']=='OK';w_ok=weighted['status']=='OK';r_ok=raw['status']=='OK'
        expected={PROB[0]:gp_ok,PROB[1]:gp_ok and w_ok,PROB[2]:gp_ok and w_ok,
                  PROB[3]:gp_ok and w_ok,PROB[4]:gp_ok,POINT[0]:w_ok,POINT[1]:r_ok}
        c.ok('component_availability',set(f['method_availability'])==set(ALL) and
             all(f['method_availability'][k]['available']==expected[k] for k in ALL))
        expected_status='OK' if all(expected.values()) else 'PARTIAL' if any(expected.values()) else 'FAIL'
        c.ok('context_status',f['status']==expected_status);states[f['status']]+=1
        c.ok('exact_available_laws',set(f['laws'])=={k for k in PROB if expected[k]})
        if gp_ok:
            c.ok('GP_both_channels',len(gp['channels'])==2 and all(r['status']=='OK' for r in gp['channels']))
            parts=[gp_check(gp['channels'][j]['result'],np.log(2*np.pi*freq[o]),np.log(y[j]),np.log(2*np.pi*freq[t]),c) for j in range(2)]
            gm=np.concatenate([r[0] for r in parts]);C=linalg.block_diag(*[r[1] for r in parts]);gp_count+=2
            c.close('complete_GP_mean',f['gp_mean'],gm,rtol=1e-9,atol=1e-8)
            c.close('complete_GP_covariance',f['full_residual_covariance'],C,rtol=2e-7,atol=2e-13)
            c.ok('complete_covariance_zero_cross_channel',bool(np.all(np.array(f['full_residual_covariance'])[:7,7:]==0)))
        if w_ok:
            wm,wdiag=nnls_check(weighted['result'],freq,o,t,y,True,c)
            c.close('Maxwell_log_location',f['maxwell_log_location'],wm,rtol=1e-12,atol=1e-12)
            c.ok('weighted_result_retained',f['maxwell_point']==weighted['result'])
        if r_ok:
            _,rdiag=nnls_check(raw['result'],freq,o,t,y,False,c)
            c.ok('raw_result_retained',f['raw_Pa_maxwell_point']==raw['result'])
        for method,law in f['laws'].items():
            m=gm if method in (PROB[0],PROB[4]) else .5*(gm+wm) if method in (PROB[1],PROB[3]) else wm
            if method in (PROB[0],PROB[1],PROB[2]):
                c.ok('Gaussian_distribution',law['distribution']=='multivariate_gaussian_log_modulus')
                c.close('law_mean',law['mean'],m,rtol=1e-9,atol=1e-8)
                c.close('full_C_retained',law['covariance'],C,rtol=2e-7,atol=2e-13)
            else:
                c.ok('Student_df4',law['df']==4)
                c.close('Student_location',law['location'],m,rtol=1e-9,atol=1e-8)
                c.close('Student_shape_half_C',law['scale_matrix'],.5*C,rtol=2e-7,atol=2e-13)
                c.close('Student_predictive_covariance',law['predictive_covariance'],C,rtol=2e-7,atol=2e-13)
            law_count+=1
        c.ok('no_target_or_old_prior',f['target_response_values_received']==0 and not f['old_viscosity_parameters_loaded'] and
             not f['learned_transport_fraction'] and f['observed_scalar_budget']==12 and f['target_scalar_dimension']==14 and
             not f['generated_residual_draws_guaranteed_Maxwell_or_passive'])
        diagnostics.append({'context':f['context'],'status':f['status'],
                            'weighted_KKT':wdiag if w_ok else None,'raw_Pa_KKT':rdiag if r_ok else None})
    actual_access=[tuple(a[k] for k in ['context','role','file','source_line_1based','column_0based','channel','source_unit','working_unit','exact_multiplier']) for a in ledger['access_sites']]
    c.ok('exact_own_role_access_multiset',collections.Counter(actual_access)==collections.Counter(expected_access))
    unique=len({(a[2],a[3],a[4]) for a in expected_access})
    c.ok('access_counts',ledger['numeric_scalar_accesses']==len(actual_access)==216 and ledger['unique_raw_scalar_cells']==unique==144)
    c.ok('no_nonselected_or_extra_columns',ledger['non_selected_temperature_numeric_responses_read']==0 and ledger['other_response_columns_decoded']==0)
    for row in p['method_availability']:
        ids=[f['context'] for f in p['forecasts'] if f['method_availability'][row['method']]['available']]
        c.ok('method_denominator18',row['planned_contexts']==18 and row['available_context_ids']==ids and row['available_contexts']==len(ids) and row['unavailable_contexts']==18-len(ids))
        c.close('failure_rate_denominator18',row['failure_rate'],(18-len(ids))/18,rtol=0,atol=0)
    c.ok('top_counts',p['probability_laws']==law_count and p['contexts_fully_available']==states['OK'] and
         p['contexts_partial']==states['PARTIAL'] and p['contexts_all_methods_failed']==states['FAIL'])
    return {'contexts':18,'condition_states':6,'probability_laws_audited':law_count,
            'given_kernel_GP_channels_audited':gp_count,'weighted_NNLS_audited':18,
            'raw_Pa_NNLS_audited':18,'context_status_counts':dict(states),
            'scalar_anchor_role_accesses':216,'distinct_original_anchor_scalar_cells':unique,
            'held_target_G_numeric_reads_by_this_auditor':0,'GP_hyperparameter_refits':0,'production_imports':0,
            'diagnostics':diagnostics,'limitation':'Context-specific masking reuses original cells across masks; one source material, not18 independent materials or specimens. Given-kernel reconstruction certifies arithmetic, not global hyperparameter optimality.'}


def source_values(q,indices,phase,access):
    """POST-only scoped original numeric access. Never invoked by PRE."""
    wanted={q['native_rows'][i]['source_line_1based']:i for i in indices};found={}
    with (SOURCE/'darus_smp_originals'/q['file']).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);header=next(reader);unit=next(reader)
        columns=[header.index('Storage Modulus'),header.index('Loss Modulus')]
        if [unit[j] for j in columns]!=['[MPa]','[MPa]']: raise AssertionError('MPa units changed')
        for row in reader:
            if reader.line_num not in wanted: continue
            i=wanted[reader.line_num]
            if float(row[header.index('Frequency')])!=q['frequency_Hz'][i]: raise AssertionError('row frequency differs')
            values=[]
            for ch,col in enumerate(columns):
                try: number=float(row[col])*1e6
                except ValueError: number=float('nan')
                values.append(number)
                access.append({'context':q['context'],'phase':phase,'file':q['file'],'line':reader.line_num,'column':col,'channel':ch})
            found[i]=values
    if set(found)!=set(indices): raise AssertionError('Scoped original rows incomplete')
    return np.asarray([found[i] for i in indices]).T


def independent_score(law,y):
    if law['distribution']=='multivariate_gaussian_log_modulus':
        m=np.asarray(law['mean']);C=np.asarray(law['covariance']);s=np.sqrt(np.diag(C));z=(y-m)/s
        crps=s*(z*special.erf(z/math.sqrt(2))+2*np.exp(-z*z/2)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
        residual=y-m;L=np.linalg.cholesky(C);a=linalg.solve_triangular(L,residual,lower=True)
        nll=(.5*a@a+np.log(np.diag(L)).sum()+7*math.log(2*math.pi))/14
        quant=m+stats.norm.ppf([.05,.5,.95])[:,None]*s
    else:
        m=np.asarray(law['location']);C=np.asarray(law['scale_matrix']);s=np.sqrt(np.diag(C));z=(y-m)/s
        # Derive Gini constant with independent CDF energy integral, not production closed constant.
        crps=s*(z*(2*stats.t.cdf(z,4)-1)+2*(4+z*z)/3*stats.t.pdf(z,4)-STUDENT_GINI)
        residual=y-m;L=np.linalg.cholesky(C);a=linalg.solve_triangular(L,residual,lower=True);d=14;nu=4
        logdet=2*np.log(np.diag(L)).sum()
        logp=special.gammaln((nu+d)/2)-special.gammaln(nu/2)-.5*(d*math.log(nu*math.pi)+logdet)-.5*(nu+d)*math.log1p(float(a@a)/nu)
        nll=-logp/d;quant=m+stats.t.ppf([.05,.5,.95],4)[:,None]*s
    metrics={'joint_nll_per_dim':float(nll),'log_crps':float(crps.mean()),
             'pointwise_log_crps':crps,'log_rmse':float(np.sqrt(np.mean((y-m)**2))),
             'coverage90':float(np.mean((y>=quant[0])&(y<=quant[2]))),
             'log_width90':float(np.mean(quant[2]-quant[0]))}
    channels=[]
    for j,name in enumerate(('Gprime','Gdoubleprime')):
        sl=slice(j*7,(j+1)*7)
        channels.append({'channel':name,'log_rmse':float(np.sqrt(np.mean((y[sl]-m[sl])**2))),
                         'log_crps':float(crps[sl].mean()),'coverage90':float(np.mean((y[sl]>=quant[0,sl])&(y[sl]<=quant[2,sl]))),
                         'log_width90':float(np.mean(quant[2,sl]-quant[0,sl]))})
    return metrics,channels


STUDENT_GINI=None


def post(reg_path,out,authorization,c):
    global STUDENT_GINI
    auth=read(authorization)
    if auth.get('authorization')!='AUDIT_RELEASED_FUNCTIONAL_TARGETS_AND_SCORES':
        raise PermissionError('Separate root explicit POST audit authorization required')
    c.ok('post_authorization_nonfuture',dt.datetime.fromisoformat(auth['actual_utc'])<=dt.datetime.now(dt.timezone.utc))
    for key,file in [('predictions_sha256','PREDICTIONS.json'),('targets_sha256','TARGETS.json'),('scores_sha256','SCORES.json')]:
        c.ok('post_exact_authorization',auth[key]==sha(out/file))
    reg,pred_input,p=check_binding(reg_path,out,c);targets=read(out/'TARGETS.json');scores=read(out/'SCORES.json')
    pre_report=read(out/'INDEPENDENT_PRE_Y.json')
    c.ok('prior_PRE_PASS',pre_report['status']=='PASS' and pre_report['predictions_sha256']==sha(out/'PREDICTIONS.json') and targets['pre_audit_sha256']==sha(out/'INDEPENDENT_PRE_Y.json'))
    c.ok('post_payload_bindings',targets['predictions_sha256']==scores['predictions_sha256']==sha(out/'PREDICTIONS.json') and scores['targets_sha256']==sha(out/'TARGETS.json') and scores['registration_sha256']==sha(reg_path))
    STUDENT_GINI=2*integrate.quad(lambda z:float(stats.t.cdf(z,4)*stats.t.sf(z,4)),0,np.inf,epsabs=3e-12,epsrel=3e-12)[0]
    c.close('Student_Gini_independent_CDF_energy',STUDENT_GINI,15*math.pi/64,rtol=0,atol=3e-12)
    qs={q['context']:q for q in pred_input['contexts']};ys={t['context']:t for t in targets['contexts']}
    actual_rows={(r['context'],r['method']):r for r in scores['rows']+scores['point_rows']}
    expected_rows={};access=[];raw_terms=0;direct_integrals=0
    for f in p['forecasts']:
        q=qs[f['context']];own=source_values(q,q['O'],'released_post_original_O',access)
        target=source_values(q,q['T'],'released_post_original_T',access)
        c.close('original_O_MPa_to_Pa',f['observed_moduli_Pa'],own,rtol=0,atol=0)
        c.close('original_T_MPa_to_Pa',ys[f['context']]['values_Pa'],target,rtol=0,atol=0)
        admitted=bool(np.isfinite(target).all() and np.all(target>0))
        c.ok('target_admission',ys[f['context']]['target_admissible']==admitted)
        if not admitted: continue
        y=np.log(target.reshape(14))
        for method in PROB:
            if not f['method_availability'][method]['available']: continue
            law=f['laws'][method];expected,channels=independent_score(law,y);key=(f['context'],method)
            row=actual_rows[key];expected_rows[key]={'context':f['context'],'state_block':f['state_block'],'mask':f['mask'],'method':method,**expected}
            for k,v in expected.items(): c.close('score_'+k,row[k],v,rtol=5e-9,atol=3e-9)
            c.ok('channel_labels',[r['channel'] for r in row['channels']]==[r['channel'] for r in channels])
            for a,b in zip(row['channels'],channels):
                for k in ('log_rmse','log_crps','coverage90','log_width90'):c.close('channel_'+k,a[k],b[k],rtol=5e-9,atol=3e-9)
            raw_terms+=14
            # One independent direct CDF integral per context and distribution family.
            if method in ('full_anchor_gaussian','full_anchor_student4'):
                m=np.array(law['mean'] if method==PROB[0] else law['location']);s=np.sqrt(np.diag(np.array(law['covariance'] if method==PROB[0] else law['scale_matrix'])))
                z=float((y[0]-m[0])/s[0]);dist=stats.norm if method==PROB[0] else stats.t(4)
                direct=s[0]*(integrate.quad(lambda x:float(dist.cdf(x)**2),-np.inf,z,epsabs=2e-10,epsrel=2e-10)[0]+
                             integrate.quad(lambda x:float(dist.sf(x)**2),z,np.inf,epsabs=2e-10,epsrel=2e-10)[0])
                c.close('direct_CDF_CRPS',expected['pointwise_log_crps'][0],direct,rtol=1e-8,atol=1e-9);direct_integrals+=1
        for field,method in [('maxwell_point',POINT[0]),('raw_Pa_maxwell_point',POINT[1])]:
            if not f['method_availability'][method]['available']:continue
            point=f[field];m=np.log(np.r_[point['predicted_Gprime_Pa'],point['predicted_Gdoubleprime_Pa']]);key=(f['context'],method)
            value=float(np.sqrt(np.mean((y-m)**2)));row=actual_rows[key]
            c.close('point_log_RMSE',row['log_rmse'],value,rtol=1e-12,atol=1e-12)
            for j in range(2):c.close('point_channel_RMSE',row['channels'][j]['log_rmse'],np.sqrt(np.mean((y[7*j:7*(j+1)]-m[7*j:7*(j+1)])**2)),rtol=1e-12,atol=1e-12)
            c.ok('point_has_no_probability_scores',row['probability_scores_defined'] is False and 'joint_nll_per_dim' not in row)
            expected_rows[key]={'context':f['context'],'state_block':f['state_block'],'mask':f['mask'],'method':method,'log_rmse':value}
    c.ok('all_score_rows_no_deletion_or_duplicate',set(actual_rows)==set(expected_rows) and len(actual_rows)==len(scores['rows'])+len(scores['point_rows']))
    for row in scores['macro_by_mask']:
        subset=[r for r in expected_rows.values() if r['mask']==row['mask'] and r['method']==row['method']]
        c.ok('mask_macro_denominator6',row['planned_state_conditions']==6 and row['scored_state_conditions']==len(subset) and row['failed_or_unscored']==6-len(subset) and row['successful_case_diagnostic_only']==(len(subset)!=6))
        keys=('joint_nll_per_dim','log_crps','log_rmse','coverage90','log_width90') if row['method'] in PROB else ('log_rmse',)
        for k in keys:
            if subset:c.close('equal_state_macro_'+k,row[k],np.mean([r[k] for r in subset]),rtol=5e-9,atol=3e-9)
            else:c.ok('empty_macro_None',row[k] is None)
    c.ok('macro_cardinality',len(scores['macro_by_mask'])==21)
    for row in scores['paired_comparisons']:
        a={r['context']:r for r in expected_rows.values() if r['mask']==row['mask'] and r['method']==row['method_a']}
        b={r['context']:r for r in expected_rows.values() if r['mask']==row['mask'] and r['method']==row['method_b']}
        common=sorted(set(a)&set(b));c.ok('paired_availability6',row['planned_state_conditions']==6 and row['paired_state_conditions']==len(common) and row['unpaired_or_failed']==6-len(common) and row['conditional_paired_subset']==(len(common)!=6))
        c.ok('paired_context_set',[r['context'] for r in row['paired_context_deltas']]==common)
        keys=['log_rmse']+(['joint_nll_per_dim','log_crps','coverage90','log_width90'] if row['method_a'] in PROB and row['method_b'] in PROB else [])
        for k in keys:
            values=[a[i][k]-b[i][k] for i in common]
            if common:
                c.close('paired_case_'+k,[r[k] for r in row['paired_context_deltas']],values,rtol=5e-9,atol=3e-9)
                c.close('paired_macro_'+k,row['mean_delta_'+k],np.mean(values),rtol=5e-9,atol=3e-9)
            else:c.ok('empty_pair_None',row['mean_delta_'+k] is None)
    c.ok('paired_cardinality',len(scores['paired_comparisons'])==63)
    c.ok('all18_denominator',scores['all_contexts_denominator']==18 and len(scores['method_context_availability'])==126)
    for row in scores['method_context_availability']:
        forecast=next(f for f in p['forecasts'] if f['context']==row['context']);prediction=forecast['method_availability'][row['method']]['available'];target=ys[row['context']]['target_admissible']
        c.ok('availability_each_case',row['prediction_available']==prediction and row['target_log_admissible']==target and row['scored']==(prediction and target))
    for row in scores['method_score_availability']:
        count=sum(r['method']==row['method'] for r in expected_rows.values())
        c.ok('method_score_denominator18',row['planned_contexts']==18 and row['scored_contexts']==count and row['unavailable_or_unscored']==18-count and row['complete_cohort_scored']==(count==18))
        c.close('method_score_failure_rate',row['failure_rate'],(18-count)/18,rtol=0,atol=0)
    c.ok('material_scope_unchanged',scores['physical_material_count']==1 and scores['independent_specimen_count'] is None and not scores['independent_material_confirmation'] and not scores['original_resin_final20_accessed'])
    return {'contexts':18,'probability_rows':len(scores['rows']),'probability_scalar_terms':raw_terms,
            'point_rows':len(scores['point_rows']),'direct_CDF_integrals':direct_integrals,
            'macro_rows':21,'paired_rows':63,'case_method_availability':126,
            'original_scalar_role_accesses':len(access),'distinct_original_scalar_cells':len({(r['file'],r['line'],r['column']) for r in access}),
            'new_material_measurements':0,'production_imports':0,'model_refits':0,
            'source_access_log':access,'primary_mask':'interior_holes','main_method':'half_maxwell_mean_gaussian'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['pre','post']);parser.add_argument('--registration',required=True);parser.add_argument('--output',required=True);parser.add_argument('--authorization');parser.add_argument('--report');args=parser.parse_args()
    out=Path(args.output);report=Path(args.report) if args.report else out/('INDEPENDENT_PRE_Y.json' if args.phase=='pre' else 'INDEPENDENT_POST_Y.json')
    selfreg=out/('INDEPENDENT_'+args.phase.upper()+'_AUDIT_REGISTRATION.json')
    # Root explicitly authorized PRE by team message; POST requires exact-hash file.
    registration={'schema':'darus-functional-independent-audit-registration-v1','actual_utc':utc(),
                  'phase':args.phase,'source_path':str(Path(__file__).resolve()),'source_sha256':sha(__file__),
                  'root_registration_sha256':sha(args.registration),'prepared_sha256':sha(out/'PREPARED.json'),
                  'predictions_sha256':sha(out/'PREDICTIONS.json'),
                  'authorization':'Root team message permits saved own12 PRE only' if args.phase=='pre' else args.authorization,
                  'production_imports':0,'refits':0,'read_original_modulus_cells':args.phase=='post',
                  'NNLS_KKT_tolerance_relative':1e-7,'GP_covariance_rtol':2e-7,'GP_mean_atol':1e-8}
    write_new(selfreg,registration);c=Checks();payload={}
    try:
        if args.phase=='pre':payload=pre(args.registration,out,c)
        else:
            if not args.authorization:raise PermissionError('POST authorization required')
            payload=post(args.registration,out,args.authorization,c)
        status='PASS';error=None
    except Exception as exc:
        status='FAIL';error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    result={'schema':'darus-functional-independent-audit-v1','status':status,'actual_utc':utc(),
            'phase':args.phase,'predictions_sha256':sha(out/'PREDICTIONS.json'),
            'prepared_sha256':sha(out/'PREPARED.json'),'audit_source_sha256':sha(__file__),
            'audit_registration_sha256':sha(selfreg),'check_count':c.count,'checks':c.groups,
            'failure':error,**payload}
    if args.phase=='post':result.update(targets_sha256=sha(out/'TARGETS.json'),scores_sha256=sha(out/'SCORES.json'))
    write_new(report,result)
    print(json.dumps({'status':status,'phase':args.phase,'checks':c.count,'report':str(report),'sha256':sha(report),'failure':error['message'] if error else None},ensure_ascii=False))
    if status!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
