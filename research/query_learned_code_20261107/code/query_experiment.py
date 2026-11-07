"""Four genuine feedback batches, final10 forecast freeze, then one384 census.

No hidden-label simulation of querying: every selected source label is released
only after a persisted exact ticket. Queried128 train every936 baseline equally.
"""
import argparse
from fractions import Fraction as F
import json
import math
from pathlib import Path
import random
import secrets
from statistics import NormalDist
import query_io as io
from train_forecasts import core_modules,empirical_candidate,fit_record,jsonable

METHODS=('F808','priorglobal','priorcontext','Beta','geometryRiskCode','plainQueryERM',
         'RF936','GP936','CRPS936','NGBoost936')


def frozen_initial(p):
    pin=p['inputs']['initial_freeze'];io.verify_pin(pin);f=io.read(pin['file'])
    io.require(f['status']=='INITIAL512_F808_PRIORS_GEOMETRY_FROZEN_BEFORE_CURRENT_Y'
               and f['plan_sha256']==p['_plan_sha256'],'immutable initial forecast phase required')
    io.verify_payloads(Path(pin['file']).parent,f['files_sha256'])
    return pin,Path(pin['file']).parent


def adaptive_queries(p,folder):
    io.require(p['_phase']=='adaptive_queries','registered actualquery phase required')
    freeze,initial=frozen_initial(p);geo,rc=core_modules(p)
    contexts=io.read(initial/'CONTEXTS512.json')['records'];preds=io.read(initial/'INITIAL512_LAWS.json')['records']
    io.require([r['component_id'] for r in contexts]==[r['component_id'] for r in preds],
               'frozen512 context/law order mismatch')
    ids=[r['component_id'] for r in contexts]
    geos=[geo.geometry(r['F808']).lift(c['X'][7]) for r,c in zip(preds,contexts)]
    prior=tuple(map(F,io.read(initial/'PRIOR_HEADS.json')['context']['theta']));theta=prior
    # Draw seed only now, after verifying initial forecast freeze. Seed is saved,
    # not secretly changed if draws or later outcomes prove inconvenient.
    seed_bytes=secrets.token_bytes(32);seed_int=int.from_bytes(seed_bytes,'big')
    io.write(folder/'SEED.json',{'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],
        'forecast_freeze_sha256':freeze['sha256'],'seed_hex':seed_bytes.hex(),
        'generator':'Python random.Random integer-seeded MT19937; exact randrange tickets',
        'OS_entropy_requested_after_forecast_freeze':True})
    seedpin=io.pin(folder/'SEED.json');rng=random.Random(seed_int);state=rc.QueryState(geos)
    reader=io.TrustedReader(p,io.pool(p));actual=[];query_indices=[];query_targets=[];fits=[]
    for batch in range(4):
        state.start_batch(theta)
        io.write(folder/('BATCH_'+str(batch+1)+'_POLICY.json'),{'actual_utc':io.utc(),
            'batch':batch+1,'past_queries':len(actual),'theta_before_batch':list(map(str,theta)),
            'mixture':['1/4','3/8','3/8'],'width_max_coordinate_exact':list(map(str,state.widths)),
            'current_gap_width_exact':list(map(str,state.gaps)),
            'formula':'q_i=1/4/n+3/8*w_i/sum(w)+3/8*g_i/sum(g); zero-sum component=uniform',
            'gap_width_uses_only_frozenCDFs_and_past_batch_theta':True})
        for _ in range(32):
            before=list(state.remaining);draw=state.draw(rng);index=draw['index'];key=ids[index];t=len(actual)+1
            ticketpath=folder/('TICKET_'+str(t).zfill(3)+'.json')
            ticket={'status':'EXACT_PREDRAW_TICKET_PERSISTED_BEFORE_Y','actual_utc':io.utc(),
                'plan_sha256':p['_plan_sha256'],'forecast_freeze_sha256':freeze['sha256'],
                'seed_sha256':seedpin['sha256'],'component_id':key,'query_index':t,'batch':batch+1,
                'remaining_indices_before':before,'policy_file':'BATCH_'+str(batch+1)+'_POLICY.json',
                **jsonable(draw)}
            io.write(ticketpath,ticket)  # exclusive creation and close BEFORE trueY
            permit={'component_id':key,'plan_sha256':p['_plan_sha256'],'forecast_freeze':freeze,
                    'seed':seedpin,'ticket':io.pin(ticketpath)}
            target=reader.reveal(key,permit);observation=state.observe(target)
            observation.update(actual_utc=io.utc(),component_id=key,native_target=target,
                ticket_sha256=io.sha(ticketpath),true_source='original computed bulk report, not new wet/DFT')
            io.write(folder/('OBSERVE_'+str(t).zfill(3)+'.json'),jsonable(observation))
            actual.append({'component_id':key,'identity':contexts[index]['identity'],'native_target':target,
                           'query_index':t,'context_index':index,'ticket_sha256':io.sha(ticketpath)})
            query_indices.append(index);query_targets.append(target)
        Q=state.remaining_Q();chat=state.remaining_estimate()
        fit=rc.fit_code(Q,chat,prior,len(state.remaining));theta=fit.theta;fits.append(fit)
        io.write(folder/('BATCH_'+str(batch+1)+'_UPDATE.json'),{'actual_utc':io.utc(),
            'queries_complete':len(actual),'remaining_indices':list(state.remaining),'Q_remaining_exact':jsonable(Q),
            'Chat_remaining_exact':jsonable(chat),'sum_Z_total_exact':jsonable(state.sum_Z),
            'sum_c_queried_exact':jsonable(state.sum_c),'prior_theta_penalty_center_fixed808':list(map(str,prior)),
            'fit':fit_record(fit),'fallback_is_prior_and_not_success':not fit.accepted,
            'projection_is_point_estimate_not_CS':True})
    io.require(len(actual)==128 and len(state.remaining)==384,'128/384 budget mismatch')
    erm=rc.fit_query_erm([geos[i] for i in query_indices],query_targets,state.remaining_Q(),prior,384)
    io.write(folder/'PLAIN_QUERY_ERM.json',{'actual_utc':io.utc(),'fit':fit_record(erm),
        'scope':'ordinary unit-weight queried CRPS, adaptive sample; not unbiased active ERM',
        'same_actual128_labels_and_same_prior_tau':True})
    io.write(folder/'ACTUAL128_TARGETS.json',{'records':actual,'synthetic_or_predicted_targets':0})
    remaining=[ids[i] for i in state.remaining]
    io.write(folder/'QUERY_CODE_REMAINING_LAWS.json',{'records':[{'component_id':ids[i],
        'geometryRiskCode':empirical_candidate(geos[i],theta),
        'plainQueryERM':empirical_candidate(geos[i],erm.theta)} for i in state.remaining],
        'fit_control':{'geometryRiskCode_accepted':fits[-1].accepted,'plainQueryERM_accepted':erm.accepted},
        'no_confidence_gate_forcing_zero':True,'prior808_methods_have_less_label_information':True})
    io.write(folder/'COMPLETE.json',{'status':'ACTUAL128_QUERIES_AND_FOUR_UPDATES_COMPLETE_BEFORE_REMAINING_Y',
        'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],'initial_freeze_sha256':freeze['sha256'],
        'seed':seedpin,'queried_ids':[r['component_id'] for r in actual],'remaining_ids':remaining,
        'actual_queries':128,'adaptive_QP_tasks':4,'ordinary_query_ERM_QP_tasks':1,
        'adaptive_optimizer_calls':sum(f.diagnostics['optimizer_calls'] for f in fits),
        'queryERM_optimizer_calls':erm.diagnostics['optimizer_calls'],
        'files_sha256':{f.name:io.sha(f) for f in folder.iterdir() if f.is_file()},
        'remaining_targets_read':0,'confirm_extra_property_X_Y_disclosed_to_models':0,'confirm_identity_composition_metadata_already_accessible':True})



def law_summary(law):
    """Deterministic native summaries; empirical generalized inverse is exact."""
    probabilities=(F(1,40),F(1,20),F(1,10),F(1,2),F(9,10),F(19,20),F(39,40))
    if law['family']=='weighted_empirical':
        mass={}
        for x,w in zip(law['support_native'],law['weight_fractions']):
            weight=F(w);point=float(x)
            io.require(weight>=0 and math.isfinite(point),'valid finite empirical atom')
            if weight:mass[point]=mass.get(point,F())+weight
        io.require(mass and sum(mass.values(),F())==1,'exact empirical normalization')
        atoms=sorted(mass.items());values={};running=F();todo=list(probabilities)
        for point,weight in atoms:
            running+=weight
            while todo and running>=todo[0]:values[str(todo.pop(0))]=point
        io.require(not todo,'exact quantile probabilities not reached')
        exact_mean=sum((F.from_float(point)*weight for point,weight in atoms),F())
        mean=float(exact_mean);mean_exact=str(exact_mean)
        quantile_rule='inf{x:F(x)>=p}, exact Fraction cumulative probability; selected native atom'
    else:
        io.require(law['family']=='normal','registered Normal/empirical only')
        mean=float(law['location']);sd=float(law['scale'])
        io.require(math.isfinite(mean) and math.isfinite(sd) and sd>0,'finite positive NormalSD')
        dist=NormalDist(mean,sd);values={str(v):dist.inv_cdf(float(v)) for v in probabilities}
        mean_exact=None;quantile_rule='statistics.NormalDist.inv_cdf, deterministic analytic Normal quantile'
    intervals={}
    for level,lo,hi in (('80',F(1,10),F(9,10)),('90',F(1,20),F(19,20)),('95',F(1,40),F(39,40))):
        lower,upper=values[str(lo)],values[str(hi)]
        io.require(math.isfinite(lower) and math.isfinite(upper) and lower<=upper,'finite ordered native interval')
        intervals[level]={'lower':lower,'upper':upper,'width':upper-lower,'tail_probability_exact':str(lo)}
    io.require(math.isfinite(mean),'finite native predictive mean')
    return {'mean':mean,'mean_exact_empirical':mean_exact,'median':values['1/2'],
        'equal_tail_intervals':intervals,'quantile_rule':quantile_rule,'stochastic_samples':0}


def final_forecasts(p,folder):
    io.require(p['_phase']=='final_forecasts','registered final forecast phase required')
    initialpin,initial=frozen_initial(p)
    querypin=p['inputs']['query_complete'];updatedpin=p['inputs']['updated_complete']
    query=io.read(querypin['file']);updated=io.read(updatedpin['file'])
    io.require(query['plan_sha256']==updated['plan_sha256']==p['_plan_sha256']
        and query['initial_freeze_sha256']==initialpin['sha256'],'stage provenance mismatch')
    io.verify_payloads(Path(querypin['file']).parent,query['files_sha256'])
    io.verify_payloads(Path(updatedpin['file']).parent,updated['files_sha256'])
    io.require(updated['status']=='FOUR_SAME_INFORMATION936_BASELINES_FROZEN_BEFORE_REMAINING_Y'
        and updated['remaining_ids']==query['remaining_ids'],'same-information common384 baseline freeze required')
    base=io.read(initial/'INITIAL512_LAWS.json')['records']
    learned=io.read(Path(querypin['file']).parent/'QUERY_CODE_REMAINING_LAWS.json')['records']
    modern=io.read(Path(updatedpin['file']).parent/'UPDATED_BASELINE_LAWS.json')['records']
    b={r['component_id']:r for r in base};l={r['component_id']:r for r in learned};u={r['component_id']:r for r in modern}
    remaining=query['remaining_ids'];io.require(len(remaining)==len(set(remaining))==384 and set(l)==set(u)==set(remaining),'exact common384 forecasts required')
    records=[]
    for key in remaining:
        row={'component_id':key,**{n:b[key][n] for n in METHODS[:4]},
             **{n:l[key][n] for n in METHODS[4:6]},**{n:u[key][n] for n in METHODS[6:]}}
        io.require(set(row)=={'component_id',*METHODS},'all10 distributions required')
        records.append(row)
    io.write(folder/'FINAL10_LAWS384.json',{'methods':list(METHODS),'records':records,
        'primary':'geometryRiskCode native properCRPS vs same936label RF/GP/CRPSForest/NGBoost',
        'lower_information_ablations':['F808','priorglobal','priorcontext','Beta'],
        'plainQueryERM_is_same128control_not_unbiased':True})
    io.write(folder/'FINAL10_SUMMARIES384.json',{'methods':list(METHODS),'records':[{'component_id':row['component_id'],**{m:law_summary(row[m]) for m in METHODS}} for row in records],
        'frozen_before_remaining_Y':True,'point_prediction':'distribution mean','intervals':'native equal-tail80/90/95; deterministic no stochastic samples'})
    io.write(folder/'FROZEN.json',{'status':'ALL10_METHODS384_LAWS_FROZEN_BEFORE_REMAINING_Y',
        'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],'initial_freeze':initialpin,
        'query_complete':querypin,'updated_complete':updatedpin,'seed':query['seed'],
        'remaining_ids':remaining,'queried_ids':query['queried_ids'],'methods':list(METHODS),
        'files_sha256':{f.name:io.sha(f) for f in folder.iterdir() if f.is_file()},
        'model_calls_total':10,'remaining_targets_read':0,'confirm_extra_property_X_Y_disclosed_to_models':0,'confirm_identity_composition_metadata_already_accessible':True})


def score_law(law,y,native):
    if law['family']=='weighted_empirical':return native.crps(law,y)
    io.require(law['family']=='normal','registered empirical/Normal families only')
    mu,s=float(law['location']),float(law['scale']);io.require(s>0 and math.isfinite(s),'positive finite NormalSD')
    z=(y-mu)/s;v=s*(z*math.erf(z/math.sqrt(2))+math.sqrt(2/math.pi)*math.exp(-z*z/2)-1/math.sqrt(math.pi))
    io.require(v>=0 and math.isfinite(v),'finite properNormal CRPS required');return v


def remaining_score(p,folder):
    io.require(p['_phase']=='remaining_score','registered remaining384 score phase required')
    initialpin,initial=frozen_initial(p);finalpin=p['inputs']['final_freeze'];final=io.read(finalpin['file'])
    io.require(final['status']=='ALL10_METHODS384_LAWS_FROZEN_BEFORE_REMAINING_Y' and final['plan_sha256']==p['_plan_sha256'],'all10 forecast freeze required')
    io.verify_payloads(Path(finalpin['file']).parent,final['files_sha256'])
    audit=p['inputs']['independent_forecast_audit'];ar=io.read(audit['file'])
    io.require(ar.get('status','').startswith('PASS') and ar['final_freeze_sha256']==finalpin['sha256'],'independent saved-state/forecast audit before any384score required')
    native=io.load_dependency(p,'empirical_native');reader=io.TrustedReader(p,io.pool(p))
    records=io.read(Path(finalpin['file']).parent/'FINAL10_LAWS384.json')['records'];truths=[];scores=[];diagnostics=[]
    summaries={r['component_id']:r for r in io.read(Path(finalpin['file']).parent/'FINAL10_SUMMARIES384.json')['records']}
    for row in records:
        key=row['component_id'];permit={'component_id':key,'plan_sha256':p['_plan_sha256'],
            'forecast_freeze':initialpin,'seed':final['seed'],'final_freeze':finalpin,'independent_forecast_audit':audit}
        y=reader.reveal(key,permit);truths.append({'component_id':key,'native_target':y})
        scores.append({'component_id':key,'native_target':y,**{m:score_law(row[m],y,native) for m in METHODS}})
        details={'component_id':key,'native_target':y}
        for m in METHODS:
            summary=summaries[key][m];error=summary['mean']-y
            details[m]={**summary,'prediction_error_mean_minus_Y':error,'absolute_error':abs(error),'squared_error':error*error,
                'interval_observation':{level:{'covered_inclusive':v['lower']<=y<=v['upper'],'width':v['width']} for level,v in summary['equal_tail_intervals'].items()}}
        diagnostics.append(details)
    io.require(len(truths)==384,'complete remaining384 census required')
    io.write(folder/'REMAINING384_TARGETS.json',{'records':truths,'new_measurements':0,'source':'original computedDFT reports'})
    io.write(folder/'SCORES384.json',{'records':scores,'unit':io.UNIT,'loss':'native properCRPS'})
    io.write(folder/'POINT_INTERVAL_DIAGNOSTICS384.json',{'records':diagnostics,'point_prediction':'native predictive mean','coverage_rule':'lower<=Y<=upper; exact empirical generalized inverse quantiles'})
    means={m:math.fsum(r[m] for r in scores)/384 for m in METHODS}
    secondary={m:{'RMSE':math.sqrt(math.fsum(r[m]['squared_error'] for r in diagnostics)/384),
        'MAE':math.fsum(r[m]['absolute_error'] for r in diagnostics)/384,
        'coverage':{level:sum(r[m]['interval_observation'][level]['covered_inclusive'] for r in diagnostics)/384 for level in ('80','90','95')},
        'mean_interval_width':{level:math.fsum(r[m]['interval_observation'][level]['width'] for r in diagnostics)/384 for level in ('80','90','95')}} for m in METHODS}
    io.write(folder/'AGGREGATE.json',{'status':'RAW_MODEL_EFFICACY_DEVELOPMENT_TRIAL_NOT_INDEPENDENT_FINAL_CONFIRM',
        'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],'n_common_remaining':384,'mean_CRPS_native':means,'secondary_point_interval_metrics_native':secondary,
        'relative_change_code_vs_controls':{m:(means['geometryRiskCode']/means[m]-1) for m in METHODS[5:] if means[m]>0},
        'all_current512_consumed_after_this_census':True,'confirm1024_extra_property_X_Y_not_disclosed_to_models':True,'confirm_identity_composition_metadata_already_accessible':True,
        'posthoc_tuning_or_recut_on384_allowed':False,
        'scope':'single computed bulk target; not wet hydrogel experiment, universal materials generation or full Eigen hypercycle'})
    io.write(folder/'COMPLETE.json',{'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],
        'final_freeze_sha256':finalpin['sha256'],'files_sha256':{f.name:io.sha(f) for f in folder.iterdir() if f.is_file()},
        'status':'ONE_COMMON384_CENSUS_AND_SCORE_COMPLETE','target_cells_revealed_this_phase':384})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--phase',choices=['adaptive_queries','final_forecasts','remaining_score'],required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    p=io.validate(a.plan,a.approval,a.phase);io.require(not a.output_dir.exists(),'one-shot phase directory exists; no automatic rerun')
    a.output_dir.mkdir(parents=True,exist_ok=False);io.write(a.output_dir/'START.json',{'actual_utc':io.utc(),'plan_sha256':p['_plan_sha256'],'phase':a.phase})
    {'adaptive_queries':adaptive_queries,'final_forecasts':final_forecasts,'remaining_score':remaining_score}[a.phase](p,a.output_dir)
if __name__=='__main__':main()
