"""Query/decision/census and separately approved first full remaining-risk score.

No learning or tuning. Operational score gaps are distinct from full predictive
scores. The latter run only after an actually matching state/query audit PASS.
"""
import argparse
from fractions import Fraction
import math
import os
from pathlib import Path
import pilot_io as io


def append(path,value):
    import json
    with Path(path).open('a',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False);f.write('\n')


def frozen_inputs(plan,folder):
    pin=folder/'FORECAST_FREEZE.json';freeze=io.read(pin)
    if (freeze['status']!='ALL512_F_G_LAWS_FROZEN_BEFORE_ANY_TARGET' or freeze['plan_sha256']!=plan['_plan_sha256']
        or freeze['source_sha256']!=io.sources() or freeze['identity_roles_sha256']!=plan['inputs']['identity_roles']['sha256']):
        raise ValueError('complete matching512 forecast freeze required')
    io.verify_payloads(folder,freeze['files_sha256'])
    forecasts=io.read(folder/'FORECASTS512.json')['records'];geometry=io.read(folder/'GEOMETRY512.json')['records']
    identities=io.pool(plan);keys=[r['component_id'] for r in identities]
    if (len(forecasts)!=512 or len(geometry)!=512 or [r['component_id'] for r in forecasts]!=keys
        or [r['component_id'] for r in geometry]!=keys):raise ValueError('exact registered512 forecast/geometry positions required')
    if any(r['identity']!=ident for r,ident in zip(forecasts,identities)):raise ValueError('complete identities changed')
    return forecasts,geometry,identities,pin


def query(plan,approvalfile,forecast_dir,output):
    forecasts,geometry,identities,fp=frozen_inputs(plan,forecast_dir)
    approval=io.read(approvalfile);reviewpin=approval.get('forecast_review')
    if not isinstance(reviewpin,dict) or io.sha(reviewpin['file'])!=reviewpin['sha256']:
        raise ValueError('actual current independent forecast audit pin required before query')
    forecast_review=io.read(reviewpin['file'])
    if (forecast_review['status']!='PASS_INDEPENDENT_REMAINING512_FORECASTS'
        or forecast_review['scientific_plan_sha256']!=plan['_plan_sha256']
        or forecast_review['forecast_freeze_sha256']!=io.sha(fp)
        or forecast_review['exact_denominator_lcm_bit_length']>200000
        or forecast_review['new_target_queries']!=0):
        raise ValueError('matching current forecast arithmetic/LCM audit before any query target')
    frame=io.load_dependency(plan,'affine_reference')
    io.load_dependency(plan,'directed_confidence');core=io.load_dependency(plan,'remaining_risk_core')
    output.mkdir(parents=True,exist_ok=False);(output/'tickets').mkdir()
    io.write(output/'START.json',{'actual_utc':io.utc(),'plan_sha256':plan['_plan_sha256'],'forecast_freeze_sha256':io.sha(fp),
             'forecast_review_sha256':reviewpin['sha256'],'full_predictive_score_calls':0,'no_new_fit_or_optimizer':True})
    seedbytes=os.urandom(32);seed=int.from_bytes(seedbytes,'big')
    seedfile=output/'OS_SEED_AFTER_FREEZE.json'
    io.write(seedfile,{'status':'ACTUAL_OS_SEED_RECORDED_AFTER_FORECAST_FREEZE','actual_utc':io.utc(),
             'plan_sha256':plan['_plan_sha256'],'forecast_freeze_sha256':io.sha(fp),
             'seed_hex':seedbytes.hex(),'seed_integer':seed,'entropy_source':'os.urandom32bytes',
             'sampling_model':'coreRandom(seed) exactintegerPPS; confidence theorem assumes ideal conditional-uniform tickets'})
    units=[core.RiskUnit(r['component_id'],Fraction(r['envelope']['lower_exact_fraction']),
                       Fraction(r['envelope']['upper_exact_fraction']),Fraction(r['curvature_exact_fraction'])) for r in geometry]
    try:engine=core.ExactRemainingRisk(units,seed)
    except (ValueError,ArithmeticError,OverflowError) as error:
        io.write(output/'CORE_INITIALIZATION_STOP.json',{'status':'STOP_BEFORE_ANY_TARGET_NO_CENSUS',
                 'actual_utc':io.utc(),'plan_sha256':plan['_plan_sha256'],'forecast_freeze_sha256':io.sha(fp),
                 'seed_sha256':io.sha(seedfile),'error_type':type(error).__name__,'error':str(error),
                 'target_queries':0,'no_seed_reset_or_samepool_retry_authorized':True})
        raise
    reader=io.TrustedReader(plan,identities);bygeometry={r['component_id']:r for r in geometry}
    byforecast={r['component_id']:r for r in forecasts};targets={};queried=[]
    decision=engine.decision()
    append(output/'DECISION_PREFIXES.jsonl',{'actual_utc':io.utc(),'query_count':0,'decision':decision})
    try:
        while (decision['status']=='valid' and not decision['strict_improvement']
               and len(queried)<io.MAX_QUERIES and decision['remaining_count']>io.MIN_REMAINING):
            ticket=engine.draw()
            if ticket is None:break
            ticketfile=output/'tickets'/('ticket_'+format(ticket.query_index,'04d')+'.json')
            io.write(ticketfile,{'status':'TICKET_PERSISTED_BEFORE_TARGET','actual_utc':io.utc(),
                     'plan_sha256':plan['_plan_sha256'],'forecast_freeze_sha256':io.sha(fp),
                     'seed_sha256':io.sha(seedfile),'ticket':ticket.to_record()})
            append(output/'TICKETS.jsonl',{'query_index':ticket.query_index,'component_id':ticket.identity,
                   'ticket_file':ticketfile.relative_to(output).as_posix(),'ticket_sha256':io.sha(ticketfile)})
            permit={'phase':'query','plan_sha256':plan['_plan_sha256'],'component_id':ticket.identity,
                    'forecast_freeze':{'file':str(fp),'sha256':io.sha(fp)},'seed':{'file':str(seedfile),'sha256':io.sha(seedfile)},
                    'ticket':{'file':str(ticketfile),'sha256':io.sha(ticketfile)}}
            before=io.utc();value=reader.reveal(ticket.identity,permit)
            targets[ticket.identity]=value;queried.append(ticket.identity)
            gap=frame.gap_at(bygeometry[ticket.identity]['envelope'],frame.rational(value))
            event=engine.observe(ticket,gap)
            append(output/'QUERIES.jsonl',{'actual_request_utc':before,'actual_return_utc':io.utc(),
                   'component_id':ticket.identity,'native_target':value,'gap_exact_fraction':str(gap),
                   'permit':permit,'core_event':event})
            decision=engine.decision()
            append(output/'DECISION_PREFIXES.jsonl',{'actual_utc':io.utc(),'query_count':len(queried),'decision':decision})
        if decision['remaining_count']<128:raise AssertionError('minimum remaining set violated')
        reason=('first_strict_remaining_certificate' if decision['strict_improvement'] else
                'confidence_abstain' if decision['status']!='valid' else
                'maximum384queries_no_strict_certificate' if len(queried)==384 else 'no_nonzero_width_ticket')
        alpha=Fraction(decision['alpha']) if decision['status']=='valid' and decision['strict_improvement'] else Fraction()
        remaining=list(decision['remaining_ids'])
        if set(remaining)!=set(byforecast)-set(queried) or len(remaining)+len(queried)!=512:
            raise AssertionError('decision remaining set must be exactly unqueried')
        snapshot=engine.snapshot()
        if snapshot.get('pending') is not None:
            raise AssertionError('an unobserved pending draw cannot cross the alpha freeze')
        if set(reader.revealed)!=set(queried) or set(targets)!=set(queried):
            raise AssertionError('every released target must be present in the operational query ledger')
        if set(engine.observed)!=set(queried) or len(snapshot['events'])!=len(queried):
            raise AssertionError('every released target must be observed exactly once before alpha freeze')
        io.write(output/'CORE_FINAL_SNAPSHOT.json',snapshot)
        decisionfile=output/'ADOPTION_FREEZE.json'
        io.write(decisionfile,{'status':'ALPHA_AND_REMAINING_SET_FROZEN_BEFORE_CENSUS','actual_utc':io.utc(),
                 'plan_sha256':plan['_plan_sha256'],'source_sha256':io.sources(),'forecast_freeze_sha256':io.sha(fp),
                 'seed_sha256':io.sha(seedfile),'core_snapshot_sha256':io.sha(output/'CORE_FINAL_SNAPSHOT.json'),
                 'forecast_review_sha256':reviewpin['sha256'],
                 'queried_ids':queried,'remaining_ids':remaining,'query_count':len(queried),'remaining_count':len(remaining),
                 'alpha':str(alpha),'strict_certificate':bool(alpha>0 and decision['strict_improvement']),
                 'stop_reason':reason,'decision':decision,'confidence_delta':'1/20',
                 'primary_population':'this frozen remaining set, native uniform mean, not whole512 forecast mean',
                 'no_new_targets_after_this_freeze_used_to_choose_alpha':True})
        census=[]
        for key in remaining:
            permit={'phase':'remaining_census','plan_sha256':plan['_plan_sha256'],'component_id':key,
                    'forecast_freeze':{'file':str(fp),'sha256':io.sha(fp)},'seed':{'file':str(seedfile),'sha256':io.sha(seedfile)},
                    'decision':{'file':str(decisionfile),'sha256':io.sha(decisionfile)}}
            before=io.utc();value=reader.reveal(key,permit);targets[key]=value
            record={'component_id':key,'identity':byforecast[key]['identity'],'native_target':value,
                    'actual_request_utc':before,'actual_return_utc':io.utc(),'permit':permit}
            census.append(record);append(output/'CENSUS_REQUESTS.jsonl',record)
        io.write(output/'CENSUS_REMAINING.json',{'status':'ACTUAL_REMAINING_TARGETS_AFTER_ALPHA_FREEZE',
                 'actual_utc':io.utc(),'decision_freeze_sha256':io.sha(decisionfile),'records':census})
        io.write(output/'ALL512_TARGETS.json',{'actual_utc':io.utc(),'plan_sha256':plan['_plan_sha256'],
                 'forecast_freeze_sha256':io.sha(fp),'decision_freeze_sha256':io.sha(decisionfile),
                 'records':[{'component_id':r['component_id'],'identity':r['identity'],'native_target':targets[r['component_id']],
                             'role_at_alpha_freeze':'queried' if r['component_id'] in set(queried) else 'remaining'} for r in forecasts]})
        pins={p.relative_to(output).as_posix():io.sha(p) for p in output.rglob('*') if p.is_file()}
        io.write(output/'PILOT_CLOSED.json',{'status':'QUERY_DECISION_AND512_TARGETS_CLOSED_AWAITING_INDEPENDENT_STATE_AUDIT',
                 'actual_utc':io.utc(),'plan_sha256':plan['_plan_sha256'],'source_sha256':io.sources(),
                 'forecast_freeze_sha256':io.sha(fp),'decision_freeze_sha256':io.sha(decisionfile),'files_sha256':pins,
                 'primary_query_targets':len(queried),'research_census_targets':len(remaining),'total_actual_targets':512,
                 'alpha':str(alpha),'remaining_count':len(remaining),'strict_certificate':bool(alpha>0 and decision['strict_improvement']),
                 'full_predictive_score_calls':0,'operational_gap_calculations':len(queried),'new_fits':0,
                 'new_wet_measurements_or_DFT':0,'physical_cost_savings_established':False})
        print({'status':'PILOT_CLOSED','queries':len(queried),'remaining':len(remaining),'alpha':str(alpha),'fullscore':0})
    except Exception as error:
        io.write(output/'FAILED_STOP.json',{'actual_utc':io.utc(),'type':type(error).__name__,'message':str(error),
                 'already_revealed_target_ids':list(targets),'no_seed_reset_reselection_or_samepool_rescue':True});raise


def mixture(frame,F,G,alpha):
    masses={}
    for law,coefficient in ((F,1-alpha),(G,alpha)):
        for value,mass in frame.atoms(law):masses[value]=masses.get(value,Fraction())+coefficient*mass
    support=sorted(v for v,w in masses.items() if w>0)
    if any(Fraction.from_float(float(v))!=v for v in support):raise ValueError('pilot expects original binary native supports')
    return {'family':'weighted_empirical','coordinate':'native','unit':io.UNIT,'support_native':list(map(float,support)),
            'weight_fractions':[str(masses[v]) for v in support],'alpha_exact_fraction':str(alpha)}


def metric(frame,law,y):
    exact=frame.crps_exact(law,y);pairs=frame.atoms(law);target=frame.rational(y)
    def q(p):
        cumulative=Fraction()
        for x,w in pairs:
            cumulative+=w
            if cumulative>=p:return x
        raise AssertionError('quantile not attained')
    median=q(Fraction(1,2));intervals={}
    for level,lo,hi in [('0.8',Fraction(1,10),Fraction(9,10)),('0.9',Fraction(1,20),Fraction(19,20)),('0.95',Fraction(1,40),Fraction(39,40))]:
        a,b=q(lo),q(hi);intervals[level]={'lower_exact_fraction':str(a),'upper_exact_fraction':str(b),
                                       'covered':a<=target<=b,'width_native':float(b-a)}
    return {'CRPS_native':float(exact),'CRPS_exact_fraction':str(exact),'median_native':float(median),
            'squared_error_native':float((median-target)**2),'intervals':intervals}


def aggregate(rows):
    n=len(rows)
    if n==0:return None
    return {'groups':n,'CRPS_native_mean':math.fsum(r['CRPS_native'] for r in rows)/n,
            'RMSE_median_native':math.sqrt(math.fsum(r['squared_error_native'] for r in rows)/n),
            'coverage':{z:sum(r['intervals'][z]['covered'] for r in rows)/n for z in ('0.8','0.9','0.95')},
            'width_native_mean':{z:math.fsum(r['intervals'][z]['width_native'] for r in rows)/n for z in ('0.8','0.9','0.95')}}


def score(plan,approvalfile,forecast_dir,query_dir,output):
    forecasts,geometry,identities,fp=frozen_inputs(plan,forecast_dir)
    cp=query_dir/'PILOT_CLOSED.json';close=io.read(cp);dp=query_dir/'ADOPTION_FREEZE.json';decision=io.read(dp)
    if (close['status']!='QUERY_DECISION_AND512_TARGETS_CLOSED_AWAITING_INDEPENDENT_STATE_AUDIT'
        or close['plan_sha256']!=plan['_plan_sha256'] or close['forecast_freeze_sha256']!=io.sha(fp)
        or close['decision_freeze_sha256']!=io.sha(dp)):raise ValueError('matching actual closed query/census required')
    io.verify_payloads(query_dir,close['files_sha256'])
    approval=io.read(approvalfile);reviewpin=approval['state_review']
    if io.sha(reviewpin['file'])!=reviewpin['sha256']:raise ValueError('state/query review changed')
    review=io.read(reviewpin['file'])
    if (not review['status'].startswith('PASS') or review['scientific_plan_sha256']!=plan['_plan_sha256']
        or review['forecast_freeze_sha256']!=io.sha(fp) or review['pilot_close_sha256']!=io.sha(cp)):
        raise ValueError('actual current state/queryauditPASS beforefirstfullscore')
    truths=io.read(query_dir/'ALL512_TARGETS.json');lookup={r['component_id']:r for r in truths['records']}
    allids={r['component_id'] for r in identities};remaining=set(decision['remaining_ids']);queried=set(decision['queried_ids'])
    if (len(lookup)!=512 or set(lookup)!=allids or len(remaining)<128 or remaining|queried!=allids or remaining&queried
        or truths['decision_freeze_sha256']!=io.sha(dp) or truths['forecast_freeze_sha256']!=io.sha(fp)
        or truths['plan_sha256']!=plan['_plan_sha256']):raise ValueError('frozen role/true target closure mismatch')
    alpha=Fraction(decision['alpha'])
    frame=io.load_dependency(plan,'affine_reference');output.mkdir(parents=True,exist_ok=False)
    io.write(output/'START.json',{'actual_utc':io.utc(),'plan_sha256':plan['_plan_sha256'],
             'state_query_review_sha256':reviewpin['sha256'],'pilot_close_sha256':io.sha(cp),'first_fullscore':True})
    records=[];gstrata={'g_zero':[],'g_positive':[]};gapR=Fraction();gapG=Fraction()
    for row in forecasts:
        key=row['component_id'];target=lookup[key]['native_target'];H=mixture(frame,row['F_law'],row['G_law'],alpha)
        values={m:metric(frame,law,target) for m,law in [('F',row['F_law']),('G',row['G_law']),('H',H)]}
        role='remaining' if key in remaining else 'queried'
        record={'component_id':key,'role_at_alpha_freeze':role,'native_target':target,
                'g5':row['bounded_descriptors8'][5],'scores':values};records.append(record)
        if key in remaining:
            delta=Fraction(values['H']['CRPS_exact_fraction'])-Fraction(values['F']['CRPS_exact_fraction'])
            gapR+=delta;gapG+=Fraction(values['G']['CRPS_exact_fraction'])-Fraction(values['F']['CRPS_exact_fraction'])
            gstrata['g_zero' if record['g5']==0 else 'g_positive'].append(record)
    summaries={role:{m:aggregate([r['scores'][m] for r in records if role=='all512_forecast_diagnostic' or r['role_at_alpha_freeze']==role]) for m in ('F','G','H')}
               for role in ('all512_forecast_diagnostic','remaining','queried')}
    group_summaries={k:{'groups':len(rows),'summaries':{m:aggregate([r['scores'][m] for r in rows]) for m in ('F','G','H')},
                        'descriptive_only_no_risk_reweighting':True} for k,rows in gstrata.items()}
    primary=gapR/len(remaining);certificate=bool(decision['strict_certificate'] and alpha>0)
    bound=Fraction(decision['decision'].get('risk_upper_sum','0'))
    io.write(output/'RESULTS.json',{'status':'COMPLETE_REMAINING_RISK_FROZEN_REFERENCE_PILOT','actual_utc':io.utc(),
             'plan_sha256':plan['_plan_sha256'],'forecast_freeze_sha256':io.sha(fp),'pilot_close_sha256':io.sha(cp),
             'state_query_audit_sha256':reviewpin['sha256'],'alpha_exact_fraction':str(alpha),'queried_groups':len(queried),
             'remaining_groups':len(remaining),'score_positions':1536,'summaries':summaries,'records':records,
             'primary_remaining_H_minus_F_sum_exact':str(gapR),'primary_remaining_H_minus_F_mean_exact':str(primary),
             'primary_remaining_H_minus_F_mean_native':float(primary),'remaining_G_minus_F_mean_native':float(gapG/len(remaining)),
             'completion_normalized512_gap_exact':str(gapR/512),'completion_normalized512_gap_native':float(gapR/512),
             'queried_cells_both_completions_use_actual_truth_pointmass_CRPS_gap0':True,
             'all512_predictive_summaries_are_retrospective_not_completion_primary':True,
             'remaining_population_fixed_before_census':True,'g0_gpositive_remaining_descriptive':group_summaries,
             'strict_certificate':certificate,'certificate_risk_upper_sum_exact':str(bound),
             'certificate_actual_remaining_bound_violation':bool(certificate and gapR>bound),
             'pilot_effective_increment':bool(certificate and alpha>0 and primary<0),
             'alpha0_is_reference_preservation_not_improvement':alpha==0,
             'confidence_scope':'single-epoch recorded PPS experiment; cannot establish empirical95%coverage',
             'actual_primary_query_targets':len(queried),'actual_research_census_targets':len(remaining),'total_actual_target_cells':512,
             'new_fits_or_optimizers':0,'new_wet_measurements_or_DFT':0,'actual_physical_experiment_savings_measured':False})
    print({'status':'COMPLETE_FIRST_FULLSCORE','remaining':len(remaining),'alpha':str(alpha),'primary_remaining_mean_gap':float(primary)})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--mode',choices=('query','score'),required=True)
    ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--forecast-dir',type=Path,required=True)
    ap.add_argument('--query-dir',type=Path);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    if (a.mode=='score')!=(a.query_dir is not None):raise ValueError('query-dir required only forscore')
    plan=io.validate(a.plan,a.approval,'query_and_census' if a.mode=='query' else 'score_closed_pilot')
    if a.mode=='query':query(plan,a.approval,a.forecast_dir,a.output_dir)
    else:score(plan,a.approval,a.forecast_dir,a.query_dir,a.output_dir)


if __name__=='__main__':main()
