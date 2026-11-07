"""First full scores after all three fixed repair/verification rounds close.

Certificate-pool score differences have already been used operationally and
are not forward confirmation. Only the192 confirmation positions evaluate
the earlier frozen adoption choices. Three rounds share growing histories.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import importlib.util
import math
from pathlib import Path
from statistics import NormalDist

HERE=Path(__file__).resolve().parent
METHODS=('directRF','directAuto','ionicRF','ionicBeta','ionicGlobal','ionicAnchor',
         'ionicAuto','GP10total','CRPSRFionic','electronic_delta')
LEVELS=(('0.8',Fraction(1,10),Fraction(9,10)),
        ('0.9',Fraction(1,20),Fraction(19,20)),
        ('0.95',Fraction(1,40),Fraction(39,40)))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def score(law,y,frame,unit):
    if law.get('coordinate') != 'native' or law.get('unit') != unit:
        raise ValueError('current source native permittivity unit required')
    y=frame.rational(y)
    if law['family'] in ('affine_empirical','weighted_empirical'):
        pairs=frame.atoms(law)
        exact=frame.crps_exact(law,y)
        if exact<0:raise ArithmeticError('negative exact proper CRPS')
        def quantile(p):
            cumulative=Fraction()
            for x,w in pairs:
                cumulative+=w
                if cumulative>=p:return x
            raise AssertionError('unit mass must attain quantile')
        median=quantile(Fraction(1,2)); value=float(exact)
        left=sum((w for x,w in pairs if x<y),Fraction())
        right=sum((w for x,w in pairs if x<=y),Fraction())
        pit_kind='discrete_CDF_interval_not_uniform_midpoint_observation'
    elif law['family']=='normal':
        mu,sd=float(law['location']),float(law['scale'])
        if not math.isfinite(mu) or not math.isfinite(sd) or sd<=0:raise ValueError('finite positive-scale normal required')
        z=(float(y)-mu)/sd; cdf=.5*(1+math.erf(z/math.sqrt(2)))
        value=sd*(z*(2*cdf-1)+2*math.exp(-.5*z*z)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
        quantile=lambda p:frame.rational(mu+sd*NormalDist().inv_cdf(float(p)))
        median=frame.rational(mu);left=right=cdf;exact=None
        pit_kind='continuous_normal_CDF'
    else:raise ValueError('unregistered predictive family')
    if not math.isfinite(value) or value<0:raise ArithmeticError('invalid proper CRPS')
    intervals={}
    for level,p0,p1 in LEVELS:
        lo,hi=quantile(p0),quantile(p1)
        intervals[level]={'lower_native':float(lo),'upper_native':float(hi),
            'lower_exact_fraction':str(lo),'upper_exact_fraction':str(hi),
            'width_native':float(hi-lo),'covered':lo<=y<=hi}
    return {'CRPS_native':value,'CRPS_exact_fraction':None if exact is None else str(exact),
            'median_native':float(median),'median_exact_fraction':str(median),
            'squared_error_native':float((median-y)**2),'intervals':intervals,
            'PIT_left':float(left),'PIT_right':float(right),'PIT_kind':pit_kind,
            'NLL':None,'density_reason':'atomic and continuous laws compared by proper CRPS'}


def aggregate(rows):
    n=len(rows)
    return {'groups':n,'CRPS_native_mean':math.fsum(r['CRPS_native'] for r in rows)/n,
            'RMSE_median_native':math.sqrt(math.fsum(r['squared_error_native'] for r in rows)/n),
            'coverage':{level:sum(r['intervals'][level]['covered'] for r in rows)/n for level,_,_ in LEVELS},
            'width_native_mean':{level:math.fsum(r['intervals'][level]['width_native'] for r in rows)/n for level,_,_ in LEVELS}}


def paired(a,b):
    diff=[x['CRPS_native']-y['CRPS_native'] for x,y in zip(a,b)]
    n=len(diff);mean=math.fsum(diff)/n
    return {'groups':n,'mean_difference':mean,
            'SE_descriptive_only':math.sqrt(math.fsum((v-mean)**2 for v in diff)/(n-1)/n),
            'better':sum(v<0 for v in diff),'worse':sum(v>0 for v in diff),'ties':sum(v==0 for v in diff),
            'differences':diff,'shared_adaptive_history_not_independent_campaign_CI':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True)
    ap.add_argument('--roles',required=True)
    ap.add_argument('--execution-root',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();io=load('physical_task_io_scoring',HERE/'task_io.py')
    frame=load('physical_affine_scoring',HERE.parent/'frame_principle/affine_reference.py')
    plan=io.validate(a.plan,a.approval,'score_closed_rounds')
    roles=io.load_roles(plan,a.roles)
    prepared=[]
    # Validate every closed round before creating a score output or scoring any outcome.
    for number in (1,2,3):
        directory=a.execution_root/('round'+str(number));fdir=directory/'forecast';vdir=directory/'verification'
        ff=fdir/'ALL1280_FORECASTS_FROZEN.json';freeze=io.read(ff);close=io.read(vdir/'CLOSED_ROUND.json')
        if (freeze['status']!='ALL1280_FORECASTS_FROZEN_NO_FUTURE_TARGET_DISCLOSURE'
            or freeze['round']!=number or freeze['plan_sha256']!=io.sha(a.plan)
            or freeze['source_sha256']!=io.sources() or freeze['roles_sha256']!=io.sha(a.roles)):
            raise ValueError('exact current-source, plan, role and round freeze required')
        if (close['status']!='ROUND_CLOSED_WITH_ALL128_TRUE_TARGETS' or close['round']!=number
            or close['plan_sha256']!=io.sha(a.plan) or close['forecast_freeze_sha256']!=io.sha(ff)):
            raise ValueError('allthree actual matching closures required before fullscore')
        for base,pins in ((fdir,freeze['files_sha256']),(vdir,close['files_sha256'])):
            for name,pin in pins.items():
                p=(base/name).resolve();p.relative_to(base.resolve())
                if io.sha(p)!=pin:raise ValueError('changed closed-round bytes')
        truth=io.read(vdir/'ALL128_TARGETS_DISCLOSED.json')['records']
        expected={r['component_id']:kind for kind,rows in roles['round'+str(number)].items() for r in rows}
        if {r['component_id']:r['role'] for r in truth}!=expected:
            raise ValueError('closed outcomes must match the exact allocated roles')
        decision=io.read(vdir/'ADOPTION_FREEZE.json')
        if decision['choice'] not in ('ionicAuto','ionicRF'):raise ValueError('precommitted adoption choices only')
        prepared.append((number,io.read(fdir/'FORECASTS1280.json')['records'],truth,decision,close))
    a.output_dir.mkdir(parents=True,exist_ok=False)
    io.write(a.output_dir/'START.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
             'plan_sha256':io.sha(a.plan),'allthree_rounds_closed':True,
             'certificate_gap_access_precedes_this_fullscore':True})
    records=[];adopted=[];rounds=[];seen=set()
    for number,forecasts,truth,decision,close in prepared:
        lookup={r['component_id']:r for r in truth}
        if len(truth)!=128 or len(lookup)!=128 or len(forecasts)!=1280:raise ValueError('complete fixed-size positions required')
        scored={}
        for row in forecasts:
            key=(row['method'],row['component_id'])
            if row['method'] not in METHODS or key in scored or row['component_id'] not in lookup:raise ValueError('unexpected forecast position')
            target=lookup[row['component_id']]
            if target['role']!=row['role']:raise ValueError('role mismatch')
            result=score(row['law'],target['native_target'],frame,io.UNIT);scored[key]=result
            records.append({'round':number,'role':row['role'],'method':row['method'],
                            'component_id':row['component_id'],'native_target':target['native_target'],'score':result})
        for target in truth:
            if target['component_id'] in seen:raise ValueError('no repeated whole groups acrossrounds')
            seen.add(target['component_id'])
            if target['role']=='confirmation':
                adopted.append({'round':number,'component_id':target['component_id'],'chosen_method':decision['choice'],
                                'score':scored[(decision['choice'],target['component_id'])],
                                'reference_score':scored[('ionicRF',target['component_id'])]})
        rounds.append({'round':number,'choice':decision['choice'],'primary_queries':close['primary_target_queries'],
                       'research_census_queries':close['research_census_queries'],'confirmation_queries':64,
                       'total_targets':128})
    summaries={m:{role:{str(n):aggregate([r['score'] for r in records if r['method']==m
        and (role=='all' or r['role']==role) and (n==0 or r['round']==n)])
        for n in (0,1,2,3)} for role in ('all','certificate','confirmation')} for m in METHODS}
    confirm={m:[r['score'] for r in records if r['method']==m and r['role']=='confirmation'] for m in METHODS}
    controls={m:paired(confirm['ionicAuto'],confirm[m]) for m in METHODS if m!='ionicAuto'}
    chosen=paired([r['score'] for r in adopted],[r['reference_score'] for r in adopted])
    io.write(a.output_dir/'RESULTS.json',{'status':'COMPLETE_FIXED_THREE_ROUND_PHYSICAL_REFERENCE_RESEARCH',
             'actual_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':io.sha(a.plan),
             'scores':records,'summaries':summaries,'confirmation_ionicAuto_minus_controls':controls,
             'adopted_confirmation_summary':aggregate([r['score'] for r in adopted]),
             'adopted_confirmation_minus_ionicRF':chosen,'adopted_records':adopted,'rounds':rounds,
             'forecasts_scored':3840,'unique_new_groups':384,'forward_confirmation_groups':192,
             'certificate_groups_used_for_selection':192,'oldgroups_refit_reuses':448,
             'new_physical_measurements':0,'actual_total_new_report_targets':384,
             'census_and_confirmation_cost_included':True,'actual_physical_experiment_savings_measured':False,
             'not_three_independent_campaigns':True,'directAuto_ionicAuto_not_pure_frame_ablation':True,
             'no_full_Eigen_hypercycle_or_universal_joint_generator_claim':True})
    print({'scores':3840,'new_groups':384,'forward_confirmation':192,
           'candidate_minus_ionicRF':controls['ionicRF']['mean_difference'],
           'adopted_minus_ionicRF':chosen['mean_difference']})


if __name__=='__main__':main()
