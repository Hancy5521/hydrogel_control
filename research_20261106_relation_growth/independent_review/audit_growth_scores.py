"""All672 saved prequential scores via independent native-CDF/energy arithmetic.

No SUT score import, material read, new fit, acquisition, or feature selection.
Mixture quantiles are native-CDF roots; empirical CRPS uses pairwise energy.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings
from scipy import special
from scipy.integrate import quad, IntegrationWarning

METHODS=('old3_BMA','grown4_BMA','H_new','H_anchor','affine','GP10','RF10')
LEVELS=('0.8','0.9','0.95')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def require(value,message):
    if not value:raise AssertionError(message)


def mixture_check(saved,own,c):
    c.number('mixture_CRPS',saved['CRPS_native'],own['CRPS_native'],2e-4,2e-5)
    c.number('mixture_native_median',saved['predictive_median_native'],own['predictive_median_native'],2e-7,2e-9)
    c.number('mixture_squared_error',saved['squared_error_native'],own['squared_error_native'])
    for key in ('PIT_left','PIT_right','PIT_midpoint'):c.number('mixture_'+key,saved[key],own[key],2e-10,0.)
    c.number('mixture_logdensity',saved['log_density_native'],own['log_density_native'],2e-9,2e-11)
    c.number('mixture_NLL',saved['negative_log_density_native'],-own['log_density_native'],2e-9,2e-11)
    for level in LEVELS:
        for key in ('lower','upper','width'):c.number('mixture_interval_'+key,saved['intervals'][level][key],own['intervals'][level][key],2e-7,2e-9)
        c.exact('mixture_coverage_event',saved['intervals'][level]['covered'],own['intervals'][level]['covered'])


def aggregate(rows):
    n=len(rows);require(n in (32,96),'exact batch32 or total96')
    return {'groups':n,'CRPS_native_mean':math.fsum(r['CRPS_native'] for r in rows)/n,
            'RMSE_median_native':math.sqrt(math.fsum(r['squared_error_native'] for r in rows)/n),
            'coverage':{l:sum(r['intervals'][l]['covered'] for r in rows)/n for l in LEVELS},
            'width_native_mean':{l:math.fsum(r['intervals'][l]['width'] for r in rows)/n for l in LEVELS}}


def audit(plan):
    pins=plan['pins']
    for pin in pins.values():require(sha(pin['file'])==pin['sha256'],'input/helper changed:'+pin['file'])
    batch=load('saved_state_reference',pins['batch_transform']['file'])
    scalar=load('_growth_independent_scalar_scores',pins['scalar_score_functions']['file'])
    native=load('_growth_independent_native_CDF',pins['native_CDF']['file']);native.special=special;native.quad=quad
    load('full_native_reference',pins['full_fixture_reference']['file'])
    mixture=load('_growth_independent_mixture_reference',pins['mixture_reference']['file'])
    own=SimpleNamespace(transform=batch.forward_yj,inverse=scalar.inverse_yj)
    c=scalar.Checks();folder=Path(plan['forecast_dir'])
    result=read(pins['score_result']['file']);freeze=read(pins['all_batch_freeze']['file'])
    science=read(pins['science_plan']['file']);state=read(pins['independent_state_audit']['file'])
    require(state['status']=='PASS','complete saved-state audit must already pass')
    c.exact('result_status',result['status'],'COMPLETE_SINGLE_SLOT_GROWTH_FRESH128_DEVELOPMENT')
    c.exact('result_science_binding',result['plan_sha256'],pins['science_plan']['sha256'])
    c.exact('result_forecast_binding',result['all_batch_freeze_sha256'],pins['all_batch_freeze']['sha256'])
    c.exact('result_seven_source_binding',result['source_sha256'],science['source_sha256'])
    for file,pin in freeze['files_sha256'].items():require(sha(folder/file)==pin,'frozen payload changed')
    c.exact('only672scores',len(result['scores']),672)
    saved={(r['method'],r['component_id']):r for r in result['scores']}
    require(len(saved)==672,'no repeated method/group score')
    cache={};qcache={};routes={'native_mixture_CDF':0,'native_scalar_CDF':0,'empirical_energy':0}
    values={};batch_by_group={};quantile_errors=[]

    def evaluate(y,law):
        encoded=json.dumps(law,sort_keys=True,separators=(',',':'),allow_nan=False);key=(y,encoded)
        if key not in cache:
            if law['family']=='mixture_student_t':
                if encoded not in qcache:qcache[encoded]={p:mixture.quantile(law,p) for p in mixture.PROBABILITIES}
                q=qcache[encoded];median=q[.5];crps,error=mixture.direct_crps(law,y);pit=mixture.cdf(law,y)
                intervals={}
                for level in LEVELS:
                    p=float(level);lo=q[min(q,key=lambda x:abs(x-(1-p)/2))];hi=q[min(q,key=lambda x:abs(x-(1+p)/2))]
                    intervals[level]={'lower':lo,'upper':hi,'width':hi-lo,'covered':lo<=y<=hi}
                cache[key]={'CRPS_native':crps,'predictive_median_native':median,'squared_error_native':(y-median)**2,
                            'PIT_left':pit,'PIT_right':pit,'PIT_midpoint':pit,'intervals':intervals,
                            'log_density_native':mixture.log_density(law,y),'quad_error_estimate_native':error}
                routes['native_mixture_CDF']+=1
            else:
                cache[key]=scalar.evaluate(y,law,native,own)
                routes['empirical_energy' if law['family']=='weighted_empirical' else 'native_scalar_CDF']+=1
        return cache[key]

    expected=set()
    for b,role in enumerate(plan['batch_roles'],1):
        forecast=read(folder/role/'FORECASTS.json')['records']
        truth=read(folder/role/'TRUE_BATCH_REVEAL.json')['rows']
        targets={r['identity']['component_id']:r for r in truth}
        require(len(targets)==len(truth)==32,'all32 distinct shared truth rows')
        for group in targets:
            require(group not in batch_by_group,'batch groups overlap');batch_by_group[group]=b
        for row in forecast:
            method=row['method'];group=row['identity']['component_id'];key=(method,group);expected.add(key)
            target=targets[group];c.exact('forecast_truth_identity',row['identity'],target['identity'])
            y=target['native_target'];scored=saved[key]
            c.exact('score_method',scored['method'],method);c.exact('score_batch',scored['batch'],b)
            c.number('score_target',scored['native_target'],y,0,0)
            value=evaluate(y,row['law']);values[key]=value
            (mixture_check if row['law']['family']=='mixture_student_t' else scalar.score_compare)(scored['score'],value,c)
    require(set(saved)==expected and len(values)==672,'all672 coordinate crossproduct')
    c.exact('96_unique_validation_groups',len(batch_by_group),96)
    tables={}
    for method in METHODS:
        tables[method]={}
        for b in (0,1,2,3):
            ownsummary=aggregate([value for (m,g),value in values.items() if m==method and (b==0 or batch_by_group[g]==b)])
            table=result['summaries'][method][str(b)];tables[method][str(b)]=ownsummary
            c.exact('summary_groups',table['groups'],ownsummary['groups'])
            c.number('summary_CRPS',table['CRPS_native_mean'],ownsummary['CRPS_native_mean'],2e-4,2e-5)
            c.number('summary_RMSE',table['RMSE_median_native'],ownsummary['RMSE_median_native'])
            for level in LEVELS:
                c.number('summary_exact_coverage',table['coverage'][level],ownsummary['coverage'][level],0,0)
                c.number('summary_interval_width',table['width_native_mean'][level],ownsummary['width_native_mean'][level],2e-7,2e-9)
    paired={}
    group_order=[r['component_id'] for r in result['scores'] if r['method']=='grown4_BMA']
    require(len(group_order)==len(set(group_order))==96,'96 primary paired groups')
    for method in METHODS:
        if method=='grown4_BMA':continue
        diffs=[];saved_diffs=[]
        table=result['paired_grown4_minus_controls'][method]
        for i,group in enumerate(group_order):
            a=values[('grown4_BMA',group)]['CRPS_native'];b=values[(method,group)]['CRPS_native'];difference=a-b
            c.number('paired_difference',table['all_differences'][i],difference,4e-4+2e-5*(abs(a)+abs(b)),0.)
            diffs.append(difference);saved_diffs.append(table['all_differences'][i])
        mean=math.fsum(diffs)/96;se=math.sqrt(math.fsum((d-mean)**2 for d in diffs)/95/96)
        c.exact('paired_groups',table['groups'],96)
        c.number('paired_mean',table['candidate_minus_other_CRPS_mean'],mean,4e-4,4e-5)
        c.number('paired_descriptive_SE',table['standard_error_descriptive_only'],se,4e-4,4e-5)
        c.exact('saved_sign_better',table['better'],sum(d<0 for d in saved_diffs))
        c.exact('saved_sign_worse',table['worse'],sum(d>0 for d in saved_diffs))
        c.exact('saved_sign_ties',table['exact_ties'],sum(d==0 for d in saved_diffs))
        paired[method]={'mean':mean,'descriptive_SE':se}
    initial=read(folder/'INITIAL_GROWTH_STATE.json');final=read(folder/'FINAL_STATE.json')
    for k,s in [('selected_descriptor_index','selected_index'),('anchor_descriptor_index','anchor_index'),('new_anchor_alias','new_anchor_exact_alias')]:
        c.exact('result_'+k,result[k],initial[s])
    c.exact('initial_library_weights',result['initial_library_weights'],initial['libraries'])
    c.exact('final_library_weights',result['final_library_weights'],final['libraries'])
    for key,value in [('new_material_groups',128),('prequential_validation_groups',96),('forecast_score_coordinates',672),('actual_sklearn_fits',6),('new_native_cell_cost',384)]:
        c.exact('result_'+key,result[key],value)
    return {'status':'PASS','all_score_coordinates':672,'coverage_events':2016,'method_batch_summaries':28,
            'paired_score_differences':576,'unique_law_outcome_scores':len(cache),'independent_routes':routes,
            'independent_summaries':tables,'independent_paired':paired,
            'primary_grown4_minus_old3_CRPS':paired['old3_BMA']['mean'],
            'comparison_counts':c.counts,'maximum_absolute_errors':c.errors,
            'maximum_quad_error_estimate_native':max(v.get('quad_error_estimate_native',0.) for v in cache.values()),
            'saved_near_zero_sign_counts_not_claimed_resolved_independently':True,
            'SUT_score_imports':0,'new_fits':0,'raw_reads':0,
            'scope':'Arithmetic verification of prequential development reports; not external validation or selection-corrected mechanism discovery.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--approval',required=True);parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();plan=read(args.plan)
    require(args.approval==plan['root_approval_token'] and sha(__file__)==plan['audit_source_sha256'],'root/source binding')
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    start=datetime.now(timezone.utc).isoformat()
    write(output/'EXECUTION_START.json',{'actual_utc':start,'plan_sha256':sha(args.plan),'new_fits':0,'raw_reads':0})
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',IntegrationWarning)
            result=audit(plan)
    except Exception as exc:
        result={'status':'FAIL','error_type':type(exc).__name__,'error':str(exc),'new_fits':0,'raw_reads':0}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),
                   'audit_source_sha256':sha(__file__),'plan_sha256':sha(args.plan)})
    file=output/'INDEPENDENT_GROWTH_ALL_SCORES_AUDIT.json';write(file,result)
    print(json.dumps({'status':result['status'],'file':str(file),'sha256':sha(file)},ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
