"""Independent all4080 frozen scores; native mixtureCDF/energy arithmetic, never SUT scoring."""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.util,itertools,json,math,sys,traceback,warnings
from pathlib import Path
from types import SimpleNamespace
from scipy import special
from scipy.integrate import quad,IntegrationWarning

POLICIES=('full_native','warm_openloop','random','marginal_native','model_MI','gp_variance')
BUDGETS=(0,4,8)
LEVELS=('0.8','0.9','0.95')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def require(x,m):
    if not x:raise AssertionError(m)


def load(pin,name):
    require(sha(pin['file'])==pin['sha256'],'independent helper changed:'+name)
    spec=importlib.util.spec_from_file_location(name,pin['file']);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def aggregate(values,discrete):
    require(len(values)==96,'all96terminalvalues')
    mean=lambda x:math.fsum(x)/96
    pits=sorted(v['PIT_midpoint'] for v in values)
    return {'formula_groups':96,'mean_CRPS_native':mean([v['CRPS_native'] for v in values]),
            'RMSE_predictive_median_native':math.sqrt(mean([v['squared_error_native'] for v in values])),
            'coverage':{l:sum(v['intervals'][l]['covered'] for v in values)/96 for l in LEVELS},
            'mean_interval_width_native':{l:mean([v['intervals'][l]['width'] for v in values]) for l in LEVELS},
            'PIT_ECDF_deviation_descriptive':None if discrete else max(max((i+1)/96-v,v-i/96) for i,v in enumerate(pits)),
            'mean_PIT_left':mean([v['PIT_left'] for v in values]),'mean_PIT_right':mean([v['PIT_right'] for v in values])}


def aggregate_check(saved,own,c):
    c.exact('summary_groups',saved['formula_groups'],96)
    c.number('summary_CRPS',saved['mean_CRPS_native'],own['mean_CRPS_native'],2e-4,2e-5)
    for key in ('RMSE_predictive_median_native','mean_PIT_left','mean_PIT_right'):c.number('summary_'+key,saved[key],own[key])
    if own['PIT_ECDF_deviation_descriptive'] is None:c.exact('empirical_not_uniformPIT',saved['PIT_ECDF_deviation_descriptive'],None)
    else:c.number('summary_PIT_ECDF',saved['PIT_ECDF_deviation_descriptive'],own['PIT_ECDF_deviation_descriptive'],2e-10,0.)
    for level in LEVELS:
        c.number('summary_coverage_exact',saved['coverage'][level],own['coverage'][level],0,0)
        c.number('summary_interval_width',saved['mean_interval_width_native'][level],own['mean_interval_width_native'][level])


def mixture_compare(saved,own,c):
    c.number('mixture_CRPS_native',saved['CRPS_native'],own['CRPS_native'],2e-4,2e-5)
    c.number('mixture_native_median',saved['predictive_median_native'],own['predictive_median_native'],2e-7,2e-9)
    c.number('mixture_squared_error',saved['squared_error_native'],own['squared_error_native'])
    for key in ('PIT_left','PIT_right','PIT_midpoint'):c.number('mixture_'+key,saved[key],own[key],2e-10,0.)
    c.number('mixture_native_logdensity',saved['log_density_native'],own['log_density_native'],2e-9,2e-11)
    c.number('mixture_native_NLL',saved['negative_log_density_native'],-own['log_density_native'],2e-9,2e-11)
    for level in LEVELS:
        for key in ('lower','upper','width'):c.number('mixture_interval_'+key,saved['intervals'][level][key],own['intervals'][level][key],2e-7,2e-9)
        c.exact('mixture_interval_event',saved['intervals'][level]['covered'],own['intervals'][level]['covered'])


def audit(plan):
    for name,digest in plan['independent_sources'].items():require(sha(Path(__file__).parent/name)==digest,'scoreaudit source changed')
    loaded={}
    for name,pin in plan['inputs'].items():
        require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name);loaded[name]=read(pin['file'])
    helpers=plan['helpers']
    batch=load(helpers['batch_transform'],'saved_state_reference')
    old=load(helpers['scalar_score_functions'],'_independent_old_score_functions')
    native=load(helpers['native_CDF'],'_independent_scalar_nativeCDF');native.special=special;native.quad=quad
    require(sha(helpers['full_fixture_reference']['file'])==helpers['full_fixture_reference']['sha256'],'fullfixture purehelper changed')
    core=load(helpers['full_fixture_reference'],'full_native_reference')
    sys.path.insert(0,str(Path(helpers['mixture_reference']['file']).parent))
    mixture=load(helpers['mixture_reference'],'_independent_mixture_scoring_reference')
    require(Path(mixture.core.__file__).resolve()==Path(helpers['full_fixture_reference']['file']).resolve(),'mixture reference dependency collision')
    own=SimpleNamespace(transform=batch.forward_yj,inverse=old.inverse_yj);c=old.Checks()
    result=loaded['score_result'];freeze=loaded['forecast_freeze'];protocol=loaded['science_plan']
    require(loaded['state_audit']['status']=='PASS','all savedstates independently passed')
    for name,digest in protocol['source_sha256'].items():require(sha(Path(plan['producer_code_directory'])/name)==digest,'producer source changed')
    folder=Path(plan['forecast_directory'])
    for name,digest in freeze['files'].items():require(sha(folder/name)==digest,'saved forecast payload changed')
    c.exact('score_plan_binding',result['plan_sha256'],plan['inputs']['science_plan']['sha256'])
    c.exact('score_freeze_binding',result['forecast_freeze_sha256'],plan['inputs']['forecast_freeze']['sha256'])
    start=loaded['score_start'];c.exact('all4032beforetruth',start['all4032terminal_and48prequery_laws_verified_before_terminalY'],True)
    require(datetime.fromisoformat(freeze['actual_utc'])<=datetime.fromisoformat(start['actual_utc'])<=datetime.fromisoformat(result['actual_end_utc']),'freeze before firstterminalscore')
    roles=loaded['roles']['representatives'];truth=loaded['terminal_truth']['truth'];require(len(truth)==len(roles['terminal'])==96,'all96targets')
    target={}
    for row,meta in zip(truth,roles['terminal']):c.exact('truth_identity',row['identity'],meta);target[meta['component_id']]=row['native_target']
    require(len(target)==96 and all(math.isfinite(x) for x in target.values()),'96uniquefinitetargets')
    main=read(folder/'TERMINAL_FORECASTS.json')['forecasts'];refs=read(folder/'REFERENCE_FORECASTS.json')['forecasts']
    laws={(r['component_id'],r['policy'],r['budget']):r for r in main}
    reference_laws={(r['component_id'],r['policy'],r['budget'],r['method']):r for r in refs}
    saved={(r['component_id'],r['policy'],r['budget']):r for r in result['main_forecast_scores']}
    saved_refs={(r['component_id'],r['policy'],r['budget'],r['method']):r for r in result['reference_forecast_scores']}
    expected={(g,p,b) for g in target for p in POLICIES for b in BUDGETS}
    expected_ref={(g,p,b,m) for g in target for p in ('full_native','random') for b in BUDGETS for m in ('GP','RF','affine_NIG')}
    expected_ref.update((g,'gp_variance',b,'GP') for g in target for b in BUDGETS)
    expected_ref.update((g,'full_native',b,'frozen_weights_BMA') for g in target for b in BUDGETS)
    require(len(main)==len(laws)==len(saved)==1728 and set(laws)==set(saved)==expected,'all1728main scores')
    require(len(refs)==len(reference_laws)==len(saved_refs)==2304 and set(reference_laws)==set(saved_refs)==expected_ref,'all2304reference scores')
    cache={};quantile_cache={};routes={'mixture_nativeCDF':0,'scalar_nativeCDF':0,'empirical_energy':0};maxquad=0.
    def evaluate(y,law):
        nonlocal maxquad
        encoded=json.dumps(law,sort_keys=True,separators=(',',':'),allow_nan=False);key=(y,encoded)
        if key not in cache:
            if law['family']=='mixture_student_t':
                if encoded not in quantile_cache:quantile_cache[encoded]={p:mixture.quantile(law,p) for p in mixture.PROBABILITIES}
                q=quantile_cache[encoded];median=q[.5];crps,error=mixture.direct_crps(law,y);pit=mixture.cdf(law,y)
                intervals={}
                for level in LEVELS:
                    p=float(level);lo=q[min(q,key=lambda x:abs(x-(1-p)/2))];hi=q[min(q,key=lambda x:abs(x-(1+p)/2))]
                    intervals[level]={'lower':lo,'upper':hi,'width':hi-lo,'covered':lo<=y<=hi}
                cache[key]={'CRPS_native':crps,'predictive_median_native':median,'squared_error_native':(y-median)**2,
                            'PIT_left':pit,'PIT_right':pit,'PIT_midpoint':pit,'intervals':intervals,
                            'log_density_native':mixture.log_density(law,y),'quad_error_estimate_native':error}
                routes['mixture_nativeCDF']+=1
            else:
                cache[key]=old.evaluate(y,law,native,own);routes['empirical_energy' if law['family']=='weighted_empirical' else 'scalar_nativeCDF']+=1
        value=cache[key];maxquad=max(maxquad,value.get('quad_error_estimate_native',0.));return value
    values={};reference_values={}
    for lawmap,scoremap,output in ((laws,saved,values),(reference_laws,saved_refs,reference_values)):
        for key,forecast in lawmap.items():
            row=scoremap[key];y=target[key[0]];c.exact('score_jid',row['jid'],forecast['jid']);c.number('exact_native_target',row['native_target'],y,0,0)
            value=evaluate(y,forecast['law']);output[key]=value
            (mixture_compare if forecast['law']['family']=='mixture_student_t' else old.score_compare)(row['score'],value,c)
    summaries={};reference_summaries={}
    for policy in POLICIES:
        summaries[policy]={}
        for budget in BUDGETS:
            table=aggregate([values[(g,policy,budget)] for g in target],False)
            aggregate_check(result['summary_BMA_prediction_by_policy'][policy][str(budget)],table,c);summaries[policy][str(budget)]=table
    for policy in ('full_native','random','gp_variance'):
        methods=('GP','RF','affine_NIG') if policy!='gp_variance' else ('GP',)
        if policy=='full_native':methods+=('frozen_weights_BMA',)
        reference_summaries[policy]={}
        for method in methods:
            reference_summaries[policy][method]={}
            for budget in BUDGETS:
                table=aggregate([reference_values[(g,policy,budget,method)] for g in target],method=='RF')
                aggregate_check(result['summary_sameinformation_references'][policy][method][str(budget)],table,c)
                reference_summaries[policy][method][str(budget)]=table
    comparators=[(p,None) for p in POLICIES if p!='full_native']+[(p,m) for p in reference_summaries for m in reference_summaries[p]]
    paired={(r['component_id'],r['comparator']):r for r in result['paired_final_differences']}
    require(len(paired)==len(result['paired_final_differences'])==1248,'all13comparatorsx96paired')
    for policy,method in comparators:
        name=policy+('_'+method if method else '_BMA');differences=[]
        for g in target:
            a=values[(g,'full_native',8)]['CRPS_native'];b=reference_values[(g,policy,8,method)]['CRPS_native'] if method else values[(g,policy,8)]['CRPS_native'];d=a-b
            row=paired[(g,name)];c.exact('paired_final_budget',row['budget'],8)
            c.number('paired_final_difference',row['full_native_minus_reference_CRPS'],d,4e-4+2e-5*(abs(a)+abs(b)),0.);differences.append(d)
        table=result['paired_final_summaries_descriptive'][name];mean=math.fsum(differences)/96
        se=math.sqrt(math.fsum((d-mean)**2 for d in differences)/95)/math.sqrt(96)
        c.number('paired_descriptive_mean',table['mean_signed_CRPS_difference'],mean,4e-4,4e-5)
        c.number('paired_descriptive_SE',table['standard_error_descriptive'],se,4e-4,4e-5)
        # Literal signs near floating-point zero are bookkeeping of the saved
        # producer deltas, not independently resolved statistical improvements.
        reported=[paired[(g,name)]['full_native_minus_reference_CRPS'] for g in target]
        c.exact('paired_better_count',table['candidate_better_groups'],sum(d<0 for d in reported));c.exact('paired_worse_count',table['candidate_worse_groups'],sum(d>0 for d in reported))
        c.exact('paired_exact_ties',table['exact_score_ties'],sum(d==0 for d in reported))
    final=summaries['full_native']['8']['mean_CRPS_native']
    losses={p+'_BMA':summaries[p]['8']['mean_CRPS_native'] for p in POLICIES if p!='full_native'}
    losses.update({p+'_'+m:reference_summaries[p][m]['8']['mean_CRPS_native'] for p in reference_summaries for m in reference_summaries[p]})
    primary=result['primary_comparison'];c.number('primary_final_CRPS',primary['full_native_BMA_CRPS_budget8'],final,2e-4,2e-5)
    require(set(primary['other_final_losses'])==set(losses),'all13finalcomparators')
    for name,value in losses.items():c.number('primary_other_final_CRPS',primary['other_final_losses'][name],value,2e-4,2e-5)
    best=min(losses.values());c.number('primary_best_ratio',primary['candidate_to_best_reference_ratio'],final/best,2e-5,2e-5)
    saved_best=min(primary['other_final_losses'].values())
    c.exact('primary_within2percent_descriptive',primary['within2percent_descriptive_only'],primary['full_native_BMA_CRPS_budget8']<=1.02*saved_best)
    indices=read(folder/'PREQUENTIAL_INDEX.json')['records'];secondary=result['secondary_prequery_scores'];require(len(indices)==len(secondary)==48,'48prequeryscores')
    for row,item in zip(secondary,indices):
        before=read(folder/item['forecast_file']);after=read(folder/item['outcome_file']);y=after['true_Y_reveal']['native_target']
        for key in ('policy','step','pool_index'):c.exact('secondary_'+key,row[key],item[key])
        c.exact('secondary_identity',row['component_id'],before['identity']['component_id']);c.number('secondary_trueY',row['native_target'],y,0,0)
        c.exact('secondary_forecastSHA',row['forecast_sha256'],item['forecast_sha256']);c.exact('secondary_outcomeSHA',row['outcome_sha256'],item['outcome_sha256'])
        require(datetime.fromisoformat(row['prediction_utc'])<=datetime.fromisoformat(row['reveal_utc']),'prequery beforetrueY')
        mixture_compare(row['score'],evaluate(y,before['law']),c)
    states=read(folder/'MODEL_STATES.json')['checkpoint_states'];sequences={p:next(r['selected_pool_indices'] for r in states if r['policy']==p and r['budget']==8) for p in POLICIES}
    c.exact('actual_final_sequences',result['actual_selected_pool_sequences'],sequences)
    paths=[{'left':a,'right':b,'exact_sequence_equal':sequences[a]==sequences[b],'same_final_set':set(sequences[a])==set(sequences[b])} for a,b in itertools.combinations(POLICIES,2)]
    c.exact('policy_path_aliases',result['policy_path_comparisons'],paths)
    c.exact('all_terminal_scores',result['terminal_score_coordinates'],4032);c.exact('all_prequery_scores',result['prequery_score_coordinates'],48)
    c.exact('common_setup352',result['common_learning_setup_cells'],352);c.exact('perpolicy8',result['perpolicy_acquiredY_cells'],8);c.exact('all18fits',result['actual_sklearn_fits'],18)
    return {'status':'PASS','all_score_coordinates':4080,'all_terminal_scores':4032,'all_prequery_scores':48,
            'coverage_events':12240,'method_budget_summaries':42,'final_paired_differences':1248,
            'unique_independent_law_outcome_scores':len(cache),'reference_routes':routes,'max_nativeCDF_quad_error_estimate':maxquad,
            'independent_BMA_summary':summaries,'independent_reference_summary':reference_summaries,
            'independent_final_CRPS':final,'independent_best_reference_ratio':final/best,
            'within2percent_descriptive':final<=1.02*best,'comparison_counts':c.counts,'max_absolute_errors':c.errors,
            'literal_floating_sign_counts_checked_from_saved_deltas_not_claimed_resolved_in_reference':True,
            'SUT_score_runs_fits_or_raw_reads':0,'scope':'arithmetic verification of all fixed160 development source scores, not independent source/wet experiment confirmation'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--approval',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();plan=read(a.plan);approval=read(a.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_FRESH160_ALL_SCORES_ONCE' and approval['plan_sha256']==sha(a.plan)
            and approval['script_sha256']==sha(__file__),'exact root allscoreapproval')
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat()
    try:
        with warnings.catch_warnings():warnings.simplefilter('error',IntegrationWarning);result=audit(plan)
    except Exception as problem:result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
                   'plan_sha256':sha(a.plan),'approval_sha256':sha(a.approval)})
    f=out/'INDEPENDENT_FRESH160_ALL_SCORES_AUDIT.json'
    with f.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
