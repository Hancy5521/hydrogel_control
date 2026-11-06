"""One fixed synthetic full-NIG BMA gate, never material loading or model fitting.

Only a root-registered plan with exact source/fixture pins authorizes execution.
A diagnostic failure remains a complete saved result; criteria are never relaxed.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from scipy import special
import scipy
from relation_posterior import feature_map,normalize_logweights,update_actual,scalar_law,pairs,conditional_target
from risk_field import LEVELS,native_value,conditional_cdfs

HERE=Path(__file__).resolve().parent
BASE=HERE.parent
PROPOSAL_SHA='6ac2e7fc802bc22afb7dbd658d29e00efcae6d9c119bc3d5f9549b1efc715521'
SOURCE_NAMES=('relation_posterior.py','risk_field.py','run_gate.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def save(path,value):
    with path.open('x',encoding='utf-8') as s:json.dump(value,s,ensure_ascii=False,indent=2,allow_nan=False);s.write('\n')


def ensemble_from_proposal(p):
    models=[]
    for model in p['models']:
        state={k:copy.deepcopy(v) for k,v in model.items() if k!='id'}
        models.append({'id':model['id'],'state':state})
    return {'models':models,'log_weights':normalize_logweights(np.log(p['weights'])),
        'expansion_transform':None,'response_coordinate':'shared_fixed_synthetic_response',
        'true_label_updates':0}


def probes(ensemble,p):
    pairlaws=pairs(ensemble,p['query_context'],p['target_context'])
    points=np.asarray(p['target_modeled_CDF_points']);result=[]
    for value in p['fixed_observed_modeled_query_Y']:
        updated,evidence=update_actual(ensemble,p['query_context'],value)
        conditional=conditional_cdfs(pairlaws,ensemble['log_weights'],np.asarray([value]),points)
        component_laws=[];updated_cdfs=[]
        for old,new,pair in zip(ensemble['models'],updated['models'],pairlaws):
            phi=feature_map(new['id'],p['target_context'])
            law=scalar_law(new['state'],phi)
            direct=conditional_target(pair,(value-pair['mu_query'])/pair['scale_query'])
            errors={key:abs(law[key]-direct[key]) for key in ('mu','scale','nu')}
            updated_cdfs.append(special.stdtr(law['nu'],(points-law['mu'])/law['scale']))
            component_laws.append({'id':new['id'],'updated_NIG':new['state'],
                'updated_NIG_target_law':law,'bivariate_conditional_target_law':direct,
                'conditional_parameter_absolute_differences':errors})
        weights=np.exp(updated['log_weights']);updated_mix=np.column_stack(updated_cdfs)@weights
        cdf=conditional['mixture_CDF'][:,0]
        result.append({'artificial_modeled_query_Y':value,'not_a_material_measurement':True,
            'evidence':evidence,'updated_models':component_laws,'updated_log_weights':updated['log_weights'],
            'bivariate_query_posterior_weights':conditional['posterior_weights'][0].tolist(),
            'weight_max_absolute_difference':float(np.max(np.abs(weights-conditional['posterior_weights'][0]))),
            'target_modeled_CDF_points':points.tolist(),'mixture_CDF_from_updated_NIG':updated_mix.tolist(),
            'mixture_CDF_from_bivariate_conditionals':cdf.tolist(),
            'mixture_CDF_max_absolute_difference':float(np.max(np.abs(updated_mix-cdf)))})
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    plan=json.loads(args.plan.read_text('utf-8'))
    if plan.get('approved_for_execution') is not True or not plan.get('actual_registered_utc'):
        raise ValueError('root actual-UTC registered approval required')
    actual_sources={name:sha(HERE/name) for name in SOURCE_NAMES}
    if plan['source_sha256']!=actual_sources:raise ValueError('source hash pin mismatch')
    proposal_path=BASE/'FIXED_DATAFREE_GATE_PROPOSAL.json'
    if plan['proposal_sha256']!=PROPOSAL_SHA or sha(proposal_path)!=PROPOSAL_SHA:raise ValueError('fixed pre-output proposal mismatch')
    p=json.loads(proposal_path.read_text('utf-8'))
    if tuple(p['quadrature_levels'])!=LEVELS or plan['criteria']!=p['criteria']:raise ValueError('frozen levels/criteria mismatch')
    if args.output.exists():raise FileExistsError('one execution output must not already exist')
    args.output.mkdir(parents=True)
    started=utc();save(args.output/'EXECUTION_START.json',{'actual_utc':started,'plan_sha256':sha(args.plan),
        'proposal_sha256':PROPOSAL_SHA,'source_sha256':actual_sources,'python':platform.python_version(),
        'numpy':np.__version__,'scipy':scipy.__version__,'scope':'ONE_SYNTHETIC_FIXTURE_NO_MATERIAL'})
    before=time.perf_counter();ensemble=ensemble_from_proposal(p)
    evaluations=[native_value(ensemble,p['query_context'],p['target_context'],p['response_transform'],level) for level in LEVELS]
    probe_results=probes(ensemble,p)
    quantities=('native_full_VOI','native_weighted_within_total','native_B_before','native_E_B_after','native_signed_between')
    changes={key:abs(evaluations[-1][key]-evaluations[-2][key]) for key in quantities}
    within_changes=[abs(a-b) for a,b in zip(evaluations[-1]['native_weighted_within_components'],evaluations[-2]['native_weighted_within_components'])]
    maxchange=max(list(changes.values())+within_changes)
    criteria=p['criteria'];tail=evaluations[-1]['tail'];last=evaluations[-1]
    maxparam=max(v for probe in probe_results for model in probe['updated_models'] for v in model['conditional_parameter_absolute_differences'].values())
    maxweight=max(x['weight_max_absolute_difference'] for x in probe_results)
    maxcdf=max(x['mixture_CDF_max_absolute_difference'] for x in probe_results)
    closure_allowance=tail['finite_query_decomposition_closure_allowance']+3*maxchange
    checks={'conditional_parameter_algebra':maxparam<=criteria['conditional_NIG_vs_bivt_parameter_absolute'],
        'model_weight_algebra':maxweight<=criteria['Hweight_absolute'],
        'conditional_mixture_CDF_algebra':maxcdf<=criteria['conditional_mixture_CDF_absolute'],
        'final_refinement_diagnostic':maxchange<=criteria['final_refinement_diagnostic_absolute'],
        'ideal_tail_account':tail['total_value_tail_bound_ideal_arithmetic']<=criteria['ideal_tail_account_absolute'],
        'finite_cutoff_decomposition_diagnostic':abs(last['decomposition_closure'])<=closure_allowance,
        'no_negative_between_clipping':last['signed_between_clipped'] is False}
    result={'status':'COMPLETE_ALL_PRODUCER_DIAGNOSTICS_PASS' if all(checks.values()) else 'COMPLETE_WITH_DIAGNOSTIC_FAILURE',
        'actual_start_utc':started,'actual_end_utc':utc(),'elapsed_seconds':time.perf_counter()-before,
        'plan_sha256':sha(args.plan),'proposal_sha256':PROPOSAL_SHA,'source_sha256':actual_sources,
        'fixed_fixture':p,'initial_ensemble':ensemble,'evaluations_by_level':evaluations,
        'synthetic_fixed_label_probes':probe_results,'last_refinement_differences':changes,
        'last_within_component_refinement_differences':within_changes,'max_refinement_proxy':maxchange,
        'decomposition_closure_diagnostic_allowance':closure_allowance,
        'max_conditional_parameter_algebra_difference':maxparam,'max_weight_algebra_difference':maxweight,
        'max_mixture_CDF_algebra_difference':maxcdf,'checks':checks,'criteria_unchanged':True,
        'signed_between_is_negative_beyond_recorded_numerical_proxy':last['native_signed_between'] < -(3*maxchange+tail['total_value_tail_bound_ideal_arithmetic']),
        'not_independent_reference_PASS_until_separate_audit':True,'material_reads_fits_or_queries':0,
        'classical_method_not_claimed_new_Bayesian_algorithm':True,
        'limitations':['Single artificial fixture; no material benefit or active-policy effect.',
            'Finite quadrature refinement is a diagnostic, not rigorous error certificate.',
            'CDF library and floating-point errors are not enclosed by analytic tail bounds.',
            'No parameter scans, old material rescue, new model stack or signed-term clipping.']}
    save(args.output/'RESULTS.json',result)
    save(args.output/'EXECUTION_COMPLETE.json',{'actual_utc':utc(),'results_sha256':sha(args.output/'RESULTS.json'),
        'status':result['status'],'checks':checks,'model_fits':0,'material_label_reads':0})
    print(json.dumps({'status':result['status'],'results_sha256':sha(args.output/'RESULTS.json'),
        'full_native_VOI':last['native_full_VOI'],'within_total':last['native_weighted_within_total'],
        'signed_between':last['native_signed_between'],'checks':checks},indent=2))

if __name__=='__main__':main()
