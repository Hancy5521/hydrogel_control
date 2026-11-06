"""Six fixed fresh-material only-Y trajectories; all laws freeze before terminal Y.

Pure conjugate/BMA core is frozen separately. Actual-label access is owned by the
trusted disclosure layer. No generated label updates, policy tuning or early STOP.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode=True
CORE=Path(__file__).resolve().parents[2]/'code'
sys.path.insert(0,str(CORE))
import data_io as task
import material_model as model
import acquisition
import reference_models as references
import relation_posterior as posterior
import risk_field as field

CORE_PINS={'relation_posterior.py':'1d74cde91519ce282c6ebee54e53807a2a79be0cebf19e06221eb4f16e8b7528',
    'risk_field.py':'cda5178b9c5138df647a0b9fffab2da30a19bde6082aa6fd4cc8b0910c6b92aa'}
ACQUISITION_LEVELS=(32,64,128,256)


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def versions():
    actual={name:importlib.metadata.version(name) for name in task.configuration()['libraries']}
    if actual!=task.configuration()['libraries']:raise ValueError('registered library versions required')
    return actual


def verify_foundation(plan):
    bound={str(Path(item['file']).resolve()):item['sha256'] for item in plan['external_sources'].values()}
    for module in (posterior,field):
        path=Path(module.__file__).resolve();expected=(CORE/path.name).resolve()
        if path!=expected or task.sha(path)!=CORE_PINS[path.name] or bound.get(str(path))!=CORE_PINS[path.name]:
            raise ValueError('exact original frozen mathematical core must be plan-bound')


def converged_acquisition(ensemble,pool,terminal,response,mode,failure_file,context):
    tolerance=2e-8 if mode=='model_MI' else 2e-5
    levels=[];previous=None;difference=None
    for count in ACQUISITION_LEVELS:
        result=acquisition.risk_values(ensemble,pool,terminal,response,mode=mode,count=count)
        scores=np.asarray(result['values'],dtype=float)
        if len(scores)!=len(pool) or not np.isfinite(scores).all():raise ArithmeticError('finite every-action readout required')
        difference=None if previous is None else float(np.max(np.abs(scores-previous)))
        levels.append({'count':count,'values':scores.tolist(),'max_absolute_consecutive_level_value_difference':difference,
            'diagnostics':result['diagnostics']})
        if difference is not None and difference<=tolerance:
            return scores.tolist(),{'mode':mode,'levels':levels,'selected_count':count,'absolute_tolerance':tolerance,
                'converged_by_registered_diagnostic':True,'rigorous_error_certificate':False}
        previous=scores
    task.write(failure_file,{'status':'ACQUISITION_NUMERICAL_DIAGNOSTIC_NOT_MET_SAFE_STOP','actual_utc':task.utc(),
        'context':context,'mode':mode,'levels':levels,'absolute_tolerance':tolerance,
        'no_new_label_requested_for_this_action':True,'no_model_or_budget_change':True})
    raise ArithmeticError('registered acquisition convergence unmet at256; saved all levels and stopped before next Y')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True)
    parser.add_argument('--output-dir',required=True);args=parser.parse_args()
    plan,roles=task.validate(args.plan,args.approval,'forecast');verify_foundation(plan);libraries=versions()
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False)
    task.write(output/'EXECUTION_START.json',{'actual_utc':task.utc(),'source_sha256':task.sources(),
        'plan_sha256':task.sha(args.plan),'approval_sha256':task.sha(args.approval),'terminalY_model_access':0})
    task.write(output/'LIBRARY_VERSIONS.json',{'libraries':libraries,'Python':sys.version})
    try:
        oracle=task.Disclosure(plan['inputs']['raw_source']['file'],roles);visible=oracle.initial()
        task.write(output/'VISIBLE_INPUTS.json',visible)
        warm_native=[r['native_values'] for r in visible['warm']]
        pool_native=[r['native_proxies'] for r in visible['pool']]
        terminal_native=[r['native_proxies'] for r in visible['terminal']]
        warm=model.fit_warm(warm_native);warm_ensemble=warm['ensemble'];response=warm['response']
        pool=model.proxy_z(warm['transform'],pool_native);terminal=model.proxy_z(warm['transform'],terminal_native)
        native_features=warm['native_features'];models_dir=output/'models';models_dir.mkdir()
        preq_dir=output/'prequential';preq_dir.mkdir()
        fits=[];prequential=[];primary=[];ref_forecasts=[];checkpoints=[]

        def fit_reference(method,train,prediction,fit_id,context):
            laws,state=(references.fit_gp(train,prediction,native_features) if method=='GP'
                else references.fit_forest(train,prediction,native_features))
            path=models_dir/(fit_id+'.json')
            task.write(path,{'fit_id':fit_id,'method':method,'context':context,'source_sha256':task.sources(),
                'actual_library_versions':versions(),'train_native_values_sha256':json_sha(train),'state':state})
            pin=task.sha(path);fits.append({'fit_id':fit_id,'method':method,'file':'models/'+path.name,
                'sha256':pin,'train_rows':len(train)})
            return laws,pin

        def pure_reference(method,ensemble,z):
            if method=='affine_NIG':return model.affine_law(ensemble,z,response)
            if method=='frozen_weights_BMA':return model.law(ensemble,z,response,logweights_override=warm_ensemble['log_weights'])
            raise ValueError('registered exact-state reference required')

        def append_pure_refs(policy,budget,ensemble,selected):
            methods=(('affine_NIG',) if policy in task.REF_POLICIES else ())+(('frozen_weights_BMA',) if policy=='full_native' else ())
            for method in methods:
                for item,z in zip(visible['terminal'],terminal):
                    ref_forecasts.append({'component_id':item['identity']['component_id'],'jid':item['identity']['jid'],
                        'policy':policy,'budget':budget,'method':method,'law':pure_reference(method,ensemble,z),
                        'same_actual_train_queries':list(selected),'exact_component_states_not_extra_fit':True,
                        'source_ensemble_sha256':json_sha(ensemble),'shared_warm_law_alias':budget==0})

        gp_warm_all,gp_warm_pin=fit_reference('GP',warm_native,pool_native+terminal_native,'GP_warm_shared','sharedwarmReferenceAndGPpolicy0')
        rf_warm,rf_warm_pin=fit_reference('RF',warm_native,terminal_native,'RF_warm_shared','sharedwarmReference')
        gp_warm_terminal=gp_warm_all[32:];warm_laws=[model.law(warm_ensemble,z,response) for z in terminal]
        seed=task.configuration()['random_seed'];random_order=np.random.default_rng(seed).permutation(32).tolist()
        for policy in task.POLICIES:
            ensemble=copy.deepcopy(warm_ensemble);selected=[];train=copy.deepcopy(warm_native)
            for item,predicted in zip(visible['terminal'],warm_laws):
                primary.append({'component_id':item['identity']['component_id'],'jid':item['identity']['jid'],
                    'policy':policy,'budget':0,'law':predicted,'shared_warm_law_alias':True})
            if policy in task.REF_POLICIES:
                for method,laws,pin in [('GP',gp_warm_terminal,gp_warm_pin),('RF',rf_warm,rf_warm_pin)]:
                    for item,predicted in zip(visible['terminal'],laws):
                        ref_forecasts.append({'component_id':item['identity']['component_id'],'jid':item['identity']['jid'],
                            'policy':policy,'budget':0,'method':method,'law':predicted,'fit_state_sha256':pin,
                            'shared_warm_actual_fit_alias':True})
            if policy=='gp_variance':
                for item,predicted in zip(visible['terminal'],gp_warm_terminal):
                    ref_forecasts.append({'component_id':item['identity']['component_id'],'jid':item['identity']['jid'],
                        'policy':policy,'budget':0,'method':'GP','law':predicted,'fit_state_sha256':gp_warm_pin,
                        'shared_warm_actual_fit_alias':True})
            append_pure_refs(policy,0,ensemble,selected)
            checkpoints.append({'policy':policy,'budget':0,'ensemble':copy.deepcopy(ensemble),
                'selected_pool_indices':[],'actual_utc':task.utc(),'source_sha256':task.sources()})
            for step in range(8):
                available=[i for i in range(32) if i not in selected]
                acquisition_ensemble=model.openloop_from_warm(warm_ensemble,ensemble) if policy=='warm_openloop' else copy.deepcopy(ensemble)
                diagnostics={};gp_policy_pin=None
                if policy in ('full_native','warm_openloop','marginal_native','model_MI'):
                    mode='full_native' if policy in ('full_native','warm_openloop') else policy
                    scores,diagnostics=converged_acquisition(acquisition_ensemble,pool[available],terminal,response,mode,
                        output/('ACQUISITION_FAILURE_%s_%02d.json'%(policy,step)),{'policy':policy,'budget_before_query':step,'available_pool_indices':available})
                elif policy=='random':
                    scores=[float(32-random_order.index(i)) for i in available]
                    diagnostics={'fixed_seed':seed,'fixed_permutation':random_order,'ranking_is_random_schedule_not_risk':True}
                elif policy=='gp_variance':
                    if step==0:predictions=gp_warm_all[:32];gp_policy_pin=gp_warm_pin
                    else:
                        predictions_all,gp_policy_pin=fit_reference('GP',train,pool_native+terminal_native,
                            'GP_policy_step%02d'%step,'gp_variance_policy_acquisition_and_own_predictions')
                        predictions=predictions_all[:32]
                        if step==4:
                            for item,predicted in zip(visible['terminal'],predictions_all[32:]):
                                ref_forecasts.append({'component_id':item['identity']['component_id'],'jid':item['identity']['jid'],
                                    'policy':policy,'budget':4,'method':'GP','law':predicted,'fit_state_sha256':gp_policy_pin,
                                    'same_actual_train_queries':list(selected),'reused_step4_fit_not_additional_fit':True})
                    scores=[predictions[i]['scale']**2 for i in available]
                    diagnostics={'native_GP_predictive_variance_with_White':True,'GP_actual_fit_state_sha256':gp_policy_pin,
                        'step0_shared_not_refit':step==0}
                else:raise ValueError('registered fixed policy required')
                if len(scores)!=len(available) or not np.isfinite(scores).all():raise ArithmeticError('finite every-action acquisition scores required')
                choice=max(range(len(available)),key=lambda j:scores[j]);chosen=available[choice]
                item=visible['pool'][chosen];before=copy.deepcopy(ensemble);predicted=model.law(ensemble,pool[chosen],response)
                stem=policy+'_step%02d'%step;before_file=preq_dir/(stem+'_FORECAST.json')
                task.write(before_file,{'status':'ACTION_AND_PREQUERY_MIXTURE_FROZEN','actual_utc':task.utc(),
                    'source_sha256':task.sources(),'plan_sha256':task.sha(args.plan),'policy':policy,'budget_before_query':step,
                    'pool_index':chosen,'identity':item['identity'],'native_query_target_visible':False,
                    'known_native_proxies':item['native_proxies'],'law':predicted,'actual_predictive_ensemble_before':before,
                    'acquisition_ensemble':acquisition_ensemble,'warmopenloop_acquisition_surrogate_not_actual_posterior':policy=='warm_openloop',
                    'available_pool_indices':available,'acquisition_values':scores,'acquisition_diagnostics':diagnostics,
                    'selected_position':choice,'sameunit_query':[item['identity']['component_id'],'bulk modulus'],
                    'source_GP_fit_sha256_if_policy':gp_policy_pin,'common_setup_cost':352})
                revealed=oracle.query_Y(policy,chosen,before_file,task.sources(),task.sha(args.plan))
                native_row=item['native_proxies']+[revealed['native_target']]
                ensemble,evidence=model.update_native(ensemble,pool[chosen],revealed['native_target'],response)
                modeled_y=float(model.response_z(response,revealed['native_target']))
                train.append(native_row);selected.append(chosen)
                after_file=preq_dir/(stem+'_OUTCOME_UPDATE.json')
                task.write(after_file,{'actual_utc':task.utc(),'policy':policy,'budget_after_query':step+1,'pool_index':chosen,
                    'identity':item['identity'],'prequery_forecast_sha256':task.sha(before_file),'true_Y_reveal':revealed,
                    'native_complete_row':native_row,'modeled_zY':modeled_y,'ensemble_after':ensemble,
                    'old_predictive_evidence_then_NIG_updates':evidence,'no_generated_label':True})
                prequential.append({'policy':policy,'step':step,'pool_index':chosen,'forecast_file':'prequential/'+before_file.name,
                    'forecast_sha256':task.sha(before_file),'outcome_file':'prequential/'+after_file.name,'outcome_sha256':task.sha(after_file)})
                if step+1 in task.BUDGETS:
                    budget=step+1
                    for target,z in zip(visible['terminal'],terminal):
                        primary.append({'component_id':target['identity']['component_id'],'jid':target['identity']['jid'],
                            'policy':policy,'budget':budget,'law':model.law(ensemble,z,response)})
                    checkpoints.append({'policy':policy,'budget':budget,'ensemble':copy.deepcopy(ensemble),
                        'selected_pool_indices':list(selected),'observed_native_training_sha256':json_sha(train),
                        'actual_utc':task.utc(),'source_sha256':task.sources(),'actual_library_versions':versions()})
                    append_pure_refs(policy,budget,ensemble,selected)
                    if policy in task.REF_POLICIES:
                        for method in ('GP','RF'):
                            laws,pin=fit_reference(method,train,terminal_native,'%s_%s_budget%d'%(method,policy,budget),
                                'sameactualtrainingStrongPredictionReference')
                            for target,predicted in zip(visible['terminal'],laws):
                                ref_forecasts.append({'component_id':target['identity']['component_id'],'jid':target['identity']['jid'],
                                    'policy':policy,'budget':budget,'method':method,'law':predicted,'fit_state_sha256':pin,
                                    'same_actual_train_queries':list(selected)})
            if policy=='gp_variance':
                laws,pin=fit_reference('GP',train,terminal_native,'GP_policy_final_budget8','gp_variance_policy_own_final_predictions')
                for target,predicted in zip(visible['terminal'],laws):
                    ref_forecasts.append({'component_id':target['identity']['component_id'],'jid':target['identity']['jid'],
                        'policy':policy,'budget':8,'method':'GP','law':predicted,'fit_state_sha256':pin,
                        'same_actual_train_queries':list(selected)})
            if len(selected)!=8:raise AssertionError('exactly8distinctYqueries for every policy')
        if (len(primary),len(ref_forecasts),len(prequential),len(fits))!=(1728,2304,48,18):raise AssertionError('fixed scientific counts changed')
        if task.sources()!=plan['source_sha256']:raise ValueError('science source changed during run')
        task.write(output/'MODEL_STATES.json',{'transform':warm['transform'],'response_transform':response,
            'native_reference_feature_transform':native_features,'warm_ensemble':warm_ensemble,
            'warm_log_evidence_by_H_modeled_response':warm['log_evidence_by_H_modeled_response'],
            'warm_log_evidence_by_H_native_response':warm['log_evidence_by_H_native_response'],
            'common_response_native_logJacobian_sum':warm['common_response_native_logJacobian_sum'],
            'warm_evidence_history':warm['warm_evidence_history'],'checkpoint_states':checkpoints,
            'allpool_standardized_known_proxies':pool.tolist(),'allterminal_standardized_known_proxies':terminal.tolist(),
            'fixed3H_no_weight_floor_all_withinH_retained':True})
        task.write(output/'QUERY_LEDGER.json',oracle.ledger());task.write(output/'TERMINAL_FORECASTS.json',{'forecasts':primary})
        task.write(output/'REFERENCE_FORECASTS.json',{'forecasts':ref_forecasts});task.write(output/'PREQUENTIAL_INDEX.json',{'records':prequential})
        task.write(output/'FIT_INDEX.json',{'records':fits,'actual_GP':13,'actual_RF':5,
            'warm_NIG_initializations':3,'feedback_component_NIG_updates':144,
            'sharedwarm_and_step4_GP_aliases_not_additional_fit':True,'affine_and_frozenweights_laws_no_new_fit':True})
        names=list(task.BASE_FILES)+[r[key] for r in prequential for key in ('forecast_file','outcome_file')]+[r['file'] for r in fits]
        if len(names)!=123 or len(set(names))!=123:raise AssertionError('exact123distinct freeze payloads required')
        task.write(output/'FORECAST_FREEZE.json',{'status':task.FORECAST_STATUS,'actual_utc':task.utc(),'source_sha256':task.sources(),
            'plan_sha256':task.sha(args.plan),'files':{name:task.sha(output/name) for name in names},'terminal_Y_model_access':0,
            'logical_terminal_laws':4032,'logical_prequery_laws':48,'unique_material_roles':160,
            'actual_sklearn_fits':18,'no_real_new_wet_or_DFT':True})
        print(json.dumps({'status':task.FORECAST_STATUS,'sklearn_fits':18,'primary_laws':1728,'reference_laws':2304}))
    except Exception as problem:
        task.write(output/'FAILED_STOP.json',{'actual_utc':task.utc(),'type':type(problem).__name__,'message':str(problem),
            'source_sha256':task.sources(),'retain_no_samecohort_recipe_rescue':True})
        raise

if __name__=='__main__':main()
