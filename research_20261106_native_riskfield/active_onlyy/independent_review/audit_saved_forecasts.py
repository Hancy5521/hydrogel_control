"""Audit frozen active-onlyY state, information ledgers and laws; never fit or open raw data."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import traceback
import numpy as np
import model_reference as model

POLICIES = ('native_student','native_gaussian','latent_ivr','random','gp_variance','warm_openloop')
BUDGETS = (0,4,8)


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text('utf-8-sig'))
def jsha(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(value, message): model.require(value,message)


def independent_module(pin, name):
    require(sha(pin['file']) == pin['sha256'],'pinned independent helper changed')
    spec = importlib.util.spec_from_file_location(name,pin['file'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(plan,c):
    root = Path(__file__).parent
    for name,digest in plan['independent_sources'].items():
        require(sha(root/name) == digest,'independent source changed: '+name)
    for key,pin in plan['inputs'].items():
        require(sha(pin['file']) == pin['sha256'],'pinned input changed: '+key)
    ref = independent_module(plan['independent_batch_helper'],'_independent_batch_reference')
    h = independent_module(plan['independent_baseline_helper'],'_independent_scalar_tree_reference')
    protocol = read(plan['inputs']['science_plan']['file'])
    for name,digest in protocol['source_sha256'].items():
        require(sha(Path(plan['producer_code_directory'])/name) == digest,'producer source changed')
    roles = read(plan['inputs']['roles']['file'])['representatives']
    folder = Path(plan['forecast_directory'])
    freeze = read(plan['inputs']['forecast_freeze']['file'])
    c.exact('freeze_status',freeze['status'],'ALL2112_TERMINAL_AND48_PREQUENTIAL_LAWS_FROZEN')
    c.exact('freeze_plan',freeze['plan_sha256'],plan['inputs']['science_plan']['sha256'])
    c.exact('freeze_sources',freeze['source_sha256'],protocol['source_sha256'])
    c.exact('freeze_payload_count',len(freeze['files']),123)
    for name,digest in freeze['files'].items():
        file = (folder/name).resolve()
        file.relative_to(folder.resolve())
        require(sha(file) == digest,'frozen payload changed: '+name)
    visible = read(folder/'VISIBLE_INPUTS.json')
    states = read(folder/'MODEL_STATES.json')
    ledger = read(folder/'QUERY_LEDGER.json')
    indices = read(folder/'PREQUENTIAL_INDEX.json')['records']
    fit_index = read(folder/'FIT_INDEX.json')
    primary = read(folder/'TERMINAL_FORECASTS.json')['forecasts']
    refs = read(folder/'REFERENCE_FORECASTS.json')['forecasts']
    c.exact('visible_counts',[len(visible[k]) for k in ('warm','pool','terminal')],[64,24,64])
    for role in ('warm','pool','terminal'):
        field = 'native_values' if role == 'warm' else 'native_proxies'
        for saved,meta in zip(visible[role],roles[role]):
            c.exact('identity_'+role,saved['identity'],meta)
            require(set(saved) == {'identity',field},'unrevealed target exposed in visible '+role)
    warm = [r['native_values'] for r in visible['warm']]
    pool = [r['native_proxies'] for r in visible['pool']]
    terminal = [r['native_proxies'] for r in visible['terminal']]
    transform = ref.warm_transform(warm)
    for key,value in transform.items(): c.array('fixed_warm_'+key,states['transform'][key],value)
    response = {'center':transform['u_center_warm'][2], 'sd':transform['u_sample_SD_warm'][2],
                's0':transform['native_first_scale_s0'][2]}
    for key,value in response.items(): c.number('response_'+key,states['response_transform'][key],value)
    feature = {'center':transform['proxy_native_center_warm'],'sd':transform['proxy_native_sample_SD_warm']}
    for key,value in feature.items(): c.array('reference_feature_'+key,states['native_reference_feature_transform'][key],value)
    pz = [ref.transform_row(transform,row) for row in pool]
    tz = [ref.transform_row(transform,row) for row in terminal]
    c.array('known_pool_z',states['allpool_standardized_known_proxies'],pz)
    c.array('known_terminal_z',states['allterminal_standardized_known_proxies'],tz)
    warm_joint = ref.joint_state(transform,warm)
    for key in ('dimension','n','kappa','nu'): c.number('warm_NIW_'+key,states['warm_pooled_NIW'][key],warm_joint['pooled'][key])
    for key in ('mean','Psi'): c.array('warm_NIW_'+key,states['warm_pooled_NIW'][key],warm_joint['pooled'][key])
    warm_nig = model.induced_nig(warm_joint['pooled'],0)
    model.nig_check(states['warm_NIG'],warm_nig,c,'warm_NIG')
    inventory = {(r['component_id'],p,b) for r in roles['terminal'] for p in POLICIES for b in BUDGETS}
    laws = {(r['component_id'],r['policy'],r['budget']):r for r in primary}
    require(len(primary) == len(laws) == 1152 and set(laws) == inventory,'all 1152 primary law identities')
    reference_inventory = {(r['component_id'],p,b,m) for r in roles['terminal'] for p in ('native_student','random') for b in BUDGETS for m in ('GP','RF')}
    reference_inventory.update((r['component_id'],'gp_variance',b,'GP') for r in roles['terminal'] for b in BUDGETS)
    reference_laws = {(r['component_id'],r['policy'],r['budget'],r['method']):r for r in refs}
    require(len(refs) == len(reference_laws) == 960 and set(reference_laws) == reference_inventory,'all960reference identities')
    checkpoint = {(r['policy'],r['budget']):r for r in states['checkpoint_states']}
    require(len(checkpoint) == 18,'18 NIG checkpoints')
    for row in primary+refs:
        meta = next(r for r in roles['terminal'] if r['component_id'] == row['component_id'])
        c.exact('terminal_forecast_jid',row['jid'],meta['jid'])
    preq = {(r['policy'],r['step']):r for r in indices}
    events = {(r['policy'],r['policy_total_query_cost']-1):r for r in ledger['events']}
    require(len(indices) == len(preq) == len(events) == len(ledger['events']) == 48,'all48events unique')
    # Every fitted native GP/forest is reconstructed after exact training prefixes are known.
    training = {}
    prefixes = {}
    chosen_per_policy = {}
    risk_diagnostics = []
    pending_GP_actions = []
    fixed_random = np.random.default_rng(20261106).permutation(24).tolist()
    for policy in POLICIES:
        train = list(warm)
        selected = []
        for step in range(9):
            expected_joint = ref.joint_state(transform,train)
            expected_nig = model.induced_nig(expected_joint['pooled'],step)
            training[(policy,step)] = list(train)
            prefixes[(policy,step)] = list(selected)
            if step in BUDGETS:
                snap = checkpoint[(policy,step)]
                model.nig_check(snap['nig'],expected_nig,c,'checkpoint_NIG')
                c.exact('checkpoint_selected',snap['selected_pool_indices'],selected)
                if step:
                    c.exact('checkpoint_train_sha',snap['observed_native_training_sha256'],jsha(train))
                for meta,own in zip(roles['terminal'],terminal):
                    model.t_check(laws[(meta['component_id'],policy,step)]['law'],ref.target_law(expected_joint,own,pooled=True),c)
            if step == 8: break
            item = preq[(policy,step)]
            for key,hashkey in (('forecast_file','forecast_sha256'),('outcome_file','outcome_sha256')):
                c.exact('prequential_frozen_sha',sha(folder/item[key]),item[hashkey])
            before = read(folder/item['forecast_file'])
            after = read(folder/item['outcome_file'])
            chosen = before['pool_index']
            require(chosen not in selected and 0 <= chosen < 24,'new legal pool action')
            c.exact('preq_index',item['pool_index'],chosen)
            for record in (before,after):
                c.exact('query_identity',record['identity'],roles['pool'][chosen])
                c.exact('query_policy',record['policy'],policy)
            c.exact('query_target_hidden',before['native_query_target_visible'],False)
            c.exact('query_known_proxies',before['known_native_proxies'],pool[chosen])
            c.exact('query_budget_before',before['budget_before_query'],step)
            c.exact('query_budget_after',after['budget_after_query'],step+1)
            c.exact('query_plan',before['plan_sha256'],plan['inputs']['science_plan']['sha256'])
            c.exact('query_sources',before['source_sha256'],protocol['source_sha256'])
            model.nig_check(before['actual_predictive_nig_before'],expected_nig,c,'prequery_NIG')
            model.t_check(before['law'],ref.target_law(expected_joint,pool[chosen],pooled=True),c)
            acquisition = dict(expected_nig)
            if policy == 'warm_openloop':
                acquisition = dict(warm_nig)
                acquisition['V'] = expected_nig['V']
                acquisition['alpha'] = warm_nig['alpha']+step/2
                acquisition['b'] = (acquisition['alpha']-1)*warm_nig['b']/(warm_nig['alpha']-1)
                acquisition['labeled_updates'] = step
            model.nig_check(before['acquisition_nig'],acquisition,c,'acquisition_NIG')
            c.exact('openloop_surrogate_tag',before['warmopenloop_acquisition_surrogate_not_actual_posterior'],policy=='warm_openloop')
            available = [i for i in range(24) if i not in selected]
            c.exact('available_actions',before['available_pool_indices'],available)
            scores = before['acquisition_values']
            require(len(scores) == len(available) and all(math.isfinite(v) for v in scores),'finite all-action scores')
            argmax = max(range(len(scores)),key=lambda i:scores[i])
            c.exact('action_argmax',before['selected_position'],argmax)
            c.exact('action_chosen',chosen,available[argmax])
            if policy == 'random':
                c.exact('random_seed_permutation',before['acquisition_diagnostics']['fixed_permutation'],fixed_random)
                c.exact('random_action_values',scores,[float(24-fixed_random.index(i)) for i in available])
            elif policy == 'latent_ivr':
                matrix = np.asarray(expected_nig['V'])
                values=[]
                for i in available:
                    q=np.array([1.]+pz[i]);qd=1+q@matrix@q
                    values.append(expected_nig['b']/(expected_nig['alpha']-1)*math.fsum(float(q@matrix@np.array([1.]+x))**2 for x in tz)/(64*qd))
                c.array('latent_IVR_values',scores,values)
            elif policy == 'gp_variance':
                pending_GP_actions.append((before,available,scores))
            else:
                diag=before['acquisition_diagnostics']
                c.exact('risk_family',diag['family'],'normal' if policy=='native_gaussian' else 'student_t')
                c.exact('risk_target_count',diag['target_count'],64)
                c.exact('risk_query_count',diag['query_count'],len(available))
                c.number('risk_df',diag['nu'],2*acquisition['alpha'])
                c.exact('risk_target_average',diag['mean_over_all_targets_not_sum'],True)
                levels=diag['levels_evaluated']
                require([r['nodes_per_segment'] for r in levels] in ([32,64],[32,64,128],[32,64,128,256]),'fixed refinement levels')
                require(levels[-1]['max_mean_native_change'] <= 2e-5,'actual risk refinement diagnostic')
                require(diag['mean_tail_bound_ideal_arithmetic'] >= 0,'nonnegative ideal tail account')
                mu=[response['center']+response['sd']*float(np.array([1.]+x)@np.array(acquisition['mean'])) for x in tz]
                c.exact('risk_positive_decomposition_count',diag['positive_branch_decomposition_targets'],sum(v>=0 for v in mu))
                c.exact('risk_negative_direct_count',diag['negative_mean_direct_integral_targets'],sum(v<0 for v in mu))
                rho=[]
                v=np.array(acquisition['V'])
                for i in available:
                    q=np.array([1.]+pz[i])
                    for x in tz:
                        t=np.array([1.]+x);rho.append(abs(float(q@v@t)/math.sqrt((1+q@v@q)*(1+t@v@t))))
                c.number('risk_max_abs_rho',diag['max_abs_rho'],max(rho))
                risk_diagnostics.append({'policy':policy,'step':step,'last_change':levels[-1]['max_mean_native_change'],
                                         'mean_ideal_tail':diag['mean_tail_bound_ideal_arithmetic'],
                                         'native_action_values_independently_reintegrated':False})
            event=events[(policy,step)]
            c.exact('outcome_event',after['true_Y_reveal'],event)
            c.exact('event_identity',event['identity'],roles['pool'][chosen])
            c.exact('event_pool_index',event['pool_index'],chosen)
            c.exact('event_one_cell_cost',event['incremental_cost'],1)
            c.exact('event_policy_cost',event['policy_total_query_cost'],step+1)
            c.exact('event_not_synthetic',event['synthetic_label'],False)
            c.exact('event_prefreeze_sha',event['prequery_file_sha256'],item['forecast_sha256'])
            c.exact('after_prefreeze_sha',after['prequery_forecast_sha256'],item['forecast_sha256'])
            require(datetime.fromisoformat(before['actual_utc']) <= datetime.fromisoformat(event['actual_utc']) <= datetime.fromisoformat(after['actual_utc']),'true-label prequential chronology')
            native = pool[chosen]+[event['native_target']]
            c.exact('actual_complete_row',after['native_complete_row'],native)
            c.number('actual_label_coordinate',after['modeled_zY'],ref.transform_row(transform,native)[2])
            train.append(native);selected.append(chosen)
            next_nig=model.induced_nig(ref.joint_state(transform,train)['pooled'],step+1)
            model.nig_check(after['nig_after'],next_nig,c,'after_NIG')
        chosen_per_policy[policy]=selected
    c.exact('common_cost',ledger['common_setup_native_cells'],368)
    c.exact('perpolicy_cost',ledger['cost_per_policy'],{p:8 for p in POLICIES})
    c.exact('terminalY_access',ledger['terminalY_model_access'],0)
    physical_union=len(set(v for values in chosen_per_policy.values() for v in values))
    c.exact('physical_pool_union',ledger['union_pool_components_with_Y_extracted'],physical_union)
    # Reconstruct every saved optimizer result from state, not by calling sklearn.
    fit_laws={};fit_reports=[];fit_rows=fit_index['records']
    require(len(fit_rows)==18 and sum(r['method']=='GP' for r in fit_rows)==13 and sum(r['method']=='RF' for r in fit_rows)==5,'13 GP and5RF actual fits')
    for item in fit_rows:
        record=read(folder/item['file']);fid=item['fit_id'];method=item['method']
        c.exact('fit_hash',sha(folder/item['file']),item['sha256'])
        c.exact('fit_id',record['fit_id'],fid);c.exact('fit_method',record['method'],method)
        if fid in ('GP_warm_shared','RF_warm_shared'): policy,step='native_student',0
        elif fid.startswith('GP_policy_step'): policy,step='gp_variance',int(fid[-2:])
        elif fid=='GP_policy_final_budget8': policy,step='gp_variance',8
        else:
            suffix=fid[len(method)+1:];policy,budget=suffix.rsplit('_budget',1);step=int(budget)
        train=training[(policy,step)]
        query=pool+terminal if fid=='GP_warm_shared' or fid.startswith('GP_policy_step') else terminal
        c.exact('fit_training_hash',record['train_native_values_sha256'],jsha(train))
        c.exact('fit_source',record['source_sha256'],protocol['source_sha256'])
        xfit=[[(r[j]-feature['center'][j])/feature['sd'][j] for j in (0,1)] for r in train]
        xquery=[[(r[j]-feature['center'][j])/feature['sd'][j] for j in (0,1)] for r in query]
        y=[r[2] for r in train]
        fn=model.gp_check if method=='GP' else model.forest_check
        predictions,report=fn(record['state'],xfit,xquery,y,c,h)
        fit_laws[item['sha256']]=(predictions,24 if len(query)==88 else 0)
        fit_reports.append({'fit_id':fid,'method':method,**report})
    for row in refs:
        predictions,offset=fit_laws[row['fit_state_sha256']]
        position=next(i for i,r in enumerate(roles['terminal']) if r['component_id']==row['component_id'])
        model.baseline_law_check(row['law'],predictions[offset+position],c)
        if row['budget']:
            c.exact('reference_same_actual_training_queries',row['same_actual_train_queries'],prefixes[(row['policy'],row['budget'])])
    for before,available,scores in pending_GP_actions:
        predictions,offset=fit_laws[before['source_GP_fit_sha256_if_policy']]
        require(offset==24,'GP query+terminal predictions preserved')
        for value,i in zip(scores,available):
            # Squaring native SD propagates the independently predeclared GP law tolerance.
            sd=predictions[i]['scale']
            c.number('GP_variance_acquisition',value,sd*sd,4e-6*abs(sd)+4e-12,4e-8)
    aliases=[]
    for i,left in enumerate(POLICIES):
        for right in POLICIES[i+1:]:
            budget_rows=[]
            for budget in BUDGETS:
                error=max(abs(laws[(r['component_id'],left,budget)]['law'][key]-laws[(r['component_id'],right,budget)]['law'][key])
                          for r in roles['terminal'] for key in ('location','scale','df','native_first_scale_s0'))
                budget_rows.append({'budget':budget,'same_selected_set':set(prefixes[(left,budget)])==set(prefixes[(right,budget)]),
                                    'max_primary_law_parameter_abs_difference':error})
            aliases.append({'left':left,'right':right,'same_exact_action_sequence':chosen_per_policy[left]==chosen_per_policy[right],
                            'same_final_selected_set':set(chosen_per_policy[left])==set(chosen_per_policy[right]),
                            'budget_alias_diagnostics':budget_rows})
    return {'status':'PASS','primary_t_laws':1152,'reference_native_laws':960,'secondary_prequential_laws':48,
            'true_label_updates':48,'actual_baseline_fit_states_audited':18,'fit_state_checks':fit_reports,
            'physical_unique_pool_groups_queried':physical_union,'selected_sequences':chosen_per_policy,
            'policy_aliases':aliases,
            'saved_risk_diagnostics':risk_diagnostics,'terminal_target_reads':0,'SUT_imports_or_refits':0,
            'independent_native_risk_material_reintegration':False,
            'risk_audit_scope':'NIG context/coordinate/family/refinement/argmax checks and previously actual independently tested fixed helper; not reintegration of all material acquisition values',
            'scope':'saved-state and forecast arithmetic on consumed development source, not effect evaluation or new materials'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();plan=read(args.plan);approval=read(args.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_ACTIVE_ONLYY_SAVED_FORECAST_ONCE'
            and approval['plan_sha256']==sha(args.plan) and approval['script_sha256']==sha(__file__),'exact root audit approval')
    output=Path(args.output_dir);output.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat();c=model.Checks()
    try: result=audit(plan,c)
    except Exception as problem:
        result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
                   'plan_sha256':sha(args.plan),'approval_sha256':sha(args.approval),'numeric_counts':c.counts,'max_absolute_errors':c.errors})
    file=output/'INDEPENDENT_ACTIVE_ONLYY_STATE_AUDIT.json'
    with file.open('x',encoding='utf-8') as stream: json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(file),'sha256':sha(file)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
