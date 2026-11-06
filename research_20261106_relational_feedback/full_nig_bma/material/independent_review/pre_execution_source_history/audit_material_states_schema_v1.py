"""Independent saved-statistics/law audit of the fixed160 material feedback pilot; no refit."""
import argparse
from datetime import datetime,timezone
import hashlib,importlib.util,json,math
from pathlib import Path
import traceback
import numpy as np
import bma_state_reference as bma

POLICIES=('full_native','warm_openloop','random','marginal_native','model_MI','gp_variance')
BUDGETS=(0,4,8)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def jsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def require(x,m):
    if not x:raise AssertionError(m)


def load(pin,name):
    require(sha(pin['file'])==pin['sha256'],'independent helper changed:'+name)
    spec=importlib.util.spec_from_file_location(name,pin['file']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def warm_common_jacobian(rows,t):
    first=t['native_first_scale_s0'][2];second=t['u_sample_SD_warm'][2]
    return math.fsum(-math.log(first)-math.log(second)+(-.5*math.log1p(r[2]/first) if r[2]>=0 else .5*math.log1p(-r[2]/first)) for r in rows)


def acquisition_check(before,policy,step,available,c):
    scores=before['acquisition_values'];require(len(scores)==len(available) and all(math.isfinite(v) for v in scores),'finite everylegalaction score')
    choice=max(range(len(scores)),key=lambda i:scores[i]);c.exact('argmax_position',before['selected_position'],choice)
    c.exact('argmax_action',before['pool_index'],available[choice])
    if policy not in ('full_native','warm_openloop','marginal_native','model_MI'):return None
    diag=before['acquisition_diagnostics'];mode='full_native' if policy in ('full_native','warm_openloop') else policy
    c.exact('readout_mode',diag['mode'],mode);tol=2e-8 if mode=='model_MI' else 2e-5
    c.number('registered_readout_tol',diag['absolute_tolerance'],tol,0,0)
    levels=diag['levels'];nodes=[r['count'] for r in levels]
    require(nodes in ([32,64],[32,64,128],[32,64,128,256]),'fixed consecutive readout levels')
    for i,level in enumerate(levels):
        values=level['values'];require(len(values)==len(available) and all(math.isfinite(v) for v in values),'readout everyaction atalllevels')
        item=level['diagnostics'];c.exact('all96targetcontexts',item['target_contexts'],96)
        c.exact('allremainingquerycontexts',item['available_queries'],len(available));c.exact('readout_targetlabels0',item['target_labels_read'],0)
        if i:
            difference=max(abs(a-b) for a,b in zip(values,levels[i-1]['values']))
            c.number('readout_refinement_difference',level['max_absolute_consecutive_level_value_difference'],difference)
        if mode=='full_native':
            c.exact('fullreadout_paircount',item['full_pair_evaluations'],96*len(available))
            for value,q in zip(values,item['per_query']):
                target=q['per_target_values'];require(len(target)==96,'all96 targets retained')
                mean=math.fsum(r['full'] for r in target)/96
                c.number('nativefield_target_average',value,mean)
                c.number('nativefield_diag_mean',q['mean_full_value'],mean)
                c.number('nativefield_within_mean',q['mean_within_value'],math.fsum(r['within'] for r in target)/96)
                c.number('nativefield_signedbetween_mean',q['mean_signed_between'],math.fsum(r['signed_between'] for r in target)/96)
                for r in target:
                    c.number('nativefield_components_sum',r['within'],math.fsum(r['within_components']))
                    c.number('nativefield_signed_B_identity',r['signed_between'],r['B_before']-r['E_B_after'])
                    c.number('nativefield_closure_identity',r['closure'],r['full']-r['within']-r['signed_between'])
        else:
            for value,q in zip(values,item['per_query']):c.number('scalar_readout_value_saved',value,q['value'])
    c.exact('selected_readout_nodes',diag['selected_count'],nodes[-1]);c.array('actual_action_values_are_lastlevel',scores,levels[-1]['values'])
    require(levels[-1]['max_absolute_consecutive_level_value_difference']<=tol,'actual lastrefinement passbeforeY')
    return {'policy':policy,'step':step,'mode':mode,'selected_nodes':nodes[-1],
            'last_max_abs_difference':levels[-1]['max_absolute_consecutive_level_value_difference'],
            'fullmaterial_2D_reintegration':False}


def audit(plan,c,models):
    for name,digest in plan['independent_sources'].items():require(sha(Path(__file__).parent/name)==digest,'audit source changed')
    for name,pin in plan['inputs'].items():require(sha(pin['file'])==pin['sha256'],'pinned input changed:'+name)
    old=load(plan['helpers']['batch_transform'],'_old_independent_warm_transform')
    trees=load(plan['helpers']['scalar_tree'],'_old_independent_scalar_tree')
    protocol=read(plan['inputs']['science_plan']['file']);roles=read(plan['inputs']['roles']['file'])['representatives']
    for name,digest in protocol['source_sha256'].items():require(sha(Path(plan['producer_code_directory'])/name)==digest,'producer source changed')
    folder=Path(plan['forecast_directory']);freeze=read(plan['inputs']['forecast_freeze']['file'])
    c.exact('forecast_freeze_status',freeze['status'],'ALL4032_TERMINAL_AND48_PREQUERY_MIXTURE_LAWS_FROZEN')
    c.exact('forecast_plan_binding',freeze['plan_sha256'],plan['inputs']['science_plan']['sha256'])
    c.exact('forecast_sources_binding',freeze['source_sha256'],protocol['source_sha256']);c.exact('123frozenpayloads',len(freeze['files']),123)
    for name,digest in freeze['files'].items():
        file=(folder/name).resolve();file.relative_to(folder.resolve());require(sha(file)==digest,'frozen payload changed:'+name)
    visible=read(folder/'VISIBLE_INPUTS.json');saved=read(folder/'MODEL_STATES.json');ledger=read(folder/'QUERY_LEDGER.json')
    c.exact('role_counts',[len(visible[k]) for k in ('warm','pool','terminal')],[32,32,96])
    for role in ('warm','pool','terminal'):
        field='native_values' if role=='warm' else 'native_proxies'
        for row,meta in zip(visible[role],roles[role]):
            c.exact('visible_'+role+'_identity',row['identity'],meta);require(set(row)=={'identity',field},'unexpected target exposure')
    warm=[r['native_values'] for r in visible['warm']];pool=[r['native_proxies'] for r in visible['pool']];terminal=[r['native_proxies'] for r in visible['terminal']]
    transform=old.warm_transform(warm)
    for name,value in transform.items():c.array('warm_transform_'+name,saved['transform'][name],value)
    c.exact('warm_transform_rows',saved['transform']['warm_rows'],32)
    response={'center':transform['u_center_warm'][2],'sd':transform['u_sample_SD_warm'][2],'s0':transform['native_first_scale_s0'][2]}
    for name,value in response.items():c.number('response_'+name,saved['response_transform'][name],value)
    feature={'center':transform['proxy_native_center_warm'],'sd':transform['proxy_native_sample_SD_warm']}
    for name,value in feature.items():c.array('reference_feature_'+name,saved['native_reference_feature_transform'][name],value)
    wz=[bma.transform_row(transform,r,old.forward_yj) for r in warm]
    pz=[bma.transform_row(transform,r,old.forward_yj) for r in pool];tz=[bma.transform_row(transform,r,old.forward_yj) for r in terminal]
    c.array('known_pool_coordinates',saved['allpool_standardized_known_proxies'],pz)
    c.array('known_terminal_coordinates',saved['allterminal_standardized_known_proxies'],tz)
    expansion=bma.expansion_transform(wz);warm_ensemble,warm_evidence=bma.batch_ensemble(wz,expansion)
    bma.ensemble_check(saved['warm_ensemble'],warm_ensemble,c,'warm')
    c.array('closed_warm_log_evidence',saved['warm_log_evidence_by_H_modeled_response'],warm_evidence)
    jac=warm_common_jacobian(warm,transform);c.number('common_native_jacobian',saved['common_response_native_logJacobian_sum'],jac)
    c.array('native_warm_log_evidence',saved['warm_log_evidence_by_H_native_response'],[v+jac for v in warm_evidence])
    require(len(saved['warm_evidence_history'])==32,'32warm evidence steps')
    previous_evidence=[0.]*3
    for i,row in enumerate(saved['warm_evidence_history']):
        post,evidence=bma.batch_ensemble(wz[:i+1],expansion)
        c.exact('warm_evidence_order',row['warm_row_order_zero_based'],i)
        c.array('warm_predictive_evidence_increment',row['preupdate_logpredictive'],[a-b for a,b in zip(evidence,previous_evidence)])
        c.array('warm_model_logweights',row['posterior_log_weights'],post['log_weights']);previous_evidence=evidence
    primary=read(folder/'TERMINAL_FORECASTS.json')['forecasts'];refs=read(folder/'REFERENCE_FORECASTS.json')['forecasts']
    laws={(r['component_id'],r['policy'],r['budget']):r for r in primary}
    reference_laws={(r['component_id'],r['policy'],r['budget'],r['method']):r for r in refs}
    inventory={(m['component_id'],p,b) for m in roles['terminal'] for p in POLICIES for b in BUDGETS}
    expected_ref={(m['component_id'],p,b,k) for m in roles['terminal'] for p in ('full_native','random') for b in BUDGETS for k in ('GP','RF','affine_NIG')}
    expected_ref.update((m['component_id'],'gp_variance',b,'GP') for m in roles['terminal'] for b in BUDGETS)
    expected_ref.update((m['component_id'],'full_native',b,'frozen_weights_BMA') for m in roles['terminal'] for b in BUDGETS)
    require(len(primary)==len(laws)==1728 and set(laws)==inventory,'all1728main laws')
    require(len(refs)==len(reference_laws)==2304 and set(reference_laws)==expected_ref,'all2304reference laws')
    checkpoints={(r['policy'],r['budget']):r for r in saved['checkpoint_states']};require(len(checkpoints)==18,'18 checkpoints')
    index=read(folder/'PREQUENTIAL_INDEX.json')['records'];preq={(r['policy'],r['step']):r for r in index}
    events={(r['policy'],r['policy_total_query_cost']-1):r for r in ledger['events']}
    require(len(index)==len(preq)==len(events)==len(ledger['events'])==48,'all48 distincttrue disclosures')
    sequences={};training={};prefix={};ensembles={};readout_diagnostics=[];GP_actions=[];physical_values={}
    random=np.random.default_rng(20261106).permutation(32).tolist()
    for policy in POLICIES:
        train=list(warm);selected=[]
        for step in range(9):
            train_z=[bma.transform_row(transform,r,old.forward_yj) for r in train]
            expected,evidence=bma.batch_ensemble(train_z,expansion)
            training[(policy,step)]=list(train);prefix[(policy,step)]=list(selected);ensembles[(policy,step)]=expected
            if step in BUDGETS:
                snap=checkpoints[(policy,step)];bma.ensemble_check(snap['ensemble'],expected,c,'checkpoint')
                c.exact('checkpoint_prefix',snap['selected_pool_indices'],selected)
                if step:c.exact('checkpoint_train_hash',snap['observed_native_training_sha256'],jsha(train))
                for meta,x in zip(roles['terminal'],tz):
                    row=laws[(meta['component_id'],policy,step)];c.exact('main_jid',row['jid'],meta['jid'])
                    predicted=bma.law(expected,x,response);bma.law_check(row['law'],predicted,c)
                    if policy in ('full_native','random'):
                        affine=reference_laws[(meta['component_id'],policy,step,'affine_NIG')]
                        wanted={**predicted['components'][0],'native_first_scale_s0':response['s0']}
                        models.t_check(affine['law'],wanted,c)
                    if policy=='full_native':
                        frozen=reference_laws[(meta['component_id'],policy,step,'frozen_weights_BMA')]
                        bma.law_check(frozen['law'],bma.law(expected,x,response,warm_ensemble['log_weights']),c)
            if step==8:break
            item=preq[(policy,step)]
            for f,k in (('forecast_file','forecast_sha256'),('outcome_file','outcome_sha256')):c.exact('preq_file_hash',sha(folder/item[f]),item[k])
            before=read(folder/item['forecast_file']);after=read(folder/item['outcome_file']);chosen=before['pool_index']
            available=[i for i in range(32) if i not in selected]
            require(chosen in available,'newlegal pooledcell only')
            c.exact('available_pool',before['available_pool_indices'],available);c.exact('index_pool',item['pool_index'],chosen)
            for row in (before,after):c.exact('query_identity',row['identity'],roles['pool'][chosen]);c.exact('query_policy',row['policy'],policy)
            c.exact('query_before_budget',before['budget_before_query'],step);c.exact('query_after_budget',after['budget_after_query'],step+1)
            c.exact('query_noY',before['native_query_target_visible'],False);c.exact('query_knownX',before['known_native_proxies'],pool[chosen])
            c.exact('query_plan',before['plan_sha256'],plan['inputs']['science_plan']['sha256'])
            c.exact('query_sources',before['source_sha256'],protocol['source_sha256'])
            bma.ensemble_check(before['actual_predictive_ensemble_before'],expected,c,'prequery')
            bma.law_check(before['law'],bma.law(expected,pz[chosen],response),c)
            acquisition=bma.openloop(warm_ensemble,expected,step) if policy=='warm_openloop' else expected
            bma.ensemble_check(before['acquisition_ensemble'],acquisition,c,'acquisition')
            c.exact('openloop_tag',before['warmopenloop_acquisition_surrogate_not_actual_posterior'],policy=='warm_openloop')
            diagnostics=acquisition_check(before,policy,step,available,c)
            if diagnostics:readout_diagnostics.append(diagnostics)
            if policy=='random':
                c.exact('random_permutation',before['acquisition_diagnostics']['fixed_permutation'],random)
                c.exact('random_scores',before['acquisition_values'],[float(32-random.index(i)) for i in available])
            if policy=='gp_variance':GP_actions.append((before,available))
            event=events[(policy,step)];c.exact('event_matches_update',after['true_Y_reveal'],event)
            c.exact('onecell_fee',event['incremental_cost'],1);c.exact('fee_count',event['policy_total_query_cost'],step+1)
            c.exact('event_component_identity',event['identity'],roles['pool'][chosen]);c.exact('event_pool',event['pool_index'],chosen)
            c.exact('event_true_source',event['true_origin_raw_sha256'],'3d0fc1315b74cf84071f16aa25bb93434dafbc83867beb75872064ac245bd747')
            c.exact('not_synthetic',event['synthetic_label'],False);c.exact('event_prefreeze_hash',event['prequery_file_sha256'],item['forecast_sha256'])
            require(datetime.fromisoformat(before['actual_utc'])<=datetime.fromisoformat(event['actual_utc'])<=datetime.fromisoformat(after['actual_utc']),'freeze/reveal/updatechronology')
            if chosen in physical_values:c.exact('samephysicalcell_samevalue_acrosspolicies',event['native_target'],physical_values[chosen])
            physical_values[chosen]=event['native_target'];native=pool[chosen]+[event['native_target']]
            c.exact('complete_native_observation',after['native_complete_row'],native)
            c.number('actual_modeled_response',after['modeled_zY'],bma.transform_row(transform,native,old.forward_yj)[2])
            train.append(native);selected.append(chosen)
            next_ensemble,next_evidence=bma.batch_ensemble([bma.transform_row(transform,r,old.forward_yj) for r in train],expansion)
            bma.ensemble_check(after['ensemble_after'],next_ensemble,c,'afterquery')
            update=after['old_predictive_evidence_then_NIG_updates']
            c.array('query_evidence_from_closedmarginal',update['preupdate_logpredictive'],[a-b for a,b in zip(next_evidence,evidence)])
            c.array('query_old_model_logweights',update['old_log_weights'],expected['log_weights'])
            c.array('query_new_model_logweights',update['new_log_weights'],next_ensemble['log_weights'])
            c.exact('old_density_before_NIG_update',update['evidence_before_NIG_update'],True);c.exact('probability_not_floored',update['probability_floor'],False)
        sequences[policy]=selected
    c.exact('common_setup352',ledger['common_setup_native_cells'],352);c.exact('validationYseparate96',ledger['terminal_Y_validation_native_cells_separate'],96)
    c.exact('perpolicy8cells',ledger['cost_per_policy'],{p:8 for p in POLICIES});c.exact('terminalYnotread',ledger['terminalY_model_access'],0)
    c.exact('actualphysicalpoolunion',ledger['union_pool_components_with_Y_extracted'],len(physical_values))
    fit_index=read(folder/'FIT_INDEX.json')['records'];require(len(fit_index)==18 and sum(x['method']=='GP' for x in fit_index)==13,'18fits13GP5RF')
    fit_laws={};fit_reports=[]
    for item in fit_index:
        c.exact('fit_payload_hash',sha(folder/item['file']),item['sha256']);record=read(folder/item['file']);fid=item['fit_id'];method=item['method']
        if fid in ('GP_warm_shared','RF_warm_shared'):policy,step='full_native',0
        elif fid.startswith('GP_policy_step'):policy,step='gp_variance',int(fid[-2:])
        elif fid=='GP_policy_final_budget8':policy,step='gp_variance',8
        else:policy,number=fid[len(method)+1:].rsplit('_budget',1);step=int(number)
        train=training[(policy,step)];query=pool+terminal if fid=='GP_warm_shared' or fid.startswith('GP_policy_step') else terminal
        c.exact('fit_training_hash',record['train_native_values_sha256'],jsha(train));c.exact('fit_sources',record['source_sha256'],protocol['source_sha256'])
        xfit=[[(r[j]-feature['center'][j])/feature['sd'][j] for j in (0,1)] for r in train]
        xquery=[[(r[j]-feature['center'][j])/feature['sd'][j] for j in (0,1)] for r in query]
        function=models.gp_check if method=='GP' else models.forest_check
        reconstructed,report=function(record['state'],xfit,xquery,[r[2] for r in train],c,trees)
        fit_laws[item['sha256']]=(reconstructed,32 if len(query)==128 else 0);fit_reports.append({'fit_id':fid,'method':method,**report})
    for row in refs:
        meta=next(r for r in roles['terminal'] if r['component_id']==row['component_id']);c.exact('reference_jid',row['jid'],meta['jid'])
        if row['method'] in ('GP','RF'):
            predicted,offset=fit_laws[row['fit_state_sha256']];position=next(i for i,r in enumerate(roles['terminal']) if r['component_id']==row['component_id'])
            models.baseline_law_check(row['law'],predicted[offset+position],c)
        else:c.exact('exact_reference_currentstate_hash',row['source_ensemble_sha256'],jsha(checkpoints[(row['policy'],row['budget'])]['ensemble']))
        if 'same_actual_train_queries' in row:c.exact('reference_same_actual_training',row['same_actual_train_queries'],prefix[(row['policy'],row['budget'])])
    for before,available in GP_actions:
        predictions,offset=fit_laws[before['source_GP_fit_sha256_if_policy']];require(offset==32,'GPpolicyallpoolpredictions')
        for value,i in zip(before['acquisition_values'],available):
            sd=predictions[i]['scale'];c.number('GP_variance_action_value',value,sd*sd,4e-6*abs(sd)+4e-12,4e-8)
    aliases=[]
    for i,left in enumerate(POLICIES):
        for right in POLICIES[i+1:]:
            aliases.append({'left':left,'right':right,'same_exact_sequence':sequences[left]==sequences[right],
                            'same_final_set':set(sequences[left])==set(sequences[right])})
    return {'status':'PASS','primary_mixture_laws':1728,'reference_laws':2304,'total_terminal_laws':4032,
            'secondary_prequery_mixture_laws':48,'feedback_component_NIG_updates':144,'warm_component_uses':96,
            'baseline_saved_fit_states':18,'baseline_state_checks':fit_reports,'physical_poolY_union':len(physical_values),
            'actual_policy_sequences':sequences,'policy_alias_diagnostics':aliases,'readout_diagnostics':readout_diagnostics,
            'all96terminalcontexts_retained':True,'terminalY_reads':0,'SUT_imports_optimizer_or_model_refits':0,
            'materialfullVOI2Dreintegration':False,
            'scope':'independent dense closed statistics/evidence/laws; native risk field audited through frozen passedpuregate plus actualstate/coordinates/averages/refinement/argmax; not terminal effect evaluation'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--approval',required=True);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();plan=read(args.plan);approval=read(args.approval)
    require(approval.get('approved') is True and approval['scope']=='INDEPENDENT_FRESH160_SAVED_STATES_ONCE'
            and approval['plan_sha256']==sha(args.plan) and approval['script_sha256']==sha(__file__),'exact rootindependent approval')
    models=load(plan['helpers']['baseline_state'],'_independent_baseline_state_math');c=models.Checks()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False);start=datetime.now(timezone.utc).isoformat()
    try:result=audit(plan,c,models)
    except Exception as problem:result={'status':'FAIL','type':type(problem).__name__,'message':str(problem),'traceback':traceback.format_exc()}
    result.update({'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
                   'plan_sha256':sha(args.plan),'approval_sha256':sha(args.approval),'comparison_counts':c.counts,'max_absolute_errors':c.errors})
    f=out/'INDEPENDENT_FRESH160_STATES_AUDIT.json'
    with f.open('x',encoding='utf-8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':result['status'],'file':str(f),'sha256':sha(f)}))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
