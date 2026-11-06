"""Audit only saved growth128 arithmetic and freezes, without fits or raw data.

All transforms, conjugate sufficient statistics, absolute evidence, OOF CDFs,
composition selection, 672 forecasts, and six saved GP/RF states are rebuilt
from previously revealed values. Root must pin the completed run before use.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy import special

METHODS = ('old3_BMA','grown4_BMA','H_new','H_anchor','affine','GP10','RF10')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--approval',required=True)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args();plan=read(args.plan)
    if args.approval != plan['root_approval_token'] or sha(__file__) != plan['audit_source_sha256']:
        raise ValueError('root/source binding mismatch')
    for pin in plan['pins'].values():
        if sha(pin['file']) != pin['sha256']:raise ValueError('changed pin '+pin['file'])
    output=args.output_dir;output.mkdir(parents=True,exist_ok=False)
    write(output/'EXECUTION_START.json',{'actual_utc':datetime.now(timezone.utc).isoformat(),
         'plan_sha256':sha(args.plan),'new_fits':0,'new_raw_reads':0,'scope':plan['scope']})
    c=None
    try:
        pins=plan['pins'];folder=Path(plan['forecast_dir'])
        ref=load('_growth_independent_reference',pins['growth_reference']['file'])
        scalar=load('_independent_scalar_tree',pins['scalar_tree']['file'])
        base=load('_independent_saved_baseline',pins['baseline_reference']['file'])
        transform=load('_independent_scalar_transform',pins['transform_reference']['file'])
        c=base.Checks()
        freeze=read(pins['all_batch_freeze']['file'])
        c.exact('complete_run_status',freeze['status'],'ALL672_PREQUENTIAL_LAWS_AND96_TRUE_OUTCOMES_COMPLETE')
        for name,pin in freeze['files_sha256'].items():
            path=(folder/name).resolve();path.relative_to(folder.resolve())
            c.exact('closed_payload_sha',sha(path),pin)
        science=read(pins['science_plan']['file'])
        c.exact('science_plan_sha',freeze['plan_sha256'],pins['science_plan']['sha256'])
        c.exact('seven_science_sources',freeze['source_sha256'],science['source_sha256'])
        history=read(folder/'HISTORY40_ONLY.json')
        oldvisible=read(science['inputs']['old_visible']['file'])
        oldstates=read(science['inputs']['old_states']['file'])
        oldindex=read(science['inputs']['old_prequential_index']['file'])
        oldfreeze=read(science['inputs']['old_forecast_freeze']['file'])
        oldfolder=Path(science['inputs']['old_forecast_freeze']['file']).parent
        expected_history=list(oldvisible['warm'])
        oldqueries=sorted((item for item in oldindex['records'] if item['policy']=='full_native'),key=lambda r:r['step'])
        c.exact('old_history_selected_steps',[item['step'] for item in oldqueries],list(range(8)))
        for item in oldqueries:
            for field,hashfield in (('forecast_file','forecast_sha256'),('outcome_file','outcome_sha256')):
                p=(oldfolder/item[field]).resolve();p.relative_to(oldfolder.resolve())
                c.exact('old_selected_file_sha',sha(p),item[hashfield])
                c.exact('old_selected_freeze_sha',oldfreeze['files'][item[field]],item[hashfield])
            before=read(oldfolder/item['forecast_file']);after=read(oldfolder/item['outcome_file'])
            reveal=after['true_Y_reveal'];c.exact('old_query_identity',before['identity'],reveal['identity'])
            expected_history.append({'identity':before['identity'],
                'native_values':before['known_native_proxies']+[reveal['native_target']]})
        c.exact('only_authorized_old40',history['rows'],expected_history)
        identities=[row['identity']['component_id'] for row in history['rows']]
        c.exact('unique_history40',len(set(identities)),40)
        native=[row['native_values'] for row in history['rows']]
        descriptors=[ref.composition(row['identity']) for row in history['rows']]
        coordinate=transform.warm_transform(native[:32])
        saved_coordinate=history['old_snapshot']['transform']
        for key in ('native_first_scale_s0','u_center_warm','u_sample_SD_warm'):
            c.array('frozen_old32_'+key,saved_coordinate[key],coordinate[key])
        expanded=ref.expansion(ref.transform(native[:32],coordinate,transform.forward_yj))
        for key in ('center','sd'):
            c.array('old32_expansion_'+key,history['old_snapshot']['warm_ensemble']['expansion_transform'][key],expanded[key])
        ds=ref.descriptor_scale(descriptors);dz=ref.standardized(descriptors,ds)
        z=ref.transform(native,coordinate,transform.forward_yj)
        anchor=read(folder/'ANCHOR_PREPARED.json');anchorfreeze=read(folder/'ANCHOR_FREEZE.json')
        c.exact('anchor_freeze_sha',anchorfreeze['anchor_sha256'],sha(folder/'ANCHOR_PREPARED.json'))
        c.exact('anchor_freeze_beforeOOF_declared',anchorfreeze['before_any_OOF_fit'],True)
        c.exact('anchor_component_ids',anchor['component_ids'],identities)
        c.array('anchor_native_proxies',anchor['history_proxy_native40'],[row[:2] for row in native])
        c.array('composition_descriptors',anchor['history_descriptors40'],descriptors)
        for key in ('center','sd','raw_sample_SD'):
            c.array('descriptor_transform_'+key,anchor['descriptor_transform'][key],ds[key])
        c.exact('descriptor_constant_flags',anchor['descriptor_transform']['constant_or_numerically_constant'],ds['constant_or_numerically_constant'])
        phi=np.array([ref.design('fullquad',row,expanded) for row in z])
        u,s,vh=np.linalg.svd(phi,full_matrices=False);rank=int(sum(s>1e-10*s[0]));q=u[:,:rank]
        residual=np.array(dz)-q@(q.T@np.array(dz))
        c.array('anchor_Phi',anchor['projection_fullquad_Phi'],phi.tolist())
        c.exact('anchor_projection_rank',anchor['projection_rank'],rank)
        savedq=np.array(anchor['projection_Q'])
        c.array('anchor_projection_matrix',(savedq@savedq.T).tolist(),(q@q.T).tolist())
        c.array('residualized_descriptors',anchor['residualized_standardized_descriptors'],residual.tolist())
        eligibility=[]
        for j in range(8):
            norm=float(np.linalg.norm(np.array(dz)[:,j]));rn=float(np.linalg.norm(residual[:,j]))
            eligible=not ds['constant_or_numerically_constant'][j] and rn>1e-10*norm
            eligibility.append(eligible);c.exact('descriptor_eligible',anchor['eligibility'][j]['eligible'],bool(eligible))
        anchor_index=next((j for j,ok in enumerate(eligibility) if ok),None)
        c.exact('metadata_only_anchor_index',anchor['anchor_index'],anchor_index)
        order=sorted(range(40),key=lambda i:(hashlib.sha256(('relation_growth_oof5_v1|20261106|'+identities[i]).encode()).hexdigest(),identities[i]))
        folds=[None]*40
        for j,i in enumerate(order):folds[i]=j//8
        c.exact('fixed_OOF_folds',anchor['OOF_fold_by_original_row'],folds)
        proposal=read(folder/'AUTOMATIC_PROPOSAL.json');residual_y=[None]*40
        c.exact('OOF_uses_frozen_anchor',proposal['prepared'],anchor)
        for item in proposal['OOF_fold_records']:
            fold=item['fold'];train=[i for i in range(40) if folds[i]!=fold];held=[i for i in range(40) if folds[i]==fold]
            c.exact('OOF_train32',item['train_original_indices'],train);c.exact('OOF_held8',item['heldout_original_indices'],held)
            trainrows=[native[i] for i in train];fc=transform.warm_transform(trainrows)
            fz=ref.transform(trainrows,fc,transform.forward_yj);fe=ref.expansion(fz)
            for key in ('native_first_scale_s0','u_center_warm','u_sample_SD_warm'):
                c.array('OOF_fold_only_'+key,item['transform_fit_only_on_train32'][key],fc[key])
            for key in ('center','sd'):c.array('OOF_expansion_'+key,item['expansion_fit_only_on_train32'][key],fe[key])
            states={};logs=[]
            for kind in ref.IDS:
                state,ev=ref.dense([ref.design(kind,row,fe) for row in fz],[row[2] for row in fz])
                ref.state_check(item['trained_components'][kind]['state'],state,c,'OOF_'+kind)
                c.number('OOF_absolute_evidence',item['absolute_conditional_log_evidence'][kind],ev)
                states[kind]=state;logs.append(ev)
            logweights=ref.normalized(logs);c.array('OOF_logweights',item['model_log_weights'],logweights)
            for index in held:
                row=next(row for row in proposal['OOF_rows'] if row['original_row']==index)
                c.exact('OOF_held_identity',row['component_id'],identities[index])
                c.exact('OOF_held_fold',row['fold'],fold)
                c.number('OOF_held_original_target',row['actual_old_native_target'],native[index][2],0,0)
                hz=ref.transform([native[index]],fc,transform.forward_yj)[0]
                components=[]
                for kind in ref.IDS:
                    law=ref.predictive(states[kind],ref.design(kind,hz,fe),fc)
                    components.append({'id':kind,**law})
                wanted={'coordinate':'u=YJ_half(native/s0)','family':'mixture_student_t','lambda':.5,
                        'native_first_scale_s0':fc['native_first_scale_s0'][2],'components':components,'log_weights':logweights}
                ref.law_check(row['native_law'],wanted,c,'OOF_law')
                uvalue=transform.forward_yj(native[index][2]/wanted['native_first_scale_s0'])
                pit=math.fsum(math.exp(w)*float(special.stdtr(comp['df'],(uvalue-comp['location'])/comp['scale']))
                              for w,comp in zip(logweights,components))
                c.number('OOF_native_CDF',row['native_OOF_CDF'],pit)
                c.number('OOF_direction_residual',row['bounded_direction_residual'],pit-.5)
                residual_y[index]=pit-.5
        c.array('OOF_all40_residuals',proposal['raw_residual'],residual_y)
        r=np.array(residual_y);rr=r-q@(q.T@r);rnorm=float(np.linalg.norm(rr))
        low=rnorm<=1e-12*max(1.,float(np.linalg.norm(r)))
        c.array('residualized_direction',proposal['residualized_r'],rr.tolist())
        scores=[]
        for j,eligible in enumerate(eligibility):
            score=(0. if low else abs(float(residual[:,j]@rr)/(float(np.linalg.norm(residual[:,j]))*rnorm))) if eligible else None
            scores.append(score)
            if eligible:c.number('descriptor_proposal_score',proposal['scores_all8'][j]['absolute_score'],score)
        maximum=max(score for score,ok in zip(scores,eligibility) if ok)
        selected=next(j for j,score in enumerate(scores) if eligibility[j] and maximum-score<=1e-12)
        c.exact('one_fixed_selected_descriptor',proposal['selected_index'],selected)
        alias=selected==anchor_index
        initial=read(folder/'INITIAL_GROWTH_STATE.json')
        c.exact('new_anchor_alias',initial['new_anchor_exact_alias'],alias)
        c.exact('anchor_id',initial['anchor_id'],'new' if alias else 'anchor')
        component_kinds={name:(part['kind'],part['descriptor_index']) for name,part in initial['components'].items()}
        c.exact('unique_component_count',len(component_kinds),4 if alias else 5)
        c.exact('selected_component_basis',list(component_kinds['new']),['descriptor',selected])
        if not alias:c.exact('anchor_component_basis',list(component_kinds['anchor']),['descriptor',anchor_index])

        def batch_stats(rows,desc):
            zz=ref.transform(rows,coordinate,transform.forward_yj);dd=ref.standardized(desc,ds)
            stats={};ev={}
            for name,(kind,index) in component_kinds.items():
                designs=[ref.design(kind,row,expanded,d,index) for row,d in zip(zz,dd)]
                stats[name],ev[name]=ref.dense(designs,[row[2] for row in zz])
            return stats,ev

        def suite_check(suite,rows,desc,label):
            stats,ev=batch_stats(rows,desc)
            for key in ('native_first_scale_s0','u_center_warm','u_sample_SD_warm'):
                c.array(label+'_fixed_coordinate_'+key,suite['transform'][key],coordinate[key])
            for key in ('center','sd'):
                c.array(label+'_fixed_expansion_'+key,suite['expansion_transform'][key],expanded[key])
                c.array(label+'_fixed_descriptor_'+key,suite['descriptor_transform'][key],ds[key])
            for name,state in stats.items():ref.state_check(suite['components'][name]['state'],state,c,label+'_'+name)
            for library,ids in (('old3_BMA',list(ref.IDS)),('grown4_BMA',list(ref.IDS)+['new'])):
                c.exact(label+'_library_ids',suite['libraries'][library]['component_ids'],ids)
                c.array(label+'_library_full_evidence',suite['libraries'][library]['log_weights'],ref.normalized([ev[name] for name in ids]))
            c.exact(label+'_history_rows',suite['history_rows_seen'],len(rows))
            c.exact(label+'_fresh_rows',suite['fresh_rows_seen'],len(rows)-40)
            return stats,ev

        original_stats,original_ev=suite_check(initial,native,descriptors,'initial40')
        for name,value in original_ev.items():c.number('history40_absolute_evidence',initial['absolute_history40_log_evidence'][name],value)
        for old in history['old_final3']['models']:
            ref.state_check(initial['components'][old['id']]['state'],old['state'],c,'old40_preserved')
        visible=read(folder/'VISIBLE_INPUTS.json');roles=read(pins['roles']['file'])
        for role,rows in visible.items():
            c.exact('fresh_role_identities',[r['identity'] for r in rows],roles['representatives'][role])
        if set(identities)&set(roles['new_development_consumed_component_ids']):raise AssertionError('new128 overlap history40')
        newfit=visible['fit'];fitnative=[row['native_values'] for row in newfit]
        fitdesc=[ref.composition(row['identity']) for row in newfit]
        allnative=list(native);alldesc=list(descriptors)

        def increments_check(saved,newrows,newdesc,label):
            oldstats,oldev=batch_stats(allnative,alldesc)
            for item,row,d in zip(saved['records'],newrows,newdesc):
                allnative.append(row);alldesc.append(d);stats,ev=batch_stats(allnative,alldesc)
                for name in component_kinds:c.number(label+'_predictive_evidence',item['old_predictive_loglikelihood'][name],ev[name]-oldev[name])
                for library,ids in (('old3_BMA',list(ref.IDS)),('grown4_BMA',list(ref.IDS)+['new'])):
                    c.array(label+'_postquery_logweights',item['library_log_weights'][library],ref.normalized([ev[name] for name in ids]))
                oldev=ev
            c.exact(label+'_updated_rows',saved['rows_updated_after_whole_batch_reveal'],len(newrows))
            c.exact(label+'_unique_updates',saved['unique_components_updated_each_row'],len(component_kinds))

        increments_check(read(folder/'NEW32_FIT_UPDATE.json'),fitnative,fitdesc,'initial32')
        baseline=read(folder/'BASELINE_TRANSFORM.json')
        proxy_center=coordinate['proxy_native_center_warm'];proxy_sd=coordinate['proxy_native_sample_SD_warm']
        expected_baseline={'center':proxy_center+ds['center'],'sd':proxy_sd+ds['sd']}
        c.array('fixed10D_center',baseline['center'],expected_baseline['center'])
        c.array('fixed10D_SD',baseline['sd'],expected_baseline['sd'])
        roles_order=plan['batch_roles'];all_counts=0;baselines=[]
        for index,role in enumerate(roles_order):
            bdir=folder/role;before=read(bdir/'STATE_BEFORE.json');suite=before['suite']
            c.exact('batch_training_native_rows',before['training_native_rows'],allnative)
            c.array('batch_training_descriptors',before['training_descriptors'],alldesc)
            c.exact('batch_training_count',before['training_components'],72+32*index)
            states,ev=suite_check(suite,allnative,alldesc,'batch'+str(index))
            rows=visible[role];queries=[row['native_proxies'] for row in rows]
            desc=[ref.composition(row['identity']) for row in rows]
            train10=[row[:2]+d for row,d in zip(allnative,alldesc)]
            query10=[row+d for row,d in zip(queries,desc)]
            xfit=ref.standardized(train10,expected_baseline);xquery=ref.standardized(query10,expected_baseline)
            yfit=[row[2] for row in allnative]
            gp,gpdiag=ref.gp10_check(read(bdir/'GP10_STATE.json'),xfit,xquery,yfit,c,scalar)
            rf,rfdiag=base.forest_check(read(bdir/'RF10_STATE.json'),xfit,xquery,yfit,c,scalar)
            baselines.append({'role':role,'GP':gpdiag,'RF':rfdiag})
            records=read(bdir/'FORECASTS.json')['records']
            c.exact('batch_all224',len(records),224)
            expected_pairs={(method,row['identity']['component_id']) for method in METHODS for row in rows}
            c.exact('batch_method_group_crossproduct',set((r['method'],r['identity']['component_id']) for r in records),expected_pairs)
            for j,row in enumerate(rows):
                context=ref.transform([queries[j]],coordinate,transform.forward_yj)[0]
                standardized_d=ref.standardized([desc[j]],ds)[0]
                for method in METHODS:
                    actual=next(r for r in records if r['method']==method and r['identity']['component_id']==row['identity']['component_id'])
                    c.exact('forecast_same_identity',actual['identity'],row['identity'])
                    c.array('forecast_same_proxies',actual['known_native_proxies'],queries[j])
                    c.array('forecast_descriptor',actual['composition_descriptors'],desc[j])
                    if method in ('GP10','RF10'):
                        base.baseline_law_check(actual['law'],gp[j] if method=='GP10' else rf[j],c)
                    else:
                        ids=(list(ref.IDS) if method=='old3_BMA' else list(ref.IDS)+['new'] if method=='grown4_BMA'
                             else ['new'] if method=='H_new' else [initial['anchor_id']] if method=='H_anchor' else ['affine'])
                        components=[]
                        for name in ids:
                            kind,di=component_kinds[name];phi=ref.design(kind,context,expanded,standardized_d,di)
                            components.append({'id':name,**ref.predictive(states[name],phi,coordinate)})
                        wanted={'coordinate':'u=YJ_half(native/s0)','lambda':.5,'native_first_scale_s0':coordinate['native_first_scale_s0'][2]}
                        if len(ids)==1:wanted.update({'family':'student_t',**{k:components[0][k] for k in ('location','scale','df')}})
                        else:wanted.update({'family':'mixture_student_t','components':components,'log_weights':ref.normalized([ev[name] for name in ids])})
                        ref.law_check(actual['law'],wanted,c,'forecast_'+method)
                    all_counts+=1
            bf=read(bdir/'FORECAST_FREEZE.json');truth=read(bdir/'TRUE_BATCH_REVEAL.json')
            c.exact('batch_frozen_status',bf['status'],'ALL224_BATCH_LAWS_FROZEN_BEFORE_32Y')
            c.exact('truth_bound_to_freeze',truth['forecast_freeze_sha256'],sha(bdir/'FORECAST_FREEZE.json'))
            for name,pin in bf['files_sha256'].items():c.exact('batch_payload_sha',sha(bdir/name),pin)
            if datetime.fromisoformat(bf['actual_utc'])>datetime.fromisoformat(truth['actual_utc']):raise AssertionError('truth precedes forecast freeze')
            c.exact('all_batch_truth_identities',[r['identity'] for r in truth['rows']],[r['identity'] for r in rows])
            disclosed=[query+[r['native_target']] for query,r in zip(queries,truth['rows'])]
            after=read(bdir/'UPDATE_AFTER_REVEAL.json')
            increments_check(after['evidence'],disclosed,desc,'batch'+str(index))
            suite_check(after['suite'],allnative,alldesc,'after_batch'+str(index))
        suite_check(read(folder/'FINAL_STATE.json'),allnative,alldesc,'final168')
        c.exact('all_forecast_coordinates',all_counts,672)
        c.exact('final_rows',len(allnative),168)
        fits=read(folder/'FIT_INDEX.json');c.exact('GP_actual_fits',fits['GP_fits'],3);c.exact('RF_actual_fits',fits['RF_fits'],3)
        ledger=read(folder/'DISCLOSURE_LEDGER.json');c.exact('new_native_cell_cost',ledger['total_new_native_cells'],384)
        result={'status':'PASS','actual_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(args.plan),
                'scope':plan['scope'],'checks':c.counts,'maximum_absolute_errors':c.errors,
                'saved_baseline_diagnostics':baselines,'all672_forecasts_audited':all_counts,
                'saved_OOF_component_states':15,'saved_OOF_CDFs':40,'all_unique_component_updates_audited':128*len(component_kinds),
                'selected_descriptor':ref.DESCRIPTORS[selected],'metadata_anchor':ref.DESCRIPTORS[anchor_index],
                'new_anchor_exact_alias':alias,'old40_absolute_evidence_audited':original_ev,
                'new_fits':0,'new_raw_reads':0,'labels_used':'only already disclosed saved reports; no new target read',
                'conjugate_batch_statistics_are_audit_not_additional_model_fit':True,
                'not_proof_of_postselection_valid_model_probabilities':True}
    except Exception as exc:
        result={'status':'FAIL','actual_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':sha(args.plan),
                'error_type':type(exc).__name__,'error':str(exc),'checks_completed':{} if c is None else c.counts,
                'errors_before_stop':{} if c is None else c.errors,'new_fits':0,'new_raw_reads':0}
    write(output/'INDEPENDENT_GROWTH_STATES_AUDIT.json',result)
    print(json.dumps({'status':result['status'],'result_sha256':sha(output/'INDEPENDENT_GROWTH_STATES_AUDIT.json')},ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
