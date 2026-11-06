"""One fixed old40 OOF composition proposal; every fit is explicit and caller-authorized.

prepare_anchor uses no Y. Root must freeze its output before calling propose.
Residualization is ranking-only; the admitted descriptor remains its raw40-scaled
physical composition quantity. No future row or target is available to this module.
"""
import copy
import hashlib
import math
import numpy as np
import growth_models as models
import probability

HASH_PREFIX='relation_growth_oof5_v1|20261106|'
SVD_RCOND=1e-10
RESIDUAL_REL_TOL=1e-10
R_ZERO_TOL=1e-12
TIE_ABS_TOL=1e-12


def prepare_anchor(component_ids,history_proxy_native40,descriptors40,old_snapshot):
    if len(component_ids)!=40 or len(set(component_ids))!=40 or any(not isinstance(x,str) or not x for x in component_ids):raise ValueError('40unique component IDs required')
    proxy=np.asarray(history_proxy_native40,dtype=float)
    if proxy.shape!=(40,2):raise ValueError('40known dual proxy rows required')
    dt=models.descriptor_transform(descriptors40);d=models.standard_descriptors(dt,descriptors40)
    x=models.proxies_z(old_snapshot['transform'],proxy);expansion=old_snapshot['warm_ensemble']['expansion_transform']
    phi=np.vstack([models.basis('fullquad',row,expansion) for row in x])
    u,s,vh=np.linalg.svd(phi,full_matrices=False);cut=SVD_RCOND*float(s[0]);rank=int((s>cut).sum());q=u[:,:rank]
    residual=d-q@(q.T@d);eligibility=[]
    for j,name in enumerate(models.DICTIONARY):
        norm=float(np.linalg.norm(d[:,j]));resid=float(np.linalg.norm(residual[:,j]));constant=dt['constant_or_numerically_constant'][j]
        reason='constant_or_numerically_constant' if constant else 'in_fullquad_column_span' if resid<=RESIDUAL_REL_TOL*norm else None
        eligibility.append({'index':j,'descriptor_id':name,'eligible':reason is None,'exclusion_reason':reason,
            'standardized_norm':norm,'residual_norm':resid,'relative_residual_norm':resid/norm if norm else 0.})
    candidates=[e['index'] for e in eligibility if e['eligible']];anchor=candidates[0] if candidates else None
    ordered=sorted(range(40),key=lambda i:(hashlib.sha256((HASH_PREFIX+component_ids[i]).encode()).hexdigest(),component_ids[i]))
    folds=[None]*40
    for rank_index,i in enumerate(ordered):folds[i]=rank_index//8
    return {'status':'METADATA_PROXY_ONLY_ANCHOR_PREPARED_MUST_FREEZE_BEFORE_OOF','component_ids':list(component_ids),
        'anchor_index':anchor,'anchor_id':None if anchor is None else models.DICTIONARY[anchor],
        'history_proxy_native40':proxy.tolist(),'history_descriptors40':np.asarray(descriptors40,dtype=float).tolist(),
        'descriptor_transform':dt,'eligibility':eligibility,'projection_fullquad_Phi':phi.tolist(),
        'projection_Q':q.tolist(),'projection_singular_values':s.tolist(),'projection_rank':rank,'projection_cutoff':cut,
        'residualized_standardized_descriptors':residual.tolist(),'OOF_fold_by_original_row':folds,
        'OOF_hash_prefix':HASH_PREFIX,'SVD_rcond':SVD_RCOND,'residual_relative_tol':RESIDUAL_REL_TOL,
        'Y_values_consumed':0,'anchor_selection_uses_response':False}


def propose(history_native40,prepared):
    native=np.asarray(history_native40,dtype=float)
    if native.shape!=(40,3) or not np.isfinite(native).all():raise ValueError('40complete authorized historical rows required')
    if not np.array_equal(native[:,:2],np.asarray(prepared['history_proxy_native40'])):raise ValueError('history row order/proxies changed after anchor freeze')
    if prepared['anchor_index'] is None:return {'status':'NO_IDENTIFIABLE_CANDIDATE','selected_index':None,'prepared':copy.deepcopy(prepared),'OOF_fits':0}
    fold_ids=prepared['OOF_fold_by_original_row'];records=[];r=np.empty(40);folds=[]
    for fold in range(5):
        train=[i for i in range(40) if fold_ids[i]!=fold];held=[i for i in range(40) if fold_ids[i]==fold]
        if len(train)!=32 or len(held)!=8:raise ValueError('fixed5fold32/8assignment required')
        trained=models.fit_old_three(native[train])
        folds.append({'fold':fold,'train_original_indices':train,'heldout_original_indices':held,
            'transform_fit_only_on_train32':trained['transform'],'expansion_fit_only_on_train32':trained['expansion_transform'],
            'trained_components':trained['components'],'absolute_conditional_log_evidence':trained['absolute_log_evidence'],
            'model_log_weights':trained['log_weights']})
        for i in held:
            x=models.proxies_z(trained['transform'],native[i,:2])
            predicted=models.law_from_components(trained['components'],trained['component_ids'],trained['log_weights'],
                trained['transform'],trained['expansion_transform'],x)
            value=probability.cdf_native(predicted,float(native[i,2]));r[i]=value-.5
            records.append({'original_row':i,'component_id':prepared['component_ids'][i],'fold':fold,'native_law':predicted,
                'actual_old_native_target':float(native[i,2]),'native_OOF_CDF':value,'bounded_direction_residual':float(r[i])})
    q=np.asarray(prepared['projection_Q']);rres=r-q@(q.T@r);rnorm=float(np.linalg.norm(rres));rawnorm=float(np.linalg.norm(r))
    low=rnorm<=R_ZERO_TOL*max(1.,rawnorm);g=np.asarray(prepared['residualized_standardized_descriptors']);scores=[]
    for item in prepared['eligibility']:
        j=item['index'];entry=dict(item)
        if item['eligible']:
            signed=0. if low else float(g[:,j]@rres)/(float(np.linalg.norm(g[:,j]))*rnorm)
            entry.update(signed_standardized_inner_product=signed,absolute_score=abs(signed))
        else:entry.update(signed_standardized_inner_product=None,absolute_score=None)
        scores.append(entry)
    maximum=max(x['absolute_score'] for x in scores if x['eligible'])
    selected=next(x['index'] for x in scores if x['eligible'] and maximum-x['absolute_score']<=TIE_ABS_TOL)
    return {'status':'ONE_CANDIDATE_SELECTED_EXPLORATORY_LOW_RESIDUAL' if low else 'ONE_CANDIDATE_SELECTED_FROM_OLD40',
        'selected_index':selected,'selected_id':models.DICTIONARY[selected],'new_anchor_exact_alias':selected==prepared['anchor_index'],
        'prepared':copy.deepcopy(prepared),'scores_all8':scores,'OOF_fold_records':folds,
        'OOF_rows':sorted(records,key=lambda x:x['original_row']),'raw_residual':r.tolist(),'residualized_r':rres.tolist(),
        'residualized_r_norm':rnorm,'r_low_signal':low,'tie_absolute_tol':TIE_ABS_TOL,
        'conditional_development_proposal_not_independent_confirmation':True,'OOF_folds':5,'component_NIG_initializations':15,
        'new_or_validation_Y_used':0,'anchor_was_response_independent':True}
