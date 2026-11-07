"""Independent saved-only phased audit. No SUT imports, fits or raw properties.

Phases are explicit; metadata can read only saved identity metadata. Model,
query and final phases are added before their saved artifacts become available.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re


def require(ok,label):
    if not ok:raise AssertionError(label)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def read(p):
    def unique(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,'duplicate JSON key');out[k]=v
        return out
    return json.loads(Path(p).read_text('utf-8-sig'),object_pairs_hook=unique)


def pinned(item):
    require(sha(item['file'])==item['sha256'],'input hash changed')
    return read(item['file'])


def metadata(args):
    plan=read(args.plan)
    inputs=plan['inputs']
    old=pinned(inputs['remaining_identity_roles'])
    eligible=pinned(inputs['eligible_metadata'])['datasets']['jarvis_dft_3d']
    j=pinned(inputs['all_JARVIS_identity'])
    d=pinned(inputs['all_dielectric_identity'])
    role=read(args.stage/'metadata_once'/'ROLE_SELECTION.json')
    jr,dr=j['rows'],d['rows']
    require(len(jr)==25923 and len(dr)==1056 and len(eligible)==11020,'all source frames')
    require(j['property_fields_emitted']==[] and d['target_values_emitted_to_metadata_or_models']==0,'identity-only frames')
    rows=jr+dr
    parent=list(range(len(rows)))
    def find(i):
        root=i
        while parent[root]!=root:root=parent[root]
        while parent[i]!=i:
            nxt=parent[i];parent[i]=root;i=nxt
        return root
    owners={}
    for i,r in enumerate(rows):
        for field in ('normalized_composition','jid','mpid'):
            token=r.get(field)
            if token is None:continue
            if field=='jid':require(re.fullmatch(r'JVASP-[0-9]+',token),'valid normalized JID')
            if field=='mpid':require(re.fullmatch(r'(?:mp|mvc)-[0-9]+',token),'valid normalized MPID')
            key=(field,token)
            if key in owners:
                a,b=find(i),find(owners[key])
                if a!=b:parent[a]=b
            else:owners[key]=i
    groups=defaultdict(list)
    for r in eligible:
        rawrow=r['raw_row_zero_based'];orig=jr[rawrow]
        require(orig['artifact_native_index']==r['artifact_native_index'],'native source index')
        counts={k:Fraction(str(v)) for k,v in r['composition_amounts'].items()}
        require(counts and all(v>0 for v in counts.values()),'positive original composition')
        total=sum(counts.values(),Fraction())
        signature='|'.join(k+':'+str(counts[k]/total) for k in sorted(counts))
        require(signature==orig['normalized_composition']==r['normalized_composition'],'literal composition identity')
        require(orig['jid_original']==r['jid'] and orig['mpid_original']==r['mpid'],'literal original aliases')
        groups[r['component_id']].append(r)
    roots={k:{find(r['raw_row_zero_based']) for r in v} for k,v in groups.items()}
    require(all(len(v)==1 for v in roots.values()),'whole component cannot split')
    roots={k:next(iter(v)) for k,v in roots.items()}
    require(len(groups)==8842 and len(set(roots.values()))==8842,'eligible unique original groups')
    d_roots={find(len(jr)+i) for i in range(len(dr))}
    require(len(d_roots)==964,'all dielectric964 groups excluded')
    old_current={r['component_id'] for r in old['selected_groups']}
    consumed=set(old['all_prior_consumed_component_ids'])|old_current
    require(len(consumed)==1920,'all previously consumed whole groups')
    blocked=d_roots|{roots[k] for k in consumed}
    available={k for k,r in roots.items() if r not in blocked}
    require(available==set(old['remaining_unselected_component_ids_in_registered_hash_order']) and len(available)==6279,'true unused6279 closure')
    confirm=sorted(available,key=lambda k:(hashlib.sha256(('query_code_confirm_v1|'+k).encode()).hexdigest(),k))[:1024]
    remaining=available-set(confirm)
    ordered=sorted(remaining,key=lambda k:(hashlib.sha256(('query_risk_code_v1|'+k).encode()).hexdigest(),k))
    require(role['confirm_component_ids']==confirm and role['trial_component_ids']==ordered[:512]
            and role['future_development_component_ids']==ordered[512:],'deterministic two-stage untouched role order')
    require(set(role['all_prior_consumed_component_ids'])==consumed,'cumulative consumed identity ledger')
    closure=defaultdict(list)
    for i,r in enumerate(rows):
        closure[find(i)].append({'source':'jarvis_dft_3d' if i<len(jr) else 'dielectric_constant',
             'raw_row_zero_based':i if i<len(jr) else i-len(jr),'artifact_native_index':r['artifact_native_index']})
    checked=0
    for expected,saved,name in ((confirm,role['confirm_groups'],'FINAL_CONFIRM_NOT_FOR_DEVELOPMENT'),
                                 (ordered[:512],role['selected_groups'],'TRIAL512_DEVELOPMENT')):
        require(expected==[r['component_id'] for r in saved],'saved role row order')
        for key,r in zip(expected,saved):
            siblings=sorted(groups[key],key=lambda a:a['raw_row_zero_based'])
            require(r['role']==name and r['representative']==siblings[0] and r['eligible_siblings']==siblings,'minimum original representative and all eligible siblings')
            members=closure[roots[key]]
            require(all(m['source']=='jarvis_dft_3d' for m in members),'no excluded dielectric aliases')
            require(r['all_original_source_siblings_reserved']==members,'all original/ineligible siblings reserved')
            require(roots[key] not in blocked,'no priorconsumed or dielectric root')
            checked+=1
    require(len(set(confirm)|set(ordered[:512])|set(ordered[512:]))==6279,'exhaustive nonoverlap')
    return {'status':'PASS_INDEPENDENT_METADATA_ONLY_ROLE_AUDIT','role_selection_sha256':sha(args.stage/'metadata_once'/'ROLE_SELECTION.json'),
       'all_source_rows_checked':len(rows),'eligible_rows_checked':len(eligible),'whole_groups_reserved_and_siblings_checked':checked,
       'prior_consumed_groups':1920,'confirm1024':1024,'development512':512,'future_development4743':4743,
       'fits':0,'new_property_cells_read':0,'SUT_imports':0,
       'composition_is_identity_metadata_not_unread_all_features_claim':True}

def verify_manifest(folder,receipt):
    for name,digest in receipt['files_sha256'].items():
        p=(folder/name).resolve();p.relative_to(folder.resolve())
        require(sha(p)==digest,'saved payload changed: '+name)


def fraction_json(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,dict):return {k:fraction_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [fraction_json(v) for v in value]
    return value


def exact_geo(law,y=None,g=None):
    from reference_math import geometry,scalar
    base=geometry(law,y,None)
    if g is None:return base
    g=scalar(g);multipliers=(1-g,g)
    Q=base['Q'];bounds=base['bounds']
    return {'Q':tuple(tuple(multipliers[j//3]*multipliers[k//3]*Q[j%3][k%3]
                           for k in range(6)) for j in range(6)),
            'c':None if y is None else tuple(m*c for m in multipliers for c in base['c']),
            'bounds':tuple((m*a,m*b) for m in multipliers for a,b in bounds),
            'knot_c':tuple(tuple(m*c for m in multipliers for c in row) for row in base['knot_c'])}


def initial(args):
    import gzip
    import numpy as np
    from reference_math import vec,matrix
    from audit_saved_gate_v2 import audit_fit,msum,vsum
    from audit_saved_models import audit_saved_model,compare_laws
    folder=args.stage/'initial_once';freeze=read(folder/'FROZEN.json')
    verify_manifest(folder,freeze)
    plan=read(args.plan)
    require(freeze['plan_sha256']==sha(args.plan),'initial immutable main contract')
    history=read(folder/'HISTORY808_RECOVERED.json')['rows']
    require(len(history)==len({r['component_id'] for r in history})==808,'actual808 original groups')
    role=pinned(plan['inputs']['new_roles'])
    ids=[r['component_id'] for r in history]
    require(not set(ids)&set(role['confirm_component_ids']+role['trial_component_ids']),'new roles do not enter training')
    oldrf=pinned(plan['inputs']['old_RF296'])
    require(ids[:296]==oldrf['training_component_ids'],'original296 training order')
    transform=oldrf['fixed_feature_transform']
    for i,r in enumerate(history[:296]):
        x=np.asarray([(v-a)/b for v,a,b in zip(r['X'],transform['center'],transform['sd'])],dtype=np.float32).astype(float).tolist()
        require(x==oldrf['X_train_standardized_then_float32'][i] and r['y']==oldrf['y_train_native'][i],'literal296 saved-training mapping')
    oldctx=pinned(plan['inputs']['current512_contexts'])['records']
    oldys={r['component_id']:r for r in pinned(plan['inputs']['current512_actual_targets'])['records']}
    require(ids[296:]==[r['component_id'] for r in oldctx],'consumed512 training order')
    for r,c in zip(history[296:],oldctx):
        require(r['identity']==c['identity'] and r['X']==c['native_proxies']+c['raw_descriptors8']
                and r['y']==oldys[r['component_id']]['native_target'],'literal consumed512 labels/context')
    ordered=sorted(range(808),key=lambda i:(hashlib.sha256(('query_code_oof_v1|'+ids[i]).encode()).hexdigest(),ids[i]))
    folds={i:rank%5 for rank,i in enumerate(ordered)}
    assignment=read(folder/'OOF_ASSIGNMENT_BEFORE_FITS.json')
    require(assignment['fold_sizes']==[162,162,162,161,161],'fixed whole-group fold sizes')
    require(assignment['rows']==[{'component_id':ids[i],'history_index':i,'fold':folds[i]} for i in range(808)],'fold assignment before fit')
    oof=read(folder/'OOF808_LAWS.json')['rows']
    require([r['component_id'] for r in oof]==ids,'OOF law full original order')
    model_reports=[]
    for fold in range(5):
        state=read(folder/('OOF_RF_STATE_'+str(fold)+'.json'))
        train=[i for i in range(808) if folds[i]!=fold];held=[i for i in range(808) if folds[i]==fold]
        require(state['training_component_ids']==[ids[i] for i in train]
                and state['heldout_component_ids']==[ids[i] for i in held],'whole-group OOF exclusion')
        laws,report=audit_saved_model(state,[history[i]['X'] for i in train],[history[i]['y'] for i in train],
                                     [history[i]['X'] for i in held])
        report['laws']=compare_laws([oof[i]['law'] for i in held],laws)
        model_reports.append(report)
    contexts=read(folder/'CONTEXTS512.json')['records']
    predictions=read(folder/'INITIAL512_LAWS.json')['records']
    selected=role['selected_groups']
    require([r['component_id'] for r in contexts]==role['trial_component_ids']==[r['component_id'] for r in predictions],
            'frozen512 original role order')
    require(all(r['identity']==s['representative'] for r,s in zip(contexts,selected)),'literal selected representatives')
    state=read(folder/'RF808_STATE.json')
    require(state['training_component_ids']==ids and state['prediction_component_ids']==role['trial_component_ids'],
            'RF808 provenance')
    laws,report=audit_saved_model(state,[r['X'] for r in history],[r['y'] for r in history],[r['X'] for r in contexts])
    report['laws']=compare_laws([r['F808'] for r in predictions],laws);model_reports.append(report)
    # No new target is consulted: geometry and prior heads use only original808.
    gbase=[];gcontext=[]
    for row,h in zip(oof,history):
        gb=exact_geo(row['law'],h['y'])
        gbase.append(gb)
        gcontext.append(exact_geo(row['law'],h['y'],h['X'][7]))
    heads=read(folder/'PRIOR_HEADS.json');head_reports={}
    from reference_math import P
    for name,geos,prior in (('global',gbase,P),('context',gcontext,P*2)):
        Q=msum(g['Q'] for g in geos);c=vsum(g['c'] for g in geos)
        head_reports[name]=audit_fit(heads[name],Q,c,prior,808)
    geometry_saved=read(folder/'GEOMETRY512.json')['records'];cache=[]
    from reference_math import law_from_theta
    for c,p,s in zip(contexts,predictions,geometry_saved):
        g=c['X'][7];gg=exact_geo(p['F808'],None,g)
        require(s['component_id']==c['component_id'] and matrix(s['Q_exact'])==gg['Q']
                and tuple(vec(v) for v in s['c_bounds_exact'])==gg['bounds'],'all512 exact truth-free Q/bounds')
        for key,theta,contextg in (('priorglobal',heads['global']['theta'],None),
                                  ('priorcontext',heads['context']['theta'],g)):
            rebuilt=law_from_theta(p['F808'],theta,contextg)
            from reference_math import canonical
            require(canonical(p[key])==rebuilt,'all512 actual prior code distributions')
        cache.append({'component_id':c['component_id'],'g':s['g_exact'],'Q':gg['Q'],
                      'bounds':gg['bounds'],'knot_c':gg['knot_c']})
    cachepath=args.output.with_name('INITIAL_INDEPENDENT_GEOMETRY_CACHE.json.gz')
    require(not cachepath.exists(),'independent cache append-only')
    with gzip.open(cachepath,'xt',encoding='utf8') as f:json.dump(fraction_json({'records':cache}),f,separators=(',',':'))
    return {'status':'PASS_INDEPENDENT_INITIAL_RF_OOF_AND_EXACT_GEOMETRY_AUDIT',
       'initial_freeze_sha256':sha(folder/'FROZEN.json'),'model_reports':model_reports,'head_reports':head_reports,
       'OOF_exclusion_groups':808,'initial_truth_free_geometries_exact':512,'actual_code_laws_checked':1024,
       'cache':{'file':str(cachepath),'sha256':sha(cachepath)},'new_targets_read':0,'SUT_imports':0,'fits':0,
       'scope':'All six RF saved states, original labels/train exclusion, global/context head objective and every frozen512 Q/bounds/code law; Beta prediction arithmetic is checked in final phase.'}


def trace_queried_source(args,contexts,actual):
    import gzip,math
    plan=read(args.plan);pin=plan['inputs']['raw_source']
    require(sha(pin['file'])==pin['sha256'],'original public source pin')
    with gzip.open(pin['file'],'rt',encoding='utf8') as f:table=json.load(f)
    cols={k:i for i,k in enumerate(table['columns'])}
    byid={r['component_id']:r for r in contexts}
    for c in contexts:
        ident=c['identity'];index=ident['raw_row_zero_based'];r=table['data'][index]
        require(table['index'][index]==ident['artifact_native_index'] and r[cols['jid']]==ident['jid']
                and r[cols['mpid']]==ident['mpid'],'original selected source identity')
        amounts={k:Fraction(str(v)) for k,v in r[cols['composition']].items()
                 if not k.startswith('@') and Fraction(str(v))>0}
        total=sum(amounts.values(),Fraction());f={k:float(v/total) for k,v in amounts.items()}
        sig='|'.join(k+':'+str(amounts[k]/total) for k in sorted(amounts))
        require(sig==ident['normalized_composition'],'original composition normalization')
        subset=lambda elements:math.fsum(f.get(k,0.) for k in elements)
        d=[float('O' in f),subset(('O',)),subset(('H',)),subset(('C','N')),subset(('F','Cl','Br','I')),
           subset(('S','Se','Te')),float(len(f)),-math.fsum(v*math.log(v) for v in f.values())]
        require(c['X']==[float(r[cols['e_form']]),float(r[cols['gap opt']])]+d,'all512 previously disclosed original context cells')
    for a in actual:
        c=byid[a['component_id']];r=table['data'][c['identity']['raw_row_zero_based']]
        require(float(r[cols['bulk modulus']])==a['native_target'],'128 actual queried original literal targets')
    return {'original_identity_rows':512,'already_disclosed_proxy_cells':1024,'already_queried_target_cells':len(actual),
            'unqueried_target_columns_accessed':0,'physical_public_JSON_decode_not_crypto_isolation':True}


def queries(args):
    import gzip
    from reference_math import vec,matrix,P,scalar,inner,theta_beta,policy,ticket_position
    from audit_saved_gate_v2 import audit_fit,msum
    folder=args.stage/'query_once';complete=read(folder/'COMPLETE.json');verify_manifest(folder,complete)
    initial_folder=args.stage/'initial_once';initial_report=read(args.stage/'independent_audit'/'INITIAL_SAVED_AUDIT.json')
    require(initial_report['status'].startswith('PASS'),'initial independent audit prerequisite')
    cachepath=Path(initial_report['cache']['file']);require(sha(cachepath)==initial_report['cache']['sha256'],'cache provenance')
    with gzip.open(cachepath,'rt',encoding='utf8') as f:raw=json.load(f)['records']
    gs=[{'Q':matrix(r['Q']),'bounds':tuple(vec(v) for v in r['bounds']),'knot_c':tuple(vec(v) for v in r['knot_c'])} for r in raw]
    predictions=read(initial_folder/'INITIAL512_LAWS.json')['records']
    contexts=read(initial_folder/'CONTEXTS512.json')['records'];ids=[r['component_id'] for r in contexts]
    prior=vec(read(initial_folder/'PRIOR_HEADS.json')['context']['theta']);theta=prior
    d=6;n=len(gs);remaining=list(range(n));sumc=sumz=past=(Fraction(),)*d
    mid=[tuple((a+b)/2 for a,b in g['bounds']) for g in gs]
    widths=[max(b-a for a,b in g['bounds']) for g in gs]
    M=tuple(sum((m[j] for m in mid),Fraction()) for j in range(d))
    selected=[];trueys=[];fits=[];gaps=None
    for t in range(1,129):
        if (t-1)%32==0:
            b=(t-1)//32+1;policy_saved=read(folder/('BATCH_'+str(b)+'_POLICY.json'))
            require(vec(policy_saved['theta_before_batch'])==theta and vec(policy_saved['width_max_coordinate_exact'])==tuple(widths),
                    'predraw batch theta/static geometry widths')
            beta=theta_beta(theta);gaps=[]
            for geo in gs:
                values=[inner(beta,c) for c in geo['knot_c']]
                gaps.append(2*(max(values)-min(values)))
            require(vec(policy_saved['current_gap_width_exact'])==tuple(gaps),'all512 batch gapwidth exact')
        q=policy([widths[i] for i in remaining],[gaps[i] for i in remaining])
        ticket=read(folder/('TICKET_'+str(t).zfill(3)+'.json'));obs=read(folder/('OBSERVE_'+str(t).zfill(3)+'.json'))
        position=ticket_position(q,ticket['ticket'],ticket['ticket_total']);index=remaining[position]
        require(ticket['remaining_indices_before']==remaining and ticket['index']==index and ticket['position']==position
                and scalar(ticket['q'])==q[position] and ticket['component_id']==ids[index],'exact predraw proposal/ticket/identity')
        require(obs['component_id']==ids[index] and obs['ticket_sha256']==sha(folder/('TICKET_'+str(t).zfill(3)+'.json')),'observation follows actual ticket')
        require(obs['actual_utc']>=ticket['actual_utc'],'receipt draw-before-observation time')
        target=obs['native_target'];geo=exact_geo(predictions[index]['F808'],target,contexts[index]['X'][7]);c=geo['c']
        Z=tuple(m+p+(cc-mm)/q[position] for m,p,cc,mm in zip(M,past,c,mid[index]))
        require(vec(obs['c_exact'])==c and vec(obs['Z_total_exact'])==Z,'actual queried native label c/Z')
        sumz=tuple(a+b for a,b in zip(sumz,Z));sumc=tuple(a+b for a,b in zip(sumc,c))
        past=tuple(a+b-mm for a,b,mm in zip(past,c,mid[index]))
        selected.append(index);trueys.append(target);del remaining[position]
        if t%32==0:
            update=read(folder/('BATCH_'+str(t//32)+'_UPDATE.json'))
            Q=msum(gs[i]['Q'] for i in remaining);chat=tuple(z/t-c for z,c in zip(sumz,sumc))
            require(Q==matrix(update['Q_remaining_exact']) and chat==vec(update['Chat_remaining_exact'])
                    and sumz==vec(update['sum_Z_total_exact']) and sumc==vec(update['sum_c_queried_exact'])
                    and remaining==update['remaining_indices'],'current remaining Q/c estimate exact')
            fit=audit_fit(update['fit'],Q,chat,prior,len(remaining));fits.append(fit);theta=vec(update['fit']['theta'])
    require([ids[i] for i in selected]==complete['queried_ids'] and [ids[i] for i in remaining]==complete['remaining_ids'],
            'actual128 disjoint/common384')
    actual=read(folder/'ACTUAL128_TARGETS.json')['records']
    require([r['component_id'] for r in actual]==complete['queried_ids'] and [r['native_target'] for r in actual]==trueys,'actual query training table')
    Qr=msum(gs[i]['Q'] for i in remaining);scale=Fraction(384,128)
    Qq=tuple(tuple(scale*v for v in row) for row in msum(gs[i]['Q'] for i in selected))
    cq=tuple(scale*v for v in sumc)
    erm_record=read(folder/'PLAIN_QUERY_ERM.json')['fit'];erm=audit_fit(erm_record,Qq,cq,prior,384,penalty=Qr)
    from reference_math import law_from_theta,canonical
    laws=read(folder/'QUERY_CODE_REMAINING_LAWS.json')['records']
    require([r['component_id'] for r in laws]==complete['remaining_ids'],'remaining predicted law order')
    for i,row in zip(remaining,laws):
        for name,weights in (('geometryRiskCode',theta),('plainQueryERM',erm_record['theta'])):
            require(canonical(row[name])==law_from_theta(predictions[i]['F808'],weights,contexts[i]['X'][7]),'all768 final learned laws')
    return {'status':'PASS_INDEPENDENT_ACTUAL128_QUERY_ESTIMATOR_AND_HEAD_AUDIT','query_complete_sha256':sha(folder/'COMPLETE.json'),
        'exact_tickets_queries':128,'exact_batch_fullpool_gap_arrays':4,'Q_Chat_updates':4,'head_fit_records':fits,'ordinary_query_erm':erm,
        'source_trace':trace_queried_source(args,contexts,actual),'remaining_laws_exact':768,'new_unqueried_targets_read':0,'SUT_imports':0,'optimizer_calls':0}

def native_summary(law):
    import math
    from scipy.special import ndtri
    from reference_math import canonical
    probabilities=(Fraction(1,40),Fraction(1,20),Fraction(1,10),Fraction(1,2),
                   Fraction(9,10),Fraction(19,20),Fraction(39,40))
    if law['family']=='weighted_empirical':
        atoms=canonical(law);running=Fraction();quantiles={}
        for x,w in atoms:
            running+=w
            for p in probabilities:
                if p not in quantiles and running>=p:quantiles[p]=float(x)
        mean_exact=sum((x*w for x,w in atoms),Fraction())
        mean=float(mean_exact)
    else:
        require(law['family']=='normal','native Normal or empirical only')
        mean,sd=float(law['location']),float(law['scale'])
        require(math.isfinite(mean) and math.isfinite(sd) and sd>0,'finite Normal')
        quantiles={p:mean+sd*float(ndtri(float(p))) for p in probabilities}
        mean_exact=None
    intervals={}
    for level,low,high in (('80',Fraction(1,10),Fraction(9,10)),
                            ('90',Fraction(1,20),Fraction(19,20)),
                            ('95',Fraction(1,40),Fraction(39,40))):
        a,b=quantiles[low],quantiles[high]
        intervals[level]={'lower':a,'upper':b,'width':b-a}
    return {'mean':mean,'mean_exact':mean_exact,'median':quantiles[Fraction(1,2)],'intervals':intervals}


def close_native(x,y,label,tol=2e-10):
    import math
    x,y=float(x),float(y)
    require(math.isfinite(x) and math.isfinite(y) and abs(x-y)<=tol*max(1,abs(x),abs(y)),label)


def final(args):
    import math
    from scipy.special import betainc
    from reference_math import canonical
    from audit_saved_models import audit_saved_model,compare_laws
    initial_folder=args.stage/'initial_once';query_folder=args.stage/'query_once'
    updated_folder=args.stage/'updated_once';final_folder=args.stage/'final_once'
    frozen=read(final_folder/'FROZEN.json');verify_manifest(final_folder,frozen)
    updated=read(updated_folder/'COMPLETE.json');verify_manifest(updated_folder,updated)
    query=read(query_folder/'COMPLETE.json');verify_manifest(query_folder,query)
    require(frozen['plan_sha256']==updated['plan_sha256']==query['plan_sha256']==sha(args.plan),
            'unchanged main contract all phases')
    ia=read(args.stage/'independent_audit'/'INITIAL_SAVED_AUDIT.json')
    qa=read(args.stage/'independent_audit'/'QUERY_SAVED_AUDIT.json')
    require(ia['status'].startswith('PASS') and qa['status'].startswith('PASS')
            and qa['query_complete_sha256']==sha(query_folder/'COMPLETE.json'),'independent prior stages closed')
    history=read(initial_folder/'HISTORY808_RECOVERED.json')['rows']
    contexts=read(initial_folder/'CONTEXTS512.json')['records'];ctx={r['component_id']:r for r in contexts}
    actual=read(query_folder/'ACTUAL128_TARGETS.json')['records']
    train=history+[{'component_id':r['component_id'],'X':ctx[r['component_id']]['X'],'y':r['native_target']} for r in actual]
    require(len(train)==len({r['component_id'] for r in train})==936,'same936 actual labels every updated baseline')
    remaining=query['remaining_ids'];require(len(remaining)==384 and frozen['remaining_ids']==updated['remaining_ids']==remaining,'common384')
    require(not set(remaining)&{r['component_id'] for r in train},'no remaining target in training')
    laws=read(updated_folder/'UPDATED_BASELINE_LAWS.json')['records']
    require([r['component_id'] for r in laws]==remaining,'updated law order')
    model_reports={}
    for name,filename in (('RF936','RF936_STATE.json'),('GP936','GP936_STATE.json'),
                           ('CRPS936','CRPS936_STATE.json'),('NGBoost936','NGBOOST936_STATE.json')):
        state=read(updated_folder/filename)
        require(state['training_component_ids']==[r['component_id'] for r in train]
                and state['prediction_component_ids']==remaining,'same original train/prediction identities '+name)
        rebuilt,report=audit_saved_model(state,[r['X'] for r in train],[r['y'] for r in train],[ctx[k]['X'] for k in remaining])
        report['laws']=compare_laws([r[name] for r in laws],rebuilt,model_kind=state['kind'])
        model_reports[name]=report
    base=read(initial_folder/'INITIAL512_LAWS.json')['records'];baseby={r['component_id']:r for r in base}
    learned=read(query_folder/'QUERY_CODE_REMAINING_LAWS.json')['records'];learnedby={r['component_id']:r for r in learned}
    updatedby={r['component_id']:r for r in laws}
    finaldata=read(final_folder/'FINAL10_LAWS384.json');records=finaldata['records'];methods=finaldata['methods']
    require(methods==['F808','priorglobal','priorcontext','Beta','geometryRiskCode','plainQueryERM',
                      'RF936','GP936','CRPS936','NGBoost936'],'all10 no selection of favorable controls')
    require([r['component_id'] for r in records]==remaining,'final common384 identity order')
    for row in records:
        key=row['component_id']
        for method in methods:
            source=baseby[key] if method in methods[:4] else learnedby[key] if method in methods[4:6] else updatedby[key]
            require(row[method]==source[method],'all10 frozen law collection unchanged')
    # Independent beta CDF-transform verification. Floating incomplete-beta
    # arithmetic has fixed 2e-10 CDF tolerance; no claim of exact special function.
    head=read(initial_folder/'PRIOR_HEADS.json')['Beta'];a,b=head['alpha'],head['beta'];max_beta_error=0.
    require(.05<=a<=20 and .05<=b<=20,'declared beta parameter bounds')
    for row in base:
        atom=canonical(row['F808']);betas=canonical(row['Beta'])
        require([x for x,w in atom]==[x for x,w in betas],'Beta retains every atom')
        cp=cq=Fraction()
        for (_,w),(_,v) in zip(atom,betas):
            cp+=w;cq+=v
            error=abs(float(cq)-float(betainc(a,b,float(cp))))
            require(error<=2e-10,'independent numerical Beta CDF')
            max_beta_error=max(max_beta_error,error)
    summaries=read(final_folder/'FINAL10_SUMMARIES384.json')['records']
    require([r['component_id'] for r in summaries]==remaining,'pre-Y summary identity order')
    checked=0
    for row,saved in zip(records,summaries):
        for method in methods:
            summary=native_summary(row[method]);ss=saved[method]
            close_native(summary['mean'],ss['mean'],'predictive mean')
            close_native(summary['median'],ss['median'],'predictive median')
            if summary['mean_exact'] is not None:
                require(str(summary['mean_exact'])==ss['mean_exact_empirical'],'exact empirical mean')
            for level,interval in summary['intervals'].items():
                for k in ('lower','upper','width'):
                    close_native(interval[k],ss['equal_tail_intervals'][level][k],'frozen interval '+level+' '+k)
            checked+=1
    return {'status':'PASS_INDEPENDENT_FINAL10_FORECAST_AND_SAVED_STATE_AUDIT',
        'final_freeze_sha256':sha(final_folder/'FROZEN.json'),'initial_freeze_sha256':sha(initial_folder/'FROZEN.json'),
        'query_complete_sha256':sha(query_folder/'COMPLETE.json'),'same936_training_groups':936,
        'updated_state_inference_reports':model_reports,'all10_laws_frozen_on_common384':3840,
        'deterministic_native_summaries_checked':checked,'Beta_frozen512_maximum_CDF_error':max_beta_error,
        'fits_or_optimizers':0,'SUT_imports':0,'remaining_Y_cells_read':0,
        'scope':'Saved-state inference and fair information/provenance, no claim to certify training optimization or wet experiments.'}


def scores(args):
    import math
    from scipy.special import ndtr
    from reference_math import energy_crps
    final_folder=args.stage/'final_once';score_folder=args.stage/'score_once'
    frozen=read(final_folder/'FROZEN.json');verify_manifest(final_folder,frozen)
    complete=read(score_folder/'COMPLETE.json');verify_manifest(score_folder,complete)
    records=read(final_folder/'FINAL10_LAWS384.json')['records'];methods=frozen['methods']
    targets=read(score_folder/'REMAINING384_TARGETS.json')['records']
    saved=read(score_folder/'SCORES384.json')['records']
    diag=read(score_folder/'POINT_INTERVAL_DIAGNOSTICS384.json')['records']
    aggregate=read(score_folder/'AGGREGATE.json')
    require([r['component_id'] for r in records]==[r['component_id'] for r in targets]
            ==[r['component_id'] for r in saved]==[r['component_id'] for r in diag]==frozen['remaining_ids'],
            'only common remaining384 endpoint')
    sums={m:[] for m in methods};squares={m:[] for m in methods};absolutes={m:[] for m in methods}
    coverage={m:{k:[] for k in ('80','90','95')} for m in methods}
    widths={m:{k:[] for k in ('80','90','95')} for m in methods};max_error=0.
    for row,target,score,details in zip(records,targets,saved,diag):
        y=target['native_target']
        require(y==score['native_target']==details['native_target'],'same actual original remainingY')
        for method in methods:
            law=row[method]
            if law['family']=='weighted_empirical':value=float(energy_crps(law,y))
            else:
                mu,sd=law['location'],law['scale'];z=(y-mu)/sd
                value=sd*(z*(2*float(ndtr(z))-1)+2*math.exp(-z*z/2)/math.sqrt(2*math.pi)-1/math.sqrt(math.pi))
            require(math.isfinite(value) and value>=0,'finite nonnegative proper CRPS')
            close_native(value,score[method],'independent native CRPS',tol=2e-10)
            max_error=max(max_error,abs(value-score[method]));sums[method].append(value)
            summary=native_summary(law);err=summary['mean']-y
            squares[method].append(err*err);absolutes[method].append(abs(err))
            close_native(err,details[method]['prediction_error_mean_minus_Y'],'point error')
            for level,interval in summary['intervals'].items():
                covered=interval['lower']<=y<=interval['upper']
                require(covered==details[method]['interval_observation'][level]['covered_inclusive'],'inclusive actual coverage')
                close_native(interval['width'],details[method]['interval_observation'][level]['width'],'interval width')
                coverage[method][level].append(int(covered));widths[method][level].append(interval['width'])
    means={m:math.fsum(sums[m])/384 for m in methods}
    for method in methods:
        close_native(means[method],aggregate['mean_CRPS_native'][method],'aggregate384 CRPS')
        ss=aggregate['secondary_point_interval_metrics_native'][method]
        close_native(math.sqrt(math.fsum(squares[method])/384),ss['RMSE'],'aggregate RMSE')
        close_native(math.fsum(absolutes[method])/384,ss['MAE'],'aggregate MAE')
        for level in ('80','90','95'):
            close_native(sum(coverage[method][level])/384,ss['coverage'][level],'aggregate coverage')
            close_native(math.fsum(widths[method][level])/384,ss['mean_interval_width'][level],'aggregate width')
    return {'status':'PASS_INDEPENDENT_COMMON384_NATIVE_SCORE_AUDIT','final_freeze_sha256':sha(final_folder/'FROZEN.json'),
       'scores_receipt_sha256':sha(score_folder/'COMPLETE.json'),'native_CRPS_values_checked':3840,
       'maximum_native_CRPS_absolute_difference':max_error,'mean_CRPS_native_independent':means,
       'source_trace':trace_queried_source(args,read(args.stage/'initial_once'/'CONTEXTS512.json')['records'],targets),
       'new_unique_targets':0,'fits_or_optimizers':0,'SUT_imports':0,
       'scope':'Saved already-consumed384 targets only; queries do not enter main remaining CRPS, no posthoc recut or refit.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['metadata','initial','queries','final','scores'],required=True)
    ap.add_argument('--stage',type=Path,required=True);ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    require(not args.output.exists(),'append-only audit output')
    start=datetime.now(timezone.utc).isoformat()
    result={'metadata':metadata,'initial':initial,'queries':queries,'final':final,'scores':scores}[args.phase](args)
    result.update(actual_start_utc=start,actual_end_utc=datetime.now(timezone.utc).isoformat(),
                  audit_source_sha256=sha(__file__),plan_sha256=sha(args.plan))
    with args.output.open('x',encoding='utf8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':result['status'],'sha256':sha(args.output)}))

if __name__=='__main__':main()
