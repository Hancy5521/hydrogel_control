"""Registered phase IO, original-group roles and trusted field release.

No estimator/optimizer is imported here. Public table decoding physically reads
all JSON bytes; only phase-permitted cells leave TrustedReader. This is a
software disclosure boundary, not encryption or a newly performed experiment.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

SCOPE = 'ACTUAL_QUERY_RISK_CODE_V1'
UNIT = 'source bulk GPa attribution, aggregation qualified'
RECIPE = {'historical_groups':808,'oof_folds':5,'pool_groups':512,'queries':128,
          'batch_size':32,'remaining':384,'confirm_reserved':1024,'available_before':6279,
          'future_development_remaining':4743,'oof_hash_prefix':'query_code_oof_v1|',
          'confirm_hash_prefix':'query_code_confirm_v1|','trial_hash_prefix':'query_risk_code_v1|',
          'g_descriptor_index':5,'epsilon':'1/100','tau':'1/4',
          'pit_bins':['0','1/20','1/2','19/20','1'],
          'query_mixture':['1/4','3/8','3/8'], 'model_calls':10}


def require(ok, message):
    if not ok: raise ValueError(message)
def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,'duplicate JSON key'); d[k]=v
    return d
def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'),object_pairs_hook=unique,
        parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def pin(path): return {'file':str(Path(path).resolve()),'sha256':sha(path)}
def verify_pin(p): require(sha(p['file'])==p['sha256'],'pinned file changed: '+str(p['file']))
def verify_payloads(root,manifest):
    root=Path(root).resolve()
    for name,digest in manifest.items():
        p=(root/name).resolve();p.relative_to(root)
        require(sha(p)==digest,'frozen payload changed: '+name)
def load_dependency(plan,name):
    p=plan['dependencies'][name];verify_pin(p)
    spec=importlib.util.spec_from_file_location(name,p['file'])
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def validate(planfile,approvalfile,phase):
    # Before identity selection, imported science code, new field access or fit.
    require(getattr(sys,'get_int_max_str_digits',lambda:0)()==0,
            'run Python with -X int_max_str_digits=0')
    p,a=read(planfile),read(approvalfile)
    require(p.get('status')=='REGISTERED' and p.get('scope')==SCOPE
        and a.get('approved') is True and a.get('scope')==SCOPE
        and a.get('plan_sha256')==sha(planfile) and phase in a.get('allowed_phases',[]),
        'exact actual root registration and phase approval required')
    require(p.get('recipe')==RECIPE,'fixed preregistered recipe required')
    for section in ('sources','inputs','dependencies'):
        for item in p[section].values(): verify_pin(item)
    require(isinstance(p.get('registered_actual_utc'),str) and isinstance(a.get('actual_utc'),str),
            'actual UTC required')
    # Later artifacts do not exist at initial registration. Each later ROOT
    # approval binds their exact hashes to this immutable experiment plan.
    for name,item in a.get('phase_artifacts',{}).items():
        require(name not in p['inputs'],'phase artifacts cannot replace static inputs')
        verify_pin(item);p['inputs'][name]=dict(item)
    p['_plan_sha256']=sha(planfile);p['_phase']=phase
    return p

def descriptor(identity):
    counts={k:F(str(v)) for k,v in identity['composition_amounts'].items() if not k.startswith('@')}
    require(all(v>=0 for v in counts.values()),'negative atomic count')
    counts={k:v for k,v in counts.items() if v>0}; total=sum(counts.values(),F())
    require(total>0,'empty original composition')
    frac={k:float(v/total) for k,v in counts.items()}; get=lambda names:math.fsum(frac.get(k,0.) for k in names)
    raw=[float('O' in frac),get(('O',)),get(('H',)),get(('C','N')),get(('F','Cl','Br','I')),
         get(('S','Se','Te')),float(len(frac)),-math.fsum(v*math.log(v) for v in frac.values())]
    require(all(math.isfinite(v) for v in raw),'finite native descriptor required')
    return raw

def recover808(plan):
    """Only already consumed808; compare old nativeX with originalfloat32 state."""
    keys=('old_RF296','old_HISTORY168','old_visible_contexts','old_validation1_actual_targets',
          'old_validation2_actual_targets','current512_contexts','current512_actual_targets')
    x={k:read(plan['inputs'][k]['file']) for k in keys}
    rf=x['old_RF296']; rows=list(x['old_HISTORY168']['rows'])
    for role in ('validation1','validation2'):
        contexts=x['old_visible_contexts'][role]; targets=x['old_'+role+'_actual_targets']['rows']
        require(len(contexts)==len(targets)==64,'old64 batch required')
        for c,t in zip(contexts,targets):
            require(c['identity']==t['identity'],'old context/Y identity mismatch')
            rows.append({'identity':c['identity'],'native_values':c['native_proxies']+[t['native_target']]})
    byid={r['identity']['component_id']:r for r in rows}; result=[]
    require(len(byid)==296 and set(byid)==set(rf['training_component_ids']),'originalRF296 group mismatch')
    transform=rf['fixed_feature_transform']
    for i,key in enumerate(rf['training_component_ids']):
        r=byid[key]; xx=r['native_values'][:2]+descriptor(r['identity']); y=float(r['native_values'][2])
        actual=[struct.unpack('<f',struct.pack('<f',(v-c)/s))[0]
                for v,c,s in zip(xx,transform['center'],transform['sd'])]
        require(actual==rf['X_train_standardized_then_float32'][i] and y==rf['y_train_native'][i],
                'native history does not match original fitted state')
        result.append({'component_id':key,'identity':r['identity'],'X':xx,'y':y,'history':'original296'})
    truths={r['component_id']:r for r in x['current512_actual_targets']['records']}
    for c in x['current512_contexts']['records']:
        key=c['component_id'];t=truths[key]; d=descriptor(c['identity'])
        require(c['identity']==t['identity'] and c['raw_descriptors8']==d,'consumed512 metadata mismatch')
        result.append({'component_id':key,'identity':c['identity'],'X':c['native_proxies']+d,
                       'y':float(t['native_target']),'history':'consumed512'})
    require(len(result)==len({r['component_id'] for r in result})==808,'unique808 actual history required')
    require(all(len(r['X'])==10 and all(type(v) in (int,float) and math.isfinite(v) for v in r['X']+[r['y']]) for r in result),'finite808 observed rows required')
    return result,transform

def select_metadata(plan):
    require(plan['_phase']=='metadata_roles','metadata-only phase required')
    old=read(plan['inputs']['remaining_identity_roles']['file'])
    require(old['status']=='IDENTITY_ONLY_POOL_RESERVED_NO_PROPERTY_AUTHORIZATION','prior roles contract required')
    available=old['remaining_unselected_component_ids_in_registered_hash_order']
    require(len(available)==len(set(available))==6279,'exact never-used6279 required')
    old_current={r['component_id'] for r in old['selected_groups']}
    consumed=set(old['all_prior_consumed_component_ids'])|old_current
    require(len(consumed)==1920 and not consumed & set(available),'prior1920 whole-group closure required')
    eligible=read(plan['inputs']['eligible_metadata']['file'])['datasets']['jarvis_dft_3d']
    jmeta=read(plan['inputs']['all_JARVIS_identity']['file']);dmeta=read(plan['inputs']['all_dielectric_identity']['file'])
    require(jmeta['property_fields_emitted']==[] and dmeta['target_values_emitted_to_metadata_or_models']==0,'identity-only whole tables required')
    jrows,drows=jmeta['rows'],dmeta['rows']; require((len(jrows),len(drows),len(eligible))==(25923,1056,11020),'exact source frame sizes required')
    legacy=load_dependency(plan,'identity_graph');dsu=legacy.DSU(len(jrows)+len(drows));legacy.connect(jrows+drows,dsu)
    groups=defaultdict(list)
    for r in eligible:
        require(set(r)==legacy.ELIGIBLE,'unknown eligible identity field')
        orig=jrows[r['raw_row_zero_based']]
        require(orig['artifact_native_index']==r['artifact_native_index'] and orig['normalized_composition']==r['normalized_composition'] and legacy.signature(r['composition_amounts'])==r['normalized_composition'],'eligible/source identity mismatch')
        groups[r['component_id']].append(r)
    roots={k:{dsu.find(r['raw_row_zero_based']) for r in v} for k,v in groups.items()}
    require(all(len(v)==1 for v in roots.values()),'alias graph split an eligible group')
    roots={k:next(iter(v)) for k,v in roots.items()};newroots={dsu.find(len(jrows)+i) for i in range(len(drows))}
    blocked=newroots|{roots[c] for c in consumed}
    computed={k for k,r in roots.items() if r not in blocked}
    require(computed==set(available) and len({roots[k] for k in available})==6279,'prior6279 alias-closure no longer exact')
    confirm=sorted(available,key=lambda k:(hashlib.sha256((RECIPE['confirm_hash_prefix']+k).encode()).hexdigest(),k))[:1024]
    confirmset=set(confirm); rest=[k for k in available if k not in confirmset]
    ordered=sorted(rest,key=lambda k:(hashlib.sha256((RECIPE['trial_hash_prefix']+k).encode()).hexdigest(),k))
    trial,left=ordered[:512],ordered[512:]
    closure=defaultdict(list)
    for i,r in enumerate(jrows): closure[dsu.find(i)].append({'source':'jarvis_dft_3d','raw_row_zero_based':i,'artifact_native_index':r['artifact_native_index']})
    for i,r in enumerate(drows): closure[dsu.find(len(jrows)+i)].append({'source':'dielectric_constant','raw_row_zero_based':i,'artifact_native_index':r['artifact_native_index']})
    def record(key,role):
        siblings=sorted(groups[key],key=lambda r:r['raw_row_zero_based']);members=closure[roots[key]]
        require(all(m['source']=='jarvis_dft_3d' for m in members),'reserved group touches excluded source')
        return {'component_id':key,'role':role,'representative':siblings[0],'eligible_siblings':siblings,'all_original_source_siblings_reserved':members}
    return {'status':'PERMANENT_CONFIRM_AND_DEVELOPMENT_ROLES_NO_PROPERTY_AUTHORIZATION','actual_utc':utc(),
        'recipe':RECIPE,'confirm_groups':[record(k,'FINAL_CONFIRM_NOT_FOR_DEVELOPMENT') for k in confirm],
        'selected_groups':[record(k,'TRIAL512_DEVELOPMENT') for k in trial],
        'confirm_component_ids':confirm,'trial_component_ids':trial,'future_development_component_ids':left,
        'all_prior_consumed_component_ids':sorted(consumed),'counts':{'confirm_reserved':1024,'trial':512,'future_development':4743,'prior_consumed':1920},
        'new_properties_read':0,'fits':0,'confirm_extra_property_X_Y_allowed_for_models':False,'identity_composition_metadata_already_accessible':True,
        'grouping':'full original normalized-composition/JID/MPID transitive closure; minimum original eligible rawrow representative'}

def pool(plan):
    roles=read(plan['inputs']['new_roles']['file'])
    require(roles['status']=='PERMANENT_CONFIRM_AND_DEVELOPMENT_ROLES_NO_PROPERTY_AUTHORIZATION','permanent roles required')
    identities=[r['representative'] for r in roles['selected_groups']]
    require(len(identities)==len({r['component_id'] for r in identities})==512,'trial512 groups required')
    require(not set(roles['confirm_component_ids']) & {r['component_id'] for r in identities},'confirm contamination')
    return identities

class TrustedReader:
    def __init__(self,plan,identities):
        require(plan.get('_phase') in ('initial_forecasts','adaptive_queries','remaining_score') and plan.get('_plan_sha256'),'validated new-field phase required')
        self.plan=plan;self.identities={r['component_id']:r for r in identities};self.revealed=set()
        require(len(self.identities)==512,'only current development512 can be decoded')
        with gzip.open(plan['inputs']['raw_source']['file'],'rt',encoding='utf8') as f: self.table=json.load(f,object_pairs_hook=unique)
        require(len(self.table['data'])==25923,'exact raw source dimensions required')
        self.columns={v:i for i,v in enumerate(self.table['columns'])};self._freeze_cache=None;self._final_cache=None
    def _row(self,key):
        ident=self.identities[key];i=ident['raw_row_zero_based'];r=self.table['data'][i]
        counts={k:F(str(v)) for k,v in r[self.columns['composition']].items() if not k.startswith('@') and F(str(v))>0};total=sum(counts.values(),F())
        norm='|'.join(k+':'+str(counts[k]/total) for k in sorted(counts))
        require(self.table['index'][i]==ident['artifact_native_index'] and r[self.columns['jid']]==ident['jid'] and r[self.columns['mpid']]==ident['mpid'] and norm==ident['normalized_composition'],'original representative identity mismatch')
        return r
    def contexts(self):
        require(self.plan['_phase']=='initial_forecasts','X release only initial phase')
        out=[]
        for key,ident in self.identities.items():
            row=self._row(key);proxies=[row[self.columns[k]] for k in ('e_form','gap opt')]
            require(all(type(v) in (int,float) and math.isfinite(v) for v in proxies),'invalid original proxy; no replacement')
            out.append({'component_id':key,'identity':ident,'X':list(map(float,proxies))+descriptor(ident)})
        return out
    def reveal(self,key,permit):
        require(self.plan['_phase'] in ('adaptive_queries','remaining_score') and key in self.identities and key not in self.revealed,'single permitted true target required')
        require(permit['component_id']==key and permit['plan_sha256']==self.plan['_plan_sha256'],'permit identity/plan mismatch')
        freeze=permit['forecast_freeze'];verify_pin(freeze);f=read(freeze['file'])
        require(f['status']=='INITIAL512_F808_PRIORS_GEOMETRY_FROZEN_BEFORE_CURRENT_Y' and f['plan_sha256']==self.plan['_plan_sha256'],'initial frozen laws required')
        if self._freeze_cache!=freeze['sha256']:
            verify_payloads(Path(freeze['file']).parent,f['files_sha256']);self._freeze_cache=freeze['sha256']
        seed=permit['seed'];verify_pin(seed);s=read(seed['file'])
        require(s['forecast_freeze_sha256']==freeze['sha256'] and s['plan_sha256']==self.plan['_plan_sha256'] and s['actual_utc']>=f['actual_utc'],'post-forecast actual OS seed required')
        if self.plan['_phase']=='adaptive_queries':
            tpin=permit['ticket'];verify_pin(tpin);ticket=read(tpin['file'])
            require(ticket['status']=='EXACT_PREDRAW_TICKET_PERSISTED_BEFORE_Y' and ticket['component_id']==key and ticket['query_index']==len(self.revealed)+1 and ticket['forecast_freeze_sha256']==freeze['sha256'] and ticket['seed_sha256']==seed['sha256'],'predraw current ticket required')
            require(len(self.revealed)<128,'128 actualquery cap')
        else:
            finalpin=permit['final_freeze'];verify_pin(finalpin);final=read(finalpin['file'])
            require(final['status']=='ALL10_METHODS384_LAWS_FROZEN_BEFORE_REMAINING_Y' and final['plan_sha256']==self.plan['_plan_sha256'] and key in final['remaining_ids'],'all10 final remaining laws must be frozen')
            if self._final_cache!=finalpin['sha256']:
                verify_payloads(Path(finalpin['file']).parent,final['files_sha256']);self._final_cache=finalpin['sha256']
            audit=permit['independent_forecast_audit'];verify_pin(audit);aread=read(audit['file'])
            require(aread.get('status','').startswith('PASS') and aread['final_freeze_sha256']==finalpin['sha256'],'independent forecast/state arithmetic audit before score required')
        y=self._row(key)[self.columns['bulk modulus']]
        require(type(y) in (int,float) and math.isfinite(y),'invalid original true target; stop without replacement')
        self.revealed.add(key);return float(y)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args()
    p=validate(a.plan,a.approval,'metadata_roles');require(not a.output_dir.exists(),'one-shot output already exists')
    result=select_metadata(p);a.output_dir.mkdir(parents=True,exist_ok=False)
    write(a.output_dir/'ROLE_SELECTION.json',result);write(a.output_dir/'COMPLETE.json',{'actual_utc':utc(),'plan_sha256':p['_plan_sha256'],'selection':pin(a.output_dir/'ROLE_SELECTION.json'),'new_properties_read':0,'fits':0})
    print(json.dumps({'status':result['status'],'counts':result['counts']}))
if __name__=='__main__':main()
