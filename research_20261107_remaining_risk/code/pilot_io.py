"""Frozen-reference pilot contracts and target-cell release; no model fitting.

The trusted reader decodes the entire public table, but only permitted cells
leave it. Existing reports are not new experiments or physically hidden labels.
"""
from datetime import datetime,timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SCOPE='REMAINING_RISK_MATERIAL_V1'
FILES=('pilot_io.py','freeze_forecasts.py','query_and_score.py')
UNIT='source bulk GPa attribution, aggregation qualified'
FIELDS=('e_form','gap opt','bulk modulus')
N=512
MAX_QUERIES=384
MIN_REMAINING=128


def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def _unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate JSON key')
        d[k]=v
    return d
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_unique,
                      parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))
def write(path,value):
    with Path(path).open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def sources():return {n:sha(Path(__file__).resolve().parent/n) for n in FILES}
def verify_payloads(folder,pins):
    root=Path(folder).resolve()
    for n,h in pins.items():
        p=(root/n).resolve();p.relative_to(root)
        if sha(p)!=h:raise ValueError('frozen payload changed: '+n)


def validate(planfile,approvalfile,phase):
    if getattr(sys,'get_int_max_str_digits',lambda:0)()!=0:
        raise ValueError('start with -X int_max_str_digits=0 before any new property access')
    plan,a=read(planfile),read(approvalfile)
    if (plan.get('status')!='REGISTERED' or plan.get('scope')!=SCOPE or a.get('approved') is not True
        or a.get('scope')!=SCOPE or a.get('plan_sha256')!=sha(planfile) or phase not in a.get('allowed_phases',[])):
        raise ValueError('exact registered phase approval required')
    if sources()!=plan['source_sha256']:raise ValueError('registered pilot source changed')
    for section in ('inputs','dependencies'):
        for name,pin in plan[section].items():
            if sha(pin['file'])!=pin['sha256']:raise ValueError('changed '+section+' '+name)
    if (plan['pool_groups']!=512 or plan['max_primary_queries']!=384 or plan['min_remaining']!=128
        or plan['delta']!='1/20' or plan['lambda_exponents']!=list(range(-6,3)) or plan['decimal_precision']!=100):
        raise ValueError('fixed remaining-risk pilot recipe required')
    for key in ('identity_review','old_state_review','core_review'):
        if not read(plan['inputs'][key]['file']).get('status','').startswith('PASS'):
            raise ValueError('independent '+key+' must pass')
    plan['_plan_sha256']=sha(planfile);plan['_phase']=phase
    return plan


def pool(plan):
    roles=read(plan['inputs']['identity_roles']['file'])
    if roles['status']!='IDENTITY_ONLY_POOL_RESERVED_NO_PROPERTY_AUTHORIZATION':raise ValueError('exact reserved identity pool required')
    rows=roles['selected_groups']
    identities=[]
    for row in rows:
        ident=dict(row['representative'])
        if ident['component_id']!=row['component_id']:raise ValueError('group/representative identity mismatch')
        identities.append(ident)
    if len(identities)!=512 or len({r['component_id'] for r in identities})!=512:
        raise ValueError('512 distinct reserved original groups required')
    return identities


def descriptor_pair(identity):
    amounts={k:Fraction(str(v)) for k,v in identity['composition_amounts'].items() if not k.startswith('@')}
    if any(v<0 for v in amounts.values()):raise ValueError('negative atomic count')
    amounts={k:v for k,v in amounts.items() if v>0};total=sum(amounts.values(),Fraction())
    if total<=0:raise ValueError('empty atomic composition')
    f={k:float(v/total) for k,v in amounts.items()};get=lambda names:math.fsum(f.get(k,0.) for k in names)
    raw=[float('O' in f),get(('O',)),get(('H',)),get(('C','N')),get(('F','Cl','Br','I')),
         get(('S','Se','Te')),float(len(f)),-math.fsum(v*math.log(v) for v in f.values())]
    bounded=raw[:6]+[raw[6]/118.,raw[7]/math.log(118.)]
    if not all(math.isfinite(x) and 0<=x<=1 for x in bounded):raise ValueError('invalid bounded known descriptor')
    return raw,bounded


def load_dependency(plan,name):
    pin=plan['dependencies'][name];path=Path(pin['file'])
    if sha(path)!=pin['sha256']:raise ValueError('dependency changed before import')
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


class TrustedReader:
    def __init__(self,plan,identities):
        if plan.get('_phase') not in ('freeze_forecasts','query_and_census') or not plan.get('_plan_sha256'):
            raise ValueError('validated current phase required before decoding')
        self.plan=plan;self.identities={r['component_id']:r for r in identities};self.revealed=set();self.query_ids=[]
        if len(self.identities)!=512:raise ValueError('512 original groups required')
        with gzip.open(plan['inputs']['raw_source']['file'],'rt',encoding='utf8') as f:self.table=json.load(f,object_pairs_hook=_unique)
        if len(self.table['data'])!=25923:raise ValueError('exact public JARVIS table required')
        self.columns={k:i for i,k in enumerate(self.table['columns'])}
        if len(self.columns)!=len(self.table['columns']):raise ValueError('unique source columns required')
        self._freeze_cache=None

    def _row(self,key):
        identity=self.identities[key];i=identity['raw_row_zero_based'];row=self.table['data'][i]
        c=row[self.columns['composition']]
        amounts={k:Fraction(str(v)) for k,v in c.items() if not k.startswith('@') and Fraction(str(v))>0}
        total=sum(amounts.values(),Fraction());norm='|'.join(k+':'+str(amounts[k]/total) for k in sorted(amounts))
        if (self.table['index'][i]!=identity['artifact_native_index'] or row[self.columns['jid']]!=identity['jid']
            or row[self.columns['mpid']]!=identity['mpid'] or norm!=identity['normalized_composition']):
            raise ValueError('original source identity mismatch')
        return row

    def contexts(self):
        if self.plan['_phase']!='freeze_forecasts':raise ValueError('context release only in forecast phase')
        result=[]
        for key,identity in self.identities.items():
            row=self._row(key);values=[row[self.columns[k]] for k in FIELDS[:2]]
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('invalid known proxy; no replacement')
            raw,bounded=descriptor_pair(identity)
            result.append({'component_id':key,'identity':identity,'native_proxies':list(map(float,values)),
                           'raw_descriptors8':raw,'bounded_descriptors8':bounded})
        return result

    def reveal(self,key,permit):
        if self.plan['_phase']!='query_and_census' or key not in self.identities or key in self.revealed:
            raise ValueError('single registered target release required')
        if permit.get('plan_sha256')!=self.plan['_plan_sha256'] or permit.get('component_id')!=key:
            raise ValueError('plan/key permit mismatch')
        fp=permit['forecast_freeze'];path=Path(fp['file'])
        if sha(path)!=fp['sha256']:raise ValueError('forecast freeze changed')
        freeze=read(path)
        if (freeze['status']!='ALL512_F_G_LAWS_FROZEN_BEFORE_ANY_TARGET' or freeze['plan_sha256']!=self.plan['_plan_sha256']
            or freeze['source_sha256']!=sources()):raise ValueError('current frozen laws required')
        if self._freeze_cache!=(str(path.resolve()),fp['sha256']):
            verify_payloads(path.parent,freeze['files_sha256']);self._freeze_cache=(str(path.resolve()),fp['sha256'])
        seedpin=permit['seed'];seed=read(seedpin['file'])
        if (sha(seedpin['file'])!=seedpin['sha256'] or seed['forecast_freeze_sha256']!=fp['sha256']
            or seed['plan_sha256']!=self.plan['_plan_sha256'] or seed['actual_utc']<freeze['actual_utc']):
            raise ValueError('actual post-freeze seed required')
        if permit['phase']=='query':
            if len(self.query_ids)>=MAX_QUERIES:raise ValueError('query cap exceeded')
            tp=permit['ticket'];ticket=read(tp['file'])
            if (sha(tp['file'])!=tp['sha256'] or ticket['status']!='TICKET_PERSISTED_BEFORE_TARGET'
                or ticket['ticket']['identity']!=key or ticket['ticket']['query_index']!=len(self.query_ids)+1
                or ticket['forecast_freeze_sha256']!=fp['sha256'] or ticket['seed_sha256']!=seedpin['sha256']):
                raise ValueError('current persisted query ticket required')
            self.query_ids.append(key)
        elif permit['phase']=='remaining_census':
            dp=permit['decision'];decision=read(dp['file'])
            if (sha(dp['file'])!=dp['sha256'] or decision['status']!='ALPHA_AND_REMAINING_SET_FROZEN_BEFORE_CENSUS'
                or decision['plan_sha256']!=self.plan['_plan_sha256'] or decision['forecast_freeze_sha256']!=fp['sha256']
                or decision['seed_sha256']!=seedpin['sha256'] or key not in decision['remaining_ids']
                or set(decision['queried_ids'])!=set(self.query_ids)
                or set(decision['remaining_ids'])!=set(self.identities)-set(self.query_ids)):
                raise ValueError('frozen exact remaining set/alpha required before census')
            alpha=Fraction(decision['alpha'])
            if not 0<=alpha<=1:raise ValueError('invalid frozen step')
        else:raise ValueError('unknown target release phase')
        value=self._row(key)[self.columns[FIELDS[2]]]
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('invalid target; stop without replacement')
        self.revealed.add(key)
        return float(value)
