"""Explicit source/role/label contracts for three physical-reference rounds.

The trusted decoder reads a whole existing public JSON table. Current targets
leave this reader only through a freeze-bound permit. This is logical
disclosure, not physical hiding, new DFT or an experimental saving claim.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

SCOPE = 'PHYSICAL_REFERENCE_VALIDATION_V1'
FILES = ('task_io.py', 'numerical_adapter.py', 'forecast_stage.py', 'verification_stage.py', 'score_stage.py')
METHODS = ('directRF', 'directAuto', 'ionicRF', 'ionicBeta', 'ionicGlobal', 'ionicAnchor', 'ionicAuto', 'GP10total', 'CRPSRFionic', 'electronic_delta')
ROLES = {'certificate': 64, 'confirmation': 64}
ROUNDS = 3
UNIT = 'dimensionless relative static permittivity; poly_total = mean dielectric-tensor eigenvalues'
RESIDUAL_UNIT = 'dimensionless modeled ionic residual; fl(poly_total - poly_electronic)'
FEATURE_IDS = ['band_gap_eV', 'poly_electronic_dimensionless', 'O_present', 'f_O', 'f_H', 'f_C_plus_f_N', 'f_F_Cl_Br_I', 'f_S_Se_Te', 'element_count', 'composition_entropy']
HASH_PREFIX = 'physical_reference_v1|'


def utc(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def _unique(pairs):
    result = {}
    for k,v in pairs:
        if k in result: raise ValueError('duplicate JSON key')
        result[k] = v
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError('nonfinite JSON '+s)))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def sources():
    root = Path(__file__).resolve().parent
    return {name: sha(root/name) for name in FILES}


def verify_payloads(folder, pins):
    root = Path(folder).resolve()
    for name, digest in pins.items():
        p = (root/name).resolve(); p.relative_to(root)
        if sha(p) != digest: raise ValueError('frozen payload changed: '+name)


def validate(planfile, approvalfile, phase):
    plan, approval = read(planfile), read(approvalfile)
    if (plan.get('status') != 'REGISTERED' or plan.get('scope') != SCOPE
        or approval.get('approved') is not True or approval.get('scope') != SCOPE
        or approval.get('plan_sha256') != sha(planfile) or phase not in approval.get('allowed_phases', [])):
        raise ValueError('exact registered phase approval required')
    if sources() != plan['source_sha256']: raise ValueError('registered local source changed')
    for section in ('inputs', 'dependencies'):
        for name,pin in plan[section].items():
            if sha(pin['file']) != pin['sha256']: raise ValueError('changed pinned '+section+': '+name)
    if (plan['rounds'] != 3 or plan['role_sizes'] != ROLES or plan['methods'] != list(METHODS)
        or plan['hash_prefix'] != HASH_PREFIX or plan['fixed_auto_g'] != 5 or plan['anchor_g'] != 0):
        raise ValueError('fixed three-round/same-information recipe required')
    if read(plan['inputs']['old_score_audit']['file'])['status'] != 'PASS':
        raise ValueError('old448 experiment must be independently closed')
    if not read(plan['inputs']['stable_numerical_review']['file'])['status'].startswith('PASS'):
        raise ValueError('audited stable Newton source required')
    if not read(plan['inputs']['CRPS_baseline_review']['file'])['status'].startswith('PASS'):
        raise ValueError('independent CRPS baseline review required before material work')
    result = deepcopy(plan)
    result['_plan_sha256'] = sha(planfile)
    result['_plan_file'] = str(Path(planfile).resolve())
    return result


def expected_roles(plan):
    old = read(plan['inputs']['old_roles']['file'])
    remainder = old['remaining_unselected_component_ids']
    if len(remainder) != len(set(remainder)) or len(remainder) != 403:
        raise ValueError('exact403 previously unconsumed whole groups required')
    metadata = read(plan['inputs']['identity_metadata']['file'])
    groups = {r['component_id']: r for r in metadata['groups']}
    ordered = sorted(remainder, key=lambda key: (hashlib.sha256((HASH_PREFIX+key).encode()).hexdigest(), key))
    result = {'plan_sha256': plan['_plan_sha256'], 'hash_prefix': HASH_PREFIX,
              'rounds': 3, 'role_sizes': dict(ROLES), 'remaining_component_ids': ordered[384:],
              'selected_component_ids': ordered[:384], 'metadata_only_selection': True}
    for number in range(1,4):
        current = {}; start = 128*(number-1)
        for j,role in enumerate(ROLES):
            identities = []
            for key in ordered[start+64*j:start+64*(j+1)]:
                identity = deepcopy(groups[key]['representative'])
                identity['component_id'] = key
                identity['source_row'] = identity['raw_row_zero_based']
                identities.append(identity)
            current[role] = identities
        result['round'+str(number)] = current
    return result


def load_roles(plan, rolefile):
    roles = read(rolefile)
    expected = expected_roles(plan)
    if roles != expected: raise ValueError('roles differ from exact precommitted metadata-only allocation')
    return roles


def descriptor_pair(identity):
    # Same finite atomic-amount convention as the frozen calibration module.
    amounts = {e: Fraction(str(float(Fraction(v)))) for e,v in identity['composition_amount_fractions'].items()}
    total = sum(amounts.values(), Fraction()); f = {e:a/total for e,a in amounts.items()}
    get = lambda e:f.get(e,Fraction())
    entropy = -math.fsum(float(v)*math.log(float(v)) for v in f.values())
    bounded = [float(get('O')>0),float(get('O')),float(get('H')),float(get('C')+get('N')),
               float(sum((get(e) for e in ('F','Cl','Br','I')),Fraction())),
               float(sum((get(e) for e in ('S','Se','Te')),Fraction())),len(f)/118.,entropy/math.log(118.)]
    bounded = [min(1.,max(0.,v)) for v in bounded]
    return bounded[:6]+[bounded[6]*118,bounded[7]*math.log(118)],bounded


def feature_row(record):
    raw,_ = descriptor_pair(record['identity'])
    return record['raw_known_proxies2']+raw


def initial_history(plan):
    records = read(plan['inputs']['old_final_history']['file'])['calibration_records']
    initial = read(plan['inputs']['old_initial_fit']['file'])['rows']
    visible = read(plan['inputs']['old_visible_contexts']['file'])
    proxies = {r['identity']['component_id']:r['native_values'][:2] for r in initial}
    for rows in visible.values():
        for r in rows: proxies[r['identity']['component_id']] = r['native_proxies']
    if len(records) != 448 or len(proxies) != 448: raise ValueError('closed448 history required')
    native = []
    for r in records:
        p = proxies[r['identity']['component_id']]
        native.append({'identity':deepcopy(r['identity']),'native_target':r['native_target'],
                       'raw_known_proxies2':p,'known_electronic':p[1],
                       'bounded_descriptors8':r['bounded_descriptors8']})
    return native,deepcopy(records),read(plan['inputs']['old_transform']['file'])


def load_modules(plan):
    order = ('empirical_native','simplex_transport','calibration_views','stable_simplex','affine_reference',
             'crps_tree','crps_forest','reference_baselines')
    modules = {}
    for name in order:
        pin = plan['dependencies'][name]; path = Path(pin['file'])
        if sha(path) != pin['sha256']: raise ValueError('dependency changed before import')
        spec = importlib.util.spec_from_file_location(name,path)
        module = importlib.util.module_from_spec(spec);sys.modules[name] = module
        spec.loader.exec_module(module);modules[name] = module
    return modules


class TrustedReader:
    def __init__(self, plan, roles, round_index=1):
        if round_index not in (1,2,3): raise ValueError('registered round required')
        self.plan,self.roles,self.round = plan,roles,round_index
        self.by = {r['component_id']:(role,r) for role,rows in roles['round'+str(round_index)].items() for r in rows}
        if len(self.by) != 128: raise ValueError('128 distinct current identities required')
        with gzip.open(plan['inputs']['raw_source']['file'],'rt',encoding='utf8') as f:
            self.table = json.load(f,object_pairs_hook=_unique)
        cols = self.table['columns']; self.columns = {name:cols.index(name) for name in ('band_gap','poly_electronic','poly_total','material_id')}
        if len(self.table['data']) != 1056: raise ValueError('exactsource1056 rows required')
        self.revealed = set(); self.events = []; self._verified_freeze = None

    def _source_row(self,identity):
        i = identity['raw_row_zero_based']; row = self.table['data'][i]
        if (self.table['index'][i] != identity['artifact_native_index'] or row[self.columns['material_id']] != identity['material_id']):
            raise ValueError('source identity/index mismatch')
        return row

    def contexts(self):
        result = {}
        for role, identities in self.roles['round'+str(self.round)].items():
            result[role] = []
            for identity in identities:
                row = self._source_row(identity)
                p = [row[self.columns['band_gap']],row[self.columns['poly_electronic']]]
                if not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in p):
                    raise ValueError('known admitted proxy changed')
                _,d = descriptor_pair(identity)
                result[role].append({'identity':deepcopy(identity),'raw_known_proxies2':p,
                                     'known_electronic':p[1],'bounded_descriptors8':d})
        return result

    def past448(self): return initial_history(self.plan)

    def reveal_one(self,component_id,permit):
        if component_id not in self.by or component_id in self.revealed: raise ValueError('single registered current target request required')
        if (permit.get('plan_sha256') != self.plan['_plan_sha256'] or permit.get('round') != self.round
            or permit.get('authorized_component_id') != component_id or permit.get('request_index') != len(self.revealed)):
            raise ValueError('exact current plan/round/component/request-index permit required')
        pin = permit['forecast_freeze']; path = Path(pin['file']).resolve()
        if sha(path) != pin['sha256']: raise ValueError('forecast freeze changed')
        frozen = read(path)
        if (frozen['status'] != 'ALL1280_FORECASTS_FROZEN_NO_FUTURE_TARGET_DISCLOSURE'
            or frozen['round'] != self.round or frozen['plan_sha256'] != self.plan['_plan_sha256']
            or frozen['source_sha256'] != sources()): raise ValueError('registered current forecast freeze required')
        if self._verified_freeze != (str(path),pin['sha256']):
            verify_payloads(path.parent,frozen['files_sha256']);self._verified_freeze = (str(path),pin['sha256'])
        role,identity = self.by[component_id]; phase = permit.get('phase')
        if phase == 'certificate_primary':
            if role != 'certificate': raise ValueError('primary verification only queries certificate pool')
        elif phase in ('post_decision_census','confirmation'):
            if role != ('certificate' if phase=='post_decision_census' else 'confirmation'):
                raise ValueError('disclosure phase/role mismatch')
            dp = permit['decision_freeze']; decision_path = Path(dp['file'])
            if sha(decision_path) != dp['sha256']: raise ValueError('decision freeze changed')
            d = read(decision_path)
            if (d['status'] != 'ADOPTION_FROZEN_BEFORE_CENSUS_AND_CONFIRMATION' or d['round'] != self.round
                or d['plan_sha256'] != self.plan['_plan_sha256'] or d['forecast_freeze_sha256'] != pin['sha256']
                or d['choice'] not in ('ionicAuto','ionicRF')): raise ValueError('prior bound adoption decision required')
        else: raise ValueError('unknown target-disclosure phase')
        value = self._source_row(identity)[self.columns['poly_total']]
        if type(value) not in (int,float) or not math.isfinite(value) or value<=0: raise ValueError('previously admitted target changed; stop without replacement')
        self.revealed.add(component_id)
        self.events.append({'actual_utc':utc(),'phase':phase,'component_id':component_id,'request_index':permit['request_index'],
                            'forecast_freeze_sha256':pin['sha256'],'new_true_target_count':1})
        return value


def main():
    ap = argparse.ArgumentParser(description='One metadata-only role allocation, no raw table decode')
    ap.add_argument('--plan',required=True);ap.add_argument('--approval',required=True);ap.add_argument('--output-dir',type=Path,required=True)
    a = ap.parse_args();plan = validate(a.plan,a.approval,'allocate_roles')
    roles = expected_roles(plan);a.output_dir.mkdir(parents=True,exist_ok=False)
    write(a.output_dir/'ROLES.json',roles)
    write(a.output_dir/'COMPLETE.json',{'status':'COMPLETE_384_UNCONSUMED_ROLES_NO_PROPERTIES','actual_utc':utc(),
          'plan_sha256':plan['_plan_sha256'],'roles_sha256':sha(a.output_dir/'ROLES.json'),
          'selected':384,'untouched':19,'properties_emitted':0,'raw_source_decodes':0})


if __name__=='__main__':main()
