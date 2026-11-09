"""SOURCE ONLY: actualUTC-gated paired dynamic-modulus masking pilot.

Only six input-selected state/temperature curves are admitted. Each context
receives its own6 paired observations; all forecasts, including failures, freeze
before root-released targets. Different masks reuse native cells across roles.
This is one-material state stress testing, not18 independent material samples.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import traceback
import numpy as np
from pure_laws import build_context,score_context,METHODS,MAIN

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'sources/generalization_admission_20261109/other_polymer_functions'
INPUT_GATE=SOURCE/'darus_functional_gate/INPUT_GATE_RESULT.json'
PLAN=ROOT/'generalization/DARUS_FUNCTIONAL_RESPONSE_PROTOCOL_PREPARED.json'
INPUT_SHA='10086c3dfd5bab37b252a895cd283966e50e333e526752abe975c59ff9a4a550'


def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,obj):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def authorize(registration,phase):
    reg=read(registration)
    if reg['schema']!='darus-functional-mean-transport-registration-v1' or phase not in reg['authorized_phases']:
        raise PermissionError('Explicit actualUTC functional protocol required')
    stamp=dt.datetime.fromisoformat(reg['actual_utc'])
    if stamp.tzinfo is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture UTC required')
    if reg['main_method']!=MAIN or reg['methods']!=list(METHODS) or reg['numeric_moduli_before_registration']!=0:
        raise PermissionError('Frozen pre-response candidate/budget required')
    if reg['observed_scalar_budget']!=12 or reg['target_scalar_dimension']!=14 or reg['contexts']!=18:
        raise ValueError('Fixed six-frequency paired budget differs')
    if reg['mean_transport_fraction']!=.5 or reg['weighted_teacher']!='own6_channel_RMS_relative_NNLS':
        raise ValueError('No transport/teacher search admitted')
    for name,digest in reg['source_pins'].items():
        if sha(name)!=digest:raise ValueError('Registered source/input changed: '+name)
    if sha(INPUT_GATE)!=INPUT_SHA:raise ValueError('Independent source/input gate changed')
    gate=read(reg['toy_gate_path'])
    if gate['status']!='PASS' or sha(reg['toy_gate_path'])!=reg['toy_gate_sha256']:
        raise PermissionError('Frozen synthetic numerical gate PASS required')
    return reg


def prepare(out,registration):
    plan=read(PLAN);gate=read(INPUT_GATE)
    by_id={r['condition_id']:r for r in gate['condition_curves']};contexts=[]
    for selected in plan['first_bounded_stage']['selected_input_conditions']:
        c=by_id[selected['condition_id']]
        if c['source_sha256']!=sha(SOURCE/'darus_smp_originals'/c['file']):
            raise ValueError('Original measurement file bytes changed')
        for mask,indices in plan['first_bounded_stage']['masks'].items():
            o,t=indices['O'],indices['T']
            if len(o)!=6 or len(t)!=7 or set(o)&set(t) or set(o)|set(t)!=set(range(13)):
                raise ValueError('Exact native6/7 partition required')
            contexts.append({'context':c['condition_id']+'|'+mask,'condition_id':c['condition_id'],
                'state_block':c['state_block'],'file':c['file'],'source_sha256':c['source_sha256'],
                'mask':mask,'O':o,'T':t,'native_rows':c['native_rows'],
                'frequency_Hz':[r['frequency_Hz'] for r in c['native_rows']],
                'temperature_range_C':c['temperature_range_C'],
                'source_modulus_units':'MPa','working_modulus_units':'Pa'})
    if len(contexts)!=18 or len({c['condition_id'] for c in contexts})!=6:
        raise ValueError('Exactly six selected conditions and three masks required')
    write(out/'PREPARED.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
        'input_gate_sha256':sha(INPUT_GATE),'source_protocol_sha256':sha(PLAN),
        'contexts':contexts,'numeric_moduli_read':0,'material_fits':0,
        'material_count':1,'independent_specimen_count':None,'methods':list(METHODS)})


class RoleReader:
    def __init__(self):self.access=[]
    def values(self,context,indices,role):
        path=SOURCE/'darus_smp_originals'/context['file']
        if sha(path)!=context['source_sha256']:raise ValueError('Source bytes changed')
        wanted={context['native_rows'][i]['source_line_1based']:i for i in indices}
        values={};invalid=[]
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);header=next(reader);units=next(reader)
            cols=[header.index('Storage Modulus'),header.index('Loss Modulus')]
            if [units[i] for i in cols]!=['[MPa]','[MPa]']:
                raise ValueError('Pinned MPa source units differ')
            for row in reader:
                line=reader.line_num
                if line not in wanted:continue
                i=wanted[line];node=context['native_rows'][i]
                if float(row[header.index('Frequency')])!=node['frequency_Hz']:
                    raise ValueError('Original frequency/row mapping differs')
                pair=[]
                for channel,col in enumerate(cols):
                    try:v=float(row[col])*1e6
                    except ValueError:v=float('nan')
                    pair.append(v)
                    if not np.isfinite(v) or v<=0:
                        invalid.append({'source_line_1based':line,'channel':channel,
                                        'reason':'nonfinite_or_nonpositive_modulus'})
                    self.access.append({'context':context['context'],'role':role,'file':context['file'],
                        'source_line_1based':line,'column_0based':col,'channel':channel,
                        'source_unit':'MPa','working_unit':'Pa','exact_multiplier':1000000})
                values[i]=pair
        if set(values)!=set(indices):raise ValueError('Missing whitelisted original measurement row')
        return np.array([values[i] for i in indices]).T,invalid
    def receipt(self):
        return {'actual_utc':utc(),'numeric_scalar_accesses':len(self.access),
            'unique_raw_scalar_cells':len({(a['file'],a['source_line_1based'],a['column_0based']) for a in self.access}),
            'access_sites':self.access,'non_selected_temperature_numeric_responses_read':0,
            'other_response_columns_decoded':0,'scope':'offline context-specific roles, not globally unseen labels'}


def serializable(a):
    return [[float(v) if np.isfinite(v) else None for v in row] for row in a]


def predict(out,registration):
    prepared=read(out/'PREPARED.json')
    if prepared['registration_sha256']!=sha(registration):raise ValueError('Registration differs')
    reader=RoleReader();forecasts=[];failures=[]
    for c in prepared['contexts']:
        observed,invalid=reader.values(c,c['O'],'own_observed6_frequency_12_scalar')
        base={'context':c['context'],'condition_id':c['condition_id'],'state_block':c['state_block'],
            'mask':c['mask'],'file':c['file'],'O':c['O'],'T':c['T'],
            'observed_moduli_Pa':serializable(observed)}
        if invalid:
            failure={'context':c['context'],'stage':'observed_admission','reasons':invalid}
            failures.append(failure);forecasts.append({**base,'status':'FAIL','failure':failure});continue
        try:
            payload=build_context(c['frequency_Hz'],c['O'],c['T'],observed)
            forecasts.append({**base,'status':'OK',**payload})
        except Exception as exc:
            # Numerical/model failures remain cases. No rescue fit, floor,
            # source substitution or successful-case material claim.
            failure={'context':c['context'],'stage':'model_fit_or_law','exception_type':type(exc).__name__,
                     'message':str(exc),'traceback':traceback.format_exc()}
            failures.append(failure);forecasts.append({**base,'status':'FAIL','failure':failure})
    if len(reader.access)!=216:raise ValueError('Exactly18 contexts times12 own scalar observations required')
    write(out/'ANCHOR_ACCESS.json',reader.receipt())
    success=sum(f['status']=='OK' for f in forecasts)
    write(out/'PREDICTIONS.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
        'prepared_sha256':sha(out/'PREPARED.json'),'forecasts':forecasts,
        'contexts_total':18,'contexts_successful':success,'contexts_failed':18-success,
        'failure_rate':(18-success)/18,'failures':failures,'probability_laws':4*success,
        'planned_GP_fits':36,'planned_weighted_NNLS_fits':18,'planned_raw_Pa_NNLS_fits':18,
        'new_materials':0,'physical_material_count':1,'target_numeric_accesses':0,
        'different_masks_reuse_cells_in_other_roles':True,
        'primary_gate_requires_zero_failed_contexts':True})
    write(out/'PREDICTION_COMPLETE.json',{'actual_utc':utc(),'predictions_sha256':sha(out/'PREDICTIONS.json'),
        'anchor_access_sha256':sha(out/'ANCHOR_ACCESS.json'),'all_cases_including_failures_frozen':True})


def reveal(out,registration,pre_audit,release):
    complete=read(out/'PREDICTION_COMPLETE.json');audit=read(pre_audit);rel=read(release)
    if (audit['status']!='PASS' or audit['predictions_sha256']!=sha(out/'PREDICTIONS.json') or
            complete['predictions_sha256']!=sha(out/'PREDICTIONS.json')):
        raise PermissionError('Independent frozen forecast pre-audit PASS required')
    if (rel['schema']!='darus-functional-target-release-v1' or rel['authorization']!='REVEAL_FIXED18_CONTEXT_TARGETS' or
            rel['registration_sha256']!=sha(registration) or rel['predictions_sha256']!=sha(out/'PREDICTIONS.json') or
            rel['pre_audit_sha256']!=sha(pre_audit)):
        raise PermissionError('Separate root exact-hash target release required')
    reader=RoleReader();targets=[]
    for c in read(out/'PREPARED.json')['contexts']:
        values,invalid=reader.values(c,c['T'],'released_target7_frequency_14_scalar')
        targets.append({'context':c['context'],'values_Pa':serializable(values),
                        'target_admissible':not invalid,'invalid_reasons':invalid})
    if len(reader.access)!=252:raise ValueError('Exactly18 contexts times14 released target scalars required')
    write(out/'TARGET_ACCESS.json',reader.receipt())
    write(out/'TARGETS.json',{'actual_utc':utc(),'predictions_sha256':sha(out/'PREDICTIONS.json'),
        'registration_sha256':sha(registration),'pre_audit_sha256':sha(pre_audit),
        'release_sha256':sha(release),'contexts':targets,'new_wet_measurements':0})


def score(out,registration):
    p=read(out/'PREDICTIONS.json');targets=read(out/'TARGETS.json')
    if targets['predictions_sha256']!=sha(out/'PREDICTIONS.json'):raise ValueError('Frozen predictions differ')
    by_context={t['context']:t for t in targets['contexts']};rows=[];points=[];failures=[]
    for f in p['forecasts']:
        target=by_context[f['context']]
        meta={k:f[k] for k in ('context','condition_id','state_block','mask')}
        if f['status']!='OK' or not target['target_admissible']:
            failures.append({**meta,'prediction_status':f['status'],'target_admissible':target['target_admissible'],
                             'failure':f.get('failure'),'target_reasons':target['invalid_reasons']});continue
        scores,point=score_context(f,target['values_Pa'])
        rows.extend({**meta,**r} for r in scores);points.extend({**meta,**r} for r in point)
    macro=[]
    for mask in read(PLAN)['first_bounded_stage']['masks']:
        for method in METHODS:
            subset=[r for r in rows if r['mask']==mask and r['method']==method]
            macro.append({'mask':mask,'method':method,'planned_state_conditions':6,
                'scored_state_conditions':len(subset),'failed_or_unscored':6-len(subset),
                'successful_case_diagnostic_only':len(subset)!=6,
                **{k:float(np.mean([r[k] for r in subset])) if subset else None
                   for k in ('joint_nll_per_dim','log_crps','log_rmse','coverage90','log_width90')}})
    write(out/'SCORES.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
        'predictions_sha256':sha(out/'PREDICTIONS.json'),'targets_sha256':sha(out/'TARGETS.json'),
        'rows':rows,'point_rows':points,'macro_by_mask':macro,
        'all_contexts_denominator':18,'failed_or_unscored_contexts':len(failures),
        'failure_rate':len(failures)/18,'failures':failures,
        'complete_benchmark_admitted':not failures,'physical_material_count':1,
        'independent_specimen_count':None,'independent_material_confirmation':False,
        'original_resin_final20_accessed':False,'old_viscosity_prior_loaded':False})
    write(out/'SCORE_COMPLETE.json',{'actual_utc':utc(),'scores_sha256':sha(out/'SCORES.json')})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','predict','reveal','score'])
    parser.add_argument('--registration',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pre-audit');parser.add_argument('--release');args=parser.parse_args()
    authorize(args.registration,args.phase);out=Path(args.output)
    if args.phase=='prepare':out.mkdir(parents=True,exist_ok=False);prepare(out,args.registration)
    elif args.phase=='predict':predict(out,args.registration)
    elif args.phase=='reveal':reveal(out,args.registration,args.pre_audit,args.release)
    else:score(out,args.registration)


if __name__=='__main__':main()
