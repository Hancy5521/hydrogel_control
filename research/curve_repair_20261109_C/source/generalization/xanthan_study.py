"""Registered, context-isolated cross-source masked curve benchmark.

prepare: only input metadata/coordinates and frozen reference parameters.
predict: each context receives exactly its own six log-viscosity observations.
score: all eight response targets are read only after every law is frozen and
an independent pre-score audit has passed. Different predefined masks may
reuse a measured cell in different roles: this is a retrospective masking
benchmark, not an assertion of global target secrecy or new wet experiments.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from conditional_laws import build_laws, score_context, METHODS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'sources/generalization_admission_20261109/polymer_flow/candidate2_xanthan'
WINDOWS = {'overlap_resin_1_to_50': (1., 50.),
           'full_reference_support': (0.012787740677595104, 754.5143432617188)}
MASKS = {'near_tail': ([0,1,13,14,15,16], list(range(17,25))),
         'prefix_tail': ([0,1,2,3,4,5], list(range(17,25))),
         'interior_hole': ([0,4,6,17,20,24], list(range(8,16)))}
SOURCE_SHA = 'c50696f0776fba1a721e6a2804480b2d2aa900e232f3e93eacbfd35403c904a4'
INPUT_SHA = '0ac7b5f64663d6bf89ad80ed1602ec26df665f036084a3201e6f87ce1edd0c92'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def authorize(path):
    reg = read(path)
    if reg['schema'] != 'xanthan-cross-source-fixed-context-v1':
        raise PermissionError('Separate actual-UTC benchmark registration required')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'])
    if stamp.tzinfo is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Nonfuture actual UTC required')
    for p, digest in reg['pins'].items():
        if sha(p) != digest:
            raise ValueError('Pinned benchmark bytes changed: '+p)
    if list(METHODS) != reg['methods'] or reg['own_scalar_budget'] != 6 or reg['target_dimension'] != 8:
        raise ValueError('Frozen same-budget benchmark specification differs')
    if sha(SOURCE/'INPUT_ONLY.json') != INPUT_SHA or sha(SOURCE/'Rheological_data.zip') != SOURCE_SHA:
        raise ValueError('Original source/input bytes changed')
    if reg['numeric_eta_before_registration'] != 0:
        raise PermissionError('This protocol requires a source unconsumed by this project before registration')
    return reg


def frozen_adapter():
    path = ROOT/'material_stress/frozen_grid_reference.py'
    if sha(path) != 'd0c77cbc3a09b776c5a870d1a770b9afe2a4ea8b726dddfa3394e924fbc011bc':
        raise ValueError('Frozen reference adapter changed')
    name = '_xanthan_frozen_adapter'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        spec.loader.exec_module(module)
    return sys.modules[name]


def prepare(out, registration, checkpoint_root=None):
    data=read(SOURCE/'INPUT_ONLY.json')
    curves=[]
    adapter=frozen_adapter()
    for g in data['groups']:
        coordinates=g['coordinates']
        native=np.array([c['gamma_per_s'] for c in coordinates])
        if (np.diff(native)<=0).any() or (native<=0).any():
            raise ValueError('Positive strictly increasing native coordinates required')
        for window,(low,high) in WINDOWS.items():
            eligible=np.flatnonzero((native>=low)&(native<=high))
            if len(eligible)<25:
                raise ValueError('Insufficient native nodes, no interpolation/replacement')
            ranks=np.floor(np.linspace(0,len(eligible)-1,25)+.5).astype(int)
            nodes=[coordinates[int(eligible[r])] for r in ranks]
            x=np.log([n['gamma_per_s'] for n in nodes])
            options={} if checkpoint_root is None else {'checkpoint_root':Path(checkpoint_root)}
            bundle=adapter.build_frozen_grid_reference(x, **options)
            mean,cov=bundle.reference.joint(bundle.theta_gp)
            curves.append({'preparation_group':g['preparation_group'],
                'salinity_M':g['NaCl_molarity_M'],'preparation_order':g['preparation_order'],
                'member':g['member'],'member_sha256':g['source_member_sha256'],
                'window':window,'nodes':nodes,'x':x.tolist(),
                'frozen_mean':mean.tolist(),'frozen_covariance':cov.tolist(),
                'observation_unit':'Pa.s','temperature_C':21.0})
    if len(curves)!=22 or len({c['preparation_group'] for c in curves})!=11:
        raise ValueError('All eleven preparation conditions and both coordinate windows required')
    write(out/'PREPARED.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
        'curves':curves,'masks':{k:{'O':o,'T':t} for k,(o,t) in MASKS.items()},
        'source_numeric_eta_accessed':0,'methods':list(METHODS),
        'reference_theta_updates':0,'new_fits':0})


class RoleReader:
    """CSV oracle with an explicit native-row whitelist for one call only."""
    def __init__(self):
        self.zip=zipfile.ZipFile(SOURCE/'Rheological_data.zip')
        self.access=[]
    def log_values(self,curve,indices,role,context):
        nodes=curve['nodes']
        wanted={int(nodes[i]['source_line_1based']):i for i in indices}
        if len(wanted)!=len(indices):
            raise ValueError('Distinct original row addresses required')
        raw=self.zip.read(curve['member'])
        if hashlib.sha256(raw).hexdigest()!=curve['member_sha256']:
            raise ValueError('Native measurement file changed')
        values={}
        for line,row in enumerate(csv.reader(io.StringIO(raw.decode('utf-8-sig'))),1):
            if line==1 and row!=['Var1','Var2','Var3','Var4']:
                raise ValueError('Raw field order changed')
            if line==2 and row[2]!='viscosity [Pa s]':
                raise ValueError('Raw viscosity unit changed, no resin cP conversion allowed')
            if line not in wanted:
                continue
            i=wanted[line]
            gamma=float(row[0])
            if gamma!=nodes[i]['gamma_per_s'] or row[0]!=nodes[i]['gamma_raw_token']:
                raise ValueError('Exact original input/row mapping differs')
            # Only column2 of whitelisted rows is converted to response floats.
            eta=float(row[2])
            if not np.isfinite(eta) or eta<=0:
                raise ValueError('Nonpositive/unrepresentable selected viscosity; stop without floor or deletion')
            values[i]=float(np.log(eta))
            self.access.append({'context':context,'role':role,'member':curve['member'],
                                'line':line,'column':2,'unit':'Pa.s'})
        if set(values)!=set(indices):
            raise ValueError('Missing whitelisted measurement row')
        return np.array([values[i] for i in indices])
    def receipt(self):
        return {'actual_utc':utc(),'numeric_access_calls':len(self.access),
            'unique_response_cells':len({(r['member'],r['line'],r['column']) for r in self.access}),
            'access_sites':self.access,'stress_or_torque_numeric_accessed':0,
            'scope':'All values read from public measured source; context O/T isolation, not global secrecy'}
    def close(self):
        self.zip.close()


def condition_gaussian(m,k,o,y,t):
    m,k=np.asarray(m),np.asarray(k)
    koo=k[np.ix_(o,o)];kto=k[np.ix_(t,o)]
    mean=m[t]+kto@np.linalg.solve(koo,y-m[o])
    cov=k[np.ix_(t,t)]-kto@np.linalg.solve(koo,kto.T)
    cov=(cov+cov.T)/2
    np.linalg.cholesky(cov)
    return {'mean':mean.tolist(),'covariance':cov.tolist()}


def predict(out,registration):
    prepared=read(out/'PREPARED.json')
    if prepared['registration_sha256']!=sha(registration):
        raise ValueError('Prepared laws belong to another protocol')
    reader=RoleReader();forecasts=[]
    try:
        for curve in prepared['curves']:
            x=np.array(curve['x'])
            for mask,(o,tidx) in MASKS.items():
                context=curve['preparation_group']+'|'+curve['window']+'|'+mask
                y=reader.log_values(curve,o,'observed_own6',context)
                frozen=condition_gaussian(curve['frozen_mean'],curve['frozen_covariance'],o,y,tidx)
                payload=build_laws(x[o],y,x[tidx],frozen)
                forecasts.append({'context':context,'preparation_group':curve['preparation_group'],
                    'salinity_M':curve['salinity_M'],'preparation_order':curve['preparation_order'],
                    'window':curve['window'],'mask':mask,'member':curve['member'],
                    'O':o,'T':tidx,'x_observed':x[o].tolist(),'y_observed':y.tolist(),
                    'x_target':x[tidx].tolist(),**payload})
                with (out/'CONTEXT_LAWS.jsonl').open('a',encoding='utf-8',newline='\n') as log:
                    log.write(json.dumps({'actual_utc':utc(),**forecasts[-1]},allow_nan=False)+'\n')
    finally:
        write(out/'ANCHOR_ACCESS.json',reader.receipt());reader.close()
    if len(forecasts)!=66 or len(reader.access)!=396:
        raise ValueError('Exactly 66 six-observation contexts required')
    write(out/'PREDICTIONS.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
        'prepared_sha256':sha(out/'PREPARED.json'),'forecasts':forecasts,
        'new_physical_fits':66,'new_anchor_gp_fits':66,'covariance_fits_from_target_labels':0,
        'laws':594,'unique_original_curves':11,'unique_salinity_conditions':7,
        'context_target_read_during_own_prediction':False,
        'different_masks_can_reuse_source_cells_in_other_roles':True})
    write(out/'PREDICTION_COMPLETE.json',{'actual_utc':utc(),'predictions_sha256':sha(out/'PREDICTIONS.json')})


def score(out,registration,audit):
    prediction=read(out/'PREDICTIONS.json')
    receipt=read(out/'PREDICTION_COMPLETE.json')
    audit_receipt=read(audit)
    if (receipt['predictions_sha256']!=sha(out/'PREDICTIONS.json') or
        audit_receipt['status']!='PASS' or audit_receipt['predictions_sha256']!=sha(out/'PREDICTIONS.json')):
        raise PermissionError('Frozen predictions and independent pre-score law/source audit PASS required')
    if prediction['registration_sha256']!=sha(registration):
        raise ValueError('Registration changed')
    curves={(c['preparation_group'],c['window']):c for c in read(out/'PREPARED.json')['curves']}
    reader=RoleReader();rows=[];target_rows=[]
    try:
        for f in prediction['forecasts']:
            c=curves[(f['preparation_group'],f['window'])]
            y=reader.log_values(c,f['T'],'score_target_own8',f['context'])
            target_rows.append({'context':f['context'],'values':y.tolist()})
            for row in score_context(f['laws'],y):
                rows.append({**{k:f[k] for k in ('context','preparation_group','salinity_M','preparation_order','window','mask')},**row})
    finally:
        write(out/'TARGET_ACCESS.json',reader.receipt());reader.close()
    write(out/'TARGETS.json',{'actual_utc':utc(),'predictions_sha256':sha(out/'PREDICTIONS.json'),
        'independent_pre_score_sha256':sha(audit),'contexts':target_rows,
        'unique_original_curves':11,'new_wet_measurements':0})
    metrics=('joint_nll_per_dim','crps','log_rmse','coverage90','width90')
    by_salinity=[];macro=[]
    for window in WINDOWS:
        for mask in MASKS:
            for method in METHODS:
                for salinity in sorted({r['salinity_M'] for r in rows}):
                    subset=[r for r in rows if (r['window'],r['mask'],r['method'],r['salinity_M'])==(window,mask,method,salinity)]
                    by_salinity.append({'window':window,'mask':mask,'method':method,'salinity_M':salinity,
                        'preparation_conditions':len(subset),
                        **{k:float(np.mean([r[k] for r in subset])) for k in metrics}})
                subset=[r for r in by_salinity if (r['window'],r['mask'],r['method'])==(window,mask,method)]
                macro.append({'window':window,'mask':mask,'method':method,'salinity_groups':7,
                    **{k:float(np.mean([r[k] for r in subset])) for k in metrics}})
    write(out/'SCORES.json',{'actual_utc':utc(),'registration_sha256':sha(registration),
        'predictions_sha256':sha(out/'PREDICTIONS.json'),'targets_sha256':sha(out/'TARGETS.json'),
        'rows':rows,'by_salinity':by_salinity,'macro':macro,
        'not_independent_sample_count':66,'original_preparation_conditions':11,'salinity_levels':7,
        'new_laboratory_batch_confirmation':False,'original_resin_final20_accessed':False})
    write(out/'SCORE_COMPLETE.json',{'actual_utc':utc(),'scores_sha256':sha(out/'SCORES.json')})


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','predict','score'])
    p.add_argument('--registration',required=True);p.add_argument('--output',required=True);p.add_argument('--audit')
    p.add_argument('--checkpoint-root')
    args=p.parse_args();authorize(args.registration);out=Path(args.output)
    if args.phase=='prepare':out.mkdir(parents=True,exist_ok=False)
    write(out/('STARTED_'+args.phase+'.json'),{'actual_utc':utc(),'registration_sha256':sha(args.registration),
          'argv':sys.argv,'phase':args.phase})
    try:
        if args.phase=='score':score(out,args.registration,args.audit)
        elif args.phase=='prepare':prepare(out,args.registration,args.checkpoint_root)
        else:predict(out,args.registration)
    except BaseException as error:
        write(out/('FAILED_'+args.phase+'.json'),{'actual_utc':utc(),'type':type(error).__name__,
            'message':str(error),'decision':'Preserve partial context laws/access; no automatic refit/restart or substitution'})
        raise


if __name__=='__main__':main()
