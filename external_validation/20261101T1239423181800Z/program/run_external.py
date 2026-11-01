"""Two-step external test: freeze forecasts, then reveal/score native targets once."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from external_models import predict,PROBABILISTIC,POINT
from external_scoring import score_one,aggregate

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
def now():return datetime.now(timezone.utc).isoformat()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','score']);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    protocol=read(a.protocol)
    for file,digest in protocol['bindings'].items():
        if sha(file)!=digest:raise ValueError('Protocol binding changed: '+file)
    if a.output.exists():raise FileExistsError('Never overwrite a completed phase')
    a.output.mkdir(parents=True)
    started=now();save(a.output/'START.json',{'actual_utc':started,'mode':a.mode,'protocol_sha256':sha(a.protocol)})
    curves=read(protocol['admitted_curves'])['curves']
    if not curves or len({c['curve_id'] for c in curves})!=len(curves):raise ValueError('Nonempty unique admitted curve IDs required')
    if a.mode=='freeze':
        model=read(protocol['model']);baselines=read(protocol['baselines']);records=[]
        for c in curves:
            gamma=np.asarray(c['gamma'],dtype=float);eta=np.asarray(c['eta'],dtype=float)
            limits=np.exp(np.asarray(model['q'])[[0,-1]])
            if gamma.ndim!=1 or eta.shape!=gamma.shape or len(gamma)<12 or not np.isfinite(gamma).all() or not np.isfinite(eta).all() or not np.all(np.diff(gamma)>0) or not np.all(eta>0) or np.any(gamma<limits[0]) or np.any(gamma>limits[1]):raise ValueError('Invalid admitted curve')
            input_payload={'observed_gamma':gamma[:6].tolist(),'observed_eta':eta[:6].tolist(),'target_gamma':gamma[6:].tolist()}
            # Model receives no target responses. Source inspection has already
            # exposed data; this separation does not make a blind confirmation.
            laws=predict(protocol['repo'],model,baselines,**{'anchors_gamma':gamma[:6],'anchors_eta':eta[:6],'target_gamma':gamma[6:]})
            records.append({'curve_id':c['curve_id'],'group_id':c['group_id'],'source_doi':c['source_doi'],
                'input':input_payload,'forecasts':laws})
        save(a.output/'PREDICTIONS.json',{'actual_utc':now(),'protocol_sha256':sha(a.protocol),
            'source_inspected_not_blind':True,'frozen_model_retraining_calls':0,
            'per_curve_anchor_OLS_point_estimates':len(curves),'records':records})
        save(a.output/'FROZEN.json',{'actual_utc':now(),'prediction_sha256':sha(a.output/'PREDICTIONS.json'),
            'curve_count':len(records),'methods':list(PROBABILISTIC+POINT),'target_response_scoring_performed':False})
    else:
        pred_path=Path(protocol['prediction_dir'])/'PREDICTIONS.json';frozen=read(Path(protocol['prediction_dir'])/'FROZEN.json')
        if sha(pred_path)!=frozen['prediction_sha256']:raise ValueError('Frozen predictions changed')
        lookup={c['curve_id']:c for c in curves};predictions=read(pred_path);rows=[]
        if len(predictions['records'])!=len(curves) or {r['curve_id'] for r in predictions['records']}!=set(lookup):raise ValueError('Frozen cohort differs from admitted cohort')
        for record in predictions['records']:
            curve=lookup[record['curve_id']]
            if record['source_doi']!=curve['source_doi'] or record['group_id']!=curve['group_id'] or set(record['forecasts'])!=set(PROBABILISTIC+POINT):raise ValueError('Frozen identity or method roster changed')
            expected={'observed_gamma':curve['gamma'][:6],'observed_eta':curve['eta'][:6],'target_gamma':curve['gamma'][6:]}
            if record['input']!=expected:raise ValueError('Frozen input differs from admitted source')
            for method,forecast in record['forecasts'].items():
                rows.append({'curve_id':record['curve_id'],'group_id':record['group_id'],'source_doi':record['source_doi'],
                             'method':method,'scores':score_one(forecast,curve['eta'][6:])})
        save(a.output/'SCORES.json',{'actual_utc':now(),'protocol_sha256':sha(a.protocol),
            'prediction_sha256':sha(pred_path),'score_rows':rows,'sources':aggregate(rows),
            'scope':'Source-inspected zero-refit external chemistry/operator stress; not compatible new-batch or independent calibration certification'})
    save(a.output/'COMPLETED.json',{'actual_utc':now(),'started_utc':started,'mode':a.mode,'status':'PASS'})
    print(json.dumps({'status':'PASS','mode':a.mode,'curves':len(curves),'output':str(a.output)}))

if __name__=='__main__':main()
