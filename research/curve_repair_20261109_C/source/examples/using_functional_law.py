"""Generate labeled synthetic log-modulus draws from an exact saved law.

No fitting, target reads, scoring, API access or observed-value replacement.
Draws are model outputs, never additional physical measurements.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--sha256',required=True)
    p.add_argument('--context',required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--count',type=int,default=32)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    raw=a.predictions.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=a.sha256:
        raise ValueError('Explicit saved-forecast SHA256 differs')
    if a.count<=0:
        raise ValueError('Positive sample count required')
    j=json.loads(raw)
    # The saved experiment's schema is discovered explicitly; no old
    # viscosity model or parameter table is imported as a substitute.
    contexts=j.get('contexts',j.get('forecasts',[]))
    chosen=[f for f in contexts if f.get('context',f.get('context_id'))==a.context]
    if len(chosen)!=1:
        raise ValueError('One exact saved context identity is required')
    f=chosen[0]
    payload=f.get('payload',f.get('prediction',f))
    if a.method not in payload.get('laws',{}):
        raise ValueError('Requested law unavailable; inspect recorded component failures')
    law=payload['laws'][a.method]
    rng=np.random.default_rng(a.seed)
    if law['distribution']=='multivariate_gaussian_log_modulus':
        m=np.array(law['mean']);matrix=np.array(law['covariance']);df=None
    elif law['distribution']=='multivariate_student4_log_modulus':
        m=np.array(law['location']);matrix=np.array(law['scale_matrix']);df=4
    else:
        raise ValueError('This example accepts only the declared functional laws')
    if m.shape!=(14,) or matrix.shape!=(14,14) or not np.isfinite(m).all() or not np.isfinite(matrix).all():
        raise ValueError('Complete finite 14-dimensional log-modulus law required')
    z=rng.standard_normal((a.count,14))@np.linalg.cholesky(matrix).T
    if df is not None:
        z/=np.sqrt(rng.chisquare(df,a.count)/df)[:,None]
    samples=m+z
    if not np.isfinite(samples).all():
        raise ArithmeticError('Unrepresentable generated values, no clipping')
    a.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.output/'GENERATED_LOG_MODULUS.npz',log_G_over_Pa=samples)
    meta={'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'kind':'GENERATED_MODEL_DRAWS_NOT_MEASUREMENTS','source_predictions_sha256':a.sha256,
          'context':a.context,'method':a.method,'count':a.count,'seed':a.seed,
          'draw_order':'Gprime seven targets, then Gdoubleprime same seven',
          'unit':'ln(G/(1Pa))','new_material_measurements':0,'new_fits':0,
          'new_target_reads':0,'generated_curves_guaranteed_Maxwell_passive':False,
          'log_Student_has_no_finite_arithmetic_Pa_expectation':df is not None}
    with (a.output/'GENERATED_METADATA.json').open('x',encoding='utf-8') as file:
        json.dump(meta,file,ensure_ascii=False,indent=2,allow_nan=False)


if __name__=='__main__':main()
