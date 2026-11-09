"""Read-only arithmetic diagnosis of the already-revealed eighteen-context pilot.

No refitting, new predictions, parameter search, source-G decoding, or new score
protocol. Exact mean-change decomposition explains the saved Gaussian NLL.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import block_diag

HERE=Path(__file__).resolve().parent;OUT=HERE/'pilot_once'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
now=dt.datetime.now(dt.timezone.utc)
write(OUT/'SAVED_DIAGNOSIS_REGISTRATION.json',{'actual_utc':now.isoformat(),
    'asia_shanghai':now.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
    'source_sha256':sha(__file__),'inputs':{name:sha(OUT/name) for name in ['PREDICTIONS.json','TARGETS.json','SCORES.json','INDEPENDENT_POST_Y_V2.json']},
    'role':'Read-only interpretation of saved, already revealed pilot, not new independent confirmation',
    'no_refit_no_search_no_new_raw_reads':True,'fixed_alpha':.5})
p=read(OUT/'PREDICTIONS.json');s=read(OUT/'SCORES.json');y={r['context']:r for r in read(OUT/'TARGETS.json')['contexts']}
scores={(r['context'],r['method']):r for r in s['rows']}
rows=[]
for f in p['forecasts']:
    m=np.array(f['gp_mean']);C=np.array(f['full_residual_covariance']);d=np.array(f['maxwell_log_location'])-m
    truth=np.log(np.array(y[f['context']]['values_Pa']).reshape(14));e=m-truth
    a=float(d@np.linalg.solve(C,d)/14);b=float(d@np.linalg.solve(C,e)/14)
    nll_change=.125*a+.5*b
    direct=scores[(f['context'],'half_maxwell_mean_gaussian')]['joint_nll_per_dim']-scores[(f['context'],'full_anchor_gaussian')]['joint_nll_per_dim']
    if abs(nll_change-direct)>2e-8*(1+abs(direct)):raise AssertionError('Mean-only exact quadratic differs')
    eig=np.linalg.eigvalsh(C)
    row={'context':f['context'],'mask':f['mask'],'state_block':f['state_block'],
         'mean_only_Gaussian_NLL_change_per_dim':nll_change,'saved_direct_NLL_change':direct,
         'quadratic_nonnegative_penalty':.125*a,'linear_alignment_term':.5*b,
         'Euclidean_mean_square_change':float(np.mean((e+.5*d)**2)-np.mean(e*e)),
         'covariance_smallest_eigenvalue':float(eig[0]),'covariance_largest_eigenvalue':float(eig[-1]),
         'covariance_condition_number':float(eig[-1]/eig[0]),
         'mean_GP_sd':float(np.mean(np.sqrt(np.diag(C)))),
         'channel_effects':[]}
    for ch,name in enumerate(['Gprime','Gdoubleprime']):
        sl=slice(7*ch,7*(ch+1));cc=C[sl,sl];dd=d[sl];ee=e[sl]
        row['channel_effects'].append({'channel':name,
            'Gaussian_NLL_change_per_dim':float((.125*dd@np.linalg.solve(cc,dd)+.5*dd@np.linalg.solve(cc,ee))/7),
            'Euclidean_mean_square_change':float(np.mean((ee+.5*dd)**2)-np.mean(ee*ee)),
            'teacher_mean_abs_displacement':float(np.mean(np.abs(dd)))})
    rows.append(row)
summaries=[]
for mask in sorted({r['mask'] for r in rows}):
    sub=[r for r in rows if r['mask']==mask]
    gp={r['context']:r for r in s['rows'] if r['mask']==mask and r['method']=='full_anchor_gaussian'}
    half={r['context']:r for r in s['rows'] if r['mask']==mask and r['method']=='half_maxwell_mean_gaussian'}
    summaries.append({'mask':mask,'contexts':len(sub),'new_independent_materials':0,
        'crps_improved_states':[k for k in gp if half[k]['log_crps']<gp[k]['log_crps']],
        'crps_worsened_states':[k for k in gp if half[k]['log_crps']>gp[k]['log_crps']],
        'joint_NLL_improved_states':[k for k in gp if half[k]['joint_nll_per_dim']<gp[k]['joint_nll_per_dim']],
        'joint_NLL_worsened_states':[k for k in gp if half[k]['joint_nll_per_dim']>gp[k]['joint_nll_per_dim']],
        'mean_quadratic_penalty':float(np.mean([r['quadratic_nonnegative_penalty'] for r in sub])),
        'mean_linear_alignment_term':float(np.mean([r['linear_alignment_term'] for r in sub])),
        'mean_NLL_change':float(np.mean([r['mean_only_Gaussian_NLL_change_per_dim'] for r in sub])),
        'mean_Euclidean_MSE_change':float(np.mean([r['Euclidean_mean_square_change'] for r in sub]))})
result={'actual_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'role':'saved-output diagnosis only',
    'fixed_method_not_reselected':True,'fixed_alpha':.5,'GP_covariance_not_modified':True,
    'formula':'Delta NLL/d = (alpha^2/2)*(d^T C^-1 d/dimension) + alpha*(d^T C^-1(mG-y)/dimension)',
    'rows':rows,'geometry_summaries':summaries,'saved_macro_by_mask':s['macro_by_mask'],
    'interpretation':'Marginal Euclidean/CRPS improvement and joint Mahalanobis improvement are different claims. Componentwise successful optimization is not extrapolation calibration or physical-spectrum identification.'}
write(OUT/'SAVED_RESULT_DIAGNOSIS.json',result)
print(json.dumps({'output':str(OUT/'SAVED_RESULT_DIAGNOSIS.json'),'sha256':sha(OUT/'SAVED_RESULT_DIAGNOSIS.json'),'geometry_summaries':summaries},ensure_ascii=False))
