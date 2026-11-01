"""Proper Gaussian marginal scores and honest group aggregation in log units."""
import math
import numpy as np
from scipy.special import ndtr,ndtri

def score_one(forecast,eta):
    truth=np.log(np.asarray(eta,dtype=float));mean=np.asarray(forecast['mean'])
    if mean.shape!=truth.shape or not np.isfinite(mean).all():raise ValueError('Forecast/target mismatch')
    error=mean-truth
    result={'n_targets':len(truth),'mse_log':float(np.mean(error**2)),
            'rmse_log':float(np.sqrt(np.mean(error**2))),'mae_log':float(np.mean(np.abs(error)))}
    if 'covariance' not in forecast:return result
    covariance=np.asarray(forecast['covariance']);variance=np.diag(covariance)
    if not (variance>0).all():raise ValueError('Nonpositive forecast variance')
    sd=np.sqrt(variance);z=(truth-mean)/sd
    crps=sd*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    half=ndtri(.95)*sd;low=mean-half;high=mean+half
    covered=(truth>=low)&(truth<=high)
    interval=(high-low)+20*np.maximum(low-truth,0)+20*np.maximum(truth-high,0)
    result.update(crps_log=float(np.mean(crps)),coverage90=float(np.mean(covered)),
        whole_curve_covered90=bool(np.all(covered)),width90_log=float(np.mean(high-low)),
        interval_score90_log=float(np.mean(interval)),
        marginal_nll_log=float(np.mean(.5*np.log(2*np.pi*variance)+.5*z*z)),
        max_abs_standardized_error=float(np.max(np.abs(z))))
    return result

def aggregate(rows):
    """One equal vote per formulation group, averaging its native replicate runs."""
    sources=sorted(set(r['source_doi'] for r in rows));answer={}
    for source in sources:
        part=[r for r in rows if r['source_doi']==source];groups=sorted(set(r['group_id'] for r in part))
        methods=sorted(set(r['method'] for r in part));summaries={}
        for method in methods:
            subset=[r for r in part if r['method']==method]
            keys=sorted(set(k for r in subset for k in r['scores'])-{'n_targets','rmse_log'})
            group_values=[]
            for group in groups:
                runs=[r['scores'] for r in subset if r['group_id']==group]
                if not runs:raise ValueError('Missing method/group results')
                group_values.append({'group_id':group,'runs':len(runs),
                    **{key:float(np.mean([r[key] for r in runs])) for key in keys}})
            summary={key:float(np.mean([g[key] for g in group_values])) for key in keys}
            summary['root_group_macro_mse_log']=math.sqrt(summary['mse_log'])
            summary['groups']=group_values;summaries[method]=summary
        answer[source]={'formulation_groups':len(groups),'native_curves':len(part)//len(methods),
            'weighting':'equal formulation groups; equal native replicate curves within a group; equal target positions within a curve',
            'methods':summaries}
    return answer
