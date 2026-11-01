"""Small pre-execution mathematical contract checks; no material files or models."""
import json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from external_scoring import score_one,aggregate

checks=[]
def check(name,truth):
    checks.append({'name':name,'passed':bool(truth)})
    if not truth:raise AssertionError(name)
normal=score_one({'mean':[0.],'covariance':[[1.]]},[1.])
check('standard_normal_at_mean_CRPS',abs(normal['crps_log']-(math.sqrt(2)-1)/math.sqrt(math.pi))<1e-15)
check('normal_marginal_density',abs(normal['marginal_nll_log']-.5*math.log(2*math.pi))<1e-15)
check('inside_marginal_interval',normal['coverage90']==1 and normal['whole_curve_covered90'])
point=score_one({'mean':[0.]},[math.e])
check('point_control_has_no_probability_scores','crps_log' not in point and 'coverage90' not in point and abs(point['mse_log']-1)<1e-15)
rows=[{'source_doi':'toy','group_id':'a','method':'p','scores':{'n_targets':1,'mse_log':1.,'rmse_log':1.,'mae_log':1.}},
      {'source_doi':'toy','group_id':'a','method':'p','scores':{'n_targets':9,'mse_log':9.,'rmse_log':3.,'mae_log':3.}},
      {'source_doi':'toy','group_id':'b','method':'p','scores':{'n_targets':5,'mse_log':25.,'rmse_log':5.,'mae_log':5.}}]
out=aggregate(rows)['toy']['methods']['p']
check('equal_groups_then_equal_runs',out['mse_log']==15.)
check('root_macro_mse_not_average_rmse',out['root_group_macro_mse_log']==math.sqrt(15.))
result={'actual_utc':datetime.now(timezone.utc).isoformat(),'role':'pre-execution toy mathematical checks; no material score',
        'checks':checks,'status':'PASS','material_files_accessed':0,'model_fits':0}
with (Path(__file__).parent/'PREEXECUTION_SCORE_CONTRACT.json').open('x',encoding='utf-8') as handle:json.dump(result,handle,indent=2)
print(json.dumps(result))
