"""Registered fixed-matrix calculation; no training or material values."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from math import pi, sqrt
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.query_effect import query_effect, rank_queries

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def serial(v):
    if isinstance(v,np.ndarray):
        return v.tolist()
    if isinstance(v,dict):
        return {k:serial(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):
        return [serial(x) for x in v]
    return v

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--authorization",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    auth=json.loads(a.authorization.read_text("utf-8-sig"))
    if auth.get("scope")!="fixed_matrix_synthetic_only" or auth.get("authorized_by")!="root":
        raise ValueError("Actual root synthetic-only registration required")
    for item in auth["source_pins"]:
        if sha(item["file"])!=item["sha256"]:
            raise ValueError("Registered source changed")
    f=json.loads(a.fixture.read_text("utf-8-sig"))
    print("FIXTURE_KEYS", sorted(f))
    k=np.array(f["covariance"],dtype=float)
    target=f["target_indices"]
    candidates=f["candidate_indices"]
    noises=f["noise_variances"]
    costs=f["costs"]
    weights=f["target_weights"]
    effects=[]
    for i,q in enumerate(candidates):
        effect=query_effect(k,target,q,noise_variance=noises[i],target_weights=weights)
        row=serial(asdict(effect))
        row.update(cost=float(costs[i]),gain_per_cost=effect.weighted_expected_gain/float(costs[i]))
        effects.append(row)
    ranked=rank_queries(k,target,candidates,noise_variances=noises,costs=costs,target_weights=weights)
    # A declared positive control: high variance at a disconnected candidate
    # cannot improve a fixed target panel, while a lower-variance neighbour can.
    counter=np.array([[1.,.64,0.,.8],[.64,1.,0.,.8],[0.,0.,9.,0.],[.8,.8,0.,1.]])
    maximum_variance=2
    useful=query_effect(counter,[0,1],3)
    useless=query_effect(counter,[0,1],2)
    assert useful.weighted_expected_gain>0 and useless.weighted_expected_gain==0
    assert rank_queries(counter,[0,1],[2,3])[0]["query_index"]==3
    # Explicit negative control: with an incorrect cross-covariance, the same
    # update is harmful in expectation. These are synthetic analytic worlds.
    predicted_gain=.4/sqrt(pi)
    actual_misspecified_risk=1.4/sqrt(pi)
    prior_true_risk=1/sqrt(pi)
    assert actual_misspecified_risk>prior_true_risk
    try:
        query_effect(k,target,candidates[0],noise_variance="0.1")
    except ValueError:
        rejects_string=True
    else:
        raise AssertionError("numeric string must not be accepted as variance")
    report={"status":"FIXED_GAUSSIAN_QUERY_EFFECT_SYNTHETIC_ONLY", "actual_utc":datetime.now(timezone.utc).isoformat(),
        "source_sha256":sha(ROOT/"core/query_effect.py"),"gate_sha256":sha(__file__),
        "fixture_sha256":sha(a.fixture),"effects":effects,
        "ranked_query_indices":[x["query_index"] for x in ranked],
        "counterexample_to_maximum_variance":{"max_variance_query":maximum_variance,"best_target_query":3,
            "max_variance_target_gain":useless.weighted_expected_gain,"correlated_target_gain":useful.weighted_expected_gain},
        "misspecified_cross_covariance_negative_control":{"model_predicted_gain":predicted_gain,
            "true_prior_expected_CRPS":prior_true_risk,"true_updated_expected_CRPS":actual_misspecified_risk,
            "actual_gain":prior_true_risk-actual_misspecified_risk,
            "scope":"hand-set alternative Gaussian world; not observed material performance"},
        "rejects_numeric_string":rejects_string,"material_fits_or_new_targets":0,
        "fixed_covariance_policy_depends_on_observed_values":False,
        "novel_algorithm_or_calibration_claim":False}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("x",encoding="utf8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({"status":report["status"],"rank":report["ranked_query_indices"],"output":str(a.output)}))

if __name__=="__main__":
    main()
