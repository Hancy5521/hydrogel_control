"""Independent once-frozen H3 score audit, after explicit root target release.

Only already released TARGETS.json is read. There is no CSV reader, fitting,
gate change, selected endpoint or parameter search. Presence is not authority.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from resin_independent_math import (Comparisons,TARGET,actual_utc,load,
                                    score_gaussian,sha,write_new)

METRICS=("crps","joint_nll_per_dimension","log_rmse","coverage90","width90")

def audit(args):
    pilot,targets,scores,release,pre=(load(path) for path in
        (args.predictions,args.targets,args.scores,args.release,args.pre_audit))
    prediction_sha,target_sha=sha(args.predictions),sha(args.targets)
    if pre.get("status")!="PASS" or pre.get("audited_early_predictions_sha256")!=prediction_sha:
        raise PermissionError("Exact independent H3 pre-Y PASS required")
    if (release.get("authorization")!="SCORE_FROZEN_EARLY_UPDATE_ONLY"
        or release.get("predictions_sha256")!=prediction_sha
        or release.get("targets_sha256")!=target_sha):
        raise PermissionError("Exact root H3 score release required")
    if (scores["predictions_sha256"]!=prediction_sha or scores["targets_sha256"]!=target_sha
        or scores["release_sha256"]!=sha(args.release)
        or pilot["main_predictions_sha256"]!=targets["predictions_sha256"]):
        raise AssertionError("H3 score/release/main-target byte chain changed")
    truth={row["member"]:row for row in targets["curves"]}
    if len(targets["curves"])!=15 or len(truth)!=15 or any(row["target_indices"]!=TARGET for row in truth.values()):
        raise AssertionError("Exact common120dev target panel required")
    forecasts={(row["member"],row["method_id"]):row for row in pilot["forecasts"]}
    actual={(row["member"],row["method_id"]):row for row in scores["curve_scores"]}
    if len(pilot["forecasts"])!=45 or len(actual)!=45 or set(actual)!=set(forecasts):
        raise AssertionError("Exact45 early-pilot score rows required")
    comparisons=Comparisons()
    expected={}
    for key,row in forecasts.items():
        if row["target_indices"]!=TARGET or row["recipe"]!=truth[row["member"]]["recipe"]:
            raise AssertionError("H3 forecast group/target panel changed")
        metric=score_gaussian(row["mean"],row["covariance"],truth[row["member"]]["values"])
        metric["joint_nll_per_dimension"]=metric.pop("joint_nll_per_target")
        expected[key]=metric
        for name in METRICS:
            comparisons.close("curve_"+name,actual[key][name],metric[name],atol=3e-8,rtol=3e-9)
    recipes=sorted({row["recipe"] for row in truth.values()})
    recipe_rows={(row["recipe"],row["method_id"]):row for row in scores["recipe_scores"]}
    if len(scores["recipe_scores"])!=9 or len(recipe_rows)!=9 or len(recipes)!=3:
        raise AssertionError("Exact9 recipe-method rows required")
    independent=[]
    for recipe in recipes:
        for method in ("base","candidate","gated"):
            selected=[metric for (member,name),metric in expected.items()
                      if name==method and truth[member]["recipe"]==recipe]
            if len(selected)!=5:
                raise AssertionError("All five native runs must remain in recipe")
            row={"recipe":recipe,"method_id":method,"runs":5}
            for metric in METRICS:
                row[metric]=float(np.mean([s[metric] for s in selected]))
                comparisons.close("recipe_"+metric,recipe_rows[recipe,method][metric],row[metric],atol=3e-8,rtol=3e-9)
            independent.append(row)
    by={(row["recipe"],row["method_id"]):row for row in independent}
    per_recipe=[]
    sentinel={row["recipe"] for row in truth.values() if row["thermal_history"]=="descending_35_last"}
    if len(sentinel)!=1:
        raise AssertionError("Exact single sentinel recipe required")
    for recipe in recipes:
        per_recipe.append({"recipe":recipe,"is_sentinel_development":recipe in sentinel,
            "candidate_minus_base_CRPS":by[recipe,"candidate"]["crps"]-by[recipe,"base"]["crps"],
            "gated_minus_base_CRPS":by[recipe,"gated"]["crps"]-by[recipe,"base"]["crps"]})
    if scores["gate_accept"]!=pilot["gate_accept"] or scores["early_report"]!=pilot["early_report"]:
        raise AssertionError("Early gate/report modified after target reveal")
    return {"status":"PASS","actual_utc":actual_utc(),
        "audited_scores_sha256":sha(args.scores),"audited_early_predictions_sha256":prediction_sha,
        "audited_targets_sha256":target_sha,"root_release_sha256":sha(args.release),
        "pre_y_audit_sha256":sha(args.pre_audit),"auditor_source_sha256":sha(__file__),
        "auditor_math_sha256":sha(Path(__file__).with_name("resin_independent_math.py")),
        "comparisons":comparisons.groups,"independent_recipe_scores":independent,
        "independent_per_recipe_CRPS_change":per_recipe,
        "curve_scores":45,"common_target_cells":120,"CRPS_terms":360,
        "parameter_updates_in_audit":0,"raw_source_opened":False,"SUT_imported":False,
        "scope":"one predeclared anchor-only parameter update; sentinel-tail is development, not independent confirmation"}

def main():
    parser=argparse.ArgumentParser()
    for option in ("predictions","targets","scores","release","pre-audit","output"):
        parser.add_argument("--"+option,required=True)
    args=parser.parse_args()
    try:
        report=audit(args)
    except Exception as error:
        write_new(args.output,{"status":"FAIL","actual_utc":actual_utc(),
            "error_type":type(error).__name__,"error":str(error)})
        raise
    write_new(args.output,report)
    print("Independent early-update score audit PASS;45laws,9recipe-method rows; no new model update.")

if __name__=="__main__":
    main()
