"""Summarize immutable saved development results; never fit or rescore.

Reads only previously saved public-development prediction/score artifacts.
CRPS, NLL, RMSE, intervals and coverage below are copied/averaged from existing
scores. New arithmetic is descriptive parameter/covariance/bias decomposition.
No SUT import, raw source file, query selection, rho grid or future eta exists.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

BASE=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
FILES={
    "prepare":BASE/"resin_dev_once/prepare/PREPARED.json",
    "universal_predictions":BASE/"universal_dev_once/predict/PREDICTIONS.json",
    "universal_scores":BASE/"universal_dev_once/score/SCORES.json",
    "physical_predictions":BASE/"student_dev_once/predict/PREDICTIONS.json",
    "physical_scores":BASE/"student_dev_once/score/SCORES.json",
    "gaussian_scores":BASE/"resin_dev_once/score/SCORES.json",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(path,payload):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def stats(values):
    a=np.asarray(values,float)
    return {"mean":float(np.mean(a)),"min":float(np.min(a)),"max":float(np.max(a))}

def main():
    now=dt.datetime.now(dt.timezone.utc)
    write_new(HERE/"TRANSFER_FAILURE_SUMMARY_REGISTERED.json",{
        "actual_utc":now.isoformat(),"asia_shanghai":now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_sha256":sha(__file__),"inputs":{key:{"path":str(path),"sha256":sha(path)} for key,path in FILES.items()},
        "task":"descriptive analysis of saved development artifacts, not new forecast/score/fit",
        "prior_access":"root and auditor already read saved aggregate macros; no claim of outcome blindness",
        "forbidden":["fit","score recomputation","rho grid","new masks","raw source eta","future35run eta"]})
    data={key:load(path) for key,path in FILES.items()}
    curves={row["member"]:row for row in data["prepare"]["curves"]}
    up=data["universal_predictions"]["forecasts"]
    physical={ (row["member"],row["mask_method_id"]):row
        for row in data["physical_predictions"]["forecasts"] if row["baseline_kind"]=="power_law"}
    points={}
    for row in data["universal_scores"]["point_terms"]:
        points.setdefault((row["member"],row["method_id"]),[]).append(row)
    pp={}
    for row in data["physical_scores"]["point_terms"]:
        if row["baseline_kind"]=="power_law":
            pp.setdefault((row["member"],row["mask_method_id"]),[]).append(row)
    score_lookup={(row["member"],row["method_id"]):row for row in data["universal_scores"]["curve_scores"]}
    pscore_lookup={(row["member"],row["mask_method_id"]):row for row in data["physical_scores"]["curve_scores"] if row["baseline_kind"]=="power_law"}
    curve_rows=[]
    for law in up:
        key=(law["member"],law["method_id"])
        original=(law["member"],law["mask_method_id"])
        p=physical[original]
        rows=sorted(points[key],key=lambda r:r["target_index"])
        p_rows=sorted(pp[original],key=lambda r:r["target_index"])
        if [r["truth"] for r in rows]!=[r["truth"] for r in p_rows]:
            raise AssertionError("Compared methods have different saved target cells")
        target=np.column_stack((np.ones(8),np.array(curves[law["member"]]["log_gamma"])[law["target_indices"]]))
        trend=target@np.array(law["beta"])
        location=np.array(law["location"])
        truth=np.array([row["truth"] for row in rows])
        conditional=np.diag(law["conditional_reference_covariance"])
        uncertainty=np.diag(law["trend_uncertainty_covariance"])
        total=conditional+uncertainty
        scale=law["sse"]/4
        reconstruction=scale*(np.array(law["conditional_reference_covariance"])+np.array(law["trend_uncertainty_covariance"]))
        row={"member":law["member"],"recipe":law["recipe"],"thermal_history":law["thermal_history"],
            "mask_method_id":law["mask_method_id"],"covariance_id":law["covariance_id"],
            "beta_intercept":law["beta"][0],"beta_log_gamma_slope":law["beta"][1],
            "whitened_sse":law["sse"],"whitened_scale_SSE_over4":scale,
            "physical_OLS_intercept":p["beta"][0],"physical_OLS_slope":p["beta"][1],
            "physical_OLS_SSE":p["sse"],"physical_OLS_scale_SSE_over4":p["sse"]/4,
            "mean_C_diagonal":float(np.mean(conditional)),"mean_U_diagonal":float(np.mean(uncertainty)),
            "trend_uncertainty_fraction":float(np.sum(uncertainty)/np.sum(total)),
            "mean_shape_diagonal":float(np.mean(np.diag(law["scale_matrix"]))),
            "mean_C_scaled_diagonal":float(scale*np.mean(conditional)),
            "mean_U_scaled_diagonal":float(scale*np.mean(uncertainty)),
            "mean_location_minus_truth":float(np.mean(location-truth)),
            "mean_OLS_location_minus_truth":float(np.mean(np.array(p["location"])-truth)),
            "mean_affine_trend_minus_truth":float(np.mean(trend-truth)),
            "mean_correlated_residual_shift":float(np.mean(location-trend)),
            "mean_abs_correlated_residual_shift":float(np.mean(np.abs(location-trend))),
            "max_scale_reconstruction_error":float(np.max(np.abs(reconstruction-np.array(law["scale_matrix"])))),
            "saved_metrics":{name:score_lookup[key][name] for name in
                ("crps","joint_nll_per_dimension","log_rmse","coverage90","width90")},
            "saved_OLS_metrics":{name:pscore_lookup[original][name] for name in
                ("crps","joint_nll_per_dimension","log_rmse","coverage90","width90")}}
        curve_rows.append(row)
    fields=[key for key,value in curve_rows[0].items() if isinstance(value,(int,float))]
    recipes=sorted({row["recipe"] for row in curve_rows})
    masks=sorted({row["mask_method_id"] for row in curve_rows})
    refs=sorted({row["covariance_id"] for row in curve_rows})
    by_recipe=[]
    by_macro=[]
    for mask in masks:
        for ref in refs:
            macro=[]
            for recipe in recipes:
                selected=[r for r in curve_rows if r["mask_method_id"]==mask and r["covariance_id"]==ref and r["recipe"]==recipe]
                if len(selected)!=5: raise AssertionError("All five run summaries required")
                row={"recipe":recipe,"mask_method_id":mask,"covariance_id":ref,"runs":5,
                    "thermal_history":selected[0]["thermal_history"],
                    "summaries":{field:stats([s[field] for s in selected]) for field in fields},
                    "saved_metrics_mean":{name:float(np.mean([s["saved_metrics"][name] for s in selected])) for name in selected[0]["saved_metrics"]},
                    "saved_OLS_metrics_mean":{name:float(np.mean([s["saved_OLS_metrics"][name] for s in selected])) for name in selected[0]["saved_OLS_metrics"]}}
                by_recipe.append(row);macro.append(row)
            by_macro.append({"mask_method_id":mask,"covariance_id":ref,"recipes":3,
                "equal_recipe_means":{f:float(np.mean([r["summaries"][f]["mean"] for r in macro])) for f in fields},
                "saved_metrics_macro":{name:float(np.mean([r["saved_metrics_mean"][name] for r in macro])) for name in macro[0]["saved_metrics_mean"]},
                "saved_OLS_metrics_macro":{name:float(np.mean([r["saved_OLS_metrics_mean"][name] for r in macro])) for name in macro[0]["saved_OLS_metrics_mean"]}})
    write_new(HERE/"TRANSFER_REFERENCE_FAILURE_NUMBERS.json",{
        "actual_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "registration_sha256":sha(HERE/"TRANSFER_FAILURE_SUMMARY_REGISTERED.json"),
        "curve_parameter_and_saved_score_summaries":curve_rows,"recipe_mask_summaries":by_recipe,
        "macro_mask_summaries":by_macro,"saved_universal_macros":data["universal_scores"]["macro_by_method"],
        "saved_powerlaw_macros":[r for r in data["physical_scores"]["macro_by_method"] if r["method_id"].startswith("power_law__")],
        "counts":{"universal_laws_summarized":420,"recipe_mask_ref_cells":84,"macro_mask_ref_cells":28,
            "new_fits":0,"new_forecasts":0,"new_CRPS_or_NLL_evaluations":0,"raw_source_eta_cells_read":0,"future35_eta_cells_read":0},
        "study_role":"POST_REVEAL_DEV_DIAGNOSIS_OF_SAVED_RESULTS_NOT_NEW_CONFIRMATION"})
    for row in by_macro:
        if row["mask_method_id"]=="curv_nearest_target6":
            print(json.dumps(row,ensure_ascii=False))
    print("SUMMARY_COMPLETE;0newfit/forecast/score/raweta")

if __name__=="__main__":main()
