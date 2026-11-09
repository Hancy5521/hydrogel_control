"""Summarize saved expansion scores/audit receipts; no fit/forecast/rescore/Y.

Pareto comparison below uses only saved CRPS, mean run log-RMSE and joint NLL.
Interval width and empirical coverage are reported jointly, not optimized as
standalone monotonically better quantities. No final-confirm recipe is read.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

BASE=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
FILES={"scores":BASE/"expansion_once/score/SCORES.json",
       "pre_audit":BASE/"expansion_once/INDEPENDENT_PRE_Y.json",
       "post_audit":BASE/"expansion_once/INDEPENDENT_POST_Y.json",
       "selection":BASE/"expansion_study/DEVELOPMENT_SELECTION_RECEIPT.json"}
AXES=("crps","log_rmse","joint_nll_per_dimension")

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def write_new(path,payload):
    with Path(path).open("x",encoding="utf-8") as stream:
        json.dump(payload,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")

def main():
    now=dt.datetime.now(dt.timezone.utc)
    registration={"actual_utc":now.isoformat(),"asia_shanghai":now.astimezone(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_sha256":sha(__file__),"inputs":{k:{"path":str(p),"sha256":sha(p)} for k,p in FILES.items()},
        "scope":"descriptive summary of already saved scores/audits; no raw Y,fit,new prediction or rescore",
        "prior_knowledge":"root already disclosed main outcomes; no claim of result-blind analysis"}
    write_new(HERE/"EXPANSION_REVIEW_SUMMARY_REGISTERED.json",registration)
    scores,pre,post=(load(FILES[k]) for k in ["scores","pre_audit","post_audit"])
    if pre["status"]!="PASS" or post["status"]!="PASS":raise AssertionError("Actual independent PASS required")
    if post["audited_expansion_scores_sha256"]!=sha(FILES["scores"]):raise AssertionError("Audit/score bytes differ")
    macro={r["method_id"]:r for r in scores["macro_by_method"]}
    recipe={(r["recipe"],r["method_id"]):r for r in scores["recipe_scores"]}
    selected="copula_frozen_curvature_posterior"
    student="physical_full_student";gp="copula_frozen_gp_posterior";ggp="gaussian_frozen_gp"
    names=sorted(macro)
    dominance=[]
    for a in names:
        for b in names:
            if a!=b and all(macro[a][k]<=macro[b][k] for k in AXES) and any(macro[a][k]<macro[b][k] for k in AXES):
                dominance.append({"dominates":a,"dominated":b})
    primary=macro[selected]["joint_nll_per_dimension"]-macro[student]["joint_nll_per_dimension"]
    secondary=macro[selected]["joint_nll_per_dimension"]-macro[gp]["joint_nll_per_dimension"]
    rows=[]
    for name in sorted({r["recipe"] for r in scores["recipe_scores"]}):
        rows.append({"recipe":name,"runs":5,
            "selected_minus_Student_joint_delta":recipe[name,selected]["joint_nll_per_dimension"]-recipe[name,student]["joint_nll_per_dimension"],
            "selected_minus_GPcopula_joint_delta":recipe[name,selected]["joint_nll_per_dimension"]-recipe[name,gp]["joint_nll_per_dimension"],
            "selected_Student_marginal_metrics":{k:recipe[name,selected][k] for k in ["crps","log_rmse","coverage90","width90"]},
            "GaussianGP_metrics":{k:recipe[name,ggp][k] for k in ["crps","joint_nll_per_dimension","log_rmse","coverage90","width90"]}})
    coverage=[]
    for method in names:
        chosen=[p for p in scores["point_terms"] if p["method_id"]==method]
        coverage.append({"method_id":method,"same_target_cells":len(chosen),
                         "covered90":sum(bool(p["coverage90"]) for p in chosen)})
    result={"actual_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
        "registration_sha256":sha(HERE/"EXPANSION_REVIEW_SUMMARY_REGISTERED.json"),
        "main_claim":"Selected candidate improves joint NLL against full physicalStudent on held-out recipe groups; no marginal gain by copula construction; no all-baseline dominance",
        "same_source_not_new_lab":True,"groups":{"recipes":3,"runs":15,"unique_anchor_cells":90,"unique_target_cells":120,
            "forecast_laws":105,"point_score_terms":840,"final20_numeric_eta_read":0},
        "actual_pre_audit_status":pre["status"],"actual_post_audit_status":post["status"],
        "macro_by_method":scores["macro_by_method"],"recipe_scores":scores["recipe_scores"],
        "primary_selected_minus_Student":primary,"secondary_selected_minus_GPcopula":secondary,
        "reported_primary":scores["primary"],"reported_secondary":scores["secondary"],
        "recipe_contrasts":rows,"coverage_counts_from_saved_points":coverage,
        "Pareto_axes_lower_better":list(AXES),"Pareto_dominance_from_saved_macros":dominance,
        "selected_vs_GaussianGP":{
            "CRPS_difference":macro[selected]["crps"]-macro[ggp]["crps"],
            "CRPS_ratio":macro[selected]["crps"]/macro[ggp]["crps"],
            "mean_run_logRMSE_difference":macro[selected]["log_rmse"]-macro[ggp]["log_rmse"],
            "width90_ratio":macro[selected]["width90"]/macro[ggp]["width90"],
            "joint_NLL_difference":macro[selected]["joint_nll_per_dimension"]-macro[ggp]["joint_nll_per_dimension"]},
        "no_new_fits":True,"no_new_predictions":True,"no_CRPS_or_density_recomputation":True,"raw_source_or_target_file_opened":False,
        "actual_code_defects_found_by_current_checks":[],
        "claim_limits":["three held-out recipes from same public source, not independent laboratory/batch validation",
            "all four copulas retain Student marginals exactly, including RMSE/coverage/width",
            "selected curvature copula worse than GPcopula on secondary endpoint in all three recipes",
            "GaussianGP has better marginal CRPS and joint NLL, while physicalStudent-based means have better aggregate RMSE",
            "coverage counts use correlated target points, not120independent calibration observations",
            "do not switch primary method after opening holdout, pool development as independent confirmation, or read final20"]}
    write_new(HERE/"EXPANSION_RESULT_REVIEW_NUMBERS.json",result)
    print(json.dumps({"primary":primary,"secondary":secondary,"selected_vs_GaussianGP":result["selected_vs_GaussianGP"],"Pareto":dominance},ensure_ascii=False))

if __name__=="__main__":main()
