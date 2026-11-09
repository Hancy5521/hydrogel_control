"""Independent, anchor-only audit of the single declared shared update.

No production/frozen-runtime imports and no CSV/target loading. Gradients are
reconstructed by differentiating the explicit two-dimensional Schur law,
rather than subtracting two joint-NLL gradients, and independently checked with
two central finite-difference step sizes. Presence is not execution authority.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
from resin_independent_math import (Comparisons, TARGET, actual_utc, condition,
    joint, load, sha, write_new)

CONTEXT, CHECK = [0,1,3,4], [2,5]
BOUNDS = np.array([[-1.,1.],[-1.,1.],[-.75,.75],[-1.,1.],[-.75,.75]])
FINITE_STEPS = (1e-4, 2e-5)

def conditional_score(reference, theta, values):
    m0, k0, b = (np.asarray(reference[k],float) for k in ("mean","covariance","cross"))
    theta, y = np.asarray(theta,float), np.asarray(values,float)
    mean, cov = joint(m0,k0,b,theta)
    o,h = CONTEXT,CHECK
    a,c = cov[np.ix_(o,o)],cov[np.ix_(h,o)]
    w = np.linalg.inv(a)
    ro = y[o]-mean[o]
    mu = mean[h]+c@w@ro
    v = cov[np.ix_(h,h)]-c@w@c.T
    v = (v+v.T)/2
    e = y[h]-mu
    sign,logdet = np.linalg.slogdet(v)
    if sign != 1:
        raise AssertionError("Independent conditional covariance not positive definite")
    vi = np.linalg.inv(v)
    alpha = vi@e
    nll = float((2*math.log(2*math.pi)+logdet+e@alpha)/2)
    # Differentiate the Schur conditional directly, not a difference of two
    # full-joint Gaussian objectives as in the production score implementation.
    logl0,l10,logl1 = theta[2],theta[3],theta[4]
    l0,l1 = math.exp(logl0),math.exp(logl1)
    dcode = [np.array([[2*l0*l0,l0*l10],[l0*l10,0.]]),
             np.array([[0.,l0],[l0,2*l10]]),
             np.array([[0.,0.],[0.,2*l1*l1]])]
    gradient=[]
    for coordinate in range(5):
        dm = b[:,coordinate] if coordinate < 2 else np.zeros(len(mean))
        dk = np.zeros_like(cov) if coordinate < 2 else b@dcode[coordinate-2]@b.T
        da,dc = dk[np.ix_(o,o)],dk[np.ix_(h,o)]
        dmu = dm[h]+dc@w@ro-c@w@dm[o]-c@w@da@w@ro
        dv = dk[np.ix_(h,h)]-dc@w@c.T-c@w@dc.T+c@w@da@w@c.T
        gradient.append(float(-dmu@alpha + .5*np.trace((vi-np.outer(alpha,alpha))@dv)))
    return nll,np.asarray(gradient)

def finite_gradient(reference,theta,values,step):
    gradient=[]
    for coordinate in range(5):
        delta=np.zeros(5)
        delta[coordinate]=step
        gradient.append((conditional_score(reference,theta+delta,values)[0]
                        -conditional_score(reference,theta-delta,values)[0])/(2*step))
    return np.asarray(gradient)

def audit(args):
    prepared,main,pilot,previous = (load(path) for path in
        (args.main_prepare,args.main_predictions,args.pilot,args.main_pre_audit))
    prepared_sha,main_sha = sha(args.main_prepare),sha(args.main_predictions)
    if previous.get("status")!="PASS" or previous.get("audited_prepare_sha256")!=prepared_sha or previous.get("audited_predictions_sha256")!=main_sha:
        raise PermissionError("Original independent target-free main audit PASS required")
    if (pilot["schema"]!="mlst-resin-early-update-predictions-v1"
        or pilot["prepare_sha256"]!=prepared_sha or pilot["main_predictions_sha256"]!=main_sha):
        raise AssertionError("Early-update payload does not bind original main artifacts")
    comparisons=Comparisons()
    comparisons.close("declared_bounds",pilot["bounds"],BOUNDS,atol=0,rtol=0)
    if pilot["context_indices"]!=CONTEXT or pilot["check_indices"]!=CHECK or pilot["step_length"]!=.05:
        raise AssertionError("Registered pseudo-hole layout or single step changed")
    if (pilot["parameter_update_steps"]!=1 or pilot["optimizer_loop_runs"]!=0
            or pilot["step_search"] or pilot["gate_uses_exact_endpoint"]
            or pilot["target_labels_accessed"]!=0 or pilot["new_source_eta_access"]!=0):
        raise AssertionError("Early pilot exceeds one anchor-only declared update")
    sources={curve["member"]:curve for curve in prepared["curves"]}
    first={row["member"]:row for row in main["forecasts"]
           if row["method_id"]=="curvature_frozen_theta__fixed_first6"}
    payload={row["member"]:row for row in pilot["curves"]}
    if len(first)!=15 or len(payload)!=15 or set(first)!=set(payload):
        raise AssertionError("Exact fifteen saved own-six payloads required")
    training=[m for m,r in first.items() if r["thermal_history"]=="ascending_35_first"]
    sentinel=[m for m,r in first.items() if r["thermal_history"]=="descending_35_last"]
    if len(training)!=10 or len(sentinel)!=5 or pilot["training_members"]!=training or pilot["sentinel_members"]!=sentinel:
        raise AssertionError("History-declared train/sentinel population changed")
    trgroups={first[m]["recipe"] for m in training}
    segroups={first[m]["recipe"] for m in sentinel}
    if len(trgroups)!=2 or len(segroups)!=1 or trgroups&segroups:
        raise AssertionError("Two train recipes and disjoint single sentinel recipe required")
    theta=np.asarray(pilot["base_theta"],float)
    per_curve={}
    for member,row in first.items():
        for identity in ("member","recipe","run_id","thermal_history","anchor_indices","anchor_values"):
            if payload[member][identity]!=row[identity]:
                raise AssertionError("Early pilot changed own-six saved observations or identity")
        if row["anchor_indices"]!=list(range(6)):
            raise AssertionError("Own-six observation layout changed")
        comparisons.close("frozen_base_theta",theta,sources[member]["laws"]["curvature_frozen_theta"]["theta"],atol=0,rtol=0)
        score,gradient=conditional_score(sources[member]["reference"],theta,row["anchor_values"])
        finite={}
        for step in FINITE_STEPS:
            g=finite_gradient(sources[member]["reference"],theta,row["anchor_values"],step)
            comparisons.close("direct_gradient_vs_FD_"+str(step),gradient,g,atol=3e-5,rtol=3e-5)
            finite[str(step)]=g.tolist()
        per_curve[member]={"score":score,"gradient":gradient.tolist(),"finite_difference_gradients":finite}
    train_g=np.mean([per_curve[m]["gradient"] for m in training],axis=0)
    sentinel_g=np.mean([per_curve[m]["gradient"] for m in sentinel],axis=0)
    train_score=float(np.mean([per_curve[m]["score"] for m in training]))
    sentinel_score=float(np.mean([per_curve[m]["score"] for m in sentinel]))
    comparisons.close("training_gradient_for_step",pilot["training_gradient_for_step"],train_g,atol=3e-7,rtol=3e-7)
    norm=float(np.linalg.norm(train_g))
    comparisons.close("training_gradient_norm",pilot["training_gradient_norm"],norm,atol=3e-7,rtol=3e-7)
    candidate=np.clip(theta-.05*train_g/norm,BOUNDS[:,0],BOUNDS[:,1]) if norm else theta.copy()
    comparisons.close("single_candidate",pilot["candidate_theta"],candidate,atol=5e-9,rtol=5e-9)
    # Also ensure production used exactly its own declared analytic gradient.
    stored_g=np.asarray(pilot["training_gradient_for_step"],float)
    stored_norm=float(np.linalg.norm(stored_g))
    formula=np.clip(theta-.05*stored_g/stored_norm,BOUNDS[:,0],BOUNDS[:,1]) if stored_norm else theta.copy()
    comparisons.close("exact_recorded_gradient_step",pilot["candidate_theta"],formula,atol=1e-15,rtol=1e-15)
    delta=candidate-theta
    comparisons.close("actual_projected_delta",pilot["delta"],delta,atol=5e-9,rtol=5e-9)
    h=float(sentinel_g@delta)
    report=pilot["early_report"]
    comparisons.close("report_training_gradient",report["training_gradient"],train_g,atol=3e-7,rtol=3e-7)
    comparisons.close("report_sentinel_gradient",report["development_gradient"],sentinel_g,atol=3e-7,rtol=3e-7)
    comparisons.close("report_training_nll",report["training_conditional_nll"],train_score,atol=3e-7,rtol=3e-7)
    comparisons.close("report_sentinel_nll",report["development_conditional_nll"],sentinel_score,atol=3e-7,rtol=3e-7)
    comparisons.close("early_h",report["development_directional_derivative"],h,atol=3e-7,rtol=3e-7)
    if bool(pilot["gate_accept"])!=(h<0):
        raise AssertionError("Gate differs from preregistered strict negative h rule")
    endpoint=float(np.mean([conditional_score(sources[m]["reference"],candidate,first[m]["anchor_values"])[0] for m in sentinel]))
    if [step["step"] for step in report["steps"]]!=[0.,1.]:
        raise AssertionError("Unexpected endpoint/step scan")
    comparisons.close("endpoint_actual_delta",report["steps"][1]["actual_development_nll_change"],endpoint-sentinel_score,atol=3e-7,rtol=3e-7)
    comparisons.close("endpoint_predicted_delta",report["steps"][1]["predicted_development_nll_change"],h,atol=3e-7,rtol=3e-7)
    candidates={"base":theta,"candidate":candidate,"gated":candidate if h<0 else theta}
    forecasts={(row["member"],row["method_id"]):row for row in pilot["forecasts"]}
    if len(pilot["forecasts"])!=45 or set(forecasts)!={(m,name) for m in first for name in candidates}:
        raise AssertionError("Forty-five complete frozen predictions required")
    for (member,method),row in forecasts.items():
        if row["anchor_indices"]!=list(range(6)) or row["target_indices"]!=TARGET or row["recipe"]!=first[member]["recipe"]:
            raise AssertionError("Forecast identity/own-six/panel changed")
        source=sources[member]["reference"]
        m,k=joint(np.asarray(source["mean"]),np.asarray(source["covariance"]),np.asarray(source["cross"]),candidates[method])
        mu,cov=condition(m,k,list(range(6)),first[member]["anchor_values"])
        comparisons.close("forecast_theta",row["theta"],candidates[method],atol=5e-9,rtol=5e-9)
        comparisons.close("forecast_mean",row["mean"],mu[TARGET],atol=2e-7,rtol=2e-8)
        comparisons.close("forecast_covariance",row["covariance"],cov[np.ix_(TARGET,TARGET)],atol=3e-9,rtol=3e-9)
        if method=="base":
            comparisons.close("unchanged_main_baseline_mean",row["mean"],first[member]["mean"],atol=0,rtol=0)
            comparisons.close("unchanged_main_baseline_covariance",row["covariance"],first[member]["covariance"],atol=0,rtol=0)
    return {"status":"PASS","actual_utc":actual_utc(),
        "audited_early_predictions_sha256":sha(args.pilot),
        "audited_main_predictions_sha256":main_sha,"audited_prepare_sha256":prepared_sha,
        "main_pre_audit_sha256":sha(args.main_pre_audit),"auditor_source_sha256":sha(__file__),
        "auditor_math_sha256":sha(Path(__file__).with_name("resin_independent_math.py")),
        "comparisons":comparisons.groups,"per_curve_direct_schur_and_finite_differences":per_curve,
        "independent_training_gradient":train_g.tolist(),"independent_sentinel_gradient":sentinel_g.tolist(),
        "independent_candidate":candidate.tolist(),"independent_delta":delta.tolist(),
        "independent_h":h,"independent_gate_accept":bool(h<0),
        "independent_exact_sentinel_change":endpoint-sentinel_score,
        "forecasts":45,"source_CSV_opened":False,"target_artifact_opened":False,
        "new_numeric_eta_access":0,"SUT_imported":False,"optimizer_or_parameter_search":False,
        "limitations":["One projected gradient update is learning, despite zero optimizer loops.",
            "Sentinel is development and thermal order is confounded with recipe.",
            "A negative anchor-sentinel derivative neither guarantees finite-step benefit nor tail accuracy."]}

def main():
    parser=argparse.ArgumentParser()
    for option in ("main-prepare","main-predictions","main-pre-audit","pilot","output"):
        parser.add_argument("--"+option,required=True)
    args=parser.parse_args()
    try:
        report=audit(args)
    except Exception as error:
        write_new(args.output,{"status":"FAIL","actual_utc":actual_utc(),
            "error_type":type(error).__name__,"error":str(error),
            "source_CSV_opened":False,"target_artifact_opened":False})
        raise
    write_new(args.output,report)
    print("Independent early-update pre-Y audit PASS; all45 laws and one candidate checked, no targets.")

if __name__=="__main__":
    main()
