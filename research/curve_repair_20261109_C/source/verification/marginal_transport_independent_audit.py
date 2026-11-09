"""Independent audit of fixed-panel Student marginals plus Gaussian copula.

SOURCE ONLY until exact actual-UTC root authorization. There are no production
imports, CSV readers, fits, covariance selection or future-recipe labels. The
same120 development targets are already consumed; this is no confirmation.
"""
from __future__ import annotations
import argparse
import datetime as dt
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.special import ndtri_exp
from scipy.stats import multivariate_normal,multivariate_t,norm,t
from resin_independent_math import Comparisons,TARGET,actual_utc,load,sha,write_new

CONTROL_IDS=("independence","physical_student_shape","frozen_gp_posterior","frozen_curvature_posterior")
MARGINAL_METRICS=("crps","log_rmse","coverage90","width90")

def authorize(args):
    reg=load(args.authorization)
    if (reg.get("scope")!="marginal_transport_independent_audit" or reg.get("allowed_phase")!=args.phase
        or reg.get("auditor_source_sha256")!=sha(__file__)):
        raise PermissionError("Exact root scope/phase/auditor hash required")
    if sha(Path(__file__).with_name("resin_independent_math.py"))!="f7858fecc1ccb049e89e0c2c6d2edb052e142fa2a180f26e0f18fa70ace40897":
        raise PermissionError("Qualified independent JSON/numeric helper bytes changed")
    stamp=dt.datetime.fromisoformat(reg["actual_utc"].replace("Z","+00:00"))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture timezone-aware UTC required")
    required=(['gate','samples'] if args.phase=='gates' else
              ['main_predictions','student_predictions','predictions','main_audit','student_audit','gate_audit'])
    if args.phase=='scores':required+=['targets','scores','student_scores']
    elif args.phase=='laws' and any(getattr(args,name) is not None for name in ['targets','scores','student_scores']):
        raise PermissionError("Law-only audit accepts no target/score payload")
    for name in required:
        path=getattr(args,name)
        if path is None or reg["input_sha256"].get(name)!=sha(path):
            raise PermissionError("Exact root-authorized input hash required: "+name)
    return reg

def marginal_normal_scores(value,location,scale):
    """Independent library definition; preserve the accurate tail on each side."""
    standardized=(np.asarray(value,float)-np.asarray(location,float))/np.asarray(scale,float)
    out=np.empty_like(standardized)
    negative=standardized<=0
    out[negative]=ndtri_exp(t.logcdf(standardized[negative],4))
    out[~negative]=-ndtri_exp(t.logsf(standardized[~negative],4))
    if not np.isfinite(out).all():
        raise ArithmeticError("Unresolved Student tail: no clipping/fallback")
    return out

def independent_density(value,location,scale,correlation):
    y,m,s,r=(np.asarray(a,float) for a in (value,location,scale,correlation))
    z=marginal_normal_scores(y,m,s)
    marginal=t.logpdf((y-m)/s,4)-np.log(s)
    # SciPy's eigensystem/PSD factorization differs from production Cholesky.
    gaussian=multivariate_normal.logpdf(z,mean=np.zeros(len(z)),cov=r,allow_singular=False)
    return float(gaussian-np.sum(norm.logpdf(z))+np.sum(marginal)),z,float(np.sum(marginal))

def mp4_normal_score(x):
    """Independent t4 closed CDF +120digit erf inverse, not scipy's t CDF."""
    with mp.workdps(120):
        x=mp.mpf(float(x))
        a=abs(x)
        radius=mp.sqrt(4+a*a)
        u=a/radius
        delta=4/(radius*(radius+a))
        sf=delta*delta*(u+2)/4
        positive=mp.sqrt(2)*mp.erfinv(1-2*sf)
        return float(positive if x>=0 else -positive)

def normalized_covariance(value):
    covariance=np.asarray(value,float)
    sd=np.sqrt(np.diag(covariance))
    return covariance/sd[:,None]/sd[None,:]

def law_audit(args):
    main,student,saved,main_audit,student_audit,gate_audit=(load(path) for path in
        [args.main_predictions,args.student_predictions,args.predictions,args.main_audit,args.student_audit,args.gate_audit])
    if (main_audit.get("status")!="PASS" or main_audit.get("audited_predictions_sha256")!=sha(args.main_predictions)
        or student_audit.get("status")!="PASS"
        or student_audit.get("audited_baseline_predictions_sha256")!=sha(args.student_predictions)
        or gate_audit.get("status")!="PASS"):
        raise PermissionError("Exact original main/physicalStudent audits and new synthetic gate PASS required")
    if (saved["main_predictions_sha256"]!=sha(args.main_predictions)
        or saved["student_predictions_sha256"]!=sha(args.student_predictions)
        or saved["synthetic_gate_sha256"]!=gate_audit["audited_gate_sha256"]
        or saved["independent_gate_audit_sha256"]!=sha(args.gate_audit)
        or saved["new_fits"]!=0 or saved["new_source_numeric_eta_access"]!=0
        or saved["future_runs_protected"]!=35):
        raise AssertionError("Fixed-law source identities/gate/role accounting changed")
    gaussian={(r["member"],r["method_id"]):r for r in main["forecasts"]}
    physical={r["member"]:r for r in student["forecasts"]
              if r["baseline_kind"]=="power_law" and r["mask_method_id"]=="curv_nearest_target6"}
    if len(physical)!=15 or len(saved["forecasts"])!=60:
        raise AssertionError("Canonical own-six15parents and60copula laws required")
    check,seen,forecasts=Comparisons(),set(),{}
    for law in saved["forecasts"]:
        member,control=law["member"],law["control_id"]
        if (member,control) in seen or member not in physical or control not in CONTROL_IDS:
            raise AssertionError("Duplicate/unregistered fixed copula control")
        seen.add((member,control));parent=physical[member]
        for field in ["recipe","run_id","thermal_history","anchor_indices","anchor_values","target_indices"]:
            if law[field]!=parent[field]:
                raise AssertionError("Copula source/group/mask changed: "+field)
        if law["anchor_indices"]!=[0,1,13,14,15,16] or law["target_indices"]!=TARGET or law["df"]!=4:
            raise AssertionError("Fixed own-six/fixed8panel/df4 changed")
        if (law["marginal_parent_method_id"]!="power_law__curv_nearest_target6"
            or law["distribution"]!="gaussian_copula_student4_marginals"
            or law["scale_is_marginal_sd"] is not False or law["marginal_variance_factor"]!=2):
            raise AssertionError("Physical Student marginal identity/scale semantics changed")
        check.close("same_Student_location",law["location"],parent["location"],atol=0,rtol=0)
        check.close("same_Student_scale",law["scale"],np.sqrt(np.diag(parent["scale_matrix"])),atol=0,rtol=0)
        check.close("same_Student_quantiles",law["quantiles05_50_95"],
            np.array(law["location"])[None,:]+np.array(law["scale"])[None,:]*t.ppf(np.array([.05,.5,.95])[:,None],4),atol=3e-10,rtol=3e-10)
        covariance={"physical_student_shape":parent["scale_matrix"],
            "frozen_gp_posterior":gaussian[member,"GP_nearest_target6"]["covariance"],
            "frozen_curvature_posterior":gaussian[member,"curv_nearest_target6"]["covariance"]}
        r=np.asarray(law["correlation"],float)
        if r.shape!=(8,8) or np.linalg.eigvalsh(r).min()<=0 or not np.allclose(r,r.T,atol=1e-12,rtol=0):
            raise AssertionError("Valid SPD correlation required, without jitter or floor")
        expected=np.eye(8) if control=="independence" else normalized_covariance(covariance[control])
        check.close("correlation_from_saved_law",r,expected,atol=3e-12,rtol=3e-12)
        check.close("correlation_unit_diagonal",np.diag(r),np.ones(8),atol=3e-12,rtol=0)
        forecasts[member,law["method_id"]]=law
    if seen!={(member,control) for member in physical for control in CONTROL_IDS}:
        raise AssertionError("All four fixed controls per devcurve required")
    return check,forecasts,physical,{
        "audited_copula_predictions_sha256":sha(args.predictions),
        "audited_transport_predictions_sha256":sha(args.predictions),
        "audited_main_predictions_sha256":sha(args.main_predictions),
        "audited_student_predictions_sha256":sha(args.student_predictions),
        "copula_laws":60,"dev_curves":15,"dev_recipes":3,
        "new_fits":0,"new_raw_source_eta_access":0,"future_final_runs_protected":35,
        "marginal_equality_is_construction_not_new_performance":True,
        "scope":"fixed8D panels; no across-mask conditional consistency claim; same consumed120dev targets"}

def score_audit(args,check,forecasts,physical,result):
    targets,scores,old_scores=(load(path) for path in [args.targets,args.scores,args.student_scores])
    if targets["predictions_sha256"]!=sha(args.main_predictions):
        raise AssertionError("Only original120dev targets allowed")
    if (scores["predictions_sha256"]!=sha(args.predictions) or scores["targets_sha256"]!=sha(args.targets)
        or scores["parent_student_scores_sha256"]!=sha(args.student_scores)
        or old_scores["predictions_sha256"]!=sha(args.student_predictions)
        or old_scores["targets_sha256"]!=sha(args.targets)):
        raise AssertionError("Fixed copula/parent Student score byte chain changed")
    truths={r["member"]:r for r in targets["curves"]}
    old={r["member"]:r for r in old_scores["curve_scores"]
         if r["baseline_kind"]=="power_law" and r["mask_method_id"]=="curv_nearest_target6"}
    old_points={(r["member"],r["target_index"]):r for r in old_scores["point_terms"]
                if r["baseline_kind"]=="power_law" and r["mask_method_id"]=="curv_nearest_target6"}
    rows={(r["member"],r["method_id"]):r for r in scores["curve_scores"]}
    points={(r["member"],r["method_id"],r["target_index"]):r for r in scores["point_terms"]}
    if len(truths)!=15 or len(old)!=15 or len(rows)!=60 or set(rows)!=set(forecasts):
        raise AssertionError("Exact60laws/same15targetcurves required")
    if len(points)!=480 or len(scores["point_terms"])!=480:
        raise AssertionError("Exactly480same-marginal point terms required")
    original_full_t={}
    for member,parent in physical.items():
        original_full_t[member]=-float(multivariate_t.logpdf(truths[member]["values"],
            loc=parent["location"],shape=parent["scale_matrix"],df=4))/8
        check.close("original_full_Student_joint_NLL",old[member]["joint_nll_per_dimension"],original_full_t[member],atol=3e-8,rtol=3e-8)
    independent_joint={}
    for key,law in forecasts.items():
        truth=truths[key[0]]
        if truth["target_indices"]!=TARGET or truth["recipe"]!=law["recipe"]:
            raise AssertionError("Fixed panel/group changed")
        logp,z,marginal=independent_density(truth["values"],law["location"],law["scale"],law["correlation"])
        nll=-logp/8
        independent_joint[key]=nll
        check.close("independent_joint_NLL",rows[key]["joint_nll_per_dimension"],nll,atol=3e-8,rtol=3e-8)
        check.close("strong_original_Student_joint_retained",rows[key]["original_student_joint_nll_per_dimension"],old[key[0]]["joint_nll_per_dimension"],atol=0,rtol=0)
        check.close("joint_delta_vs_original_Student",rows[key]["joint_nll_delta_vs_original_student"],nll-original_full_t[key[0]],atol=3e-8,rtol=3e-8)
        if law["control_id"]=="independence":
            check.close("R_I_equals_product_of_Student_marginals",logp,marginal,atol=2e-10,rtol=2e-10)
        for metric in MARGINAL_METRICS:
            check.close("unchanged_marginal_"+metric,rows[key][metric],old[key[0]][metric],atol=0,rtol=0)
        for j,index in enumerate(TARGET):
            point=points[key[0],key[1],index]
            source=old_points[key[0],index]
            for name in ["truth","location","t_scale","t_sd","df","crps","squared_error","coverage90","width90"]:
                if point[name]!=source[name]:
                    raise AssertionError("Marginal point score changed: "+name)
            check.close("same_point_quantiles",point["quantiles05_50_95"],
                [law["quantiles05_50_95"][i][j] for i in range(3)],atol=0,rtol=0)
            if point["marginal_metrics_inherited_exactly"] is not True:
                raise AssertionError("Inherited marginal score falsely described")
    recipe_rows={(r["recipe"],r["method_id"]):r for r in scores["recipe_scores"]}
    macro_rows={r["method_id"]:r for r in scores["macro_by_method"]}
    recipes=sorted({r["recipe"] for r in truths.values()})
    if len(recipe_rows)!=12 or len(macro_rows)!=4 or len(recipes)!=3:
        raise AssertionError("Exactly12recipe and4macro rows required")
    metrics=(*MARGINAL_METRICS,"joint_nll_per_dimension","original_student_joint_nll_per_dimension","joint_nll_delta_vs_original_student")
    independent_macro={}
    for control in CONTROL_IDS:
        per_recipe=[]
        for recipe in recipes:
            selected=[row for (member,method),row in rows.items() if method==control and row["recipe"]==recipe]
            if len(selected)!=5: raise AssertionError("All five original runs must remain in recipe")
            calculated={metric:float(np.mean([r[metric] for r in selected])) for metric in metrics}
            # New-density entries are also independently reconstructed rather
            # than merely trusting copied curve values for aggregation.
            calculated["joint_nll_per_dimension"]=float(np.mean([value for (member,method),value in independent_joint.items() if method==control and truths[member]["recipe"]==recipe]))
            calculated["original_student_joint_nll_per_dimension"]=float(np.mean([value for member,value in original_full_t.items() if truths[member]["recipe"]==recipe]))
            calculated["joint_nll_delta_vs_original_student"]=calculated["joint_nll_per_dimension"]-calculated["original_student_joint_nll_per_dimension"]
            for metric,value in calculated.items():
                check.close("recipe_"+metric,recipe_rows[recipe,control][metric],value,atol=3e-8,rtol=3e-8)
            per_recipe.append(calculated)
        independent_macro[control]={metric:float(np.mean([r[metric] for r in per_recipe])) for metric in metrics}
        for metric,value in independent_macro[control].items():
            check.close("macro_"+metric,macro_rows[control][metric],value,atol=3e-8,rtol=3e-8)
    result.update({"audited_scores_sha256":sha(args.scores),"audited_targets_sha256":sha(args.targets),
        "joint_NLL_recomputed":60,"independent_joint_NLL_rows":[{"member":k[0],"method_id":k[1],
            "joint_nll_per_dimension":v} for k,v in independent_joint.items()],
        "original_full_Student_baseline_retained":True,"original_full_Student_joint_recomputed":15,
        "independent_macro":independent_macro,"same_marginal_point_terms":480})

def synthetic_audit(args):
    gate=load(args.gate)
    check=Comparisons()
    if gate["status"]!="PASS":
        raise AssertionError("Production fixed synthetic gate has not passed")
    fixture=gate["fixture"]
    m,s,r=(np.asarray(fixture[key],float) for key in ["location","scale","correlation"])
    check.close("fixed_fixture_location",m,[.2,-.4,.9],atol=0,rtol=0)
    check.close("fixed_fixture_scale",s,[.3,.7,1.1],atol=0,rtol=0)
    check.close("fixed_fixture_R",r,[[1,.4,-.2],[.4,1,.3],[-.2,.3,1]],atol=0,rtol=0)
    if fixture["df"]!=4 or np.linalg.eigvalsh(r).min()<=0:
        raise AssertionError("Fixed synthetic SPD/df fixture changed")
    normal=np.array([[-9,0,9],[9,-9,0],[0,0,0],[-2,1,.5],[.1,-.2,3]],float)
    check.close("fixed_normal_probes",gate["normal_probes"],normal,atol=0,rtol=0)
    transported=np.asarray(gate["transported_probes"],float)
    if transported.shape!=(5,3): raise AssertionError("Exactly five synthetic transport probes required")
    independent_z=[]
    for index,y in enumerate(transported):
        high=np.array([mp4_normal_score(v) for v in (y-m)/s])
        independent_z.append(high)
        check.close("mp120_inverse_CDF_transport",high,normal[index],atol=3e-9,rtol=3e-9)
        logp,z,marginal=independent_density(y,m,s,r)
        check.close("library_vs_mp120_normal_scores",z,high,atol=3e-10,rtol=3e-10)
        check.close("synthetic_density",gate["logpdf_probes"][index],logp,atol=3e-8,rtol=3e-8)
        identity,_,_=independent_density(y,m,s,np.eye(3))
        check.close("synthetic_I_is_product",identity,marginal,atol=3e-10,rtol=3e-10)
        check.close("production_I_is_product",gate["independence_logpdf_probes"][index],marginal,atol=3e-10,rtol=3e-10)
    check.close("reported_roundtrip_normal",gate["recovered_normal"],normal,atol=3e-9,rtol=3e-9)
    for dimension in range(3):
        order=np.argsort(normal[:,dimension],kind='stable')
        a,b=normal[order,dimension],transported[order,dimension]
        if np.any(np.diff(b)[np.diff(a)>0]<=0):
            raise AssertionError("Transport is not increasing")
    tail_count=0
    check.close("fixed_extreme_tail_probes",gate["tail_standardized_probes"],
        [[-1e12,0.,1e12],[1e12,-1e12,0.]],atol=0,rtol=0)
    for i,values in enumerate(gate["tail_y_probes"]):
        standardized=np.asarray(gate["tail_standardized_probes"][i],float)
        y=np.asarray(values,float)
        check.close("tail_y_construction",y,m+s*standardized,atol=0,rtol=0)
        high=np.array([mp4_normal_score(v) for v in (y-m)/s])
        check.close("mp120_extreme_tail_normal_score",gate["tail_normal_probes"][i],high,atol=3e-9,rtol=3e-9)
        logp,z,_=independent_density(y,m,s,r)
        check.close("tail_density",gate["tail_logpdf_probes"][i],logp,atol=3e-8,rtol=3e-8)
        tail_count+=1
    if tail_count<2 or not gate["reject_singular"] or not gate["reject_indefinite"]:
        raise AssertionError("Both extreme tails and invalid-R rejection probes required")
    if sha(args.samples)!=gate["samples_sha256"]:
        raise AssertionError("Fixed synthetic sample bytes changed")
    with np.load(args.samples,allow_pickle=False) as saved:
        samples=np.array(saved["samples"],float)
    if samples.shape!=(20000,3) or not np.isfinite(samples).all() or gate["draws"]!=20000 or gate["seed"]!=20261109:
        raise AssertionError("Exactly once-declared20k synthetic samples required")
    probabilities=np.array([.1,.5,.9])
    check.close("fixed_sample_probabilities",gate["probabilities"],probabilities,atol=0,rtol=0)
    quantiles=m[None,:]+s[None,:]*t.ppf(probabilities[:,None],4)
    check.close("fixed_marginal_quantiles",gate["quantiles"],quantiles,atol=3e-10,rtol=3e-10)
    counts=np.array([[np.count_nonzero(samples[:,j]<=quantiles[i,j]) for j in range(3)] for i in range(3)])
    check.close("sample_CDF_counts",gate["sample_counts"],counts,atol=0,rtol=0)
    tolerance=(6*np.sqrt(20000*probabilities*(1-probabilities))+2)[:,None]
    if np.any(np.abs(counts-20000*probabilities[:,None])>tolerance):
        raise AssertionError("Fixed synthetic empirical marginal CDF exceeds6sigma+2 binomial tolerance")
    latent=np.column_stack([marginal_normal_scores(samples[:,j],m[j],s[j]) for j in range(3)])
    mean=np.mean(latent,axis=0)
    empirical=np.cov(latent,rowvar=False,ddof=1)
    if np.max(np.abs(mean))>6/math.sqrt(20000) or np.max(np.abs(empirical-r))>6*math.sqrt(2/19999):
        raise AssertionError("Synthetic latent-normal moments inconsistent with declared correlation")
    return check,{"audited_gate_sha256":sha(args.gate),"audited_synthetic_gate_sha256":sha(args.gate),"audited_samples_sha256":sha(args.samples),
        "mpmath_version":mp.__version__,"independent_tail_precision_digits":120,
        "synthetic_draws":20000,"new_measurements":0,"sample_CDF_counts":counts.tolist(),
        "CDF_count_tolerance_6sigma_plus2":tolerance[:,0].tolist(),
        "latent_normal_empirical_mean":mean.tolist(),"latent_normal_empirical_covariance":empirical.tolist(),
        "Monte_Carlo_is_software_check_not_material_calibration":True}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["gates","laws","scores"])
    parser.add_argument("--authorization",required=True)
    parser.add_argument("--output",required=True)
    for name in ["gate","samples","main-predictions","student-predictions","predictions","main-audit",
                 "student-audit","gate-audit","targets","scores","student-scores"]:
        parser.add_argument("--"+name)
    args=parser.parse_args()
    try:
        authorize(args)
        if args.phase=="gates":checks,result=synthetic_audit(args)
        else:
            checks,forecasts,physical,result=law_audit(args)
            if args.phase=="scores":score_audit(args,checks,forecasts,physical,result)
        report={"status":"PASS","actual_utc":actual_utc(),"phase":args.phase,
            "auditor_source_sha256":sha(__file__),"authorization_sha256":sha(args.authorization),
            "comparisons":checks.groups,"SUT_imported":False,"raw_CSV_opened":False,**result}
    except Exception as error:
        write_new(args.output,{"status":"FAIL","actual_utc":actual_utc(),
            "error_type":type(error).__name__,"error":str(error)})
        raise
    write_new(args.output,report)
    print("Independent marginal transport audit PASS; phase="+args.phase)

if __name__=="__main__":main()
