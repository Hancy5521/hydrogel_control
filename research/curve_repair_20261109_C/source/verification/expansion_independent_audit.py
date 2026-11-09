"""Independent native-grid, observed-only forecast and released-score audit.

SOURCE ONLY until actual root registration. This expansion contains three
previously numerically unconsumed recipes from the SAME public source, not a
new laboratory experiment. Final-confirm20 runs remain outside every eta API.
No production import, optimizer, new mask/seed, or raw CSV before post-Y.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import math
from pathlib import Path
import zipfile

HELPER_PINS={
    "resin_independent_math.py":"f7858fecc1ccb049e89e0c2c6d2edb052e142fa2a180f26e0f18fa70ace40897",
    "student_independent_audit.py":"76b1b16887591ec3ffe90d504cb746c6a87d0f7bc772a58677bc3c4bf0d28491",
    "marginal_transport_independent_audit.py":"009b1c0d9095d1d864e7d756fcb92f385bd175dbbd8e1acf7d9f679fb15c2727",
}
for _name,_pin in HELPER_PINS.items():
    if hashlib.sha256(Path(__file__).with_name(_name).read_bytes()).hexdigest()!=_pin:
        raise ImportError("Qualified independent helper bytes changed: "+_name)

import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_t,t
from resin_independent_math import (Comparisons,INPUT_SHA,MODEL_SHA,SOURCE_SHA,TARGET,
    actual_utc,condition,joint,load,native_reference,recipe_roles,score_gaussian,sha,write_new)
from student_independent_audit import qr_prediction
from marginal_transport_independent_audit import independent_density

ANCHORS=[0,1,13,14,15,16]
METHODS=("physical_full_student","copula_independence","copula_physical_student_shape",
    "copula_frozen_gp_posterior","copula_frozen_curvature_posterior",
    "gaussian_frozen_gp","gaussian_frozen_curvature")
METRICS=("crps","joint_nll_per_dimension","log_rmse","coverage90","width90")

def authorize(args):
    reg=load(args.authorization)
    if (reg.get("scope")!="expansion_independent_audit" or reg.get("allowed_phase")!=args.phase
        or reg.get("auditor_source_sha256")!=sha(__file__)):
        raise PermissionError("Exact expansion independent scope/phase/auditor hash required")
    stamp=dt.datetime.fromisoformat(reg["actual_utc"].replace("Z","+00:00"))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture UTC required")
    required=["prepare","predictions","inputs","model","source_zip"]
    if args.phase=="post_y":required += ["targets","scores","pre_audit","release"]
    elif any(getattr(args,name) is not None for name in ["targets","scores","pre_audit","release"]):
        raise PermissionError("Pre-Y audit must not receive target/score payloads")
    for name in required:
        path=getattr(args,name)
        if path is None or reg["input_sha256"].get(name)!=sha(path):
            raise PermissionError("Exact authorized input identity required: "+name)
    if sha(args.source_zip)!=SOURCE_SHA:
        raise AssertionError("Original public ZIP changed")
    return reg

def correlations(k):
    k=np.asarray(k,float)
    s=np.sqrt(np.diag(k))
    return k/s[:,None]/s[None,:]

def pre_audit(args):
    inputs,model=load(args.inputs,INPUT_SHA),load(args.model,MODEL_SHA)
    prepared,predicted=load(args.prepare),load(args.predictions)
    if (predicted["prepare_sha256"]!=sha(args.prepare) or predicted["authorized_role"]!="future_expansion"
        or predicted["OLS_fits"]!=15 or predicted["covariance_updates"]!=0
        or predicted["target_labels_accessed"]!=0 or predicted["anchor_cells_accessed"]!=90
        or predicted["final_confirm_runs_protected"]!=20
        or prepared["source_zip_sha256"]!=SOURCE_SHA or prepared["input_sha256"]!=INPUT_SHA
        or prepared["numerical_eta_accessed"]!=0 or prepared["final_confirm_runs_protected"]!=20):
        raise AssertionError("Expansion input/fit/access or protection accounting changed")
    roles,specs=recipe_roles(inputs)
    allowed={member for member,spec in specs.items() if roles[spec["recipe"]]=="future_expansion"}
    if len(allowed)!=15:
        raise AssertionError("Exactly original3expansion recipes15runs required")
    curves={row["member"]:row for row in prepared["curves"]}
    if len(prepared["curves"])!=15 or set(curves)!=allowed:
        raise AssertionError("Expansion input population changed or contains final/dev members")
    if {name:row["role"] for name,row in prepared["roles"].items()}!=roles:
        raise AssertionError("Original3dev/3expansion/4final recipe roles changed")
    check=Comparisons()
    laws={}
    minimum_residual=float("inf")
    temperature_minimum,temperature_maximum=float("inf"),-float("inf")
    history_counts={}
    for member,curve in curves.items():
        spec=specs[member]
        if any(curve[key]!=spec[key] for key in ["recipe","run_id","thermal_history","source_member_sha256"]):
            raise AssertionError("Expansion original-source/group identity changed")
        gamma=np.array([float(token) for token in spec["native_gamma_raw_tokens_sorted"]])
        coordinates=np.log(gamma)
        if (gamma.shape!=(25,) or not np.isfinite(gamma).all() or np.any(np.diff(gamma)<=0)
            or coordinates.min()<model["q"][0] or coordinates.max()>model["q"][-1]):
            raise AssertionError("Native25 must be positive/distinct/in frozen physical support")
        low,high=spec["actual35_temperature_min_C"],spec["actual35_temperature_max_C"]
        if not 34.5<=low<=high<35.5 or set(spec["status_raw_counts"])!={"Dy_auto"}:
            raise AssertionError("Admitted35degree block/status inputs changed")
        history=spec["thermal_history"]
        position=0 if history=="ascending_35_first" else -1
        if history not in ["ascending_35_first","descending_35_last"] or not 34.5<=spec["nominal_temperature_sequence_C"][position]<35.5:
            raise AssertionError("Nominal35degree thermal-order identity differs")
        temperature_minimum=min(temperature_minimum,low);temperature_maximum=max(temperature_maximum,high)
        history_counts[history]=history_counts.get(history,0)+1
        check.close("admitted_native_gamma",curve["native_gamma"],gamma,atol=0,rtol=0)
        check.close("admitted_log_gamma",curve["log_gamma"],coordinates,atol=1e-14,rtol=1e-14)
        m,k,b,residual=native_reference(coordinates,model)
        minimum_residual=min(minimum_residual,residual)
        for key,value in [("mean",m),("covariance",k),("cross",b)]:
            check.close("independent_native_reference_"+key,curve["reference"][key],value)
        for ref,theta in [("gp_theta0",np.zeros(5)),("curvature_frozen_theta",np.asarray(model["theta"]))]:
            check.close("unchanged_theta",curve["laws"][ref]["theta"],theta,atol=0,rtol=0)
            mm,kk=joint(m,k,b,theta)
            check.close("native_joint_mean",curve["laws"][ref]["mean"],mm)
            check.close("native_joint_covariance",curve["laws"][ref]["covariance"],kk)
            laws[member,ref]=mm,kk
    rows={(row["member"],row["method_id"]):row for row in predicted["forecasts"]}
    if len(predicted["forecasts"])!=105 or set(rows)!={(member,method) for member in allowed for method in METHODS}:
        raise AssertionError("Exactly7frozen methods per15curves required")
    expected_student={}
    expected_gaussian={}
    unique_anchor_cells=set()
    for member in sorted(allowed):
        student=rows[member,"physical_full_student"]
        values=student["anchor_values"]
        if len(values)!=6 or not np.isfinite(values).all():
            raise AssertionError("Only six finite observed anchor values required")
        curve=curves[member]
        x=np.array(curve["log_gamma"])
        expected=qr_prediction(x[ANCHORS],values,x[TARGET],"power_law")
        expected_student[member]=expected
        for method in METHODS:
            row=rows[member,method]
            if (row["anchor_indices"]!=ANCHORS or row["target_indices"]!=TARGET
                or row["anchor_values"]!=values or row["role"]!="future_expansion"
                or any(row[key]!=curve[key] for key in ["recipe","run_id","thermal_history"])):
                raise AssertionError("Forecast own-six/role/source group/panel changed")
            unique_anchor_cells.update((member,index) for index in row["anchor_indices"])
        if student["df"]!=4 or student["n"]!=6 or student["p"]!=2 or student["scale_is_covariance"] is not False:
            raise AssertionError("Student shape/df/p/n semantics changed")
        for key in ["beta","sse","location","scale_matrix","predictive_covariance"]:
            check.close("independent_QR_Student_"+key,student[key],expected[key],atol=2e-9,rtol=2e-8)
        for key in ["residual_norm_resolution","sse_resolution_threshold"]:
            check.close("Student_unresolved_SSE_rule_"+key,student[key],expected[key],atol=0,rtol=3e-8)
        if student["resolution_multiplier"]!=64:
            raise AssertionError("Fixed unresolved-SSE policy multiplier changed")
        for ref,method in [("gp_theta0","gaussian_frozen_gp"),("curvature_frozen_theta","gaussian_frozen_curvature")]:
            m,k=laws[member,ref]
            mu,cov=condition(m,k,ANCHORS,values)
            mm,vv=mu[TARGET],cov[np.ix_(TARGET,TARGET)]
            expected_gaussian[member,method]=(mm,vv)
            check.close("Gaussian_target_mean",rows[member,method]["mean"],mm,atol=1e-7,rtol=1e-8)
            check.close("Gaussian_target_covariance",rows[member,method]["covariance"],vv)
        rsets={"copula_independence":np.eye(8),
            "copula_physical_student_shape":correlations(expected["scale_matrix"]),
            "copula_frozen_gp_posterior":correlations(expected_gaussian[member,"gaussian_frozen_gp"][1]),
            "copula_frozen_curvature_posterior":correlations(expected_gaussian[member,"gaussian_frozen_curvature"][1])}
        scale=np.sqrt(np.diag(student["scale_matrix"]))
        for method,r in rsets.items():
            row=rows[member,method]
            if row["df"]!=4 or row["scale_is_marginal_sd"] is not False:
                raise AssertionError("Fixed Student4 marginal semantics changed")
            check.close("copula_exact_Student_marginal_mean",row["location"],student["location"],atol=0,rtol=0)
            check.close("copula_exact_Student_marginal_scale",row["scale"],scale,atol=0,rtol=0)
            check.close("independent_copula_R",row["correlation"],r,atol=2e-8,rtol=2e-8)
            rr=np.asarray(row["correlation"])
            if np.linalg.eigvalsh(rr).min()<=0:
                raise AssertionError("SPD Gaussian copula R required, no repair/floor")
            check.close("copula_R_diagonal",np.diag(rr),np.ones(8),atol=3e-12,rtol=0)
    if len(unique_anchor_cells)!=90:
        raise AssertionError("Exact90observed-anchor cells required")
    return check,rows,specs,expected_student,{
        "audited_expansion_predictions_sha256":sha(args.predictions),
        "audited_expansion_prepare_sha256":sha(args.prepare),"input_masks_sha256":sha(args.inputs),
        "source_zip_sha256":sha(args.source_zip),"frozen_model_sha256":sha(args.model),
        "forecast_laws":105,"Student_QR_laws":15,"Gaussian_Schur_laws":30,"copula_laws":60,
        "minimum_full_reference_residual_eigenvalue":minimum_residual,
        "actual35_temperature_range_C":[temperature_minimum,temperature_maximum],
        "history_run_counts":history_counts,
        "unique_observed_anchor_cells":90,"raw_CSV_opened":False,"target_artifact_opened":False,
        "new_raw_source_eta_access":0,"final20_eta_access":0,
        "same_source_new_recipes_not_new_lab_experiment":True}

def student_scores(location,shape,y,constant):
    mu,s,y=np.asarray(location),np.asarray(shape),np.asarray(y)
    scale=np.sqrt(np.diag(s));error=y-mu;z=error/scale
    points=scale*(z*(2*t.cdf(z,4)-1)+2*(4+z*z)*t.pdf(z,4)/3-constant)
    cutoff=t.ppf(.95,4)
    return {"crps":float(np.mean(points)),"joint_nll_per_dimension":-float(multivariate_t.logpdf(y,loc=mu,shape=s,df=4))/8,
        "log_rmse":float(np.sqrt(np.mean(error**2))),"coverage90":float(np.mean(np.abs(error)<=cutoff*scale)),
        "width90":float(np.mean(2*cutoff*scale)),"crps_points":points.tolist()}

def post_audit(args,check,rows,specs,student,result):
    previous,release=load(args.pre_audit),load(args.release)
    predicted_sha,prepared_sha=sha(args.predictions),sha(args.prepare)
    if (previous.get("status")!="PASS"
        or previous.get("audited_expansion_predictions_sha256")!=predicted_sha
        or previous.get("audited_expansion_prepare_sha256")!=prepared_sha):
        raise PermissionError("Exact independent pre-Y expansion audit PASS required")
    if (release.get("schema")!="mlst-resin-expansion-target-release-v1"
        or release.get("authorization")!="REVEAL_ONLY_120_ORIGINAL_FUTURE_EXPANSION_TARGETS"
        or release.get("predictions_sha256")!=predicted_sha
        or release.get("prepare_sha256")!=prepared_sha or release.get("audit_sha256")!=sha(args.pre_audit)):
        raise PermissionError("Separate exact root expansion-target release required")
    stamp=dt.datetime.fromisoformat(release["actual_utc"].replace("Z","+00:00"))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture target release UTC required")
    targets,scores=load(args.targets),load(args.scores)
    if (targets["predictions_sha256"]!=predicted_sha or targets["prepare_sha256"]!=prepared_sha
        or targets["audit_sha256"]!=sha(args.pre_audit) or targets["root_release_sha256"]!=sha(args.release)
        or scores["predictions_sha256"]!=predicted_sha or scores["targets_sha256"]!=sha(args.targets)):
        raise AssertionError("Frozen-prediction/release/score byte chain changed")
    truth={row["member"]:row for row in targets["curves"]}
    if len(targets["curves"])!=15 or set(truth)!=set(student):
        raise AssertionError("Exact fifteen authorized expansion target curves required")
    raw_values,addresses={},[]
    with zipfile.ZipFile(args.source_zip) as archive:
        for member in sorted(student):
            spec=specs[member]
            raw=archive.read(member)
            if hashlib.sha256(raw).hexdigest()!=spec["source_member_sha256"]:
                raise AssertionError("Original expansion run bytes changed")
            lines=raw.decode("utf-16").splitlines()
            # Exactly90already-used anchors +120released targets. No other
            # response token is converted or used for validity/QC inspection.
            for index in ANCHORS+TARGET:
                line=spec["source_line_by_rank_1based"][index]
                parts=lines[line-1].split("\t")
                if (len(parts)!=7 or parts[5].strip()!=spec["native_gamma_raw_tokens_sorted"][index]
                    or parts[4]!="Dy_auto" or not 34.5<=float(parts[2])<35.5):
                    raise AssertionError("Independent raw input/status/temperature address mismatch")
                cp=float(parts[3])
                if not math.isfinite(cp) or cp<=0:
                    raise AssertionError("STOP_NO_REPLACEMENT invalid/nonpositive released source eta")
                raw_values[member,index]=math.log(cp/1000)
                addresses.append({"member":member,"rank_1based":index+1,"source_line_1based":line,
                    "role":"released_expansion_target" if index in TARGET else "already_used_expansion_anchor"})
            target=truth[member]
            if (target["role"]!="future_expansion" or target["target_indices"]!=TARGET
                or any(target[name]!=spec[name] for name in ["recipe","run_id","thermal_history"])):
                raise AssertionError("Target panel/source/expansion-role identity changed")
            check.close("raw_original_expansion_targets",target["values"],
                [raw_values[member,index] for index in TARGET],atol=1e-14,rtol=1e-14)
            check.close("raw_original_expansion_anchors",rows[member,"physical_full_student"]["anchor_values"],
                [raw_values[member,index] for index in ANCHORS],atol=1e-14,rtol=1e-14)
    actual={(row["member"],row["method_id"]):row for row in scores["curve_scores"]}
    points={(row["member"],row["method_id"],row["target_index"]):row for row in scores["point_terms"]}
    if len(actual)!=105 or len(scores["curve_scores"])!=105 or set(actual)!=set(rows) or len(points)!=840 or len(scores["point_terms"])!=840:
        raise AssertionError("Exactly105curve scores/840unique point terms required")
    constant,constant_error=quad(lambda z:2*t.cdf(z,4)*t.sf(z,4),0,np.inf,
                                 epsabs=2e-12,epsrel=2e-12,limit=180)
    expected={}
    for member in student:
        law=rows[member,"physical_full_student"]
        expected[member,"physical_full_student"]=student_scores(law["location"],law["scale_matrix"],truth[member]["values"],constant)
    independence_count=0
    for key,law in rows.items():
        member,method=key
        y=np.asarray(truth[member]["values"])
        base=expected[member,"physical_full_student"]
        if method in METHODS[5:]:
            value=score_gaussian(law["mean"],law["covariance"],y)
            value["joint_nll_per_dimension"]=value.pop("joint_nll_per_target")
        elif method in METHODS[1:5]:
            value=dict(base)
            logpdf,_,product=independent_density(y,law["location"],law["scale"],law["correlation"])
            value["joint_nll_per_dimension"]=-logpdf/8
            if method=="copula_independence":
                check.close("I_equals_product_Student_marginals",logpdf,product,atol=2e-10,rtol=2e-10)
                independence_count+=1
        else:value=base
        value["joint_nll_delta_vs_full_student"]=value["joint_nll_per_dimension"]-base["joint_nll_per_dimension"]
        expected[key]=value
        for metric in (*METRICS,"joint_nll_delta_vs_full_student"):
            check.close("independent_curve_"+metric,actual[key][metric],value[metric],atol=3e-8,rtol=3e-8)
        if method in METHODS[1:5]:
            for metric in ["crps","log_rmse","coverage90","width90"]:
                check.close("copula_marginal_score_identity_"+metric,actual[key][metric],actual[member,"physical_full_student"][metric],atol=0,rtol=0)
        for j,index in enumerate(TARGET):
            point=points[member,method,index]
            for identity in ["recipe","run_id","thermal_history","role"]:
                if point[identity]!=law[identity] or actual[key][identity]!=law[identity]:
                    raise AssertionError("Score/source group identity changed")
            mu=law["mean"][j] if method in METHODS[5:] else law["location"][j]
            if method in METHODS[5:]:
                scale=math.sqrt(law["covariance"][j][j]);cutoff=1.6448536269514722
                check.close("Gaussian_point_sd",point["sd"],scale)
            else:
                parent=rows[member,"physical_full_student"]
                scale=math.sqrt(parent["scale_matrix"][j][j]);cutoff=t.ppf(.95,4)
                check.close("Student_point_scale",point["t_scale"],scale)
                check.close("Student_point_sd",point["t_sd"],scale*math.sqrt(2))
                if point["df"]!=4: raise AssertionError("Student point df changed")
            for metric,v in [("truth",y[j]),("location",mu),("crps",value["crps_points"][j]),
                ("squared_error",(y[j]-mu)**2),("width90",2*cutoff*scale)]:
                check.close("independent_point_"+metric,point[metric],v,atol=3e-8,rtol=3e-8)
            if point["coverage90"]!=bool(abs(y[j]-mu)<=cutoff*scale):
                raise AssertionError("Point interval coverage mismatch")
            if method in METHODS[1:5]:
                original=points[member,"physical_full_student",index]
                if point["marginal_metrics_inherited_exactly"] is not True:
                    raise AssertionError("Copula marginal scores cannot be claimed new")
                for name in ["truth","location","t_scale","t_sd","df","crps","squared_error","coverage90","width90"]:
                    if point[name]!=original[name]:raise AssertionError("Exact point marginal identity broken")
    recipes=sorted({row["recipe"] for row in truth.values()})
    recipe_rows={(row["recipe"],row["method_id"]):row for row in scores["recipe_scores"]}
    macro_rows={row["method_id"]:row for row in scores["macro_by_method"]}
    if len(recipes)!=3 or len(recipe_rows)!=21 or len(macro_rows)!=7:
        raise AssertionError("Exactly21recipe and7macro score rows required")
    grouped,macro={},{}
    for method in METHODS:
        for recipe in recipes:
            chosen=[v for (member,name),v in expected.items() if name==method and truth[member]["recipe"]==recipe]
            if len(chosen)!=5:raise AssertionError("Each original recipe keeps all five runs")
            grouped[recipe,method]={metric:float(np.mean([v[metric] for v in chosen])) for metric in (*METRICS,"joint_nll_delta_vs_full_student")}
            for metric,value in grouped[recipe,method].items():
                check.close("recipe_"+metric,recipe_rows[recipe,method][metric],value,atol=3e-8,rtol=3e-8)
        macro[method]={metric:float(np.mean([grouped[recipe,method][metric] for recipe in recipes])) for metric in (*METRICS,"joint_nll_delta_vs_full_student")}
        for metric,value in macro[method].items():
            check.close("macro_"+metric,macro_rows[method][metric],value,atol=3e-8,rtol=3e-8)
    primary=macro[METHODS[4]]["joint_nll_per_dimension"]-macro[METHODS[0]]["joint_nll_per_dimension"]
    secondary=macro[METHODS[4]]["joint_nll_per_dimension"]-macro[METHODS[3]]["joint_nll_per_dimension"]
    check.close("primary_joint_delta",scores["primary"]["recipe_macro_joint_nll_delta"],primary,atol=3e-8,rtol=3e-8)
    check.close("secondary_curv_minus_GP",scores["secondary"]["recipe_macro_joint_nll_delta"],secondary,atol=3e-8,rtol=3e-8)
    per_recipe=[]
    reported={row["recipe"]:row for row in scores["primary"]["per_recipe"]}
    for recipe in recipes:
        first=grouped[recipe,METHODS[4]]["joint_nll_per_dimension"]-grouped[recipe,METHODS[0]]["joint_nll_per_dimension"]
        second=grouped[recipe,METHODS[4]]["joint_nll_per_dimension"]-grouped[recipe,METHODS[3]]["joint_nll_per_dimension"]
        check.close("primary_per_recipe",reported[recipe]["primary_delta"],first,atol=3e-8,rtol=3e-8)
        check.close("secondary_per_recipe",reported[recipe]["secondary_delta_curv_minus_gp"],second,atol=3e-8,rtol=3e-8)
        per_recipe.append({"recipe":recipe,"primary_delta":first,"secondary_delta_curv_minus_gp":second})
    if scores["unique_anchor_cells"]!=90 or scores["unique_target_cells"]!=120 or scores["final_confirm_runs_protected"]!=20:
        raise AssertionError("Measurement budget/final protection misreported")
    result.update({"audited_expansion_scores_sha256":sha(args.scores),"audited_expansion_targets_sha256":sha(args.targets),
        "root_target_release_sha256":sha(args.release),"pre_y_audit_sha256":sha(args.pre_audit),
        "raw_CSV_opened":True,"target_artifact_opened":True,
        "raw_access_scope":{"already_used_expansion_anchor_cells":90,"released_expansion_target_cells":120,
            "final_confirm_eta_cells":0,"old_dev_eta_cells":0,"source_addresses":addresses},
        "independent_primary_joint_delta":primary,"independent_secondary_delta":secondary,
        "independent_per_recipe":per_recipe,"independent_macro":macro,
        "score_curves":105,"point_terms":840,"I_product_checks":independence_count,
        "independent_Student_CRPS_CDF_constant":constant,"constant_quad_error":constant_error,
        "marginal_CRPS_identity_not_new_gain":True,
        "scope":"Same-source previously held-out recipe groups, not new laboratory, certified new batches or final confirmation"})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["pre_y","post_y"])
    for name in ["authorization","prepare","predictions","inputs","model","source-zip","output"]:
        parser.add_argument("--"+name,required=True)
    for name in ["targets","scores","pre-audit","release"]:
        parser.add_argument("--"+name)
    args=parser.parse_args()
    try:
        authorize(args)
        checks,rows,specs,student,result=pre_audit(args)
        if args.phase=="post_y":post_audit(args,checks,rows,specs,student,result)
        report={"status":"PASS","actual_utc":actual_utc(),"phase":args.phase,
            "auditor_source_sha256":sha(__file__),"authorization_sha256":sha(args.authorization),
            "independent_helper_sha256":HELPER_PINS,"comparisons":checks.groups,
            "production_imported":False,**result}
    except Exception as error:
        write_new(args.output,{"status":"FAIL","actual_utc":actual_utc(),
            "error_type":type(error).__name__,"error":str(error)})
        raise
    write_new(args.output,report)
    print("Expansion independent audit PASS; phase="+args.phase+"; final20untouched")

if __name__=="__main__":main()
