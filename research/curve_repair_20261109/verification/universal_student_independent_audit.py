"""Independent augmented-system audit of universal-Kriging Student transfer.

SOURCE ONLY UNTIL ROOT AUTHORIZATION. This is a post-reveal development
diagnostic: the same120 targets were already consumed by the original pilot.
No CSV/future-recipe reader, production import, parameter search or explicit
matrix inverse is used. Location and shape use >=70digit block linear solves.
"""
from __future__ import annotations
import argparse
import datetime as dt
import math
from pathlib import Path
import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_t,t
from resin_independent_math import Comparisons,TARGET,actual_utc,load,sha,write_new

REFERENCES=("gp_theta0","curvature_frozen_theta")
METRICS=("crps","joint_nll_per_dimension","log_rmse","coverage90","width90")

def authorize(args):
    reg=load(args.authorization)
    if (reg.get("scope")!="universal_student_independent_audit"
            or reg.get("allowed_phase")!=args.phase
            or reg.get("auditor_source_sha256")!=sha(__file__)):
        raise PermissionError("Exact root scope/phase/auditor source authorization required")
    if sha(Path(__file__).with_name("resin_independent_math.py"))!="f7858fecc1ccb049e89e0c2c6d2edb052e142fa2a180f26e0f18fa70ace40897":
        raise PermissionError("Original qualified independent JSON/comparison helpers changed")
    stamp=dt.datetime.fromisoformat(reg["actual_utc"].replace("Z","+00:00"))
    if stamp.utcoffset() is None or stamp>dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture timezone-aware UTC required")
    required=["prepare","main_predictions","predictions","main_audit","gate"]
    if args.phase=="scores":
        required += ["targets","scores"]
    elif args.targets is not None or args.scores is not None:
        raise PermissionError("Law-only audit accepts no target/score path")
    for name in required:
        path=getattr(args,name)
        if path is None or reg["input_sha256"].get(name)!=sha(path):
            raise PermissionError("Exact authorized input hash required: "+name)
    return reg

def gate_audit(path,check):
    gate=load(path)
    if gate["status"]!="PASS" or gate["new_material_labels"]!=0 or gate["synthetic_fixture_only"] is not True:
        raise AssertionError("Bounded synthetic gate identity mismatch")
    trace=gate["numeric_trace"]
    fixture=trace["fixture"]
    x=np.array([-1.8,-1.1,-.6,-.2,.2,.7,1.,1.4,1.9,2.3,2.8,3.2])
    o,h=np.array([0,2,4,5,7,8]),np.array([1,3,6,9,10,11])
    y=.3-.4*x[o]+np.array([.1,-.08,.03,.12,-.02,-.06])
    k=np.exp(-.5*((x[:,None]-x[None,:])/.9)**2)+.07*np.eye(len(x))
    order=np.array([3,0,5,2,1,4])
    for name,value in (("log_gamma",x),("anchor_values",y),("correlated_K",k)):
        check.close("synthetic_fixture_"+name,fixture[name],value,atol=1e-15,rtol=1e-15)
    if (fixture["anchor_indices"]!=o.tolist() or fixture["target_indices"]!=h.tolist()
        or fixture["anchor_permutation"]!=order.tolist() or fixture["scaled_K_multiplier"]!=7.):
        raise AssertionError("Synthetic fixture grouping/scale changed")
    identity=augmented_prediction(x,y,o,h,np.eye(len(x)))
    correlated=augmented_prediction(x,y,o,h,k)
    scaled=augmented_prediction(x,y,o,h,7*k)
    permuted=augmented_prediction(x,y[order],o[order],h[::-1],k)
    for label,expected in (("identity_universal",identity),("identity_powerlaw",identity),
            ("correlated_universal",correlated),("scaled_correlated_universal",scaled),
            ("permuted_universal",permuted)):
        for name in ("beta","sse","location","scale_matrix","predictive_covariance"):
            check.close("synthetic_augmented70_"+label+"_"+name,trace[label][name],expected[name],atol=2e-10,rtol=2e-10)
    for name in ("beta","sse","location","scale_matrix"):
        check.close("synthetic_dense_formula_"+name,trace["correlated_dense_formula"][name],correlated[name],atol=2e-10,rtol=2e-10)
    for name in ("beta","location","scale_matrix","predictive_covariance"):
        check.close("synthetic_scale_gauge_"+name,scaled[name],correlated[name],atol=2e-12,rtol=2e-12)
    # Direct ordinary least-squares law is an additional K=I equivalence path.
    xo=np.column_stack((np.ones(6),x[o]))
    xt=np.column_stack((np.ones(6),x[h]))
    q,r=np.linalg.qr(xo,mode="reduced")
    beta=np.linalg.solve(r,q.T@y)
    residual=y-xo@beta
    factor=np.linalg.solve(r.T,xt.T)
    ordinary_shape=(residual@residual/4)*(np.eye(6)+factor.T@factor)
    check.close("synthetic_identity_equals_QR_OLS_mean",identity["location"],xt@beta,atol=2e-12,rtol=2e-12)
    check.close("synthetic_identity_equals_QR_OLS_shape",identity["scale_matrix"],ordinary_shape,atol=2e-12,rtol=2e-12)
    rejected=trace["exactline_rejected_fixture"]
    if (rejected["log_gamma"]!=list(np.arange(12.)) or rejected["anchor_indices"]!=list(range(6))
        or rejected["target_indices"]!=list(range(6,12))
        or rejected["anchor_values"]!=list(1.+2*np.arange(6.))
        or "numerically_unresolved_residual" not in rejected["error"]):
        raise AssertionError("Production exactline rejection trace changed")
    try:
        augmented_prediction(np.arange(12.),1.+2*np.arange(6.),list(range(6)),list(range(6,12)),np.eye(12))
    except ArithmeticError as error:
        if "numerically unresolved residual" not in str(error): raise
    else:
        raise AssertionError("Independent exactline residual must be rejected")
    return {"synthetic_numeric_trace_audited":True,"synthetic_gate_sha256":sha(path),
        "K_identity_equals_existing_powerlaw_Student":True,"global_K_scale_gauge":True,
        "exactline_SSE_rejection":True}

def augmented_prediction(log_gamma,values,anchor_indices,target_indices,covariance):
    x,y,k=np.asarray(log_gamma,float),np.asarray(values,float),np.asarray(covariance,float)
    o,h=list(anchor_indices),list(target_indices)
    if len(o)!=6 or len(set(o))!=6 or y.shape!=(6,) or not len(h):
        raise AssertionError("Exactly own6 distinct anchors and nonempty targets required")
    design=np.column_stack((np.ones(6),x[o]))
    target=np.column_stack((np.ones(len(h)),x[h]))
    if np.linalg.matrix_rank(design)!=2:
        raise ArithmeticError("REJECT rank deficient universal mean design")
    with mp.workdps(70):
        def matrix(a):
            return mp.matrix([[mp.mpf(float(v)) for v in row] for row in np.asarray(a)])
        ko=matrix(k[np.ix_(o,o)])
        kt=matrix(k[np.ix_(h,h)])
        kto=matrix(k[np.ix_(h,o)])
        xo,xt=matrix(design),matrix(target)
        ym=mp.matrix([mp.mpf(float(v)) for v in y])
        # Universal-Kriging saddle-point system. No K^-1 or Gram inverse.
        block=mp.matrix(8,8)
        for i in range(6):
            for j in range(6): block[i,j]=ko[i,j]
            for j in range(2): block[i,6+j]=xo[i,j];block[6+j,i]=xo[i,j]
        rhs=mp.matrix(list(ym)+[mp.mpf(0),mp.mpf(0)])
        solution=mp.lu_solve(block,rhs)
        u,beta=mp.matrix(solution[:6]),mp.matrix(solution[6:])
        location=kto*u+xt*beta
        cross=mp.matrix(len(h),8)
        for i in range(len(h)):
            for j in range(6): cross[i,j]=kto[i,j]
            for j in range(2): cross[i,6+j]=xt[i,j]
        solved=mp.matrix(8,len(h))
        for j in range(len(h)):
            column=mp.lu_solve(block,mp.matrix([cross[j,i] for i in range(8)]))
            for i in range(8): solved[i,j]=column[i]
        unscaled=kt-cross*solved
        conditional=mp.matrix(kt)
        for j in range(len(h)):
            ordinary_column=mp.lu_solve(ko,mp.matrix([kto[j,i] for i in range(6)]))
            for i in range(len(h)):
                conditional[i,j]-=(kto[i,:]*ordinary_column)[0]
        trend_uncertainty=unscaled-conditional
        residual=ym-xo*beta
        weighted=mp.lu_solve(ko,residual)
        sse=(residual.T*weighted)[0]
        shape=(sse/4)*unscaled
        bf=np.array([float(v) for v in beta])
        sf=float(sse)
        # Different whitening path; any K^-1/2 and production L^-1 differ by
        # an orthogonal transformation, so both norms in the rule agree.
        eigenvalues,vectors=np.linalg.eigh(k[np.ix_(o,o)])
        if eigenvalues.min()<=0:
            raise ArithmeticError("REJECT nonpositive anchor covariance")
        whitening=(vectors*(1/np.sqrt(eigenvalues)))@vectors.T
        resolution=float(64*np.finfo(float).eps*(np.linalg.norm(whitening@design,2)
                       *np.linalg.norm(bf,2)+np.linalg.norm(whitening@y,2)))
        threshold=resolution**2
        if not math.isfinite(sf) or sf<=threshold:
            raise ArithmeticError("REJECT numerically unresolved residual; no variance floor")
        return {"beta":bf,"sse":sf,"n":6,"p":2,"df":4,
                "location":np.array([float(v) for v in location]),
                "scale_matrix":np.array(shape.tolist(),float),
                "predictive_covariance":2*np.array(shape.tolist(),float),
                "conditional_reference_covariance":np.array(conditional.tolist(),float),
                "trend_uncertainty_covariance":np.array(trend_uncertainty.tolist(),float),
                "residual_norm_resolution":resolution,"sse_resolution_threshold":threshold}

def law_audit(args):
    prepared,main,saved,previous=(load(path) for path in
        (args.prepare,args.main_predictions,args.predictions,args.main_audit))
    if (previous.get("status")!="PASS"
        or previous.get("audited_predictions_sha256")!=sha(args.main_predictions)
        or previous.get("audited_prepare_sha256")!=sha(args.prepare)):
        raise PermissionError("Exact original target-free main audit PASS required")
    if saved["main_prepare_sha256"]!=sha(args.prepare) or saved["main_predictions_sha256"]!=sha(args.main_predictions):
        raise AssertionError("Universal diagnostic is not bound to original inputs")
    if (saved["study_role"]!="POST_REVEAL_DEV_DIAGNOSTIC" or saved["future_final_runs_protected"]!=35
        or saved["new_raw_eta_access"]!=0 or saved["new_target_access"]!=0
        or saved["covariance_hyperparameter_updates"]!=0 or saved["GLS_fits"]!=420):
        raise AssertionError("Transfer study role/target access/fit accounting changed")
    curves={row["member"]:row for row in prepared["curves"]}
    mains={(row["member"],row["method_id"]):row for row in main["forecasts"]}
    if len(curves)!=15 or len(mains)!=210 or len(saved["forecasts"])!=420:
        raise AssertionError("Exact15curves,210own6mask,420universal laws required")
    check=Comparisons()
    gate_result=gate_audit(args.gate,check)
    forecasts,seen={},set()
    minratio=float("inf")
    for law in saved["forecasts"]:
        key=law["member"],law["mask_method_id"],law["covariance_id"]
        if key in seen or key[:2] not in mains or key[2] not in REFERENCES:
            raise AssertionError("Duplicate/unregistered universal law")
        seen.add(key)
        parent,curve=mains[key[:2]],curves[key[0]]
        for field in ("recipe","run_id","thermal_history","anchor_indices","anchor_values","target_indices"):
            if law[field]!=parent[field]:
                raise AssertionError("Universal model received different information: "+field)
        if law["method_id"]!=key[2]+"__universal_student__"+key[1] or law["target_indices"]!=TARGET:
            raise AssertionError("Universal method identity or fixed panel changed")
        if law["n"]!=6 or law["p"]!=2 or law["df"]!=4 or law["scale_is_covariance"] is not False:
            raise AssertionError("Predictive t has df4 and shape is not covariance")
        if (law["frozen_reference_mean_used"] is not False or law["full_frozen_covariance_used"] is not True
            or law["covariance_hyperparameter_updates"]!=0
            or law["prior"]!="p(beta,sigma_squared) proportional to 1/sigma_squared"):
            raise AssertionError("Universal trend/reference/prior changed")
        expected=augmented_prediction(curve["log_gamma"],law["anchor_values"],law["anchor_indices"],TARGET,
                                      curve["laws"][key[2]]["covariance"])
        for name in ("beta","sse","location","scale_matrix","predictive_covariance",
                     "conditional_reference_covariance","trend_uncertainty_covariance"):
            check.close("augmented70_"+name,law[name],expected[name],atol=2e-8,rtol=2e-8)
        for name in ("residual_norm_resolution","sse_resolution_threshold"):
            check.close("symmetric_whitening_"+name,law[name],expected[name],atol=0,rtol=3e-8)
        if law["resolution_multiplier"]!=64 or np.linalg.eigvalsh(np.asarray(law["scale_matrix"])).min()<=0:
            raise AssertionError("Residual rule or positive scale changed")
        minratio=min(minratio,expected["sse"]/expected["sse_resolution_threshold"])
        forecasts[law["member"],law["method_id"]]=law
    if seen!={(member,method,ref) for member,method in mains for ref in REFERENCES}:
        raise AssertionError("Some matched controls omitted")
    return check,forecasts,{"audited_transfer_predictions_sha256":sha(args.predictions),
        "audited_predictions_sha256":sha(args.predictions),
        "audited_prepare_sha256":sha(args.prepare),"audited_main_predictions_sha256":sha(args.main_predictions),
        "forecast_laws":420,"minimum_SSE_to_resolution_threshold_ratio":minratio,
        "independent_formula":"70digit saddle-point solve; no explicit inverse; orthogonally invariant whitening norms",
        "scope":"POST_REVEAL_DEV_DIAGNOSTIC_ON_ORIGINAL_THREE_RECIPES",
        "new_material_eta_access":0,"new_future_final_eta_access":0,"future_final_runs_protected":35,
        "scale_not_covariance":"predictive covariance=df/(df-2)*shape=2shape",**gate_result}

def crps_integral(location,scale,value):
    z=(value-location)/scale
    a,ea=quad(lambda u:t.cdf(u,4)**2,-np.inf,z,epsabs=2e-10,epsrel=2e-10,limit=180)
    b,eb=quad(lambda u:t.sf(u,4)**2,z,np.inf,epsabs=2e-10,epsrel=2e-10,limit=180)
    return scale*(a+b),scale*(ea+eb)

def score_audit(args,check,forecasts,result):
    target,scores=load(args.targets),load(args.scores)
    if target["predictions_sha256"]!=sha(args.main_predictions) or target["prepare_sha256"]!=sha(args.prepare):
        raise AssertionError("Only original common120target cells may be reused")
    if scores["predictions_sha256"]!=sha(args.predictions) or scores["targets_sha256"]!=sha(args.targets):
        raise AssertionError("Scores not bound to declared diagnostic predictions and old targets")
    if scores["study_role"]!="POST_REVEAL_DEV_DIAGNOSTIC" or scores["new_raw_eta_access"]!=0 or scores["new_target_access"]!=0:
        raise AssertionError("Consumed targets misclassified or new raw access occurred")
    truths={row["member"]:row for row in target["curves"]}
    rows={(row["member"],row["method_id"]):row for row in scores["curve_scores"]}
    points={(row["member"],row["method_id"],row["target_index"]):row for row in scores["point_terms"]}
    if len(truths)!=15 or len(rows)!=420 or len(scores["curve_scores"])!=420 or set(rows)!=set(forecasts):
        raise AssertionError("Invalid diagnostic target/forecast population")
    if len(points)!=3360 or len(scores["point_terms"])!=3360:
        raise AssertionError("Exact3360marginal terms required")
    # CRPS energy constant is computed by its CDF integral, not production's
    # beta-function expression; all3360 values are independently reconstructed.
    half_energy,constant_error=quad(lambda z:2*t.cdf(z,4)*t.sf(z,4),0,np.inf,
                                   epsabs=2e-12,epsrel=2e-12,limit=180)
    metric_rows={}
    for key,law in forecasts.items():
        truth=truths[key[0]]
        if truth["target_indices"]!=TARGET or truth["recipe"]!=law["recipe"]:
            raise AssertionError("Diagnostic fixed panel/group changed")
        y=np.asarray(truth["values"])
        mu,shape=np.asarray(law["location"]),np.asarray(law["scale_matrix"])
        scale=np.sqrt(np.diag(shape))
        error=y-mu
        z=error/scale
        crps=scale*(z*(2*t.cdf(z,4)-1)+2*(4+z*z)*t.pdf(z,4)/3-half_energy)
        cutoff=t.ppf(.95,4)
        expected={"crps":float(np.mean(crps)),
            "joint_nll_per_dimension":-float(multivariate_t.logpdf(y,loc=mu,shape=shape,df=4))/8,
            "log_rmse":float(np.sqrt(np.mean(error**2))),
            "coverage90":float(np.mean(np.abs(error)<=cutoff*scale)),
            "width90":float(np.mean(2*cutoff*scale))}
        metric_rows[key]=expected
        for metric,value in expected.items():
            check.close("independent_curve_"+metric,rows[key][metric],value,atol=2e-8,rtol=2e-8)
        for j,index in enumerate(TARGET):
            point=points[key[0],key[1],index]
            for field in ("recipe","run_id","thermal_history","mask_method_id","covariance_id"):
                if point[field]!=law[field] or rows[key][field]!=law[field]:
                    raise AssertionError("Scored law identity mismatch: "+field)
            for metric,value in (("truth",y[j]),("location",mu[j]),("t_scale",scale[j]),
                ("t_sd",scale[j]*math.sqrt(2)),("crps",crps[j]),
                ("squared_error",error[j]**2),("width90",2*cutoff*scale[j])):
                check.close("independent_point_"+metric,point[metric],value,atol=2e-8,rtol=2e-8)
            if point["df"]!=4 or point["coverage90"]!=bool(abs(error[j])<=cutoff*scale[j]):
                raise AssertionError("Point t scale/df/coverage mismatch")
    recipes=sorted({row["recipe"] for row in truths.values()})
    methodids=sorted({method for _,method in forecasts})
    recipes_saved={(row["recipe"],row["method_id"]):row for row in scores["recipe_scores"]}
    macros_saved={row["method_id"]:row for row in scores["macro_by_method"]}
    if len(recipes)!=3 or len(methodids)!=28 or len(recipes_saved)!=84 or len(macros_saved)!=28:
        raise AssertionError("Expected84recipe rows and28macro rows")
    independent_macro={}
    for method in methodids:
        collected={name:[] for name in METRICS}
        for recipe in recipes:
            selected=[value for (member,m),value in metric_rows.items() if m==method and truths[member]["recipe"]==recipe]
            if len(selected)!=5:
                raise AssertionError("All five runs per recipe required")
            for metric in METRICS:
                average=float(np.mean([row[metric] for row in selected]))
                check.close("recipe_"+metric,recipes_saved[recipe,method][metric],average,atol=2e-8,rtol=2e-8)
                collected[metric].append(average)
        independent_macro[method]={name:float(np.mean(values)) for name,values in collected.items()}
        for metric,value in independent_macro[method].items():
            check.close("macro_"+metric,macros_saved[method][metric],value,atol=2e-8,rtol=2e-8)
    selected_terms=[]
    for ref in REFERENCES:
        universe=[(key,index) for key in sorted(forecasts) if forecasts[key]["covariance_id"]==ref for index in TARGET]
        selected_terms.extend(universe[int(i)] for i in np.linspace(0,len(universe)-1,20,dtype=int))
    direct=[]
    for key,index in selected_terms:
        law=forecasts[key]
        j=TARGET.index(index)
        value,err=crps_integral(law["location"][j],math.sqrt(law["scale_matrix"][j][j]),truths[key[0]]["values"][j])
        check.close("direct_CDF_integral_CRPS",points[key[0],key[1],index]["crps"],value,atol=3e-8,rtol=3e-8)
        direct.append({"member":key[0],"method_id":key[1],"target_index":index,"quadrature_error":err})
    result.update({"audited_scores_sha256":sha(args.scores),"audited_targets_sha256":sha(args.targets),
        "CRPS_terms_reconstructed":3360,"joint_student_NLL_recomputed":420,
        "CRPS_constant_CDF_integral":half_energy,"constant_quad_error":constant_error,
        "direct_CDF_quadrature_terms":direct,"independent_macro":independent_macro,
        "target_scope":"Only original120consumed development targets; not a new confirmation"})

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("phase",choices=["laws","scores"])
    for name in ("authorization","prepare","main-predictions","predictions","main-audit","gate","output"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--targets")
    parser.add_argument("--scores")
    args=parser.parse_args()
    try:
        authorize(args)
        checks,forecasts,result=law_audit(args)
        if args.phase=="scores": score_audit(args,checks,forecasts,result)
        report={"status":"PASS","actual_utc":actual_utc(),"phase":args.phase,
            "authorization_sha256":sha(args.authorization),"auditor_source_sha256":sha(__file__),
            "auditor_math_sha256":sha(Path(__file__).with_name("resin_independent_math.py")),
            "comparisons":checks.groups,"production_imported":False,"CSV_opened":False,
            "new_model_update_in_audit":False,"mpmath_version":mp.__version__,"minimum_decimal_precision":70,**result}
    except Exception as error:
        write_new(args.output,{"status":"FAIL","actual_utc":actual_utc(),
            "error_type":type(error).__name__,"error":str(error)})
        raise
    write_new(args.output,report)
    print("Universal Student independent audit PASS;420augmented-system laws; phase="+args.phase)

if __name__=="__main__":
    main()
