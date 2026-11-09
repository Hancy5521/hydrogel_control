"""Saved-law Student audit, deliberately independent of production baselines.

SOURCE ONLY until an exact-hash, actual-UTC root authorization exists.
Pre-Y reads only already released six-anchor main forecasts and coordinates.
Post-Y reads only the same authorized 120 development targets, never raw CSV.
No production baseline, frozen runtime or source-response reader is imported.
OLS/scale use QR; CRPS uses its defining CDF integral, not a Beta formula.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.linalg import solve_triangular
from scipy.stats import multivariate_t, t

PANEL = list(range(17, 25))
KINDS = ("constant_log_viscosity", "power_law")
METRICS = ("crps", "joint_nll_per_dimension", "log_rmse", "coverage90", "width90")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


class Check:
    def __init__(self):
        self.elements = 0
        self.maximum_absolute_error = {}

    def close(self, label, actual, expected, atol=2e-9, rtol=2e-8):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(label + ": different shape/nonfinite")
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        self.maximum_absolute_error[label] = max(error, self.maximum_absolute_error.get(label, 0.0))
        self.elements += actual.size
        if not np.allclose(actual, expected, atol=atol, rtol=rtol):
            raise AssertionError(label + ": difference " + str(error))


def authorize(args):
    reg = load(args.authorization)
    if (reg.get("scope") != "student_baseline_independent_audit" or
            reg.get("allowed_phase") != args.phase or
            reg.get("auditor_source_sha256") != sha(__file__)):
        raise PermissionError("Exact auditor hash, scope and phase authorization required")
    stamp = dt.datetime.fromisoformat(reg["actual_utc"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture timezone-aware UTC required")
    required = ["prepare", "gaussian_predictions", "student_predictions", "main_audit"]
    if args.phase == "post_y":
        required += ["targets", "scores", "student_pre_audit"]
    elif any(getattr(args, name) is not None for name in ("targets", "scores", "student_pre_audit")):
        raise PermissionError("Pre-Y accepts no target, score or post-Y payload arguments")
    for name in required:
        path = getattr(args, name)
        if path is None or reg["input_sha256"].get(name) != sha(path):
            raise PermissionError("Root-authorized exact input hash required: " + name)
    return reg


def qr_prediction(x, y, xt, kind):
    """QR solve differs from production lstsq plus Gram Cholesky."""
    x, y, xt = np.asarray(x, float), np.asarray(y, float), np.asarray(xt, float)
    if x.shape != (6,) or y.shape != (6,) or xt.shape != (8,) or len(set(x)) != 6:
        raise AssertionError("Exactly six distinct native anchors and eight targets required")
    if not np.isfinite(np.r_[x, y, xt]).all() or np.all(y == y[0]):
        raise AssertionError("Nonfinite/constant response is not an admissible Student fit")
    if kind not in KINDS:
        raise AssertionError("Unregistered baseline kind")
    design = np.ones((6, 1)) if kind == KINDS[0] else np.column_stack((np.ones(6), x))
    target = np.ones((8, 1)) if kind == KINDS[0] else np.column_stack((np.ones(8), xt))
    q, r = np.linalg.qr(design, mode="reduced")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise AssertionError("Rank-deficient baseline")
    beta = solve_triangular(r, q.T @ y, lower=False)
    residual = y - design @ beta
    sse = float(residual @ residual)
    # Verify the declared backward-error resolution bound using our QR beta.
    # This rejects an unresolved residual; it never sets an SSE/variance floor.
    resolution = 64 * np.finfo(float).eps * (np.linalg.norm(design, 2) * np.linalg.norm(beta, 2) + np.linalg.norm(y, 2))
    p = design.shape[1]
    df = 6 - p
    if not math.isfinite(sse) or sse <= resolution ** 2:
        raise AssertionError("Numerically unresolved Student residual; no replacement or floor")
    # X* (X'X)^-1 X*' = (R^-T X*')' (R^-T X*').
    transformed = solve_triangular(r.T, target.T, lower=True)
    shape = sse / df * (np.eye(8) + transformed.T @ transformed)
    return {"beta": beta, "sse": sse, "p": p, "df": df,
            "residual_norm_resolution": resolution, "sse_resolution_threshold": resolution ** 2,
            "location": target @ beta, "scale_matrix": shape,
            "predictive_covariance": shape * df / (df - 2)}


def pre_audit(args):
    prepared, gaussian, saved = load(args.prepare), load(args.gaussian_predictions), load(args.student_predictions)
    main_audit = load(args.main_audit)
    if (main_audit.get("status") != "PASS" or
            main_audit.get("audited_predictions_sha256") != sha(args.gaussian_predictions) or
            main_audit.get("audited_prepare_sha256") != sha(args.prepare)):
        raise AssertionError("Original independently audited exact main prediction bytes required")
    if (saved.get("schema") != "mlst-resin-anchor-baseline-predictions-v1" or
            saved.get("main_predictions_sha256") != sha(args.gaussian_predictions) or
            saved.get("main_prepare_sha256") != sha(args.prepare) or
            saved.get("new_material_eta_access") != 0 or saved.get("new_target_access") != 0 or
            saved.get("future_final_runs_protected") != 35):
        raise AssertionError("Saved Student law source/role binding mismatch")
    curves = {row["member"]: row for row in prepared["curves"]}
    mains = {(row["member"], row["method_id"]): row for row in gaussian["forecasts"]}
    if len(curves) != 15 or len(mains) != 210 or len(saved["forecasts"]) != 420:
        raise AssertionError("Exact preregistered curve/method population required")
    if len({row["recipe"] for row in curves.values()}) != 3:
        raise AssertionError("Three development recipe groups required")
    checks, seen, unique_anchors, forecast_map = Check(), set(), set(), {}
    for law in saved["forecasts"]:
        key = law["member"], law["mask_method_id"], law["baseline_kind"]
        if key in seen or key[:2] not in mains or key[2] not in KINDS:
            raise AssertionError("Duplicate/unregistered Student prediction")
        seen.add(key)
        parent, curve = mains[key[:2]], curves[key[0]]
        if law["method_id"] != key[2] + "__" + key[1] or law.get("kind") != key[2]:
            raise AssertionError("Baseline method identity mismatch")
        for field in ("recipe", "run_id", "thermal_history", "anchor_indices", "anchor_values", "target_indices"):
            if law[field] != parent[field]:
                raise AssertionError("Student received different information: " + field)
        if (law["source_reference_id"] != parent["reference_id"] or
                law["source_policy"] != parent["policy"] or law["target_indices"] != PANEL):
            raise AssertionError("Parent reference/mask/panel mismatch")
        anchors = law["anchor_indices"]
        if (len(set(anchors)) != 6 or len(anchors) != 6 or not {0, 1}.issubset(anchors) or
                not set(anchors).issubset(range(17))):
            raise AssertionError("Student uses more than its own six anchors")
        unique_anchors.update((key[0], index) for index in anchors)
        x = np.asarray(curve["log_gamma"], float)
        expected = qr_prediction(x[anchors], parent["anchor_values"], x[PANEL], key[2])
        if (law.get("distribution") != "multivariate_student_t" or
                law["n"] != 6 or law["p"] != expected["p"] or law["df"] != expected["df"] or
                law.get("scale_is_covariance") is not False or law.get("residual_noise_in_prediction") is not True or
                law.get("prior") != "p(beta,sigma_squared) proportional to 1/sigma_squared"):
            raise AssertionError("Degrees of freedom/scale/likelihood/prior mismatch")
        for field in ("beta", "sse", "location", "scale_matrix", "predictive_covariance"):
            checks.close("QR_" + field, law[field], expected[field])
        if law.get("resolution_multiplier") != 64:
            raise AssertionError("Registered backward-error multiplier differs")
        for field in ("residual_norm_resolution", "sse_resolution_threshold"):
            checks.close("QR_" + field, law[field], expected[field], atol=0, rtol=2e-8)
        if np.linalg.eigvalsh(np.asarray(law["scale_matrix"])).min() <= 0:
            raise AssertionError("Student scale matrix not SPD")
        forecast_map[(law["member"], law["method_id"])] = law
    if seen != {(member, method, kind) for member, method in mains for kind in KINDS}:
        raise AssertionError("Not all two-baseline matched-parent predictions are present")
    return checks, forecast_map, {
        "forecast_laws_checked": 420, "own_anchor_count_each": 6,
        "unique_existing_anchor_cells": len(unique_anchors),
        "new_material_eta_access": 0, "new_target_access": 0,
        "audited_baseline_predictions_sha256": sha(args.student_predictions),
        "audited_main_predictions_sha256": sha(args.gaussian_predictions),
        "audited_prepare_sha256": sha(args.prepare),
        "future_final_runs_protected": 35,
        "independent_OLS": "QR beta and triangular-R predictive Gram, no production imports",
        "unresolved_residual_policy": "reject at declared backward-error bound, never variance flooring",
        "scale_vs_covariance": "shape times df/(df-2), df5 constant and df4 power-law"}


def crps_cdf_integral(location, scale, df, value):
    a = (value - location) / scale
    left, le = quad(lambda z: t.cdf(z, df) ** 2, -np.inf, a, epsabs=2e-10, epsrel=2e-10, limit=150)
    right, re = quad(lambda z: t.sf(z, df) ** 2, a, np.inf, epsabs=2e-10, epsrel=2e-10, limit=150)
    return scale * (left + right), scale * (le + re)


def post_audit(args, checks, forecasts, result):
    previous, targets, scores = load(args.student_pre_audit), load(args.targets), load(args.scores)
    if (previous.get("status") != "PASS" or
            previous.get("audited_baseline_predictions_sha256") != sha(args.student_predictions) or
            previous.get("audited_main_predictions_sha256") != sha(args.gaussian_predictions)):
        raise AssertionError("Exact saved pre-Y Student audit PASS required")
    if (targets.get("predictions_sha256") != sha(args.gaussian_predictions) or
            targets.get("prepare_sha256") != sha(args.prepare) or len(targets["curves"]) != 15 or
            targets.get("future_final_runs_protected") != 35 or
            scores.get("predictions_sha256") != sha(args.student_predictions) or
            scores.get("targets_sha256") != sha(args.targets) or
            scores.get("new_material_eta_access") != 0 or scores.get("new_target_access") != 0):
        raise AssertionError("Only originally released common target cells may be scored")
    truths = {row["member"]: row for row in targets["curves"]}
    rows = {(row["member"], row["method_id"]): row for row in scores["curve_scores"]}
    points = {(row["member"], row["method_id"], row["target_index"]): row for row in scores["point_terms"]}
    if len(rows) != 420 or len(scores["curve_scores"]) != 420 or len(points) != 3360 or len(scores["point_terms"]) != 3360:
        raise AssertionError("Duplicate/missing score entries")
    if set(rows) != set(forecasts):
        raise AssertionError("Scored forecasts differ from frozen Student laws")
    curve_metrics = {}
    for key, law in forecasts.items():
        truth = truths[key[0]]
        if truth["target_indices"] != PANEL or truth["recipe"] != law["recipe"]:
            raise AssertionError("Truth grouping/panel mismatch")
        y = np.asarray(truth["values"], float)
        m, shape, df = np.asarray(law["location"]), np.asarray(law["scale_matrix"]), law["df"]
        scale, err = np.sqrt(np.diag(shape)), y - m
        inside = np.abs(err) <= t.ppf(.95, df) * scale
        width = 2 * t.ppf(.95, df) * scale
        nll = -float(multivariate_t.logpdf(y, loc=m, shape=shape, df=df)) / 8
        for i, idx in enumerate(PANEL):
            point = points[(key[0], key[1], idx)]
            for name in ("recipe", "run_id", "thermal_history", "baseline_kind", "mask_method_id"):
                if point[name] != law[name] or rows[key][name] != law[name]:
                    raise AssertionError("Score identity mismatch")
            for label, expected in (("truth", y[i]), ("location", m[i]), ("t_scale", scale[i]),
                                    ("t_sd", scale[i] * math.sqrt(df / (df - 2))),
                                    ("squared_error", err[i] ** 2), ("width90", width[i])):
                checks.close("point_" + label, point[label], expected)
            if point.get("df") != df or point["coverage90"] != bool(inside[i]):
                raise AssertionError("Student quantile coverage differs")
            if not math.isfinite(point["crps"]) or point["crps"] < 0:
                raise AssertionError("Invalid stored CRPS")
        # Check every aggregation; independent CRPS integrals below cover twenty
        # declared representative marginals rather than claiming all3360 quadratures.
        crps = np.mean([points[(key[0], key[1], idx)]["crps"] for idx in PANEL])
        expected = {"crps": crps, "joint_nll_per_dimension": nll,
                    "log_rmse": float(np.sqrt(np.mean(err ** 2))),
                    "coverage90": float(np.mean(inside)), "width90": float(np.mean(width))}
        for metric in METRICS:
            checks.close("curve_" + metric, rows[key][metric], expected[metric])
        curve_metrics[key] = expected
    selected = []
    for kind in KINDS:
        universe = [(key, idx) for key in sorted(forecasts) if forecasts[key]["baseline_kind"] == kind for idx in PANEL]
        positions = np.linspace(0, len(universe) - 1, 10, dtype=int)
        selected.extend(universe[int(i)] for i in positions)
    integral_rows = []
    for key, idx in selected:
        law = forecasts[key]
        i = PANEL.index(idx)
        expected, error = crps_cdf_integral(law["location"][i], math.sqrt(law["scale_matrix"][i][i]),
                                           law["df"], truths[key[0]]["values"][i])
        point = points[(key[0], key[1], idx)]
        checks.close("CRPS_direct_CDF_integral", point["crps"], expected, atol=2e-8, rtol=2e-8)
        if error > 2e-7:
            raise AssertionError("CDF quadrature error estimate exceeds audit bound")
        integral_rows.append({"member": key[0], "method_id": key[1], "target_index": idx,
                              "df": law["df"], "CDF_integral": expected, "quad_error_estimate": error,
                              "saved_CRPS": point["crps"]})
    recipe_rows = {(row["method_id"], row["recipe"]): row for row in scores["recipe_scores"]}
    macros = {row["method_id"]: row for row in scores["macro_by_method"]}
    methods = sorted({key[1] for key in forecasts})
    recipes = sorted({row["recipe"] for row in forecasts.values()})
    if len(recipe_rows) != 84 or len(scores["recipe_scores"]) != 84 or len(macros) != 28 or len(scores["macro_by_method"]) != 28:
        raise AssertionError("Exact recipe/method macro cardinalities required")
    for method in methods:
        aggregate = []
        for recipe in recipes:
            keys = [key for key, law in forecasts.items() if key[1] == method and law["recipe"] == recipe]
            if len(keys) != 5 or recipe_rows[method, recipe]["runs"] != 5:
                raise AssertionError("Five runs/recipe, not five independent recipes")
            metrics = {name: float(np.mean([curve_metrics[key][name] for key in keys])) for name in METRICS}
            for name in METRICS:
                checks.close("recipe_" + name, recipe_rows[method, recipe][name], metrics[name])
            aggregate.append(metrics)
        if macros[method]["recipes"] != 3:
            raise AssertionError("Equal three-recipe macro required")
        for name in METRICS:
            checks.close("macro_" + name, macros[method][name], np.mean([r[name] for r in aggregate]))
    result.update({"authorized_existing_target_cells_read": 120, "new_target_measurements": 0,
                   "joint_NLL_independent_scipy_checks": 420, "marginal_interval_checks": 3360,
                   "CRPS_independent_CDF_integrals": 20,
                   "CRPS_integral_selection": "per-kind10 evenly spaced over sorted(member,method,target) universe",
                   "CRPS_integral_rows": integral_rows,
                   "audited_targets_sha256": sha(args.targets), "audited_scores_sha256": sha(args.scores),
                   "CRPS_scope": "20 independent defining-integral checks; all3360 stored-term aggregation, not all3360 integrations"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("pre_y", "post_y"))
    for name in ("authorization", "prepare", "gaussian_predictions", "student_predictions", "main_audit", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    for name in ("targets", "scores", "student_pre_audit"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    authorize(args)
    checks, forecasts, result = pre_audit(args)
    if args.phase == "post_y":
        post_audit(args, checks, forecasts, result)
    result.update({"status": "PASS", "phase": args.phase,
                   "actual_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "authorization_sha256": sha(args.authorization), "auditor_source_sha256": sha(__file__),
                   "numeric_elements_compared": checks.elements,
                   "maximum_absolute_error": checks.maximum_absolute_error,
                   "production_baseline_imported": False, "raw_response_CSV_opened": False,
                   "material_targets_opened": args.phase == "post_y"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "PASS", "phase": args.phase, "output": str(args.output),
                      "numeric_elements_compared": checks.elements}))


if __name__ == "__main__":
    main()
