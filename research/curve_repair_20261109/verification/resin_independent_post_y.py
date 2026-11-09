"""Independent released-cell source trace and score audit; no SUT imports.

THIS SOURCE DOES NOT AUTHORIZE EXECUTION OR TARGET ACCESS. Root must release
the exact 120 development target cells after the independent pre-Y audit PASS.
This auditor then reads only the already used anchor union and released targets
from the original fifteen dev ZIP members, never future/final numeric eta.
"""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import zipfile
import numpy as np
from scipy.integrate import quad
from resin_independent_math import (Comparisons, INPUT_SHA, SOURCE_SHA, TARGET,
    actual_utc, load, recipe_roles, score_gaussian, sha, write_new)

KEYS = ("crps", "joint_nll_per_dimension", "log_rmse", "coverage90", "width90")

def audit(args):
    inputs, prepared = load(args.inputs, INPUT_SHA), load(args.prepare)
    predictions, targets, scores = load(args.predictions), load(args.targets), load(args.scores)
    preaudit, release = load(args.pre_audit), load(args.release)
    prediction_sha, prepare_sha = sha(args.predictions), sha(args.prepare)
    if preaudit["status"] != "PASS" or preaudit["audited_predictions_sha256"] != prediction_sha:
        raise PermissionError("Exact pre-Y audit PASS required")
    if preaudit["audited_prepare_sha256"] != prepare_sha:
        raise PermissionError("Pre-Y audit preparation identity differs")
    if (release.get("schema") != "mlst-resin-target-release-v1"
            or release.get("authorization") != "REVEAL_ONLY_120_DEV_TARGET_CELLS"
            or release.get("predictions_sha256") != prediction_sha
            or release.get("prepare_sha256") != prepare_sha
            or release.get("audit_sha256") != sha(args.pre_audit)):
        raise PermissionError("Exact root release scope/byte identities required")
    if (targets["predictions_sha256"] != prediction_sha or targets["prepare_sha256"] != prepare_sha
            or targets["root_release_sha256"] != sha(args.release)
            or targets["audit_sha256"] != sha(args.pre_audit)
            or scores["targets_sha256"] != sha(args.targets)
            or scores["predictions_sha256"] != prediction_sha or scores["prepare_sha256"] != prepare_sha):
        raise AssertionError("Released target/score byte chain changed")
    roles, specs = recipe_roles(inputs)
    dev = {m for m, s in specs.items() if roles[s["recipe"]] == "dev"}
    target_lookup = {c["member"]: c for c in targets["curves"]}
    if len(targets["curves"]) != 15 or set(target_lookup) != dev:
        raise AssertionError("Exact fifteen dev target curves required")
    forecasts = {(c["member"], c["method_id"]): c for c in predictions["forecasts"]}
    if len(forecasts) != 210:
        raise AssertionError("Exact 210 distinct frozen forecasts required")
    cells = set()
    for forecast in forecasts.values():
        if forecast["member"] not in dev:
            raise PermissionError("Future/final or unknown member in predictions")
        cells.update((forecast["member"], i) for i in forecast["anchor_indices"])
    anchor_count = len(cells)
    cells.update((member, i) for member in dev for i in TARGET)
    if sha(args.source_zip) != SOURCE_SHA:
        raise AssertionError("Original source ZIP changed")
    raw_values, raw_addresses = {}, []
    with zipfile.ZipFile(args.source_zip) as archive:
        for member in sorted(dev):
            spec = specs[member]
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest() != spec["source_member_sha256"]:
                raise AssertionError("Original dev run hash changed")
            lines = raw.decode("utf-16").splitlines()
            selected_indices = sorted(i for m, i in cells if m == member)
            for i in selected_indices:
                address = spec["source_line_by_rank_1based"][i]
                parts = lines[address-1].split("\t")
                if (len(parts) != 7 or parts[5].strip() != spec["native_gamma_raw_tokens_sorted"][i]
                        or parts[4] != "Dy_auto" or not 34.5 <= float(parts[2]) < 35.5):
                    raise AssertionError("Independent raw source input/address/status mismatch")
                cp = float(parts[3])
                if not math.isfinite(cp) or cp <= 0:
                    raise AssertionError("STOP_NO_REPLACEMENT invalid released source cell")
                raw_values[member, i] = math.log(cp/1000.0)
                raw_addresses.append({"member": member, "rank_1based": i+1,
                    "source_line_1based": address,
                    "scope": "released_dev_target" if i in TARGET else "already_used_dev_anchor"})
    comparisons = Comparisons()
    for member, target in target_lookup.items():
        if target["target_indices"] != TARGET or target["recipe"] != specs[member]["recipe"]:
            raise AssertionError("Released target panel/group changed")
        comparisons.close("raw_released_targets", target["values"],
                          [raw_values[member, i] for i in TARGET], atol=1e-14, rtol=1e-14)
    for (member, _), forecast in forecasts.items():
        comparisons.close("raw_already_used_anchors", forecast["anchor_values"],
            [raw_values[member, i] for i in forecast["anchor_indices"]], atol=1e-14, rtol=1e-14)
    expected_rows = {}
    quadrature_count, quadrature_max_error = 0, 0.0
    first_methods = {m: min(method for member, method in forecasts if member == m) for m in dev}
    for key, forecast in forecasts.items():
        member, method = key
        independent = score_gaussian(forecast["mean"], forecast["covariance"], target_lookup[member]["values"])
        independent["joint_nll_per_dimension"] = independent.pop("joint_nll_per_target")
        expected_rows[key] = independent
        # One predetermined target (rank18) of lexically first method per curve:
        # direct normal-energy quadrature, not the usual Gaussian CRPS shortcut.
        if method == first_methods[member]:
            mu = forecast["mean"][0]
            sd = math.sqrt(forecast["covariance"][0][0])
            truth = target_lookup[member]["values"][0]
            crossing = (truth-mu)/sd
            points = [crossing] if -12 < crossing < 12 else None
            absolute, _ = quad(lambda z: abs(mu+sd*z-truth)*math.exp(-z*z/2)/math.sqrt(2*math.pi),
                               -12, 12, points=points, epsabs=2e-12, epsrel=2e-12)
            crps_integral = absolute - sd/math.sqrt(math.pi)
            error = abs(crps_integral-independent["crps_points"][0])
            if error > 2e-9:
                raise AssertionError("Independent CRPS energy quadrature mismatch")
            quadrature_count += 1
            quadrature_max_error = max(quadrature_max_error, error)
    score_rows = {(r["member"],r["method_id"]): r for r in scores["curve_scores"]}
    if len(scores["curve_scores"]) != 210 or set(score_rows) != set(expected_rows):
        raise AssertionError("Score rows missing or repeated")
    for key, expected in expected_rows.items():
        for metric in KEYS:
            comparisons.close("curve_score_"+metric, score_rows[key][metric], expected[metric],
                              atol=3e-8, rtol=3e-9)
    point_rows = {(r["member"],r["method_id"],r["target_index"]):r for r in scores["point_terms"]}
    if len(scores["point_terms"]) != 1680 or len(point_rows) != 1680:
        raise AssertionError("Exactly 1680 unique same-target marginal CRPS terms required")
    for key, expected in expected_rows.items():
        member, method = key
        for j, target_index in enumerate(TARGET):
            point = point_rows[member, method, target_index]
            forecast = forecasts[key]
            truth = raw_values[member, target_index]
            mu = forecast["mean"][j]
            sd = math.sqrt(forecast["covariance"][j][j])
            for metric, value in (("crps",expected["crps_points"][j]), ("truth",truth), ("mean",mu),
                                  ("sd",sd), ("squared_error",(truth-mu)**2),
                                  ("width90",2*1.6448536269514722*sd)):
                comparisons.close("point_"+metric, point[metric], value, atol=3e-9,rtol=3e-9)
            if point["coverage90"] != (abs(truth-mu) <= 1.6448536269514722*sd):
                raise AssertionError("Independent point coverage differs")
    methods = sorted({method for _,method in forecasts})
    recipes = sorted({specs[m]["recipe"] for m in dev})
    if len(methods) != 14 or len(recipes) != 3:
        raise AssertionError("Unexpected method/recipe aggregation cardinality")
    recipe_rows = {(r["recipe"],r["method_id"]):r for r in scores["recipe_scores"]}
    macro_rows = {r["method_id"]: r for r in scores["macro_by_method"]}
    if len(recipe_rows) != 42 or len(scores["recipe_scores"]) != 42 or len(macro_rows) != 14:
        raise AssertionError("Expected 42 recipe and14 macro rows")
    computed_recipe, computed_macro = {}, {}
    for method in methods:
        for recipe in recipes:
            selected = [expected for (member,m),expected in expected_rows.items()
                        if m == method and specs[member]["recipe"] == recipe]
            if len(selected) != 5:
                raise AssertionError("Each recipe keeps all five runs")
            computed_recipe[recipe,method] = {k: float(np.mean([s[k] for s in selected])) for k in KEYS}
            for metric,value in computed_recipe[recipe,method].items():
                comparisons.close("recipe_macro_"+metric,recipe_rows[recipe,method][metric],value,
                                  atol=3e-8,rtol=3e-9)
        computed_macro[method] = {k:float(np.mean([computed_recipe[r,method][k] for r in recipes])) for k in KEYS}
        for metric,value in computed_macro[method].items():
            comparisons.close("all_recipe_macro_"+metric,macro_rows[method][metric],value,atol=3e-8,rtol=3e-9)
    first, proposed = "curvature_frozen_theta__fixed_first6", "curvature_frozen_theta__expected_crps6"
    deltas = {r: computed_recipe[r,proposed]["crps"]-computed_recipe[r,first]["crps"] for r in recipes}
    primary = float(np.mean(list(deltas.values())))
    comparisons.close("primary_recipe_macro_CRPS_delta",scores["primary"]["recipe_macro_CRPS_delta"],primary)
    reported_deltas = {row["recipe"]:row for row in scores["primary"]["per_recipe"]}
    if len(reported_deltas) != 3:
        raise AssertionError("Three primary recipe deltas required")
    for recipe,value in deltas.items():
        comparisons.close("primary_per_recipe_delta",reported_deltas[recipe]["delta"],value)
    if scores["cost"]["unique_anchor_union_cells"] != anchor_count:
        raise AssertionError("Research-level anchor union cost underreported")
    if scores["cost"]["per_method_anchor_cells"] != 90 or scores["cost"]["unique_scoring_target_cells"] != 120:
        raise AssertionError("Per-method or target accounting differs")
    return {"status":"PASS", "actual_utc":actual_utc(),
            "audited_scores_sha256":sha(args.scores), "audited_targets_sha256":sha(args.targets),
            "audited_predictions_sha256":prediction_sha, "audited_prepare_sha256":prepare_sha,
            "root_release_sha256":sha(args.release), "pre_y_audit_sha256":sha(args.pre_audit),
            "auditor_source_sha256":sha(__file__),
            "auditor_math_sha256":sha(Path(__file__).with_name("resin_independent_math.py")),
            "source_zip_sha256":sha(args.source_zip),"input_masks_sha256":sha(args.inputs),
            "numeric_comparisons":comparisons.groups,
            "raw_access_scope":{"dev_runs":15,"dev_recipes":3,"already_used_anchor_union":anchor_count,
                "released_target_cells":120,"source_addresses":raw_addresses,
                "future_final_eta_numeric_cells":0,"new_source_outcome_cells_beyond_released_scope":0},
            "quadrature_check":{"terms":quadrature_count,"max_absolute_error":quadrature_max_error},
            "independent_primary_delta":primary,"independent_per_recipe_delta":deltas,
            "independent_macro":computed_macro,"fits":0,
            "production_or_original_runtime_imported":False,
            "limitation":"Three recipe development stress only; no independent hydrogel confirmation, reliable calibration, or measured real experiment saving."}

def main():
    parser=argparse.ArgumentParser()
    for option in ("prepare","predictions","targets","scores","inputs","source-zip","pre-audit","release","output"):
        parser.add_argument("--"+option,required=True)
    args=parser.parse_args()
    try:
        report=audit(args)
    except Exception as error:
        write_new(args.output,{"status":"FAIL","actual_utc":actual_utc(),
            "error_type":type(error).__name__,"error":str(error),
            "target_read_authorization":"Exact root release checked before raw target access"})
        raise
    write_new(args.output,report)
    print("Independent post-Y source/score audit PASS: 120 released targets,210forecast laws,1680CRPS terms.")

if __name__=="__main__":
    main()
