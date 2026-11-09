"""Target-free independent native-reference, mask and forecast audit.

No production module, original runtime, ZIP/CSV response reader or target
artifact is imported/opened. This command is safe only after root registers the
auditor and freezes the preparation/prediction artifacts. It never fits.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from resin_independent_math import (Comparisons, INPUT_SHA, MODEL_SHA, REFERENCES,
    TARGET, actual_utc, condition, joint, load, native_reference, planned_mask,
    recipe_roles, sha, write_new)

def audit(args):
    inputs, model = load(args.inputs, INPUT_SHA), load(args.model, MODEL_SHA)
    prepared, predictions = load(args.prepare), load(args.predictions)
    if predictions["prepare_sha256"] != sha(args.prepare):
        raise AssertionError("Prediction artifact does not bind exact prepared bytes")
    roles, specs = recipe_roles(inputs)
    if {name: record["role"] for name, record in prepared["roles"].items()} != roles:
        raise AssertionError("Prepared recipe role manifest differs from independent hash allocation")
    expected_members = {member for member, spec in specs.items() if roles[spec["recipe"]] == "dev"}
    curves = {curve["member"]: curve for curve in prepared["curves"]}
    if len(curves) != 15 or set(curves) != expected_members:
        raise AssertionError("Prepared population is not exact preregistered fifteen dev runs")
    comparisons, laws, method_lookup = Comparisons(), {}, {}
    seed = int(prepared["seed_128bit_hex"], 16)
    if len(prepared["seed_128bit_hex"]) != 32 or seed < 0 or seed >= 2**128:
        raise AssertionError("Exact saved OS128-bit seed required")
    rng = np.random.Generator(np.random.PCG64(seed))
    seed_receipt = load(Path(args.prepare).with_name("RANDOM_SEED.json"))
    if seed_receipt["seed_128bit_hex"] != prepared["seed_128bit_hex"] or seed_receipt["algorithm"] != "PCG64":
        raise AssertionError("Prepared random seed differs from original once-only seed receipt")
    if list(curves) != sorted(curves):
        raise AssertionError("Registered random generator iteration order changed")
    residual_eigenvalues = []
    expected_curve_methods = set()
    for member, curve in curves.items():
        spec = specs[member]
        if (curve["recipe"], curve["thermal_history"], curve["source_member_sha256"]) != (
                spec["recipe"], spec["thermal_history"], spec["source_member_sha256"]):
            raise AssertionError("Prepared group/source alias mismatch")
        # Independent admitted native coordinates; no response token read.
        gamma = np.array([float(value) for value in spec["native_gamma_raw_tokens_sorted"]])
        if gamma.shape != (25,) or np.any(np.diff(gamma) <= 0):
            raise AssertionError("Independent native grid invalid")
        t = np.log(gamma)
        comparisons.close("native_gamma", curve["native_gamma"], gamma, atol=0, rtol=0)
        comparisons.close("log_gamma", curve["log_gamma"], t, atol=1e-14, rtol=1e-14)
        m, k, b, mineig = native_reference(t, model)
        residual_eigenvalues.append(mineig)
        for key, expected in (("mean", m), ("covariance", k), ("cross", b)):
            comparisons.close("independent_reference_"+key, curve["reference"][key], expected)
        for refid, theta in (("gp_theta0", np.zeros(5)),
                             ("curvature_frozen_theta", np.asarray(model["theta"]))):
            law = curve["laws"][refid]
            comparisons.close("theta", law["theta"], theta, atol=0, rtol=0)
            mm, kk = joint(m, k, b, theta)
            comparisons.close("independent_joint_mean", law["mean"], mm)
            comparisons.close("independent_joint_covariance", law["covariance"], kk)
            laws[member, refid] = mm, kk
        own_methods = {}
        for method in curve["methods"]:
            methodid, refid, policy = method["method_id"], method["reference_id"], method["policy"]
            if methodid in own_methods or refid not in REFERENCES:
                raise AssertionError("Invalid/duplicate prepared method")
            if method["target_indices"] != TARGET:
                raise AssertionError("Scoring panel changed or query removed target")
            anchors = method["anchor_indices"]
            if (len(anchors) != 6 or len(set(anchors)) != 6 or not {0, 1}.issubset(anchors)
                    or not set(anchors).issubset(range(17))):
                raise AssertionError("Not six own admissible anchor cells")
            own_methods[methodid] = method
            method_lookup[member, methodid] = method
        if len(own_methods) != 14:
            raise AssertionError("Expected fourteen controls per dev curve")
        if not expected_curve_methods:
            expected_curve_methods = set(own_methods)
        elif set(own_methods) != expected_curve_methods:
            raise AssertionError("Method identities differ between dev runs")
        expected_random = [0,1] + rng.choice(np.arange(2,17), 4, replace=False).tolist()
        # Policies depend on covariance only; compare independent greedy plans.
        for refid in REFERENCES:
            for policy, fixed in (("fixed_first6", [0,1,2,3,4,5]),
                                  ("spread6", [0,1,2,7,11,16]),
                                  ("nearest_target6", [0,1,13,14,15,16]),
                                  ("max_variance6", None), ("expected_crps6", None),
                                  ("random_uniform6", "random")):
                matches = [method for method in own_methods.values()
                           if method["reference_id"] == refid and method["policy"] == policy
                           and method.get("mask_from_reference_id", refid) == refid]
                if len(matches) != 1:
                    raise AssertionError("Missing or duplicate own-reference policy: "+refid+policy)
                method = matches[0]
                if fixed is None:
                    expected_mask, _ = planned_mask(laws[member, refid][1], policy)
                    if method["anchor_indices"] != expected_mask:
                        raise AssertionError("Independent covariance planner mask differs")
                elif fixed != "random" and method["anchor_indices"] != fixed:
                    raise AssertionError("Registered fixed/spread/near-target anchor layout changed")
                elif fixed == "random" and method["anchor_indices"] != expected_random:
                    raise AssertionError("PCG64 mask differs from once-frozen OS seed and member order")
        random_masks = [method["anchor_indices"] for method in own_methods.values()
                        if method["policy"] == "random_uniform6"]
        if len(random_masks) != 2 or random_masks[0] != random_masks[1]:
            raise AssertionError("One random mask per curve must be shared by both references")
        cross_count = 0
        for method in own_methods.values():
            from_ref = method.get("mask_from_reference_id", method["reference_id"])
            if from_ref != method["reference_id"]:
                if method["policy"] != "cross_expected_crps6" or from_ref not in REFERENCES:
                    raise AssertionError("Only registered two cross-reference masks permitted")
                expected_mask, _ = planned_mask(laws[member, from_ref][1], "expected_crps6")
                if method["anchor_indices"] != expected_mask:
                    raise AssertionError("Cross-reference same-information mask mismatch")
                cross_count += 1
        if cross_count != 2:
            raise AssertionError("Both same-information cross-reference controls required")
    forecasts = predictions["forecasts"]
    if len(forecasts) != 210:
        raise AssertionError("Exact 210 pre-Y forecasts required")
    observed = {}
    seen = set()
    unique_anchors = set()
    for forecast in forecasts:
        key = forecast["member"], forecast["method_id"]
        if key in seen or key not in method_lookup:
            raise AssertionError("Duplicate or unregistered forecast")
        seen.add(key)
        method = method_lookup[key]
        if forecast["anchor_indices"] != method["anchor_indices"] or forecast["target_indices"] != TARGET:
            raise AssertionError("Forecast mask not frozen prepared mask")
        if forecast["reference_id"] != method["reference_id"]:
            raise AssertionError("Forecast reference differs from frozen method")
        spec = specs[forecast["member"]]
        if forecast["recipe"] != spec["recipe"] or forecast["thermal_history"] != spec["thermal_history"]:
            raise AssertionError("Forecast group/history alias mismatch")
        values = forecast["anchor_values"]
        if len(values) != 6 or not np.isfinite(values).all():
            raise AssertionError("Observed-only payload must be finite own-six readings")
        for index, value in zip(method["anchor_indices"], values):
            cell = forecast["member"], index
            if cell in observed and observed[cell] != value:
                raise AssertionError("Different methods receive inconsistent same observed cell")
            observed[cell] = value
            unique_anchors.add(cell)
        m, k = laws[forecast["member"], method["reference_id"]]
        pm, pk = condition(m, k, method["anchor_indices"], values)
        comparisons.close("full_conditional_mean", forecast["full_mean"], pm, atol=1e-7, rtol=1e-8)
        comparisons.close("full_conditional_covariance", forecast["full_covariance"], pk)
        comparisons.close("panel_conditional_mean", forecast["mean"], pm[TARGET], atol=1e-7, rtol=1e-8)
        comparisons.close("panel_conditional_covariance", forecast["covariance"], pk[np.ix_(TARGET,TARGET)])
        if np.linalg.eigvalsh(np.asarray(forecast["covariance"])).min() <= 0:
            raise AssertionError("Saved panel covariance not SPD")
    if seen != set(method_lookup):
        raise AssertionError("Missing authorized forecast")
    return {"status": "PASS", "actual_utc": actual_utc(),
            "audited_predictions_sha256": sha(args.predictions),
            "audited_prepare_sha256": sha(args.prepare),
            "input_masks_sha256": sha(args.inputs), "frozen_model_sha256": sha(args.model),
            "auditor_source_sha256": sha(__file__),
            "auditor_math_sha256": sha(Path(__file__).with_name("resin_independent_math.py")),
            "numeric_comparisons": comparisons.groups,
            "forecasts": len(forecasts), "dev_curves": len(curves), "dev_recipes": 3,
            "minimum_full_residual_eigenvalue": min(residual_eigenvalues),
            "unique_observed_anchor_cells": len(unique_anchors),
            "independent_formula": "direct Matern3/2+intercept+slope; B=K_rq F^T(FK_qqF^T)^-1/2; dense Schur",
            "material_targets_opened": False, "raw_response_csv_opened": False,
            "production_or_original_runtime_imported": False, "fits": 0,
            "random_seed_mask_reconstruction": "PASS: saved OS128seed, PCG64, sorted member order, one4withoutreplacement mask shared by refs",
            "limitations": ["Audits arithmetic and software roles, not physical batch independence or calibration."]}

def main():
    parser = argparse.ArgumentParser()
    for option in ("prepare", "predictions", "inputs", "model", "output"):
        parser.add_argument("--"+option, required=True)
    args = parser.parse_args()
    try:
        report = audit(args)
    except Exception as error:
        report = {"status": "FAIL", "actual_utc": actual_utc(), "error_type": type(error).__name__,
                  "error": str(error), "material_targets_opened": False}
        write_new(args.output, report)
        raise
    write_new(args.output, report)
    print("Independent pre-Y audit PASS: 15 native references, 210 forecast laws; targets unopened.")

if __name__ == "__main__":
    main()
