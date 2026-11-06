"""First evaluation of closed, saved prequential laws; no refitting or selection."""
import argparse
from bisect import bisect_right
from fractions import Fraction
import math
from pathlib import Path

import transport_io as io
from empirical_native import canonical, crps


def empirical_score(target, law):
    value = canonical(law)
    support, mass = value["support"], value["mass"]
    y = float(target)
    if not math.isfinite(y):
        raise ValueError("finite actual evaluator response required")
    def quantile(probability):
        total = Fraction(0)
        for x, weight in zip(support, mass):
            total += weight
            if total >= probability:
                return x
        raise AssertionError("unit exact mass must reach quantile")
    median = quantile(Fraction(1, 2))
    intervals = {}
    for level, lower, upper in (("0.8", Fraction(1, 10), Fraction(9, 10)),
                                 ("0.9", Fraction(1, 20), Fraction(19, 20)),
                                 ("0.95", Fraction(1, 40), Fraction(39, 40))):
        a, b = quantile(lower), quantile(upper)
        intervals[level] = {"lower": a, "upper": b, "width": b - a, "covered": a <= y <= b}
    left = sum((w for x, w in zip(support, mass) if x < y), Fraction(0))
    right = sum((w for x, w in zip(support, mass) if x <= y), Fraction(0))
    return {"CRPS_native": crps(law, y), "squared_error_native": (median - y) ** 2,
            "predictive_median_native": median, "intervals": intervals,
            "PIT_left": float(left), "PIT_right": float(right), "PIT_midpoint": float((left + right) / 2),
            "PIT_kind": "discrete_CDF_interval_not_randomized_uniform_sample",
            "log_density_native": None, "density_reason": "finite atomic prediction;continuous NLL not comparable",
            "quantile_probabilities_are_exact_rationals": True}


def aggregate(rows):
    n = len(rows)
    return {"groups": n, "CRPS_native_mean": math.fsum(r["CRPS_native"] for r in rows) / n,
            "RMSE_median_native": math.sqrt(math.fsum(r["squared_error_native"] for r in rows) / n),
            "coverage": {level: sum(r["intervals"][level]["covered"] for r in rows) / n
                         for level in ("0.8", "0.9", "0.95")},
            "width_native_mean": {level: math.fsum(r["intervals"][level]["width"] for r in rows) / n
                                  for level in ("0.8", "0.9", "0.95")}}


def evaluate_saved_envelope(envelope, target):
    """Exact stored-law score difference at any real binary-float target.

    This checks the observed score gap using a pretarget piecewise-linear
    certificate. It neither certifies future risk improvement nor calibration.
    """
    knots = [Fraction.from_float(float(v)) for v in envelope["knots_native"]]
    values = [Fraction(v) for v in envelope["knot_gaps_exact_fraction"]]
    y = Fraction.from_float(float(target))
    index = bisect_right(knots, y) - 1
    if index < 0:
        result = values[0]
    elif index >= len(knots) - 1:
        result = values[-1]
    else:
        t = (y - knots[index]) / (knots[index + 1] - knots[index])
        result = (1 - t) * values[index] + t * values[index + 1]
    if not Fraction(envelope["lower_exact_fraction"]) <= result <= Fraction(envelope["upper_exact_fraction"]):
        raise AssertionError("exact saved-law score gap outside pretarget envelope")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--forecast-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plan, roles = io.validate(args.plan, args.approval, "score_saved_batches")
    modules = io.bootstrap(plan)
    scalar = io.scalar_scorer(plan)
    folder = args.forecast_dir.resolve()
    complete = io.read(folder / "ALL_BATCHES_FROZEN.json")
    if (complete["status"] != "ALL1344_PREQUENTIAL_LAWS_AND192_TRUE_OUTCOMES_COMPLETE"
            or complete["source_sha256"] != io.sources() or complete["plan_sha256"] != io.sha(args.plan)):
        raise ValueError("complete exact three-batch run required")
    io.verify_payloads(folder, complete["files_sha256"])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    io.write(output / "SCORE_START.json", {"actual_utc": io.utc(), "source_sha256": io.sources(),
             "plan_sha256": io.sha(args.plan), "all_batch_freeze_sha256": io.sha(folder / "ALL_BATCHES_FROZEN.json")})
    scalar_cache, law_cache = {}, {}
    def scalar_cached(y, law):
        key = (y, json_key(law))
        if key not in scalar_cache:
            scalar_cache[key] = scalar(y, law)
        return scalar_cache[key]
    records, envelopes_observed = [], []
    for batch, role in enumerate(io.BATCHES):
        truth = io.read(folder / role / "TRUE_BATCH_REVEAL.json")["rows"]
        lookup = {r["identity"]["component_id"]: r["native_target"] for r in truth}
        forecasts = io.read(folder / role / "FORECASTS.json")["records"]
        scored = {}
        for row in forecasts:
            component = row["identity"]["component_id"]
            y = lookup[component]
            key = (y, json_key(row["law"]))
            if key not in law_cache:
                law_cache[key] = (empirical_score(y, row["law"]) if row["law"]["family"] == "weighted_empirical"
                                  else modules["probability"].score_law(y, row["law"], scalar_cached))
            result = {"batch": batch + 1, "method": row["method"], "component_id": component,
                      "native_target": y, "score": law_cache[key]}
            records.append(result)
            scored[(row["method"], component)] = result["score"]["CRPS_native"]
        envs = io.read(folder / role / "PRETARGET_LOSS_GAP_ENVELOPES.json")["records"]
        for row in envs:
            component, method = row["component_id"], row["method"]
            exact = evaluate_saved_envelope(row["loss_gap_envelope"], lookup[component])
            floating = scored[(method, component)] - scored[("RF_identity", component)]
            error = abs(floating - float(exact))
            if error > 1e-9 + 1e-11 * max(abs(floating), abs(float(exact))):
                raise ArithmeticError("floating CRPS gap disagrees with exact saved-law envelope")
            envelopes_observed.append({"batch": batch + 1, "method": method, "component_id": component,
                                       "exact_observed_loss_gap_fraction": str(exact),
                                       "floating_CRPS_subtraction_error": error})
    if len(records) != 1344 or len(envelopes_observed) != 768:
        raise ValueError("exact seven-by192 scores/four-by192 gap checks required")
    summaries = {m: {str(batch): aggregate([r["score"] for r in records if r["method"] == m
                        and (batch == 0 or r["batch"] == batch)]) for batch in (0, 1, 2, 3)} for m in io.METHODS}
    candidate = {r["component_id"]: r["score"]["CRPS_native"] for r in records if r["method"] == "RF_auto4"}
    paired = {}
    for method in io.METHODS:
        if method == "RF_auto4":
            continue
        other = {r["component_id"]: r["score"]["CRPS_native"] for r in records if r["method"] == method}
        differences = [candidate[key] - other[key] for key in candidate]
        mean = math.fsum(differences) / 192
        paired[method] = {"groups": 192, "candidate_minus_other_CRPS_mean": mean,
                          "standard_error_descriptive_only": math.sqrt(math.fsum((v - mean) ** 2 for v in differences) / 191 / 192),
                          "better": sum(v < 0 for v in differences), "worse": sum(v > 0 for v in differences),
                          "exact_ties": sum(v == 0 for v in differences), "all_differences": differences,
                          "not_formal_IID_campaign_interval_or_future_risk_certificate": True}
    selected = io.read(folder / "DESCRIPTOR_SELECTION_FROZEN.json")
    result = {"status": "COMPLETE_REFERENCE_TRANSPORT_FRESH192_DEVELOPMENT",
              "actual_utc": io.utc(), "source_sha256": io.sources(), "plan_sha256": io.sha(args.plan),
              "all_batch_freeze_sha256": io.sha(folder / "ALL_BATCHES_FROZEN.json"),
              "summaries": summaries, "paired_RF_auto4_minus_controls": paired, "scores": records,
              "observed_loss_gap_envelope_checks": envelopes_observed,
              "selected_descriptor_index": selected["auto_index"], "anchor_descriptor_index": selected["anchor_index"],
              "new_material_groups": 192, "prequential_validation_groups": 192, "score_coordinates": 1344,
              "new_native_cells": 576, "reused_historical168_cells_separate": 504, "actual_sklearn_fits": 11,
              "all192_same_computational_source_development_not_external_wet_confirmation": True,
              "fixed_disclosure_schedule_not_activequery_or_experimental_saving_proof": True,
              "support_preserving_scalar_predictive_repair_not_arbitrary_mask_joint_generation": True,
              "legacy_growth128_not_refitted_or_rescored": True}
    io.write(output / "RESULTS.json", result)
    print({m: summaries[m]["0"]["CRPS_native_mean"] for m in io.METHODS})


def json_key(value):
    import json
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


if __name__ == "__main__":
    main()
