"""Read-only secondary composition and recorded-compute diagnostics.

Uses all four predeclared element signatures, all 42 method/budget summaries,
and stored acquisition timings. No model imports, fits or new probability scores.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values):
    return math.fsum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    base = args.material_dir.resolve()
    resultfile = base / "score_once/RESULTS.json"
    auditfile = base / "independent_review/independent_score_execution_once/INDEPENDENT_FRESH160_ALL_SCORES_AUDIT.json"
    auditplan = base / "independent_review/ALL_SCORE_AUDIT_PLAN_REGISTERED.json"
    annotationfile = base / "diagnostics/VISIBLE_ELEMENT_ANNOTATIONS.json"
    audit = read(auditfile)
    assert audit["status"] == "PASS" and audit["plan_sha256"] == sha(auditplan)
    assert any(Path(v["file"]).resolve() == resultfile and v["sha256"] == sha(resultfile)
               for v in read(auditplan)["inputs"].values())
    result = read(resultfile)
    annotations = read(annotationfile)
    signatures = {r["component_id"]: r["signature"] for r in annotations["annotations"] if r["role"] == "terminal"}
    assert len(signatures) == 96
    grouped = {}
    for key in ("main_forecast_scores", "reference_forecast_scores"):
        for row in result[key]:
            groupkey = (row["policy"], row.get("method", "BMA"), row["budget"])
            grouped.setdefault(groupkey, []).append(row)
    assert len(grouped) == 42
    summaries = []
    closures = []
    for (policy, predictor, budget), rows in grouped.items():
        assert len(rows) == 96 and {r["component_id"] for r in rows} == set(signatures)
        blocks = []
        for signature in sorted(set(signatures.values())):
            scores = [r["score"] for r in rows if signatures[r["component_id"]] == signature]
            entry = {"policy": policy, "predictor": predictor, "budget": budget,
                     "element_signature": signature, "groups": len(scores),
                     "CRPS_native_mean": mean([s["CRPS_native"] for s in scores]),
                     "RMSE_median_native": math.sqrt(mean([s["squared_error_native"] for s in scores])),
                     "coverage90": mean([int(s["intervals"]["0.9"]["covered"]) for s in scores]),
                     "width90_native_mean": mean([s["intervals"]["0.9"]["width"] for s in scores])}
            summaries.append(entry)
            blocks.append(entry)
        globalmean = mean([r["score"]["CRPS_native"] for r in rows])
        blockmean = math.fsum(r["groups"] * r["CRPS_native_mean"] for r in blocks) / 96
        assert abs(globalmean - blockmean) <= 1e-11
        closures.append(abs(globalmean - blockmean))
    timings = {}
    for policy in result["actual_selected_pool_sequences"]:
        counts = {}
        elapsed = []
        pairs = 0
        signed_min, signed_max = [], []
        for step in range(8):
            record = read(base / f"forecast_once/prequential/{policy}_step{step:02}_FORECAST.json")
            diagnostic = record["acquisition_diagnostics"]
            for level in diagnostic.get("levels", []):
                d = level["diagnostics"]
                elapsed.append(float(d["elapsed_seconds"]))
                pairs += d["full_pair_evaluations"]
                for row in d["per_query"]:
                    if "mean_signed_between" in row:
                        signed_min.append(row["mean_signed_between"])
                        signed_max.append(row["mean_signed_between"])
            if "selected_count" in diagnostic:
                n = str(diagnostic["selected_count"])
                counts[n] = counts.get(n, 0) + 1
        timings[policy] = {"recorded_readout_seconds_sum": math.fsum(elapsed) if elapsed else None,
                           "readout_level_calls": len(elapsed),
                           "full_pair_evaluations_all_refinement_levels": pairs,
                           "accepted_quadrature_level_counts": counts,
                           "minimum_signed_between_mean_across_actual_calls": min(signed_min) if signed_min else None,
                           "maximum_signed_between_mean_across_actual_calls": max(signed_max) if signed_max else None,
                           "null_timing_means_not_instrumented_not_zero": not bool(elapsed),
                           "not_total_fit_runtime_or_physical_experiment_cost": True}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "ALL_ELEMENT_SUMMARIES.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    final = {"status": "READ_ONLY_SECONDARY_ALL_GROUP_DIAGNOSTICS",
             "actual_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "source_sha256": sha(__file__), "results_sha256": sha(resultfile),
             "accepted_score_audit_sha256": sha(auditfile),
             "pre_reveal_annotation_sha256": sha(annotationfile),
             "group_sizes": annotations["counts"]["terminal"],
             "all_method_budget_subgroups": summaries,
             "maximum_group_weighted_CRPS_closure": max(closures),
             "stored_compute_diagnostics": timings,
             "scope": "Exploratory composition diagnosis within one consumed development cohort; no independent subgroup efficacy, new model selection, refit, score recomputation or statistical significance claim."}
    with (args.output_dir / "DIAGNOSTICS.json").open("x", encoding="utf-8") as stream:
        json.dump(final, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"subgroup_rows": len(summaries), "maximum_mean_closure": max(closures),
                      "stored_compute_diagnostics": timings}))


if __name__ == "__main__":
    main()
