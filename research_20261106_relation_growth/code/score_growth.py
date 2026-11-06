"""Score only the saved new-batch laws; never fit or propose another hypothesis."""
import argparse
import math
from pathlib import Path

import experiment_io as io


def aggregate(rows):
    return {"groups": len(rows), "CRPS_native_mean": math.fsum(r["CRPS_native"] for r in rows) / len(rows),
            "RMSE_median_native": math.sqrt(math.fsum(r["squared_error_native"] for r in rows) / len(rows)),
            "coverage": {level: sum(r["intervals"][level]["covered"] for r in rows) / len(rows)
                         for level in ("0.8", "0.9", "0.95")},
            "width_native_mean": {level: math.fsum(r["intervals"][level]["width"] for r in rows) / len(rows)
                                  for level in ("0.8", "0.9", "0.95")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--forecast-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plan, roles = io.validate(args.plan, args.approval, "score_saved_batches")
    io.bootstrap_core(plan)
    import probability
    scalar = io.scalar_scorer(plan)
    folder = args.forecast_dir.resolve()
    allfreeze = io.read(folder / "ALL_BATCHES_FROZEN.json")
    if (allfreeze["status"] != "ALL672_PREQUENTIAL_LAWS_AND96_TRUE_OUTCOMES_COMPLETE"
            or allfreeze["source_sha256"] != io.sources() or allfreeze["plan_sha256"] != io.sha(args.plan)):
        raise ValueError("complete matching three-batch run required")
    for name, pin in allfreeze["files_sha256"].items():
        path = (folder / name).resolve()
        path.relative_to(folder)
        if io.sha(path) != pin:
            raise ValueError("saved prequential payload changed")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    io.write(output / "SCORE_START.json", {"actual_utc": io.utc(), "all_batch_freeze_sha256": io.sha(folder / "ALL_BATCHES_FROZEN.json"),
             "plan_sha256": io.sha(args.plan), "source_sha256": io.sources()})
    cache, scalar_cache = {}, {}
    def scalar_score(y, law):
        key = (y, io.jsha(law))
        if key not in scalar_cache:
            scalar_cache[key] = scalar(y, law)
        return scalar_cache[key]
    records = []
    for batch, role in enumerate(io.BATCHES):
        truth = io.read(folder / role / "TRUE_BATCH_REVEAL.json")
        lookup = {r["identity"]["component_id"]: r["native_target"] for r in truth["rows"]}
        before = io.read(folder / role / "FORECASTS.json")
        for row in before["records"]:
            y = lookup[row["identity"]["component_id"]]
            key = (y, io.jsha(row["law"]))
            if key not in cache:
                cache[key] = probability.score_law(y, row["law"], scalar_score)
            records.append({"batch": batch + 1, "method": row["method"],
                            "component_id": row["identity"]["component_id"], "native_target": y, "score": cache[key]})
    if len(records) != 672:
        raise ValueError("exact seven by96 score coordinates required")
    summaries = {method: {str(batch): aggregate([r["score"] for r in records if r["method"] == method
                         and (batch == 0 or r["batch"] == batch)]) for batch in (0, 1, 2, 3)} for method in io.METHODS}
    candidate = {r["component_id"]: r["score"]["CRPS_native"] for r in records if r["method"] == "grown4_BMA"}
    paired = {}
    for method in io.METHODS:
        if method == "grown4_BMA":
            continue
        other = {r["component_id"]: r["score"]["CRPS_native"] for r in records if r["method"] == method}
        differences = [candidate[key] - other[key] for key in candidate]
        mu = math.fsum(differences) / 96
        paired[method] = {"groups": 96, "candidate_minus_other_CRPS_mean": mu,
                          "standard_error_descriptive_only": math.sqrt(math.fsum((v - mu) ** 2 for v in differences) / 95 / 96),
                          "better": sum(v < 0 for v in differences), "worse": sum(v > 0 for v in differences),
                          "exact_ties": sum(v == 0 for v in differences), "all_differences": differences,
                          "not_iid_campaign_uncertainty_or_formal_noninferiority": True}
    initial = io.read(folder / "INITIAL_GROWTH_STATE.json")
    final = io.read(folder / "FINAL_STATE.json")
    result = {"status": "COMPLETE_SINGLE_SLOT_GROWTH_FRESH128_DEVELOPMENT",
              "actual_utc": io.utc(), "plan_sha256": io.sha(args.plan), "source_sha256": io.sources(),
              "all_batch_freeze_sha256": io.sha(folder / "ALL_BATCHES_FROZEN.json"),
              "summaries": summaries, "paired_grown4_minus_controls": paired, "scores": records,
              "selected_descriptor_index": initial["selected_index"], "anchor_descriptor_index": initial["anchor_index"],
              "new_anchor_alias": initial["new_anchor_exact_alias"],
              "initial_library_weights": initial["libraries"], "final_library_weights": final["libraries"],
              "new_material_groups": 128, "prequential_validation_groups": 96, "forecast_score_coordinates": 672,
              "actual_sklearn_fits": 6, "new_native_cell_cost": 384,
              "history40_is_reused_development_not_new_measurements": True,
              "fixed_disclosure_schedule_not_new_active_query_policy": True,
              "all128_development_same_DFT_source_not_external_confirmation": True,
              "no_generated_labels_or_old96_rescoring": True}
    io.write(output / "RESULTS.json", result)
    print({method: summaries[method]["0"]["CRPS_native_mean"] for method in io.METHODS})


if __name__ == "__main__":
    main()
