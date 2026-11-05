"""Reaggregate saved score columns only; never fit, predict, or read measurements.

Standard library only. The archived six-model aggregates are copied with their
provenance; the ten new arms are checked against per-coordinate saved losses.
This is a review replay, not an independent material validation or target score.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
import statistics


MASKS = ("spread", "middle_gap", "one_sided")
NEW_RANDOMS = tuple(f"random_{seed}" for seed in range(20261105, 20261113))
NEW_METHODS = NEW_RANDOMS + ("prior_kl", "curvature45")
ORDER = (
    "gp", "gaussian", "generic", "local", "random_gaussian",
    "joint_code_gaussian", "prior_kl", "curvature45", *NEW_RANDOMS,
)
LABELS = {
    "gp": "Coordinate GP",
    "gaussian": "Conditional Gaussian",
    "generic": "Original random bounded",
    "local": "Curvature bounded",
    "random_gaussian": "Original random Gaussian",
    "joint_code_gaussian": "Fixed curvature Gaussian",
    "prior_kl": "Prior-KL Gaussian",
    "curvature45": "Curvature-45-degree Gaussian",
    **{method: f"Random Gaussian (seed {method.split('_')[1]})" for method in NEW_RANDOMS},
}
LOSS_FIELDS = (
    "crps", "squared_error", "covered90", "width90",
    "interval_score90", "marginal_nll",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty collection")
    return sum(values, Decimal(0)) / Decimal(len(values))


def json_ready(value):
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Refusing an empty table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(json_ready(rows))


def loss_summary(rows):
    """Equal formulation weight, then equal queried coordinates within a layout.

    All layouts have 33 targets, so equal-layout and equal-coordinate aggregation
    agree. That balance is checked explicitly rather than assumed from row count.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["P"], row["mask"])].append(row)
    query_means = [
        {field: mean(row[field] for row in query) for field in LOSS_FIELDS}
        for query in groups.values()
    ]
    return {
        "CRPS": mean(row["crps"] for row in query_means),
        "RMSE": mean(row["squared_error"] for row in query_means).sqrt(),
        "coverage90": mean(row["covered90"] for row in query_means),
        "width90": mean(row["width90"] for row in query_means),
        "interval_score90": mean(row["interval_score90"] for row in query_means),
        "marginal_nll": mean(row["marginal_nll"] for row in query_means),
        "formulations": len({row["P"] for row in rows}),
        "queries": len(groups),
        "scored_positions": len(rows),
        "distinct_formulation_position_pairs": len({(row["P"], row["target_index"]) for row in rows}),
        "zero_coverage_queries": sum(row["covered90"] == 0 for row in query_means),
        "all_targets_in_separate_marginal_bands_queries": sum(row["covered90"] == 1 for row in query_means),
    }


def run(result_path: Path, coordinate_path: Path, candidate_path: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(f"Output must be new: {output_dir}")
    started = datetime.now(timezone.utc).isoformat()
    result = json.loads(result_path.read_text(encoding="utf-8"), parse_float=Decimal)
    if sha256(candidate_path) != "8800800920bad262fc06bdfc18ccdf319f080387eacbaf5808d1f28851ed0b43":
        raise ValueError("Candidate saved-loss input differs from the authorized pinned file")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"), parse_float=Decimal)
    if result["status"] != "COMPLETED":
        raise ValueError("Saved score result is not complete")
    if sha256(coordinate_path) != result["coordinate_metrics_sha256"]:
        raise ValueError("Coordinate-score hash differs from completed result")
    if set(result["aggregate"]) != set(ORDER):
        raise ValueError("Unexpected saved method inventory")
    coordinate = []
    with coordinate_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"method", "P", "mask", "target_index", *LOSS_FIELDS}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError("Unexpected score-only CSV schema")
        for original in reader:
            row = {**original, "P": int(original["P"]), "target_index": int(original["target_index"])}
            for field in LOSS_FIELDS:
                row[field] = Decimal(original[field])
                if not row[field].is_finite():
                    raise ValueError(f"Non-finite saved score: {field}")
            if row["covered90"] not in (0, 1):
                raise ValueError("Saved interval indicator is not binary")
            if any(row[field] < 0 for field in ("crps", "squared_error", "width90", "interval_score90")):
                raise ValueError("Saved nonnegative loss is negative")
            coordinate.append(row)

    inventory = defaultdict(list)
    query_coordinates = defaultdict(set)
    for row in coordinate:
        key = (row["method"], row["P"], row["mask"])
        if row["target_index"] in query_coordinates[key]:
            raise ValueError("Duplicate saved coordinate key")
        query_coordinates[key].add(row["target_index"])
        inventory[row["method"]].append(row)
    if set(inventory) != set(NEW_METHODS):
        raise ValueError("Unexpected new-arm inventory")
    if len(coordinate) != 24750 or len(query_coordinates) != 750:
        raise ValueError("Expected 24,750 score entries across 750 new queries")
    formulation_ids = sorted({row["P"] for row in coordinate})
    if len(formulation_ids) != 25:
        raise ValueError("Expected 25 distinct original formulation identifiers")
    for method in NEW_METHODS:
        for formulation in formulation_ids:
            for mask in MASKS:
                indices = query_coordinates[(method, formulation, mask)]
                if len(indices) != 33:
                    raise ValueError("Each saved query must contain 33 distinct target positions")
                if indices != query_coordinates[("prior_kl", formulation, mask)]:
                    raise ValueError("Target layouts differ across arms")
    candidate_query = {(row["P"], row["mask"]): row for row in candidate["new_rows"]}
    if len(candidate_query) != 75 or set(candidate_query) != {(p, mask) for p in formulation_ids for mask in MASKS}:
        raise ValueError("Archived candidate query inventory differs from new arms")
    for (formulation, mask), row in candidate_query.items():
        if set(row["target_indices"]) != query_coordinates[("prior_kl", formulation, mask)]:
            raise ValueError("Candidate target layout differs from new arms")

    summary = {}
    all_model_rows, by_mask_rows, new_uq_rows, p0_rows = [], [], [], []
    comparisons = []
    max_difference = Decimal(0)
    checks = 0
    for method in ORDER:
        for mask in ("overall", *MASKS):
            archived = result["aggregate"][method] if mask == "overall" else result["by_mask"][mask][method]
            metrics = dict(archived)
            origin = "Archived aggregate copied; no old targets rescored"
            if method in NEW_METHODS:
                subset = [r for r in inventory[method] if mask == "overall" or r["mask"] == mask]
                replay = loss_summary(subset)
                for field in ("CRPS", "RMSE", "coverage90", "width90"):
                    difference = abs(replay[field] - archived[field])
                    max_difference = max(max_difference, difference)
                    if difference > Decimal("1e-12") * max(Decimal(1), abs(archived[field])):
                        raise ValueError(f"Saved aggregate mismatch: {method}/{mask}/{field}")
                    checks += 1
                density_rows = [r for r in result["new_rows"] if r["model"] == method and (mask == "overall" or r["mask"] == mask)]
                density = mean(r["joint_log_density"] for r in density_rows)
                difference = abs(density - archived["mean_joint_log_density"])
                max_difference = max(max_difference, difference)
                if difference > Decimal("1e-12") * max(Decimal(1), abs(archived["mean_joint_log_density"])):
                    raise ValueError("Saved joint-density aggregate mismatch")
                checks += 1
                metrics.update(replay)
                metrics["mean_joint_log_density"] = density
                origin = "Reaggregated saved coordinate scores; saved joint density averaged"
                uq = {"method": method, "label": LABELS[method], "layout": mask, **metrics}
                new_uq_rows.append(uq)
                p0 = loss_summary([r for r in subset if r["P"] == 0])
                p0_rows.append({"method": method, "label": LABELS[method], "layout": mask, "P": 0, **p0})
            row = {
                "method": method, "label": LABELS[method], "layout": mask,
                "CRPS": metrics["CRPS"], "RMSE": metrics["RMSE"],
                "mean_joint_log_density": metrics["mean_joint_log_density"],
                "coverage90": metrics["coverage90"], "width90": metrics["width90"],
                "interval_score90": metrics.get("interval_score90", ""),
                "marginal_nll": metrics.get("marginal_nll", ""),
                "formulations": 25, "queries": 75 if mask == "overall" else 25,
                "scored_positions": 2475 if mask == "overall" else 825,
                "provenance": origin,
            }
            (all_model_rows if mask == "overall" else by_mask_rows).append(row)
            summary[(method, mask)] = metrics

    curve = result["aggregate"]["joint_code_gaussian"]["CRPS"]
    kl = result["aggregate"]["prior_kl"]["CRPS"]
    random_values = [result["aggregate"][method]["CRPS"] for method in NEW_RANDOMS]
    random_stats = {
        "prespecified_seeds": list(NEW_RANDOMS), "count": 8,
        "CRPS_values": dict(zip(NEW_RANDOMS, random_values)),
        "mean_CRPS": mean(random_values), "median_CRPS": statistics.median(random_values),
        "min_CRPS": min(random_values), "max_CRPS": max(random_values),
        "population_SD_across_seeds_descriptive_only": mean((x - mean(random_values)) ** 2 for x in random_values).sqrt(),
        "seeds_with_lower_CRPS_than_frozen_curvature": sum(x < curve for x in random_values),
        "frozen_curvature_relative_reduction_from_random_median": (statistics.median(random_values) - curve) / statistics.median(random_values),
        "warning": "Eight constructions on the same consumed formulations; no independent-material sampling distribution or seed selection.",
    }
    curvature_kl_rows = []
    curvature_kl_stats = {}
    for mask in ("overall", *MASKS):
        differences = []
        for formulation in formulation_ids:
            selected_masks = MASKS if mask == "overall" else (mask,)
            old_score = mean(candidate_query[(formulation, layout)]["crps"] for layout in selected_masks)
            new_score = loss_summary([r for r in inventory["prior_kl"] if r["P"] == formulation and r["mask"] in selected_masks])["CRPS"]
            difference = new_score - old_score
            differences.append(difference)
            curvature_kl_rows.append({
                "P": formulation, "layout": mask, "curvature_CRPS": old_score,
                "prior_KL_CRPS": new_score, "prior_KL_minus_curvature_CRPS": difference,
                "lower_CRPS": "prior_kl" if difference < 0 else "curvature" if difference > 0 else "tie",
            })
        curvature_kl_stats[mask] = {
            "paired_formulations": 25, "mean_difference": mean(differences),
            "median_difference": statistics.median(differences),
            "min_difference": min(differences), "max_difference": max(differences),
            "prior_KL_strictly_lower": sum(x < 0 for x in differences),
            "curvature_strictly_lower": sum(x > 0 for x in differences),
            "ties_exact_saved_decimal": sum(x == 0 for x in differences),
            "mean_difference_excluding_P0_descriptive_only": mean(x for p, x in zip(formulation_ids, differences) if p != 0),
        }
    if curvature_kl_stats["overall"]["curvature_strictly_lower"] != result["candidate_comparison"]["prior_kl"]["candidate_lower_crps_materials"]:
        raise ValueError("Saved paired win count differs from independently reaggregated saved losses")
    if abs(curvature_kl_stats["overall"]["mean_difference"] - (kl - curve)) > Decimal("1e-12"):
        raise ValueError("Paired saved-loss macro difference does not match aggregate")
    paired = {
        "curvature_vs_prior_kl": {
            "curvature_macro_CRPS": curve, "prior_KL_macro_CRPS": kl,
            "prior_KL_minus_curvature_macro_CRPS": kl - curve,
            "prior_KL_relative_CRPS_reduction_from_curvature": (curve - kl) / curve,
            "paired_formulations": 25,
            "curvature_lower_CRPS_formulations_from_saved_comparison": result["candidate_comparison"]["prior_kl"]["candidate_lower_crps_materials"],
            "direct_saved_loss_pairing": curvature_kl_stats,
            "inferential_confidence_interval_or_p_value": None,
            "pairing_provenance": "Archived candidate query CRPS losses paired with new prior-KL coordinate CRPS losses. No old measurement targets rescored.",
            "by_layout": {
                mask: {
                    "prior_KL_minus_curvature_CRPS": result["by_mask"][mask]["prior_kl"]["CRPS"] - result["by_mask"][mask]["joint_code_gaussian"]["CRPS"],
                    "prior_KL_minus_curvature_RMSE": result["by_mask"][mask]["prior_kl"]["RMSE"] - result["by_mask"][mask]["joint_code_gaussian"]["RMSE"],
                    "prior_KL_minus_curvature_mean_joint_log_density": result["by_mask"][mask]["prior_kl"]["mean_joint_log_density"] - result["by_mask"][mask]["joint_code_gaussian"]["mean_joint_log_density"],
                }
                for mask in MASKS
            },
            "interpretation": "KL improves aggregate CRPS/RMSE but has lower average joint log density; this narrows curvature-specific optimality. A new developmental winner is not an independently confirmed deployment replacement.",
        },
        "candidate_comparisons_from_saved_result": result["candidate_comparison"],
    }
    for mask in ("overall", *MASKS):
        differences = []
        for formulation in formulation_ids:
            values = {}
            for method in ("prior_kl", "curvature45"):
                rows = [r for r in inventory[method] if r["P"] == formulation and (mask == "overall" or r["mask"] == mask)]
                values[method] = loss_summary(rows)["CRPS"]
            differences.append(values["prior_kl"] - values["curvature45"])
        comparisons.append({
            "layout": mask, "comparison": "Prior-KL minus curvature-45-degree; saved CRPS",
            "paired_formulations": 25, "mean_difference": mean(differences),
            "median_difference": statistics.median(differences),
            "min_difference": min(differences), "max_difference": max(differences),
            "prior_KL_strictly_lower": sum(d < 0 for d in differences),
            "ties_exact_saved_decimal": sum(d == 0 for d in differences),
        })
    paired["new_prior_KL_vs_rotation_direct_saved_pairing"] = comparisons

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "ALL_MODELS_OVERALL.csv", all_model_rows)
    write_csv(output_dir / "ALL_MODELS_BY_MASK.csv", by_mask_rows)
    write_csv(output_dir / "NEW_ARMS_UNCERTAINTY.csv", new_uq_rows)
    write_csv(output_dir / "P0_NEW_ARMS.csv", p0_rows)
    write_csv(output_dir / "CURVATURE_VS_KL_FORMULATION.csv", curvature_kl_rows)
    write_json(output_dir / "RANDOM_DISTRIBUTION.json", random_stats)
    write_json(output_dir / "PAIRED_DESCRIPTIVE.json", paired)
    receipt = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "result_input": {"path": str(result_path), "sha256": sha256(result_path)},
        "coordinate_input": {"path": str(coordinate_path), "sha256": sha256(coordinate_path)},
        "candidate_saved_loss_input": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "source_sha256": sha256(Path(__file__)),
        "status": "PASS", "decimal_precision": 60,
        "saved_aggregate_comparisons": checks, "max_absolute_aggregate_difference": max_difference,
        "inventory": {
            "models_overall": 16, "models_by_layout_rows": 48,
            "new_arms": 10, "new_arm_uq_rows_including_overall": 40,
            "P0_new_arm_rows_including_overall": 40,
            "new_model_formulation_combinations": 250,
            "new_saved_queries": 750, "new_saved_coordinate_scores": 24750,
            "unique_consumed_formulations": 25,
            "original_formulation_ids": formulation_ids,
            "queries_per_model": 75, "scored_positions_per_model": 2475,
            "distinct_formulation_position_pairs_per_new_model": 950,
            "new_independent_materials": 0,
        },
        "scope": "Saved-score review replay only. No fits, inference, hidden-S targets or measurement values read. Old aggregate scores copied, candidate paired CRPS reaggregated from saved query losses; no old measurements rescored. Fit receipt audit remains separate.",
        "statistical_limits": "No confidence intervals, significance tests, new calibration claim, batch-generalization claim, or deployment selection. Coverage is marginal and empirical; an entire query inside separate marginal bands is not a simultaneous coverage guarantee.",
    }
    write_json(output_dir / "REPLAY_RECEIPT.json", receipt)
    (output_dir / "README.txt").write_text(
        "SAVED-SCORE REVIEW REPLAY\n\n"
        "ALL_MODELS_OVERALL.csv: all16 saved model aggregates. ALL_MODELS_BY_MASK.csv: all48 model-layout rows.\n"
        "NEW_ARMS_UNCERTAINTY.csv and P0_NEW_ARMS.csv: ten new arms, each overall and three layouts. Missing archived NLL/interval scores are blank rather than invented.\n"
        "New metrics are means of already saved losses, not recomputed predictions or measurement scoring. Decimal60 reaggregation checks 200 scalar aggregate values.\n"
        "RANDOM_DISTRIBUTION.json retains all8 fixed seeds. PAIRED_DESCRIPTIVE.json and CURVATURE_VS_KL_FORMULATION.csv use an additional hard-pinned saved candidate score file for direct descriptive pairing.\n"
        "There are25 consumed formulations,75 correlated queries and2475 scored positions per model; 250 new model-formulation combinations and750 new saved queries do not constitute new materials.\n"
        "The prior-KL result weakens curvature-optimality claims. Its improved CRPS does not establish density dominance, calibrated intervals or transfer. No best-seed/deployment selection is performed.\n",
        encoding="utf-8",
    )
    write_json(output_dir / "MANIFEST.json", {
        str(path.relative_to(output_dir)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(output_dir.iterdir()) if path.is_file()
    })
    print(json.dumps(json_ready({"status": "PASS", "output": str(output_dir), "checks": checks, "max_difference": max_difference, "prior_KL_CRPS": kl}), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--coordinate", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with localcontext() as context:
        context.prec = 60
        run(args.result.resolve(), args.coordinate.resolve(), args.candidate.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
