"""Pure arithmetic over saved score scalars. No model or raw-data imports."""
import math
from statistics import median

MODELS = ("gp", "generic", "local", "gaussian", "joint_code_gaussian",
          "random_gaussian", "bayesian_cross", "rb_carreau", "fixed_student_t")
ORIGINAL = MODELS[:5]
LAYOUTS = ("spread", "middle_gap", "one_sided")
CANDIDATE = "joint_code_gaussian"
VS_MODELS = ("gp", "conditional_gaussian", "random_gaussian", "curvature_gaussian", "fixed_student_t")
LABELS = {"gp": "Coordinate GP", "generic": "Generic bounded tilt",
          "local": "Curvature bounded tilt", "gaussian": "Conditional Gaussian",
          "joint_code_gaussian": "Gaussian curvature correction",
          "random_gaussian": "Matched random Gaussian", "bayesian_cross": "Bayesian Cross",
          "rb_carreau": "Bayesian Carreau a2", "fixed_student_t": "Fixed Student-t diagnostic"}


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot average an empty group")
    return math.fsum(values) / len(values)


def validate_rows(rows, models, expected_formulations):
    expected = {(model, p, mask) for model in models for p in expected_formulations for mask in LAYOUTS}
    keys = [(r["model"], r["P"], r["mask"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Missing, duplicate or unexpected model/formulation/layout score row")
    for row in rows:
        if row["formulation_id"] != "P" + str(row["P"]):
            raise ValueError("Formulation identities are inconsistent")


def validate_marginal(rows, expected_formulations):
    validate_rows(rows, MODELS, expected_formulations)
    for r in rows:
        if r["target_count"] != 33:
            raise ValueError("The frozen evaluation has exactly 33 targets per query")
        for field in ("crps", "mean_mse", "joint_log_density", "coverage90", "width90"):
            if not isinstance(r[field], (int, float)) or not math.isfinite(r[field]):
                raise ValueError("Invalid score scalar: " + field)
        if min(r["crps"], r["mean_mse"], r["width90"]) < 0:
            raise ValueError("Negative loss or interval width")
        if not (0 <= r["covered_count90"] <= 33):
            raise ValueError("Invalid marginal coverage count")
        if abs(r["coverage90"] - r["covered_count90"] / 33) > 1e-12:
            raise ValueError("Coverage count and saved fraction disagree")
        if r["mask"] == "one_sided":
            if (r["inside_count"], r["continuation_count"]) != (10, 23):
                raise ValueError("One-sided score partition changed")
            partition_mean = (r["inside_crps_sum"] + r["continuation_crps_sum"]) / 33
            if abs(partition_mean - r["crps"]) > 1e-12:
                raise ValueError("One-sided saved score sums do not recover query CRPS")


def aggregate(rows):
    n = sum(r["target_count"] for r in rows)
    if n == 0:
        raise ValueError("No scored targets")
    pooled_mse = math.fsum(r["target_count"] * r["mean_mse"] for r in rows) / n
    return {"CRPS": math.fsum(r["target_count"] * r["crps"] for r in rows) / n,
            "RMSE": math.sqrt(pooled_mse), "pooled_MSE": pooled_mse,
            "mean_joint_log_density": mean(r["joint_log_density"] for r in rows),
            "coverage90": sum(r["covered_count90"] for r in rows) / n,
            "width90": math.fsum(r["target_count"] * r["width90"] for r in rows) / n,
            "formulations": len({r["P"] for r in rows}), "queries": len(rows), "scored_positions": n}


def paired_scores(rows, comparators):
    material = {}
    for model in MODELS:
        own = [r for r in rows if r["model"] == model]
        material[model] = {p: mean(r["crps"] for r in own if r["P"] == p) for p in sorted({r["P"] for r in own})}
    pairs = {}
    for comparator in comparators:
        differences = [{"P": p, "formulation_id": "P" + str(p),
                        "candidate_CRPS": material[CANDIDATE][p],
                        "comparator_CRPS": material[comparator][p],
                        "candidate_minus_comparator": material[CANDIDATE][p] - material[comparator][p]}
                       for p in material[CANDIDATE]]
        delta = [r["candidate_minus_comparator"] for r in differences]
        pairs[comparator] = {"mean_difference": mean(delta), "median_difference": median(delta),
                             "wins": sum(d < 0 for d in delta), "losses": sum(d > 0 for d in delta),
                             "ties": sum(d == 0 for d in delta), "formulations": len(delta), "rows": differences}
    return pairs


def make_tables(rows, variogram, points, facts, provenance):
    pids = provenance["formulation_P_ids"]
    validate_marginal(rows, pids)
    validate_rows(variogram, VS_MODELS, pids)
    validate_rows(points, ("pchip_endpoint_cubic",), pids)
    if any(r["pairs"] != 528 or not math.isfinite(r["VS1"]) or r["VS1"] < 0 for r in variogram):
        raise ValueError("Invalid fixed variogram score")
    if any(r["target_count"] != 33 or not math.isfinite(r["mean_mse"]) or r["mean_mse"] < 0 for r in points):
        raise ValueError("Invalid point-only MSE")
    aggregates = {m: aggregate([r for r in rows if r["model"] == m]) for m in MODELS}
    by_layout = {layout: {m: aggregate([r for r in rows if r["model"] == m and r["mask"] == layout])
                          for m in MODELS} for layout in LAYOUTS}
    pairs = paired_scores(rows, [m for m in MODELS if m != CANDIDATE])
    table1 = [{"model": LABELS[m], **{k: aggregates[m][k] for k in
              ("CRPS", "RMSE", "mean_joint_log_density", "coverage90", "width90")}} for m in ORIGINAL]
    table2_all = []
    for m in MODELS:
        own = [r for r in rows if r["model"] == m and r["mask"] == "one_sided"]
        table2_all.append({"model": LABELS[m],
                           "inside_10_CRPS": math.fsum(r["inside_crps_sum"] for r in own) / sum(r["inside_count"] for r in own),
                           "continuation_23_CRPS": math.fsum(r["continuation_crps_sum"] for r in own) / sum(r["continuation_count"] for r in own)})
    table3 = [{"model": LABELS[m], "macro_CRPS": aggregates[m]["CRPS"],
               "one_sided_CRPS": by_layout["one_sided"][m]["CRPS"],
               "mean_joint_log_density": aggregates[m]["mean_joint_log_density"],
               "coverage90": aggregates[m]["coverage90"]}
              for m in (CANDIDATE, "random_gaussian", "bayesian_cross", "rb_carreau", "fixed_student_t")]
    s1 = []
    for m in ("gp", "generic", "local", "gaussian", "random_gaussian"):
        saved = facts["saved_bootstrap95"][m]
        s1.append({"comparator": LABELS[m], "mean_difference": pairs[m]["mean_difference"],
                   "saved_descriptive_interval_lower": saved["interval"][0],
                   "saved_descriptive_interval_upper": saved["interval"][1],
                   "candidate_lower_CRPS_formulations": pairs[m]["wins"], "formulations": 25,
                   "interval_role": "Copied original fixed-forecast bootstrap, not recomputed",
                   "interval_source_id": saved["source_id"]})
    s2 = [{"layout": layout, **{LABELS[m]: by_layout[layout][m]["CRPS"] for m in ORIGINAL}} for layout in LAYOUTS]
    s3 = [{"representation": "Fixed random QR", "bounded_CRPS": aggregates["generic"]["CRPS"], "Gaussian_CRPS": aggregates["random_gaussian"]["CRPS"]},
          {"representation": "Fixed curvature contrasts", "bounded_CRPS": aggregates["local"]["CRPS"], "Gaussian_CRPS": aggregates[CANDIDATE]["CRPS"]}]
    vs_macro = {m: mean(r["VS1"] for r in variogram if r["model"] == m) for m in VS_MODELS}
    vs_layout = {layout: {m: mean(r["VS1"] for r in variogram if r["model"] == m and r["mask"] == layout)
                          for m in VS_MODELS} for layout in LAYOUTS}
    s5 = [{"model": m, **{layout: vs_layout[layout][m] for layout in LAYOUTS}, "macro_VS1": vs_macro[m]} for m in VS_MODELS]
    s6 = [{"layout": layout, "candidate_CRPS": by_layout[layout][CANDIDATE]["CRPS"],
           "conditional_Gaussian_CRPS": by_layout[layout]["gaussian"]["CRPS"],
           "difference": by_layout[layout][CANDIDATE]["CRPS"] - by_layout[layout]["gaussian"]["CRPS"]} for layout in LAYOUTS]
    s6.append({"layout": "macro", "candidate_CRPS": aggregates[CANDIDATE]["CRPS"],
               "conditional_Gaussian_CRPS": aggregates["gaussian"]["CRPS"],
               "difference": aggregates[CANDIDATE]["CRPS"] - aggregates["gaussian"]["CRPS"]})
    point_table = []
    for layout in (*LAYOUTS, "macro"):
        selected = [r for r in points if layout == "macro" or r["mask"] == layout]
        n = sum(r["target_count"] for r in selected)
        mse = math.fsum(r["mean_mse"] * r["target_count"] for r in selected) / n
        point_table.append({"layout": layout, "pooled_MSE": mse, "pooled_RMSE": math.sqrt(mse),
                            "queries": len(selected), "score_role": "Point-only; no probability law assigned"})
    coverage = [{"P": p, **{layout: next(r["covered_count90"] for r in rows
                 if r["P"] == p and r["model"] == CANDIDATE and r["mask"] == layout) for layout in LAYOUTS}}
                for p in pids]
    paired_table = [{"comparator": LABELS[m], **{k: pairs[m][k] for k in
                    ("mean_difference", "median_difference", "wins", "losses", "ties", "formulations")}} for m in pairs]
    # Numeric plot inputs are deterministic; no raw response/interval curve is reconstructed.
    tables = {"main_table1": table1, "main_table2": table2_all[:5], "main_table2_extended": table2_all,
              "main_table3": table3, "SI_tableS1": s1, "SI_tableS2": s2, "SI_tableS3": s3,
              "SI_tableS5": s5, "SI_tableS6": s6, "paired_all": paired_table,
              "point_only_PCHIP": point_table, "figure2b_paired_data": pairs["gaussian"]["rows"],
              "figureS1_coverage_data": coverage}
    return {"scope": "Arithmetic replay of saved development score ledgers; no new measurements or raw-data scoring.",
            "counts": {"formulations": 25, "queries_per_method": 75, "probabilistic_rows": len(rows),
                       "VS_rows": len(variogram), "point_rows": len(points), "independent_new_materials": 0},
            "aggregate": aggregates, "by_layout": by_layout, "VS_macro": vs_macro, "VS_by_layout": vs_layout,
            "paired": pairs, "tables": tables}


def validate_reported(result, facts, absolute_tolerance=2e-11):
    comparisons = []

    def check(label, actual, expected):
        error = abs(actual - expected)
        comparisons.append({"field": label, "absolute_error": error})
        if error > absolute_tolerance:
            raise ValueError("Saved table replay mismatch: " + label)

    for m in MODELS:
        for metric in ("CRPS", "RMSE", "mean_joint_log_density", "coverage90", "width90"):
            check("aggregate/" + m + "/" + metric, result["aggregate"][m][metric], facts["aggregate"][m][metric])
            for layout in LAYOUTS:
                check("by_layout/" + layout + "/" + m + "/" + metric,
                      result["by_layout"][layout][m][metric], facts["by_mask"][layout][m][metric])
        record = result["tables"]["main_table2_extended"][MODELS.index(m)]
        check("continuation23/" + m, record["continuation_23_CRPS"], facts["reported_diagnostics"][m]["posthoc_23_continuation_CRPS"])
        check("inside10/" + m, record["inside_10_CRPS"], facts["reported_diagnostics"][m]["posthoc_10_in_support_CRPS"])
        if m != CANDIDATE:
            check("paired_mean/" + m, result["paired"][m]["mean_difference"], facts["reported_paired"][m]["candidate_minus_control_mean"])
            check("paired_wins/" + m, result["paired"][m]["wins"], facts["reported_paired"][m]["candidate_lower_CRPS_formulations"])
    for m in VS_MODELS:
        check("VS_macro/" + m, result["VS_macro"][m], facts["variogram_aggregate"][m])
        for layout in LAYOUTS:
            check("VS_layout/" + layout + "/" + m, result["VS_by_layout"][layout][m], facts["variogram_by_mask"][layout][m])
    for row in result["tables"]["point_only_PCHIP"]:
        expected = facts["PCHIP_RMSE"] if row["layout"] == "macro" else facts["PCHIP_by_mask_RMSE"][row["layout"]]
        check("PCHIP_RMSE/" + row["layout"], row["pooled_RMSE"], expected)
    return {"status": "PASS", "compared_saved_scalars": len(comparisons),
            "maximum_absolute_error": max(x["absolute_error"] for x in comparisons),
            "absolute_tolerance": absolute_tolerance,
            "new_fits": 0, "new_forecasts": 0, "raw_target_scoring_calls": 0}
