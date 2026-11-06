"""One registered prequential study: reference preservation and low-D repair.

OOF reference production, past-only calibration, current reference fitting and
true-report disclosure have explicit boundaries. No current-batch Y is used to
select a descriptor, fit a correction or rank a current prediction.
"""
from copy import deepcopy
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np

import transport_io as io
from empirical_native import apply_spline, loss_gap_envelope
from simplex_transport import fit_global, fit_context
from calibration_views import fit_views, fit_beta, apply_beta


def fit_fixed_views(records, anchor, selected):
    laws = [r["law"] for r in records]
    targets = [r["native_target"] for r in records]
    context = np.asarray([r["bounded_descriptors8"] for r in records])
    global_fit = fit_global(laws, targets)
    fitted = {}
    for index in sorted({j for j in (anchor, selected) if j is not None}):
        fitted[index] = fit_context(laws, targets, context[:, index])
        fitted[index]["descriptor_index"] = index
    identity = {"status": "IDENTITY_NO_ELIGIBLE_CONTEXT", "kind": "identity"}
    return {"global": global_fit,
            "anchor": deepcopy(fitted.get(anchor, identity)),
            "auto": deepcopy(fitted.get(selected, identity)),
            "anchor_index": anchor, "auto_index": selected,
            "descriptor_reselected_on_future_batches": False}


def corrected(law, fitted, descriptor, index):
    if fitted["kind"] == "identity" or fitted["status"].startswith("FAILED"):
        return deepcopy(law)
    if fitted["kind"] == "global":
        return apply_spline(law, fitted["theta0"])
    if index not in range(8):
        raise ValueError("a frozen eligible descriptor index is required")
    return apply_spline(law, fitted["theta0"], fitted["theta1"], descriptor[index])


def produce_oof(history, models, references, folder):
    rows = history["rows"]
    identities = [r["identity"] for r in rows]
    native = [r["native_values"] for r in rows]
    raw, bounded = io.descriptors(models, identities)
    X = [r[:2] + d for r, d in zip(native, raw)]
    targets = [r[2] for r in native]
    ordered = sorted(range(168), key=lambda i: (
        hashlib.sha256(("transport_oof_v1|" + identities[i]["component_id"]).encode()).hexdigest(),
        identities[i]["component_id"]))
    fold_for_row = {row: rank % 5 for rank, row in enumerate(ordered)}
    assignment = [{"history_row": i, "component_id": r["component_id"], "fold": fold_for_row[i]}
                  for i, r in enumerate(identities)]
    io.write(folder / "OOF_ASSIGNMENT_BEFORE_FITS.json", {
        "actual_utc": io.utc(), "records": assignment,
        "fold_sizes": [sum(v == k for v in fold_for_row.values()) for k in range(5)],
        "fold_hash_prefix": "transport_oof_v1|", "old_labels_already_development": True})
    records = [None] * 168
    fits = []
    for fold in range(5):
        held = [i for i in range(168) if fold_for_row[i] == fold]
        train = [i for i in range(168) if fold_for_row[i] != fold]
        laws, state = references.fit_forest([X[i] for i in train], [targets[i] for i in train],
                                            [X[i] for i in held], history["baseline_transform"])
        state.update(training_component_ids=[identities[i]["component_id"] for i in train],
                     heldout_component_ids=[identities[i]["component_id"] for i in held], fold=fold)
        name = "RF_OOF_STATE_" + str(fold) + ".json"
        io.write(folder / name, state)
        fits.append({"kind": "RF_OOF", "fold": fold, "train_rows": len(train),
                     "prediction_rows": len(held), "state_file": name, "state_sha256": io.sha(folder / name)})
        if len(laws) != len(held):
            raise ValueError("one heldout reference law per original group required")
        for i, law in zip(held, laws):
            records[i] = {"history_row": i, "identity": identities[i], "law": law,
                          "native_target": targets[i], "bounded_descriptors8": bounded[i],
                          "mode": "OOF_EXCLUDED_WHOLE_FOLD", "fold": fold,
                          "reference_state_file": name, "reference_state_sha256": io.sha(folder / name)}
    if any(r is None for r in records):
        raise ValueError("complete OOF coverage required")
    io.write(folder / "OOF_RECORDS168.json", {"actual_utc": io.utc(), "records": records,
             "OOF_only_reference_not_independent_calibrator_validation": True})
    return records, fits, X, native, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plan, roles = io.validate(args.plan, args.approval, "produce_prequential_forecasts")
    modules = io.bootstrap(plan)
    models, references = modules["growth_models"], modules["baselines"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    io.write(output / "START.json", {"actual_utc": io.utc(), "plan_sha256": io.sha(args.plan),
             "source_sha256": io.sources(), "fresh192_target_access": 0})
    try:
        history = io.history168(plan)
        oldids = {r["identity"]["component_id"] for r in history["rows"]}
        if oldids & set(roles["new_development_consumed_component_ids"]):
            raise ValueError("historical and fresh component groups overlap")
        io.write(output / "HISTORY168_ONLY.json", history)
        calibration, fits, X, native, raw_training = produce_oof(history, models, references, output)
        laws, targets = [r["law"] for r in calibration], [r["native_target"] for r in calibration]
        bounded_training = [r["bounded_descriptors8"] for r in calibration]
        initial_views = fit_views(laws, targets, bounded_training)
        initial_beta = fit_beta(laws, targets)
        anchor, selected = initial_views["anchor_index"], initial_views["auto_index"]
        io.write(output / "INITIAL_CALIBRATION_ONLY168.json", {"views": initial_views, "beta": initial_beta,
                 "actual_utc": io.utc(), "all_scores_are_past_OOF_training_objectives": True})
        io.write(output / "DESCRIPTOR_SELECTION_FROZEN.json", {"actual_utc": io.utc(),
                 "anchor_index": anchor, "auto_index": selected,
                 "initial_fit_sha256": io.sha(output / "INITIAL_CALIBRATION_ONLY168.json"),
                 "future_descriptor_reselection": False, "fresh192_properties_not_yet_parsed": True})
        # The trusted raw parser is instantiated only after the descriptor choice.
        disclosure = io.Disclosure(plan["inputs"]["raw_source"]["file"], roles)
        visible = disclosure.initial()
        io.write(output / "VISIBLE_CONTEXTS_ONLY.json", visible)
        nig3 = deepcopy(history["nig3"])
        batch_records = []
        for number, role in enumerate(io.BATCHES):
            folder = output / role
            folder.mkdir()
            if len(X) != 168 + 64 * number or len(calibration) != len(X):
                raise ValueError("same common past training/calibration record count required")
            if len({r["identity"]["component_id"] for r in calibration}) != len(calibration):
                raise ValueError("one historical calibration pair per source component")
            io.write(folder / "CALIBRATION_HISTORY_BEFORE.json", {"records": calibration,
                     "never_recomputed_old_CDF_from_current_in_sample_RF": True})
            if number == 0:
                views, beta = initial_views, initial_beta
            else:
                views = fit_fixed_views(calibration, anchor, selected)
                beta = fit_beta([r["law"] for r in calibration], [r["native_target"] for r in calibration])
            io.write(folder / "STATE_BEFORE.json", {"nig3": nig3, "views": views, "beta": beta,
                     "actual_utc": io.utc(), "training_rows": len(X), "calibration_rows": len(calibration),
                     "anchor_index": anchor, "auto_index": selected,
                     "calibration_history_sha256": io.sha(folder / "CALIBRATION_HISTORY_BEFORE.json")})
            identities = [r["identity"] for r in visible[role]]
            query = [r["native_proxies"] for r in visible[role]]
            raw_query, bounded_query = io.descriptors(models, identities)
            queryX = [p + d for p, d in zip(query, raw_query)]
            trainY = [r[2] for r in native]
            rf_laws, rf_state = references.fit_forest(X, trainY, queryX, history["baseline_transform"])
            gp_laws, gp_state = references.fit_gp(X, trainY, queryX, history["baseline_transform"])
            for method, state in (("RF10", rf_state), ("GP10", gp_state)):
                state["training_component_ids"] = [r["identity"]["component_id"] for r in calibration]
                state["prediction_component_ids"] = [r["component_id"] for r in identities]
                io.write(folder / (method + "_STATE.json"), state)
                fits.append({"kind": method, "batch": number + 1, "train_rows": len(X),
                             "prediction_rows": 64, "state_file": role + "/" + method + "_STATE.json",
                             "state_sha256": io.sha(folder / (method + "_STATE.json"))})
            if len(rf_laws) != 64 or len(gp_laws) != 64:
                raise ValueError("complete64 reference predictions required")
            forecasts, envelopes = [], []
            for i, identity in enumerate(identities):
                reference = rf_laws[i]
                predicted = {"RF_identity": reference, "RF_beta": apply_beta(reference, beta),
                             "RF_global4": corrected(reference, views["global"], bounded_query[i], None),
                             "RF_anchor4": corrected(reference, views["anchor"], bounded_query[i], anchor),
                             "RF_auto4": corrected(reference, views["auto"], bounded_query[i], selected),
                             "GP10": gp_laws[i],
                             "old3_NIG": models.law(nig3, "old3_BMA", query[i], raw_query[i])}
                for method in io.METHODS:
                    forecasts.append({"method": method, "identity": identity,
                                      "known_native_proxies": query[i], "bounded_descriptors8": bounded_query[i],
                                      "law": predicted[method]})
                for method in ("RF_beta", "RF_global4", "RF_anchor4", "RF_auto4"):
                    envelopes.append({"method": method, "component_id": identity["component_id"],
                                      "loss_gap_envelope": loss_gap_envelope(reference, predicted[method])})
            io.write(folder / "FORECASTS.json", {"actual_utc": io.utc(), "records": forecasts,
                     "current_batch_targets_given_to_models": 0})
            io.write(folder / "PRETARGET_LOSS_GAP_ENVELOPES.json", {"records": envelopes,
                     "exact_rational_arithmetic_for_saved_supports_and_masses": True,
                     "not_a_guarantee_of_risk_improvement": True})
            if io.sources() != plan["source_sha256"]:
                raise ValueError("scientific source changed before next truth")
            names = ("CALIBRATION_HISTORY_BEFORE.json", "STATE_BEFORE.json", "RF10_STATE.json",
                     "GP10_STATE.json", "FORECASTS.json", "PRETARGET_LOSS_GAP_ENVELOPES.json")
            io.write(folder / "FORECAST_FREEZE.json", {"status": "ALL448_BATCH_LAWS_FROZEN_BEFORE64Y",
                     "actual_utc": io.utc(), "role": role, "source_sha256": io.sources(),
                     "plan_sha256": io.sha(args.plan), "methods": list(io.METHODS),
                     "prediction_count": 448, "files_sha256": {name: io.sha(folder / name) for name in names}})
            truth = disclosure.reveal(role, folder / "FORECAST_FREEZE.json", io.sources(), io.sha(args.plan))
            io.write(folder / "TRUE_BATCH_REVEAL.json", truth)
            values = [r["native_target"] for r in truth["rows"]]
            nig3, increments = models.update_batch(nig3, query, raw_query, values)
            io.write(folder / "NIG3_UPDATE_AFTER_REVEAL.json", {"suite": nig3, "evidence": increments,
                     "actual_utc": io.utc(), "unique_old_components_per_row": 3})
            for i, (identity, y) in enumerate(zip(identities, values)):
                calibration.append({"identity": identity, "law": rf_laws[i], "native_target": y,
                                    "bounded_descriptors8": bounded_query[i], "mode": "PRETARGET_FROZEN_RF",
                                    "role": role, "reference_forecasts_sha256": io.sha(folder / "FORECASTS.json"),
                                    "reference_freeze_sha256": io.sha(folder / "FORECAST_FREEZE.json")})
            X += queryX
            native += [p + [y] for p, y in zip(query, values)]
            raw_training += raw_query
            batch_records.append({"batch": number + 1, "role": role, "training_rows_before": 168 + number * 64,
                                  "freeze_sha256": io.sha(folder / "FORECAST_FREEZE.json"),
                                  "truth_sha256": io.sha(folder / "TRUE_BATCH_REVEAL.json")})
        io.write(output / "FINAL_HISTORY360.json", {"calibration_records": calibration, "nig3": nig3,
                 "third_batch_not_reused_for_extra_evaluation": True})
        io.write(output / "FIT_INDEX.json", {"records": fits, "RF_OOF_fits": 5, "RF_batch_fits": 3,
                 "GP_batch_fits": 3, "actual_sklearn_fits": 11})
        io.write(output / "DISCLOSURE_LEDGER.json", disclosure.ledger())
        payloads = {p.relative_to(output).as_posix(): io.sha(p) for p in sorted(output.rglob("*.json"))}
        io.write(output / "ALL_BATCHES_FROZEN.json", {"status": "ALL1344_PREQUENTIAL_LAWS_AND192_TRUE_OUTCOMES_COMPLETE",
                 "actual_utc": io.utc(), "plan_sha256": io.sha(args.plan), "source_sha256": io.sources(),
                 "files_sha256": payloads, "batches": batch_records, "actual_sklearn_fits": 11,
                 "validation_score_calls": 0, "past_calibration_CRPS_used_for_training": True,
                 "new_native_cells": 576, "all_new_components_development": 192,
                 "elapsed_seconds": time.perf_counter() - started})
        print({"status": "ALL_BATCHES_COMPLETE", "forecasts": 1344, "actual_sklearn_fits": 11,
               "new_native_cells": 576, "selected_g": selected, "anchor_g": anchor})
    except Exception as problem:
        io.write(output / "FAILED_STOP.json", {"actual_utc": io.utc(), "type": type(problem).__name__,
                 "message": str(problem), "source_sha256": io.sources(),
                 "no_silent_same_cohort_reselection_or_recipe_rescue": True})
        raise


if __name__ == "__main__":
    main()
