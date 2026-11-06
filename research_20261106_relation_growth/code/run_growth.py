"""One registered hypothesis birth and three shared prequential batches.

All seven methods forecast a whole batch before its true reports are disclosed.
Feature proposal is frozen once using only the historical40 development groups.
"""
import argparse
import importlib.metadata
from pathlib import Path
import time

import experiment_io as io


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plan, roles = io.validate(args.plan, args.approval, "prequential_run")
    actual_versions = {key: importlib.metadata.version(key) for key in plan["libraries"]}
    if actual_versions != plan["libraries"]:
        raise ValueError("registered library versions required")
    io.bootstrap_core(plan)
    import growth_models as models
    import proposal
    import baselines

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    io.write(output / "EXECUTION_START.json", {"actual_utc": io.utc(), "source_sha256": io.sources(),
             "plan_sha256": io.sha(args.plan), "approval_sha256": io.sha(args.approval),
             "libraries": actual_versions, "new128_response_cells_revealed": 0})
    try:
        history = io.history40(plan)
        io.write(output / "HISTORY40_ONLY.json", history)
        historical_rows = history["rows"]
        native = [r["native_values"] for r in historical_rows]
        descriptors = [models.composition_descriptors(r["identity"]["composition_amounts"])
                       for r in historical_rows]
        ids = [r["identity"]["component_id"] for r in historical_rows]
        if set(ids) & set(roles["new_development_consumed_component_ids"]):
            raise ValueError("fresh groups overlap historical training")
        anchor = proposal.prepare_anchor(ids, [row[:2] for row in native], descriptors, history["old_snapshot"])
        io.write(output / "ANCHOR_PREPARED.json", anchor)
        io.write(output / "ANCHOR_FREEZE.json", {"actual_utc": io.utc(), "before_any_OOF_fit": True,
                 "anchor_sha256": io.sha(output / "ANCHOR_PREPARED.json"), "source_sha256": io.sources()})
        proposed = proposal.propose(native, anchor)
        io.write(output / "AUTOMATIC_PROPOSAL.json", proposed)
        if proposed["selected_index"] is None:
            raise ValueError("all descriptor candidates degenerate; no fabricated growth")
        suite = models.initialize_growth(native, descriptors, history["old_snapshot"], history["old_final3"], proposed)
        io.write(output / "INITIAL_GROWTH_STATE.json", suite)
        baseline_transform = models.baseline_transform(history["old_snapshot"], descriptors)
        io.write(output / "BASELINE_TRANSFORM.json", baseline_transform)

        disclosure = io.Disclosure(plan["inputs"]["raw_source"]["file"], roles)
        visible = disclosure.initial()
        io.write(output / "VISIBLE_INPUTS.json", visible)
        newfit = visible["fit"]
        fitnative = [r["native_values"] for r in newfit]
        fitdesc = [models.composition_descriptors(r["identity"]["composition_amounts"]) for r in newfit]
        suite, fit_update = models.update_batch(suite, [r[:2] for r in fitnative], fitdesc, [r[2] for r in fitnative])
        io.write(output / "NEW32_FIT_UPDATE.json", fit_update)
        train = list(native) + fitnative
        train_desc = list(descriptors) + fitdesc
        fits, batch_records = [], []
        for batch, role in enumerate(io.BATCHES):
            folder = output / role
            folder.mkdir()
            rows = visible[role]
            query = [r["native_proxies"] for r in rows]
            desc = [models.composition_descriptors(r["identity"]["composition_amounts"]) for r in rows]
            baseline_train = [list(r[:2]) + list(d) for r, d in zip(train, train_desc)]
            baseline_query = [list(r) + list(d) for r, d in zip(query, desc)]
            io.write(folder / "STATE_BEFORE.json", {"suite": suite, "training_native_rows": train,
                     "training_descriptors": train_desc, "training_components": len(train),
                     "actual_utc": io.utc(), "all_prior_labels_authorized": True})
            gp, gpstate = baselines.fit_gp(baseline_train, [r[2] for r in train], baseline_query, baseline_transform)
            rf, rfstate = baselines.fit_forest(baseline_train, [r[2] for r in train], baseline_query, baseline_transform)
            for method, state in (("GP10", gpstate), ("RF10", rfstate)):
                io.write(folder / (method + "_STATE.json"), state)
                fits.append({"method": method, "role": role, "train_rows": len(train),
                             "file": role + "/" + method + "_STATE.json",
                             "sha256": io.sha(folder / (method + "_STATE.json"))})
            forecasts = []
            for method in io.METHODS:
                for i, row in enumerate(rows):
                    predicted = gp[i] if method == "GP10" else rf[i] if method == "RF10" else models.law(suite, method, query[i], desc[i])
                    forecasts.append({"method": method, "batch": batch + 1, "identity": row["identity"],
                                      "known_native_proxies": query[i], "composition_descriptors": desc[i], "law": predicted})
            io.write(folder / "FORECASTS.json", {"actual_utc": io.utc(), "records": forecasts,
                     "batch_truth_model_access_before_freeze": 0})
            if io.sources() != plan["source_sha256"]:
                raise ValueError("source changed before next disclosure")
            payloads = ("STATE_BEFORE.json", "GP10_STATE.json", "RF10_STATE.json", "FORECASTS.json")
            frozen = {"status": "ALL224_BATCH_LAWS_FROZEN_BEFORE_32Y", "actual_utc": io.utc(),
                      "role": role, "source_sha256": io.sources(), "plan_sha256": io.sha(args.plan),
                      "methods": list(io.METHODS), "prediction_count": 224,
                      "files_sha256": {name: io.sha(folder / name) for name in payloads}}
            io.write(folder / "FORECAST_FREEZE.json", frozen)
            truth = disclosure.reveal(role, folder / "FORECAST_FREEZE.json", io.sources(), io.sha(args.plan))
            io.write(folder / "TRUE_BATCH_REVEAL.json", truth)
            values = [r["native_target"] for r in truth["rows"]]
            suite, evidence = models.update_batch(suite, query, desc, values)
            io.write(folder / "UPDATE_AFTER_REVEAL.json", {"suite": suite, "evidence": evidence, "actual_utc": io.utc()})
            train += [list(x) + [y] for x, y in zip(query, values)]
            train_desc += desc
            batch_records.append({"role": role, "batch": batch + 1, "training_rows_before": 72 + 32 * batch,
                                  "freeze_sha256": io.sha(folder / "FORECAST_FREEZE.json"),
                                  "truth_sha256": io.sha(folder / "TRUE_BATCH_REVEAL.json")})
        io.write(output / "FIT_INDEX.json", {"records": fits, "GP_fits": 3, "RF_fits": 3})
        io.write(output / "DISCLOSURE_LEDGER.json", disclosure.ledger())
        io.write(output / "FINAL_STATE.json", suite)
        files = {p.relative_to(output).as_posix(): io.sha(p) for p in sorted(output.rglob("*.json"))}
        io.write(output / "ALL_BATCHES_FROZEN.json", {"status": "ALL672_PREQUENTIAL_LAWS_AND96_TRUE_OUTCOMES_COMPLETE",
                 "actual_utc": io.utc(), "plan_sha256": io.sha(args.plan), "source_sha256": io.sources(),
                 "files_sha256": files, "batches": batch_records, "actual_sklearn_fits": 6,
                 "score_calls": 0, "new_native_cells": 384, "all_new_components_development": 128,
                 "elapsed_seconds": time.perf_counter() - started})
        print({"status": "ALL_BATCHES_COMPLETE", "native_cells": 384, "forecasts": 672,
               "selected_descriptor": proposed["selected_index"], "anchor_alias": suite["new_anchor_exact_alias"]})
    except Exception as problem:
        io.write(output / "FAILED_STOP.json", {"actual_utc": io.utc(), "type": type(problem).__name__,
                 "message": str(problem), "source_sha256": io.sources(),
                 "no_silent_reselection_or_same_cohort_recipe_rescue": True})
        raise


if __name__ == "__main__":
    main()
