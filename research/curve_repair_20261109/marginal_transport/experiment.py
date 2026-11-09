"""Bounded fixed-marginal dependence diagnostic; SOURCE is not authorization.

Predict consumes saved own-six nearest-target forecasts only. Score uses the
same already consumed 120 dev targets and the frozen physical Student scores.
It never reads source CSV, changes an old model, fits, selects or blends laws.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import t

from pure import GaussianCopulaStudent, correlation_from_covariance

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "TRANSPORT_PROTOCOL.json"
ANCHORS = [0, 1, 13, 14, 15, 16]
PANEL = list(range(17, 25))
PARENT = "power_law__curv_nearest_target6"
CONTROLS = ("independence", "physical_student_shape", "frozen_gp_posterior", "frozen_curvature_posterior")
MARGINAL_METRICS = ("crps", "log_rmse", "coverage90", "width90")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("Duplicate JSON key: " + k)
            result[k] = v
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def authorize(path, phase):
    reg = read(path)
    if reg.get("schema") != "mlst-marginal-transport-registration-v1" or phase not in reg.get("authorized_phases", []):
        raise PermissionError("Root phase registration required")
    stamp = dt.datetime.fromisoformat(reg["actual_utc"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture UTC required")
    sources = [HERE / "pure.py", HERE / "experiment.py", PROTOCOL]
    for source in sources:
        if reg["source_pins"].get(str(source)) != sha(source):
            raise PermissionError("Source bytes not registered: " + str(source))
    for input_path, expected in reg.get("input_pins", {}).items():
        if sha(input_path) != expected:
            raise PermissionError("Input bytes changed: " + input_path)
    protocol = read(PROTOCOL)
    if protocol["controls"] != list(CONTROLS) or protocol["anchor_indices"] != ANCHORS or protocol["target_indices"] != PANEL:
        raise ValueError("Protocol differs from fixed four-law diagnostic")
    return reg, protocol


def start(output, phase, registration):
    path = Path(output) / phase
    path.mkdir(parents=True, exist_ok=False)
    write(path / "STARTED.json", {"phase": phase, "actual_utc": utc(),
          "registration_sha256": sha(registration), "new_source_eta_access": 0, "new_fits": 0})
    return path


def finish(path, payload, **facts):
    write(path / "COMPLETE.json", {"status": "COMPLETE", "actual_utc": utc(),
          "payload": payload, "payload_sha256": sha(path / payload), **facts})


def gate(output, registration):
    path = start(output, "gate", registration)
    fixture = {"location": [.2, -.4, .9], "scale": [.3, .7, 1.1],
               "correlation": [[1., .4, -.2], [.4, 1., .3], [-.2, .3, 1.]], "df": 4}
    law = GaussianCopulaStudent(**fixture)
    normal = np.asarray([[-9., 0., 9.], [9., -9., 0.], [0., 0., 0.], [-2., 1., .5], [.1, -.2, 3.]])
    transported = law.from_latent(normal)
    recovered = law.to_latent(transported)
    error = float(np.max(np.abs(normal - recovered)))
    if error > 5e-8:
        raise AssertionError("Smaller-tail transport roundtrip failed")
    tail_standard = np.asarray([[-1e12, 0., 1e12], [1e12, -1e12, 0.]])
    tail_values = law.location + law.scale * tail_standard
    tail_normal = law.to_latent(tail_values)
    rejected = {}
    for name, matrix in (("singular", np.ones((3, 3))),
                         ("indefinite", np.asarray([[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]]))):
        try:
            GaussianCopulaStudent(fixture["location"], fixture["scale"], matrix)
        except (ValueError, np.linalg.LinAlgError):
            rejected[name] = True
        else:
            raise AssertionError("Non-SPD correlation was silently repaired")
    draws = law.sample(20000, np.random.Generator(np.random.PCG64(20261109)))
    quantiles = law.quantile([.1, .5, .9])
    counts = np.asarray([(draws <= q).sum(axis=0) for q in quantiles])
    if np.max(np.abs(counts / 20000 - np.asarray([.1, .5, .9])[:, None])) > .02:
        raise AssertionError("Fixed synthetic marginal CDF frequency check failed")
    samples = path / "DRAW_SAMPLES.npz"
    with samples.open("xb") as stream:
        np.savez_compressed(stream, samples=draws)
    write(path / "GATE.json", {"schema": "mlst-marginal-transport-synthetic-gate-v1", "status": "PASS",
          "actual_utc": utc(), "registration_sha256": sha(registration), "fixture": fixture,
          "normal_probes": normal.tolist(), "transported_probes": transported.tolist(),
          "recovered_normal": recovered.tolist(), "roundtrip_max_abs": error,
          "logpdf_probes": law.logpdf(transported).tolist(),
          "independence_logpdf_probes": GaussianCopulaStudent(law.location, law.scale, np.eye(3)).logpdf(transported).tolist(),
          "tail_standardized_probes": tail_standard.tolist(), "tail_y_probes": tail_values.tolist(),
          "tail_normal_probes": tail_normal.tolist(), "tail_logpdf_probes": law.logpdf(tail_values).tolist(),
          "reject_singular": rejected["singular"], "reject_indefinite": rejected["indefinite"],
          "draws": 20000, "seed": 20261109, "algorithm": "PCG64",
          "probabilities": [.1, .5, .9], "quantiles": quantiles.tolist(), "sample_counts": counts.tolist(),
          "sample_npz": samples.name, "samples_sha256": sha(samples), "sample_array_key": "samples",
          "cdf_tolerance": .02, "new_material_reads": 0, "new_fits": 0,
          "scope": "fixed synthetic implementation check, not material evidence"})
    finish(path, "GATE.json", draws=20000, new_material_reads=0, new_fits=0)


def predict(output, registration, reg, audit_path):
    gate_path = Path(output) / "gate" / "GATE.json"
    gate_result = read(gate_path)
    if gate_result.get("status") != "PASS" or gate_result["registration_sha256"] != sha(registration):
        raise PermissionError("This registration's synthetic gate PASS required")
    independent = read(audit_path)
    if independent.get("status") != "PASS" or independent.get("audited_gate_sha256") != sha(gate_path):
        raise PermissionError("Independent exact synthetic-gate audit required")
    prepared, main, student = (read(reg[key]) for key in ("prepare", "main_predictions", "student_predictions"))
    if (main["prepare_sha256"] != sha(reg["prepare"]) or student["main_predictions_sha256"] != sha(reg["main_predictions"]) or
            student["main_prepare_sha256"] != sha(reg["prepare"])):
        raise ValueError("Existing preparation/forecast identity mismatch")
    curves = {row["member"]: row for row in prepared["curves"]}
    gaussian = {(row["member"], row["method_id"]): row for row in main["forecasts"]}
    physical = {(row["member"], row["method_id"]): row for row in student["forecasts"]}
    if len(curves) != 15 or len({r["recipe"] for r in curves.values()}) != 3:
        raise ValueError("Only original fifteen development curves allowed")
    path = start(output, "predict", registration)
    forecasts = []
    for member in sorted(curves):
        source = physical[member, PARENT]
        gp, curv = gaussian[member, "GP_nearest_target6"], gaussian[member, "curv_nearest_target6"]
        for row in (source, gp, curv):
            if (row["anchor_indices"] != ANCHORS or row["target_indices"] != PANEL or
                    row["recipe"] != curves[member]["recipe"] or row["anchor_values"] != source["anchor_values"]):
                raise ValueError("Same original six known anchors and eight target coordinates required")
        if source["df"] != 4 or source["baseline_kind"] != "power_law":
            raise ValueError("Frozen strongest physical marginal control required")
        scale = np.sqrt(np.diag(np.asarray(source["scale_matrix"])))
        correlations = (np.eye(8), correlation_from_covariance(source["scale_matrix"]),
                        correlation_from_covariance(gp["covariance"]), correlation_from_covariance(curv["covariance"]))
        for control, correlation in zip(CONTROLS, correlations):
            law = GaussianCopulaStudent(source["location"], scale, correlation)
            identity = {key: source[key] for key in ("member", "recipe", "run_id", "thermal_history",
                        "anchor_indices", "anchor_values", "target_indices")}
            forecasts.append({**identity, "method_id": control, "control_id": control,
                "marginal_parent_method_id": PARENT, **law.as_dict(),
                "quantiles05_50_95": law.quantile([.05, .5, .95]).tolist()})
    write(path / "PREDICTIONS.json", {"schema": "mlst-marginal-transport-predictions-v1", "actual_utc": utc(),
          "registration_sha256": sha(registration), "protocol_sha256": sha(PROTOCOL),
          "prepare_sha256": sha(reg["prepare"]), "main_predictions_sha256": sha(reg["main_predictions"]),
          "student_predictions_sha256": sha(reg["student_predictions"]), "synthetic_gate_sha256": sha(gate_path),
          "independent_gate_audit_sha256": sha(audit_path), "forecasts": forecasts,
          "laws": 60, "new_fits": 0, "new_source_numeric_eta_access": 0, "future_runs_protected": 35,
          "marginal_identity": "all four laws have exactly the same saved physical Student4 marginals"})
    finish(path, "PREDICTIONS.json", laws=60, new_fits=0, new_source_numeric_eta_access=0)


def score(output, registration, reg, args):
    predicted = Path(output) / "predict" / "PREDICTIONS.json"
    finished = read(Path(output) / "predict" / "COMPLETE.json")
    audit, release = read(args.audit), read(args.release)
    if (finished["payload_sha256"] != sha(predicted) or audit.get("status") != "PASS" or
            audit.get("audited_transport_predictions_sha256") != sha(predicted)):
        raise PermissionError("Frozen sixty laws and their independent audit PASS required")
    if (release.get("authorization") != "SCORE_FROZEN_MARGINAL_TRANSPORT_ON_SAME120" or
            release.get("predictions_sha256") != sha(predicted) or release.get("targets_sha256") != sha(args.targets) or
            release.get("student_scores_sha256") != sha(args.student_scores) or release.get("audit_sha256") != sha(args.audit)):
        raise PermissionError("Exact root same120-target score release required")
    stamp = dt.datetime.fromisoformat(release["actual_utc"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError("Actual nonfuture root release required")
    saved, targets, parent_scores = read(predicted), read(args.targets), read(args.student_scores)
    if (saved["registration_sha256"] != sha(registration) or
            targets["predictions_sha256"] != sha(reg["main_predictions"]) or
            parent_scores["predictions_sha256"] != sha(reg["student_predictions"]) or
            parent_scores["targets_sha256"] != sha(args.targets)):
        raise ValueError("Saved law/target/strong baseline byte chain changed")
    truth = {row["member"]: row for row in targets["curves"]}
    parent_rows = {(row["member"], row["method_id"]): row for row in parent_scores["curve_scores"]}
    parent_points = {(row["member"], row["method_id"], row["target_index"]): row for row in parent_scores["point_terms"]}
    if len(saved["forecasts"]) != 60 or len(truth) != 15:
        raise ValueError("Sixty fixed laws and fifteen already consumed target curves required")
    path = start(output, "score", registration)
    rows, points = [], []
    for forecast in saved["forecasts"]:
        member = forecast["member"]
        original = parent_rows[member, PARENT]
        y = np.asarray(truth[member]["values"])
        if truth[member]["target_indices"] != PANEL or truth[member]["recipe"] != forecast["recipe"]:
            raise ValueError("Fixed common target panel and grouping required")
        law = GaussianCopulaStudent(forecast["location"], forecast["scale"], forecast["correlation"])
        identity = {key: forecast[key] for key in ("member", "recipe", "run_id", "thermal_history", "method_id",
                    "control_id", "marginal_parent_method_id")}
        joint_nll = -law.logpdf(y) / 8
        rows.append({**identity, **{key: original[key] for key in MARGINAL_METRICS},
                     "joint_nll_per_dimension": joint_nll,
                     "original_student_joint_nll_per_dimension": original["joint_nll_per_dimension"],
                     "joint_nll_delta_vs_original_student": joint_nll - original["joint_nll_per_dimension"]})
        for i, index in enumerate(PANEL):
            old = parent_points[member, PARENT, index]
            if (old["truth"] != float(y[i]) or old["location"] != forecast["location"][i] or
                    old["t_scale"] != forecast["scale"][i] or old["df"] != 4):
                raise ValueError("Marginal score is not the same frozen physical marginal and truth")
            points.append({**identity, "target_index": index,
                **{key: old[key] for key in ("truth", "location", "t_scale", "t_sd", "df", "crps", "squared_error", "coverage90", "width90")},
                "quantiles05_50_95": [forecast["quantiles05_50_95"][j][i] for j in range(3)],
                "marginal_metrics_inherited_exactly": True})
    metrics = (*MARGINAL_METRICS, "joint_nll_per_dimension", "original_student_joint_nll_per_dimension", "joint_nll_delta_vs_original_student")
    recipes = sorted({row["recipe"] for row in rows})
    recipe_rows, macro = [], []
    for control in CONTROLS:
        for recipe in recipes:
            selected = [r for r in rows if r["method_id"] == control and r["recipe"] == recipe]
            if len(selected) != 5:
                raise ValueError("Original five runs/recipe retained")
            recipe_rows.append({"method_id": control, "recipe": recipe, "runs": 5,
                                **{key: float(np.mean([r[key] for r in selected])) for key in metrics}})
        selected = [r for r in recipe_rows if r["method_id"] == control]
        macro.append({"method_id": control, "recipes": 3,
                      **{key: float(np.mean([r[key] for r in selected])) for key in metrics}})
    write(path / "SCORES.json", {"schema": "mlst-marginal-transport-scores-v1", "actual_utc": utc(),
          "registration_sha256": sha(registration), "predictions_sha256": sha(predicted),
          "targets_sha256": sha(args.targets), "parent_student_scores_sha256": sha(args.student_scores),
          "independent_laws_audit_sha256": sha(args.audit), "root_score_release_sha256": sha(args.release),
          "curve_scores": rows, "point_terms": points, "recipe_scores": recipe_rows, "macro_by_method": macro,
          "primary": "all four recipe-macro joint NLL deltas vs original full multivariate Student; no selection or mixture",
          "marginal_performance": "Inherited exact same probabilities and scores, structural identity not new material improvement",
          "future_runs_protected": 35, "new_source_numeric_eta_access": 0, "new_fits": 0,
          "scope": "same consumed three-recipe development diagnostic, not fresh confirmation"})
    finish(path, "SCORES.json", laws_scored=60, repeated_existing_target_cells=120, new_source_numeric_eta_access=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("gate", "predict", "score"))
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("audit", "release", "targets", "student-scores"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args(argv)
    reg, protocol = authorize(args.registration, args.phase)
    if args.phase == "gate":
        gate(args.output, args.registration)
    elif args.phase == "predict":
        if args.audit is None:
            raise PermissionError("Independent synthetic gate audit required")
        predict(args.output, args.registration, reg, args.audit)
    else:
        if any(getattr(args, key) is None for key in ("audit", "release", "targets", "student_scores")):
            raise PermissionError("Independent frozen-law audit and root release/old scores/targets required")
        score(args.output, args.registration, reg, args)
    print(json.dumps({"status": "COMPLETE", "phase": args.phase, "output": str(args.output)}))


if __name__ == "__main__":
    main()
