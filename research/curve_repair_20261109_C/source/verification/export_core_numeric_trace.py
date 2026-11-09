"""Export the already registered fixed core fixture's numeric trace, not gates.

Preparation does not execute the source under test. At execution, a separate
root actual-UTC/hash authorization is required. The original core source bytes
and registration remain unchanged. No materials, model JSON, optimizer or new
fixture are used. The independent auditor consumes the resulting JSON only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


REGISTERED_SHA256 = "2788bd792706620caa4c1a305df5dfec8b2d2fa128e6b0e6ebd8bbd197a2df36"
AUTHORIZATION_SCOPE = "fixed_matrix_numeric_trace_and_independent_audit_only"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _authorize(registered_path, authorization_path):
    registered_path = Path(registered_path)
    authorization_path = Path(authorization_path)
    if _digest(registered_path) != REGISTERED_SHA256:
        raise ValueError("Original root registration differs from the pinned bytes")
    registered = json.loads(registered_path.read_text(encoding="utf-8-sig"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8-sig"))
    if (authorization.get("authorized_by") != "root" or
            authorization.get("scope") != AUTHORIZATION_SCOPE or
            authorization.get("root_registered_sha256") != REGISTERED_SHA256):
        raise ValueError("Separate root fixed-numeric-trace authorization required")
    stamp = datetime.fromisoformat(authorization["root_actual_utc"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("root_actual_utc must carry a UTC offset")
    if stamp > datetime.now(timezone.utc):
        raise ValueError("Trace authorization timestamp is in the future")
    original_stamp = datetime.fromisoformat(registered["root_actual_utc"].replace("Z", "+00:00"))
    if stamp < original_stamp:
        raise ValueError("Trace authorization predates the original registration")
    root = Path(__file__).resolve().parent.parent
    current_core = {p.name: _digest(p) for p in sorted((root / "core").glob("*.py"))}
    if current_core != registered["source_sha256"]:
        raise ValueError("Core source bytes changed after original registration")
    pins = authorization.get("source_sha256")
    required = {Path(__file__).name, "independent_core_numeric_check.py"}
    if not isinstance(pins, dict) or set(pins) != required:
        raise ValueError("Pin exactly the trace exporter and independent core auditor")
    for name, expected in pins.items():
        if Path(name).name != name or _digest(root / "verification" / name) != expected:
            raise ValueError("Verification source differs from root trace authorization: " + name)
    return root, registered, authorization, current_core


def export_trace(registered_path, authorization_path):
    root, registered, authorization, core_hashes = _authorize(registered_path, authorization_path)
    sys.dont_write_bytecode = True
    # Import the SUT only after verifying its unchanged registered bytes.
    # Running this script by path places verification/, rather than root/, on
    # sys.path. Add the extension root solely for its explicit package imports.
    sys.path.insert(0, str(root))
    try:
        import numpy as np
        from core.early_influence import (
            GaussianConditionalScore, ScoringReadings, inspect_parameter_update,
        )
        from core.frozen_loader import FROZEN_CORE_SHA256, load_frozen_core
        from core.synthetic_gates import _fixture, _make
    finally:
        sys.path.pop(0)
    if FROZEN_CORE_SHA256 != registered["frozen_core_sha256"]:
        raise ValueError("Frozen-core SHA differs from original registration")
    frozen = load_frozen_core()
    theta = np.array([.07, -.08, .03, .04, -.02])
    target = np.array([6, 1, 4, 3])
    train_conditioner, fixed_values, _ = _make(frozen)
    noisy_conditioner, _, _ = _make(frozen, noise=[.06, .10, 0.0])
    dev_conditioner, dev_fixed_values, _ = _make(frozen, split="dev", noise=[.02, .05, .01])
    exact = train_conditioner.condition(theta, target_indices=target)
    noisy = noisy_conditioner.condition(theta)
    train_labels = ScoringReadings(
        train_conditioner.roles.identity, "pseudo_hole_train",
        train_conditioner.roles.pseudo_hole_train,
        fixed_values[train_conditioner.roles.pseudo_hole_train])
    dev_labels = ScoringReadings(
        dev_conditioner.roles.identity, "sentinel_dev", dev_conditioner.roles.sentinel_dev,
        dev_fixed_values[dev_conditioner.roles.sentinel_dev], [.03, .04])
    train_score = GaussianConditionalScore(train_conditioner, train_labels)
    dev_score = GaussianConditionalScore(dev_conditioner, dev_labels)
    train_value, train_gradient = train_score(theta)
    dev_value, dev_gradient = dev_score(theta)
    early = inspect_parameter_update(
        theta, [train_score], [dev_score], step_sizes=[0.0, .001],
        evaluate_development_steps=True, development_use_id="fixed_synthetic_gate_v1")
    mean, covariance, cross, residual = _fixture()

    def context(conditioner):
        return {
            "identity": asdict(conditioner.roles.identity),
            "observed_indices": conditioner.observed_indices.tolist(),
            "observed_values": conditioner.observed_values.tolist(),
            "observed_noise_variance": conditioner.noise_variance.tolist(),
            "roles": {name: getattr(conditioner.roles, name).tolist()
                      for name in ("observed", "pseudo_hole_train", "sentinel_dev", "unseen")},
        }

    def posterior(conditioner, result):
        return {
            **context(conditioner), "theta": result.theta.tolist(),
            "target_indices": result.target_indices.tolist(),
            "mean": result.mean.tolist(), "covariance": result.covariance.tolist(),
            "coordinate_system": result.coordinate_system,
        }

    def score_payload(score, value, gradient):
        return {
            **context(score.conditioner), "theta": theta.tolist(), "role": score.role,
            "label_indices": score.indices.tolist(), "label_values": score.values.tolist(),
            "label_noise_variance": score.noise_variance.tolist(),
            "nll": float(value), "gradient": gradient.tolist(),
        }

    return {
        "schema_version": "core_numeric_trace_v1",
        "classification": "FIXED_SYNTHETIC_NUMERIC_TRACE_EXPORT_NOT_GATE_RERUN",
        "actual_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_actual_utc": authorization["root_actual_utc"],
        "authorization_sha256": _digest(authorization_path),
        "root_registered_sha256": REGISTERED_SHA256,
        "core_source_sha256": core_hashes,
        "verification_source_sha256": authorization["source_sha256"],
        "frozen_core_sha256": FROZEN_CORE_SHA256,
        "fixture": {
            "mean": mean.tolist(), "covariance": covariance.tolist(),
            "cross": cross.tolist(), "residual": residual.tolist(),
            "fixed_values": fixed_values.tolist(), "theta": theta.tolist(),
            "target_indices": target.tolist(),
        },
        "exact": posterior(train_conditioner, exact),
        "noisy": posterior(noisy_conditioner, noisy),
        "train": score_payload(train_score, train_value, train_gradient),
        "dev": score_payload(dev_score, dev_value, dev_gradient),
        "early_report": early,
        "trace_score_calls": {"train": 2, "dev": 3},
        "original_gate_runner_called": False,
        "new_fixture": False, "optimizer_runs": 0,
        "materials_loaded": 0, "material_fits": 0, "old_25_scores": 0,
        "model_json_loaded": False,
        "limitation": "Recomputes the same registered fixed matrix only; independent numeric audit is separate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root-registered", type=Path, required=True)
    parser.add_argument("--trace-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = export_trace(args.root_registered, args.trace_authorization)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()),
                      "classification": result["classification"],
                      "material_fits": 0, "old_25_scores": 0}))


if __name__ == "__main__":
    main()
