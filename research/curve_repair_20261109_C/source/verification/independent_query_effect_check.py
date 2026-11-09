"""Independent synthetic quadrature/Schur reference; never imports the SUT.

Run only after this software qualification fixture has been registered. Output
is created exclusively and contains no material observations or model weights.
The normal CRPS is integrated over actual Gaussian truth densities, rather than
using the SUT's closed-form expected-risk expression as the audit reference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from math import exp, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def fixture():
    covariance = np.zeros((6, 6))
    covariance[1:, 1:] = np.array([
        [2.0, .6, .2, .4, .1],
        [.6, 1.7, .3, .2, .25],
        [.2, .3, 1.3, .5, .1],
        [.4, .2, .5, 2.4, .3],
        [.1, .25, .1, .3, 1.1],
    ])
    return {
        "covariance": covariance.tolist(),
        "target_indices": [4, 1],
        "candidate_indices": [3, 2, 5],
        "noise_variances": [0.0, 0.4, 0.2],
        "costs": [0.5, 1.25, 2.0],
        "target_weights": [3.0, 1.0],
    }


def gaussian_crps_at_truth(truth, forecast_mean, forecast_sd):
    """Direct univariate normal CRPS; no expected-risk shortcut."""
    z = (truth - forecast_mean) / forecast_sd
    density = exp(-.5 * z * z) / sqrt(2.0 * pi)
    return forecast_sd * (z * (2.0 * ndtr(z) - 1.0) +
                          2.0 * density - 1.0 / sqrt(pi))


def integrated_self_risk(mean, variance):
    """Integrate over truth after standardizing only the integration variable."""
    sd = sqrt(variance)

    def integrand(standard_truth):
        truth = mean + sd * standard_truth
        density = exp(-.5 * standard_truth ** 2) / sqrt(2.0 * pi)
        return gaussian_crps_at_truth(truth, mean, sd) * density

    value, error = quad(integrand, -np.inf, np.inf,
                        epsabs=2e-12, epsrel=2e-12, limit=200)
    if error > 1e-10:
        raise ArithmeticError("Quadrature precision insufficient")
    return float(value), float(error)


def independently_evaluate(data):
    k = np.array(data["covariance"], dtype=float)
    targets = np.array(data["target_indices"], dtype=int)
    weights = np.array(data["target_weights"], dtype=float)
    weights /= weights.sum()
    initial_covariance = k[np.ix_(targets, targets)]
    before = np.array([integrated_self_risk(0.0, v)[0]
                       for v in np.diag(initial_covariance)])
    rows = []
    maximum_value_dependence = 0.0
    maximum_closed_form_error = 0.0
    for q, noise, cost in zip(data["candidate_indices"],
                              data["noise_variances"], data["costs"]):
        sub = np.r_[targets, q]
        np.linalg.cholesky(k[np.ix_(sub, sub)])
        measurement_covariance = np.array([[k[q, q] + noise]])
        target_measurement_cross = k[np.ix_(targets, [q])]
        after_covariance = initial_covariance - (
            target_measurement_cross @ np.linalg.solve(
                measurement_covariance, target_measurement_cross.T))
        np.linalg.cholesky(after_covariance)

        # Two direct dense Gaussian mean updates produce the response by a
        # finite difference. This does not call the SUT's response expression.
        observed_values = (-0.65, 1.35)
        means = [target_measurement_cross @ np.linalg.solve(
            measurement_covariance, np.array([value]))
                 for value in observed_values]
        response = (means[1] - means[0]) / (
            observed_values[1] - observed_values[0])
        risks_across_observations = []
        errors = []
        for z in (-2.0, 0.0, 1.5):
            observed = z * sqrt(measurement_covariance[0, 0])
            posterior_mean = target_measurement_cross @ np.linalg.solve(
                measurement_covariance, np.array([observed]))
            integrated = [integrated_self_risk(mu, v)
                          for mu, v in zip(posterior_mean,
                                           np.diag(after_covariance))]
            risks_across_observations.append([x[0] for x in integrated])
            errors.extend(x[1] for x in integrated)
        risks = np.array(risks_across_observations)
        maximum_value_dependence = max(
            maximum_value_dependence, float(np.ptp(risks, axis=0).max()))
        after = risks[1]
        gain = before - after
        analytic_gain_for_check_only = (
            np.sqrt(np.diag(initial_covariance)) -
            np.sqrt(np.diag(after_covariance))) / sqrt(pi)
        maximum_closed_form_error = max(
            maximum_closed_form_error,
            float(np.max(np.abs(gain - analytic_gain_for_check_only))))
        rows.append({
            "query_index": q,
            "innovation_variance": float(measurement_covariance[0, 0]),
            "response": response.tolist(),
            "variance_after": np.diag(after_covariance).tolist(),
            "full_target_covariance_after": after_covariance.tolist(),
            "expected_gain_by_target": gain.tolist(),
            "normalized_target_weights": weights.tolist(),
            "reference_bayes_crps": float(weights @ before),
            "posterior_bayes_crps": float(weights @ after),
            "weighted_expected_gain": float(weights @ gain),
            "cost": cost,
            "gain_per_cost": float(weights @ gain) / cost,
            "max_quad_error_estimate": max(errors),
        })
    if maximum_value_dependence > 1e-10 or maximum_closed_form_error > 1e-10:
        raise ArithmeticError("Independent integration gate failed")
    return {
        "effects": rows,
        "ranked_query_indices": [r["query_index"] for r in sorted(
            rows, key=lambda r: (-r["gain_per_cost"], r["query_index"]))],
        "max_observation_value_dependence": maximum_value_dependence,
        "max_closed_form_expected_gain_error": maximum_closed_form_error,
    }


def compare_saved_actual(expected, actual):
    """Compare root-exported SUT values; the audit never imports that source."""
    expected_by_q = {row["query_index"]: row for row in expected["effects"]}
    actual_by_q = {row["query_index"]: row for row in actual["effects"]}
    if set(expected_by_q) != set(actual_by_q):
        raise ValueError("Actual query identities differ from registered fixture")
    fields = ("innovation_variance", "response", "variance_after",
              "expected_gain_by_target", "normalized_target_weights",
              "reference_bayes_crps", "posterior_bayes_crps",
              "weighted_expected_gain", "cost", "gain_per_cost")
    checks = 0
    maximum_error = 0.0
    for q, ref in expected_by_q.items():
        for name in fields:
            e, a = np.asarray(ref[name]), np.asarray(actual_by_q[q][name])
            if e.shape != a.shape or not np.isfinite(a).all():
                raise ValueError(f"Invalid actual field q={q} {name}")
            error = float(np.max(np.abs(e-a)))
            if not np.allclose(e, a, rtol=1e-10, atol=1e-11):
                raise ArithmeticError(f"Actual mismatch q={q} {name}: {error}")
            maximum_error = max(maximum_error, error)
            checks += int(e.size)
    if actual["ranked_query_indices"] != expected["ranked_query_indices"]:
        raise ArithmeticError("Actual ranking differs")
    return {"passed": True, "numeric_elements_compared": checks,
            "max_abs_error": maximum_error, "ranking_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actual", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fixture_bytes = args.fixture.read_bytes()
    data = json.loads(fixture_bytes)
    if data != fixture():
        raise ValueError("Fixture differs from the registered fixed audit design")
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
    result = {
        "classification": "INDEPENDENT_SYNTHETIC_SOFTWARE_ONLY",
        "actual_utc": datetime.now(timezone.utc).isoformat(),
        "fixture": data,
        "fixture_file_path": str(args.fixture.resolve()),
        "fixture_file_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "fixture_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "method": "Dense Gaussian Schur solve and direct normal CRPS quadrature",
        "sut_imported": False,
        "material_records_read": 0,
        **independently_evaluate(data),
    }
    if args.actual is not None:
        actual_bytes = args.actual.read_bytes()
        result["actual_receipt_path"] = str(args.actual.resolve())
        result["actual_receipt_sha256"] = hashlib.sha256(actual_bytes).hexdigest()
        result["comparison"] = compare_saved_actual(result, json.loads(actual_bytes))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output),
                      "comparison": result.get("comparison"),
                      "material_records_read": 0}))


if __name__ == "__main__":
    main()
