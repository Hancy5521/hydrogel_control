"""Audit saved synthetic wrapper/gradient numbers without importing the SUT.

This uses a direct residual-plus-code covariance construction, dense Gaussian
conditioning and finite differences of a conditional target density. Execute
only after root registers this source and its SUT trace exporter. No materials,
frozen weights or original scientific model modules are imported or read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def fixed_joint(theta):
    b = np.array([[.12, -.04], [.20, .06], [.10, .09], [-.08, .11],
                  [.04, -.16], [.13, .14], [-.10, .03]])
    grid = np.arange(7)
    residual = .18 * .5 ** np.abs(grid[:, None] - grid[None, :]) + .35 * np.eye(7)
    baseline_mean = np.array([-.20, -.10, .02, .09, .12, .14, .17])
    l = np.array([[np.exp(theta[2]), 0.0], [theta[3], np.exp(theta[4])]])
    return baseline_mean + b @ theta[:2], residual + b @ l @ l.T @ b.T


def full_posterior(theta, noise):
    mean, covariance = fixed_joint(theta)
    o = np.array([5, 0, 2])
    fixed_values = np.array([-.11, .20, -.06, .32, .01, .41, -.14])
    innovation_covariance = covariance[np.ix_(o, o)] + np.diag(noise)
    cross = covariance[:, o]
    posterior_mean = mean + cross @ np.linalg.solve(
        innovation_covariance, fixed_values[o] - mean[o])
    posterior_covariance = covariance - cross @ np.linalg.solve(
        innovation_covariance, cross.T)
    # Exact readings condition the same coordinate, hence zero covariance.
    for idx, variance in zip(o, noise):
        if variance == 0:
            posterior_mean[idx] = fixed_values[idx]
            posterior_covariance[idx, :] = 0.0
            posterior_covariance[:, idx] = 0.0
    return posterior_mean, (posterior_covariance + posterior_covariance.T) / 2


def conditional_nll(theta, observed_noise, label_noise):
    mean, covariance = full_posterior(theta, observed_noise)
    t = np.array([1, 3])
    actual_labels = np.array([.20, .32])
    target_covariance = covariance[np.ix_(t, t)] + np.diag(label_noise)
    np.linalg.cholesky(target_covariance)
    error = actual_labels - mean[t]
    sign, logdet = np.linalg.slogdet(target_covariance)
    if sign != 1:
        raise ArithmeticError("Synthetic target covariance is not positive")
    return float(.5 * (len(t) * np.log(2 * np.pi) + logdet +
                       error @ np.linalg.solve(target_covariance, error)))


def finite_gradient(theta, observed_noise, label_noise, h):
    result = []
    for unit in np.eye(5):
        plus = conditional_nll(theta+h*unit, observed_noise, label_noise)
        minus = conditional_nll(theta-h*unit, observed_noise, label_noise)
        result.append((plus-minus)/(2*h))
    return np.array(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    actual_bytes = args.actual.read_bytes()
    actual = json.loads(actual_bytes)
    theta = np.array([.07, -.08, .03, .04, -.02])
    records = []

    def compare(name, expected, saved, atol=2e-11, rtol=2e-10):
        e, a = np.asarray(expected), np.asarray(saved)
        if e.shape != a.shape or not np.isfinite(a).all():
            raise ValueError("Bad saved shape/nonfinite value: " + name)
        error = float(np.max(np.abs(e-a))) if e.size else 0.0
        if not np.allclose(e, a, atol=atol, rtol=rtol):
            raise ArithmeticError(f"Independent saved-number mismatch {name}: {error}")
        records.append({"name": name, "elements": int(e.size),
                        "max_abs_error": error, "atol": atol, "rtol": rtol})

    for name, noise in (("exact", [0.0, 0.0, 0.0]),
                        ("noisy", [.06, .10, 0.0])):
        mean, covariance = full_posterior(theta, noise)
        compare(name+".mean", mean, actual[name]["mean"])
        compare(name+".covariance", covariance, actual[name]["covariance"])
    compare("exact.target_indices", [6, 1, 4, 3], actual["exact"]["target_indices"], 0, 0)

    noises = {"train": ([0.0, 0.0, 0.0], [0.0, 0.0]),
              "dev": ([.02, .05, .01], [.03, .04])}
    gradients = {}
    for name, (obs, label) in noises.items():
        value = conditional_nll(theta, obs, label)
        gradient = finite_gradient(theta, obs, label, 1e-5)
        gradient_check = finite_gradient(theta, obs, label, 5e-6)
        compare(name+".fd_step_stability", gradient_check, gradient,
                atol=3e-9, rtol=1e-7)
        compare(name+".nll", value, actual[name]["nll"])
        compare(name+".gradient", gradient, actual[name]["gradient"],
                atol=3e-9, rtol=2e-7)
        gradients[name] = gradient

    report = actual["early_report"]
    direction = -gradients["train"]
    slope = float(gradients["dev"] @ direction)
    compare("early.direction", direction, report["direction"], atol=3e-9, rtol=2e-7)
    compare("early.slope", slope, report["development_directional_derivative"],
            atol=3e-9, rtol=2e-7)
    if [row["step"] for row in report["steps"]] != [0.0, .001]:
        raise ValueError("Early diagnostic steps differ from fixed design")
    for record in report["steps"]:
        step = record["step"]
        compare("early.predicted_change."+str(step), step*slope,
                record["predicted_development_nll_change"], atol=3e-11, rtol=2e-7)
        expected_change = conditional_nll(theta+step*direction, *noises["dev"]) - conditional_nll(theta, *noises["dev"])
        compare("early.actual_change."+str(step), expected_change,
                record["actual_development_nll_change"], atol=3e-11, rtol=2e-7)
    if not (report["development_score_calls"] == 2 and
            report["selected_step"] is None and
            report["unseen_labels_used"] is False and
            report["optimizer_runs"] == 0):
        raise ValueError("Early diagnostic role/count boundaries differ")
    output = {
        "status": "passed",
        "classification": "INDEPENDENT_SAVED_SYNTHETIC_NUMERIC_AUDIT",
        "actual_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actual_receipt_path": str(args.actual.resolve()),
        "actual_receipt_sha256": hashlib.sha256(actual_bytes).hexdigest(),
        "sut_imported": False, "materials_read": 0, "model_weights_read": False,
        "method": "Independent residual+BSB^T, dense conditional target NLL, central finite differences at two scales",
        "checks": records,
        "numeric_elements_compared": sum(r["elements"] for r in records),
        "limitation": "This checks the fixed synthetic trace, not every possible input or material accuracy.",
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "status": "passed",
                      "numeric_elements_compared": output["numeric_elements_compared"]}))


if __name__ == "__main__":
    main()
