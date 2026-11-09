"""Prepared fixed-matrix gates. This file does not run on import.

No dataset, model JSON or material outcome is opened. Before running, root must
record actual UTC and hashes of every Python source in this directory. A source
edit invalidates that authorization. The authorization is an audit boundary,
not a claim that caller-supplied timestamps prove clock truth.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .arbitrary_mask import (
    ArbitraryMaskConditioner, CurveIdentity, MaskRoles, ObservedReadings,
)
from .early_influence import GaussianConditionalScore, ScoringReadings, inspect_parameter_update
from .frozen_loader import FROZEN_CORE_SHA256, load_frozen_core


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def _authorized(path):
    receipt = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if receipt.get("scope") != "fixed_matrix_synthetic_only" or receipt.get("authorized_by") != "root":
        raise ValueError("Root fixed-matrix authorization required")
    if receipt.get("frozen_core_sha256") != FROZEN_CORE_SHA256:
        raise ValueError("Frozen-core hash differs from root authorization")
    stamp = datetime.fromisoformat(receipt["root_actual_utc"].replace("Z", "+00:00"))
    if stamp.utcoffset() is None or stamp.utcoffset().total_seconds() != 0:
        raise ValueError("root_actual_utc must carry UTC offset")
    if stamp > datetime.now(timezone.utc):
        raise ValueError("Root authorization timestamp is in the future")
    current = source_hashes()
    if receipt.get("source_sha256") != current:
        raise ValueError("Source bytes changed after root hash authorization")
    return receipt, current


def _fixture():
    # Fixed synthetic numbers, not old 25-formulation data or extracted models.
    b = np.array([[.12, -.04], [.20, .06], [.10, .09], [-.08, .11],
                  [.04, -.16], [.13, .14], [-.10, .03]])
    grid = np.arange(7)
    residual = .18 * .5 ** np.abs(grid[:, None] - grid[None, :]) + .35 * np.eye(7)
    covariance = residual + b @ b.T
    mean = np.array([-.20, -.10, .02, .09, .12, .14, .17])
    return mean, covariance, b, residual


def _make(core, *, split="train", noise=0.0, observed=(5, 0, 2)):
    mean, covariance, b, residual = _fixture()
    reference = core.CodeReference(mean, covariance, b)
    identity = CurveIdentity("fixed_synthetic_v1", "group_" + split, "curve_" + split, split)
    remaining = [i for i in range(7) if i not in observed]
    withheld = remaining[:2]
    roles = MaskRoles(identity, 7, list(observed), withheld if split == "train" else [],
                      withheld if split == "dev" else [],
                      remaining[2:] if split != "unseen" else remaining)
    # Distinct values available only as explicit observed/authorized-score payloads.
    fixed = np.array([-.11, .20, -.06, .32, .01, .41, -.14])
    readings = ObservedReadings(identity, list(observed), fixed[list(observed)], noise)
    return ArbitraryMaskConditioner(reference, roles, readings), fixed, residual


def _direct(reference, theta, o, y, r):
    m, k = reference.joint(theta)
    if not len(o):
        return m, k
    a = k[np.ix_(o, o)] + np.diag(r)
    gain = np.linalg.solve(a, k[o, :]).T
    return m + gain @ (y - m[o]), k - gain @ k[o, :]


def _raises(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected boundary rejection did not occur")


def run_gates(authorization_path):
    """Run only after root has written an actual-UTC/source-hash receipt."""
    authorization, hashes = _authorized(authorization_path)
    core = load_frozen_core()
    checks = []
    c, fixed, residual = _make(core)
    theta = np.array([.07, -.08, .03, .04, -.02])
    target = np.array([6, 1, 4, 3])  # arbitrary order and separated internal holes
    exact = c.condition(theta, target_indices=target)
    em, ek = _direct(c.reference, theta, c.observed_indices, c.observed_values, c.noise_variance)
    np.testing.assert_allclose(exact.mean, em, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(exact.covariance, ek, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(exact.target_indices, target)
    checks.append("arbitrary_order_internal_holes_full_Schur_equivalence")

    # Permuting the observed payload is harmless when values/variances move too.
    p = [2, 0, 1]
    permuted = ArbitraryMaskConditioner(c.reference, c.roles, ObservedReadings(
        c.roles.identity, c.observed_indices[p], c.observed_values[p], c.noise_variance[p]))
    other = permuted.condition(theta, target_indices=target)
    np.testing.assert_allclose(other.mean, exact.mean, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(other.covariance, exact.covariance, atol=1e-12, rtol=1e-12)
    checks.append("observed_permutation_equivariance")

    zero = c.condition(np.zeros(5), target_indices=target)
    identity = c.condition(theta, v=0, target_indices=target)
    np.testing.assert_array_equal(zero.mean, identity.mean)
    np.testing.assert_array_equal(zero.covariance, identity.covariance)
    checks.append("v_zero_exact_reference_identity")

    noisy, _, _ = _make(core, noise=[.06, .10, 0.0])
    post = noisy.condition(theta)
    nm, nk = _direct(noisy.reference, theta, noisy.observed_indices, noisy.observed_values, noisy.noise_variance)
    np.testing.assert_allclose(post.mean, nm, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(post.covariance, nk, atol=1e-12, rtol=1e-12)
    assert np.all(np.diag(post.covariance)[noisy.observed_indices[:2]] > 0)
    np.testing.assert_array_equal(post.covariance[noisy.observed_indices[2]], np.zeros(7))
    checks.append("known_variance_mixed_exact_noisy_observation_likelihood")

    # Explicitly catch the tempting but wrong unconditioned-residual surrogate.
    o, h, b = c.observed_indices, target, c.reference.cross
    k = c.reference.covariance
    code_posterior = np.eye(2) - b[o].T @ np.linalg.solve(k[np.ix_(o, o)], b[o])
    wrong = residual + b @ code_posterior @ b.T
    assert np.linalg.norm(wrong[np.ix_(h, h)] - zero.covariance[np.ix_(h, h)]) > 1e-6
    checks.append("unconditioned_residual_surrogate_is_detectably_wrong")

    # Target and unseen roles never become implicit observations.
    _raises(lambda: ArbitraryMaskConditioner(c.reference, c.roles, ObservedReadings(
        c.roles.identity, [5, 0, 2, 6], fixed[[5, 0, 2, 6]])))
    _raises(lambda: c.condition(target_indices=[5, 6]))
    _raises(lambda: c.condition(target_indices=[1, 1]))
    _raises(lambda: c.condition(target_indices=[True]))
    _raises(lambda: c.condition(target_indices=[1.0]))
    _raises(lambda: c.condition(target_indices=[7]))
    _raises(lambda: GaussianConditionalScore(c, ScoringReadings(
        c.roles.identity, "unseen", [6], fixed[[6]])))
    _raises(lambda: GaussianConditionalScore(c, ScoringReadings(
        c.roles.identity, "sentinel_dev", [6], fixed[[6]])))
    _raises(lambda: _make(core, noise=-0.1))
    checks.append("role_identity_index_and_variance_boundaries")

    # Empty and fully observed masks have well-defined generation semantics.
    for observed in ((), tuple(range(7))):
        edge, _, _ = _make(core, observed=observed)
        edge_post = edge.condition()
        draws = edge_post.sample(3, np.random.default_rng(173))
        assert draws.shape == (3, 7 - len(observed))
    draws = exact.sample(4, np.random.default_rng(173), target_only=False)
    np.testing.assert_array_equal(draws[:, c.observed_indices],
                                  np.broadcast_to(c.observed_values, (4, len(c.observed_indices))))
    checks.append("empty_full_masks_and_correlated_generation_exact_anchors")

    # Finite differences are a synthetic oracle, not material-score evaluation.
    dev, dev_fixed, _ = _make(core, split="dev", noise=[.02, .05, .01])
    labels = ScoringReadings(dev.roles.identity, "sentinel_dev", dev.roles.sentinel_dev,
                            dev_fixed[dev.roles.sentinel_dev], [.03, .04])
    dev_score = GaussianConditionalScore(dev, labels)
    value, gradient = dev_score(theta)
    step = 1e-6
    finite = np.array([(dev_score(theta + step * unit)[0] - dev_score(theta - step * unit)[0]) / (2 * step)
                       for unit in np.eye(5)])
    np.testing.assert_allclose(gradient, finite, atol=2e-8, rtol=2e-5)
    # Independent conditional target likelihood, including target label noise.
    dm, dk = dev.condition(theta).marginal(dev.roles.sentinel_dev)
    dk = dk + np.diag([.03, .04])
    error = labels.values - dm
    expected = .5 * (len(error) * np.log(2 * np.pi) + np.linalg.slogdet(dk)[1] + error @ np.linalg.solve(dk, error))
    np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-12)
    checks.append("noisy_conditional_NLL_analytic_gradient_and_independent_density")

    train_score = GaussianConditionalScore(c, ScoringReadings(
        c.roles.identity, "pseudo_hole_train", c.roles.pseudo_hole_train,
        fixed[c.roles.pseudo_hole_train]))
    report = inspect_parameter_update(theta, [train_score], [dev_score],
                                      step_sizes=[0.0, .001], evaluate_development_steps=True,
                                      development_use_id="fixed_synthetic_gate_v1")
    assert report["development_score_calls"] == 2 and report["selected_step"] is None
    assert report["unseen_labels_used"] is False and report["optimizer_runs"] == 0
    np.testing.assert_allclose(report["steps"][0]["actual_development_nll_change"], 0.0, atol=0.0)
    checks.append("early_influence_step_sensitivity_development_accounting_no_selection")

    return {
        "status": "passed", "scope": "fixed_matrix_synthetic_only",
        "completed_actual_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_actual_utc": authorization["root_actual_utc"],
        "source_sha256": hashes, "frozen_core_sha256": FROZEN_CORE_SHA256,
        "checks": checks, "materials_loaded": 0, "material_fits": 0,
        "old_25_scores": 0, "model_json_loaded": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-authorization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_gates(args.root_authorization)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Do not overwrite a prior gate receipt")
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
