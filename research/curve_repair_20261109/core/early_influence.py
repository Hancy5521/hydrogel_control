"""Local parameter-update diagnostics using explicitly permitted held-out labels.

No final/unseen values have an input path. Sentinel labels are development
labels: any repeated use is tuning, never a new independent test result.
This module does not fit, select a learning rate, or claim unknown-truth access.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .arbitrary_mask import (
    ArbitraryMaskConditioner, CurveIdentity, _array, _indices, _variance,
)


@dataclass(frozen=True)
class ScoringReadings:
    """Withheld labels, separate from the conditioning observations."""
    identity: CurveIdentity
    role: str  # exactly pseudo_hole_train or sentinel_dev
    indices: object
    values: object
    additional_noise_variance: object = 0.0


def _joint_nll_gradient(reference, theta, index, values, noise):
    """Frozen joint likelihood with optional extra independent label noise."""
    if not len(index):
        return 0.0, np.zeros(5)
    if np.all(noise == 0):
        return reference.nll_gradient(theta, values, index)
    # Same five-parameter derivative as the frozen core, with K_II + diag(R).
    mean, covariance = reference.joint(theta)
    b = reference.cross[index]
    k = covariance[np.ix_(index, index)] + np.diag(noise)
    factor = cho_factor(k, lower=True)
    error = values - mean[index]
    weight = cho_solve(factor, error)
    wb = b.T @ weight
    value = 0.5 * (len(index) * np.log(2 * np.pi) +
                   2 * np.log(np.diag(factor[0])).sum() + error @ weight)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        l00, l10, l11 = np.exp(theta[2]), theta[3], np.exp(theta[4])
        ds = np.array([[[2 * l00**2, l00 * l10], [l00 * l10, 0.0]],
                       [[0.0, l00], [l00, 2 * l10]],
                       [[0.0, 0.0], [0.0, 2 * l11**2]]])
    gs = 0.5 * (b.T @ cho_solve(factor, b) - np.outer(wb, wb))
    return float(value), np.r_[-wb, np.einsum("ij,kij->k", gs, ds)]


class GaussianConditionalScore:
    """Exact conditional NLL and analytic gradient for one authorized mask task.

    NLL(Y_T|Y_O) = NLL(Y_[O,T]) - NLL(Y_O), using complete high-dimensional
    covariance, not a score in code space alone. The target label payload can
    contain only the manifest's train pseudo-hole OR development sentinel role.
    """
    def __init__(self, conditioner, labels):
        if not isinstance(conditioner, ArbitraryMaskConditioner) or not isinstance(labels, ScoringReadings):
            raise ValueError("Explicit conditioner and scoring readings required")
        if labels.identity != conditioner.roles.identity:
            raise ValueError("Scoring identity differs from conditioning identity")
        if labels.role not in ("pseudo_hole_train", "sentinel_dev"):
            raise ValueError("Unseen/observed truth may not be supplied as scoring labels")
        permitted = getattr(conditioner.roles, labels.role)
        index = _indices(labels.indices, conditioner.roles.size, "score indices")
        if not len(index) or not set(index.tolist()).issubset(set(permitted.tolist())):
            raise ValueError("Scoring labels must be a nonempty subset of their allowed role")
        values = _array(labels.values, "score labels", 1)
        if values.shape != (len(index),):
            raise ValueError("Scoring labels must align with score indices")
        self.conditioner, self.identity, self.role = conditioner, labels.identity, labels.role
        self.indices, self.values = index, values
        self.noise_variance = _variance(labels.additional_noise_variance, len(index), "score noise variance")

    def __call__(self, theta):
        theta = _array(theta, "theta", 1)
        if theta.shape != (5,):
            raise ValueError("Exactly five Gaussian code parameters required")
        c = self.conditioner
        full_index = np.r_[c.observed_indices, self.indices]
        full_values = np.r_[c.observed_values, self.values]
        full_noise = np.r_[c.noise_variance, self.noise_variance]
        full = _joint_nll_gradient(c.reference, theta, full_index, full_values, full_noise)
        context = _joint_nll_gradient(c.reference, theta, c.observed_indices,
                                      c.observed_values, c.noise_variance)
        return float(full[0] - context[0]), full[1] - context[1]


def _mean_score(scores, theta, expected_role):
    scores = tuple(scores)
    if not scores or any(not isinstance(s, GaussianConditionalScore) or s.role != expected_role for s in scores):
        raise ValueError(f"Nonempty {expected_role} score tasks required")
    out = [score(theta) for score in scores]
    return float(np.mean([x[0] for x in out])), np.mean([x[1] for x in out], axis=0)


def inspect_parameter_update(theta, training_scores, sentinel_scores, *, direction=None,
                             step_sizes=(0.0, 0.1, 0.5, 1.0), evaluate_development_steps=False,
                             development_use_id):
    """Inspect one proposed parameter direction before a complete refit.

    Defaults to the negative mean training pseudo-hole NLL gradient. Predicts
    development NLL change by <g_dev, direction>*step; this is a first-order
    local diagnostic, not a guarantee. Optional exact development endpoints
    consume development labels at every nonzero step and are explicitly counted.
    No step is selected here; no optimizer or automatic retry is called.
    Candidate endpoints are an unconstrained local diagnostic path: the frozen
    optimizer's BOUNDS are not projected or used to admit a parameter update.

    Equal weight is assigned per score task. The caller must arrange/predeclare
    per-curve masks and weights, group/alias closure and query/compute budgets.
    """
    theta = _array(theta, "theta", 1)
    if theta.shape != (5,):
        raise ValueError("Exactly five Gaussian code parameters required")
    if not isinstance(development_use_id, str) or not development_use_id.strip():
        raise ValueError("An auditable development_use_id is required")
    train, dev = tuple(training_scores), tuple(sentinel_scores)
    if (not train or not dev or
            any(not isinstance(s, GaussianConditionalScore) or s.role != "pseudo_hole_train" for s in train) or
            any(not isinstance(s, GaussianConditionalScore) or s.role != "sentinel_dev" for s in dev)):
        raise ValueError("Separate nonempty train-pseudo-hole and dev-sentinel score lists required")
    # A split manifest is not a substitute for external alias closure; this
    # catches direct group identity contamination at this interface.
    train_groups = {(s.identity.source_id, s.identity.group_id) for s in train}
    dev_groups = {(s.identity.source_id, s.identity.group_id) for s in dev}
    if train_groups & dev_groups:
        raise ValueError("Training and development group identities overlap")
    train_value, train_gradient = _mean_score(train, theta, "pseudo_hole_train")
    dev_value, dev_gradient = _mean_score(dev, theta, "sentinel_dev")
    direction = -train_gradient if direction is None else _array(direction, "direction", 1)
    if direction.shape != (5,):
        raise ValueError("The candidate direction must have five parameters")
    steps = _array(step_sizes, "step_sizes", 1)
    if not len(steps) or (steps < 0).any() or len(np.unique(steps)) != len(steps):
        raise ValueError("Unique nonnegative explicit step sizes required")
    slope = float(dev_gradient @ direction)
    records = []
    for step in steps:
        record = {"step": float(step), "predicted_development_nll_change": float(step * slope)}
        if evaluate_development_steps:
            value = dev_value if step == 0 else _mean_score(dev, theta + step * direction, "sentinel_dev")[0]
            record["actual_development_nll_change"] = float(value - dev_value)
        records.append(record)
    return {
        "development_use_id": development_use_id,
        "theta": theta.tolist(), "direction": direction.tolist(),
        "training_conditional_nll": train_value, "development_conditional_nll": dev_value,
        "training_gradient": train_gradient.tolist(), "development_gradient": dev_gradient.tolist(),
        "development_directional_derivative": slope,
        "local_warning": "predicted_harm" if slope > 0 else "no_first_order_harm_signal",
        "steps": records,
        "training_score_calls": len(train),
        "development_score_calls": len(dev) * (1 + (int(np.count_nonzero(steps)) if evaluate_development_steps else 0)),
        "sentinel_role": "development_tuning_not_final_test",
        "unseen_labels_used": False, "optimizer_runs": 0, "selected_step": None,
        "first_order_only": True,
        "constraint_policy": "unconstrained_diagnostic_not_admitted_fit_update",
    }
