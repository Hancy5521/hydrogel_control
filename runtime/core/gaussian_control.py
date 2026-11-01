"""Independent five-coefficient conditional Gaussian control.

The caller freezes the reference GP, legal context, target coordinates and their
support normalization *before* creating target labels. This module never learns
that normalization or borrows a fitted non-Gaussian model. The exact objective is
the explicit-weight mean task NLL plus 0.02 * sum(theta**2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


RIDGE = 0.02
MAX_ITERATIONS = 400
PROJECTED_GRADIENT_GATE = 1e-5
PARAMETER_NAMES = ("a0", "a1", "a2", "b0", "b1")
BOUNDS = ((-1.0, 1.0),) * 3 + ((-0.75, 0.75),) * 2


def _array(value, name, ndim, shape=None):
    """Copy to an actually immutable, finite, real float64 array."""
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real numeric array")
    try:
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real numeric array") from exc
    if result.ndim != ndim or (shape is not None and result.shape != shape):
        raise ValueError(f"{name} has incompatible shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    # Bytes-backed storage prevents callers from re-enabling the write flag.
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)


def _scalar(value, name, *, minimum=None, strict=False):
    if any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} must not be boolean")
    raw = np.asarray(value)
    if raw.ndim != 0 or np.iscomplexobj(raw) or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        raise ValueError(f"{name} must be {'greater than' if strict else 'at least'} {minimum}")
    return result


def _covariance(value, size, name="V"):
    covariance = _array(value, name, 2, (size, size))
    if size == 0:
        raise ValueError("Nonempty Gaussian required")
    # Tolerate roundoff in a Schur complement; never add jitter or discard axes.
    scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric")
    covariance = _array(covariance * 0.5 + covariance.T * 0.5, name, 2)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite; no automatic jitter") from exc
    return covariance, _array(chol, f"Cholesky({name})", 2)


def context_feature(y_context, m_context, K_context):
    """log1p(context Mahalanobis / context count), using a Cholesky solve.

    Empty contexts have no registered role and are rejected. No target value is
    accepted by this helper. The caller supplies the reference context marginal.
    """
    y = _array(y_context, "y_context", 1)
    m = _array(m_context, "m_context", 1, y.shape)
    _, chol = _covariance(K_context, y.size, "K_context")
    residual = solve_triangular(chol, y - m, lower=True, check_finite=False)
    with np.errstate(over="raise", invalid="raise"):
        try:
            h = np.log1p(float(residual @ residual) / y.size)
        except FloatingPointError as exc:
            raise ValueError("Nonfinite context Mahalanobis distance") from exc
    return _scalar(h, "h", minimum=0.0)


@dataclass(frozen=True, eq=False, slots=True)
class GaussianDesign:
    """Target-label-free conditional design, with all matrix work precomputed."""

    m: np.ndarray
    V: np.ndarray
    M: np.ndarray
    h: float
    chol: np.ndarray = field(init=False, repr=False)
    whitened_M: np.ndarray = field(init=False, repr=False)
    logdet: float = field(init=False)
    constant: float = field(init=False)

    def __post_init__(self):
        m = _array(self.m, "m", 1)
        V, chol = _covariance(self.V, m.size)
        M = _array(self.M, "M", 2, (m.size, 3))
        h = _scalar(self.h, "h", minimum=0.0)
        whitened_M = _array(solve_triangular(chol, M, lower=True, check_finite=False),
                            "whitened M", 2)
        logdet = _scalar(2.0 * np.log(np.diag(chol)).sum(), "logdet V")
        constant = _scalar(0.5 * (m.size * np.log(2.0 * np.pi) + logdet),
                           "Gaussian constant")
        for key, value in (("m", m), ("V", V), ("M", M), ("h", h),
                           ("chol", chol), ("whitened_M", whitened_M),
                           ("logdet", logdet), ("constant", constant)):
            object.__setattr__(self, key, value)

    @classmethod
    def from_reference(cls, m, V, u, h):
        """Build M=diag(sqrt(diag(V)))[1,u,h] using caller-frozen u.

        ``u`` is already normalized on the caller's frozen physical support.
        It is never normalized using this task's extent or any target label.
        """
        m = _array(m, "m", 1)
        V, _ = _covariance(V, m.size)
        u = _array(u, "caller-frozen u", 1, m.shape)
        h = _scalar(h, "h", minimum=0.0)
        with np.errstate(over="raise", invalid="raise"):
            try:
                M = np.sqrt(np.diag(V))[:, None] * np.column_stack(
                    (np.ones(m.size), u, np.full(m.size, h)))
            except FloatingPointError as exc:
                raise ValueError("Nonfinite standardized mean design") from exc
        return cls(m, V, M, h)

    @property
    def n(self):
        return self.m.size


@dataclass(frozen=True, eq=False, slots=True)
class GaussianTrainingTask:
    """One immutable training task. Labels only enter whitened residuals."""

    design: GaussianDesign
    y: np.ndarray
    weight: float = 1.0
    whitened_residual: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.design, GaussianDesign):
            raise ValueError("design must be a GaussianDesign")
        y = _array(self.y, "y", 1, self.design.m.shape)
        weight = _scalar(self.weight, "weight", minimum=0.0, strict=True)
        with np.errstate(over="raise", invalid="raise"):
            try:
                residual = y - self.design.m
            except FloatingPointError as exc:
                raise ValueError("Nonfinite training residual") from exc
        whitened = _array(solve_triangular(self.design.chol, residual, lower=True,
                                           check_finite=False), "whitened residual", 1)
        object.__setattr__(self, "y", y)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "whitened_residual", whitened)

    @classmethod
    def from_arrays(cls, m, V, M, h, y, weight=1.0):
        return cls(GaussianDesign(m, V, M, h), y, weight)

    @classmethod
    def from_reference(cls, m, V, u, h, y, weight=1.0):
        return cls(GaussianDesign.from_reference(m, V, u, h), y, weight)


def _tasks_weights(tasks):
    tasks = tuple(tasks)
    if not tasks or not all(isinstance(task, GaussianTrainingTask) for task in tasks):
        raise ValueError("Nonempty GaussianTrainingTask collection required")
    weights = np.array([task.weight for task in tasks])
    # This scale-invariant normalization avoids overflow in a raw weight sum.
    weights /= weights.max()
    weights /= weights.sum()
    if not np.all(weights > 0.0):
        raise ValueError("Weights have a nonrepresentable relative dynamic range")
    return tasks, _array(weights, "normalized weights", 1)


@dataclass(frozen=True, eq=False, slots=True)
class GaussianObjective:
    """Weighted mean NLL + fixed RIDGE*theta@theta and analytic gradient."""

    tasks: tuple[GaussianTrainingTask, ...]
    normalized_weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        tasks, weights = _tasks_weights(self.tasks)
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "normalized_weights", weights)

    def __call__(self, theta):
        theta = _array(theta, "theta", 1, (5,))
        a, b = theta[:3], theta[3:]
        value, gradient = 0.0, np.zeros(5)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for task, weight in zip(self.tasks, self.normalized_weights):
                    design = task.design
                    b_row = np.array((1.0, design.h))
                    eta = float(b_row @ b)
                    scale_precision = np.exp(-2.0 * eta)
                    if scale_precision == 0.0:
                        raise FloatingPointError("Scale precision underflow")
                    residual = task.whitened_residual - design.whitened_M @ a
                    q = float(residual @ residual)
                    value += weight * (design.constant + design.n * eta
                                       + 0.5 * scale_precision * q)
                    gradient[:3] -= weight * scale_precision * (design.whitened_M.T @ residual)
                    gradient[3:] += weight * (design.n - scale_precision * q) * b_row
                value += RIDGE * float(theta @ theta)
                gradient += 2.0 * RIDGE * theta
            except FloatingPointError as exc:
                raise ValueError("Nonfinite Gaussian objective; fail fast without clipping") from exc
        return _scalar(value, "objective"), np.array(_array(gradient, "gradient", 1), copy=True)


@dataclass(frozen=True, eq=False, slots=True)
class GaussianRankDiagnostic:
    mean_information: np.ndarray
    mean_singular_values: np.ndarray
    mean_rank: int
    mean_rank_tolerance: float
    b_design: np.ndarray
    weighted_b_design: np.ndarray
    b_singular_values: np.ndarray
    b_rank: int
    b_rank_tolerance: float
    normalized_weights: np.ndarray
    n_tasks: int
    n_targets: int

    def __post_init__(self):
        for name, ndim, shape in (
                ("mean_information", 2, (3, 3)),
                ("mean_singular_values", 1, None),
                ("b_design", 2, (self.n_tasks, 2)),
                ("weighted_b_design", 2, (self.n_tasks, 2)),
                ("b_singular_values", 1, None),
                ("normalized_weights", 1, (self.n_tasks,))):
            object.__setattr__(self, name, _array(getattr(self, name), name, ndim, shape))
        if not (0 <= self.mean_rank <= 3 and 0 <= self.b_rank <= 2):
            raise ValueError("Invalid design rank")
        if self.n_tasks <= 0 or self.n_targets < self.n_tasks:
            raise ValueError("Invalid training design counts")
        for name in ("mean_rank_tolerance", "b_rank_tolerance"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name, minimum=0.0))

    @property
    def identifiable_parameter_count(self):
        return self.mean_rank + self.b_rank

    def to_dict(self):
        return {"nominal_parameter_count": 5,
                "identifiable_parameter_count": self.identifiable_parameter_count,
                "mean_rank": self.mean_rank, "b_rank": self.b_rank,
                "mean_information": self.mean_information.tolist(),
                "mean_singular_values": self.mean_singular_values.tolist(),
                "mean_rank_tolerance": self.mean_rank_tolerance,
                "b_design": self.b_design.tolist(),
                "weighted_b_design": self.weighted_b_design.tolist(),
                "b_singular_values": self.b_singular_values.tolist(),
                "b_rank_tolerance": self.b_rank_tolerance,
                "normalized_weights": self.normalized_weights.tolist(),
                "n_tasks": self.n_tasks, "n_targets": self.n_targets,
                "rank_policy": "All five axes retained under registered ridge; rank is data-design identifiability, not ridge-created information.",
                "mean_information_definition": "sum normalized_weight * M.T @ solve(V,M)",
                "rank_definition": "SVD of stacked sqrt(weight)*solve(chol(V),M), and sqrt(weight)*[1,h]; tol=max(shape)*eps*largest_singular_value"}


def _singular_rank(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    tolerance = float(max(matrix.shape) * np.finfo(float).eps * singular[0])
    return _array(singular, "singular values", 1), int(np.sum(singular > tolerance)), tolerance


def rank_diagnostic(tasks):
    """Use every positive-weight training task, never target outcomes, for rank."""
    tasks, weights = _tasks_weights(tasks)
    mean_design = np.vstack([np.sqrt(weight) * task.design.whitened_M
                             for task, weight in zip(tasks, weights)])
    b_design = np.array([[1.0, task.design.h] for task in tasks])
    weighted_b = np.sqrt(weights)[:, None] * b_design
    mean_singular, mean_rank, mean_tol = _singular_rank(mean_design)
    b_singular, b_rank, b_tol = _singular_rank(weighted_b)
    with np.errstate(over="raise", invalid="raise"):
        try:
            information = mean_design.T @ mean_design
        except FloatingPointError as exc:
            raise ValueError("Nonfinite mean information matrix") from exc
    return GaussianRankDiagnostic(
        _array(information, "mean information", 2), mean_singular, mean_rank, mean_tol,
        _array(b_design, "b design", 2), _array(weighted_b, "weighted b design", 2),
        b_singular, b_rank, b_tol, weights, len(tasks), sum(task.design.n for task in tasks))


def projected_gradient(theta, gradient, *, box_constraints=True):
    """Bound KKT residual: zero outward gradient at an active bound."""
    theta = _array(theta, "theta", 1, (5,))
    gradient = np.array(_array(gradient, "gradient", 1, (5,)), copy=True)
    if not isinstance(box_constraints, (bool, np.bool_)):
        raise ValueError("box_constraints must be boolean")
    if box_constraints:
        lower, upper = np.array(BOUNDS).T
        if np.any(theta < lower) or np.any(theta > upper):
            raise ValueError("theta outside registered bounds")
        gradient[(theta <= lower) & (gradient > 0.0)] = 0.0
        gradient[(theta >= upper) & (gradient < 0.0)] = 0.0
    return gradient


@dataclass(frozen=True, eq=False, slots=True)
class GaussianFit:
    theta: np.ndarray
    objective: float
    gradient: np.ndarray
    projected_gradient: np.ndarray
    rank_diagnostic: GaussianRankDiagnostic
    box_constraints: bool
    iterations: int
    function_evaluations: int
    gradient_evaluations: int
    optimizer_status: int
    optimizer_message: str

    def __post_init__(self):
        for name in ("theta", "gradient", "projected_gradient"):
            object.__setattr__(self, name, _array(getattr(self, name), name, 1, (5,)))
        object.__setattr__(self, "objective", _scalar(self.objective, "fitted objective"))
        if not isinstance(self.rank_diagnostic, GaussianRankDiagnostic):
            raise ValueError("GaussianRankDiagnostic required")
        if not isinstance(self.box_constraints, (bool, np.bool_)):
            raise ValueError("box_constraints must be boolean")
        expected = projected_gradient(self.theta, self.gradient,
                                      box_constraints=self.box_constraints)
        if not np.array_equal(expected, self.projected_gradient):
            raise ValueError("Inconsistent fitted projected gradient")
        if self.optimizer_status != 0 or np.max(np.abs(expected)) > PROJECTED_GRADIENT_GATE:
            raise ValueError("Only admitted successful Gaussian fits may be constructed")
        if not (0 <= self.iterations <= MAX_ITERATIONS):
            raise ValueError("Invalid optimizer iteration count")

    @property
    def admitted(self):
        # Construction through fit_gaussian_control is possible only after both
        # the optimizer-success and projected-gradient checks pass.
        return True

    def to_dict(self):
        return {"theta": self.theta.tolist(), "parameter_names": list(PARAMETER_NAMES),
                "objective": self.objective, "gradient": self.gradient.tolist(),
                "projected_gradient": self.projected_gradient.tolist(),
                "projected_gradient_inf": float(np.max(np.abs(self.projected_gradient))),
                "accepted": True, "success": True, "runs": 1,
                "method": "L-BFGS-B", "initial_theta": [0.0] * 5,
                "iterations": self.iterations, "nfev": self.function_evaluations,
                "njev": self.gradient_evaluations,
                "optimizer_status": self.optimizer_status,
                "optimizer_message": self.optimizer_message,
                "objective_definition": "sum(normalized task weight * target-vector NLL) + 0.02 * sum(theta**2)",
                "ridge": RIDGE, "maxiter": MAX_ITERATIONS,
                "optimizer_gtol": 1e-7, "optimizer_ftol": 1e-15,
                "optimizer_maxls": 40, "projected_gradient_gate": PROJECTED_GRADIENT_GATE,
                "box_constraints": self.box_constraints,
                "bounds": [list(bound) for bound in BOUNDS] if self.box_constraints else None,
                "rank_diagnostic": self.rank_diagnostic.to_dict()}


class GaussianFitError(RuntimeError):
    """Failed optimization or admission; no fitted control is returned."""

    def __init__(self, message, *, receipt=None):
        super().__init__(message)
        self.receipt = receipt


def _failure_receipt(diagnostic, box_constraints):
    return {"admitted": False, "runs": 1, "method": "L-BFGS-B",
            "initial_theta": [0.0] * 5, "ridge": RIDGE,
            "objective_definition": "sum(normalized task weight * target-vector NLL) + 0.02 * sum(theta**2)",
            "maxiter": MAX_ITERATIONS, "optimizer_gtol": 1e-7,
            "optimizer_ftol": 1e-15, "optimizer_maxls": 40,
            "projected_gradient_gate": PROJECTED_GRADIENT_GATE,
            "box_constraints": bool(box_constraints),
            "bounds": [list(bound) for bound in BOUNDS] if box_constraints else None,
            "rank_diagnostic": diagnostic.to_dict()}


def fit_gaussian_control(tasks: Iterable[GaussianTrainingTask], *, box_constraints=True):
    """One deterministic fit from zero, with fixed ridge/budget/admission.

    There are no restarts, rank-based axis removal, jitter, output-based tuning,
    or tolerance retries. Rank deficiency is exposed in the returned diagnostic;
    ridge retains all axes and supplies a unique penalized mean for fixed scale.
    """
    if not isinstance(box_constraints, (bool, np.bool_)):
        raise ValueError("box_constraints must be boolean")
    objective = GaussianObjective(tuple(tasks))
    diagnostic = rank_diagnostic(objective.tasks)
    try:
        result = minimize(objective, np.zeros(5), jac=True, method="L-BFGS-B",
                          bounds=BOUNDS if box_constraints else None,
                          options={"maxiter": MAX_ITERATIONS, "gtol": 1e-7,
                                   "ftol": 1e-15, "maxls": 40})
        value, gradient = objective(result.x)
        projected = projected_gradient(result.x, gradient, box_constraints=box_constraints)
    except Exception as exc:
        # Preserve an explicit unknown-endpoint receipt and re-raise, including
        # unexpected optimizer failures. This is not a fallback or retry.
        receipt = _failure_receipt(diagnostic, box_constraints)
        receipt.update({"endpoint_known": False, "success": None,
                        "theta": None, "objective": None, "gradient": None,
                        "projected_gradient": None, "projected_gradient_inf": None,
                        "iterations": None, "nfev": None, "njev": None,
                        "optimizer_status": None, "optimizer_message": None,
                        "exception_type": type(exc).__name__, "exception_message": str(exc)})
        raise GaussianFitError(f"Gaussian fit failed fast: {exc}", receipt=receipt) from exc
    residual = float(np.max(np.abs(projected)))
    if not result.success or residual > PROJECTED_GRADIENT_GATE:
        receipt = _failure_receipt(diagnostic, box_constraints)
        receipt.update({"endpoint_known": True, "success": bool(result.success),
                        "theta": np.asarray(result.x).tolist(), "objective": value,
                        "gradient": gradient.tolist(), "projected_gradient": projected.tolist(),
                        "projected_gradient_inf": residual, "iterations": int(result.nit),
                        "nfev": int(result.nfev), "njev": int(result.njev),
                        "optimizer_status": int(result.status), "optimizer_message": str(result.message)})
        raise GaussianFitError(f"Gaussian fit rejected: success={bool(result.success)}, "
                               f"status={result.status}, projected_gradient_inf={residual:.17g}, "
                               f"gate={PROJECTED_GRADIENT_GATE}, message={result.message}",
                               receipt=receipt)
    return GaussianFit(_array(result.x, "fitted theta", 1, (5,)), value,
                       _array(gradient, "fitted gradient", 1, (5,)),
                       _array(projected, "projected gradient", 1, (5,)), diagnostic,
                       bool(box_constraints), int(result.nit), int(result.nfev),
                       int(result.njev), int(result.status), str(result.message))


@dataclass(frozen=True, eq=False, slots=True)
class GaussianPrediction:
    """Full multivariate conditional Gaussian; covariance is never diagonalized."""

    mean: np.ndarray
    covariance: np.ndarray
    eta: float
    chol: np.ndarray = field(init=False, repr=False)
    logdet: float = field(init=False)

    def __post_init__(self):
        mean = _array(self.mean, "prediction mean", 1)
        covariance, chol = _covariance(self.covariance, mean.size, "prediction covariance")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "eta", _scalar(self.eta, "eta"))
        object.__setattr__(self, "chol", chol)
        object.__setattr__(self, "logdet", _scalar(2 * np.log(np.diag(chol)).sum(), "prediction logdet"))

    def logpdf(self, y):
        y = _array(y, "prediction target y", 1, self.mean.shape)
        with np.errstate(over="raise", invalid="raise"):
            try:
                residual = solve_triangular(self.chol, y - self.mean, lower=True, check_finite=False)
                value = -0.5 * (self.mean.size * np.log(2 * np.pi) + self.logdet + residual @ residual)
            except FloatingPointError as exc:
                raise ValueError("Nonfinite prediction log density") from exc
        return _scalar(value, "prediction log density")


def predict_gaussian(theta, design):
    """Callable without fitting, including theta=0 to recover the reference GP."""
    theta = _array(theta, "theta", 1, (5,))
    if not isinstance(design, GaussianDesign):
        raise ValueError("design must be a GaussianDesign")
    with np.errstate(over="raise", invalid="raise", under="raise"):
        try:
            eta = float(theta[3] + theta[4] * design.h)
            mean = design.m + design.M @ theta[:3]
            covariance = np.exp(2.0 * eta) * design.V
        except FloatingPointError as exc:
            raise ValueError("Nonfinite Gaussian prediction; no scale clipping") from exc
    return GaussianPrediction(mean, covariance, eta)
