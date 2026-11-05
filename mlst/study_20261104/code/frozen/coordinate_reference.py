"""Independent-recipe GP in actual log shear-rate coordinates.

Inputs are t=ln(gamma/(1 s^-1)), y=ln(eta/(1 Pa s)), never sequence
addresses or interpolated labels. The report discrepancy is a statistical
nugget, not certified instrument noise. No numerical jitter is added.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


HYPER_NAMES = ("level", "slope", "smooth", "length", "report_discrepancy")


def _vector(value, name):
    original = np.asarray(value)
    if (original.ndim != 1 or original.dtype.kind not in "iuf"
            or any(isinstance(item, (bool, np.bool_)) for item in np.asarray(value, dtype=object).flat)):
        raise ValueError(f"{name} must be a real numeric one-dimensional array")
    result = np.array(original, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")
    return result


@dataclass(frozen=True)
class Curve:
    recipe_id: str
    t: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        if not isinstance(self.recipe_id, str) or not self.recipe_id.strip():
            raise ValueError("recipe_id must be a nonempty string")
        t, y = _vector(self.t, "t"), _vector(self.y, "y")
        if len(t) == 0 or len(t) != len(y):
            raise ValueError("Each curve needs matching nonempty visible t and y")
        t.setflags(write=False)
        y.setflags(write=False)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "y", y)


def _curves(curves):
    curves = tuple(curves)
    if not curves or len({c.recipe_id for c in curves}) != len(curves):
        raise ValueError("Supply nonempty curves with unique recipe ids")
    return curves


@dataclass(frozen=True)
class Normalization:
    t_center: float
    t_scale: float
    y_center: float
    y_scale: float

    def __post_init__(self):
        for value in (self.t_center, self.t_scale, self.y_center, self.y_scale):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)):
                raise ValueError("Normalization must contain finite real numeric scalars")
        if self.t_scale <= 0 or self.y_scale <= 0:
            raise ValueError("Coordinate range and visible y standard deviation must be positive")

    @classmethod
    def from_curves(cls, curves: Sequence[Curve]):
        """Use only the supplied visible TRAIN observations; population y std."""
        curves = _curves(curves)
        t, y = np.concatenate([c.t for c in curves]), np.concatenate([c.y for c in curves])
        return cls(float((t.min() + t.max()) / 2), float(np.ptp(t) / 2),
                   float(y.mean()), float(y.std(ddof=0)))

    def u(self, t):
        return (_vector(t, "t") - self.t_center) / self.t_scale


@dataclass(frozen=True)
class FitConfig:
    """One deterministic L-BFGS-B run; amplitudes refer to standardized y."""
    initial: tuple = (0.5, 0.5, 0.5, 0.5, 0.05)
    # Modeling bounds constrain the statistical prior; they are not numerical jitter.
    bounds: tuple = ((1e-3, 10.0), (1e-3, 10.0), (1e-3, 10.0),
                     (0.02, 10.0), (1e-3, 1.0))
    maxiter: int = 150
    maxfun: int = 240
    ftol: float = 1e-10
    gtol: float = 1e-6

    def __post_init__(self):
        initial, bounds = _vector(self.initial, "initial"), np.asarray(self.bounds)
        if (initial.shape != (5,) or bounds.shape != (5, 2) or bounds.dtype.kind not in "iuf"
                or any(isinstance(item, (bool, np.bool_)) for item in np.asarray(self.bounds, dtype=object).flat)):
            raise ValueError("Exactly five initial hyperparameters and bounds are required")
        bounds = np.array(bounds, dtype=float, copy=True)
        if (not np.isfinite(initial).all() or not np.isfinite(bounds).all()
                or (bounds <= 0).any() or (bounds[:, 0] >= bounds[:, 1]).any()
                or (initial < bounds[:, 0]).any() or (initial > bounds[:, 1]).any()):
            raise ValueError("Initial positive hyperparameters must lie in finite positive bounds")
        for budget in (self.maxiter, self.maxfun):
            if (isinstance(budget, (bool, np.bool_))
                    or not isinstance(budget, (int, np.integer)) or budget < 1):
                raise ValueError("Optimization budgets must be positive integers")
        for tolerance in (self.ftol, self.gtol):
            if (isinstance(tolerance, (bool, np.bool_))
                    or not isinstance(tolerance, (int, float, np.integer, np.floating))
                    or not np.isfinite(tolerance) or tolerance <= 0):
                raise ValueError("Optimization tolerances must be finite positive real scalars")
        object.__setattr__(self, "initial", tuple(initial.tolist()))
        object.__setattr__(self, "bounds", tuple(map(tuple, bounds.tolist())))
        object.__setattr__(self, "maxiter", int(self.maxiter))
        object.__setattr__(self, "maxfun", int(self.maxfun))
        object.__setattr__(self, "ftol", float(self.ftol))
        object.__setattr__(self, "gtol", float(self.gtol))


def _kernel(u, v, log_hyper, report_diagonal=False, gradient=False):
    theta = _vector(log_hyper, "log_hyper")
    if len(theta) != 5:
        raise ValueError("Five log hyperparameters are required")
    level, slope, smooth, ell, discrepancy = np.exp(theta)
    z = np.sqrt(3.0) * np.abs(u[:, None] - v[None, :]) / ell
    a = np.full((len(u), len(v)), level ** 2)
    b = slope ** 2 * u[:, None] * v[None, :]
    c = smooth ** 2 * (1 + z) * np.exp(-z)
    d = np.zeros_like(a)
    if report_diagonal:
        if not np.array_equal(u, v):
            raise ValueError("Report diagonal applies only to the same ordered variables")
        d = discrepancy ** 2 * np.eye(len(u))
    covariance = a + b + c + d
    if not gradient:
        return covariance
    return covariance, np.stack((2 * a, 2 * b, 2 * c,
                                  smooth ** 2 * z ** 2 * np.exp(-z), 2 * d))


def observed_nll(log_hyper, curves: Sequence[Curve], normalization: Normalization):
    """Profiled ML (not REML) NLL and analytic gradient, in physical log-y units.

    beta is shared and profiled by GLS. The envelope theorem removes beta's
    derivative from the covariance gradient. No training target is imputed.
    """
    curves = _curves(curves)
    records, precision, score = [], np.zeros((2, 2)), np.zeros(2)
    for curve in curves:
        u = normalization.u(curve.t)
        x = np.column_stack((np.ones(len(u)), u))
        y = (curve.y - normalization.y_center) / normalization.y_scale
        k, dk = _kernel(u, u, log_hyper, report_diagonal=True, gradient=True)
        factor = cho_factor(k, lower=True, check_finite=True)
        ki_x, ki_y = cho_solve(factor, x), cho_solve(factor, y)
        precision += x.T @ ki_x
        score += x.T @ ki_y
        records.append((x, y, factor, dk))
    beta = np.linalg.solve(precision, score)
    nll, grad = 0.0, np.zeros(5)
    for x, y, factor, dk in records:
        residual = y - x @ beta
        alpha = cho_solve(factor, residual)
        nll += (0.5 * residual @ alpha + np.log(np.diag(factor[0])).sum()
                + len(y) * (0.5 * np.log(2 * np.pi) + np.log(normalization.y_scale)))
        weight = cho_solve(factor, np.eye(len(y))) - np.outer(alpha, alpha)
        grad += 0.5 * np.einsum("ij,kij->k", weight, dk)
    return float(nll), grad, beta


@dataclass(frozen=True)
class Gaussian:
    mean: np.ndarray
    cov: np.ndarray


def condition_joint(law: Gaussian, observed_indices, values):
    """Condition the same report addresses exactly, returning the entire law.

    Use report+latent joint laws to infer latent functions from noisy reports.
    Distinct reports at equal coordinates retain independent discrepancies.
    """
    indices = np.asarray(observed_indices)
    values = _vector(values, "observed values")
    if (indices.ndim != 1 or indices.dtype.kind not in "iu"
            or any(isinstance(item, (bool, np.bool_)) for item in np.asarray(observed_indices, dtype=object).flat)):
        raise ValueError("Observation indices must be a one-dimensional integer array")
    if len(indices) != len(values) or len(set(indices)) != len(indices):
        raise ValueError("Observation indices must be unique and match values")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(law.mean)):
        raise ValueError("Observation index out of range")
    if len(indices) == 0:
        return Gaussian(law.mean.copy(), law.cov.copy())
    cross = law.cov[:, indices]
    factor = cho_factor(law.cov[np.ix_(indices, indices)], lower=True)
    mean = law.mean + cross @ cho_solve(factor, values - law.mean[indices])
    cov = law.cov - cross @ cho_solve(factor, cross.T)
    cov = (cov + cov.T) / 2
    mean[indices], cov[indices, :], cov[:, indices] = values, 0.0, 0.0
    return Gaussian(mean, cov)


@dataclass(frozen=True)
class CoordinateGP:
    normalization: Normalization
    log_hyper: np.ndarray
    beta: np.ndarray
    fit_receipt: dict = field(default_factory=dict)

    def __post_init__(self):
        hyper, beta = _vector(self.log_hyper, "log_hyper"), _vector(self.beta, "beta")
        if len(hyper) != 5 or len(beta) != 2:
            raise ValueError("Expected five hyperparameters and two mean coefficients")
        hyper.setflags(write=False)
        beta.setflags(write=False)
        object.__setattr__(self, "log_hyper", hyper)
        object.__setattr__(self, "beta", beta)

    @classmethod
    def fixed(cls, normalization, log_hyper, beta):
        """Construct a specified law without fitting, useful for qualification."""
        return cls(normalization, log_hyper, beta, {"optimization_performed": False})

    @classmethod
    def fit(cls, curves: Sequence[Curve], config: FitConfig = FitConfig()):
        curves = _curves(curves)
        normalization = Normalization.from_curves(curves)
        def objective(theta):
            value, gradient, _ = observed_nll(theta, curves, normalization)
            return value, gradient
        result = minimize(objective, np.log(config.initial), jac=True, method="L-BFGS-B",
                          bounds=np.log(config.bounds), options={"maxiter": config.maxiter,
                          "maxfun": config.maxfun, "ftol": config.ftol, "gtol": config.gtol})
        value, gradient, beta = observed_nll(result.x, curves, normalization)
        receipt = {"optimization_performed": True, "runs": 1, "method": "L-BFGS-B",
                   "success": bool(result.success), "status": int(result.status),
                   "message": str(result.message), "nit": int(result.nit),
                   "nfev": int(result.nfev), "njev": int(result.njev),
                   "observed_nll": value, "gradient_inf": float(np.max(np.abs(gradient))),
                   "n_recipes": len(curves), "n_visible": sum(len(c.y) for c in curves),
                   "config": {key: getattr(config, key) for key in config.__dataclass_fields__}}
        return cls(normalization, result.x, beta, receipt)

    def joint(self, t, report=True):
        """Joint law at actual coordinates for one independent recipe."""
        if type(report) is not bool:
            raise ValueError("report must be a bool")
        u = self.normalization.u(t)
        mean = self.normalization.y_center + self.normalization.y_scale * (self.beta[0] + self.beta[1] * u)
        cov = self.normalization.y_scale ** 2 * _kernel(u, u, self.log_hyper, report_diagonal=report)
        return Gaussian(mean, cov)

    def report_latent_joint(self, t_report, t_latent):
        """Reports then latent f at fixed support; no discrepancy on latent f."""
        report, latent = _vector(t_report, "t_report"), _vector(t_latent, "t_latent")
        law = self.joint(np.concatenate((report, latent)), report=False)
        discrepancy = self.normalization.y_scale ** 2 * np.exp(2 * self.log_hyper[4])
        law.cov[:len(report), :len(report)] += discrepancy * np.eye(len(report))
        return law

    def condition(self, t_observed, y_observed, t_query, report=True, new_replicates=False):
        """Reports or latent queries given reports; no query values accepted.

        Report coordinates overlapping observed coordinates are rejected unless
        new_replicates=True explicitly requests new independent measurements.
        Recover stored anchors with condition_joint, never as new replicates.
        Latent queries may overlap reports and retain function uncertainty.
        """
        if type(report) is not bool or type(new_replicates) is not bool:
            raise ValueError("report and new_replicates must be bools")
        observed, query = _vector(t_observed, "t_observed"), _vector(t_query, "t_query")
        values = _vector(y_observed, "y_observed")
        if len(observed) != len(values):
            raise ValueError("Observed coordinates and values must match")
        if report and not new_replicates and np.isin(query, observed).any():
            raise ValueError("Query overlaps an observed report; use condition_joint for stored anchors or explicitly request new_replicates")
        law = self.joint(np.concatenate((observed, query)), report=report)
        if not report:
            discrepancy = self.normalization.y_scale ** 2 * np.exp(2 * self.log_hyper[4])
            law.cov[:len(observed), :len(observed)] += discrepancy * np.eye(len(observed))
        posterior = condition_joint(law, np.arange(len(observed)), values)
        return Gaussian(posterior.mean[len(observed):], posterior.cov[len(observed):, len(observed):])

    def protocol_info(self):
        return {"coordinate": "ln(gamma/(1 s^-1))", "response": "ln(eta/(1 Pa s))",
                "independence_likelihood_assumption": True,
                "physical_batch_independence_certified": False, "mean": "shared beta0+beta1*u",
                "normalization": dict(self.normalization.__dict__),
                "normalization_scope": "supplied visible TRAIN observations only",
                "hyper_names": HYPER_NAMES, "hyper_values": np.exp(self.log_hyper).tolist(),
                "beta_standardized": self.beta.tolist(), "numerical_jitter": 0.0,
                "report_discrepancy": "statistical nugget, not certified instrument noise",
                "interpolated_labels": False, "fit": self.fit_receipt}
