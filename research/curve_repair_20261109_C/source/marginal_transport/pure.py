"""Gaussian copula with fixed Student4 marginals: classical, full-rank law.

No fitting, mixture, gate, parameter search, covariance flooring or data reader.
Dependence changes; every declared marginal distribution remains unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.special import ndtr, ndtri_exp, stdtrit
from scipy.stats import t


def _real(value, name, ndim):
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iuf" or any(
            isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError(name + " requires a real numeric array")
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(name + " requires finite values")
    out.setflags(write=False)
    return out


def correlation_from_covariance(covariance):
    covariance = _real(covariance, "covariance", 2)
    if not len(covariance) or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Nonempty square covariance required")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Symmetric covariance required")
    if np.any(np.diag(covariance) <= 0):
        raise ValueError("Strictly positive variances required")
    # This is definitional symmetric correlation normalization, not a PSD
    # repair, shrinkage, eigenvalue clip or additional diagonal noise.
    symmetric = (covariance + covariance.T) / 2
    np.linalg.cholesky(symmetric)
    sd = np.sqrt(np.diag(symmetric))
    correlation = symmetric / np.outer(sd, sd)
    np.fill_diagonal(correlation, 1.0)
    np.linalg.cholesky(correlation)
    correlation.setflags(write=False)
    return correlation


@dataclass(frozen=True)
class GaussianCopulaStudent:
    location: object
    scale: object
    correlation: object
    df: int = 4

    def __post_init__(self):
        m, s = _real(self.location, "location", 1), _real(self.scale, "scale", 1)
        r = _real(self.correlation, "correlation", 2)
        if not len(m) or m.shape != s.shape or r.shape != (len(m), len(m)) or np.any(s <= 0):
            raise ValueError("Aligned nonempty marginals, positive Student scales and square correlation required")
        if isinstance(self.df, (bool, np.bool_)) or self.df != 4:
            raise ValueError("This registered mechanism fixes all marginal degrees of freedom to four")
        if not np.array_equal(np.diag(r), np.ones(len(m))) or not np.allclose(r, r.T, atol=1e-12, rtol=1e-12):
            raise ValueError("Unit diagonal symmetric correlation required; no diagonal modification here")
        factor = np.linalg.cholesky(r)
        factor.setflags(write=False)
        for name, value in (("location", m), ("scale", s), ("correlation", r), ("_factor", factor)):
            object.__setattr__(self, name, value)

    def to_latent(self, values):
        raw = np.asarray(values)
        batch = raw.ndim == 2
        y = _real(values, "values", 2 if batch else 1)
        if y.shape[-1] != len(self.location):
            raise ValueError("Values must follow the saved marginal order")
        standardized = (y - self.location) / self.scale
        if not np.isfinite(standardized).all():
            raise ArithmeticError("Nonfinite standardized readings; no clipping")
        # Only evaluate the smaller tail; evaluating CDF first may round to one.
        log_lower = np.where(standardized <= 0, t.logcdf(standardized, 4), t.logsf(standardized, 4))
        if not np.isfinite(log_lower).all():
            raise ArithmeticError("Student log-tail evaluation is not finite; no probability clipping")
        negative_normal = ndtri_exp(log_lower)
        z = np.where(standardized <= 0, negative_normal, -negative_normal)
        if not np.isfinite(z).all():
            raise ArithmeticError("Normal quantile is not finite")
        return z

    def from_latent(self, latent):
        raw = np.asarray(latent)
        z = _real(latent, "latent", 2 if raw.ndim == 2 else 1)
        if z.shape[-1] != len(self.location):
            raise ValueError("Latent dimension differs from saved marginal order")
        lower = ndtr(-np.abs(z))
        if not np.isfinite(lower).all() or np.any(lower <= 0):
            raise ArithmeticError("Representable positive normal lower-tail required; no clipping")
        negative_student = stdtrit(4.0, lower)
        standardized = np.where(z > 0, -negative_student, negative_student)
        values = self.location + self.scale * standardized
        if not np.isfinite(values).all():
            raise ArithmeticError("Student transport produced nonfinite values")
        return values

    def logpdf(self, values):
        raw = np.asarray(values)
        batch = raw.ndim == 2
        y = _real(values, "values", 2 if batch else 1)
        z = self.to_latent(y)
        standardized = (y - self.location) / self.scale
        marginals = np.sum(t.logpdf(standardized, 4) - np.log(self.scale), axis=-1)
        whitened = np.linalg.solve(self._factor, z.T).T if batch else np.linalg.solve(self._factor, z)
        copula = (-np.log(np.diag(self._factor)).sum() - .5 *
                  (np.sum(whitened * whitened, axis=-1) - np.sum(z * z, axis=-1)))
        result = marginals + copula
        if not np.isfinite(result).all():
            raise ArithmeticError("Nonfinite full copula density; no fallback")
        return result if batch else float(result)

    def sample(self, count, rng):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("Positive integer draw count required")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("Explicit numpy Generator required")
        normal = rng.standard_normal((count, len(self.location))) @ self._factor.T
        return self.from_latent(normal)

    def marginal(self, indices):
        raw = np.asarray(indices)
        if raw.ndim != 1 or raw.dtype.kind not in "iu" or len(set(raw.tolist())) != len(raw) or not len(raw):
            raise ValueError("Unique nonempty integer marginal indices required")
        idx = raw.astype(int)
        if np.any(idx < 0) or np.any(idx >= len(self.location)):
            raise ValueError("Marginal index out of range")
        return GaussianCopulaStudent(self.location[idx], self.scale[idx], self.correlation[np.ix_(idx, idx)])

    def quantile(self, probability):
        probability = _real(probability, "probability", 1)
        if np.any(probability <= 0) or np.any(probability >= 1):
            raise ValueError("Quantile probabilities must lie strictly inside (0,1)")
        return self.location[None, :] + stdtrit(4.0, probability)[:, None] * self.scale[None, :]

    def as_dict(self):
        return {"distribution": "gaussian_copula_student4_marginals", "location": self.location.tolist(),
                "scale": self.scale.tolist(), "correlation": self.correlation.tolist(), "df": 4,
                "scale_is_marginal_sd": False, "marginal_variance_factor": 2,
                "conditional_scope": "fixed-observed-set only; target-subset consistent, not asserted cross-mask Bayes compatible"}
