"""Finite-grid Gaussian curve repair with explicit label roles.

This module never reads files, fits a model, accepts complete hidden curves, or
updates a split. All curve values are in the frozen reference's log-viscosity
coordinates. Extra observation noise is a VARIANCE in those coordinates.
The reference's existing discrepancy/nugget remains part of its full covariance.
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _array(value, name, ndim):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "iuf" or
            any(isinstance(x, (bool, np.bool_))
                for x in np.asarray(value, dtype=object).flat)):
        raise ValueError(f"{name} must be real numeric, not boolean")
    out = np.array(raw, dtype=float, copy=True)
    if out.ndim != ndim or not np.isfinite(out).all():
        raise ValueError(f"{name} has invalid shape or nonfinite values")
    out.setflags(write=False)
    return out


def _indices(value, n, name="indices"):
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    # An empty Python list has float dtype but contains no ambiguous index.
    if raw.size == 0:
        out = np.array([], dtype=int)
    else:
        if (raw.dtype.kind not in "iu" or
                any(isinstance(x, (bool, np.bool_))
                    for x in np.asarray(value, dtype=object).flat)):
            raise ValueError(f"{name} requires integer, nonboolean indices")
        if (raw < 0).any() or (raw >= n).any():
            raise ValueError(f"{name} is outside the frozen finite grid")
        out = np.array(raw, dtype=int, copy=True)
        if len(np.unique(out)) != len(out):
            raise ValueError(f"{name} has duplicate indices")
    out.setflags(write=False)
    return out


def _variance(value, size, name):
    raw = _array(value, name, np.asarray(value).ndim)
    if raw.ndim == 0:
        out = np.full(size, float(raw))
    elif raw.shape == (size,):
        out = raw.copy()
    else:
        raise ValueError(f"{name} must be scalar or index-aligned vector")
    if (raw < 0).any():
        raise ValueError(f"{name} must be nonnegative variance, not std")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class CurveIdentity:
    """Caller-supplied frozen provenance; alias/group closure is external."""
    source_id: str
    group_id: str
    curve_id: str
    split: Literal["train", "dev", "unseen"]

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip()
               for v in (self.source_id, self.group_id, self.curve_id)):
            raise ValueError("Nonempty source/group/curve identity required")
        if self.split not in ("train", "dev", "unseen"):
            raise ValueError("split must be train, dev or unseen")


@dataclass(frozen=True)
class MaskRoles:
    """Disjoint exhaustive roles on an already fixed finite curve grid.

    'observed' alone enters conditioning. Pseudo-hole labels are available only
    on training curves; sentinel labels only on development curves. 'unseen'
    has no truth-reading API. Prediction may target any non-observed role.
    """
    identity: CurveIdentity
    size: int
    observed: object
    pseudo_hole_train: object
    sentinel_dev: object
    unseen: object

    def __post_init__(self):
        if not isinstance(self.identity, CurveIdentity):
            raise ValueError("A CurveIdentity is required")
        if isinstance(self.size, (bool, np.bool_)) or not isinstance(self.size, (int, np.integer)) or self.size < 1:
            raise ValueError("A positive finite-grid size is required")
        fields = ("observed", "pseudo_hole_train", "sentinel_dev", "unseen")
        parts = [_indices(getattr(self, f), self.size, f) for f in fields]
        joined = np.concatenate(parts)
        if len(joined) != self.size or not np.array_equal(np.sort(joined), np.arange(self.size)):
            raise ValueError("Roles must be a disjoint exhaustive grid partition")
        if len(parts[1]) and self.identity.split != "train":
            raise ValueError("Pseudo-hole labels belong only to train curves")
        if len(parts[2]) and self.identity.split != "dev":
            raise ValueError("Sentinel labels belong only to dev curves")
        for field, part in zip(fields, parts):
            object.__setattr__(self, field, part)


@dataclass(frozen=True)
class ObservedReadings:
    """Only genuinely available readings; index and variance orders match values."""
    identity: CurveIdentity
    indices: object
    values: object
    additional_noise_variance: object = 0.0


@dataclass(frozen=True)
class PosteriorGaussian:
    """Full-grid conditional latent law, with an explicit requested target order."""
    mean: object
    covariance: object
    observed_indices: object
    target_indices: object
    additional_observation_noise_variance: object
    theta: object
    roles: MaskRoles
    coordinate_system: str = "frozen_reference_log_viscosity"

    def __post_init__(self):
        n = self.roles.size
        for name, ndim in (("mean", 1), ("covariance", 2), ("theta", 1)):
            object.__setattr__(self, name, _array(getattr(self, name), name, ndim))
        if self.mean.shape != (n,) or self.covariance.shape != (n, n) or self.theta.shape != (5,):
            raise ValueError("Posterior dimensions differ from its frozen grid")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-12, rtol=1e-12):
            raise ValueError("Symmetric posterior covariance required")
        o = _indices(self.observed_indices, n, "observed_indices")
        t = _indices(self.target_indices, n, "target_indices")
        if set(o.tolist()) != set(self.roles.observed.tolist()) or np.intersect1d(o, t).size:
            raise ValueError("Posterior observations must match roles and exclude targets")
        object.__setattr__(self, "observed_indices", o)
        object.__setattr__(self, "target_indices", t)
        object.__setattr__(self, "additional_observation_noise_variance",
                           _variance(self.additional_observation_noise_variance, len(o), "noise variance"))

    def marginal(self, indices=None):
        """Return a mean/covariance pair in caller order (default: targets)."""
        ix = self.target_indices if indices is None else _indices(indices, len(self.mean))
        return self.mean[ix].copy(), self.covariance[np.ix_(ix, ix)].copy()

    def sample(self, count, rng, *, target_only=True):
        """Generate correlated log-coordinate draws, without touching truth.

        Singular exact-observation rows are allowed. Only eigenvalues within
        roundoff tolerance of zero are set to zero; no jitter is added.
        """
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("count must be a positive integer")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("Pass an explicit numpy.random.Generator")
        ix = self.target_indices if target_only else np.arange(len(self.mean))
        m, k = self.marginal(ix)
        if not len(ix):
            return np.empty((count, 0))
        eig, vectors = np.linalg.eigh(k)
        tol = 64 * np.finfo(float).eps * max(1, len(ix)) * max(1.0, np.max(np.abs(eig)))
        if eig[0] < -tol:
            raise ValueError("Posterior is not positive semidefinite; no jitter fallback")
        factor = vectors * np.sqrt(np.maximum(eig, 0.0))
        draws = m + rng.standard_normal((count, len(ix))) @ factor.T
        if not target_only:
            exact = self.observed_indices[self.additional_observation_noise_variance == 0]
            draws[:, exact] = self.mean[exact]
        return draws


class ArbitraryMaskConditioner:
    """Condition the frozen full Gaussian on arbitrary finite-grid anchor sets.

    Zero noise delegates the missing-coordinate block to CodeReference.condition.
    Extra independent observation noise uses the same FULL joint covariance.
    This does not condition only the two-dimensional code and leave its residual
    unconditional. No optimizer, learned flow, or material-data loading occurs.
    """
    def __init__(self, reference, roles, readings):
        if not isinstance(roles, MaskRoles) or not isinstance(readings, ObservedReadings):
            raise ValueError("Explicit MaskRoles and ObservedReadings are required")
        if readings.identity != roles.identity:
            raise ValueError("Reading identity differs from the frozen role manifest")
        n = roles.size
        if np.asarray(reference.mean).shape != (n,):
            raise ValueError("Reference and role manifest grids differ")
        if not callable(getattr(reference, "joint", None)) or not callable(getattr(reference, "condition", None)):
            raise ValueError("A frozen CodeReference-compatible object is required")
        o = _indices(readings.indices, n, "observed readings")
        if set(o.tolist()) != set(roles.observed.tolist()):
            raise ValueError("Only all manifest-observed readings may enter conditioning")
        values = _array(readings.values, "observed values", 1)
        if values.shape != (len(o),):
            raise ValueError("Observed values must align with their indices")
        self.reference, self.roles = reference, roles
        self.observed_indices, self.observed_values = o, values
        self.noise_variance = _variance(readings.additional_noise_variance, len(o), "additional_noise_variance")

    def condition(self, theta=None, *, v=1.0, target_indices=None):
        """Return the full posterior, retaining caller target/index order.

        theta is the existing five-parameter code-Gaussian correction. The
        explicit path is theta_effective = v*theta, 0<=v<=1. v=0 is exactly the
        unchanged full-reference conditional law, including its residual. This
        path is a Gaussian parameter path, not a nonlinear probability flow.
        """
        theta = np.zeros(5) if theta is None else _array(theta, "theta", 1)
        if theta.shape != (5,):
            raise ValueError("Exactly five Gaussian code parameters required")
        strength = _array(v, "v", 0)
        if not 0 <= float(strength) <= 1:
            raise ValueError("v must lie in [0,1]")
        effective_theta = np.zeros(5) if float(strength) == 0 else float(strength) * theta
        n, o = self.roles.size, self.observed_indices
        missing = np.setdiff1d(np.arange(n), o)
        t = missing if target_indices is None else _indices(target_indices, n, "target_indices")
        if np.intersect1d(o, t).size:
            raise ValueError("Targets must not include manifest-observed coordinates")
        mean, covariance = self.reference.joint(effective_theta)
        mean, covariance = np.array(mean, copy=True), np.array(covariance, copy=True)
        if len(o) and np.all(self.noise_variance == 0):
            # Original mathematical core supplies the full-residual conditional.
            hm, hk = self.reference.condition(effective_theta, o, self.observed_values, missing)
            mean[missing], mean[o] = hm, self.observed_values
            covariance[:] = 0.0
            covariance[np.ix_(missing, missing)] = hk
        elif len(o):
            koo = covariance[np.ix_(o, o)] + np.diag(self.noise_variance)
            factor = cho_factor(koo, lower=True)
            cross = covariance[:, o].copy()
            mean += cross @ cho_solve(factor, self.observed_values - mean[o])
            covariance -= cross @ cho_solve(factor, cross.T)
            covariance = (covariance + covariance.T) / 2
            # Exactly measured coordinates are fixed even in mixed-noise sets.
            exact = np.flatnonzero(self.noise_variance == 0)
            if len(exact):
                at = o[exact]
                mean[at] = self.observed_values[exact]
                covariance[at, :], covariance[:, at] = 0.0, 0.0
        return PosteriorGaussian(mean, covariance, o, t, self.noise_variance,
                                 effective_theta, self.roles)
