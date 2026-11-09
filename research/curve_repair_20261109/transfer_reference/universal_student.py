"""Universal kriging with unknown affine log-rheology trend and residual scale.

Classical correlated linear-model Student prediction, not a new probability
identity. Frozen full K retains the complete high-dimensional residual. Trend
and one scale use only the curve's six already available anchors. There is no
covariance fitting, parameter search, jitter, floor or fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


class UndefinedUniversalStudent(ValueError):
    pass


def _real(value, name, ndim):
    raw = np.asarray(value)
    if (raw.ndim != ndim or raw.dtype.kind not in 'iuf' or
            any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat)):
        raise ValueError(name + ' requires a finite real array')
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(name + ' requires finite values')
    return out


def _indices(value, n, name):
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b' or
            not len(raw) or len(np.unique(raw)) != len(raw) or
            (raw < 0).any() or (raw >= n).any()):
        raise ValueError(name + ' requires unique valid integer indices')
    return np.array(raw, dtype=int)


@dataclass(frozen=True)
class UniversalStudentForecast:
    covariance_id: str
    beta: object
    sse: float
    df: int
    residual_norm_resolution: float
    sse_resolution_threshold: float
    location: object
    scale_matrix: object
    predictive_covariance: object
    conditional_reference_covariance: object
    trend_uncertainty_covariance: object

    def as_dict(self):
        return {'distribution': 'multivariate_student_t',
                'model': 'universal_kriging_affine_log_trend_unknown_scale',
                'covariance_id': self.covariance_id,
                'beta': self.beta.tolist(), 'sse': self.sse, 'df': self.df, 'n': 6, 'p': 2,
                'residual_norm_resolution': self.residual_norm_resolution,
                'sse_resolution_threshold': self.sse_resolution_threshold,
                'resolution_multiplier': 64,
                'location': self.location.tolist(), 'scale_matrix': self.scale_matrix.tolist(),
                'predictive_covariance': self.predictive_covariance.tolist(),
                'conditional_reference_covariance': self.conditional_reference_covariance.tolist(),
                'trend_uncertainty_covariance': self.trend_uncertainty_covariance.tolist(),
                'scale_is_covariance': False,
                'prior': 'p(beta,sigma_squared) proportional to 1/sigma_squared',
                'frozen_reference_mean_used': False,
                'full_frozen_covariance_used': True, 'covariance_hyperparameter_updates': 0}


def fit_predict(log_gamma, anchor_indices, anchor_values, target_indices, full_covariance,
                covariance_id):
    """Return the exact correlated-linear-model posterior-predictive t law.

K_oo=L L'; Xw=L^-1 Xo, yw=L^-1 yo; beta=OLS(Xw,yw).
Let V=L^-1 K_ot, H=Xt-V'Xw. Then full Schur C=Ktt-V'V,
and t shape=(SSE/4)*(C+H*(Xw'Xw)^-1*H').
Frozen K=I reproduces the existing six-anchor power-law Student baseline.
"""
    x = _real(log_gamma, 'log_gamma', 1)
    k = _real(full_covariance, 'full_covariance', 2)
    y = _real(anchor_values, 'anchor_values', 1)
    if not len(x) or k.shape != (len(x), len(x)) or len(np.unique(x)) != len(x):
        raise ValueError('Distinct full-grid coordinates and matching square K required')
    if not np.allclose(k, k.T, rtol=1e-12, atol=1e-12):
        raise ValueError('Symmetric full frozen covariance required')
    np.linalg.cholesky(k)
    observed = _indices(anchor_indices, len(x), 'anchor_indices')
    targets = _indices(target_indices, len(x), 'target_indices')
    if len(observed) != 6 or y.shape != (6,) or np.intersect1d(observed, targets).size:
        raise ValueError('Exactly six own anchors, disjoint from fixed targets, required')
    if not isinstance(covariance_id, str) or not covariance_id:
        raise ValueError('Explicit covariance identity required')
    design = np.column_stack([np.ones(len(x)), x])
    xo, xt = design[observed], design[targets]
    loo = np.linalg.cholesky(k[np.ix_(observed, observed)])
    xw, yw = np.linalg.solve(loo, xo), np.linalg.solve(loo, y)
    beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=None)
    if rank != 2:
        raise UndefinedUniversalStudent('STOP_NO_FALLBACK rank-deficient whitened affine trend')
    residual = y - xo @ beta
    whitened_residual = np.linalg.solve(loo, residual)
    sse = float(whitened_residual @ whitened_residual)
    resolution = float(64 * np.finfo(float).eps *
                        (np.linalg.norm(xw, 2) * np.linalg.norm(beta) + np.linalg.norm(yw)))
    threshold = resolution ** 2
    if (np.all(y == y[0]) or not math.isfinite(sse) or sse <= 0 or
            not math.isfinite(threshold) or sse <= threshold):
        raise UndefinedUniversalStudent(
            'STOP_NO_FALLBACK numerically_unresolved_residual; '
            f'SSE={sse!r}; SSE_resolution_threshold={threshold!r}; '
            f'residual_norm_resolution={resolution!r}; no floor or replacement')
    v = np.linalg.solve(loo, k[np.ix_(observed, targets)])
    conditional = k[np.ix_(targets, targets)] - v.T @ v
    conditional = (conditional + conditional.T) / 2
    np.linalg.cholesky(conditional)
    h = xt - v.T @ xw
    trend_factor = np.linalg.cholesky(xw.T @ xw)
    trend_solution = np.linalg.solve(trend_factor, h.T)
    trend_uncertainty = trend_solution.T @ trend_solution
    mean = xt @ beta + v.T @ whitened_residual
    df = 4
    scale = (sse / df) * (conditional + trend_uncertainty)
    scale = (scale + scale.T) / 2
    np.linalg.cholesky(scale)
    covariance = scale * df / (df - 2)
    if any(not np.isfinite(a).all() for a in (beta, conditional, trend_uncertainty, mean, scale, covariance)):
        raise UndefinedUniversalStudent('STOP_NO_FALLBACK nonfinite universal predictive law')
    for array in (beta, conditional, trend_uncertainty, mean, scale, covariance):
        array.setflags(write=False)
    return UniversalStudentForecast(covariance_id, beta, sse, df, resolution, threshold,
                                    mean, scale, covariance, conditional, trend_uncertainty)
