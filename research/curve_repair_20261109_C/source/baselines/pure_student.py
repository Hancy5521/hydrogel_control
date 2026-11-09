"""Pure six-anchor rheology controls with exact linear-model Student predictions.

The prior is explicitly p(beta, sigma_squared) proportional to 1/sigma_squared
(flat beta and usual reference scale prior). This is not a new model family.
No data reader, parameter search, variance floor, jitter or fallback is present.
Student scale/shape is distinguished from predictive variance/covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.special import betaln, gammaln, stdtr, stdtrit


KINDS = ('constant_log_viscosity', 'power_law')


class UndefinedStudentPrediction(ValueError):
    """The specified improper-prior regression has no valid predictive law."""


def _real(value, name, ndim):
    raw = np.asarray(value)
    if (raw.dtype.kind not in 'iuf' or raw.ndim != ndim or
            any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat)):
        raise ValueError(name + ' requires a finite real array')
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(name + ' requires finite values')
    out.setflags(write=False)
    return out


def _design(x, kind):
    if kind not in KINDS:
        raise ValueError('Unknown preregistered physical control')
    return np.ones((len(x), 1)) if kind == KINDS[0] else np.column_stack([np.ones(len(x)), x])


@dataclass(frozen=True)
class StudentForecast:
    kind: str
    beta: object
    sse: float
    n: int
    p: int
    df: int
    residual_norm_resolution: float
    sse_resolution_threshold: float
    location: object
    scale_matrix: object
    predictive_covariance: object

    def as_dict(self):
        return {'distribution': 'multivariate_student_t',
                'kind': self.kind, 'beta': self.beta.tolist(), 'sse': self.sse,
                'n': self.n, 'p': self.p, 'df': self.df,
                'residual_norm_resolution': self.residual_norm_resolution,
                'sse_resolution_threshold': self.sse_resolution_threshold,
                'resolution_multiplier': 64,
                'location': self.location.tolist(), 'scale_matrix': self.scale_matrix.tolist(),
                'predictive_covariance': self.predictive_covariance.tolist(),
                'scale_is_covariance': False,
                'prior': 'p(beta,sigma_squared) proportional to 1/sigma_squared',
                'residual_noise_in_prediction': True}


def fit_predict(anchor_log_gamma, anchor_log_eta, target_log_gamma, kind):
    """OLS on own six anchors -> joint posterior-predictive t on fixed targets.

The predictive target is a new report reading, not the latent noiseless mean:
shape=s_squared*(I+X_target*(X_anchor'X_anchor)^-1*X_target').
SSE=0/rank failure is fatal, not repaired with a chosen floor or another model.
"""
    x = _real(anchor_log_gamma, 'anchor_log_gamma', 1)
    y = _real(anchor_log_eta, 'anchor_log_eta', 1)
    xt = _real(target_log_gamma, 'target_log_gamma', 1)
    if x.shape != (6,) or y.shape != x.shape or not len(xt):
        raise ValueError('Exactly six own anchors and at least one target required')
    if len(np.unique(x)) != len(x):
        raise ValueError('Six distinct anchor coordinates required')
    design, target = _design(x, kind), _design(xt, kind)
    n, p = design.shape
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != p or n <= p:
        raise UndefinedStudentPrediction('STOP_NO_FALLBACK rank deficient OLS or no residual degrees of freedom')
    residual = y - design @ beta
    sse = float(residual @ residual)
    # A mathematically exact line may leave a tiny positive floating residual.
    # Reject residual energy unresolved at the declared backward-error scale;
    # do not turn roundoff into a purportedly precise stochastic noise law.
    # This threshold rejects a fit; it never replaces SSE or predictive scale.
    # The factor is fixed before execution, not fitted to response outcomes.
    residual_norm_resolution = float(64 * np.finfo(float).eps *
        (np.linalg.norm(design, 2) * np.linalg.norm(beta) + np.linalg.norm(y)))
    sse_resolution_threshold = residual_norm_resolution ** 2
    if (np.all(y == y[0]) or not math.isfinite(sse) or sse <= 0 or
            not math.isfinite(sse_resolution_threshold) or sse <= sse_resolution_threshold):
        raise UndefinedStudentPrediction(
            'STOP_NO_FALLBACK numerically_unresolved_residual; '
            f'SSE={sse!r}; SSE_resolution_threshold={sse_resolution_threshold!r}; '
            f'residual_norm_resolution={residual_norm_resolution!r}; '
            'no predictive variance floor or replacement')
    df = n - p
    if df <= 2:
        raise UndefinedStudentPrediction('This bounded pilot requires finite predictive covariance')
    gram = design.T @ design
    factor = np.linalg.cholesky(gram)
    solved = np.linalg.solve(factor, target.T)
    shape = (sse / df) * (np.eye(len(xt)) + solved.T @ solved)
    shape = (shape + shape.T) / 2
    np.linalg.cholesky(shape)
    location = target @ beta
    covariance = shape * df / (df - 2)
    if not np.isfinite(location).all() or not np.isfinite(shape).all() or not np.isfinite(covariance).all():
        raise UndefinedStudentPrediction('STOP_NO_FALLBACK nonfinite predictive law')
    for array in (beta, location, shape, covariance):
        array.setflags(write=False)
    return StudentForecast(kind, beta, sse, n, p, df, residual_norm_resolution,
                           sse_resolution_threshold, location, shape, covariance)


def marginal_crps(location, scale, df, y):
    """Closed-form t CRPS; `scale` is sqrt(shape diagonal), never t SD.

Jordan/Krueger/Lerch scoringRules Appendix A, also scoringrules docs:
https://scoringrules.readthedocs.io/en/latest/forecast_dists.html
No scoringRules source is copied; formula independently implemented below.
An independent CDF integral is required before actual target scoring.
"""
    m, s, y = _real(location, 'location', 1), _real(scale, 'scale', 1), _real(y, 'y', 1)
    if m.shape != s.shape or y.shape != m.shape or (s <= 0).any():
        raise ValueError('Aligned location/positive t scale/target vectors required')
    if isinstance(df, (bool, np.bool_)) or not isinstance(df, (int, float, np.integer, np.floating)):
        raise ValueError('Finite real degrees of freedom greater than one required')
    nu = float(df)
    if not math.isfinite(nu) or nu <= 1:
        raise ValueError('CRPS requires finite df > 1')
    z = (y - m) / s
    cdf = stdtr(nu, z)
    log_pdf = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi) - (nu + 1) / 2 * np.log1p(z * z / nu)
    absolute_mean_term = 2 * np.exp(log_pdf) * (nu + z * z) / (nu - 1)
    constant = 2 * math.sqrt(nu) / (nu - 1) * math.exp(betaln(0.5, nu - 0.5) - 2 * betaln(0.5, nu / 2))
    value = s * (z * (2 * cdf - 1) + absolute_mean_term - constant)
    if not np.isfinite(value).all() or (value < 0).any():
        raise ArithmeticError('Finite nonnegative Student CRPS required; no clipping')
    return value


def joint_nll(location, scale_matrix, df, y):
    """Full multivariate-t negative log density, not independent t marginals."""
    m, s, y = _real(location, 'location', 1), _real(scale_matrix, 'scale_matrix', 2), _real(y, 'y', 1)
    if y.shape != m.shape or s.shape != (len(m), len(m)) or not len(m):
        raise ValueError('Aligned multivariate t mean/shape/target required')
    nu = float(df)
    if not math.isfinite(nu) or nu <= 0:
        raise ValueError('Positive finite df required')
    if not np.allclose(s, s.T, atol=1e-12, rtol=1e-12):
        raise ValueError('Symmetric scale matrix required')
    lower = np.linalg.cholesky(s)
    standardized = np.linalg.solve(lower, y - m)
    quadratic = float(standardized @ standardized)
    d = len(m)
    return float(gammaln(nu / 2) - gammaln((nu + d) / 2) + d / 2 * math.log(nu * math.pi) +
                 np.log(np.diag(lower)).sum() + (nu + d) / 2 * math.log1p(quadratic / nu))


def score_saved_law(location, scale_matrix, df, y):
    m, s, y = _real(location, 'location', 1), _real(scale_matrix, 'scale_matrix', 2), _real(y, 'y', 1)
    if s.shape != (len(m), len(m)) or y.shape != m.shape:
        raise ValueError('Aligned saved Student law required')
    scale = np.sqrt(np.diag(s))
    crps = marginal_crps(m, scale, df, y)
    quantile = float(stdtrit(float(df), 0.95))
    error = y - m
    coverage = np.abs(error) <= quantile * scale
    width = 2 * quantile * scale
    points = [{'truth': float(y[i]), 'location': float(m[i]), 'df': float(df), 't_scale': float(scale[i]),
               't_sd': float(scale[i] * math.sqrt(df / (df - 2))),
               'crps': float(crps[i]), 'squared_error': float(error[i] ** 2),
               'coverage90': bool(coverage[i]), 'width90': float(width[i])}
              for i in range(len(m))]
    return {'crps': float(np.mean(crps)), 'joint_nll_per_dimension': joint_nll(m, s, df, y) / len(y),
            'log_rmse': float(np.sqrt(np.mean(error ** 2))),
            'coverage90': float(np.mean(coverage)), 'width90': float(np.mean(width))}, points
