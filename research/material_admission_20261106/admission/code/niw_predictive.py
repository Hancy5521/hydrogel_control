"""Classical complete three-variable NIW reference; no source or target IO."""
import math
import numpy as np

MODELS = ('niw_joint_t', 'gaussian_momentmatched', 'independent_target_marginal')
MASKS = ((), (0,), (1,), (0, 1))
MASK_NAMES = ('none', 'e_form', 'gap_opt', 'both')


def spd(matrix):
    a = np.asarray(matrix, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all()
            or not np.allclose(a, a.T, rtol=0., atol=1e-12)):
        raise ValueError('Finite symmetric matrix required')
    np.linalg.cholesky(a)
    return a


def fit_niw(values):
    """Exactly64 complete fit rows; fit-dependent affine reference, not pure Bayes."""
    x = np.asarray(values, dtype=float)
    if x.shape != (64, 3) or not np.isfinite(x).all():
        raise ValueError('Exactly64x3 finite fit-only source cells required')
    center, affine_scale = x.mean(axis=0), x.std(axis=0, ddof=1)
    if not np.isfinite(center).all() or not np.isfinite(affine_scale).all() or (affine_scale <= 0).any():
        raise ValueError('Fit-only affine SD must be strictly positive')
    z = (x - center) / affine_scale
    zbar = z.mean(axis=0)
    residual = z - zbar
    n, kappa0, nu0 = 64, 1., 7.
    kappa, nu = kappa0 + n, nu0 + n
    location = n * zbar / kappa
    lam = 3 * np.eye(3) + residual.T @ residual + (kappa0 * n / kappa) * np.outer(zbar, zbar)
    spd(lam)
    df = nu - 3 + 1
    scale = ((kappa + 1) / (kappa * df)) * lam
    spd(scale)
    return {'reference_type': 'fit-dependent affine empirical-reference NIW arithmetic',
            'fit_rows': n, 'dimension': 3, 'affine_center': center.tolist(),
            'affine_scale_sample_SD': affine_scale.tolist(),
            'prior': {'m0': [0., 0., 0.], 'kappa0': kappa0, 'nu0': nu0, 'Lambda0': (3 * np.eye(3)).tolist()},
            'posterior': {'location': location.tolist(), 'kappa': kappa, 'nu': nu, 'Lambda': lam.tolist()},
            'predictive': {'df': df, 'location': location.tolist(), 'scale_matrix_not_covariance': scale.tolist(),
                           'covariance': (df / (df - 2) * scale).tolist()},
            'diagnostic_rows_used_in_fit': 0, 'extra_parameters_optimized': 0}


def conditional_t(location, scale_matrix, df, observed_indices, observed_values, target_indices):
    """Multivariate-t conditional; returns location, scale matrix and updateddf."""
    m, s = np.asarray(location, dtype=float), spd(scale_matrix)
    o, h = tuple(observed_indices), tuple(target_indices)
    z = np.asarray(observed_values, dtype=float)
    if (m.ndim != 1 or s.shape != (len(m), len(m)) or not np.isfinite(m).all()
            or not math.isfinite(df) or df <= 0 or not h
            or len(set(o + h)) != len(o + h) or any(type(i) is not int or i < 0 or i >= len(m) for i in o + h)
            or z.shape != (len(o),) or not np.isfinite(z).all()):
        raise ValueError('Disjoint valid indices, finite t parameters and observations required')
    hh = np.ix_(h, h)
    if not o:
        return m[list(h)].copy(), s[hh].copy(), float(df)
    oo, ho, oh = np.ix_(o, o), np.ix_(h, o), np.ix_(o, h)
    difference = z - m[list(o)]
    solved = np.linalg.solve(s[oo], difference)
    delta = float(difference @ solved)
    if not math.isfinite(delta) or delta < 0:
        raise ArithmeticError('Invalid t Mahalanobis distance')
    conditional_location = m[list(h)] + s[ho] @ solved
    schur = s[hh] - s[ho] @ np.linalg.solve(s[oo], s[oh])
    conditional_scale = ((df + delta) / (df + len(o))) * schur
    spd(conditional_scale)
    return conditional_location, conditional_scale, float(df + len(o))


def conditional_gaussian(location, covariance, observed_indices, observed_values, target_indices):
    m, s = np.asarray(location, dtype=float), spd(covariance)
    o, h = tuple(observed_indices), tuple(target_indices)
    z = np.asarray(observed_values, dtype=float)
    if (m.ndim != 1 or s.shape != (len(m), len(m)) or not np.isfinite(m).all()
            or not h or len(set(o + h)) != len(o + h)
            or any(type(i) is not int or i < 0 or i >= len(m) for i in o + h)
            or z.shape != (len(o),) or not np.isfinite(z).all()):
        raise ValueError('Disjoint Gaussian indices and finite observations required')
    if not o:
        return m[list(h)].copy(), s[np.ix_(h, h)].copy()
    oo, ho, oh, hh = np.ix_(o, o), np.ix_(h, o), np.ix_(o, h), np.ix_(h, h)
    result_location = m[list(h)] + s[ho] @ np.linalg.solve(s[oo], z - m[list(o)])
    result_covariance = s[hh] - s[ho] @ np.linalg.solve(s[oo], s[oh])
    spd(result_covariance)
    return result_location, result_covariance


def predict_law(fitted, model, observed_indices, observed_values):
    o = tuple(observed_indices)
    values = np.asarray(observed_values, dtype=float)
    if model not in MODELS or o not in MASKS or values.shape != (len(o),) or not np.isfinite(values).all():
        raise ValueError('One of four declared native-proxy masks and three model roles required')
    center = np.asarray(fitted['affine_center'])
    affine = np.asarray(fitted['affine_scale_sample_SD'])
    p = fitted['predictive']
    effective = () if model == 'independent_target_marginal' else o
    standardized = (values - center[list(o)]) / affine[list(o)]
    if model == 'gaussian_momentmatched':
        loc, variance = conditional_gaussian(p['location'], p['covariance'], effective, standardized, (2,))
        df = None
    else:
        loc, variance, df = conditional_t(p['location'], p['scale_matrix_not_covariance'], p['df'],
                    effective, [] if not effective else standardized, (2,))
    native_location = float(center[2] + affine[2] * loc[0])
    native_scale = float(affine[2] * math.sqrt(float(variance[0, 0])))
    if not math.isfinite(native_location) or not math.isfinite(native_scale) or native_scale <= 0:
        raise ArithmeticError('Native prediction is outside numeric range')
    return {'family': 'normal' if df is None else 'student_t', 'location': native_location,
            'scale': native_scale, 'df': df, 'unit': 'source bulk field GPa attribution',
            'scale_is_SD_only_if_normal': True}
