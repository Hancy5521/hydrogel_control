"""Exact full-row Beta/NIW updates in fixed warm-only YJ-half coordinates.

Classic conjugate models. Native reported G==0 identifies the atom; this does not
infer a physical zero/censoring mechanism. No source I/O, selection or MC catalog.
"""
import math
import numpy as np


def yj(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('finite YJ inputs required')
    result = np.empty_like(value)
    nonnegative = value >= 0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result[nonnegative] = 2 * value[nonnegative] / (np.sqrt(1 + value[nonnegative]) + 1)
        result[~nonnegative] = -np.expm1(1.5 * np.log1p(-value[~nonnegative])) / 1.5
    if not np.isfinite(result).all():
        raise ArithmeticError('YJ outside supported numerical range')
    return result


def inverse_yj(value):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError('finite inverse-YJ inputs required')
    result = np.empty_like(value)
    nonnegative = value >= 0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result[nonnegative] = value[nonnegative] * (1 + value[nonnegative] / 4)
        result[~nonnegative] = -np.expm1((2 / 3) * np.log1p(-1.5 * value[~nonnegative]))
    if not np.isfinite(result).all():
        raise ArithmeticError('inverse YJ outside numerical range')
    return result


def log_jacobian(native, s0, affine_scale):
    value = np.asarray(native, dtype=float) / np.asarray(s0, dtype=float)
    derivative = np.where(value >= 0, -.5 * np.log1p(np.maximum(value, 0)),
                           .5 * np.log1p(np.maximum(-value, 0)))
    result = derivative - np.log(s0) - np.log(affine_scale)
    if not np.isfinite(result).all():
        raise ArithmeticError('nonfinite native-to-z Jacobian')
    return result


def positive_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12)):
        raise ValueError('finite symmetric matrix required')
    np.linalg.cholesky(matrix)
    return matrix


def niw_prior(dimension):
    if dimension not in (2, 3):
        raise ValueError('this protocol supports dimensions2and3')
    return {'dimension': dimension, 'n': 0, 'kappa': 1., 'nu': float(dimension + 4),
            'mean': [0.] * dimension, 'Psi': (3 * np.eye(dimension)).tolist()}


def niw_update(state, value):
    vector = np.asarray(value, dtype=float)
    dimension = state['dimension']
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError('one complete branch row required')
    kappa = state['kappa']
    difference = vector - np.asarray(state['mean'])
    new_mean = np.asarray(state['mean']) + difference / (kappa + 1)
    new_psi = positive_matrix(np.asarray(state['Psi']) + kappa / (kappa + 1) * np.outer(difference, difference))
    return {'dimension': dimension, 'n': state['n'] + 1, 'kappa': kappa + 1,
            'nu': state['nu'] + 1, 'mean': new_mean.tolist(), 'Psi': new_psi.tolist()}


def predictive(state):
    dimension = state['dimension']
    df = state['nu'] - dimension + 1
    if df <= 4:
        raise ArithmeticError('native inverse-quadratic first/second moment safety requires df>4')
    scale = positive_matrix((state['kappa'] + 1) / (state['kappa'] * df) * np.asarray(state['Psi']))
    return np.asarray(state['mean']), scale, float(df)


def multivariate_t_log_density(value, mean, scale, df):
    difference = np.asarray(value) - mean
    factor = np.linalg.cholesky(scale)
    whitened = np.linalg.solve(factor, difference)
    delta = float(whitened @ whitened)
    dimension = len(mean)
    logdet = 2 * float(np.log(factor.diagonal()).sum())
    return (math.lgamma((df + dimension) / 2) - math.lgamma(df / 2)
            - .5 * (dimension * math.log(df * math.pi) + logdet)
            - .5 * (df + dimension) * math.log1p(delta / df))


def transformed_row(transform, native):
    native = np.asarray(native, dtype=float)
    if native.shape != (3,) or not np.isfinite(native).all():
        raise ValueError('three complete finite native values required')
    return (yj(native / np.asarray(transform['native_first_scale_s0']))
            - np.asarray(transform['u_center_warm'])) / np.asarray(transform['u_sample_SD_warm'])


def full_row_predictive_log_density(state, native):
    """Mixed native RN density before updating: only atom or slab branch contributes."""
    native = np.asarray(native, dtype=float)
    z = transformed_row(state['transform'], native)
    zero = bool(native[1] == 0)
    indices = [0, 2] if zero else [0, 1, 2]
    branch = state['zero'] if zero else state['nonzero']
    alpha, beta = state['beta_parameters']
    probability = alpha / (alpha + beta) if zero else beta / (alpha + beta)
    location, scale, df = predictive(branch)
    transform = state['transform']
    jacobian = log_jacobian(native[indices], np.asarray(transform['native_first_scale_s0'])[indices],
                            np.asarray(transform['u_sample_SD_warm'])[indices])
    return float(math.log(probability) + multivariate_t_log_density(z[indices], location, scale, df)
                 + jacobian.sum())


def update_complete(state, native):
    native = np.asarray(native, dtype=float)
    z = transformed_row(state['transform'], native)
    zero = bool(native[1] == 0)
    updated = {'transform': state['transform'], 'zero': state['zero'], 'nonzero': state['nonzero'],
               'pooled': niw_update(state['pooled'], z),
               'beta_parameters': list(state['beta_parameters']), 'complete_rows_seen': state['complete_rows_seen'] + 1}
    if zero:
        updated['zero'] = niw_update(state['zero'], z[[0, 2]])
        updated['beta_parameters'][0] += 1
    else:
        updated['nonzero'] = niw_update(state['nonzero'], z)
        updated['beta_parameters'][1] += 1
    return updated


def fit_warm(values):
    native = np.asarray(values, dtype=float)
    if native.shape != (64, 3) or not np.isfinite(native).all():
        raise ValueError('exactly64complete warm rows required')
    s0 = native.std(axis=0, ddof=1)
    if not np.isfinite(s0).all() or (s0 <= 0).any():
        raise ValueError('all warm raw sample SD must be strictly positive; do not add arbitrary floors')
    latent = yj(native / s0)
    center, second = latent.mean(axis=0), latent.std(axis=0, ddof=1)
    if not np.isfinite(center).all() or not np.isfinite(second).all() or (second <= 0).any():
        raise ValueError('all warm YJ sample SD must be positive')
    transform = {'native_first_scale_s0': s0.tolist(), 'u_center_warm': center.tolist(),
        'u_sample_SD_warm': second.tolist(), 'lambda': .5,
        'proxy_native_center_warm': native[:, :2].mean(axis=0).tolist(),
        'proxy_native_sample_SD_warm': native[:, :2].std(axis=0, ddof=1).tolist(),
        'scope': 'fixed warm-dependent empirical reference; not an independent native pre-data prior'}
    state = {'transform': transform, 'zero': niw_prior(2), 'nonzero': niw_prior(3),
             'pooled': niw_prior(3), 'beta_parameters': [1., 1.], 'complete_rows_seen': 0}
    for row in native:
        state = update_complete(state, row)
    return state


def conditional_parameters(state, indices, observed, target):
    mean, scale, df = predictive(state)
    observed = np.asarray(observed, dtype=float)
    if (len(indices) != len(observed) or not np.isfinite(observed).all()
            or target in indices or len(set(indices)) != len(indices)):
        raise ValueError('disjoint target and finite own-row observed coordinates required')
    block = scale[np.ix_(indices, indices)]
    cross = scale[target, list(indices)]
    difference = observed - mean[list(indices)]
    solved = np.linalg.solve(block, difference)
    delta = float(difference @ solved)
    schur = float(scale[target, target] - cross @ np.linalg.solve(block, cross))
    location = float(mean[target] + cross @ solved)
    variance = (df + delta) / (df + len(indices)) * schur
    if not all(math.isfinite(x) for x in (delta, location, variance)) or delta < 0 or variance <= 0:
        raise ArithmeticError('invalid conditional t parameters')
    return location, math.sqrt(variance), df + len(indices)


def predict_target(state, native_proxies, pooled=False):
    native = np.asarray(native_proxies, dtype=float)
    if native.shape != (2,) or not np.isfinite(native).all():
        raise ValueError('exactly two own-row native proxies, no target, required')
    transform = state['transform']
    standardized = ((yj(native / np.asarray(transform['native_first_scale_s0'])[:2])
                     - np.asarray(transform['u_center_warm'])[:2])
                     / np.asarray(transform['u_sample_SD_warm'])[:2])
    if pooled:
        branch, indices, values, target = state['pooled'], [0, 1], standardized, 2
    elif native[1] == 0:
        branch, indices, values, target = state['zero'], [0], standardized[:1], 1
    else:
        branch, indices, values, target = state['nonzero'], [0, 1], standardized, 2
    location, scale, df = conditional_parameters(branch, indices, values, target)
    u_location = transform['u_center_warm'][2] + transform['u_sample_SD_warm'][2] * location
    u_scale = transform['u_sample_SD_warm'][2] * scale
    return {'coordinate': 'u=YJ_half(native/s0)', 'family': 'student_t',
            'location': float(u_location), 'scale': float(u_scale), 'df': float(df),
            'native_first_scale_s0': transform['native_first_scale_s0'][2], 'lambda': .5,
            'unit': 'source bulk GPa attribution, aggregation qualified'}
