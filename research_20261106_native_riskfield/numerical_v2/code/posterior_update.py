"""Exact scalar Normal--InverseGamma conditional regression; no data or source I/O.

Classic conjugate mechanism, not a new algorithm. Modeled responses are in fixed
YJ/affine coordinates chosen by the caller. Only actually observed labels may be
persisted by an experimental controller; hypothetical integration draws are not labels.
"""
import math
import numpy as np


def checked(state):
    mean = np.asarray(state['mean'], dtype=float)
    covariance = np.asarray(state['V'], dtype=float)
    if (mean.ndim != 1 or covariance.shape != (len(mean), len(mean))
            or not np.isfinite(mean).all() or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12)
            or not math.isfinite(state['alpha']) or state['alpha'] <= 1
            or not math.isfinite(state['b']) or state['b'] <= 0):
        raise ValueError('finite SPD NIG state with alpha>1 and b>0 required')
    np.linalg.cholesky(covariance)
    return mean, covariance


def from_niw(niw):
    """Induced exact conditional-regression NIG, target last, all other X known."""
    mu = np.asarray(niw['mean'], dtype=float)
    psi = np.asarray(niw['Psi'], dtype=float)
    p = len(mu) - 1
    if p not in (1, 2) or psi.shape != (p + 1, p + 1):
        raise ValueError('two or three variable NIW state required')
    np.linalg.cholesky(psi)
    beta = np.linalg.solve(psi[:p, :p], psi[:p, p])
    residual = float(psi[p, p] - psi[p, :p] @ beta)
    precision = np.empty_like(psi)
    precision[0, 0] = niw['kappa']
    precision[0, 1:] = niw['kappa'] * mu[:p]
    precision[1:, 0] = precision[0, 1:]
    precision[1:, 1:] = psi[:p, :p] + niw['kappa'] * np.outer(mu[:p], mu[:p])
    result = {'mean': [float(mu[p] - beta @ mu[:p])] + beta.tolist(),
              'V': np.linalg.inv(precision).tolist(), 'alpha': float(niw['nu'] / 2),
              'b': residual / 2, 'labeled_updates': 0}
    checked(result)
    return result


def design(context, dimension):
    context = np.asarray(context, dtype=float)
    if context.shape != (dimension - 1,) or not np.isfinite(context).all():
        raise ValueError('complete finite known context required')
    return np.concatenate(([1.], context))


def update_observed_label(state, context, modeled_label):
    mean, covariance = checked(state)
    phi = design(context, len(mean))
    if type(modeled_label) not in (int, float) or not math.isfinite(modeled_label):
        raise ValueError('one finite actual label in the modeled response coordinate required')
    projection = covariance @ phi
    denominator = 1 + float(phi @ projection)
    residual = modeled_label - float(phi @ mean)
    new_v = covariance - np.outer(projection, projection) / denominator
    result = {'mean': (mean + projection * residual / denominator).tolist(), 'V': new_v.tolist(),
        'alpha': state['alpha'] + .5, 'b': state['b'] + .5 * residual ** 2 / denominator,
        'labeled_updates': state.get('labeled_updates', 0) + 1}
    checked(result)
    return result


def scalar_law(state, context):
    mean, covariance = checked(state)
    phi = design(context, len(mean))
    return {'mu': float(phi @ mean),
        'scale': math.sqrt(state['b'] / state['alpha'] * (1 + float(phi @ covariance @ phi))),
        'nu': 2 * state['alpha']}


def pair_law(state, query_context, target_context):
    """Different physical rows: independent residuals, shared regression posterior."""
    mean, covariance = checked(state)
    query, target = design(query_context, len(mean)), design(target_context, len(mean))
    qdiag = 1 + float(query @ covariance @ query)
    tdiag = 1 + float(target @ covariance @ target)
    rho = float(query @ covariance @ target) / math.sqrt(qdiag * tdiag)
    if not -1 < rho < 1:
        raise ArithmeticError('nondegenerate different-row predictive correlation required')
    return {'mu_query': float(query @ mean), 'mu_target': float(target @ mean),
        'scale_query': math.sqrt(state['b'] / state['alpha'] * qdiag),
        'scale_target': math.sqrt(state['b'] / state['alpha'] * tdiag),
        'nu': 2 * state['alpha'], 'rho': rho,
        'distinct_physical_rows_independent_residuals': True}


def conditional_target(pair, standardized_query):
    nu, rho = pair['nu'], pair['rho']
    if not math.isfinite(standardized_query):
        raise ValueError('finite standardized hypothetical query value required')
    return {'mu': pair['mu_target'] + rho * pair['scale_target'] * standardized_query,
        'scale': pair['scale_target'] * math.sqrt((nu + standardized_query ** 2) / (nu + 1) * (1 - rho ** 2)),
        'nu': nu + 1}
