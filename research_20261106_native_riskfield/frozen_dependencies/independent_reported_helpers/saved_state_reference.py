"""Independent batch-stat and scalar-transform reference for saved reported-feedback states."""
import math
import numpy as np
from scipy.stats import multivariate_t


def forward_yj(v):
    return 2 * v / (math.sqrt(1 + v) + 1) if v >= 0 else -math.expm1(1.5 * math.log1p(-v)) / 1.5


def warm_transform(rows):
    n = len(rows)
    mean = [math.fsum(row[j] for row in rows) / n for j in range(3)]
    s0 = [math.sqrt(math.fsum((r[j] - mean[j]) ** 2 for r in rows) / (n - 1)) for j in range(3)]
    latent = [[forward_yj(r[j] / s0[j]) for j in range(3)] for r in rows]
    center = [math.fsum(r[j] for r in latent) / n for j in range(3)]
    second = [math.sqrt(math.fsum((r[j] - center[j]) ** 2 for r in latent) / (n - 1)) for j in range(3)]
    return {'native_first_scale_s0': s0, 'u_center_warm': center, 'u_sample_SD_warm': second,
            'proxy_native_center_warm': mean[:2], 'proxy_native_sample_SD_warm': s0[:2]}


def transform_row(t, values):
    return [(forward_yj(v / t['native_first_scale_s0'][j]) - t['u_center_warm'][j]) / t['u_sample_SD_warm'][j]
            for j, v in enumerate(values)]


def niw_batch(rows, d):
    n = len(rows)
    center = [math.fsum(r[j] for r in rows) / n if n else 0. for j in range(d)]
    scatter = [[math.fsum((r[i] - center[i]) * (r[j] - center[j]) for r in rows) for j in range(d)] for i in range(d)]
    psi = [[(3. if i == j else 0.) + scatter[i][j] + n / (n + 1) * center[i] * center[j]
            for j in range(d)] for i in range(d)]
    return {'dimension': d, 'n': n, 'kappa': float(1 + n), 'nu': float(d + 4 + n),
            'mean': [n * x / (n + 1) for x in center], 'Psi': psi}


def joint_state(t, native_rows):
    z = [transform_row(t, r) for r in native_rows]
    zero = [zz[::2] for zz, r in zip(z, native_rows) if r[1] == 0]
    nonzero = [zz for zz, r in zip(z, native_rows) if r[1] != 0]
    return {'transform': t, 'zero': niw_batch(zero, 2), 'nonzero': niw_batch(nonzero, 3),
            'pooled': niw_batch(z, 3), 'beta_parameters': [1. + len(zero), 1. + len(nonzero)],
            'complete_rows_seen': len(native_rows)}


def predictive(branch):
    k, d, nu = branch['kappa'], branch['dimension'], branch['nu']
    df = nu - d + 1
    return np.array(branch['mean']), (k + 1) / (k * df) * np.array(branch['Psi']), df


def target_law(state, proxies, pooled=False):
    t = state['transform']
    x = transform_row(t, proxies)
    if pooled:
        b, observed = state['pooled'], x
    elif proxies[1] == 0:
        b, observed = state['zero'], x[:1]
    else:
        b, observed = state['nonzero'], x
    m, C, df = predictive(b)
    k = len(observed)
    delta = np.array(observed) - m[:k]
    q = float(delta @ np.linalg.solve(C[:k, :k], delta))
    cross = C[k, :k]
    mu = float(m[k] + cross @ np.linalg.solve(C[:k, :k], delta))
    variance = float((df + q) / (df + k) * (C[k, k] - cross @ np.linalg.solve(C[:k, :k], cross)))
    return {'location': t['u_center_warm'][2] + t['u_sample_SD_warm'][2] * mu,
            'scale': t['u_sample_SD_warm'][2] * math.sqrt(variance), 'df': df + k,
            'native_first_scale_s0': t['native_first_scale_s0'][2]}


def full_row_log_RN(state, native):
    t = state['transform']
    z = transform_row(t, native)
    zero = native[1] == 0
    indices = [0, 2] if zero else [0, 1, 2]
    b = state['zero'] if zero else state['nonzero']
    m, C, df = predictive(b)
    alpha, beta = state['beta_parameters']
    probability = (alpha if zero else beta) / (alpha + beta)
    jac = []
    for j in indices:
        v = native[j] / t['native_first_scale_s0'][j]
        logderiv = -.5 * math.log1p(v) if v >= 0 else .5 * math.log1p(-v)
        jac.append(logderiv - math.log(t['native_first_scale_s0'][j]) - math.log(t['u_sample_SD_warm'][j]))
    return math.log(probability) + float(multivariate_t.logpdf([z[j] for j in indices], loc=m, shape=C, df=df)) + math.fsum(jac)
