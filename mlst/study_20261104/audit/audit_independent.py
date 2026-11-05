"""Independent full-matrix audit. Never imports the producer or performs fitting.

Synthetic mode opens no material inputs. Frozen mode requires the explicit
global-freeze SHA supplied by the root after all 750 forecasts were frozen.
Outputs refuse overwrite; events contain factual UTC actions only.
"""
import os
for _key in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import erf
from scipy.stats import norm


EXPECTED_PROTOCOL = '21c6a86096623e0a1ffa03522d16becd1e2a66b6cce373de048a5ded2348f908'
EXPECTED_FREEZE = 'bab8fdc8e87c0d13598e446badf35bac0f0706667c7c118f1a6453fbd5066c1a'
EXPECTED_TRAIN = 'cbd09c37e80548937a3b012e86e4173b3dc663cbe8e61f5ca7baa18c9002d974'
P_ROWS = [0, 1, 3, 4, 5, 6, 11, 15, 16, 20, 21, 24, 26, 27, 28,
          29, 30, 31, 33, 35, 36, 37, 40, 41, 44]
S_ROWS = [i for i in range(45) if i not in P_ROWS]
MASKS = {'spread': [0, 7, 15, 23, 31, 38],
         'middle_gap': [0, 4, 8, 30, 34, 38],
         'one_sided': [0, 3, 6, 9, 12, 15]}
METHODS = [f'random_{s}' for s in range(20261105, 20261113)] + ['prior_kl', 'curvature45']
BOUNDS = np.array([[-1., 1.], [-1., 1.], [-.75, .75], [-1., 1.], [-.75, .75]])
R45 = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def event(stage, action, status, detail):
    path = stage / 'audit/AUDIT_EVENTS.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': utc(), 'action': action, 'status': status,
                            'detail': detail}, ensure_ascii=False) + '\n')


class Checks:
    def __init__(self):
        self.count = 0
        self.max_error = {}

    def true(self, condition, name):
        self.count += 1
        if not condition:
            raise AssertionError(name)

    def close(self, actual, expected, name, atol=2e-8, rtol=2e-9):
        a, b = np.asarray(actual), np.asarray(expected)
        self.true(a.shape == b.shape, name + ': shape')
        self.true(np.isfinite(a).all() and np.isfinite(b).all(), name + ': finite')
        err = float(np.max(np.abs(a - b), initial=0))
        self.max_error[name] = max(self.max_error.get(name, 0.), err)
        self.true(np.allclose(a, b, atol=atol, rtol=rtol), name + ': values')


def law(theta, mean, covariance, cross):
    theta = np.asarray(theta)
    chol = np.array([[np.exp(theta[2]), 0.], [theta[3], np.exp(theta[4])]])
    s = chol @ chol.T
    m = mean + cross @ theta[:2]
    k = covariance + cross @ (s - np.eye(2)) @ cross.T
    return m, (k + k.T) / 2


def conditional(mean, covariance, observed, values, hidden):
    oo = covariance[np.ix_(observed, observed)]
    ho = covariance[np.ix_(hidden, observed)]
    m = mean[hidden] + ho @ np.linalg.solve(oo, values - mean[observed])
    k = covariance[np.ix_(hidden, hidden)] - ho @ np.linalg.solve(oo, ho.T)
    return m, (k + k.T) / 2


def nll(mean, covariance, values):
    sign, determinant = np.linalg.slogdet(covariance)
    if sign != 1:
        raise AssertionError('Nonpositive covariance determinant')
    error = values - mean
    return .5 * (len(values) * np.log(2 * np.pi) + determinant
                 + error @ np.linalg.solve(covariance, error))


def full_nll_gradient(theta, mean, covariance, cross, values):
    """Differentiate dense 39x39 Gaussian algebra, not producer's 2x2 ratio."""
    mu, k = law(theta, mean, covariance, cross)
    precision = np.linalg.solve(k, np.eye(len(k)))
    alpha = precision @ (values - mu)
    weight = precision - np.outer(alpha, alpha)
    l00, l10, l11 = np.exp(theta[2]), theta[3], np.exp(theta[4])
    ds = [np.array([[2*l00*l00, l00*l10], [l00*l10, 0.]]),
          np.array([[0., l00], [l00, 2*l10]]),
          np.array([[0., 0.], [0., 2*l11*l11]])]
    g = [-cross[:, 0] @ alpha, -cross[:, 1] @ alpha]
    for delta in ds:
        g.append(.5 * np.sum(weight * (cross @ delta @ cross.T)))
    return nll(mu, k, values), np.array(g)


def objective(theta, tasks):
    values, gradients, reference_values = [], [], []
    for m, k, c, y in tasks:
        full, gfull = full_nll_gradient(theta, m, k, c, y)
        context, gc, zero_context = [], [], []
        for indexes in MASKS.values():
            o = np.array(indexes)
            vv, gg = full_nll_gradient(theta, m[o], k[np.ix_(o, o)], c[o], y[o])
            context.append(vv)
            gc.append(gg)
            zero_context.append(nll(m[o], k[np.ix_(o, o)], y[o]))
        values.append(full - np.mean(context))
        gradients.append(gfull - np.mean(gc, axis=0))
        reference_values.append(nll(m, k, y) - np.mean(zero_context))
    raw = float(np.mean(values) + .02 * np.dot(theta, theta))
    constant = float(np.mean(reference_values))
    return raw, raw - constant, np.mean(gradients, axis=0) + .04*theta, constant


def dense_gp(info, t, q):
    n = info['normalization']
    h = np.asarray(info['hyper_values'])
    beta = np.asarray(info['beta_standardized'])
    u, v = (t-n['t_center'])/n['t_scale'], (q-n['t_center'])/n['t_scale']
    def kernel(a, b):
        d = np.sqrt(3.) * np.abs(a[:, None]-b[None, :])/h[3]
        return n['y_scale']**2 * (h[0]**2 + h[1]**2*np.outer(a, b)
                                 + h[2]**2*(1+d)*np.exp(-d))
    mean = n['y_center'] + n['y_scale']*(beta[0]+beta[1]*u)
    yy = kernel(u, u) + np.eye(len(u))*n['y_scale']**2*h[4]**2
    return mean, yy, kernel(u, v), kernel(v, v)


def curvature_basis(k, q):
    # Independent expression: each curvature is 2(right slope-left slope)/(a+b).
    gaps = np.diff(q)
    second = np.zeros((len(q)-2, len(q)))
    for j, (a, b) in enumerate(zip(gaps[:-1], gaps[1:])):
        second[j, j] += 2./((a+b)*a)
        second[j, j+1] -= 2./(a+b)*(1./a+1./b)
        second[j, j+2] += 2./((a+b)*b)
    span = q[-1]-q[0]
    result = []
    for f in (1./3, 2./3):
        centered = (q[1:-1]-(q[0]+span*f))/(span/3.)
        weights = np.maximum(1.-centered**2, 0.)**3 * (gaps[:-1]+gaps[1:])/2.
        result.append((weights/weights.sum()) @ second)
    f = np.array(result)
    e, v = np.linalg.eigh(f @ k @ f.T)
    return np.linalg.cholesky(k).T @ f.T @ ((v / np.sqrt(e)) @ v.T)


def basis(method, k, q):
    if method.startswith('random_'):
        return np.linalg.qr(np.random.default_rng(int(method[7:])).normal(size=(17, 2)))[0]
    if method == 'prior_kl':
        e, v = np.linalg.eigh(k)
        e, v = e[-2:][::-1], v[:, -2:][:, ::-1].copy()
        for j in range(2):
            if v[np.argmax(np.abs(v[:, j])), j] < 0:
                v[:, j] = -v[:, j]
        return np.linalg.cholesky(k).T @ (v / np.sqrt(e))
    return curvature_basis(k, q) @ R45


def scores(mean, covariance, truth):
    sd = np.sqrt(np.diag(covariance))
    z = (truth-mean)/sd
    crps = sd*(z*erf(z/np.sqrt(2.)) + np.sqrt(2./np.pi)*np.exp(-z*z/2.)-1./np.sqrt(np.pi))
    lower, upper = mean-norm.ppf(.95)*sd, mean+norm.ppf(.95)*sd
    mse = (mean-truth)**2
    coverage = ((truth >= lower) & (truth <= upper)).astype(int)
    width = upper-lower
    interval = width + 20.*np.maximum(lower-truth, 0.) + 20.*np.maximum(truth-upper, 0.)
    marginal_nll = np.log(sd*np.sqrt(2.*np.pi)) + .5*z*z
    scalar = {'crps': float(crps.mean()), 'mean_mse': float(mse.mean()),
              'joint_log_density': -nll(mean, covariance, truth),
              'coverage90': float(coverage.mean()), 'width90': float(width.mean())}
    return scalar, [crps, mse, coverage, width, interval, marginal_nll], lower, upper


def aggregate(rows):
    return {'CRPS': float(np.mean([r['crps'] for r in rows])),
            'RMSE': float(np.sqrt(np.mean([r['mean_mse'] for r in rows]))),
            'mean_joint_log_density': float(np.mean([r['joint_log_density'] for r in rows])),
            'coverage90': float(np.mean([r['coverage90'] for r in rows])),
            'width90': float(np.mean([r['width90'] for r in rows]))}


def synthetic(stage):
    c = Checks()
    rng = np.random.default_rng(6192026)
    snapshots = {}
    for n in (7, 17, 39):
        a = rng.normal(size=(n, n))
        residual = a @ a.T + np.eye(n)
        cross = rng.normal(size=(n, 2))*.2
        k = residual+cross @ cross.T
        m, y = rng.normal(size=(2, n))
        theta = np.array([.12, -.21, .13, .22, -.16])
        mu, cv = law(theta, m, k, cross)
        o = np.array([0, 2, 4])
        h = np.array([j for j in range(n) if j not in o])
        mm, kk = conditional(mu, cv, o, y[o], h)
        direct = nll(mm, kk, y[h])
        difference = nll(mu, cv, y)-nll(mu[o], cv[np.ix_(o, o)], y[o])
        c.close(direct, difference, 'synthetic_conditional_nll', atol=1e-10)
        inv = np.linalg.inv(cv)
        cov_precision = np.linalg.inv(inv[np.ix_(h, h)])
        mean_precision = mu[h]-cov_precision @ inv[np.ix_(h, o)] @ (y[o]-mu[o])
        c.close(kk, cov_precision, 'synthetic_precision_condition_cov', atol=1e-10)
        c.close(mm, mean_precision, 'synthetic_precision_condition_mean', atol=1e-10)
        _, g = full_nll_gradient(theta, m, k, cross, y)
        fd = []
        for j in range(5):
            step = np.eye(5)[j]*1e-5
            mp, kp = law(theta+step, m, k, cross)
            mn, kn = law(theta-step, m, k, cross)
            fd.append((nll(mp, kp, y)-nll(mn, kn, y))/(2e-5))
        c.close(g, fd, 'synthetic_full_nll_gradient', atol=1e-7)
        chol = np.array([[np.exp(theta[2]), 0.], [theta[3], np.exp(theta[4])]])
        s = R45.T @ (chol @ chol.T) @ R45
        l = np.linalg.cholesky(s)
        rotated_theta = np.r_[R45.T @ theta[:2], np.log(l[0, 0]), l[1, 0], np.log(l[1, 1])]
        mr, kr = law(rotated_theta, m, k, cross @ R45)
        c.close(mr, mu, 'synthetic_unrestricted_rotation_mean', atol=1e-12)
        c.close(kr, cv, 'synthetic_unrestricted_rotation_covariance', atol=1e-12)
        c.true(abs(np.dot(theta, theta)-np.dot(rotated_theta, rotated_theta)) > 1e-5,
               'ridge_not_rotation_invariant')
        snapshots[f'n{n}_mean'] = m
        snapshots[f'n{n}_covariance'] = k
        snapshots[f'n{n}_cross'] = cross
        snapshots[f'n{n}_truth'] = y
        snapshots[f'n{n}_theta'] = theta
        snapshots[f'n{n}_expected_gradient'] = g
    q = np.linspace(-4.3, 6.6, 17)
    distance = np.abs(q[:, None]-q[None, :])
    kq = (1.+np.sqrt(3.)*distance)*np.exp(-np.sqrt(3.)*distance) + .1*np.eye(17)
    for method in METHODS:
        u = basis(method, kq, q)
        c.close(u.T @ u, np.eye(2), 'synthetic_basis_orthonormal', atol=1e-11)
    uc, ur = curvature_basis(kq, q), basis('curvature45', kq, q)
    c.close(uc @ uc.T, ur @ ur.T, 'synthetic_rotation_same_projector', atol=1e-12)
    eig, vec = np.linalg.eigh(kq)
    lk = np.linalg.cholesky(kq) @ basis('prior_kl', kq, q)
    c.close(lk @ lk.T, (vec[:, -2:]*eig[-2:]) @ vec[:, -2:].T,
            'synthetic_KL_top2_covariance', atol=1e-12)
    for y in (-2., 0., 1.4):
        value = scores(np.array([0.]), np.eye(1), np.array([y]))[0]['crps']
        numerical = quad(lambda x: norm.cdf(x)**2, -np.inf, y, epsabs=1e-10)[0]
        numerical += quad(lambda x: norm.sf(x)**2, y, np.inf, epsabs=1e-10)[0]
        c.close(value, numerical, 'synthetic_CRPS_CDF_integral', atol=1e-10)
    path = stage / 'audit/SYNTHETIC_FIXTURES.npz'
    with path.open('xb') as f:
        np.savez(f, **snapshots)
    result = {'actual_utc': utc(), 'status': 'PASS', 'checks': c.count,
              'maximum_absolute_errors': c.max_error, 'source_sha256': sha(__file__),
              'synthetic_fixture_sha256': sha(path), 'material_files_read': 0,
              'new_fits': 0, 'producer_imports': 0,
              'qualification_scope': 'Independent dense mathematical implementation, not certification of unexported producer calculations.'}
    save(stage / 'audit/SYNTHETIC_AUDIT.json', result)
    print(json.dumps(result, indent=2))


def frozen(stage, freeze_sha):
    c = Checks()
    protocol_path = stage/'experiment/PROTOCOL.json'
    freeze_path = stage/'experiment/forecasts/ALL750_FROZEN.json'
    c.true(freeze_sha == EXPECTED_FREEZE == sha(freeze_path), 'authorized_global_freeze')
    c.true(sha(protocol_path) == EXPECTED_PROTOCOL, 'protocol_identity')
    protocol, manifest = read(protocol_path), read(freeze_path)
    c.true(protocol['materials'] == P_ROWS and protocol['methods'] == METHODS and protocol['masks'] == MASKS, 'fixed_census')
    c.true(manifest['protocol_sha256'] == sha(protocol_path) and manifest['fits'] == 250
           and manifest['target_metrics_calls'] == 0 and manifest['S_hidden_targets_read'] == 0, 'freeze_receipt')
    c.true(not (stage/'experiment/forecasts/STOP.json').exists(), 'no_fit_stop')
    for rec in protocol['source_bindings']:
        c.true(sha(stage/'code'/rec['file']) == rec['sha256'], 'source_binding')
    for rec in protocol['input_bindings']:
        c.true(sha(stage/'experiment/inputs'/rec['file']) == rec['sha256'], 'input_binding')
    records = manifest['forecasts']
    c.true(len(records) == 750 and len({r['file'] for r in records}) == 750, '750_distinct_files')
    c.true({(r['method'], r['P'], r['mask']) for r in records} ==
           {(m, p, k) for m in METHODS for p in P_ROWS for k in MASKS}, '750_census')
    for rec in records:
        c.true(sha(stage/rec['file']) == rec['sha256'], 'forecast_hash')
    # Only reached after root authorization and all750 forecasts verified.
    train_path = stage/'experiment/inputs/data/TRAIN.npz'
    c.true(sha(train_path) == EXPECTED_TRAIN, 'exact_closed_train')
    with np.load(train_path, allow_pickle=False) as z:
        y, gamma = z['y'], z['gamma']
        roles, ids = z['roles'].tolist(), z['group_ids'].tolist()
    hidden_s = [i for i in range(39) if i not in MASKS['spread']]
    c.true(y.shape == gamma.shape == (45, 39), 'input_shape')
    c.true([i for i, r in enumerate(roles) if r == 'P'] == P_ROWS, 'P_roles')
    c.true([i for i, r in enumerate(roles) if r == 'S'] == S_ROWS, 'S_roles')
    c.true(np.isnan(y[np.ix_(S_ROWS, hidden_s)]).all(), 'S_hidden_unavailable')
    c.true(np.isfinite(y[P_ROWS]).all() and len(set(ids)) == 45, 'measured_P_and_unique_groups')
    t, q = np.log(gamma), np.array(protocol['fixed_support'])
    c.close(q, np.linspace(t.min(), t.max(), 17), 'support_identity', atol=1e-12)
    score_path = stage/'experiment/scores/RESULT.json'
    score = read(score_path)
    c.true(score['all750_frozen_sha256'] == freeze_sha and score['protocol_sha256'] == sha(protocol_path), 'score_freeze_binding')
    c.true(score['independent_confirmation'] is False and score['new_materials'] == 0,
           'honest_material_classification')
    score_rows = {(r['model'], r['P'], r['mask']): r for r in score['new_rows']}
    c.true(len(score_rows) == 750, 'score_row_census')
    coords_path = stage/'experiment/scores/PER_COORDINATE.csv'
    c.true(sha(coords_path) == score['coordinate_metrics_sha256'], 'coordinate_score_binding')
    with coords_path.open(encoding='utf-8', newline='') as f:
        coordinates = list(csv.DictReader(f))
    c.true(len(coordinates) == 24750, 'coordinate_census')
    coord_rows = {(r['method'], int(r['P']), r['mask'], int(r['target_index'])): r for r in coordinates}
    c.true(len(coord_rows) == 24750, 'unique_coordinate_census')
    references, fits, newrows = {}, {}, []
    for held in P_ROWS:
        info = read(stage/f'experiment/inputs/inputs/baseline/P_{held}/REFERENCE.json')
        others = [p for p in P_ROWS if p != held]
        c.true(info['held_P'] == held and info['admission']['training_rows'] == others+S_ROWS
               and info['admission']['held_row_entirely_excluded'] is True, 'reference_training_exclusion')
        visible_t = np.r_[t[others].ravel(), t[np.ix_(S_ROWS, MASKS['spread'])].ravel()]
        visible_y = np.r_[y[others].ravel(), y[np.ix_(S_ROWS, MASKS['spread'])].ravel()]
        norm_expected = [(visible_t.min()+visible_t.max())/2., np.ptp(visible_t)/2., visible_y.mean(), visible_y.std()]
        c.close(list(info['normalization'].values()), norm_expected, 'normalization_training_only', atol=1e-11)
        references[held] = info
    for rec in records:
        method, held, mask = rec['method'], rec['P'], rec['mask']
        fitkey = (method, held)
        with np.load(stage/rec['file'], allow_pickle=False) as z:
            arrays = {key: z[key] for key in z.files}
        c.true(str(arrays['method']) == method and int(arrays['held_P']) == held, 'forecast_identity')
        observed, hidden = np.array(MASKS[mask]), np.array([i for i in range(39) if i not in MASKS[mask]])
        c.close(arrays['observed'], observed, 'observed_identity', atol=0, rtol=0)
        c.close(arrays['targets'], hidden, 'target_identity', atol=0, rtol=0)
        c.close(arrays['observed_values'], y[held, observed], 'anchor_identity', atol=0, rtol=0)
        c.close(arrays['t'], t[held], 'coordinate_identity', atol=0, rtol=0)
        c.close(arrays['support'], q, 'support_file_identity', atol=0, rtol=0)
        rm, rk, kyq, kq = dense_gp(references[held], t[held], q)
        u = basis(method, kq, q)
        cross = kyq @ np.linalg.solve(np.linalg.cholesky(kq).T, u)
        c.close(arrays['basis'], u, 'basis_reconstruction', atol=1e-9)
        c.close(arrays['reference_mean'], rm, 'GP_mean_reconstruction', atol=1e-10)
        c.close(arrays['reference_covariance'], rk, 'GP_cov_reconstruction', atol=1e-10)
        c.close(arrays['support_covariance'], kq, 'GP_support_reconstruction', atol=1e-10)
        c.close(arrays['cross'], cross, 'cross_reconstruction', atol=1e-9)
        c.close(u.T @ u, np.eye(2), 'basis_orthonormal', atol=1e-10)
        c.true(np.linalg.eigvalsh(rk-cross@cross.T).min() > -1e-9*np.linalg.norm(rk), 'full_residual_valid')
        if fitkey not in fits:
            fit_path = (stage/rec['file']).parent/'FIT.json'
            receipt = read(fit_path)
            theta = np.asarray(receipt['theta'])
            c.true(receipt['training_P_rows'] == [p for p in P_ROWS if p != held]
                   and receipt['held_row_entirely_excluded'] is True, 'head_training_exclusion')
            c.true(receipt['optimizer_runs'] == 1 and receipt['initial_theta'] == [0.]*5
                   and receipt['admitted'] and receipt['reference_fits'] == 0, 'optimizer_census')
            c.close(receipt['bounds'], BOUNDS, 'same_bounds', atol=0, rtol=0)
            c.true(np.all(theta >= BOUNDS[:, 0]-1e-12) and np.all(theta <= BOUNDS[:, 1]+1e-12), 'fit_within_bounds')
            tasks = []
            for other in P_ROWS:
                if other == held:
                    continue
                tm, tk, tq, _ = dense_gp(references[held], t[other], q)
                tc = tq @ np.linalg.solve(np.linalg.cholesky(kq).T, u)
                tasks.append((tm, tk, tc, y[other]))
            raw, centered, gradient, constant = objective(theta, tasks)
            c.close(receipt['raw_objective'], raw, 'dense_raw_objective', atol=2e-7)
            c.close(receipt['centered_objective'], centered, 'dense_centered_objective', atol=2e-7)
            c.close(receipt['removed_constant'], constant, 'dense_removed_constant', atol=2e-7)
            c.close(receipt['gradient'], gradient, 'dense_analytic_gradient', atol=2e-7)
            pg = gradient.copy()
            pg[((theta <= BOUNDS[:, 0]+1e-9) & (pg > 0)) |
               ((theta >= BOUNDS[:, 1]-1e-9) & (pg < 0))] = 0.
            c.true(np.max(abs(pg)) <= 1e-5+2e-7, 'independent_projected_gradient_gate')
            fits[fitkey] = theta
        theta = fits[fitkey]
        c.close(arrays['theta'], theta, 'fit_theta_identity', atol=0, rtol=0)
        mu, cov = law(theta, rm, rk, cross)
        cm, cc = conditional(mu, cov, observed, y[held, observed], hidden)
        c.close(arrays['mean'], cm, 'conditional_mean', atol=2e-8)
        c.close(arrays['covariance'], cc, 'conditional_covariance', atol=2e-8)
        c.true(np.linalg.eigvalsh(cc).min() > 0, 'conditional_covariance_SPD')
        metrics, per, lower, upper = scores(cm, cc, y[held, hidden])
        row = score_rows[(method, held, mask)]
        for key in metrics:
            c.close(row[key], metrics[key], 'score_'+key, atol=2e-7)
        for key, value in [('mean', cm), ('lower90', lower), ('upper90', upper), ('per_target_crps', per[0])]:
            c.close(row[key], value, 'score_'+key, atol=2e-8)
        for j, address in enumerate(hidden):
            coordinate = coord_rows[(method, held, mask, int(address))]
            for key, values in zip(('crps', 'squared_error', 'covered90', 'width90', 'interval_score90', 'marginal_nll'), per):
                c.close(float(coordinate[key]), values[j], 'coordinate_'+key, atol=2e-7)
        newrows.append({'model': method, 'P': held, 'mask': mask, **metrics})
        if len(newrows) % 75 == 0:
            print('AUDITED_FORECASTS', len(newrows), 'FITS', len(fits), flush=True)
    old = []
    for filename, key in [('inputs/baseline/RESULT.json', 'rows'),
                          ('inputs/FROZEN_CANDIDATE_SCORES.json', 'new_rows'),
                          ('results/RANDOM_GAUSSIAN_RESULT.json', 'new_rows')]:
        old.extend(read(stage/'experiment/inputs'/filename)[key])
    combined = old+newrows
    models = sorted({r['model'] for r in combined})
    aggregates = {}
    for method in models:
        selected = [r for r in combined if r['model'] == method]
        c.true(len(selected) == 75, 'aggregate_75_queries')
        aggregates[method] = aggregate(selected)
        for key, value in aggregates[method].items():
            c.close(score['aggregate'][method][key], value, 'aggregate_'+key, atol=2e-7)
        for mask in MASKS:
            masked = aggregate([r for r in selected if r['mask'] == mask])
            for key, value in masked.items():
                c.close(score['by_mask'][mask][method][key], value, 'mask_aggregate_'+key, atol=2e-7)
    for method in METHODS:
        delta = []
        for held in P_ROWS:
            a = np.mean([r['crps'] for r in old if r['model']=='joint_code_gaussian' and r['P']==held])
            b = np.mean([r['crps'] for r in newrows if r['model']==method and r['P']==held])
            delta.append(a-b)
        comp = score['candidate_comparison'][method]
        c.close(comp['candidate_minus_comparator_macro_crps'], np.mean(delta), 'paired_delta')
        c.true(comp['candidate_lower_crps_materials'] == int(np.sum(np.array(delta)<0)), 'paired_material_count')
    random_values = [aggregates[f'random_{s}']['CRPS'] for s in range(20261105, 20261113)]
    for name, value in [('mean_CRPS', np.mean(random_values)), ('median_CRPS', np.median(random_values)),
                        ('min_CRPS', min(random_values)), ('max_CRPS', max(random_values))]:
        c.close(score['eight_new_random_seeds'][name], value, 'random_summary_'+name)
    c.true(len(fits) == 250 and len(newrows) == 750, 'audited_all_fits_forecasts')
    result = {'actual_utc': utc(), 'status': 'PASS', 'checks': c.count,
              'maximum_absolute_errors': c.max_error, 'fit_receipts_checked': len(fits),
              'frozen_forecasts_checked': len(newrows), 'scored_coordinates_checked': len(coord_rows),
              'source_sha256': sha(__file__), 'protocol_sha256': sha(protocol_path),
              'global_freeze_sha256': sha(freeze_path), 'score_result_sha256': sha(score_path),
              'producer_imports': 0, 'new_fits': 0, 'S_hidden_available': False,
              'root_freeze_authorization': 'Received explicit root message after global750 freeze and before material reads.',
              'aggregate': aggregates, 'classification': 'Retrospective consumed25P; no independent material confirmation.'}
    save(stage/'audit/FROZEN_AUDIT.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['synthetic', 'frozen'])
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--freeze-sha256')
    args = parser.parse_args()
    stage = args.stage.resolve()
    event(stage, args.phase, 'started', {'source_sha256': sha(__file__)})
    try:
        if args.phase == 'synthetic':
            synthetic(stage)
        else:
            frozen(stage, args.freeze_sha256)
    except Exception as exc:
        event(stage, args.phase, 'failed', {'type': type(exc).__name__, 'message': str(exc)})
        raise
    event(stage, args.phase, 'succeeded', {'source_sha256': sha(__file__)})


if __name__ == '__main__':
    main()
