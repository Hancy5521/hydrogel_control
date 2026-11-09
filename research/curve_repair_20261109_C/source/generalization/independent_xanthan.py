"""Independent audit of saved, context-isolated xanthan laws and scores.

No production imports, model fit, optimizer, or production scorer. PRE parses
only raw column-one coordinates and saved own-six anchors, never raw eta. POST
may parse only the already released eight target eta rows per context. The raw
public ZIP is unencrypted; different offline masks may reuse cells in roles.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.integrate import quad
from scipy.linalg import qr, solve_triangular
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp, ndtr, ndtri
from scipy.stats import norm, t

SOURCE_SHA = 'c50696f0776fba1a721e6a2804480b2d2aa900e232f3e93eacbfd35403c904a4'
INPUT_SHA = '0ac7b5f64663d6bf89ad80ed1602ec26df665f036084a3201e6f87ce1edd0c92'
MODEL_SHA = 'd72fc262fbae0e6fecc36568f0638166fe95e991bd5f7df100ab1434097d82d1'
WINDOWS = {'overlap_resin_1_to_50': (1., 50.),
           'full_reference_support': (0.012787740677595104, 754.5143432617188)}
MASKS = {'near_tail': ([0, 1, 13, 14, 15, 16], list(range(17, 25))),
         'prefix_tail': ([0, 1, 2, 3, 4, 5], list(range(17, 25))),
         'interior_hole': ([0, 4, 6, 17, 20, 24], list(range(8, 16)))}
METHODS = ('physical_full_student', 'frozen_gaussian_gp', 'anchor_fitted_gaussian_gp',
           'student_frozen_gp_copula', 'student_anchor_gp_copula',
           'student_anchor_gp_permuted_copula', 'student_independence',
           'half_quantile_anchor_gp', 'half_density_anchor_gp')
PERMUTATION = [0, 2, 4, 6, 1, 3, 5, 7]
GROUPS = {'NaCl_0_XG', 'NaCl_0.1_XG', 'NaCl_0.3_XG', 'NaCl_0.4_XG',
          'NaCl_0.5_XG', 'NaCl_0.7_XG', 'XG_NaCl_0.1', 'XG_NaCl_0.3',
          'XG_NaCl_0.5', 'XG_NaCl_0.7', 'XG_NaCl_0.9'}
ERRORS = {}
QUAD_ERRORS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('Duplicate JSON key: ' + k)
            d[k] = v
        return d
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def close(name, actual, expected, rtol=2e-7, atol=2e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError(name + ': shape/nonfinite')
    error = float(np.max(np.abs(a-b))) if a.size else 0.
    ERRORS[name.split(':')[0]] = max(ERRORS.get(name.split(':')[0], 0.), error)
    if not np.all(np.abs(a-b) <= atol+rtol*np.abs(b)):
        raise AssertionError(name + ': independent numerical disagreement')


def spd(a):
    a = np.asarray(a, dtype=float)
    close('symmetric', a, a.T, rtol=1e-10, atol=1e-12)
    return np.linalg.cholesky(a)


def corr(k):
    spd(k)
    s = np.sqrt(np.diag(k))
    out = k/s[:, None]/s[None, :]
    np.fill_diagonal(out, 1.)
    spd(out)
    return out


def frozen_grid(model, x):
    r = model['reference']; n = r['normalization']
    u = (x-n['t_center'])/n['t_scale']
    a, b, c, ell, noise = r['hyper_values']
    distance = math.sqrt(3.)*np.abs(u[:, None]-u[None, :])/ell
    k = n['y_scale']**2*(a*a+b*b*np.outer(u, u)+c*c*(1+distance)*np.exp(-distance)+noise*noise*np.eye(len(x)))
    beta = np.asarray(r['beta_standardized'])
    m = n['y_center']+n['y_scale']*(beta[0]+beta[1]*u)
    spd(k)
    return m, k


def schur(m, k, o, y, target):
    # Augmented block calculation independently uses least-squares solve.
    koo = k[np.ix_(o, o)]; cross = k[np.ix_(target, o)]
    weight = np.linalg.lstsq(koo, cross.T, rcond=None)[0].T
    cm = m[target]+weight@(y-m[o])
    ck = k[np.ix_(target, target)]-weight@cross.T
    ck = (ck+ck.T)/2
    spd(ck)
    return cm, ck


def physical(xo, yo, xt):
    x = np.column_stack((np.ones(6), xo)); target = np.column_stack((np.ones(8), xt))
    q, r = qr(x, mode='economic')
    beta = solve_triangular(r, q.T@yo)
    residual = yo-x@beta; sse = float(residual@residual)
    resolution = 64*np.finfo(float).eps*(np.linalg.norm(x, 2)*np.linalg.norm(beta)+np.linalg.norm(yo))
    if not sse > resolution**2:
        raise AssertionError('Unresolved physical residual')
    a = solve_triangular(r.T, target.T, lower=True)
    shape = (sse/4)*(np.eye(8)+a.T@a)
    spd(shape)
    return target@beta, shape, beta, sse, residual, math.sqrt(sse/4)


def own_anchor_gp(xo, yo, xt, law):
    ms, shape, beta, sse, residual, scale = physical(xo, yo, xt)
    hyper = law['hyperparameters']
    signal, length, white = [hyper[k] for k in
        ('signal_variance_standardized', 'length_scale_log_gamma', 'white_variance_standardized')]
    assert 1e-4 <= signal <= 1e4 and 1e-2 <= length <= 1e2 and 1e-6 <= white <= 1.
    def kernel(a, b):
        d = math.sqrt(3.)*np.abs(a[:, None]-b[None, :])/length
        return signal*(1+d)*np.exp(-d)
    koo = kernel(xo, xo)+white*np.eye(6)
    kto = kernel(xt, xo)
    cross_solve = np.linalg.solve(koo, kto.T)
    standardized = residual/scale
    alpha = np.linalg.solve(koo, standardized)
    mean = ms+scale*(kto@alpha)
    covariance = scale**2*(kernel(xt, xt)+white*np.eye(8)-kto@cross_solve)
    covariance = (covariance+covariance.T)/2; lower = spd(koo); spd(covariance)
    logml = -.5*(standardized@alpha+6*math.log(2*math.pi))-np.log(np.diag(lower)).sum()
    close('gp_OLS_beta', law['OLS_beta'], beta)
    close('gp_SSE', law['SSE'], sse)
    close('gp_residual_scale', law['residual_scale'], scale)
    close('gp_standardized_O', law['standardized_anchor_residual'], standardized)
    close('gp_theta', law['kernel_theta_log'], np.log([signal, length, white]))
    close('gp_logml', law['log_marginal_likelihood_standardized'], logml, rtol=2e-6)
    close('gp_logml_log_eta', law['log_marginal_likelihood_log_eta'], logml-6*math.log(scale), rtol=2e-6)
    close('gp_white_log_eta', hyper['white_variance_log_eta'], scale**2*white)
    assert law['target_Y_read'] is False and law['same_anchor_budget'] == 6
    assert law['n_restarts_optimizer'] == 0 and law['alpha'] == 0
    assert law['covariance_includes_observational_white_variance'] is True
    assert law['no_fallback_no_jitter_no_variance_floor'] is True
    assert law['hyperparameter_posterior_uncertainty_integrated'] is False
    return mean, covariance


def read_coordinates(source_zip, inputs):
    expected = {g['member']: g for g in inputs['groups']}
    assert {g['preparation_group'] for g in expected.values()} == GROUPS
    assert len(expected) == 11
    with zipfile.ZipFile(source_zip) as z:
        for member, g in expected.items():
            raw = z.read(member)
            assert hashlib.sha256(raw).hexdigest() == g['source_member_sha256']
            stream = io.StringIO(raw.decode('utf-8-sig'), newline='')
            assert stream.readline().rstrip('\r\n') == 'Var1,Var2,Var3,Var4'
            assert stream.readline().rstrip('\r\n') == 'shear rate [1/s],shear stress [Pa],viscosity [Pa s],Torque [μN m]'
            coordinates = []
            for line_no, line in enumerate(stream, 3):
                token, sep, _not_numeric_response = line.partition(',')
                assert sep and token.strip()
                coordinates.append({'source_line_1based': line_no, 'gamma_per_s': float(token.strip()),
                                    'gamma_raw_token': token.strip()})
            assert coordinates == g['coordinates'] and len(coordinates) == 357
            assert all(a['gamma_per_s'] < b['gamma_per_s'] for a, b in zip(coordinates, coordinates[1:]))
            assert g['temperature_C'] == 21 and g['rheology_curve_count'] == 1
            assert g['independent_batch_count'] is None
    return expected


def audit_access(receipt, forecasts, curves, role, count):
    sites = receipt['access_sites']
    expected = []
    for f in forecasts:
        c = curves[(f['preparation_group'], f['window'])]
        indices = f['O'] if role == 'observed_own6' else f['T']
        for i in indices:
            expected.append((f['context'], role, f['member'], c['nodes'][i]['source_line_1based'], 2, 'Pa.s'))
    actual = [(s['context'], s['role'], s['member'], s['line'], s['column'], s['unit']) for s in sites]
    assert Counter(actual) == Counter(expected) and len(sites) == count
    assert receipt['numeric_access_calls'] == count and receipt['stress_or_torque_numeric_accessed'] == 0
    assert receipt['unique_response_cells'] == len({(s[2], s[3], s[4]) for s in actual})


def pre_audit(args):
    reg, prepared, prediction, inputs, model = map(read,
        (args.registration, args.prepare, args.predictions, args.inputs, args.model))
    assert reg['schema'] == 'xanthan-cross-source-fixed-context-v1'
    assert reg['methods'] == list(METHODS) and reg['own_scalar_budget'] == 6 and reg['target_dimension'] == 8
    assert reg['numeric_eta_before_registration'] == 0
    for path, digest in reg['pins'].items():
        assert sha(path) == digest, 'Changed registered dependency'
    assert sha(args.source_zip) == SOURCE_SHA and sha(args.inputs) == INPUT_SHA and sha(args.model) == MODEL_SHA
    assert prepared['registration_sha256'] == prediction['registration_sha256'] == sha(args.registration)
    assert prediction['prepared_sha256'] == sha(args.prepare)
    complete = read(Path(args.predictions).parent/'PREDICTION_COMPLETE.json')
    assert complete['predictions_sha256'] == sha(args.predictions)
    assert prepared['source_numeric_eta_accessed'] == 0 and prepared['new_fits'] == 0
    assert prepared['reference_theta_updates'] == 0 and prepared['methods'] == list(METHODS)
    assert prediction['laws'] == 594 and prediction['new_physical_fits'] == prediction['new_anchor_gp_fits'] == 66
    assert prediction['covariance_fits_from_target_labels'] == 0
    assert prediction['context_target_read_during_own_prediction'] is False
    assert prediction['different_masks_can_reuse_source_cells_in_other_roles'] is True
    source_groups = read_coordinates(args.source_zip, inputs)
    curves = {}
    for c in prepared['curves']:
        key = (c['preparation_group'], c['window']); assert key not in curves
        g = source_groups[c['member']]
        assert c['preparation_group'] == g['preparation_group']
        assert c['salinity_M'] == g['NaCl_molarity_M'] and c['preparation_order'] == g['preparation_order']
        assert c['member_sha256'] == g['source_member_sha256']
        assert c['observation_unit'] == 'Pa.s' and c['temperature_C'] == 21
        low, high = WINDOWS[c['window']]
        eligible = [n for n in g['coordinates'] if low <= n['gamma_per_s'] <= high]
        assert len(eligible) >= 25
        ranks = [(2*j*(len(eligible)-1)+24)//48 for j in range(25)]
        nodes = [eligible[i] for i in ranks]
        assert nodes == c['nodes'] and len(set(ranks)) == 25
        x = np.log([n['gamma_per_s'] for n in nodes]); close('native_x', c['x'], x, 1e-12, 1e-12)
        assert min(x) >= model['q'][0] and max(x) <= model['q'][-1]
        m, k = frozen_grid(model, x)
        close('frozen_grid_mean', c['frozen_mean'], m)
        close('frozen_grid_cov', c['frozen_covariance'], k)
        curves[key] = c
    assert len(curves) == 22 and len({k[0] for k in curves}) == 11
    assert prepared['masks'] == {k: {'O': o, 'T': target} for k, (o, target) in MASKS.items()}
    expected_contexts = {g+'|'+w+'|'+mask for g in GROUPS for w in WINDOWS for mask in MASKS}
    forecasts = prediction['forecasts']; assert len(forecasts) == 66
    assert {f['context'] for f in forecasts} == expected_contexts
    for f in forecasts:
        c = curves[(f['preparation_group'], f['window'])]
        assert f['member'] == c['member'] and f['salinity_M'] == c['salinity_M']
        assert f['preparation_order'] == c['preparation_order']
        o, target = MASKS[f['mask']]
        assert f['O'] == o and f['T'] == target and not set(o)&set(target)
        assert f['context'] == f['preparation_group']+'|'+f['window']+'|'+f['mask']
        x = np.array(c['x']); xo, xt = x[o], x[target]; yo = np.array(f['y_observed'])
        assert yo.shape == (6,) and np.isfinite(yo).all()
        close('observed_x', f['x_observed'], xo); close('target_x', f['x_target'], xt)
        assert f['observations_received'] == 6 and f['target_responses_received'] == 0
        assert f['physical_fits'] == f['anchor_gp_fits'] == 1 and f['permutation'] == PERMUTATION
        laws = f['laws']; assert list(laws) == list(METHODS)
        ms, shape, beta, sse, residual, scale = physical(xo, yo, xt)
        student = laws[METHODS[0]]
        assert student['distribution'] == 'multivariate_student_t' and student['df'] == 4
        assert student['n'] == 6 and student['p'] == 2 and student['scale_is_covariance'] is False
        close('physical_beta', student['beta'], beta); close('physical_SSE', student['sse'], sse)
        close('physical_location', student['location'], ms); close('physical_shape', student['scale_matrix'], shape)
        close('physical_covariance', student['predictive_covariance'], 2*shape)
        fm, fk = frozen_grid(model, x)
        m0, k0 = schur(fm, fk, o, yo, target)
        close('frozen_cond_mean', laws[METHODS[1]]['mean'], m0)
        close('frozen_cond_cov', laws[METHODS[1]]['covariance'], k0)
        mg, kg = own_anchor_gp(xo, yo, xt, laws[METHODS[2]])
        close('anchor_GP_mean', laws[METHODS[2]]['mean'], mg)
        close('anchor_GP_cov', laws[METHODS[2]]['covariance'], kg)
        r0, rg = corr(k0), corr(kg); p = np.array(PERMUTATION); wrong = rg[np.ix_(p, p)]
        ss = np.sqrt(np.diag(shape))
        for method, r in zip(METHODS[3:7], (r0, rg, wrong, np.eye(8))):
            law = laws[method]
            assert law['distribution'] == 'gaussian_copula_student4_marginals' and law['df'] == 4
            close('copula_location', law['location'], ms); close('copula_scale', law['scale'], ss)
            close('copula_R', law['correlation'], r); spd(law['correlation'])
        close('permutation_eigenvalues', np.linalg.eigvalsh(wrong), np.linalg.eigvalsh(rg))
        close('permutation_distance', f['permuted_R_frobenius_distance'], np.linalg.norm(rg-wrong))
        midpoint = laws[METHODS[7]]
        assert midpoint['distribution'] == 'fixed_half_gaussian_student4_quantile_transport'
        for key, expected in [('gaussian_location', mg), ('gaussian_sd', np.sqrt(np.diag(kg))),
                              ('student_location', ms), ('student_scale', ss), ('correlation', rg)]:
            close('midpoint_'+key, midpoint[key], expected)
        mixture = laws[METHODS[8]]
        assert mixture['distribution'] == 'equal_density_mixture'
        assert mixture['components'] == [METHODS[2], METHODS[4]] and mixture['weights'] == [.5, .5]
    audit_access(read(args.anchor_access), forecasts, curves, 'observed_own6', 396)
    return prepared, prediction, curves


def gaussian_logpdf(y, m, k):
    lower = spd(k); z = solve_triangular(lower, y-m, lower=True)
    return -.5*(len(y)*math.log(2*math.pi)+float(z@z))-np.log(np.diag(lower)).sum()


def student_logpdf(y, m, shape):
    lower = spd(shape); z = solve_triangular(lower, y-m, lower=True); d = len(y)
    return float(gammaln((4+d)/2)-gammaln(2)-d/2*math.log(4*math.pi)-
                 np.log(np.diag(lower)).sum()-(4+d)/2*math.log1p(float(z@z)/4))


def t4_normal_score(a):
    a = np.asarray(a); aa = np.abs(a); radius = np.hypot(2., aa)
    # Exact t4 lower-tail identity; stable1-u avoids catastrophic cancellation.
    one_minus_u = 4./radius/(radius+aa)
    lower = one_minus_u**2*(aa/radius+2)/4
    if np.any(lower <= 0):
        raise ArithmeticError('Independent t4 tail not representable')
    negative = ndtri(lower)
    return np.where(a > 0, -negative, negative)


def t4_from_z(z):
    lower = ndtr(-np.abs(z))
    if np.any(lower <= 0):
        raise ArithmeticError('Independent normal tail not representable')
    negative = t.ppf(lower, 4)
    return np.where(np.asarray(z) > 0, -negative, negative)


def copula_logpdf(y, m, s, r):
    standardized = (y-m)/s; z = t4_normal_score(standardized)
    return gaussian_logpdf(z, np.zeros(8), r)-float(np.sum(norm.logpdf(z)))+float(np.sum(t.logpdf(standardized, 4)-np.log(s)))


def integral_crps(y, m, s, family):
    a = (y-m)/s
    cdf = norm.cdf if family == 'normal' else lambda v: t.cdf(v, 4)
    survival = norm.sf if family == 'normal' else lambda v: t.sf(v, 4)
    lo, e1 = quad(lambda v: cdf(v)**2, -np.inf, a, epsabs=2e-10, epsrel=2e-9, limit=150)
    hi, e2 = quad(lambda v: survival(v)**2, a, np.inf, epsabs=2e-10, epsrel=2e-9, limit=150)
    QUAD_ERRORS.append(float(s*(e1+e2)))
    return s*(lo+hi)


def midpoint_details(y, mg, sg, ms, ss, r):
    def transformed(z, j):
        return .5*(mg[j]+sg[j]*z+ms[j]+ss[j]*float(t4_from_z(z)))
    z = np.empty(8); scores = []
    for j in range(8):
        bound = 1.
        while transformed(-bound, j) > y[j] or transformed(bound, j) < y[j]:
            bound *= 2
            if bound > 32:
                raise ArithmeticError('Midpoint inverse exceeds numerical domain')
        z[j] = brentq(lambda v: transformed(v, j)-y[j], -bound, bound, xtol=1e-13, rtol=1e-14)
        split = float(ndtr(z[j]))
        def quantile(p):
            normal = float(ndtri(p)); return transformed(normal, j)
        lo, e1 = quad(lambda p: 2*p*(y[j]-quantile(p)), 0., split, epsabs=2e-10, epsrel=2e-9, limit=150)
        hi, e2 = quad(lambda p: 2*(1-p)*(quantile(p)-y[j]), split, 1., epsabs=2e-10, epsrel=2e-9, limit=150)
        scores.append(lo+hi); QUAD_ERRORS.append(float(e1+e2))
    a = t4_from_z(z)
    derivative = .5*(sg+ss*np.exp(norm.logpdf(z)-t.logpdf(a, 4)))
    logp = gaussian_logpdf(z, np.zeros(8), r)-float(np.log(derivative).sum())
    q = .5*(mg+sg*norm.ppf([.05, .5, .95])[:, None]+ms+ss*t.ppf([.05, .5, .95], 4)[:, None])
    return logp, np.array(scores), q


def density_mixture_details(y, mg, sg, ms, ss, gaussian_logp, student_copula_logp):
    scores = []; q = np.empty((3, 8))
    for j in range(8):
        # CDF CRPS in centered/scaled coordinates; no production score formula.
        center = .5*(mg[j]+ms[j]); unit = max(sg[j], ss[j])
        def cdf(v):
            value = center+unit*v
            return .5*(norm.cdf((value-mg[j])/sg[j])+t.cdf((value-ms[j])/ss[j], 4))
        def sf(v):
            value = center+unit*v
            return .5*(norm.sf((value-mg[j])/sg[j])+t.sf((value-ms[j])/ss[j], 4))
        a = (y[j]-center)/unit
        lo, e1 = quad(lambda v: cdf(v)**2, -np.inf, a, epsabs=2e-10, epsrel=2e-9, limit=150)
        hi, e2 = quad(lambda v: sf(v)**2, a, np.inf, epsabs=2e-10, epsrel=2e-9, limit=150)
        scores.append(unit*(lo+hi)); QUAD_ERRORS.append(float(unit*(e1+e2)))
        for i, p in enumerate((.05, .5, .95)):
            ends = [(mg[j]+sg[j]*norm.ppf(p)-center)/unit,
                    (ms[j]+ss[j]*t.ppf(p, 4)-center)/unit]
            q[i, j] = center+unit*brentq(lambda v: cdf(v)-p, min(ends)-1e-9, max(ends)+1e-9, xtol=1e-13)
    return float(logsumexp([gaussian_logp, student_copula_logp])-math.log(2)), np.array(scores), q


def independent_scores(laws, y):
    student = laws[METHODS[0]]; ms = np.array(student['location']); shape = np.array(student['scale_matrix'])
    ss = np.sqrt(np.diag(shape))
    student_crps = np.array([integral_crps(v, m, s, 'student') for v, m, s in zip(y, ms, ss)])
    student_q = ms+t.ppf([.05, .5, .95], 4)[:, None]*ss
    output = {}; logdensities = {}
    for method in METHODS:
        law = laws[method]; family = law['distribution']
        if family == 'multivariate_student_t':
            mean = ms; logp = student_logpdf(y, ms, shape); crps, q = student_crps, student_q
        elif family == 'multivariate_gaussian':
            mean = np.array(law['mean']); k = np.array(law['covariance']); s = np.sqrt(np.diag(k))
            logp = gaussian_logpdf(y, mean, k)
            crps = np.array([integral_crps(v, m, sd, 'normal') for v, m, sd in zip(y, mean, s)])
            q = mean+norm.ppf([.05, .5, .95])[:, None]*s
        elif family == 'gaussian_copula_student4_marginals':
            mean = ms; logp = copula_logpdf(y, ms, ss, np.array(law['correlation']))
            crps, q = student_crps, student_q
        elif family == 'fixed_half_gaussian_student4_quantile_transport':
            mg, sg = np.array(law['gaussian_location']), np.array(law['gaussian_sd'])
            mean = .5*(mg+ms)
            logp, crps, q = midpoint_details(y, mg, sg, ms, ss, np.array(law['correlation']))
        elif family == 'equal_density_mixture':
            glaw = laws[METHODS[2]]; mg = np.array(glaw['mean']); sg = np.sqrt(np.diag(glaw['covariance']))
            mean = .5*(mg+ms)
            logp, crps, q = density_mixture_details(y, mg, sg, ms, ss, logdensities[METHODS[2]], logdensities[METHODS[4]])
        else:
            raise AssertionError('Unregistered probability family')
        assert np.isfinite(logp) and np.isfinite(crps).all() and np.all(crps >= 0)
        logdensities[method] = logp
        output[method] = {'joint_nll_per_dim': -logp/8, 'crps': float(crps.mean()),
            'pointwise_crps': crps, 'log_rmse': float(np.sqrt(np.mean((y-mean)**2))),
            'coverage90': float(np.mean((y >= q[0]) & (y <= q[2]))), 'width90': float(np.mean(q[2]-q[0]))}
    return output


def post_audit(args, prepared, prediction, curves):
    pre = read(args.pre_audit)
    assert pre['status'] == 'PASS' and pre['phase'] == 'pre' and pre['predictions_sha256'] == sha(args.predictions)
    targets = read(args.targets); scores = read(args.scores)
    assert targets['predictions_sha256'] == scores['predictions_sha256'] == sha(args.predictions)
    assert targets['independent_pre_score_sha256'] == sha(args.pre_audit)
    assert scores['registration_sha256'] == sha(args.registration) and scores['targets_sha256'] == sha(args.targets)
    assert scores['new_laboratory_batch_confirmation'] is False and scores['original_resin_final20_accessed'] is False
    target_rows = {r['context']: r['values'] for r in targets['contexts']}; assert len(target_rows) == 66
    rows = {(r['context'], r['method']): r for r in scores['rows']}; assert len(rows) == len(scores['rows']) == 594
    rebuilt = []
    with zipfile.ZipFile(args.source_zip) as z:
        for f in prediction['forecasts']:
            c = curves[(f['preparation_group'], f['window'])]
            wanted = {c['nodes'][i]['source_line_1based']: i for i in f['T']}
            values = {}; raw = z.read(c['member'])
            assert hashlib.sha256(raw).hexdigest() == c['member_sha256']
            for line_no, line in enumerate(io.StringIO(raw.decode('utf-8-sig')), 1):
                if line_no not in wanted:
                    continue
                columns = line.rstrip('\r\n').split(',')
                index = wanted[line_no]
                assert columns[0] == c['nodes'][index]['gamma_raw_token']
                eta = float(columns[2])
                assert math.isfinite(eta) and eta > 0
                values[index] = math.log(eta)
            assert set(values) == set(f['T'])
            y = np.array([values[i] for i in f['T']])
            close('raw_target_log_eta', target_rows[f['context']], y, rtol=1e-12, atol=1e-12)
            expected = independent_scores(f['laws'], y)
            for method, computed in expected.items():
                saved = rows[(f['context'], method)]
                for k in ('preparation_group', 'salinity_M', 'preparation_order', 'window', 'mask'):
                    assert saved[k] == f[k]
                for k, v in computed.items():
                    close('score_'+k, saved[k], v, rtol=2e-6, atol=2e-8)
                rebuilt.append({**{k: f[k] for k in ('context', 'preparation_group', 'salinity_M', 'preparation_order', 'window', 'mask')},
                                'method': method, **computed})
    metrics = ('joint_nll_per_dim', 'crps', 'log_rmse', 'coverage90', 'width90')
    salinity_rows = {(r['window'], r['mask'], r['method'], r['salinity_M']): r for r in scores['by_salinity']}
    macro_rows = {(r['window'], r['mask'], r['method']): r for r in scores['macro']}
    assert len(salinity_rows) == len(scores['by_salinity']) == 378
    assert len(macro_rows) == len(scores['macro']) == 54
    salinities = sorted({r['salinity_M'] for r in rebuilt}); assert len(salinities) == 7
    for window in WINDOWS:
        for mask in MASKS:
            for method in METHODS:
                sal_values = []
                for salt in salinities:
                    subset = [r for r in rebuilt if (r['window'], r['mask'], r['method'], r['salinity_M']) == (window, mask, method, salt)]
                    saved = salinity_rows[(window, mask, method, salt)]
                    assert saved['preparation_conditions'] == len(subset)
                    mean = {k: float(np.mean([r[k] for r in subset])) for k in metrics}
                    for k, v in mean.items():
                        close('salinity_'+k, saved[k], v, rtol=2e-6, atol=2e-8)
                    sal_values.append(mean)
                saved = macro_rows[(window, mask, method)]; assert saved['salinity_groups'] == 7
                for k in metrics:
                    close('macro_'+k, saved[k], np.mean([r[k] for r in sal_values]), rtol=2e-6, atol=2e-8)
    audit_access(read(args.target_access), prediction['forecasts'], curves, 'score_target_own8', 528)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['pre', 'post'])
    for name in ('authorization', 'registration', 'prepare', 'predictions', 'anchor-access', 'inputs', 'source-zip', 'model', 'output'):
        parser.add_argument('--'+name, required=True)
    for name in ('targets', 'scores', 'target-access', 'pre-audit'):
        parser.add_argument('--'+name)
    args = parser.parse_args(); auth = read(args.authorization)
    assert auth['scope'] == 'xanthan_independent_audit' and auth['allowed_phase'] == args.phase
    assert auth['auditor_source_sha256'] == sha(__file__)
    stamp = dt.datetime.fromisoformat(auth['actual_utc'])
    assert stamp.tzinfo is not None and stamp <= dt.datetime.now(dt.timezone.utc)
    inputs = ('registration', 'prepare', 'predictions', 'anchor_access', 'inputs', 'source_zip', 'model')
    if args.phase == 'post':
        inputs += ('targets', 'scores', 'target_access', 'pre_audit')
    assert set(auth['input_sha256']) == set(inputs)
    for key in inputs:
        assert getattr(args, key) and sha(getattr(args, key)) == auth['input_sha256'][key]
    prepared, prediction, curves = pre_audit(args)
    if args.phase == 'post':
        post_audit(args, prepared, prediction, curves)
    stamp = dt.datetime.now(dt.timezone.utc)
    report = {'schema': 'xanthan-independent-audit-v1', 'status': 'PASS', 'phase': args.phase,
              'actual_utc': stamp.isoformat(),
              'asia_shanghai': stamp.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
              'auditor_source_sha256': sha(__file__), 'authorization_sha256': sha(args.authorization),
              'predictions_sha256': sha(args.predictions), 'prepare_sha256': sha(args.prepare),
              'contexts_audited': 66, 'probability_laws_audited': 594,
              'own_anchor_budget': 6, 'target_dimension': 8,
              'preparation_conditions': 11, 'salinity_levels': 7,
              'raw_numeric_target_accesses': 528 if args.phase == 'post' else 0,
              'raw_numeric_anchor_accesses': 0, 'production_modules_imported': 0, 'new_fits': 0,
              'maximum_absolute_errors_by_family': ERRORS,
              'cdf_or_quantile_quadratures': len(QUAD_ERRORS),
              'maximum_reported_quadrature_error': max(QUAD_ERRORS) if QUAD_ERRORS else 0.,
              'optimizer_scope': 'Saved hyperparameters conditionally rebuilt; no optimizer replay/global optimum certification',
              'masking_scope': 'Offline per-context O/T isolation; no claim of globally unseen cells or new wet measurements',
              'resin_final20_accessed': False, 'new_scientific_result': False}
    if args.phase == 'post':
        report.update(scores_sha256=sha(args.scores), targets_sha256=sha(args.targets),
                      score_rows_audited=594, point_terms_audited=4752, macro_rows_audited=54)
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'PASS', 'phase': args.phase, 'output': args.output, 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
