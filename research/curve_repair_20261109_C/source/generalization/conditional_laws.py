"""Own-context conditional laws. No source readers or target responses.

The fixed matrix-transport mechanism is classical. This file separates the
material experiment's orchestration from reference fitting and proper scores.
"""
from __future__ import annotations
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr, ndtri, stdtr
from scipy.stats import multivariate_normal, multivariate_t, norm, t
from anchor_gp import fit_predict as anchor_gp

ROOT = Path(__file__).resolve().parent.parent
PINS = {
    'baselines/pure_student.py': '567212afb8a48cebcdbf2b69f051f3a1ecfbc7b3d10e5f3bf45181aebb59c892',
    'marginal_transport/pure.py': '0db9e373dbbe35e2efa805606b9edcb065cb2d920b3f3745ae2fcf6e9529acc4',
    'quantile_midpoint/quantile_law.py': '9485750fa05cb060c5d026e8225a2ecb84c63386eacaf4095a4c66f75842c9c4'}
PERMUTATION = (0, 2, 4, 6, 1, 3, 5, 7)
METHODS = ('physical_full_student', 'frozen_gaussian_gp', 'anchor_fitted_gaussian_gp',
           'student_frozen_gp_copula', 'student_anchor_gp_copula',
           'student_anchor_gp_permuted_copula', 'student_independence',
           'half_quantile_anchor_gp', 'half_density_anchor_gp')


def pinned_module(relative):
    raw = (ROOT / relative).read_bytes()
    if hashlib.sha256(raw).hexdigest() != PINS[relative]:
        raise ValueError('Probability dependency bytes changed: ' + relative)
    name = '_generalization_' + PINS[relative]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        exec(compile(raw, str(ROOT / relative), 'exec'), mod.__dict__)
    return sys.modules[name]


def build_laws(x_observed, y_observed, x_target, frozen_gaussian):
    xo, yo, xt = map(np.asarray, (x_observed, y_observed, x_target))
    if xo.shape != (6,) or yo.shape != (6,) or xt.shape != (8,):
        raise ValueError('Registered same-budget six-to-eight context required')
    physical = pinned_module('baselines/pure_student.py').fit_predict(xo, yo, xt, 'power_law').as_dict()
    fitted = anchor_gp(xo, yo, xt)
    copula = pinned_module('marginal_transport/pure.py')
    midpoint = pinned_module('quantile_midpoint/quantile_law.py')
    scale = np.sqrt(np.diag(physical['scale_matrix']))
    r_frozen = copula.correlation_from_covariance(frozen_gaussian['covariance'])
    r_fit = copula.correlation_from_covariance(fitted['covariance'])
    p = np.array(PERMUTATION)
    r_wrong = r_fit[np.ix_(p, p)]
    laws = {
        METHODS[0]: physical,
        METHODS[1]: {'distribution': 'multivariate_gaussian', **frozen_gaussian},
        METHODS[2]: {'distribution': 'multivariate_gaussian', **fitted}}
    for method, r in zip(METHODS[3:7], (r_frozen, r_fit, r_wrong, np.eye(8))):
        laws[method] = copula.GaussianCopulaStudent(physical['location'], scale, r).as_dict()
    fitted_sd = np.sqrt(np.diag(fitted['covariance']))
    laws[METHODS[7]] = midpoint.QuantileMidpoint(fitted['mean'], fitted_sd,
                                               physical['location'], scale, r_fit).as_dict()
    laws[METHODS[8]] = {'distribution': 'equal_density_mixture', 'components': [METHODS[2], METHODS[4]],
                       'weights': [.5, .5]}
    return {'laws': laws, 'physical_fits': 1, 'anchor_gp_fits': 1,
            'permutation': list(PERMUTATION),
            'permuted_R_frobenius_distance': float(np.linalg.norm(r_fit-r_wrong)),
            'permuted_R_preserves_eigenvalues': True,
            'observations_received': int(len(yo)), 'target_responses_received': 0}


def _student_crps(y, m, s):
    a = (y-m)/s
    return s * (a*(2*stdtr(4, a)-1) + 2*(4+a*a)*t.pdf(a, 4)/3 - 15*math.pi/64)


def _gaussian_crps(y, m, s):
    a = (y-m)/s
    return s*(a*(2*ndtr(a)-1)+2*norm.pdf(a)-1/math.sqrt(math.pi))


def _mixture_crps(y, mg, sg, ms, ss):
    output = []
    for v, a, b, c, d in zip(y, mg, sg, ms, ss):
        def lower(w):
            return .5*(ndtr((w-a)/b)+stdtr(4, (w-c)/d))
        def survival(w):
            return .5*(ndtr(-(w-a)/b)+stdtr(4, -(w-c)/d))
        output.append(quad(lambda w: lower(w)**2, -np.inf, v, epsabs=2e-10, epsrel=2e-9)[0]
                      + quad(lambda w: survival(-w)**2, -np.inf, -v, epsabs=2e-10, epsrel=2e-9)[0])
    return np.array(output)


def score_context(laws, values):
    """All frozen laws receive the same eight targets only at scoring."""
    y = np.asarray(values, dtype=float)
    if y.shape != (8,) or not np.isfinite(y).all():
        raise ValueError('Eight finite target log responses required')
    copula = pinned_module('marginal_transport/pure.py')
    midpoint = pinned_module('quantile_midpoint/quantile_law.py')
    rows = []
    densities = {}
    for method in METHODS:
        law = laws[method]
        kind = law['distribution']
        if kind == 'multivariate_student_t':
            m = np.array(law['location']); sd = np.sqrt(np.diag(law['scale_matrix']))
            logp = float(multivariate_t.logpdf(y, loc=m, shape=law['scale_matrix'], df=4))
            crps = _student_crps(y, m, sd)
            q = m + t.ppf([.05, .5, .95], 4)[:, None]*sd
        elif kind == 'multivariate_gaussian':
            m = np.array(law['mean']); sd = np.sqrt(np.diag(law['covariance']))
            logp = float(multivariate_normal.logpdf(y, mean=m, cov=law['covariance']))
            crps = _gaussian_crps(y, m, sd)
            q = m + ndtri([.05, .5, .95])[:, None]*sd
        elif kind == 'gaussian_copula_student4_marginals':
            obj = copula.GaussianCopulaStudent(law['location'], law['scale'], law['correlation'])
            m = np.array(law['location']); sd = np.array(law['scale'])
            logp = obj.logpdf(y); crps = _student_crps(y, m, sd)
            q = obj.quantile([.05, .5, .95])
        elif kind == 'fixed_half_gaussian_student4_quantile_transport':
            obj = midpoint.QuantileMidpoint(*[law[k] for k in
                 ('gaussian_location', 'gaussian_sd', 'student_location', 'student_scale', 'correlation')])
            m = .5*(np.array(law['gaussian_location'])+np.array(law['student_location']))
            logp = obj.logpdf(y); crps = obj.marginal_crps(y); q = obj.quantile([.05, .5, .95])
        elif kind == 'equal_density_mixture':
            g, s = [laws[k] for k in law['components']]
            mg, sg = np.array(g['mean']), np.sqrt(np.diag(g['covariance']))
            ms, ss = np.array(s['location']), np.array(s['scale'])
            m = .5*(mg+ms)
            logp = float(logsumexp([densities[k] for k in law['components']])-math.log(2))
            crps = _mixture_crps(y, mg, sg, ms, ss)
            q = np.empty((3, 8))
            for i, prob in enumerate((.05, .5, .95)):
                for j in range(8):
                    endpoints = [mg[j]+sg[j]*ndtri(prob), ms[j]+ss[j]*t.ppf(prob, 4)]
                    low, high = min(endpoints), max(endpoints)
                    q[i, j] = brentq(lambda v: .5*(ndtr((v-mg[j])/sg[j])+stdtr(4, (v-ms[j])/ss[j]))-prob,
                                     low-1e-10, high+1e-10, xtol=1e-13)
        else:
            raise ValueError('Undeclared probability law: '+kind)
        densities[method] = logp
        rows.append({'method': method, 'joint_nll_per_dim': -logp/8,
            'crps': float(np.mean(crps)), 'pointwise_crps': crps.tolist(),
            'log_rmse': float(np.sqrt(np.mean((y-m)**2))),
            'coverage90': float(np.mean((y>=q[0]) & (y<=q[2]))),
            'width90': float(np.mean(q[2]-q[0]))})
    return rows
