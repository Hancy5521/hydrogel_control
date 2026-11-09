"""Bounded reused-development diagnostic. Never a fresh confirmation.

Only existing saved forecasts and the 240 already-released targets are used.
No raw-data reader, refitting, tau search, source reselection or final-20 access.
All new candidate laws are saved before this script reads the old target files.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr, ndtri, stdtr
from scipy.stats import multivariate_normal, multivariate_t, norm, t
from quantile_law import QuantileMidpoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = ('gaussian_gp', 'physical_full_student', 'student_gp_copula',
           'half_quantile_transport', 'half_density_mixture')


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def authorize(path):
    reg = read(path)
    if reg['schema'] != 'fixed-midpoint-reused-development-v1':
        raise ValueError('Explicit reused-development registration required')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'])
    if stamp.tzinfo is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise ValueError('Actual UTC must precede execution')
    for p, digest in reg['pins'].items():
        if sha(p) != digest:
            raise ValueError('Pinned input/source changed: ' + p)
    if reg['tau'] != .5 or reg['final_runs_accessed'] != 0 or reg['fresh_confirmation']:
        raise ValueError('Immutable diagnostic scope differs')
    return reg


def parents():
    a = read(ROOT / 'resin_dev_once/predict/PREDICTIONS.json')['forecasts']
    s = read(ROOT / 'student_dev_once/predict/PREDICTIONS.json')['forecasts']
    c = read(ROOT / 'transport_dev_once/predict/PREDICTIONS.json')['forecasts']
    e = read(ROOT / 'expansion_once/predict/PREDICTIONS.json')['forecasts']
    def index(rows, method):
        return {v['member']: v for v in rows if v['method_id'] == method}
    groups = [
        ('original_development', index(a, 'GP_nearest_target6'),
         index(s, 'power_law__GP_nearest_target6'), index(c, 'frozen_gp_posterior')),
        ('former_expansion_now_development_for_new_method', index(e, 'gaussian_frozen_gp'),
         index(e, 'physical_full_student'), index(e, 'copula_frozen_gp_posterior'))]
    for label, gs, ss, cs in groups:
        if not (len(gs) == len(ss) == len(cs) == 15 and set(gs) == set(ss) == set(cs)):
            raise ValueError('All 15 original curves and exactly matching parents required')
        for member in sorted(gs):
            g, s0, c0 = gs[member], ss[member], cs[member]
            if not (g['anchor_indices'] == s0['anchor_indices'] == c0['anchor_indices'] == [0, 1, 13, 14, 15, 16]
                    and g['target_indices'] == s0['target_indices'] == c0['target_indices'] == list(range(17, 25))
                    and g['anchor_values'] == s0['anchor_values'] == c0['anchor_values']):
                raise ValueError('Same information, native positions and budget required')
            yield label, g, s0, c0


def predict(output, registration):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'STARTED.json', {'actual_utc': utc(), 'registration_sha256': sha(registration), 'argv': sys.argv})
    forecasts = []
    for label, g, s, c in parents():
        m = np.array(g['mean']); cov = np.array(g['covariance'])
        sd = np.sqrt(np.diag(cov))
        scale = np.sqrt(np.diag(s['scale_matrix']))
        r = np.array(c['correlation'])
        if not np.allclose(r, cov / np.outer(sd, sd), atol=1e-12, rtol=1e-12):
            raise ValueError('Student copula must use the exact same GP latent correlation')
        law = QuantileMidpoint(m, sd, s['location'], scale, r)
        forecasts.append({
            'member': g['member'], 'recipe': g['recipe'], 'run_id': g['run_id'], 'reuse_partition': label,
            'anchor_indices': g['anchor_indices'], 'target_indices': g['target_indices'],
            'gaussian': {'mean': g['mean'], 'covariance': g['covariance']},
            'student': {'location': s['location'], 'scale_matrix': s['scale_matrix'], 'df': s['df']},
            'student_gp_copula': {'location': c['location'], 'scale': c['scale'], 'correlation': c['correlation'], 'df': c['df']},
            'midpoint': law.as_dict(), 'midpoint_quantiles05_50_95': law.quantile([.05, .5, .95]).tolist(),
            'density_mixture_weights': [.5, .5],
            'density_mixture_parents': ['gaussian_gp', 'student_gp_copula']})
    write(out / 'PREDICTIONS.json', {'actual_utc': utc(), 'registration_sha256': sha(registration),
        'scope': 'reused-development diagnostic, not fresh confirmation',
        'forecasts': forecasts, 'new_fits': 0, 'raw_data_read': False, 'new_target_reads': 0,
        'final_runs_accessed': 0, 'tau_candidates': [.5]})
    write(out / 'COMPLETE.json', {'actual_utc': utc(), 'predictions_sha256': sha(out / 'PREDICTIONS.json')})


def normal_crps(y, m, sd):
    z = (y - m) / sd
    return sd * (z * (2 * ndtr(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))


def cdf_crps(y, cdf):
    return (quad(lambda v: float(cdf(v))**2, -np.inf, y, epsabs=2e-10, epsrel=2e-9, limit=200)[0]
            + quad(lambda v: float(cdf(-v, upper=True))**2, -np.inf, -y,
                   epsabs=2e-10, epsrel=2e-9, limit=200)[0])


def score(output, registration):
    out = Path(output)
    predictions = read(out / 'PREDICTIONS.json')
    if predictions['registration_sha256'] != sha(registration) or read(out / 'COMPLETE.json')['predictions_sha256'] != sha(out / 'PREDICTIONS.json'):
        raise ValueError('All candidate laws must already be frozen')
    targets = {}
    for rel in ('resin_dev_once/reveal/TARGETS.json', 'expansion_once/reveal/TARGETS.json'):
        payload = read(ROOT / rel)
        for curve in payload['curves']:
            targets[curve['member']] = curve
    rows = []
    for f in predictions['forecasts']:
        target = targets[f['member']]
        y = np.asarray(target['values'])
        if target['target_indices'] != f['target_indices'] or y.shape != (8,):
            raise ValueError('Target identity differs')
        g, s, c, q = f['gaussian'], f['student'], f['student_gp_copula'], f['midpoint']
        mg, cg, ms, cs = np.array(g['mean']), np.array(g['covariance']), np.array(s['location']), np.array(s['scale_matrix'])
        sg, ss = np.sqrt(np.diag(cg)), np.sqrt(np.diag(cs))
        law = QuantileMidpoint(q['gaussian_location'], q['gaussian_sd'], q['student_location'], q['student_scale'], q['correlation'])
        lg = float(multivariate_normal.logpdf(y, mean=mg, cov=cg))
        ls = float(multivariate_t.logpdf(y, loc=ms, shape=cs, df=4))
        standard = (y - ms) / ss
        # Stable existing Student-copula log density, independently reconstructed
        # here from smaller tails. No importing the old copula implementation.
        from scipy.special import ndtri_exp
        log_tail = np.where(standard <= 0, t.logcdf(standard, 4), t.logsf(standard, 4))
        negative = ndtri_exp(log_tail)
        z = np.where(standard <= 0, negative, -negative)
        r = np.array(c['correlation'])
        lc = float(np.sum(t.logpdf(standard, 4) - np.log(ss)) - .5 * np.linalg.slogdet(r)[1]
                   - .5 * z @ np.linalg.solve(r, z) + .5 * z @ z)
        crps_g = normal_crps(y, mg, sg)
        crps_s = np.array([cdf_crps(yj, lambda v, upper=False, j=j:
            stdtr(4, -(v-ms[j])/ss[j]) if upper else stdtr(4, (v-ms[j])/ss[j])) for j, yj in enumerate(y)])
        crps_mix = np.array([cdf_crps(yj, lambda v, upper=False, j=j:
            .5 * (ndtr(-(v-mg[j])/sg[j]) + stdtr(4, -(v-ms[j])/ss[j])) if upper else
            .5 * (ndtr((v-mg[j])/sg[j]) + stdtr(4, (v-ms[j])/ss[j]))) for j, yj in enumerate(y)])
        qmix = np.empty((3, 8))
        for a, p in enumerate((.05, .5, .95)):
            for j in range(8):
                a0, b0 = min(mg[j]+sg[j]*ndtri(p), ms[j]+ss[j]*t.ppf(p, 4)), max(mg[j]+sg[j]*ndtri(p), ms[j]+ss[j]*t.ppf(p, 4))
                qmix[a, j] = brentq(lambda v: .5 * (ndtr((v-mg[j])/sg[j]) + stdtr(4, (v-ms[j])/ss[j]))-p,
                                   a0-1e-10, b0+1e-10, xtol=1e-13)
        qg = mg[None, :] + ndtri([.05, .5, .95])[:, None] * sg
        qs = ms[None, :] + t.ppf([.05, .5, .95], 4)[:, None] * ss
        data = [(-lg/8, crps_g, mg, qg), (-ls/8, crps_s, ms, qs),
                (-lc/8, crps_s, ms, qs), (-law.logpdf(y)/8, law.marginal_crps(y), (mg+ms)/2, law.quantile([.05, .5, .95])),
                (-float(logsumexp([lg, lc]))/8 + math.log(2)/8, crps_mix, (mg+ms)/2, qmix)]
        for method, (nll, crps, mean, quantiles) in zip(METHODS, data):
            rows.append({'member': f['member'], 'recipe': f['recipe'], 'run_id': f['run_id'],
                'reuse_partition': f['reuse_partition'], 'method': method, 'joint_nll_per_dim': float(nll),
                'mean_crps': float(np.mean(crps)), 'pointwise_crps': crps.tolist(),
                'log_rmse': float(np.sqrt(np.mean((y-mean)**2))),
                'coverage90': float(np.mean((y>=quantiles[0]) & (y<=quantiles[2]))),
                'width90': float(np.mean(quantiles[2]-quantiles[0]))})
    metrics = ('joint_nll_per_dim', 'mean_crps', 'log_rmse', 'coverage90', 'width90')
    by_recipe = []
    for recipe in sorted({r['recipe'] for r in rows}):
        for method in METHODS:
            subset = [r for r in rows if r['recipe']==recipe and r['method']==method]
            if len(subset) != 5:
                raise ValueError('Exactly five original runs per recipe')
            by_recipe.append({'recipe': recipe, 'method': method,
                **{metric: float(np.mean([r[metric] for r in subset])) for metric in metrics}})
    macro = [{ 'method': method, **{metric: float(np.mean([r[metric] for r in by_recipe if r['method']==method])) for metric in metrics}} for method in METHODS]
    write(out / 'SCORES.json', {'actual_utc': utc(), 'registration_sha256': sha(registration),
        'predictions_sha256': sha(out / 'PREDICTIONS.json'), 'scope': 'reused-development diagnostic only',
        'rows': rows, 'by_recipe': by_recipe, 'macro': macro, 'unique_already_consumed_targets': 240,
        'fresh_material_measurements': 0, 'new_fits': 0, 'new_final_target_access': 0})
    write(out / 'SCORE_COMPLETE.json', {'actual_utc': utc(), 'scores_sha256': sha(out / 'SCORES.json')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['predict', 'score'])
    parser.add_argument('--registration', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    authorize(args.registration)
    {'predict': predict, 'score': score}[args.phase](args.output, args.registration)


if __name__ == '__main__':
    main()
