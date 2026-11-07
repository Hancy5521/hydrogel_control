"""Registered post-score identity decomposition of already USED fixed heads.

No SUT imports, fits, model predicts, candidate generation, CRPS scoring,
oracle coefficients, tau search, new grouping or new raw-source labels.
Uses independently audited frozen CDF knot statistics and consumed labels.
"""
import argparse
from bisect import bisect_right
from datetime import datetime, timezone
from fractions import Fraction as F
import gzip
import hashlib
import json
import math
from pathlib import Path

P = (F(1,20), F(9,20), F(9,20), F(1,20))
ZERO = F()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def unique(items):
        out = {}
        for key, value in items:
            require(key not in out, 'Duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique)


def vadd(a, b):
    require(len(a) == len(b), 'Vector dimension')
    return tuple(x+y for x, y in zip(a, b))


def dot(a, b):
    require(len(a) == len(b), 'Dot dimensions')
    return sum((x*y for x, y in zip(a, b)), ZERO)


def times(q, x):
    return tuple(dot(row, x) for row in q)


def beta(theta):
    theta = tuple(map(F, theta))
    require(len(theta) == 8 and all(sum(theta[k:k+4], ZERO) == 1 for k in (0,4)), 'Two simplexes')
    return tuple(theta[k+j]-P[j] for k in (0,4) for j in range(3))


def c_from_cached_knots(law, cached, target):
    """Linear c(Y) interpolation, constant outside the entire finite support."""
    mass = {}
    for x, w in zip(law['support_native'], law['weight_fractions']):
        x, w = F.from_float(float(x)), F(w)
        require(w >= 0, 'Negative mass')
        if w:
            mass[x] = mass.get(x, ZERO)+w
    require(sum(mass.values(), ZERO) == 1, 'Empirical probability one')
    xs = sorted(mass)
    knots = tuple(tuple(map(F, row)) for row in cached['knot_c'])
    require(len(xs) == len(knots) and all(len(row) == 6 for row in knots), 'Cache knot support/6D')
    y = F.from_float(float(target))
    if y <= xs[0]:
        return knots[0]
    if y >= xs[-1]:
        return knots[-1]
    j = bisect_right(xs, y)-1
    t = (y-xs[j])/(xs[j+1]-xs[j])
    return tuple(a+t*(b-a) for a, b in zip(knots[j], knots[j+1]))


def number(value):
    value = F(value)
    return {'exact_fraction': str(value), 'native_float': float(value)}


def vector_numbers(value):
    return [number(v) for v in value]


def cosine(a, b):
    a, b = tuple(map(float, a)), tuple(map(float, b))
    aa, bb = math.fsum(x*x for x in a), math.fsum(x*x for x in b)
    return None if not aa or not bb else math.fsum(x*y for x,y in zip(a,b))/math.sqrt(aa*bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--registration', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    reg = read(args.registration)
    require(reg['status'] == 'REGISTERED_POSTSCORE_FIXED_HEAD_IDENTITY_DIAGNOSTIC'
            and reg.get('actual_registration_utc'), 'Actual diagnostic registration required')
    require(reg['script_sha256'] == sha(__file__), 'Registered script changed')
    require(not args.output.exists(), 'One-shot saved diagnostic already exists')
    for pin in reg['inputs'].values():
        require(sha(pin['file']) == pin['sha256'], 'Registered input changed')
    start = datetime.now(timezone.utc).isoformat()
    data = {k: read(v['file']) for k,v in reg['inputs'].items() if k != 'cache'}
    with gzip.open(reg['inputs']['cache']['file'], 'rt', encoding='utf-8') as f:
        cache = json.load(f)
    cached = {r['component_id']:r for r in cache['records']}
    initial = data['initial_laws']['records']
    laws = {r['component_id']:r['F808'] for r in initial}
    contexts = {r['component_id']:r for r in data['contexts']['records']}
    true_remaining = data['remaining_targets']['records']
    true_query = data['queried_targets']['records']
    require(len(true_remaining) == 384 and len(true_query) == 128, 'Consumed split counts')
    rids, qids = [r['component_id'] for r in true_remaining], [r['component_id'] for r in true_query]
    require(len(set(rids)) == 384 and len(set(qids)) == 128 and not set(rids)&set(qids)
            and set(rids)|set(qids) == set(laws) == set(cached), 'Exact consumed512 partition')
    update = data['last_update']
    require(update['queries_complete'] == 128 and update['fit']['accepted'] is True, 'Actual final accepted fixed fit')
    require([initial[i]['component_id'] for i in update['remaining_indices']] == rids,
            'Same final384 order')
    theta, theta_prior = update['fit']['theta'], update['prior_theta_penalty_center_fixed808']
    require(theta_prior == data['priors']['context']['theta'], 'Same frozen808 prior penalty center')
    b, bp = beta(theta), beta(theta_prior)
    delta = tuple(x-y for x,y in zip(b,bp))
    q = [[ZERO]*6 for _ in range(6)]
    C, Cq = (ZERO,)*6, (ZERO,)*6
    per_row = []
    saved_scores = {r['component_id']:r for r in data['saved_scores']['records']}
    require(set(saved_scores) == set(rids), 'Same stored scores384')
    group_c = {'g_zero': {'n':0,'c':(ZERO,)*6}, 'g_positive': {'n':0,'c':(ZERO,)*6}}
    for row in true_remaining:
        key, y = row['component_id'], row['native_target']
        qi = tuple(tuple(map(F, a)) for a in cached[key]['Q'])
        ci = c_from_cached_knots(laws[key], cached[key], y)
        C = vadd(C,ci)
        for j in range(6):
            for k in range(6):
                q[j][k] += qi[j][k]
        local = dot(delta, times(qi,delta))+2*dot(delta,vadd(times(qi,bp),ci))
        stored = float(saved_scores[key]['geometryRiskCode'])-float(saved_scores[key]['priorcontext'])
        error = abs(float(local)-stored)
        require(error <= 1e-9, 'Per-object fixed score-difference identity')
        per_row.append({'component_id':key,'fixed_score_difference_identity':number(local),
                        'saved_score_difference_float':stored,'absolute_identity_error':error})
        group = 'g_zero' if F(cached[key]['g']) == 0 else 'g_positive'
        group_c[group]['n'] += 1
        group_c[group]['c'] = vadd(group_c[group]['c'],ci)
    for row in true_query:
        key, y = row['component_id'], row['native_target']
        Cq = vadd(Cq,c_from_cached_knots(laws[key],cached[key],y))
    q = tuple(map(tuple,q))
    require(q == tuple(tuple(map(F,row)) for row in update['Q_remaining_exact']), 'Independent remaining Gram exact')
    require(Cq == tuple(map(F,update['sum_c_queried_exact'])), 'Independent queried c total exact')
    chat = tuple(map(F,update['Chat_remaining_exact']))
    projected = tuple(map(F,update['fit']['diagnostics']['c_projected_exact']))
    N, n = 384, 128
    err = tuple(c-h for c,h in zip(C,projected))
    prior_direction_true = vadd(times(q,bp),C)
    prior_direction_estimated = vadd(times(q,bp),projected)
    curvature = dot(delta,times(q,delta))/N
    true_linear = 2*dot(delta,prior_direction_true)/N
    estimated_linear = 2*dot(delta,prior_direction_estimated)/N
    error_term = 2*dot(delta,err)/N
    real = curvature+true_linear
    plugin = curvature+estimated_linear
    penalty = F(1,4)*curvature
    require(real-plugin == error_term, 'Exact risk estimation-error identity')
    saved_delta = data['aggregate']['mean_CRPS_native']['geometryRiskCode']-data['aggregate']['mean_CRPS_native']['priorcontext']
    require(abs(float(real)-saved_delta) <= 1e-9, 'Aggregate saved-score difference identity')
    gtrue, gest = tuple(2*v/N for v in prior_direction_true), tuple(2*v/N for v in prior_direction_estimated)
    ooftotal = tuple(map(F,data['priors']['context']['diagnostics']['c_input_exact']))
    coordinates = []
    for j in range(6):
        coordinates.append({'coordinate':j,'block':'1_minus_g' if j<3 else 'g','pit_difference':(j%3)+1,
            'remaining_C_true':number(C[j]),'remaining_C_hat_raw':number(chat[j]),
            'remaining_C_hat_projected':number(projected[j]),
            'true_minus_hat':number(err[j]),'mean_true_minus_hat':number(err[j]/N),
            'delta_beta':number(delta[j]),'contribution_to_error_term':number(2*delta[j]*err[j]/N),
            'true_gradient_at_prior':number(gtrue[j]),'estimated_gradient_at_prior':number(gest[j]),
            'historical_OOF808_c_mean_descriptive':number(ooftotal[j]/808),
            'queried128_c_mean_descriptive':number(Cq[j]/n),
            'remaining384_c_mean_descriptive':number(C[j]/N)})
    groups = {name:{'n':r['n'],'c_mean':None if not r['n'] else vector_numbers(tuple(v/r['n'] for v in r['c']))}
              for name,r in group_c.items()}
    result = {'status':'PASS_POSTSCORE_FIXED_USED_HEAD_RISK_IDENTITY_NOT_NEW_CONFIRMATION',
        'actual_start_utc':start,'actual_end_utc':datetime.now(timezone.utc).isoformat(),
        'registration_sha256':sha(args.registration),'script_sha256':sha(__file__),
        'fixed_USED_theta':theta,'fixed_prior808_theta':theta_prior,
        'delta_beta':vector_numbers(delta),'remaining_N':N,'queries_n':n,
        'decomposition':{'known_nonnegative_curvature':number(curvature),
            'estimated_linear_at_prior':number(estimated_linear),'true_linear_at_prior':number(true_linear),
            'estimated_candidate_minus_prior':number(plugin),'true_candidate_minus_prior':number(real),
            'two_delta_dot_true_minus_estimated_c_over_N':number(error_term),
            'fixed_tau_prior_penalty':number(penalty),
            'estimated_regularized_objective_change':number(plugin+penalty),
            'saved_mean_score_difference_float':saved_delta,
            'saved_mean_score_identity_error':abs(float(real)-saved_delta)},
        'gradient_at_prior':{'true':vector_numbers(gtrue),'estimated':vector_numbers(gest),
            'euclidean_cosine':cosine(gtrue,gest),
            'first_block_cosine':cosine(gtrue[:3],gest[:3]),'g_block_cosine':cosine(gtrue[3:],gest[3:]),
            'delta_dot_true_gradient':number(dot(delta,gtrue)),
            'delta_dot_estimated_gradient':number(dot(delta,gest))},
        'coordinates':coordinates,'remaining_context_groups_descriptive':groups,
        'per_object_identity_checks':per_row,
        'maximum_per_object_score_difference_identity_error':max(r['absolute_identity_error'] for r in per_row),
        'scope':{'posthoc_descriptive':True,'new_fits':0,'new_model_predicts':0,
            'new_candidate_scores':0,'new_raw_source_reads':0,'new_targets_acquired':0,
            'oracle_coefficients_computed':False,'hyperparameters_searched':False,
            'OOF_vs_remaining_not_matched_reference':True,
            'no_variance_or_bias_causality_identified_by_one_realization':True}}
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'status':result['status'],'real_minus_prior':float(real),
        'plugin_minus_prior':float(plugin),'error_term':float(error_term),
        'gradient_cosine':result['gradient_at_prior']['euclidean_cosine'],
        'output':str(args.output),'sha256':sha(args.output)}))


if __name__ == '__main__':
    main()
