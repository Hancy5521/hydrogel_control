"""Six fixed generated-label numerical cases; run only AFTER actual registration.

This runner is not a materials benchmark and never imports a materials dataset.
It makes ten SLSQP calls, including four successive query-learned updates and an
ordinary queried-ERM control. Failure is preserved; there are no retries.
"""
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import argparse
import datetime as dt
import json
import random
import sys
import numpy as np

CORE = Path(__file__).resolve().parents[1]/'core'
sys.path.insert(0, str(CORE))
from cdf_geometry import (P, EPSILON, ZERO, geometry, crps, theta_beta, dot,
                          matvec, range_geometry, vector_add)
from risk_code import (fit_code, fit_query_erm, identity_theta, aggregate_Q,
                       aggregate_c, QueryState, mixture_probabilities,
                       draw_rational, _objective_parts, _s_from_theta)


def plain(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


BASE = ((F(-3), F(1, 4)), (F(-1), F(1, 4)),
        (F(1), F(1, 4)), (F(3), F(1, 4)))
GENERATING_THETA = (F(1, 10), F(3, 20), F(13, 20), F(1, 10),
                    F(1, 10), F(13, 20), F(3, 20), F(1, 10))


def law_schema(atoms):
    return {'family': 'weighted_empirical', 'coordinate': 'native',
            'support_native': [str(x) for x, _ in atoms],
            'weight_fractions': [str(w) for _, w in atoms], 'unit': 'synthetic'}


def inputs(geos, labels):
    return {'laws': [law_schema(g.atoms) for g in geos],
            'g': [None if g.g is None else str(g.g) for g in geos],
            'true_generated_targets': list(map(str, labels)),
            'label_role': 'generated numerical labels; not material measurements'}


def fit_record(geos, labels, prior=None):
    q, c = aggregate_Q(geos), aggregate_c(geos, labels)
    prior = identity_theta(geos[0].dimension) if prior is None else prior
    result = fit_code(q, c, prior, len(geos))
    out = inputs(geos, labels)
    out.update({'Q_exact': q, 'c_exact': c, 'prior_theta': prior,
                'remaining_count': len(geos), 'fit': asdict(result)})
    return result, out


def score_checks(geo, theta):
    beta = theta_beta(theta)
    lo, hi = geo.gap_bounds(theta)
    xs = geo.points
    targets = (xs[0]-100, xs[0], (xs[0]+xs[-1])/2, xs[-1], xs[-1]+100)
    records = []
    for y in targets:
        difference = crps(geo.candidate(theta), y)-crps(geo.atoms, y)
        quadratic = dot(beta, matvec(geo.Q, beta))+2*dot(beta, geo.c(y))
        records.append({'y': y, 'score_difference': difference,
                        'quadratic_difference': quadratic,
                        'identity_exact': difference == quadratic,
                        'within_all_real_envelope': lo <= difference <= hi})
    return {'checks': records, 'gap_bounds': (lo, hi),
            'passed': all(v['identity_exact'] and v['within_all_real_envelope'] for v in records)}


def gradient_check(q, c, prior, count):
    objective, _, _ = _objective_parts(q, c, prior, count)
    # Interior feasible s; coordinate derivatives are unconstrained derivatives.
    ss = np.asarray([0.10, 0.20, 0.30, 0.40]*(len(prior)//4))
    value, analytic = objective(ss)
    eps = 1e-6
    finite = []
    for j in range(len(ss)):
        step = np.zeros(len(ss)); step[j] = eps
        finite.append((objective(ss+step)[0]-objective(ss-step)[0])/(2*eps))
    error = float(np.max(np.abs(np.asarray(finite)-analytic)))
    scale = max(1.0, float(np.max(np.abs(analytic))))
    return {'s': ss.tolist(), 'objective': value, 'analytic': analytic.tolist(),
            'central_difference': finite, 'step': eps, 'maximum_error': error,
            'tolerance': 1e-6*scale, 'passed': error <= 1e-6*scale}


def expected_score(geo, theta, truth_law):
    candidate = geo.candidate(theta)
    return sum((mass*crps(candidate, y) for y, mass in truth_law), ZERO)


def fixed_cases():
    cases = []
    base = geometry(BASE)
    # 1. Full-rank global actual learner and independent derivative comparison.
    geos = (base,)*8
    labels = tuple(map(F, (-3, -1, 1, 3, 3, 3, 3, 1)))
    fit, record = fit_record(geos, labels)
    gradients = gradient_check(record['Q_exact'], record['c_exact'],
                               record['prior_theta'], len(geos))
    checks = score_checks(base, fit.theta)
    cases.append({'name': 'global_fullrank_actual_fit', **record,
        'gradient_check': gradients, 'score_checks': checks,
        'passed': fit.accepted and fit.diagnostics['rank_exact'] == 3
                  and gradients['passed'] and checks['passed']})
    # 2. A constant context yields exact rank3, not six identifiable directions.
    geo = base.lift(F(1, 3)); geos = (geo,)*8
    fit, record = fit_record(geos, labels)
    rg = range_geometry(record['Q_exact'])
    null_checks = [dot(v, record['c_exact']) == 0 for v in rg['nullspace']]
    checks = score_checks(geo, fit.theta)
    cases.append({'name': 'constant_context_singular_actual_fit', **record,
        'null_c_exact_zero': null_checks, 'score_checks': checks,
        'passed': fit.accepted and rg['rank'] == 3 and all(null_checks) and checks['passed']})
    # 3. Rank0 still keeps the complete empirical support; tiny nonzero != zero.
    geo = geometry(((F(7), F(1)),)).lift(F(2, 5))
    fit, record = fit_record((geo,)*4, tuple(map(F, (-100, 0, 7, 100))))
    tiny = ((F(1), ZERO, ZERO), (ZERO, F(1, 10**30), ZERO),
            (ZERO, ZERO, F(1, 10**60)))
    rg = range_geometry(tiny)
    cases.append({'name': 'zero_geometry_and_exact_tiny_rank', **record,
        'tiny_matrix': tiny, 'tiny_rank': rg['rank'], 'tiny_projector': rg['projector'],
        'passed': fit.accepted and fit.diagnostics['rank_exact'] == 0
                  and fit.theta == identity_theta(6) and rg['rank'] == 3})
    # 4. True labels outside the support drive a boundary optimum, without
    #    deleting the reference atoms or putting probability beyond the support.
    fit, record = fit_record((base,)*8, (F(10**6),)*8)
    candidate = base.candidate(fit.theta)
    floor = all(t >= EPSILON*p for t, p in zip(fit.theta, P))
    boundary = min(_s_from_theta(fit.theta)) < F(1, 10**8)
    checks = score_checks(base, fit.theta)
    cases.append({'name': 'outside_support_floor_boundary_actual_fit', **record,
        'floor_exact': floor, 'boundary_reached': boundary,
        'support_preserved': tuple(x for x, _ in candidate) == base.points,
        'score_checks': checks,
        'passed': fit.accepted and floor and boundary and checks['passed']
                  and len(candidate) == len(BASE)})
    # 5. The learner receives generated outcomes, never the hidden direction.
    rng = random.Random(584102)
    geos = tuple(base.lift(F(i % 2)) for i in range(256))
    truths = tuple(g.candidate(GENERATING_THETA) for g in geos)
    labels = tuple(law[draw_rational(tuple(m for _, m in law), rng)[0]][0]
                   for law in truths)
    fit, record = fit_record(geos, labels)
    prior = identity_theta(6)
    old_risk = sum((expected_score(g, prior, law) for g, law in zip(geos, truths)), ZERO)/len(geos)
    new_risk = sum((expected_score(g, fit.theta, law) for g, law in zip(geos, truths)), ZERO)/len(geos)
    cases.append({'name': 'unknown_generated_direction_learned_from_true_labels', **record,
        'generation_seed': 584102, 'generating_theta_not_passed_to_fit': GENERATING_THETA,
        'exact_generating_population_reference_risk': old_risk,
        'exact_generating_population_learned_risk': new_risk,
        'exact_generating_population_gain': new_risk-old_risk,
        'passed': fit.accepted and fit.diagnostics['rank_exact'] == 6
                  and fit.theta != prior and new_risk < old_risk})
    # 6. Four fixed32 batches, all true outcomes generated before any query;
    #    only selected outcomes enter QueryState. No posterior synthetic Y.
    rng_y, rng_query = random.Random(901824), random.Random(910423)
    geos = []
    for i in range(256):
        shift, scale = F(i % 7, 10), F(1)+F(i % 5, 10)
        atoms = tuple((shift+scale*x, m) for x, m in BASE)
        geos.append(geometry(atoms).lift(F(i % 4, 3)))
    geos = tuple(geos)
    truths = tuple(g.candidate(GENERATING_THETA) for g in geos)
    labels = tuple(law[draw_rational(tuple(m for _, m in law), rng_y)[0]][0]
                   for law in truths)
    state, theta, updates, expectation_checks = QueryState(geos), identity_theta(6), [], []
    total_c = aggregate_c(geos, labels)
    for batch in range(4):
        state.start_batch(theta)
        q = mixture_probabilities([state.widths[i] for i in state.remaining],
                                  [state.gaps[i] for i in state.remaining])
        expectation = (ZERO,)*6
        for i, qi in zip(state.remaining, q):
            ci = geos[i].c(labels[i])
            z = tuple(m+p+(c-mid)/qi for m, p, c, mid in
                      zip(state.M, state.past_residual, ci, state.midpoints[i]))
            expectation = vector_add(expectation, tuple(qi*v for v in z))
        expectation_checks.append(expectation == total_c)
        for _ in range(32):
            draw = state.draw(rng_query)
            state.observe(labels[draw['index']])
        q_remaining, chat = state.remaining_Q(), state.remaining_estimate()
        result = fit_code(q_remaining, chat, identity_theta(6), len(state.remaining))
        theta = result.theta
        updates.append({'queries': len(state.records), 'Q_remaining': q_remaining,
            'Chat_remaining': chat, 'remaining_indices': list(state.remaining),
            'prior_theta': identity_theta(6), 'fit': asdict(result)})
    queried = [r['index'] for r in state.records]
    erm = fit_query_erm([geos[i] for i in queried], [labels[i] for i in queried],
                         state.remaining_Q(), identity_theta(6), len(state.remaining))
    positive = all(F(r['q']) >= F(1, 4*r['remaining_before']) for r in state.records)
    cases.append({'name': 'adaptive_four_batch_true_query_learn_loop', **inputs(geos, labels),
        'generation_seed': 901824, 'query_seed': 910423,
        'generating_theta_not_passed_to_fit': GENERATING_THETA,
        'conditional_total_expectation_exact_checks': expectation_checks,
        'strict_positive_uniform_floor': positive, 'query_records': state.records,
        'updates': updates, 'ordinary_query_erm': asdict(erm),
        'passed': all(expectation_checks) and positive and len(set(queried)) == 128
                  and len(state.remaining) == 128
                  and all(u['fit']['accepted'] for u in updates) and erm.accepted})
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registration = json.loads(args.registration.read_text(encoding='utf-8-sig'))
    if not registration.get('actual_registration_utc'):
        raise RuntimeError('Actual UTC preregistration required before any case')
    if args.output.exists():
        raise FileExistsError('Gate output already exists; no unregistered repeat')
    start = dt.datetime.now(dt.timezone.utc).isoformat()
    cases = fixed_cases()
    calls = sum(c.get('fit', {}).get('diagnostics', {}).get('optimizer_calls', 0) for c in cases)
    calls += sum(u['fit']['diagnostics']['optimizer_calls'] for u in cases[-1]['updates'])
    calls += cases[-1]['ordinary_query_erm']['diagnostics']['optimizer_calls']
    result = {'actual_start_utc': start, 'actual_end_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'registration': str(args.registration.resolve()), 'cases': cases,
              'optimizer_calls': calls, 'expected_optimizer_calls': 10,
              'material_data_access': False,
              'passed': calls == 10 and len(cases) == 6 and all(c['passed'] for c in cases)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plain(result), ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'cases': len(cases), 'optimizer_calls': calls,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
