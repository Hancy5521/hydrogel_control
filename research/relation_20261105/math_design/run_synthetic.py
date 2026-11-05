"""Root-approved synthetic principle/cache controls; no material model training.

Closed CRPS kernels have already received an independent CDF/energy audit in
the parent run. This program does not repeat those quadratures, or claim a new
algorithm by timing a cached implementation against a deliberately slow loop.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import tracemalloc

sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

from relation_core import (adaptive_cached, adaptive_loop_reference, centered_crps_kernel,
                           fixed_pair_values, posterior_branches, prepare, relation_covariance,
                           to_float_exact_support, validate_scope)

CODE = ('relation_core.py', 'run_synthetic.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    def bad(value):
        raise ValueError('Nonstandard JSON numeric constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=bad, object_pairs_hook=unique)


def encode(value):
    if isinstance(value, Fraction):
        return {'exact_fraction': str(value), 'float_display': float(value)}
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, dict):
        return {key: encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def pure_scope():
    return {'finite_relation_model': True, 'conditional_independence_of_units_given_relation': True,
            'proxy_marginal_theta_invariant': True, 'common_target_marginal': True,
            'repair_observes_candidate_target': False,
            'generated_or_derived_values_treated_as_measurements': False,
            'declarations_are_model_assumptions_not_physical_certification': True}


def configuration():
    return {'version': 'relation_posterior_synthetic_v1', 'material_file_access': False,
            'prior_identity_quadratures_repeated': False,
            'principle_cases': ['symmetric_signed_relation', 'theta_known', 'fake_calibration_identity',
                'zero_relation', 'noncollinear_single_proxy_selector', 'same_F_different_adaptive_branches'],
            'benchmark_families': ['homogeneous_scaling', 'known_state_noncollinear', 'dense_uninformative'],
            'benchmark_sizes': [{'repairs': 16, 'proxies': 32}, {'repairs': 64, 'proxies': 128}],
            'algorithms': ['explicit_tree_reference_correctness_only', 'fully_cached_vectorized',
                           'blocked_cached_vectorized_16'],
            'hard_budget_integer_ticks': 3, 'block_rows': 16,
            'timing_repetitions': 1, 'blas_threads': 1,
            'correctness_absolute_tolerance': 2e-12, 'correctness_relative_tolerance': 2e-11,
            'screening_implemented': False,
            'new_algorithm_or_material_efficacy_claimed': False}


def binary_probe(identifier, reliability='1', component=0, hypotheses=2, cost=1):
    reliability = Fraction(reliability)
    if not Fraction(1, 2) <= reliability <= 1:
        raise ValueError('Declared reliability must lie in[1/2,1]')
    if hypotheses == 2:
        signs = (-1, 1)
    elif hypotheses == 4:
        signs = tuple((-1, -1) if j == 0 else (-1, 1) if j == 1 else (1, -1) if j == 2 else (1, 1)
                      for j in range(4))
        signs = tuple(pair[component] for pair in signs)
    else:
        raise ValueError('Only declared2/4hypothesis fixtures are used')
    positive = [reliability if sign == 1 else 1 - reliability for sign in signs]
    return {'id': identifier, 'cost': cost,
            'likelihood': [[str(1 - p) for p in positive], [str(p) for p in positive]]}


def gaussian_cases():
    base = binary_probe('reveal_theta')
    proxy = {'id': 'signed_proxy', 'cost': 1, 'rhos': [-.8, .8], 'scale': 1.}
    zero_probe = {'id': 'independent_old_relation', 'cost': 1, 'likelihood': [['1/2', '1/2'], ['1/2', '1/2']]}
    return [('symmetric_signed_relation', ['1/2', '1/2'], [base], [proxy]),
            ('theta_known', ['1', '0'], [base], [proxy]),
            ('fake_calibration_identity', ['1/2', '1/2'], [zero_probe], [proxy]),
            ('zero_relation', ['1/2', '1/2'], [base], [dict(proxy, rhos=[0., 0.])]),
            ('noncollinear_single_proxy_selector', ['1/2', '1/2'], [base],
                [{'id': 'proxy0', 'cost': 1, 'rhos': [.8, 0.], 'scale': 1.},
                 {'id': 'proxy1', 'cost': 1, 'rhos': [0., .8], 'scale': 1.}])]


def run_principles():
    rows = []
    for name, prior, probes, proxies in gaussian_cases():
        prepared = prepare(prior, probes, proxies, 3, pure_scope())
        fixed, adaptive = fixed_pair_values(prepared), adaptive_cached(prepared)
        rows.append({'case': name, 'prior': prior, 'repair_actions': probes, 'proxy_actions': proxies,
                     'F_exact': prepared['F_exact'], 'S': prepared['kernels'],
                     'fixed_values': fixed['values'], 'complementarity': fixed['complementarity'],
                     'adaptive': adaptive})
    # The same-F counterexample is a binary Brier control, not a Gaussian CRPS
    # assertion. Independent parent derivation already supplies exact fractions.
    probes = [
        {'id': 'two_branch', 'cost': 1, 'likelihood': [['3/4', '1/4'], ['1/4', '3/4']]},
        {'id': 'three_branch', 'cost': 1, 'likelihood': [['0', '1/4'], ['3/4', '3/4'], ['1/4', '0']]},
    ]
    prior = (Fraction(1, 2), Fraction(1, 2))
    kernels = (((Fraction(1, 4), Fraction(0)), (Fraction(0), Fraction(0))),
               ((Fraction(0), Fraction(0)), (Fraction(0), Fraction(1, 4))))
    exact_cases = []
    for probe in probes:
        branches = posterior_branches(prior, probe['likelihood'])
        fixed_values = [sum((b.probability * sum((b.posterior[i] * s[i][j] * b.posterior[j]
                         for i in range(2) for j in range(2)), Fraction(0)) for b in branches), Fraction(0))
                        for s in kernels]
        adaptive_value = sum((b.probability * max(sum((b.posterior[i] * s[i][j] * b.posterior[j]
                             for i in range(2) for j in range(2)), Fraction(0)) for s in kernels)
                              for b in branches), Fraction(0))
        exact_cases.append({'id': probe['id'], 'F': relation_covariance(prior, branches),
                            'fixed_gains': fixed_values, 'adaptive_gain': adaptive_value,
                            'final_Brier': Fraction(1, 4) - adaptive_value})
    rows.append({'case': 'same_F_different_adaptive_branches', 'loss': 'mean binary Brier, separate from CRPS',
                 'exact_results': exact_cases})
    return rows


def make_benchmark(family, a_count, b_count):
    h = 4 if family == 'known_state_noncollinear' else 2
    prior = ['1/4'] * 4 if h == 4 else ['1/2', '1/2']
    probes = []
    for i in range(a_count):
        reliability = Fraction(1, 2) if family == 'dense_uninformative' else Fraction(1, 2) + Fraction(i + 1, 2 * (a_count + 1))
        probes.append(binary_probe('repair_' + str(i), reliability, component=i % 2,
                                   hypotheses=h, cost=1 + i % 2))
    proxies = []
    beta = ((-1., -.5), (-1., .5), (1., -.5), (1., .5))
    for j in range(b_count):
        if family == 'known_state_noncollinear':
            state = -2. + 4. * j / (b_count - 1)
            rho = [.8 * math.tanh(intercept + slope * state) for intercept, slope in beta]
            scale = 1.
        else:
            state, rho = None, [-.6, .6]
            scale = .5 + (j + 1) / (2 * b_count) if family == 'homogeneous_scaling' else 1.
        proxies.append({'id': 'proxy_' + str(j), 'cost': 1 + j % 2,
                        'rhos': rho, 'scale': scale, 'known_synthetic_state': state})
    return prior, probes, proxies


def timed(call):
    tracemalloc.start()
    start = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'one_shot_wall_seconds_including_preparation': elapsed,
                    'tracemalloc_managed_peak_bytes': peak,
                    'whole_process_peak_RSS_measured': False,
                    'timing_is_not_an_independent_scientific_episode': True}


def run_benchmark(family, a_count, b_count):
    prior, probes, proxies = make_benchmark(family, a_count, b_count)
    reference, reference_cost = timed(lambda: adaptive_loop_reference(prior, probes, proxies, 3, pure_scope()))
    def cached(block):
        prepared = prepare(prior, probes, proxies, 3, pure_scope())
        result = adaptive_cached(prepared, block_rows=block)
        result['fixed_pair_values'] = fixed_pair_values(prepared)['values']
        return result
    vectorized, vector_cost = timed(lambda: cached(None))
    blocked, block_cost = timed(lambda: cached(16))
    error = float(np.max(np.abs(reference['values'] - vectorized['values'])))
    block_error = float(np.max(np.abs(reference['values'] - blocked['values'])))
    if not np.allclose(reference['values'], vectorized['values'], rtol=2e-11, atol=2e-12):
        raise AssertionError('Cached adaptive values differ from explicit outcome tree')
    if not np.allclose(reference['values'], blocked['values'], rtol=2e-11, atol=2e-12):
        raise AssertionError('Blocked adaptive values differ from explicit outcome tree')
    row = {'family': family, 'repairs': a_count, 'proxies': b_count, 'hypotheses': len(prior),
           'prior': prior, 'repair_actions': probes, 'proxy_actions': proxies,
           'reference': reference, 'cached': vectorized, 'blocked': blocked,
           'timing': {'explicit_tree_correctness_reference_only': reference_cost,
                      'strong_fully_cached_vectorized_baseline': vector_cost,
                      'strong_blocked_cached_vectorized_baseline': block_cost},
           'max_abs_error_against_tree': error, 'max_block_abs_error_against_tree': block_error,
           'new_algorithm_comparison': False,
           'stronger_structural_shortcuts_must_not_be_ignored': True}
    if family in ('homogeneous_scaling', 'dense_uninformative'):
        prepared = prepare(prior, probes, proxies, 3, pure_scope())
        s0 = centered_crps_kernel([-.6, .6], 1.)
        shortcut_values, choices = [], []
        for i, branches in enumerate(prepared['branches']):
            legal = np.flatnonzero(prepared['feasible'][i])
            best = int(max(legal, key=lambda j: proxies[j]['scale']))
            gain = sum(float(branch.probability) * float(to_float_exact_support(branch.posterior) @ s0 @ to_float_exact_support(branch.posterior))
                       for branch in branches) * proxies[best]['scale']
            shortcut_values.append(gain)
            choices.append(best)
        if not np.allclose(shortcut_values, vectorized['values'], rtol=2e-11, atol=2e-12):
            raise AssertionError('Separable analytic-ranking negative control failed')
        row['separable_shortcut'] = {'values': shortcut_values, 'chosen_proxy': choices,
            'conclusion': 'This family does not require pair-matrix search; never claim speed benefit over naive loops here.'}
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    source = {name: sha(here / name) for name in CODE}
    plan, approval = load(args.plan), load(args.approval)
    expected = {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'configuration',
                'prior_independent_identity_receipt', 'no_real_material_execution'}
    if (set(plan) != expected or plan['kind'] != 'relation_posterior_bounded_synthetic'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != source
            or plan['configuration'] != configuration() or plan['no_real_material_execution'] is not True):
        raise ValueError('Exact source-pinned synthetic plan required')
    if (approval.get('approved') is not True or approval.get('scope') != 'RELATION_POSTERIOR_SYNTHETIC_CONTROL'
            or approval.get('plan_sha256') != sha(args.plan)):
        raise ValueError('Root exact-plan approval required before any synthetic execution')
    prior_receipt = plan['prior_independent_identity_receipt']
    if sha(prior_receipt['file']) != prior_receipt['sha256']:
        raise ValueError('Prior independent CDF/identity receipt changed')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'source_sha256': source,
        'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
        'prior_identity_audit_reused_not_repeated': prior_receipt, 'material_data_read': False})
    try:
        with threadpool_limits(limits=1):
            principles = run_principles()
            benchmarks = [run_benchmark(family, size['repairs'], size['proxies'])
                          for family in configuration()['benchmark_families']
                          for size in configuration()['benchmark_sizes']]
        if source != {name: sha(here / name) for name in CODE}:
            raise ValueError('Source changed during the single synthetic execution')
        dump(output / 'RESULTS.json', encode({'actual_end_utc': utc(), 'status': 'COMPLETE_SYNTHETIC_CONTROL',
             'source_sha256': source, 'principle_cases': principles, 'bounded_benchmark_cases': benchmarks,
             'no_real_material_fit_prediction_or_score': True,
             'no_new_algorithm_or_topconference_evidence_claimed': True,
             'timing_scope': 'Single ordered measurements, include preparation; managed memory only; not robust universal speed rankings.'}))
        print(json.dumps({'status': 'COMPLETE_SYNTHETIC_CONTROL', 'principle_cases': len(principles),
                          'bounded_benchmark_cases': len(benchmarks), 'material_data_read': False}))
    except Exception as error:
        dump(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
             'message': str(error), 'retain_failure_and_do_not_change_configuration': True})
        raise


if __name__ == '__main__':
    main()
