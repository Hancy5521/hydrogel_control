"""One approved moment-readout control, without material data or old-pilot reruns."""
import argparse
from fractions import Fraction
from datetime import datetime, timezone
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

from moment_core import (NumericalRangeUnsupported, adaptive_value, coefficients,
    exact_cached, exact_streamed, h2_symmetric_stable, moment_streamed)

CODE = ('moment_core.py', 'run_moment_control.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError('Nonstandard JSON constant: ' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid, object_pairs_hook=unique)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def encoded(value):
    if isinstance(value, Fraction):
        return {'exact_fraction': str(value), 'float_display': float(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: encoded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(v) for v in value]
    return value


def configuration():
    return {'version': 'moment_decision_readout_v1', 'hypotheses': [8, 64, 128],
            'rmax': [.25, .6, .9, .99], 'repair_actions': 8, 'proxies': 16,
            'outcomes_per_repair': 2, 'epsilon_CRPS': 1e-6, 'degree_cap': 64,
            'degree_not_smaller_than_H': 'fallback_exact',
            'unresolved_tail_within_cap': 'fallback_exact',
            'timing_repetitions': 1, 'BLAS_threads': 1,
            'main_case_count': 12,
            'known_state_rho_function': 'rmax*tanh(1.1*z+0.2*c*z^2)/max_abs_h(tanh(...)); z uniform finite hypothesis grid,c known proxy state',
            'scale_function': '1+0.25*c^2',
            'algorithms': ['exact_streamed_per_proxy', 'exact_cached_kernel_bank', 'moment_streamed_with_exact_fallback'],
            'necessary_controls': ['D1_ranking_reversal_smoke_with_pinned_exact_audit', 'symmetric_1e-8_shape_and_stable_H2_reference',
                                   'final_gain_underflow_explicit_refusal', 'same_rho_separable_ordering'],
            'old_frozen_source_modified': False, 'material_execution': False,
            'tail_is_not_an_IEEE_or_model_correctness_certificate': True}


def timed(call):
    tracemalloc.start()
    start = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {'one_shot_wall_seconds': elapsed, 'managed_tracemalloc_peak_bytes': peak,
                    'whole_process_RSS_measured': False}


def inputs(hypotheses, rmax):
    z_exact = [Fraction(2 * h, hypotheses - 1) - 1 for h in range(hypotheses)]
    z = np.asarray([float(x) for x in z_exact])
    prior = [Fraction(1, hypotheses)] * hypotheses
    weights, probabilities = [], []
    for a in range(8):
        threshold = Fraction(2 * (a + 1), 9) - 1
        strength = Fraction(a + 1, 9)
        likelihood_plus = [(1 + strength * ((value > threshold) - (value < threshold))) / 2 for value in z_exact]
        action_p = []
        for likelihood in ([1 - q for q in likelihood_plus], likelihood_plus):
            joint = [w * q for w, q in zip(prior, likelihood)]
            p = sum(joint, Fraction(0))
            if p <= 0:
                raise AssertionError('Declared two-outcome fixtures need positive branches')
            weights.append([float(j / p) for j in joint])
            action_p.append(float(p))
        probabilities.append(action_p)
    states = np.linspace(-1., 1., 16)
    rhos, scales = [], []
    for c in states:
        profile = np.tanh(1.1 * z + .2 * c * z * z)
        profile = rmax * profile / np.max(np.abs(profile))
        rhos.append(profile)
        scales.append(1. + .25 * c * c)
    w, rho, scale, prob = np.asarray(weights), np.asarray(rhos), np.asarray(scales), np.asarray(probabilities)
    return {'weights': w, 'rhos': rho, 'scales': scale, 'branch_probabilities': prob,
            'known_proxy_states': states,
            'common_numeric_input_bytes': int(w.nbytes + rho.nbytes + scale.nbytes + prob.nbytes),
            'posterior_still_full_H': True, 'source_of_labels': 'finite declared hypothetical likelihood, not material response'}


def controls():
    profiles = np.asarray([[.75, -.75], [.125, .125]])
    first_moments = profiles @ np.asarray([.5, .5])
    d1 = coefficients(1)[0] * first_moments ** 2
    assert d1[0] == 0 and d1[1] == 1 / 128
    r = 1e-8
    power = np.asarray([[r, r * r], [-r, r * r]])
    moment = np.asarray([[.5, .5]]) @ power
    positive = float(np.sum(moment * moment * coefficients(2)[None, :]) / math.sqrt(math.pi))
    stable = h2_symmetric_stable(r)
    assert positive > 0 and math.isclose(positive, stable, rel_tol=2e-14, abs_tol=0.)
    direct, _ = exact_streamed([[.5, .5]], [[r, -r]], [1.])
    refused = []
    tests = [lambda: moment_streamed([[1., 1e-200]], [[0., 1e-150]], [1.]),
             lambda: adaptive_value([[0.], [1e-300]], [[1., 1e-200]])]
    for test in tests:
        try:
            test()
        except NumericalRangeUnsupported:
            refused.append(True)
        else:
            raise AssertionError('Unsupported positive final gain must not silently become0')
    f = inputs(8, .6)
    same = np.tile(f['rhos'][0], (16, 1))
    scales = np.linspace(.5, 1.5, 16)
    full, _ = exact_streamed(f['weights'], same, scales)
    normalized = full / scales[None, :]
    assert np.allclose(normalized, normalized[:, :1], atol=2e-14, rtol=2e-13)
    assert np.all(np.argmax(full, axis=1) == 15)
    return {'ranking_reversal_scaled_units': {'D1_implementation_lower_values': d1,
                    'D1_argmax_without_certificate_wrong': True,
                    'exact_order_tail_and_branch_regret_proofs': 'Reused pinned independent Fraction/isqrt receipt; not repeated here.'},
            'weak_shape': {'r': r, 'D2_positive_gain': positive, 'stable_H2_reference': stable,
                           'direct_matrix_float_value': float(direct[0, 0]),
                           'stable_H2_is_a_required_strong_reference': True},
            'underflow_refusals': len(refused),
            'same_rho_separable': 'exact scales collapse to one profile; pair search unnecessary',
            'no_new_material_or_old_source_fix_claim': True}


def run_case(h, r):
    data, setup_cost = timed(lambda: inputs(h, r))
    args = (data['weights'], data['rhos'], data['scales'])
    (stream, stream_stats), stream_cost = timed(lambda: exact_streamed(*args))
    (cached, cache_stats), cached_cost = timed(lambda: exact_cached(*args))
    (lower, moment_stats), moment_cost = timed(lambda: moment_streamed(*args, epsilon=1e-6, cap=64))
    tail = moment_stats['ideal_arithmetic_truncation_tail_by_proxy']
    if not np.allclose(stream, cached, atol=2e-12, rtol=2e-11):
        raise AssertionError('Strong exact implementations differ')
    if (lower - stream > 2e-12).any() or (stream - lower - tail[None, :] > 2e-12).any():
        raise AssertionError('Ideal truncation envelope violated beyond separately declared numerical tolerance')
    probabilities = data['branch_probabilities']
    exact_value = adaptive_value(stream, probabilities)
    lo = adaptive_value(lower, probabilities)
    hi = adaptive_value(lower + tail[None, :], probabilities)
    chosen_a = int(np.argmax(lo))
    selected_proxy = np.argmax(lower, axis=1).reshape(8, 2)
    true_policy_value = sum(probabilities[chosen_a, o] * stream[2 * chosen_a + o, selected_proxy[chosen_a, o]] for o in range(2))
    regret = float(exact_value.max() - true_policy_value)
    if regret > float(tail.max()) + 2e-12:
        raise AssertionError('Restricted two-stage regret exceeds tail plus numerical tolerance')
    fixed_value = np.max((probabilities[:, :, None] * stream.reshape(8, 2, 16)).sum(axis=1), axis=1)
    strict_gap = exact_value - fixed_value
    return {'H': h, 'rmax': r, 'data': data, 'scores': {'streamed_exact': stream, 'cached_exact': cached,
                    'moment_or_fallback': lower, 'tail_ideal_arithmetic': tail},
            'adaptive_exact_values': exact_value, 'fixed_best_pair_values': fixed_value,
            'strict_adaptive_over_fixed_value_gap': strict_gap,
            'strict_positive_gap_count_over_2e-12': int((strict_gap > 2e-12).sum()),
            'moment_value_lower_float': lo, 'moment_value_upper_float': hi,
            'selected_first_action': chosen_a, 'true_model_value_of_selected_approximate_policy': float(true_policy_value),
            'restricted_model_regret_float': regret,
            'IEEE_interval_certificate_implemented': False,
            'stats': {'streamed_exact': stream_stats, 'cached_exact': cache_stats, 'moments': moment_stats},
            'costs': {'common_posterior_input_setup': setup_cost, 'streamed_exact': stream_cost,
                      'cached_exact': cached_cost, 'moment_with_fallback': moment_cost},
            'reported_total_wall_seconds': {name: setup_cost['one_shot_wall_seconds'] + value['one_shot_wall_seconds']
                for name, value in (('streamed_exact', stream_cost), ('cached_exact', cached_cost), ('moment_with_fallback', moment_cost))},
            'proxy_or_index_changes_are_not_evidence_of_strict_gain': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = {name: sha(root / name) for name in CODE}
    plan, approval = load(args.plan), load(args.approval)
    keys = {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'configuration', 'prior_audit_receipts', 'no_material_execution'}
    if (set(plan) != keys or plan['kind'] != 'bounded_moment_decision_readout'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != source
            or plan['configuration'] != configuration() or plan['no_material_execution'] is not True):
        raise ValueError('Exact registered bounded synthetic plan required')
    if (approval.get('approved') is not True or approval.get('scope') != 'MOMENT_READOUT_SYNTHETIC_CONTROL'
            or approval.get('plan_sha256') != sha(args.plan)):
        raise ValueError('Root exact-plan approval required')
    if set(plan['prior_audit_receipts']) != {'centered_CRPS_identity', 'exact_moment_interval'}:
        raise ValueError('The two prior mathematical audit receipts must be pinned')
    for receipt in plan['prior_audit_receipts'].values():
        if sha(receipt['file']) != receipt['sha256']:
            raise ValueError('Prior independent proof receipt changed')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'source_sha256': source,
        'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
        'prior_audit_receipts_reused_not_reexecuted': plan['prior_audit_receipts'],
        'old_source_or_material_data_access': False})
    try:
        with threadpool_limits(limits=1):
            qualification = controls()
            cases = [run_case(h, r) for h in configuration()['hypotheses'] for r in configuration()['rmax']]
        if source != {name: sha(root / name) for name in CODE}:
            raise ValueError('Source changed during approved execution')
        write(output / 'RESULTS.json', encoded({'actual_end_utc': utc(), 'status': 'COMPLETE_MOMENT_CONTROL',
            'source_sha256': source, 'controls': qualification, 'cases': cases,
            'material_fit_prediction_score_calls': 0,
            'no_topconference_or_new_kernel_algorithm_claim': True,
            'scope': 'Ideal truncation mathematics plus bounded float implementation observations; not IEEE interval or material-model certification.'}))
        print(json.dumps({'status': 'COMPLETE_MOMENT_CONTROL', 'cases': len(cases), 'material_execution': False}))
    except Exception as error:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
             'message': str(error), 'retain_failure_no_retune': True})
        raise


if __name__ == '__main__':
    main()
