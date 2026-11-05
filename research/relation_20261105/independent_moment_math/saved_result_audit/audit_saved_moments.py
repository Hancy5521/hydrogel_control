"""Independent standard-library numerical review of frozen moment results.

Never imports/replays producer modules, controls or timing. Reads only the
registered synthetic result and hash bindings after exact root approval.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ABS_TOL = 2e-12
REL_TOL = 2e-11
SPOT_COORDINATES = ((0, 0), (7, 8), (15, 15))
EXPECTED_H = (8, 64, 128)
EXPECTED_R = (.25, .6, .9, .99)
EPSILON = 1e-6
DEGREE_CAP = 64


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), parse_constant=invalid)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def tolerance(reference):
    return ABS_TOL + REL_TOL * abs(reference)


def close(actual, expected, label, accumulator):
    if not math.isfinite(actual) or not math.isfinite(expected):
        raise AssertionError((label, 'nonfinite value'))
    error = abs(actual - expected)
    accumulator.append(error)
    if error > tolerance(expected):
        raise AssertionError((label, error, tolerance(expected)))


def coefficient(degree):
    return Fraction(math.comb(2 * degree, degree), 4 ** degree * (2 * degree - 1))


def independent_tail(r, scale, degree):
    if r == 0:
        return 0.
    alpha = Fraction(math.comb(2 * degree, degree), 4 ** degree)
    geometric = float(coefficient(degree + 1)) / (1 - r * r)
    return scale / math.sqrt(math.pi) * r ** (2 * degree + 2) * min(float(alpha), geometric)


def independent_degree(r, scale, hypotheses):
    if r == 0:
        return 0, 0., 'exact_zero_relation'
    for degree in range(1, DEGREE_CAP + 1):
        tail = independent_tail(r, scale, degree)
        if tail <= EPSILON:
            if degree >= hypotheses:
                return None, None, 'degree_not_smaller_than_H_use_exact'
            return degree, tail, 'moment_readout'
    return None, None, 'tail_not_certified_within_cap_use_exact'


def independent_quadratic(weights, rho, scale):
    """Different kernel formula and summation from producer's BLAS path."""
    return scale / math.sqrt(math.pi) * math.fsum(
        weights[h] * weights[k] * -math.expm1(.5 * math.log1p(-rho[h] * rho[k]))
        for h in range(len(weights)) for k in range(len(weights)))


def independent_moments(all_weights, rho, scale, degree):
    powers = [tuple(r ** n for r in rho) for n in range(1, degree + 1)]
    exact_coefficients = [float(coefficient(n)) for n in range(1, degree + 1)]
    answers = []
    for weights in all_weights:
        moments = [math.fsum(w * x for w, x in zip(weights, power)) for power in powers]
        answers.append(scale / math.sqrt(math.pi) * math.fsum(
            a * m * m for a, m in zip(exact_coefficients, moments)))
    return answers


def adaptive(scores, probabilities):
    return [math.fsum(probabilities[a][o] * max(scores[2 * a + o]) for o in range(2))
            for a in range(8)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    assert approval.get('approved') is True
    assert approval.get('scope') == 'INDEPENDENT_SAVED_MOMENT_NUMERICAL_REVIEW'
    assert approval.get('plan_sha256') == sha(args.plan)
    assert sha(__file__) == plan['audit_script_sha256']
    assert all(sha(path) == expected for path, expected in plan['input_sha256'].items())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'EXECUTION_START.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
        'input_sha256': plan['input_sha256'], 'producer_imports_or_replays': 0,
        'original_timings_repeated': False})
    saved = read(plan['frozen_results_file'])
    rows = saved['cases']
    reports = []

    def check(name, function):
        try:
            item = {'family': name, 'status': 'PASS', 'details': function()}
        except Exception as error:
            item = {'family': name, 'status': 'FAIL', 'error_type': type(error).__name__, 'message': str(error)}
        reports.append(item)
        with (out / 'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')

    def identities_and_shapes():
        original_plan = read(plan['producer_registered_plan_file'])
        original_start = read(plan['producer_execution_start_file'])
        assert original_start['plan_sha256'] == sha(plan['producer_registered_plan_file'])
        assert original_plan['source_sha256'] == original_start['source_sha256'] == plan['producer_source_sha256']
        assert original_plan['status'] == 'REGISTERED' and original_plan['configuration']['main_case_count'] == 12
        assert saved['status'] == 'COMPLETE_MOMENT_CONTROL'
        assert saved['material_fit_prediction_score_calls'] == 0
        assert saved['source_sha256'] == plan['producer_source_sha256']
        assert len(rows) == 12
        assert {(r['H'], r['rmax']) for r in rows} == {(h, r) for h in EXPECTED_H for r in EXPECTED_R}
        assert len({(r['H'], r['rmax']) for r in rows}) == 12
        for row in rows:
            data, h = row['data'], row['H']
            assert len(data['weights']) == len(data['rhos']) == 16
            assert all(len(x) == h for x in data['weights'] + data['rhos'])
            assert len(data['scales']) == 16 and data['posterior_still_full_H'] is True
            assert len(data['branch_probabilities']) == 8
            for w in data['weights']:
                assert all(math.isfinite(x) and x >= 0 for x in w)
                assert abs(math.fsum(w) - 1) <= 1e-13
            assert all(math.isfinite(x) and abs(x) < 1 for rr in data['rhos'] for x in rr)
            assert all(math.isfinite(s) and s > 0 for s in data['scales'])
            for p in data['branch_probabilities']:
                assert len(p) == 2 and all(x > 0 for x in p) and abs(math.fsum(p) - 1) <= 1e-13
            for key in ('streamed_exact', 'cached_exact', 'moment_or_fallback'):
                matrix = row['scores'][key]
                assert len(matrix) == 16 and all(len(x) == 16 for x in matrix)
                assert all(math.isfinite(x) for line in matrix for x in line)
        return {'unique_fixed_cases': 12, 'total_branch_proxy_coordinates': 3072,
                'full_H_branch_posteriors_retained': True, 'synthetic_not_material_data': True}

    def all_envelopes_degrees_and_moments():
        comparison_errors, independent_errors, tail_errors = [], [], []
        moment_coordinates, fallback_coordinates, point_coordinates = 0, 0, 0
        degree_rows = []
        for row in rows:
            data, scores, stats = row['data'], row['scores'], row['stats']['moments']
            w, rho, scales = data['weights'], data['rhos'], data['scales']
            exact, cached, lower = scores['streamed_exact'], scores['cached_exact'], scores['moment_or_fallback']
            tails = scores['tail_ideal_arithmetic']
            assert len(tails) == len(stats['degrees']) == len(stats['fallback_reasons']) == 16
            fallback_count = 0
            for b in range(16):
                r = max(abs(x) for x in rho[b])
                degree, expected_tail, reason = independent_degree(r, scales[b], row['H'])
                assert stats['degrees'][b] == degree and stats['fallback_reasons'][b] == reason
                close(tails[b], stats['ideal_arithmetic_truncation_tail_by_proxy'][b], 'duplicated saved tail', tail_errors)
                if degree is None:
                    fallback_count += 1
                    assert tails[b] == 0
                else:
                    assert 0 <= degree < row['H'] and degree <= DEGREE_CAP
                    close(tails[b], expected_tail, 'independent tail bound', tail_errors)
                    assert tails[b] <= EPSILON + tolerance(EPSILON)
                    independent = independent_moments(w, rho[b], scales[b], degree)
                for branch in range(16):
                    close(cached[branch][b], exact[branch][b], 'saved cached versus streamed', comparison_errors)
                    assert lower[branch][b] <= exact[branch][b] + tolerance(exact[branch][b])
                    assert exact[branch][b] <= lower[branch][b] + tails[b] + tolerance(exact[branch][b])
                    if degree is None:
                        close(lower[branch][b], exact[branch][b], 'saved exact fallback equality', comparison_errors)
                        fallback_coordinates += 1
                    else:
                        close(lower[branch][b], independent[branch], 'independent scalar moment readout', independent_errors)
                        moment_coordinates += 1
                    point_coordinates += 1
            assert fallback_count == stats['fallback_proxies']
            assert stats['posterior_compressed_for_future_updates'] is False
            assert stats['tail_excludes_roundoff_and_model_error'] is True
            degree_rows.append({'H': row['H'], 'rmax': row['rmax'], 'degrees': stats['degrees'],
                                'fallback_proxies': fallback_count})
        assert point_coordinates == 3072 and moment_coordinates + fallback_coordinates == 3072
        return {'all_stored_coordinate_envelopes_checked': point_coordinates,
                'independently_reconstructed_nonfallback_moment_coordinates': moment_coordinates,
                'fallback_coordinates_checked_against_frozen_exact': fallback_coordinates,
                'maximum_independent_moment_absolute_error': max(independent_errors, default=0.),
                'maximum_saved_exact_consistency_error': max(comparison_errors, default=0.),
                'maximum_independent_tail_absolute_error': max(tail_errors, default=0.),
                'degree_fallback_rows': degree_rows,
                'coverage_limit': 'Fallback/full-kernel independence is checked only at the separate36 predetermined spots.'}

    def independent_full_quadratic_spots():
        comparisons, errors = [], []
        for row in rows:
            data, scores = row['data'], row['scores']
            for branch, proxy in SPOT_COORDINATES:
                reference = independent_quadratic(data['weights'][branch], data['rhos'][proxy], data['scales'][proxy])
                close(scores['streamed_exact'][branch][proxy], reference, 'independent full quadratic', errors)
                comparisons.append({'H': row['H'], 'rmax': row['rmax'], 'coordinate': [branch, proxy],
                    'independent_fsum_log1p_expm1_reference': reference,
                    'saved_streamed_exact': scores['streamed_exact'][branch][proxy],
                    'absolute_error': errors[-1]})
        assert len(comparisons) == 36
        return {'independent_full_kernel_coordinates': 36, 'predeclared_coordinates_per_case': SPOT_COORDINATES,
                'maximum_absolute_error': max(errors), 'comparisons': comparisons,
                'not_claimed': 'All3072 fullquadratics were not independently recomputed.'}

    def all_actual_decisions():
        errors, comparisons = [], []
        for row in rows:
            p, s = row['data']['branch_probabilities'], row['scores']
            exact, lower, tails = s['streamed_exact'], s['moment_or_fallback'], s['tail_ideal_arithmetic']
            upper = [[lower[i][b] + tails[b] for b in range(16)] for i in range(16)]
            q_exact, q_lower, q_upper = adaptive(exact, p), adaptive(lower, p), adaptive(upper, p)
            for reported, expected in ((row['adaptive_exact_values'], q_exact),
                    (row['moment_value_lower_float'], q_lower), (row['moment_value_upper_float'], q_upper)):
                for x, y in zip(reported, expected):
                    close(x, y, 'saved adaptive arithmetic', errors)
            for a in range(8):
                assert q_lower[a] <= q_exact[a] + tolerance(q_exact[a])
                assert q_exact[a] <= q_upper[a] + tolerance(q_exact[a])
            chosen = row['selected_first_action']
            assert type(chosen) is int and 0 <= chosen < 8
            assert q_lower[chosen] >= max(q_lower) - tolerance(max(q_lower))
            proxy_map = [max(range(16), key=lambda b: lower[2 * chosen + o][b]) for o in range(2)]
            actual_value = math.fsum(p[chosen][o] * exact[2 * chosen + o][proxy_map[o]] for o in range(2))
            close(row['true_model_value_of_selected_approximate_policy'], actual_value, 'actual selected policy', errors)
            regret = max(q_exact) - actual_value
            close(row['restricted_model_regret_float'], regret, 'actual selected policy regret', errors)
            assert regret >= -tolerance(max(q_exact))
            certificate = max(q_upper) - max(q_lower)
            assert regret <= certificate + tolerance(max(q_exact))
            assert certificate <= max(tails) + tolerance(max(q_exact))
            fixed = [max(math.fsum(p[a][o] * exact[2 * a + o][b] for o in range(2)) for b in range(16))
                     for a in range(8)]
            gaps = [q_exact[a] - fixed[a] for a in range(8)]
            for old, new in zip(row['fixed_best_pair_values'], fixed):
                close(old, new, 'fixedpair arithmetic', errors)
            for old, new in zip(row['strict_adaptive_over_fixed_value_gap'], gaps):
                close(old, new, 'adaptive minus fixed arithmetic', errors)
            assert row['strict_positive_gap_count_over_2e-12'] == sum(x > ABS_TOL for x in gaps)
            assert row['IEEE_interval_certificate_implemented'] is False
            comparisons.append({'H': row['H'], 'rmax': row['rmax'], 'selected_first_action': chosen,
                'selected_proxy_per_branch': proxy_map, 'actual_policy_value_from_frozen_exact': actual_value,
                'recomputed_regret': regret, 'upper_minus_lower_gap': certificate, 'maximum_tail': max(tails)})
        assert len(comparisons) == 12
        return {'actual_selected_policy_arithmetic_cases': 12, 'maximum_absolute_error': max(errors),
                'comparisons': comparisons,
                'coverage_limit': 'Decision arithmetic uses frozen exact matrices, whose independent quadratic check covers36 spots.'}

    def decimal_weak_shape_readonly():
        control = saved['controls']['weak_shape']
        with localcontext() as context:
            context.prec = 80
            r = Decimal.from_float(control['r'])
            pi = Decimal('3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628')
            reference = (1 - ((1 - r * r).sqrt() + (1 + r * r).sqrt()) / 2) / pi.sqrt()
        assert reference > 0
        target = float(reference)
        values = {}
        for key in ('D2_positive_gain', 'stable_H2_reference'):
            value = control[key]
            assert value > 0 and abs(value - target) <= REL_TOL * abs(target)
            values[key] = {'saved': value, 'relative_error': abs(value - target) / abs(target)}
        assert saved['controls']['underflow_refusals'] == 2
        assert saved['controls']['ranking_reversal_scaled_units']['D1_argmax_without_certificate_wrong'] is True
        return {'Decimal_precision': 80, 'independent_reference': str(reference), 'saved_value_comparisons': values,
                'weak_shape_uses_relative_tolerance_only': REL_TOL,
                'saved_direct_matrix_float_value_not_used_as_oracle': control['direct_matrix_float_value'],
                'no_controls_or_CDF_integrals_replayed': True,
                'two_saved_underflow_refusals_only_read_not_reexecuted': True}

    for name, function in (('frozen_case_identities_shapes_and_scope', identities_and_shapes),
                           ('all_envelopes_degrees_and_nonfallback_moments', all_envelopes_degrees_and_moments),
                           ('independent_full_kernel_predeclared36_spots', independent_full_quadratic_spots),
                           ('all12_actual_selected_policy_and_gap_arithmetic', all_actual_decisions),
                           ('Decimal80_weak_shape_saved_control_readonly', decimal_weak_shape_readonly)):
        check(name, function)
    unchanged = all(sha(path) == expected for path, expected in plan['input_sha256'].items())
    report = {'actual_end_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if all(row['status'] == 'PASS' for row in reports) and unchanged else 'FAIL',
        'plan_sha256': sha(args.plan), 'families': reports, 'all_pinned_inputs_unchanged': unchanged,
        'producer_imports_replays_controls_or_timings': 0, 'material_arrays_reads_fits_or_scores': 0,
        'scope': 'Registered saved-synthetic-result numerical coverage only; no IEEE directed certification or performance/material efficacy claim.',
        'full_kernel_coverage': '36 predeclared independent quadratics, all3072 stored envelopes, all nonfallback moment readouts; no full3072 independent quadratics claim.'}
    write(out / 'INDEPENDENT_SAVED_MOMENT_REVIEW.json', report)
    print(json.dumps({'status': report['status'], 'families': len(reports), 'producer_replays': 0}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
