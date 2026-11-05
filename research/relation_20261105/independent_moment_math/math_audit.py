"""Pure-standard-library rational audit of moment truncation and policy bounds.

No producer import, model, material access or numerical execution before an
exact source-pinned plan and root approval. All gains are sqrt(pi)*g/sigma.
"""
import argparse
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import itertools
import json
from math import comb, isqrt
from pathlib import Path

DEGREES = (1, 2, 4, 8)
PRECISION_DIGITS = 60
GAIN_CASES = (
    ('zero', ('0',), ('1',)),
    ('one_relation', ('3/5',), ('1',)),
    ('signed_cancellation', ('3/4', '-3/4'), ('1/2', '1/2')),
    ('heterogeneous_three', ('-4/5', '1/3', '2/3'), ('1/4', '1/2', '1/4')),
    ('heterogeneous_four', ('-9/10', '-1/10', '2/5', '9/10'), ('1/10', '1/5', '3/10', '2/5')),
    ('near_one', ('999/1000', '999/1000'), ('1/2', '1/2')),
    ('near_one_signed', ('-999/1000', '999/1000'), ('1/2', '1/2')),
)
CHANNELS = {
    'uninformative': (('1', '1'),),
    'perfect': (('1', '0'), ('0', '1')),
    'soft_A': (('1/4', '3/4'), ('3/4', '1/4')),
    'same_F_B': (('0', '1/4'), ('3/4', '3/4'), ('1/4', '0')),
}
SELECTOR = (('3/4', '0'), ('0', '3/4'))
PANELS = (
    ('selector', ('uninformative', 'soft_A', 'perfect'), SELECTOR),
    ('same_F', ('same_F_B', 'soft_A'), SELECTOR),
    ('rank_reversal', ('uninformative',), (('3/4', '-3/4'), ('1/8', '1/8'))),
    ('same_matrix_tie', ('soft_A', 'perfect'), (('3/5', '-3/5'), ('3/5', '-3/5'))),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def ready(value):
    if isinstance(value, F):
        return {'fraction': str(value), 'numerator': value.numerator, 'denominator': value.denominator}
    if isinstance(value, dict):
        return {key: ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ready(item) for item in value]
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(ready(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def configuration():
    return {'precision_decimal_digits': PRECISION_DIGITS, 'degrees': list(DEGREES),
            'gain_cases': GAIN_CASES, 'channels': CHANNELS, 'panels': PANELS,
            'prior': ['1/2', '1/2'], 'sigma': '1', 'calibration_cost': '1',
            'proxy_cost': '1', 'hard_budget': '2', 'near_one_tolerance': '1/1000',
            'value_units': 'sqrt(pi)*g/sigma; all candidates share sigma and costs',
            'tie_break': 'first declared channel, then first declared proxy in each branch',
            'producer_imports_material_reads_and_fits': 0}


def square_root_interval(x):
    """Exact rational enclosure from one integer square root, no floats."""
    x = F(x)
    assert x >= 0
    scale = 10 ** PRECISION_DIGITS
    square = (x.numerator * scale * scale) // x.denominator
    k = isqrt(square)
    lo = F(k, scale)
    exact = k * k * x.denominator == x.numerator * scale * scale
    hi = lo if exact else F(k + 1, scale)
    assert lo * lo <= x <= hi * hi and hi - lo <= F(1, scale)
    return lo, hi


def f_interval(t):
    assert -1 <= t <= 1
    lo, hi = square_root_interval(1 - t)
    return 1 - hi, 1 - lo


def interval_sum(weighted):
    lo, hi = F(0), F(0)
    for weight, interval in weighted:
        assert weight >= 0 and interval[0] <= interval[1]
        lo += weight * interval[0]
        hi += weight * interval[1]
    return lo, hi


def interval_scale(value, weight):
    assert weight >= 0
    return weight * value[0], weight * value[1]


def validate_distribution(rhos, weights):
    assert rhos and len(rhos) == len(weights)
    assert all(abs(r) < 1 for r in rhos)
    assert all(w >= 0 for w in weights) and sum(weights, F(0)) == 1


def full_gain_interval(rhos, weights):
    validate_distribution(rhos, weights)
    return interval_sum((weights[h] * weights[k], f_interval(rhos[h] * rhos[k]))
                        for h in range(len(rhos)) for k in range(len(rhos)))


def coefficient(n):
    assert n >= 1
    return F(comb(2 * n, n), 4 ** n * (2 * n - 1))


def alpha(degree):
    return F(comb(2 * degree, degree), 4 ** degree)


def moment_gain(rhos, weights, degree):
    validate_distribution(rhos, weights)
    return sum((coefficient(n) * sum((w * r ** n for r, w in zip(rhos, weights)), F(0)) ** 2
                for n in range(1, degree + 1)), F(0))


def tail_bound(rhos, degree):
    r = max(abs(r) for r in rhos)
    return alpha(degree) * r ** (2 * degree + 2)


def certified_gain(rhos, weights, degree):
    full = full_gain_interval(rhos, weights)
    truncated, bound = moment_gain(rhos, weights, degree), tail_bound(rhos, degree)
    error = full[0] - truncated, full[1] - truncated
    assert 0 <= error[0] <= error[1] <= bound
    return {'full_kernel_interval': full, 'truncated_gain': truncated,
            'actual_truncation_error_interval': error, 'analytic_tail_upper_bound': bound}


def branches(name):
    prior = (F(1, 2), F(1, 2))
    likelihood = [tuple(map(F, row)) for row in CHANNELS[name]]
    assert all(len(row) == 2 and all(0 <= v <= 1 for v in row) for row in likelihood)
    assert all(sum((row[h] for row in likelihood), F(0)) == 1 for h in range(2))
    answer = []
    for outcome, row in enumerate(likelihood):
        probability = sum((prior[h] * row[h] for h in range(2)), F(0))
        assert probability > 0
        posterior = tuple(prior[h] * row[h] / probability for h in range(2))
        answer.append({'outcome': outcome, 'probability': probability, 'posterior': posterior})
    assert sum((b['probability'] for b in answer), F(0)) == 1
    assert all(sum((b['probability'] * b['posterior'][h] for b in answer), F(0)) == prior[h]
               for h in range(2))
    return answer


def posterior_covariance(name):
    return tuple(tuple(sum((b['probability'] * (b['posterior'][h] - F(1, 2))
                            * (b['posterior'][k] - F(1, 2)) for b in branches(name)), F(0))
                       for k in range(2)) for h in range(2))


def all_true_policies(channel, proxies):
    """Enumerate every branch-to-proxy map, independent of moment ranking."""
    bs = branches(channel)
    answer = []
    for mapping in itertools.product(range(len(proxies)), repeat=len(bs)):
        value = interval_sum((b['probability'], full_gain_interval(proxies[mapping[j]], b['posterior']))
                             for j, b in enumerate(bs))
        answer.append({'channel': channel, 'proxy_map_by_outcome': mapping,
                       'actual_value_interval': value, 'charged_cost': F(2)})
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    assert approval.get('approved') is True
    assert approval.get('scope') == 'INDEPENDENT_RATIONAL_MOMENT_MATH_AUDIT'
    assert approval.get('plan_sha256') == sha(args.plan)
    assert sha(__file__) == plan['audit_script_sha256']
    # JSON canonicalization only converts fixed tuples to lists.
    assert plan['configuration'] == json.loads(json.dumps(configuration()))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'EXECUTION_START.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
        'audit_script_sha256': sha(__file__), 'producer_imports_and_material_reads': 0})
    reports = []

    def check(name, function):
        try:
            row = {'family': name, 'status': 'PASS', 'details': function()}
        except Exception as error:
            row = {'family': name, 'status': 'FAIL', 'error_type': type(error).__name__, 'message': str(error)}
        reports.append(row)
        with (out / 'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(ready(row), ensure_ascii=False) + '\n')

    def coefficient_and_sqrt_arithmetic():
        recurrence = F(1, 2)
        for n in range(1, max(DEGREES) + 2):
            assert coefficient(n) == recurrence == alpha(n - 1) - alpha(n)
            recurrence *= F(2 * n - 1, 2 * n + 2)
        for d in DEGREES:
            assert sum((coefficient(n) for n in range(1, d + 1)), F(0)) + alpha(d) == 1
        for x in map(F, ('0', '1', '1/2', '7/16', '2', '1999/1000000')):
            lo, hi = square_root_interval(x)
            assert lo ** 2 <= x <= hi ** 2
        return {'coefficient_recurrence_and_telescoping_exact': True,
                'sqrt_enclosures_use_integer_isqrt_and_60_decimal_places': True,
                'perfect_squares_collapse_to_exact_endpoint': True,
                'infinite_series_identity_is_the_stated_mathematical_lemma_not_inferred_from_finite_tests': True}

    def fixed_gain_bounds():
        rows = []
        for name, raw_rhos, raw_weights in GAIN_CASES:
            rhos, weights = tuple(map(F, raw_rhos)), tuple(map(F, raw_weights))
            previous = F(-1)
            for degree in DEGREES:
                result = certified_gain(rhos, weights, degree)
                assert result['truncated_gain'] >= previous
                previous = result['truncated_gain']
                rows.append({'case': name, 'degree': degree, **result})
        return {'fixed_cases': len(GAIN_CASES), 'degrees': DEGREES,
                'all_gain_interval_and_tail_comparisons': rows,
                'rounding_tolerance_used': 'None; exact integer/Fraction comparisons'}

    def branch_and_same_F_semantics():
        all_branches = {name: branches(name) for name in CHANNELS}
        assert [b['posterior'][0] for b in all_branches['soft_A']] == [F(1, 4), F(3, 4)]
        assert [b['probability'] for b in all_branches['same_F_B']] == [F(1, 8), F(3, 4), F(1, 8)]
        assert [b['posterior'][0] for b in all_branches['same_F_B']] == [0, F(1, 2), 1]
        f = ((F(1, 16), F(-1, 16)), (F(-1, 16), F(1, 16)))
        assert posterior_covariance('soft_A') == posterior_covariance('same_F_B') == f
        proxies = tuple(tuple(map(F, row)) for row in SELECTOR)
        s = f_interval(F(9, 16))
        adaptive = {}
        for name, multiplier in (('soft_A', F(9, 16)), ('same_F_B', F(7, 16))):
            policies = all_true_policies(name, proxies)
            value = max(p['actual_value_interval'][0] for p in policies), max(p['actual_value_interval'][1] for p in policies)
            assert value == interval_scale(s, multiplier)
            adaptive[name] = value
            for proxy in proxies:
                fixed = interval_sum((b['probability'], full_gain_interval(proxy, b['posterior']))
                                     for b in branches(name))
                assert fixed == interval_scale(s, F(5, 16))
        assert adaptive['soft_A'][0] > adaptive['same_F_B'][1]
        return {'branches_exactly_recovered_from_likelihood': all_branches, 'same_F': f,
                'adaptive_A_equals_9s_over16': adaptive['soft_A'],
                'adaptive_B_equals_7s_over16': adaptive['same_F_B'],
                'fixed_pair_both_equal_5s_over16': interval_scale(s, F(5, 16)),
                'averaged_F_is_not_sufficient_for_adaptive_max': True}

    def adaptive_policy_certificates():
        results = []
        for panel, channel_names, raw_proxies in PANELS:
            proxies = tuple(tuple(map(F, row)) for row in raw_proxies)
            all_policies = [policy for name in channel_names for policy in all_true_policies(name, proxies)]
            exact_optimum = (max(p['actual_value_interval'][0] for p in all_policies),
                             max(p['actual_value_interval'][1] for p in all_policies))
            for degree in DEGREES:
                epsilons = [tail_bound(proxy, degree) for proxy in proxies]
                channels = []
                for name in channel_names:
                    lower, upper, mapping = F(0), F(0), []
                    for b in branches(name):
                        values = []
                        for proxy in proxies:
                            certified_gain(proxy, b['posterior'], degree)
                            values.append(moment_gain(proxy, b['posterior'], degree))
                        selected = max(range(len(proxies)), key=lambda i: values[i])
                        mapping.append(selected)
                        lower += b['probability'] * values[selected]
                        upper += b['probability'] * max(values[i] + epsilons[i] for i in range(len(proxies)))
                    channels.append({'channel': name, 'lower': lower, 'upper': upper,
                                     'proxy_map_by_outcome': tuple(mapping)})
                chosen = max(channels, key=lambda row: row['lower'])
                actual_policy = next(p for p in all_policies if p['channel'] == chosen['channel']
                                     and p['proxy_map_by_outcome'] == chosen['proxy_map_by_outcome'])
                selected_value = actual_policy['actual_value_interval']
                lower = max(row['lower'] for row in channels)
                upper = max(row['upper'] for row in channels)
                assert selected_value[0] >= lower and exact_optimum[1] <= upper
                certified_gap = upper - lower
                regret = (max(F(0), exact_optimum[0] - selected_value[1]),
                          max(F(0), exact_optimum[1] - selected_value[0]))
                assert 0 <= regret[0] <= regret[1] <= certified_gap <= max(epsilons)
                if panel == 'rank_reversal':
                    if degree == 1:
                        assert chosen['proxy_map_by_outcome'] == (1,) and regret[0] > 0
                    else:
                        assert chosen['proxy_map_by_outcome'] == (0,)
                if panel == 'same_F':
                    assert chosen['channel'] == 'soft_A'
                if panel == 'same_matrix_tie':
                    assert all(b == 0 for b in chosen['proxy_map_by_outcome'])
                results.append({'panel': panel, 'degree': degree,
                    'number_of_exact_branch_policies_enumerated': len(all_policies),
                    'chosen_moment_policy': chosen,
                    'selected_actual_policy_value_interval': selected_value,
                    'exact_optimal_policy_value_interval': exact_optimum,
                    'actual_selected_policy_regret_interval': regret,
                    'decision_upper_minus_lower_certificate': certified_gap,
                    'uniform_epsilon_max': max(epsilons), 'charged_cost': F(2)})
        return {'panels': len(PANELS), 'same_sigma_cost_and_hard_budget': True,
                'all_actual_selected_policy_certificates': results,
                'scope': 'Pure R-first then one proxy family; not unrestricted or whole-budget POMDP optimality'}

    def near_one_and_rank_counterexample():
        case = next(row for row in GAIN_CASES if row[0] == 'near_one')
        near = certified_gain(tuple(map(F, case[1])), tuple(map(F, case[2])), 8)
        tolerance = F(1, 1000)
        assert near['actual_truncation_error_interval'][0] > tolerance
        assert near['analytic_tail_upper_bound'] > tolerance
        # Exact inequalities proving the predeclared D1 reversal, independently
        # of interval-based policy enumeration above.
        true_b1_lower = F(1, 8) * F(9, 16) ** 2
        true_b2_upper = F(1, 64)
        assert true_b1_lower == F(81, 2048) > true_b2_upper
        assert moment_gain((F(3, 4), F(-3, 4)), (F(1, 2), F(1, 2)), 1) == 0
        assert moment_gain((F(1, 8), F(1, 8)), (F(1, 2), F(1, 2)), 1) > 0
        return {'near_one': near, 'predeclared_scaled_tolerance': tolerance,
                'degree_cap8_is_insufficient_for_this_near_one_case': True,
                'increase_degree_or_exact_fallback_needed_not_executed_here': True,
                'rank_reversal_b1_full_lower': true_b1_lower, 'b2_full_upper': true_b2_upper,
                'low_first_moment_is_not_a_no_information_certificate': True}

    for name, function in (('coefficient_and_integer_sqrt_arithmetic', coefficient_and_sqrt_arithmetic),
                           ('fixed_gain_positive_truncation_and_tails', fixed_gain_bounds),
                           ('likelihood_branches_and_same_F_counterexample', branch_and_same_F_semantics),
                           ('enumerated_actual_adaptive_policy_value_certificates', adaptive_policy_certificates),
                           ('near_one_and_degree1_rank_reversal', near_one_and_rank_counterexample)):
        check(name, function)
    unchanged = sha(__file__) == plan['audit_script_sha256']
    report = {'actual_end_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if all(r['status'] == 'PASS' for r in reports) and unchanged else 'FAIL',
        'plan_sha256': sha(args.plan), 'source_unchanged': unchanged, 'families': reports,
        'arithmetic': 'Python integers/Fraction only; no float, NumPy, SciPy or producer import',
        'root_precision': 'integer-isqrt rational enclosure at10^60 with exact endpoint collapse',
        'model_kernel_formula_assumed': 'Previously derived centered Gaussian-copula CRPS kernel; not a repeated CDF integration test',
        'producer_imports_runs_and_material_reads_fits_or_scores': 0,
        'new_material_evidence_or_new_algorithm_proof': False,
        'scope': 'Bounded mathematical audit; no performance benchmarking or universal/posterior-compression guarantee'}
    write(out / 'INDEPENDENT_MOMENT_MATH_AUDIT.json', report)
    print(json.dumps({'status': report['status'], 'families': len(reports), 'producer_imports': 0}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
