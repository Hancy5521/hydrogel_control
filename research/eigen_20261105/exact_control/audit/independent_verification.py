"""Independent GF(2)/subset audit of frozen exact-control results.

No producer import, Bellman recursion, producer execution or material access.
All comparisons are exact rational equalities. Root approval is required.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import itertools
import json
from pathlib import Path

ACTIONS = ('V1', 'V2', 'R', 'E1', 'E2')
STATIC = ('R', 'E1', 'E2')
SCENARIOS = {
    'coupled': (3, {'R': 1, 'E1': 2, 'E2': 4, 'V1': 3, 'V2': 5}, (3, 5)),
    'theta_known': (2, {'R': 0, 'E1': 1, 'E2': 2, 'V1': 1, 'V2': 2}, (1, 2)),
    'fake_pair': (4, {'R': 8, 'E1': 2, 'E2': 4, 'V1': 3, 'V2': 5}, (3, 5)),
    'no_relation': (5, {'R': 1, 'E1': 2, 'E2': 4, 'V1': 8, 'V2': 16}, (8, 16)),
}
SETTINGS = {
    'cheap_repair_expansion': ({'R': 1, 'E1': 1, 'E2': 1, 'V1': 4, 'V2': 4}, 4),
    'equal_cost': (dict.fromkeys(ACTIONS, 4), 8),
}
POLICIES = ('exact_budget_DP', 'myopic_cost_normalized', 'static_RE_cycle')
OBJECTIVES = ('brier', 'verification_shortfall')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def decode(value):
    if isinstance(value, dict) and 'fraction' in value:
        result = F(value['fraction'])
        assert (result.numerator, result.denominator) == (value['numerator'], value['denominator'])
        return result
    if isinstance(value, dict):
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def ready(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(ready(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def span(vectors):
    answer = {0}
    for vector in vectors:
        answer |= {v ^ vector for v in tuple(answer)}
    return answer


def rank(vectors):
    return len(span(vectors)).bit_length() - 1


def sign(world, mask):
    result = 1
    for bit, value in enumerate(world):
        if (mask >> bit) & 1:
            result *= value
    return result


def risk(scenario, used):
    _, masks, targets = SCENARIOS[scenario]
    image = span(masks[a] for a in used)
    return F(sum(t not in image for t in targets), 8)


def metric(scenario, used, objective):
    return risk(scenario, used) if objective == 'brier' else F(2 - sum(a in used for a in ('V1', 'V2')), 2)


def subset_optimum(scenario, used, remaining, costs, objective):
    """At most32 subsets, with no recursion or imported DP implementation.

    A uniform linear-parity posterior is affine-uniform. Whether a target is
    known depends only on span, never observed signs. Thus a best subset also
    attains the adaptive optimum for THESE declared fixtures and fixed costs.
    """
    available = [a for a in ACTIONS if a not in used]
    best, winners = None, []
    for flags in itertools.product((False, True), repeat=len(available)):
        selected = tuple(a for a, yes in zip(available, flags) if yes)
        cost = sum(costs[a] for a in selected)
        if cost > remaining:
            continue
        pair = (metric(scenario, tuple(used) + selected, objective), F(cost))
        if best is None or pair < best:
            best, winners = pair, [selected]
        elif pair == best:
            winners.append(selected)
    return best, winners


def expected_action(scenario, used, remaining, costs, objective, policy):
    available = [a for a in ACTIONS if a not in used and costs[a] <= remaining]
    if policy == 'exact_budget_DP':
        best, winners = subset_optimum(scenario, used, remaining, costs, objective)
        if best[1] == 0:
            return None
        return next(a for a in available if any(a in w for w in winners))
    if policy == 'static_RE_cycle':
        return next((a for a in STATIC if a in available), None)
    current = metric(scenario, used, objective)
    choice, gain = None, F(0)
    for a in available:
        improvement = (current - metric(scenario, tuple(used) + (a,), objective)) / costs[a]
        if improvement > gain:
            gain, choice = improvement, a
    return choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    assert approval.get('approved') is True
    assert approval.get('scope') == 'EXACT_CONTROL_GF2_AUDIT'
    assert approval.get('plan_sha256') == sha(args.plan)
    assert sha(__file__) == plan['audit_script_sha256']
    pins = plan['input_sha256']
    assert all(sha(path) == value for path, value in pins.items())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'EXECUTION_START.json', {'actual_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
        'input_sha256': pins, 'producer_import_or_execution': False})
    result = decode(read(plan['frozen_results_file']))
    records = result['policies']
    reports = []

    def check(name, function):
        try:
            item = {'family': name, 'status': 'PASS', 'details': function()}
        except Exception as error:
            item = {'family': name, 'status': 'FAIL', 'error_type': type(error).__name__, 'message': str(error)}
        reports.append(item)
        with (out / 'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(ready(item), ensure_ascii=False) + '\n')

    def configuration():
        keys = [(r['scenario'], r['cost_setting'], r['objective'], r['policy']) for r in records]
        expected = set(itertools.product(SCENARIOS, SETTINGS, OBJECTIVES, POLICIES))
        assert len(keys) == 48 and len(set(keys)) == 48 and set(keys) == expected
        assert result['policy_configurations'] == 48
        assert result['new_material_measurements_fits_predictions_or_target_scores'] == 0
        assert result['source_sha256'] == plan['producer_source_sha256']
        assert result['plan_sha256'] == plan['producer_plan_sha256']
        return {'complete_configurations': 48, 'producer_not_imported': True}

    def all_leaf_and_optimum_checks():
        leaves_checked, prefixes_checked = 0, 0
        for row in records:
            scenario, objective, policy = row['scenario'], row['objective'], row['policy']
            dim, masks, targets = SCENARIOS[scenario]
            worlds = list(itertools.product((-1, 1), repeat=dim))
            costs, budget = SETTINGS[row['cost_setting']]
            assert row['world_count'] == len(worlds) and row['costs_ticks'] == costs
            assert row['budget_ticks'] == budget and row['tick_cost'] == F(1, 4)
            totals, coverage = Counter(), Counter()
            for leaf in row['leaves']:
                used, remaining, ids = (), budget, list(range(len(worlds)))
                for event in leaf['history']:
                    a = expected_action(scenario, used, remaining, costs, objective, policy)
                    assert a == event['action'] and event['cost_ticks'] == costs[a]
                    assert a not in used and event['observation'] in (-1, 1)
                    ids = [i for i in ids if sign(worlds[i], masks[a]) == event['observation']]
                    assert ids
                    used, remaining = used + (a,), remaining - costs[a]
                    prefixes_checked += 1
                assert expected_action(scenario, used, remaining, costs, objective, policy) is None
                assert list(used) == leaf['used'] and ids == leaf['belief_ids']
                probability = F(len(ids), len(worlds))
                assert leaf['probability'] == probability == F(1, 2 ** rank(masks[a] for a in used))
                coverage.update(ids)
                q = [F(sum(sign(worlds[i], t) == 1 for i in ids), len(ids)) for t in targets]
                assert leaf['posterior_positive_probabilities'] == q
                inferred = sum(t in span(masks[a] for a in used) and 'V' + str(j + 1) not in used
                               for j, t in enumerate(targets))
                expected = {'bayes_risk': risk(scenario, used),
                    'objective_value': metric(scenario, used, objective), 'spent_ticks': budget - remaining,
                    'directly_verified_targets': sum(a in used for a in ('V1', 'V2')),
                    'inferred_but_unverified_targets': inferred, 'oracle_calls': len(used),
                    'independent_information_bits_revealed': F(rank(masks[a] for a in used))}
                assert sum((p * (1 - p) for p in q), F(0)) / 2 == expected['bayes_risk']
                assert remaining >= 0
                for field, value in expected.items():
                    assert leaf[field] == value, (scenario, policy, field)
                    totals[field] += probability * value
                leaves_checked += 1
            assert set(coverage) == set(range(len(worlds))) and set(coverage.values()) == {1}
            for field, value in totals.items():
                assert row['expected'][field] == value, (scenario, policy, 'expected', field)
            assert row['expected']['charged_oracle_cost'] == totals['spent_ticks'] / 4
            assert row['expected']['unmeasured_inference_counts_as_verified_label'] is False
            if policy == 'exact_budget_DP':
                best, _ = subset_optimum(scenario, (), budget, costs, objective)
                assert (totals['objective_value'], totals['spent_ticks']) == best
        return {'frozen_leaves_checked': leaves_checked, 'action_prefixes_checked': prefixes_checked,
                'independent_method': 'GF(2) affine span and <=32 feasible subsets, no Bellman recursion',
                'exact_rational_discrepancies': 0}

    def information_and_complementarity():
        rows = {r['scenario']: r for r in result['complementarity']}
        assert set(rows) == set(SCENARIOS)
        for scenario, row in rows.items():
            _, masks, targets = SCENARIOS[scenario]
            subsets = {'R': ('R',), 'E1': ('E1',), 'R_and_E1': ('R', 'E1')}
            losses = {'none': risk(scenario, ()), **{k: risk(scenario, v) for k, v in subsets.items()}}
            assert row['mean_two_target_bayes_risk'] == losses
            gamma = losses['R'] + losses['E1'] - losses['R_and_E1'] - losses['none']
            assert row['mean_brier_complementarity_gain'] == gamma == (F(1, 8) if scenario == 'coupled' else 0)
            for label, actions in subsets.items():
                observed = [masks[a] for a in actions]
                for key, target in (('Y1', targets[:1]), ('Y2', targets[1:]), ('joint', targets)):
                    mi = F(rank(observed) + rank(target) - rank(observed + list(target)))
                    assert row['target_mutual_information_bits'][label][key] == mi
            assert row['Y1_information_complementarity_bits'] == (1 if scenario == 'coupled' else 0)
            assert row['independently_measured_targets_from_R_E'] == 0
        return {'coupled_mean_two_target_complementarity': F(1, 8),
                'single_target_information_complementarity_bits': 1, 'three_negative_controls': 'zero'}

    def endpoint_counterexamples():
        def get(setting, policy, objective='brier'):
            return next(r['expected'] for r in records if (r['scenario'], r['cost_setting'], r['policy'], r['objective'])
                        == ('coupled', setting, policy, objective))
        cheap = get('cheap_repair_expansion', 'exact_budget_DP')
        myopic = get('cheap_repair_expansion', 'myopic_cost_normalized')
        assert (cheap['bayes_risk'], cheap['charged_oracle_cost'], cheap['directly_verified_targets']) == (0, F(3, 4), 0)
        assert (myopic['bayes_risk'], myopic['charged_oracle_cost'], myopic['directly_verified_targets']) == (F(1, 8), 1, 1)
        equal = get('equal_cost', 'exact_budget_DP')
        fixed = get('equal_cost', 'static_RE_cycle')
        assert (equal['bayes_risk'], equal['directly_verified_targets']) == (0, 2)
        assert (fixed['bayes_risk'], fixed['directly_verified_targets']) == (F(1, 8), 0)
        verified = get('cheap_repair_expansion', 'exact_budget_DP', 'verification_shortfall')
        assert (verified['objective_value'], verified['directly_verified_targets'], verified['bayes_risk']) == (F(1, 2), 1, F(1, 8))
        return {'cheap_completion_advantage_is_not_verified_measurement_advantage': True,
                'equal_cost_direct_verification_counterexample_preserved': True}

    def pseudo_control():
        p = result['pseudo_overconfidence']
        assert p['world_count'] == 16
        assert p['true_positive_probabilities_given_h_E1_E2'] == [F(1, 2), F(1, 2)]
        assert p['reported_wrong_posterior_bayes_risk'] == 0
        assert p['actual_wrong_posterior_mean_Brier'] == F(1, 2)
        assert p['proper_posterior_mean_Brier'] == F(1, 4)
        assert p['probability_both_confident_pseudo_targets_correct'] == F(1, 2)
        assert p['generated_h_measurement_count'] == p['directly_verified_target_count'] == 0
        assert p['charged_real_oracle_actions'] == ['E1', 'E2'] and p['cheap_charged_oracle_cost'] == F(1, 2)
        assert p['pseudo_predictions_are_not_measured_labels'] is True
        return {'analytic_reason': 'independent fair h matches theta with probability1/2; on mismatch both confident targets wrong',
                'proper_Brier': F(1, 4), 'false_certainty_actual_Brier': F(1, 2)}

    for name, function in (('configuration_and_pins', configuration),
                           ('GF2_subset_optima_and_every_frozen_leaf', all_leaf_and_optimum_checks),
                           ('rank_information_and_complementarity', information_and_complementarity),
                           ('cost_and_verified_endpoint_counterexamples', endpoint_counterexamples),
                           ('independent_pseudo_false_certainty', pseudo_control)):
        check(name, function)
    unchanged = all(sha(path) == value for path, value in pins.items())
    report = {'actual_end_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if all(r['status'] == 'PASS' for r in reports) and unchanged else 'FAIL',
        'plan_sha256': sha(args.plan), 'families': reports, 'pinned_inputs_unchanged': unchanged,
        'producer_imports_or_reexecutions': 0, 'Bellman_DP_calls': 0,
        'real_material_files_fits_or_scores': 0, 'new_material_evidence': False,
        'scope': 'Exact parity-world mathematical audit only; not a general theorem or material/algorithm novelty evidence.'}
    write(out / 'INDEPENDENT_GF2_AUDIT.json', report)
    print(json.dumps({'status': report['status'], 'families': len(reports), 'producer_reexecutions': 0}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
