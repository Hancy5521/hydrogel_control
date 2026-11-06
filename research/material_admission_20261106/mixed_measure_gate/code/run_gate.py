"""One registered synthetic mixed-measure / canonical-cell mechanism gate.

No material data, fitted predictor, stochastic sample, or autonomous agent is used.
All720 six-cell orders and all rejection controls are retained as calculation cases.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

import mixed_likelihood as likelihood

CODE = ('mixed_likelihood.py', 'run_gate.py')
SCOPE = 'MIXED_MEASURE_CANONICAL_LEDGER_MATH_GATE'
TOLERANCE = 1e-11
MAIN_CELLS = (('r0', 'A', .7), ('r0', 'G', 0.), ('r0', 'Y', 1.2),
              ('r1', 'A', -.4), ('r1', 'G', -.6), ('r1', 'Y', .1))


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sources():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in CODE}


def configuration():
    return {'version': 'fixed_mixed_measure_cell_ledger_v1',
        'prior_h': [.1, .2, .3, .4], 'gap_atom_probability_by_h': [.1, .3, .6, .8],
        'rho_by_h': [-.6, -.2, .25, .65], 'h_indices': [0, 1, 2, 3],
        'zero_branch': {'coordinates': ['A', 'Y'], 'mean': ['.2*h', '-.1*h'],
                        'covariance': '[[1,rho],[rho,1]]', 'G': 'point mass at zero'},
        'slab_branch': {'coordinates': ['A', 'G', 'Y'], 'mean': ['.2*h', '.3-.1*h', '-.1*h'],
                        'covariance': 'rho**abs(i-j)', 'G_domain': 'all real, including negative values'},
        'base_measure': likelihood.BASE_MEASURE,
        'row_dependence': 'conditionally independent across r0/r1 given h; correlated fields within row',
        'canonical_aliases': dict(likelihood.ALIAS_ITEMS), 'terminal_rows': list(likelihood.TERMINAL_ROWS),
        'cell_cost': 1, 'same_cell_rule': 'exact numeric equality; same value zero evidence and cost; conflict atomic reject',
        'empty_panel_rule': 'legal STOP, exactly zero evidence/cost and unchanged state',
        'main_cells': [list(cell) for cell in MAIN_CELLS], 'main_orders': 720, 'main_steps': 4320,
        'positive_variant': 'only r1G=-.6 becomes+.6; six permutations of field order, each field r0 thenr1',
        'positive_orders': 6, 'positive_steps': 36,
        'behavior_case_count': 17, 'absolute_max_norm_tolerance': TOLERANCE,
        'naive_wrong_control': 'product of six one-cell marginals versus proper within-row joint density',
        'producer_density': 'NumPy Cholesky and solve; no SciPy',
        'real_material_data_access': 0, 'fits': 0, 'agent_runs': 0,
        'scope': 'fixed known-mechanism mathematical invariance checks, not new material evidence or a new algorithm'}


def validate(plan_path, approval_path):
    plan, approval = read(plan_path), read(approval_path)
    if (set(plan) != {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'configuration'}
            or plan['kind'] != 'synthetic_mixed_measure_canonical_cell_gate'
            or plan['status'] != 'REGISTERED' or not isinstance(plan['registered_actual_utc'], str)
            or not plan['registered_actual_utc'] or plan['source_sha256'] != sources()
            or plan['configuration'] != configuration()):
        raise ValueError('exact registered fixed source and configuration required')
    registered = datetime.fromisoformat(plan['registered_actual_utc'])
    if registered.tzinfo is None or registered > datetime.now(timezone.utc):
        raise ValueError('actual timezone-aware registration before execution required')
    if (approval.get('approved') is not True or approval.get('scope') != SCOPE
            or approval.get('plan_sha256') != sha(plan_path)
            or 'math_gate' not in approval.get('allowed_phases', [])):
        raise ValueError('root approval of exact registered plan and math_gate phase required')
    return plan


def observed_from_cells(cells):
    result = {}
    for row, field, value in cells:
        result.setdefault(row, {})[field] = value
    return result


def difference(left, right):
    return max((abs(x - y) for x, y in zip(left, right)), default=0.)


def batch_reference(cells):
    log_likelihood = likelihood.joint_log_likelihood_by_h(observed_from_cells(cells))
    return {'log_likelihood_per_h': log_likelihood, 'weights': likelihood.posterior(log_likelihood),
            'total_cost': len({(row, field) for row, field, _ in cells})}


def order_case(index, cells):
    ledger = likelihood.CanonicalCellLedger()
    steps = []
    accumulated = [0.] * 4
    for cell in cells:
        step = ledger.apply_panel([cell])
        step['request'] = [list(cell)]
        for h in range(4):
            accumulated[h] += step['incremental_log_likelihood_per_h'][h]
        steps.append(step)
        if step['incremental_cost'] != 1:
            raise AssertionError('main or positive order unexpectedly repeated a physical cell')
    final = ledger.snapshot()
    direct = batch_reference(cells)
    residual = {'posterior': difference(final['weights'], direct['weights']),
                'joint_log_likelihood': difference(final['log_likelihood_per_h'], direct['log_likelihood_per_h']),
                'telescoped_log_likelihood': difference(accumulated, direct['log_likelihood_per_h'])}
    if max(residual.values()) > TOLERANCE or final['total_cost'] != 6:
        raise AssertionError('known synthetic sequential-to-batch or unique-cost identity failed')
    return {'order_index': index, 'requested_cells': [list(cell) for cell in cells], 'steps': steps,
            'final_state': final, 'direct_batch': direct, 'max_abs_residual': residual}


def safe_display(cells):
    # Invalid request diagnostics must remain strict JSON, never NaN/Infinity literals.
    return [[repr(row), repr(field), repr(value)] for row, field, value in cells]


def behavior_case(name, initial, request, expected_reject, expected_incremental_cost=None):
    ledger = likelihood.CanonicalCellLedger()
    setup = ledger.apply_panel(initial) if initial else None
    before = ledger.snapshot()
    try:
        response = ledger.apply_panel(request)
        rejected = False
        error = None
    except ValueError as problem:
        response = None
        rejected = True
        error = {'type': type(problem).__name__, 'message': str(problem)}
    after = ledger.snapshot()
    if rejected != expected_reject:
        raise AssertionError('unexpected acceptance/rejection for ' + name)
    if rejected and before != after:
        raise AssertionError('rejected transaction mutated state for ' + name)
    if not rejected and response['incremental_cost'] != expected_incremental_cost:
        raise AssertionError('incorrect unique-cell price for ' + name)
    if not rejected and expected_incremental_cost == 0:
        if before != after or response['incremental_log_likelihood_per_h'] != [0.] * 4:
            raise AssertionError('duplicate is not exactly idempotent for ' + name)
        if response['incremental_log_evidence'] != 0.:
            raise AssertionError('duplicate has nonzero evidence increment for ' + name)
    return {'name': name, 'expected_reject': expected_reject, 'actual_reject': rejected,
            'expected_incremental_cost': expected_incremental_cost, 'setup': setup,
            'request_repr': safe_display(request), 'before': before, 'after': after,
            'response': response, 'error': error, 'state_exactly_unchanged': before == after}


def behavior_cases():
    first = [MAIN_CELLS[0]]
    all_cells = list(MAIN_CELLS)
    return [
        behavior_case('same_cell_same_value', first, first, False, 0),
        behavior_case('alias_repeat_after_native', first, [('duplicate_report', 'A', .7)], False, 0),
        behavior_case('same_panel_matching_aliases', [], [('rowA', 'A', .7), ('duplicate_report', 'A', .7)], False, 1),
        behavior_case('full_panel_repeat', all_cells, all_cells, False, 0),
        behavior_case('full_panel_with_three_alias_duplicates', [], all_cells +
                      [('duplicate_report', 'A', .7), ('duplicate_report', 'G', 0.), ('rowA', 'Y', 1.2)], False, 6),
        behavior_case('partially_known_panel', all_cells[:2], all_cells, False, 4),
        behavior_case('existing_conflict_atomic_after_legal_cell', first,
                      [('r1', 'Y', .1), ('rowA', 'A', .8)], True),
        behavior_case('contradictory_aliases_in_one_panel', [],
                      [('rowA', 'A', .7), ('duplicate_report', 'A', .8)], True),
        behavior_case('unknown_field_atomic_after_legal_cell', [], [('r0', 'A', .7), ('r1', 'Z', .1)], True),
        behavior_case('unknown_row_atomic_after_legal_cell', [], [('r0', 'A', .7), ('missing', 'Y', .1)], True),
        behavior_case('nonfinite_nan_atomic_after_legal_cell', [], [('r0', 'A', .7), ('r1', 'Y', float('nan'))], True),
        behavior_case('nonfinite_posinf', first, [('r1', 'G', float('inf'))], True),
        behavior_case('nonfinite_neginf', first, [('r1', 'G', -float('inf'))], True),
        behavior_case('terminal_atomic_after_legal_cell', [], [('r0', 'A', .7), ('terminal', 'Y', .1)], True),
        behavior_case('terminal_alias_atomic_after_legal_cell', first,
                      [('r1', 'Y', .1), ('terminal_report', 'A', .1)], True),
        behavior_case('negative_and_positive_zero_same_atom', [('r0', 'G', 0.)], [('rowA', 'G', -0.)], False, 0),
        behavior_case('empty_STOP_after_partial', first, [], False, 0),
    ]


def atomic_and_marginal_controls():
    observations = [{}, {'G': 0.}, {'G': -.6}, {'G': .6}, {'A': .7, 'Y': 1.2},
                    {'A': .7, 'G': 0., 'Y': 1.2}, {'A': -.4, 'G': -.6, 'Y': .1}]
    records = []
    for observed in observations:
        log_likelihood = [likelihood.partial_log_likelihood(observed, h) for h in range(4)]
        records.append({'observed': observed, 'log_likelihood_per_h': log_likelihood,
                        'weights_for_one_row': likelihood.posterior(log_likelihood)})
    empty = records[0]['log_likelihood_per_h']
    atom = records[1]['log_likelihood_per_h']
    if empty != [0.] * 4 or difference(atom, [math.log(p) for p in likelihood.P_ZERO]) > TOLERANCE:
        raise AssertionError('empty or atom base-measure control failed')
    return records


def naive_product_control():
    correct = batch_reference(MAIN_CELLS)
    incorrect = [math.fsum(likelihood.partial_log_likelihood({field: value}, h)
                          for _, field, value in MAIN_CELLS) for h in range(4)]
    wrong_weights = likelihood.posterior(incorrect)
    gap = difference(correct['weights'], wrong_weights)
    if gap <= TOLERANCE:
        raise AssertionError('fixed naive-independence negative control did not separate')
    return {'status': 'INTENTIONAL_WRONG_CONTROL_NOT_A_SUPPORTED_MODEL',
        'correct_within_row_joint': correct,
        'wrong_product_one_cell_marginals': {'log_likelihood_per_h': incorrect, 'weights': wrong_weights},
        'max_abs_posterior_difference': gap,
        'meaning': 'naively multiplying dependent cell marginals changes relation posterior; no material advantage claimed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan = validate(args.plan, args.approval)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'plan_sha256': sha(args.plan),
        'approval_sha256': sha(args.approval), 'source_sha256': sources(),
        'real_material_data_or_model_fit': 0, 'RNG_calls': 0, 'mechanism': 'known synthetic parameters'})
    try:
        orders = [order_case(i, list(cells)) for i, cells in enumerate(itertools.permutations(MAIN_CELLS))]
        positive_values = {(row, field): (.6 if (row, field) == ('r1', 'G') else value)
                           for row, field, value in MAIN_CELLS}
        positive = [order_case(i, [(row, field, positive_values[(row, field)])
                                 for field in fields for row in likelihood.ROWS])
                    for i, fields in enumerate(itertools.permutations(likelihood.FIELDS))]
        if len(orders) != 720 or len(positive) != 6:
            raise AssertionError('fixed order count changed')
        reference = orders[0]['direct_batch']['weights']
        maximum = max(difference(row['final_state']['weights'], reference) for row in orders)
        if maximum > TOLERANCE:
            raise AssertionError('order invariance failed')
        behavior = behavior_cases()
        if len(behavior) != configuration()['behavior_case_count']:
            raise AssertionError('fixed behavior case count changed')
        result = {'status': 'PASS_SYNTHETIC_MIXED_MEASURE_CANONICAL_CELL_GATE',
            'actual_end_utc': utc(), 'source_sha256': sources(), 'plan_sha256': sha(args.plan),
            'configuration': plan['configuration'], 'main_720_orders': orders,
            'positive_gap_variant_6_orders': positive, 'behavior_cases': behavior,
            'atom_and_marginal_controls': atomic_and_marginal_controls(),
            'naive_product_control': naive_product_control(),
            'summary': {'main_order_cases': 720, 'main_steps': 4320, 'positive_order_cases': 6,
                        'positive_steps': 36, 'transaction_cases': len(behavior),
                        'max_abs_main_order_posterior_difference': maximum,
                        'independent_material_observations': 0,
                        'counterfactual_order_cases_not_independent_materials': True,
                        'all_six_new_native_cells_cost': 6,
                        'branch_parameter_estimation_or_material_performance_claim': False}}
        if sources() != plan['source_sha256']:
            raise ValueError('registered source changed during gate')
        write(output / 'RESULTS.json', result)
        write(output / 'EXECUTION_COMPLETE.json', {'actual_utc': utc(), 'status': result['status'],
            'source_sha256': sources(), 'plan_sha256': sha(args.plan),
            'files': {name: sha(output / name) for name in ('EXECUTION_START.json', 'RESULTS.json')},
            'no_new_real_labels_or_model_fits_or_agent_runs': True})
        print(json.dumps(result['summary'], ensure_ascii=True))
    except Exception as problem:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'type': type(problem).__name__,
            'message': str(problem), 'source_sha256': sources(), 'do_not_retune_or_overwrite': True})
        raise


if __name__ == '__main__':
    main()
