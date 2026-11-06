"""Independently reconstruct saved finite mixed-measure ledger paths, no SUT imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import traceback

import oracle_math as oracle

FIELDS = ('A', 'G', 'Y')
ROWS = {'r0': {'A': .7, 'G': 0., 'Y': 1.2}, 'r1': {'A': -.4, 'G': -.6, 'Y': .1}}
ALIASES = {'duplicate_report': 'r0', 'rowA': 'r0', 'rowB': 'r1', 'terminal': 'held_out', 'terminal_report': 'held_out', 'held_out': 'held_out'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError('Duplicate JSON key: ' + k)
            d[k] = v
        return d
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(Path(path).read_text('utf-8-sig'), object_pairs_hook=unique, parse_constant=invalid)


def require(ok, message):
    if not ok:
        raise AssertionError(message)


class Checks:
    def __init__(self, criteria):
        self.criteria, self.counts, self.maxerr = criteria, {}, {}
    def exact(self, name, actual, expected):
        require(type(actual) == type(expected) and actual == expected, name + ' exact mismatch')
        self.counts[name] = self.counts.get(name, 0) + 1
    def number(self, name, a, b, posterior=False):
        base = 'posterior_weights' if posterior else 'log_likelihood'
        atol, rtol = self.criteria[base + '_abs'], self.criteria[base + '_rel']
        require(math.isfinite(a) and math.isfinite(b), name + ' nonfinite')
        error = abs(a - b)
        require(error <= atol + rtol * abs(b), f'{name}: {a} vs {b}, error={error}')
        self.counts[name] = self.counts.get(name, 0) + 1
        self.maxerr[name] = max(self.maxerr.get(name, 0), error)
    def vector(self, name, actual, expected, posterior=False):
        require(len(actual) == len(expected) == 4, name + ' expected four hypotheses')
        for a, b in zip(actual, expected):
            self.number(name, a, b, posterior)


def observed_rows(state):
    cells = {}
    for cell in state['observations']:
        row, field, value = cell['row'], cell['field'], cell['value']
        require(row in ROWS and field in FIELDS and type(value) in (int, float) and math.isfinite(value),
                'Canonical finite saved observations required')
        require((row, field) not in cells, 'Duplicate canonical cell in state')
        cells[(row, field)] = value
    rows = {}
    for (row, field), value in cells.items():
        rows.setdefault(row, {})[field] = value
    return cells, rows


def audit_state(state, checks):
    cells, rows = observed_rows(state)
    checks.exact('canonical_cell_cost', state['total_cost'], len(cells))
    reference = oracle.posterior(rows)
    checks.vector('state_log_likelihood_direct_marginals', state['log_likelihood_per_h'], reference['cumulative_log_likelihood'])
    checks.vector('state_posterior_direct_marginals', state['weights'], reference['weights'], posterior=True)
    require(all(0 <= w <= 1 for w in state['weights']) and abs(math.fsum(state['weights']) - 1)
            <= checks.criteria['simplex_mass_abs'], 'Weights not on simplex')
    return cells, reference


def evidence(loglik):
    terms = [math.log(p) + ell for p, ell in zip(oracle.PRIOR, loglik)]
    pivot = max(terms)
    return pivot + math.log(math.fsum(math.exp(x - pivot) for x in terms))


def canonical_new(request, old):
    pending = {}
    for row, field, value in request:
        row = ALIASES.get(row, row)
        require(row in ROWS and field in FIELDS and type(value) in (int, float) and math.isfinite(value),
                'Expected valid request in successful trajectory')
        key = (row, field)
        if key in pending:
            require(pending[key] == value, 'Within-panel conflict cannot be a valid action')
        if key in old:
            require(old[key] == value, 'Existing-cell conflict cannot be a valid action')
        else:
            pending[key] = value
    return pending


def audit_step(step, checks):
    before, ref0 = audit_state(step['before'], checks)
    after, ref1 = audit_state(step['after'], checks)
    pending = canonical_new(step['request'], before)
    checks.exact('step_new_cell_cost', step['incremental_cost'], len(pending))
    checks.exact('transaction_state_union', after, before | pending)
    newcells = {(r, f): v for r, f, v in step['new_cells']}
    checks.exact('canonical_new_cell_identity', newcells, pending)
    increment = [b - a for a, b in zip(ref0['cumulative_log_likelihood'], ref1['cumulative_log_likelihood'])]
    checks.vector('incremental_log_likelihood_direct_ratio', step['incremental_log_likelihood_per_h'], increment)
    checks.number('incremental_log_evidence_direct_ratio', step['incremental_log_evidence'],
                  evidence(ref1['cumulative_log_likelihood']) - evidence(ref0['cumulative_log_likelihood']))
    if not pending:
        checks.exact('repeat_empty_or_stop_state_exact', step['after'], step['before'])
        checks.exact('repeat_no_evidence_exact', step['incremental_log_likelihood_per_h'], [0., 0., 0., 0.])
        checks.exact('repeat_no_evidence_normalizer_exact', step['incremental_log_evidence'], 0.)
    return pending


def audit_trajectory(trajectory, expected_rows, checks):
    requested = trajectory['requested_cells']
    require(len(requested) == len(trajectory['steps']) == 6, 'Six actual single-cell steps required')
    expected = {(row, field): value for row, values in expected_rows.items() for field, value in values.items()}
    checks.exact('six_cell_order_inventory', {(r, f): v for r, f, v in requested}, expected)
    for index, step in enumerate(trajectory['steps']):
        checks.exact('step_requested_order', step['request'], [requested[index]])
        if index == 0:
            checks.exact('fresh_trajectory_empty', step['before']['observations'], [])
        else:
            checks.exact('adjacent_state_continuity', step['before'], trajectory['steps'][index - 1]['after'])
        audit_step(step, checks)
    checks.exact('final_state_matches_last_step', trajectory['final_state'], trajectory['steps'][-1]['after'])
    cells, final = audit_state(trajectory['final_state'], checks)
    checks.exact('final_all6cells', cells, expected)
    batch = trajectory['direct_batch']
    checks.vector('direct_batch_likelihood_independent', batch['log_likelihood_per_h'], final['cumulative_log_likelihood'])
    checks.vector('direct_batch_posterior_independent', batch['weights'], final['weights'], posterior=True)
    return tuple((r, f, v) for r, f, v in requested)


def behavior_spec():
    first = [['r0', 'A', .7]]
    full = [[r, f, v] for r, values in ROWS.items() for f, v in values.items()]
    return [
        ('same_cell_same_value', first, first, False, 0),
        ('alias_repeat_after_native', first, [['duplicate_report', 'A', .7]], False, 0),
        ('same_panel_matching_aliases', [], [['rowA', 'A', .7], ['duplicate_report', 'A', .7]], False, 1),
        ('full_panel_repeat', full, full, False, 0),
        ('full_panel_with_three_alias_duplicates', [], full + [['duplicate_report', 'A', .7], ['duplicate_report', 'G', 0.], ['rowA', 'Y', 1.2]], False, 6),
        ('partially_known_panel', full[:2], full, False, 4),
        ('existing_conflict_atomic_after_legal_cell', first, [['r1', 'Y', .1], ['rowA', 'A', .8]], True, None),
        ('contradictory_aliases_in_one_panel', [], [['rowA', 'A', .7], ['duplicate_report', 'A', .8]], True, None),
        ('unknown_field_atomic_after_legal_cell', [], [['r0', 'A', .7], ['r1', 'Z', .1]], True, None),
        ('unknown_row_atomic_after_legal_cell', [], [['r0', 'A', .7], ['missing', 'Y', .1]], True, None),
        ('nonfinite_nan_atomic_after_legal_cell', [], [['r0', 'A', .7], ['r1', 'Y', float('nan')]], True, None),
        ('nonfinite_posinf', first, [['r1', 'G', float('inf')]], True, None),
        ('nonfinite_neginf', first, [['r1', 'G', -float('inf')]], True, None),
        ('terminal_atomic_after_legal_cell', [], [['r0', 'A', .7], ['terminal', 'Y', .1]], True, None),
        ('terminal_alias_atomic_after_legal_cell', first, [['r1', 'Y', .1], ['terminal_report', 'A', .1]], True, None),
        ('negative_and_positive_zero_same_atom', [['r0', 'G', 0.]], [['rowA', 'G', -0.]], False, 0),
        ('empty_STOP_after_partial', first, [], False, 0),
    ]


def audit_behaviors(cases, checks):
    expected = behavior_spec()
    require(len(cases) == len(expected) == 17, 'All17fixed transaction cases required')
    for case, (name, initial, request, reject, cost) in zip(cases, expected):
        checks.exact('behavior_name', case['name'], name)
        checks.exact('requested_invalid_display', case['request_repr'], [[repr(r), repr(f), repr(v)] for r, f, v in request])
        checks.exact('expected_reject_locked', case['expected_reject'], reject)
        checks.exact('actual_reject', case['actual_reject'], reject)
        checks.exact('expected_cost_locked', case['expected_incremental_cost'], cost)
        before, _ = audit_state(case['before'], checks)
        after, _ = audit_state(case['after'], checks)
        checks.exact('setup_canonical_cells', before, {(ALIASES.get(r, r), f): v for r, f, v in initial})
        if initial:
            setup = case['setup']
            require(setup is not None, 'Setup response absent')
            audit_step(dict(setup, request=initial), checks)
            checks.exact('setup_state_matches_behavior_before', setup['after'], case['before'])
        else:
            checks.exact('no_setup_response', case['setup'], None)
        if reject:
            checks.exact('rejection_has_no_response', case['response'], None)
            checks.exact('rejection_state_exactly_unchanged', case['before'], case['after'])
            checks.exact('rejection_ValueError', case['error']['type'], 'ValueError')
        else:
            checks.exact('accepted_no_error', case['error'], None)
            response = case['response']
            checks.exact('accepted_before_matches', response['before'], case['before'])
            checks.exact('accepted_after_matches', response['after'], case['after'])
            audit_step(dict(response, request=request), checks)
        checks.exact('unchanged_flag', case['state_exactly_unchanged'], case['before'] == case['after'])


def audit(plan, checks):
    for name, digest in plan['independent_source_sha256'].items():
        require(sha(Path(__file__).parent / name) == digest, 'Independent source changed: ' + name)
    require(sha(plan['criteria']['file']) == plan['criteria']['sha256'], 'Preoutput criteria changed')
    loaded = {}
    for name, pin in plan['inputs'].items():
        require(sha(pin['file']) == pin['sha256'], 'Bound producer input changed: ' + name)
        loaded[name] = read(pin['file'])
    protocol, start, done, result = [loaded[k] for k in ('producer_plan', 'execution_start', 'execution_complete', 'producer_result')]
    require(protocol['status'] == 'REGISTERED' and protocol['kind'] == 'synthetic_mixed_measure_canonical_cell_gate', 'Registered producer gate required')
    for name, digest in protocol['source_sha256'].items():
        require(sha(Path(plan['producer_code_directory']) / name) == digest, 'Producer source changed')
    require(start['plan_sha256'] == done['plan_sha256'] == result['plan_sha256'] == plan['inputs']['producer_plan']['sha256'], 'Producer plan binding mismatch')
    require(start['source_sha256'] == done['source_sha256'] == result['source_sha256'] == protocol['source_sha256'], 'Producer source binding mismatch')
    require(done['files']['RESULTS.json'] == plan['inputs']['producer_result']['sha256']
            and done['files']['EXECUTION_START.json'] == plan['inputs']['execution_start']['sha256'], 'Completion receipt mismatch')
    require(result['status'] == done['status'] == 'PASS_SYNTHETIC_MIXED_MEASURE_CANONICAL_CELL_GATE', 'Producer math gate not complete')
    require(datetime.fromisoformat(protocol['registered_actual_utc']) <= datetime.fromisoformat(start['actual_utc'])
            <= datetime.fromisoformat(result['actual_end_utc']) <= datetime.fromisoformat(done['actual_utc']), 'Producer chronology mismatch')
    require(result['configuration'] == protocol['configuration'], 'Actual fixture config differs')
    config = result['configuration']
    checks.exact('four_prior_weights', config['prior_h'], list(oracle.PRIOR))
    checks.exact('four_atom_probabilities', config['gap_atom_probability_by_h'], list(oracle.P0))
    checks.exact('four_rhos', config['rho_by_h'], list(oracle.RHO))
    checks.exact('terminal_canonical_identity', config['terminal_rows'], ['held_out'])
    require(config['canonical_aliases']['terminal'] == config['canonical_aliases']['terminal_report'] == 'held_out', 'Terminal aliases not rejected consistently')
    main = result['main_720_orders']
    require(len(main) == 720 and [r['order_index'] for r in main] == list(range(720)), 'All720ordered cases required')
    allorders = []
    for trajectory in main:
        allorders.append(audit_trajectory(trajectory, ROWS, checks))
    canonical = tuple((r, f, v) for r, values in ROWS.items() for f, v in values.items())
    checks.exact('all720distinct_sixcell_permutations', set(allorders), set(itertools.permutations(canonical)))
    positive = result['positive_gap_variant_6_orders']
    prows = {r: dict(values) for r, values in ROWS.items()}
    prows['r1']['G'] = .6
    require(len(positive) == 6 and [r['order_index'] for r in positive] == list(range(6)), 'All6positiveG orders required')
    expected_positive = [tuple((r, f, prows[r][f]) for f in fields for r in ROWS) for fields in itertools.permutations(FIELDS)]
    for trajectory, expected in zip(positive, expected_positive):
        checks.exact('positive_predefined_order', audit_trajectory(trajectory, prows, checks), expected)
    audit_behaviors(result['behavior_cases'], checks)
    controls = result['atom_and_marginal_controls']
    expected_controls = [{}, {'G': 0.}, {'G': -.6}, {'G': .6}, {'A': .7, 'Y': 1.2},
                         {'A': .7, 'G': 0., 'Y': 1.2}, {'A': -.4, 'G': -.6, 'Y': .1}]
    require(len(controls) == len(expected_controls), 'Seven fixed atom/negative/positive/missingG controls required')
    for c, observed in zip(controls, expected_controls):
        checks.exact('atom_fixture_observed_subset', c['observed'], observed)
        ref = oracle.posterior({'r0': observed})
        checks.vector('atom_direct_marginal_likelihood', c['log_likelihood_per_h'], ref['cumulative_log_likelihood'])
        checks.vector('atom_direct_marginal_posterior', c['weights_for_one_row'], ref['weights'], posterior=True)
    exact = oracle.posterior(ROWS)
    naive = [math.fsum(math.log(oracle.row_likelihood(h, {f: v})) for r, f, v in canonical) for h in range(4)]
    terms = [math.log(w) + x for w, x in zip(oracle.PRIOR, naive)]
    z = max(terms)
    wrongw = [math.exp(x - z) / math.fsum(math.exp(t - z) for t in terms) for x in terms]
    contrast = result['naive_product_control']
    checks.vector('true_joint_negative_control_loglik', contrast['correct_within_row_joint']['log_likelihood_per_h'], exact['cumulative_log_likelihood'])
    checks.vector('true_joint_negative_control_posterior', contrast['correct_within_row_joint']['weights'], exact['weights'], posterior=True)
    checks.vector('wrong_marginal_product_loglik', contrast['wrong_product_one_cell_marginals']['log_likelihood_per_h'], naive)
    checks.vector('wrong_marginal_product_posterior', contrast['wrong_product_one_cell_marginals']['weights'], wrongw, posterior=True)
    difference = max(abs(a - b) for a, b in zip(wrongw, exact['weights']))
    require(difference > 1e-11, 'Dependent panel did not distinguish wrong marginal product')
    checks.number('intentional_wrong_control_difference', contrast['max_abs_posterior_difference'], difference)
    summary = result['summary']
    for key, value in {'main_order_cases': 720, 'main_steps': 4320, 'positive_order_cases': 6, 'positive_steps': 36,
                       'transaction_cases': 17, 'independent_material_observations': 0, 'all_six_new_native_cells_cost': 6,
                       'counterfactual_order_cases_not_independent_materials': True,
                       'branch_parameter_estimation_or_material_performance_claim': False}.items():
        checks.exact('summary_' + key, summary[key], value)
    return {'status': 'PASS', 'all_main_orders': 720, 'all_main_steps': 4320, 'all_positive_orders': 6,
            'all_positive_steps': 36, 'transaction_behavior_cases': 17, 'atom_marginal_controls': 7,
            'wrong_independent_product_posterior_max_difference': difference,
            'source_data_or_material_model_fits': 0, 'SUT_imports_or_original_calculation_replays': 0,
            'scope': 'SciPy observed-submatrix direct marginal oracle on saved finite-h synthetic paths; no training, target prediction scoring, continuous-posterior approximation or material atom identification.'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    p.add_argument('--approval', required=True)
    p.add_argument('--output-dir', required=True)
    a = p.parse_args()
    plan, approval = read(a.plan), read(a.approval)
    source = sha(__file__)
    require(plan['independent_source_sha256']['audit_mixed_measure.py'] == source, 'Independent auditor source changed')
    require(approval.get('approved') is True and approval['scope'] == 'INDEPENDENT_MIXED_MEASURE_GATE_ONCE'
            and approval['plan_sha256'] == sha(a.plan) and approval['script_sha256'] == source, 'Exact root audit approval required')
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = datetime.now(timezone.utc).isoformat()
    criteria = read(plan['criteria']['file'])['numerical_tolerance']
    checks = Checks(criteria)
    try:
        result = audit(plan, checks)
    except Exception as error:
        result = {'status': 'FAIL', 'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    result.update({'start_utc': start, 'end_utc': datetime.now(timezone.utc).isoformat(), 'script_sha256': source,
                   'plan_sha256': sha(a.plan), 'approval_sha256': sha(a.approval), 'comparison_counts': checks.counts,
                   'max_absolute_errors': checks.maxerr, 'random_samples_or_new_measurements': 0})
    f = output / 'INDEPENDENT_MIXED_MEASURE_GATE_AUDIT.json'
    f.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'result': str(f), 'sha256': sha(f)}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
