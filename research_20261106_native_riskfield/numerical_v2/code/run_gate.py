"""One fixed12-evaluation native riskfield mechanism gate, never material fitting."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import activefield as field
import posterior_update as posterior

CODE = ('posterior_update.py', 'activefield.py', 'run_gate.py')
SCOPE = 'NATIVE_RISKFIELD_FIXED_MECHANISM_GATE'


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
    return {name: sha(Path(__file__).resolve().parent / name) for name in CODE}


def configuration():
    return {'version': 'fixed_native_NIG_riskfield_gate_v2_angle_query',
        'NIG': {'mean': [0., 1.], 'V': [[.2, 0.], [0., 1.]], 'alpha': 3., 'b': 1.5},
        'terminal_contexts': {'tplus': 1., 'tminus': -1.},
        'query_contexts': {'qplus': 1., 'qminus': -1.1}, 's0': 2.,
        'mechanism_cases': 8, 'extra_native_matched_Gaussian_cases': 4,
        'levels': [32, 64, 128, 256], 'outer_Lv': 32., 'query_Lz': 64.,
        'query_weights': 'originaltdensityinatanThetaInterval, no renormalization',
        'v_split': '0and-mu/scale insidefixedbounds',
        'Gaussian_kernel': 'OwenT1D, positivePlackett-angle fallback when negativeorrelativecancellation',
        'negative_kernel_hard_clamp': False,
        'final_refinement_abs_diagnostic': 2e-5, 'analytic_tail_bound_max': 1e-4,
        'independent_agreement_abs': 2e-4, 'independent_agreement_rel': 2e-5,
        'algebra_abs': 2e-10, 'algebra_rel': 2e-10,
        'NIG_conditional_z_probes': [-1., 0., 1.],
        'rho_bound_probe_existing_case_pairs': [[0, 5], [2, 5], [1, 3], [0, 4]],
        'libraries': {'numpy': '2.1.3', 'scipy': '1.15.3'},
        'no_new_material_source_or_fit_or_agent': True,
        'rank_reversal_or_speedup_required': False,
        'failure_policy': 'retainallfixedcases, no prior/context sweep; separate registered numerical repair if needed'}


def fixtures():
    config = configuration()
    state = dict(config['NIG'], labeled_updates=0)
    result = []
    for query_name, query in config['query_contexts'].items():
        for target_name, target in config['terminal_contexts'].items():
            pair = posterior.pair_law(state, [query], [target])
            result.append({'id': query_name + '_' + target_name, 'family': 'student_t',
                'mu': pair['mu_target'], 'scale': pair['scale_target'], 'rho': pair['rho'], 'nu': pair['nu'],
                's0': 2., 'loss_coordinate': 'YJ_half', 'query_id': query_name, 'target_id': target_name,
                'query_context': [query], 'target_context': [target], 'exact_NIG_pair': pair})
    mirror = copy.deepcopy(result[0]); mirror['id'] = 'case1_rho_sign_mirror'
    mirror['rho'] = -mirror['rho']; mirror['synthetic_symmetry_probe_not_another_pool_action'] = True
    result.append(mirror)
    zero = {'id': 'Studentt_rho0_native', 'family': 'student_t', 'mu': 1.,
            'scale': math.sqrt(1.1), 'rho': 0., 'nu': 6., 's0': 2., 'loss_coordinate': 'YJ_half'}
    result.append(zero)
    result.append(dict(zero, id='Gaussian_rho0_native', family='normal', nu=None,
                       same_scale_parameter_not_moment_matching_claim=True))
    result.append(dict(zero, id='Studentt_rho0_affine_target', loss_coordinate='affine'))
    for original in result[:4]:
        gaussian = copy.deepcopy(original)
        gaussian.update({'id': original['id'] + '_matchedGaussian_native', 'family': 'normal',
            'scale': original['scale'] * math.sqrt(original['nu'] / (original['nu'] - 2)), 'nu': None,
            'matched_latent_mean_covariance_not_native_moment_match': True})
        result.append(gaussian)
    return state, result


def validate(plan_path, approval_path):
    plan, approval = read(plan_path), read(approval_path)
    if (set(plan) != {'kind', 'status', 'registered_actual_utc', 'source_sha256', 'configuration', 'root_scope', 'repair_inputs'}
            or plan['kind'] != 'native_riskfield_fixed_mechanism_gate' or plan['status'] != 'REGISTERED'
            or not isinstance(plan['registered_actual_utc'], str) or not plan['registered_actual_utc']
            or plan['source_sha256'] != sources() or plan['configuration'] != configuration()
            or sha(plan['root_scope']['file']) != plan['root_scope']['sha256']):
        raise ValueError('exact registered source/configuration/root scope required')
    if set(plan['repair_inputs']) != {'V1_execution_complete', 'V1_independent_audit', 'coordinate_change_spec'}:
        raise ValueError('exactoriginalV1/independentreceipt/coordinatechange provenance required')
    for item in plan['repair_inputs'].values():
        if sha(item['file']) != item['sha256']:
            raise ValueError('numerical repair provenance changed')
    registered = datetime.fromisoformat(plan['registered_actual_utc'])
    if registered.tzinfo is None or registered > datetime.now(timezone.utc):
        raise ValueError('actual timezone-aware preexecution registration required')
    if (approval.get('approved') is not True or approval.get('scope') != SCOPE
            or approval.get('plan_sha256') != sha(plan_path)
            or 'math_gate' not in approval.get('allowed_phases', [])):
        raise ValueError('rootapprovalofexactplan/math_gatephase required')
    return plan


def rank_record(positive, negative, positive_error, negative_error):
    difference = positive - negative
    uncertainty = positive_error + negative_error
    return {'qplus_mean_value': positive, 'qminus_mean_value': negative,
        'signed_qplus_minus_qminus': difference, 'summed_numerical_proxy_and_tail': uncertainty,
        'higher_value_query': ('qplus' if difference > 0 else 'qminus' if difference < 0 else 'tie'),
        'resolved_on_declared_proxies_only': abs(difference) > uncertainty,
        'not_directed_global_numerical_certificate': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan = validate(args.plan, args.approval)
    library_versions = {name: importlib.metadata.version(name) for name in configuration()['libraries']}
    if library_versions != configuration()['libraries']:
        raise ValueError('registered NumPy/SciPy versions required')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'source_sha256': sources(),
        'plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
        'libraries': library_versions, 'new_real_labels_fits_or_agent_calls': 0})
    try:
        state, selected = fixtures()
        evaluations = []
        for fixture in selected:
            levels = []
            for count in field.LEVELS:
                start = time.perf_counter(); calculation = field.native_value(fixture, count)
                calculation['elapsed_seconds_descriptive_only'] = time.perf_counter() - start
                levels.append(calculation)
            final_difference = abs(levels[-1]['value'] - levels[-2]['value'])
            final_tail = levels[-1]['tail']['total_bound_ideal_arithmetic']
            evaluations.append({'fixture': fixture, 'levels': levels,
                'final_refinement_abs_diagnostic': final_difference,
                'convergence_diagnostic_pass': final_difference <= 2e-5,
                'analytic_tail_budget_pass': final_tail <= 1e-4,
                'finite_quadrature_not_analytically_bounded': True})
        probes = []
        for fixture in selected[:4]:
            pair = fixture['exact_NIG_pair']
            for z in configuration()['NIG_conditional_z_probes']:
                hypothetical_y = pair['mu_query'] + pair['scale_query'] * z
                updated = posterior.update_observed_label(state, fixture['query_context'], hypothetical_y)
                updated_law = posterior.scalar_law(updated, fixture['target_context'])
                conditional = posterior.conditional_target(pair, z)
                diffs = {name: abs(updated_law[name] - conditional[name]) for name in ('mu', 'scale', 'nu')}
                probes.append({'fixture_id': fixture['id'], 'standardized_hypothetical_z': z,
                    'hypothetical_modeled_query': hypothetical_y, 'NIG_updated_target': updated_law,
                    'conditional_bivariate_t_target': conditional, 'absolute_differences': diffs,
                    'not_a_real_measurement_or_persisted_material_training_label': True})
        algebra_ok = all(all(record['absolute_differences'][name] <= 2e-10 + 2e-10 * abs(record['conditional_bivariate_t_target'][name])
                            for name in ('mu', 'scale', 'nu')) for record in probes)
        rho_probes = []
        for left, right in configuration()['rho_bound_probe_existing_case_pairs']:
            original, other = selected[left], selected[right]
            if any(original[name] != other[name] for name in ('mu', 'scale', 's0', 'nu', 'loss_coordinate')):
                raise AssertionError('rho probe must share target marginal and native map')
            bound = field.rho_coupling_bound(original, other['rho'])
            observed_difference = abs(evaluations[left]['levels'][-1]['value'] - evaluations[right]['levels'][-1]['value'])
            rho_probes.append({'left_fixture': original['id'], 'right_fixture': other['id'],
                'observed_value_difference': observed_difference, 'analytic_coupling': bound,
                'within_analytic_plus_predeclared_agreement_tolerance': observed_difference <= bound['bound_ideal_arithmetic'] + 2e-4})
        affine = field.affine_student_value(6., math.sqrt(1.1), 0., 2.)
        affine_difference = abs(evaluations[7]['levels'][-1]['value'] - affine)
        symmetry_difference = abs(evaluations[0]['levels'][-1]['value'] - evaluations[4]['levels'][-1]['value'])
        qvalues, errors = {}, {}
        for name, indices in [('Studentt_native', [0, 1, 2, 3]), ('matchedGaussian_native', [8, 9, 10, 11])]:
            values = [evaluations[index]['levels'][-1]['value'] for index in indices]
            eta = [evaluations[index]['final_refinement_abs_diagnostic']
                   + evaluations[index]['levels'][-1]['tail']['total_bound_ideal_arithmetic'] for index in indices]
            qvalues[name] = rank_record(sum(values[:2]) / 2, sum(values[2:]) / 2,
                                       sum(eta[:2]) / 2, sum(eta[2:]) / 2)
        for name, function in [('Studentt_affine', field.affine_student_value),
                               ('matchedGaussian_affine', field.affine_matched_gaussian_value)]:
            values = [function(fixture['nu'], fixture['scale'], fixture['rho'], fixture['s0']) for fixture in selected[:4]]
            qvalues[name] = rank_record(sum(values[:2]) / 2, sum(values[2:]) / 2, 0., 0.)
        summary = {'fixed_evaluations': 12, 'NIG_conditional_probes': len(probes),
            'algebra_pass': algebra_ok, 'all_refinement_diagnostics_pass': all(item['convergence_diagnostic_pass'] for item in evaluations),
            'all_analytic_tail_budgets_pass': all(item['analytic_tail_budget_pass'] for item in evaluations),
            'rho_sign_symmetry_abs_difference': symmetry_difference,
            'finite_Student_rho0_positive_value_observed': evaluations[5]['levels'][-1]['value'] > 0,
            'Gaussian_rho0_exact_value': evaluations[6]['levels'][-1]['value'],
            'affine_rho0_closed_value': affine, 'affine_rho0_GL_vs_closed_abs_difference': affine_difference,
            'rankings': qvalues, 'rank_reversal_not_required_or_prewritten': True,
            'no_material_effectiveness_or_adaptive_campaign_claim': True}
        if len(evaluations) != 12 or sources() != plan['source_sha256']:
            raise ValueError('source or fixed evaluation count changed')
        result = {'status': 'COMPLETE_FIXED12_NATIVE_RISKFIELD_MECHANISM_GATE', 'actual_end_utc': utc(),
            'source_sha256': sources(), 'plan_sha256': sha(args.plan), 'configuration': plan['configuration'],
            'NIG_state': state, 'evaluations': evaluations, 'conditional_update_probes': probes,
            'rho_coupling_probes': rho_probes, 'summary': summary,
            'independent_reference_audit_not_yet_executed': True,
            'analytic_bound_and_refinement_proxy_not_complete_global_error_certificate': True}
        write(output / 'RESULTS.json', result)
        write(output / 'EXECUTION_COMPLETE.json', {'actual_utc': utc(), 'status': result['status'],
            'source_sha256': sources(), 'plan_sha256': sha(args.plan),
            'files': {name: sha(output / name) for name in ('EXECUTION_START.json', 'RESULTS.json')}})
        print(json.dumps(summary, ensure_ascii=True))
    except Exception as problem:
        write(output / 'FAILED_STOP.json', {'actual_utc': utc(), 'type': type(problem).__name__,
            'message': str(problem), 'source_sha256': sources(), 'do_not_resweep_fixed_fixtures': True})
        raise


if __name__ == '__main__':
    main()
