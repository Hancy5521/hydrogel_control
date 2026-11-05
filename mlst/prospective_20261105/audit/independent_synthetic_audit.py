"""Independent, preregistered artificial-number qualification; no material data.

This script is not a predictor and never reads old material measurements.
Do not execute until the root reviewer approves the source-pinned audit plan.
"""
import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def artificial_hash(value):
    return hashlib.sha256(('SYNTHETIC_ONLY:' + value).encode()).hexdigest()


def rejected(call):
    try:
        call()
    except (ValueError, TypeError, KeyError, OverflowError):
        return True
    return False


class Audit:
    def __init__(self, output):
        self.output = output
        self.results = []
        self.commands = []

    def check(self, name, function):
        start = utc()
        try:
            details = function()
            self.results.append({'name': name, 'status': 'PASS', 'actual_utc': start, 'details': details})
        except Exception as error:
            self.results.append({'name': name, 'status': 'FAIL', 'actual_utc': start,
                                 'error_type': type(error).__name__, 'message': str(error)})

    def cli(self, source, *args, expected=0):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, str(source / 'cli.py'), *map(str, args)]
        result = subprocess.run(command, cwd=str(source), env=env, capture_output=True,
                                text=True, encoding='utf-8', timeout=30)
        entry = {'actual_utc': utc(), 'command': command, 'returncode': result.returncode,
                 'stdout': result.stdout, 'stderr': result.stderr}
        self.commands.append(entry)
        with (self.output / 'COMMANDS.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')
        assert result.returncode == expected, entry
        return result


def fixtures(source):
    config = load(source / 'DESIGN_TEMPLATE.json')
    config.update({'data_mode': 'synthetic', 'protocol_id': 'SYNTHETIC_INDEPENDENT_AUDIT',
                   'scientific_goal': 'new_formulation', 'material_state': 'SYNTHETIC_NO_MATERIAL',
                   'temperature_kelvin': 298.15, 'n_calibration_units': 9,
                   'n_confirmation_units': 3, 'practical_error_tolerance': 0.1})
    for key in ('anchor_rule', 'anchor_policy_reference', 'scan_protocol_reference',
                'source_admission_reference', 'independence_assumption_evidence',
                'design_reviewed_by', 'design_review_reference', 'sample_size_basis',
                'practical_error_tolerance_unit'):
        config[key] = 'SYNTHETIC_ONLY_' + key
    for method in config['methods']:
        method['model_sha256'] = artificial_hash(method['id'] + '_model')
        method['runtime_bindings_sha256'] = artificial_hash(method['id'] + '_runtime')
    config['consumed_registry'] = {'review_complete': True,
                                  'review_reference': 'SYNTHETIC_ONLY_REVIEW',
                                  'record_keys': ['SYNTHETIC_OLD_RUN'],
                                  'preparation_batch_ids': ['SYNTHETIC_OLD_BATCH'],
                                  'formulation_ids': ['SYNTHETIC_OLD_FORMULA'],
                                  'identity_aliases': ['SYNTHETIC_OLD_ALIAS']}
    records = []
    for role, count in [('calibration', 9), ('confirmation', 3)]:
        for i in range(count):
            stem = f'SYNTHETIC_{role}_{i}'
            records.append({'record_id': stem + '_MASK0', 'unit_id': stem + '_UNIT',
                            'role': role, 'formulation_id': stem + '_FORMULA',
                            'preparation_batch_id': stem + '_BATCH', 'source_id': 'SYNTHETIC_SOURCE',
                            'source_version': 'SYNTHETIC_VERSION', 'source_record_key': stem + '_RAW',
                            'provenance_reference': 'SYNTHETIC_ONLY_NO_MEASUREMENT',
                            'material_state': config['material_state'], 'temperature_kelvin': 298.15,
                            'identity_aliases': [stem + '_PHYSICAL_ALIAS'], 'is_synthetic': True,
                            'targets_previously_inspected': False,
                            'native_gamma': [1., 2., 3., 4., 5., 6., 8., 16., 32.],
                            'observed_indices': list(range(6)), 'target_indices': [6, 7, 8]})
    repeated_mask = copy.deepcopy(records[0])
    repeated_mask['record_id'] += '_REPEATED_QUERY'
    records.append(repeated_mask)
    cohort = {'schema': config['schema'], 'data_mode': 'synthetic', 'records': records}
    return config, cohort


def envelopes(config, cohort, protocol_hash, role):
    predictions = {'schema': config['schema'], 'data_mode': 'synthetic',
                   'protocol_sha256': protocol_hash, 'records': []}
    truth = copy.deepcopy(predictions)
    for record in cohort['records']:
        if record['role'] != role:
            continue
        i = int(record['unit_id'].split('_')[2])
        targets = [record['native_gamma'][j] for j in record['target_indices']]
        n = len(targets)
        laws = {}
        for method_index, method in enumerate(config['methods']):
            sd = 1. + 0.2 * method_index
            laws[method['id']] = {'model_sha256': method['model_sha256'],
                                 'runtime_bindings_sha256': method['runtime_bindings_sha256'],
                                 'mean_log_eta': [0.2 * method_index] * n,
                                 'covariance_log_eta': [[sd * sd if j == k else 0.
                                                         for k in range(n)] for j in range(n)]}
        predictions['records'].append({'record_id': record['record_id'],
                                       'observed_gamma': [record['native_gamma'][j] for j in record['observed_indices']],
                                       'target_gamma': targets,
                                       'anchor_input_sha256': artificial_hash(record['unit_id'] + '_anchors'),
                                       'prediction_source_sha256': artificial_hash('not_a_predictor_' + record['unit_id']),
                                       'methods': laws})
        truth['records'].append({'record_id': record['record_id'], 'target_gamma': targets,
                                 'y_log_eta': [i / 10.] * n,
                                 'original_measurement_sha256': artificial_hash('artificial_outcome_' + record['unit_id'])})
    return predictions, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = load(args.plan)
    source = Path(plan['source_directory'])
    actual_hashes = {name: digest(source / name) for name in plan['source_sha256']}
    assert actual_hashes == plan['source_sha256'], 'Source changed since independent audit preregistration'
    assert digest(__file__) == plan['audit_script_sha256'], 'Audit script changed since preregistration'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    dump(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'mode': 'synthetic',
                                         'actual_source_sha256': actual_hashes,
                                         'plan_sha256': digest(args.plan), 'real_measurements_read': False,
                                         'new_material_evidence': False})
    audit = Audit(output)
    sys.path.insert(0, str(source))
    calibration = importlib.import_module('calibration')
    evaluation = importlib.import_module('evaluation')
    contract = importlib.import_module('contract')
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import ndtr

    def exact_rank_and_leave_one_out():
        scalar_cases, held_out_cases = 0, 0
        for alpha_text in ('0.1', '0.05', '0.25', '1/3', '0.70'):
            alpha = Fraction(alpha_text)
            for n in range(31):
                # Independent exact-rational generalized inverse definition.
                target_mass = 1 - alpha
                expected_k = next(k for k in range(1, n + 2) if Fraction(k, n + 1) >= target_mass)
                scores = {f'SYNTHETIC_{i}': float(i + 1) for i in range(n)}
                threshold = calibration.calibrate_unit_scores(scores, alpha_text)
                assert threshold.order_index == expected_k and threshold.n_units == n
                assert threshold.q == (float(expected_k) if expected_k <= n else math.inf)
                encoded = threshold.to_json()
                json.dumps(encoded, allow_nan=False)
                restored = calibration.threshold_from_json(encoded)
                assert restored.q == threshold.q
                scalar_cases += 1
                if n:
                    successes = 0
                    values = [float(i) for i in range(n + 1)]
                    for holdout in range(n + 1):
                        cal = {f'SYNTHETIC_{i}': value for i, value in enumerate(values) if i != holdout}
                        q = calibration.calibrate_unit_scores(cal, alpha_text).q
                        successes += values[holdout] <= q
                        held_out_cases += 1
                    assert Fraction(successes, n + 1) >= 1 - alpha
        return {'exact_order_cases': scalar_cases, 'exhaustive_held_out_assignments': held_out_cases,
                'meaning': 'Finite exchangeability rank counting, no random/material coverage estimate'}

    def ties_and_unbounded():
        results = []
        for values in ([0., 0., 0., 0., 0., 0.], [0., 0., 1., 1., 2., 2.]):
            for alpha_text in ('0.1', '0.25', '1/3'):
                passed = 0
                for j, value in enumerate(values):
                    q = calibration.calibrate_unit_scores({f'SYNTHETIC_{i}': v for i, v in enumerate(values) if i != j}, alpha_text).q
                    passed += value <= q
                assert Fraction(passed, len(values)) >= 1 - Fraction(alpha_text)
                results.append([values, alpha_text, passed])
        zero = calibration.calibrate_unit_scores({f'SYNTHETIC_{i}': 0. for i in range(9)}, '0.1')
        assert zero.q == 0.
        empty = calibration.calibrate_unit_scores({}, '0.1')
        assert math.isinf(empty.q) and empty.to_json()['unbounded'] is True
        low = calibration.calibrate_unit_scores({'SYNTHETIC_A': 1.}, '0.1')
        band = calibration.make_band([0., 1.], [1., 2.], low)
        encoded = band.to_json()
        assert encoded['unbounded'] is True and encoded['CRPS_defined_for_band'] is False
        assert rejected(lambda: band.crps([0., 1.]))
        assert rejected(lambda: band.joint_log_density([0., 1.]))
        diagnostic = evaluation.band_diagnostics([0., 1.], [1., 2.], [0., 1.], [.5, .5], low)
        assert diagnostic['coverage_is_vacuous'] is True
        assert diagnostic['mean_log_width'] is None and diagnostic['interval_score_same_alpha'] is None
        json.dumps(diagnostic, allow_nan=False)
        return {'ties_leave_one_out': results, 'zero_and_empty_and_unbounded_are_explicit': True}

    def explicit_infinity_reasons():
        rows = [{'unit_id': f'SYNTHETIC_{i}', 'mean': [0.], 'sd': [1.], 'truth': [0.]} for i in range(8)]
        rows.append({'unit_id': 'SYNTHETIC_OVERFLOW', 'mean': [-1e308], 'sd': [1.], 'truth': [1e308]})
        scores, reasons = calibration.unit_max_scores(rows, return_reasons=True)
        assert math.isinf(scores['SYNTHETIC_OVERFLOW'])
        assert reasons['SYNTHETIC_OVERFLOW'] == ['numeric_score_overflow']
        numeric = calibration.calibrate_unit_scores(scores, '0.1', score_reasons=reasons).to_json()
        assert set(numeric['infinity_reasons']) == {'selected_infinite_score', 'numeric_score_overflow'}
        explicit = calibration.calibrate_unit_scores({**{f'SYNTHETIC_{i}': 0. for i in range(8)}, 'SYNTHETIC_INF': math.inf}, '0.1').to_json()
        assert set(explicit['infinity_reasons']) == {'selected_infinite_score', 'infinite_input_score'}
        insufficient = calibration.calibrate_unit_scores({f'SYNTHETIC_{i}': 0. for i in range(8)}, '0.1').to_json()
        assert insufficient['infinity_reasons'] == ['insufficient_calibration_units']
        for encoded in (numeric, explicit, insufficient):
            assert encoded['q'] is None and encoded['unbounded'] is True
            assert math.isinf(calibration.threshold_from_json(encoded).q)
            json.dumps(encoded, allow_nan=False)
        return {'numeric_overflow_and_explicit_extended_score_and_insufficient_n_are_distinguished': True,
                'numeric_overflow_is_vacuous_set_not_material_reliability': True}

    def input_and_density_separation():
        invalid = 0
        good = {'unit_id': 'SYNTHETIC_A', 'mean': [0., 1.], 'sd': [1., 1.], 'truth': [.3, .7]}
        for value in (0., -1., math.nan, math.inf):
            bad = copy.deepcopy(good)
            bad['sd'][0] = value
            assert rejected(lambda: calibration.unit_max_scores([bad]))
            invalid += 1
        for field in ('mean', 'truth'):
            for value in (math.nan, math.inf):
                bad = copy.deepcopy(good)
                bad[field][0] = value
                assert rejected(lambda: calibration.unit_max_scores([bad]))
                invalid += 1
        for value in ('0', '1', '-0.1', 'bad', 0.1, True):
            assert rejected(lambda: calibration.alpha_fraction(value))
            invalid += 1
        large_q = calibration.calibrate_unit_scores({f'SYNTHETIC_{i}': 3. for i in range(9)}, '0.1')
        small_q = calibration.calibrate_unit_scores({f'SYNTHETIC_{i}': .5 for i in range(9)}, '0.1')
        before = evaluation.gaussian_scores([0., 1.], [[1., 0.], [0., 4.]], [.2, .8], [.5, .5], '0.1')
        b1 = calibration.make_band([0., 1.], [1., 2.], small_q)
        b2 = calibration.make_band([0., 1.], [1., 2.], large_q)
        after = evaluation.gaussian_scores([0., 1.], [[1., 0.], [0., 4.]], [.2, .8], [.5, .5], '0.1')
        assert before == after and b1.upper_log != b2.upper_log
        return {'invalid_cases_rejected': invalid, 'raw_density_scores_unchanged_by_band_q': True}

    def standalone_diagnostic_validation():
        threshold = calibration.calibrate_unit_scores({'SYNTHETIC_A': 1.}, '0.1')
        cases = [([math.nan, 1.], [.5, .5]), ([0.], [.5, .5]),
                 ([0., 1.], [1., 1.]), ([0., 1.], [-1., 2.]), ([0., 1.], [.5])]
        for truth, weights in cases:
            assert rejected(lambda: evaluation.band_diagnostics([0., 1.], [1., 1.], truth, weights, threshold)), (truth, weights)
        return {'invalid_unbounded_diagnostic_inputs_rejected': len(cases)}

    def crps_by_independent_cdf_integration():
        errors = []
        for mu, sd in [(-2., .3), (0., 1.), (3., 1.7)]:
            for z in (-5., -1., 0., .5, 3.):
                y = mu + sd * z
                left = quad(lambda x: ndtr(x) ** 2, -math.inf, z, epsabs=2e-11, epsrel=2e-11)[0]
                right = quad(lambda x: ndtr(-x) ** 2, z, math.inf, epsabs=2e-11, epsrel=2e-11)[0]
                oracle = sd * (left + right)
                actual = evaluation.gaussian_scores([mu], [[sd * sd]], [y], [1.], '0.1')['log_rate_CRPS']
                assert math.isclose(actual, oracle, rel_tol=2e-9, abs_tol=2e-10), (actual, oracle, mu, sd, z)
                errors.append(abs(actual - oracle))
        return {'cases': len(errors), 'oracle': 'CDF squared-error numerical quadrature, scipy quad/ndtr',
                'max_absolute_error': max(errors)}

    def trapezoid_oracle():
        tested = 0
        for x in ([0., 1.], [-2., -.3, .2, 2.], [0., .1, .6, 1., 3.]):
            weights = evaluation.target_weights([math.exp(v) for v in x])
            assert min(weights) > 0 and math.isclose(sum(weights), 1., abs_tol=1e-13)
            # A trapezoid is exact for every affine function, even irregular grids.
            values = [2. + .3 * v for v in x]
            actual = sum(w * value for w, value in zip(weights, values))
            expected = 2. + .3 * (x[0] + x[-1]) / 2
            assert math.isclose(actual, expected, abs_tol=2e-13)
            assert math.isclose(weights[0], (x[1] - x[0]) / (2 * (x[-1] - x[0])), abs_tol=1e-13)
            assert math.isclose(weights[-1], (x[-1] - x[-2]) / (2 * (x[-1] - x[0])), abs_tol=1e-13)
            tested += 1
        for bad in ([], [1.], [1., 1.], [2., 1.], [0., 1.], [1., math.inf]):
            assert rejected(lambda: evaluation.target_weights(bad)), bad
        # Positive distinct float rates can have numerically identical logarithms.
        close_rates = [100., math.nextafter(100., math.inf)]
        assert rejected(lambda: evaluation.target_weights(close_rates)), 'Collapsed log span must be rejected'
        return {'affine_integral_cases': tested, 'invalid_grids_rejected': 7}

    config, cohort = fixtures(source)
    protocol_stub = {'config': config, 'cohort': cohort}

    def group_max_and_roles():
        rows = [{'unit_id': 'SYNTHETIC_A', 'mean': [0., 0.], 'sd': [1., 2.], 'truth': [1., 4.]},
                {'unit_id': 'SYNTHETIC_A', 'mean': [0., 0.], 'sd': [1., 1.], 'truth': [3., 1.]},
                {'unit_id': 'SYNTHETIC_B', 'mean': [0., 0.], 'sd': [1., 1.], 'truth': [2., 1.]}]
        scores = calibration.unit_max_scores(rows + rows)
        assert scores == {'SYNTHETIC_A': 3., 'SYNTHETIC_B': 2.}
        valid = contract.check_design(config, cohort)
        assert valid['declared_unit_counts']['calibration'] == 9
        mutations = []
        bad = copy.deepcopy(cohort)
        bad['records'][9]['preparation_batch_id'] = bad['records'][0]['preparation_batch_id']
        mutations.append(bad)
        bad = copy.deepcopy(cohort)
        bad['records'][9]['formulation_id'] = bad['records'][0]['formulation_id']
        mutations.append(bad)
        bad = copy.deepcopy(cohort)
        bad['records'][9]['identity_aliases'] = bad['records'][0]['identity_aliases']
        mutations.append(bad)
        bad = copy.deepcopy(cohort)
        bad['records'][0]['source_record_key'] = 'SYNTHETIC_OLD_RUN'
        mutations.append(bad)
        bad = copy.deepcopy(cohort)
        bad['records'][0]['targets_previously_inspected'] = True
        mutations.append(bad)
        bad = copy.deepcopy(cohort)
        bad['records'][0]['target_indices'] = [5, 6, 7, 8]
        mutations.append(bad)
        for bad in mutations:
            assert rejected(lambda: contract.check_design(config, bad))
        known = copy.deepcopy(config)
        known['scientific_goal'] = 'new_preparation_known_formula'
        overlap = copy.deepcopy(cohort)
        overlap['records'][9]['formulation_id'] = overlap['records'][0]['formulation_id']
        contract.check_design(known, overlap)
        bad_config = copy.deepcopy(config)
        bad_config['truth'] = [123.]
        assert rejected(lambda: contract.check_design(bad_config, cohort))
        bad_cohort = copy.deepcopy(cohort)
        bad_cohort['truth'] = [123.]
        assert rejected(lambda: contract.check_design(config, bad_cohort))
        assert rejected(lambda: contract.check_design(load(source / 'DESIGN_TEMPLATE.json'), load(source / 'COHORT_TEMPLATE.json')))
        return {'calibration_units_with_duplicate_query': 9, 'metadata_mutations_rejected': len(mutations),
                'known_formula_new_batch_differentiated': True, 'null_real_template_rejected': True}

    def method_grid_and_truth_bindings():
        protocol_hash = artificial_hash('protocol_stub')
        prediction, revealed = envelopes(config, cohort, protocol_hash, 'calibration')
        contract.check_predictions(prediction, protocol_stub, 'calibration', protocol_hash)
        truth = contract.check_truth(revealed, prediction, protocol_stub, 'calibration', protocol_hash)
        assert len(truth) == 10
        mutations = []
        bad = copy.deepcopy(prediction)
        del bad['records'][0]['methods'][config['methods'][-1]['id']]
        mutations.append(bad)
        bad = copy.deepcopy(prediction)
        bad['records'][0]['target_gamma'][-1] += 1.
        mutations.append(bad)
        bad = copy.deepcopy(prediction)
        bad['records'][0]['methods'][config['primary_method_id']]['model_sha256'] = '0' * 64
        mutations.append(bad)
        bad = copy.deepcopy(prediction)
        bad['records'][0]['y_log_eta'] = [0., 0., 0.]
        mutations.append(bad)
        bad = copy.deepcopy(prediction)
        bad['records'] = bad['records'][:-1]
        mutations.append(bad)
        for bad in mutations:
            assert rejected(lambda: contract.check_predictions(bad, protocol_stub, 'calibration', protocol_hash))
        bad_truth = copy.deepcopy(revealed)
        bad_truth['records'][0]['target_gamma'][-1] += 1.
        assert rejected(lambda: contract.check_truth(bad_truth, prediction, protocol_stub, 'calibration', protocol_hash))
        conflicting = copy.deepcopy(revealed)
        conflicting['records'][-1]['y_log_eta'][0] += 1.
        assert rejected(lambda: contract.check_truth(conflicting, prediction, protocol_stub, 'calibration', protocol_hash))
        return {'method_grid_hash_label_contamination_or_subset_mutations_rejected': len(mutations),
                'truth_grid_mismatch_rejected': True, 'conflicting_same_curve_overlapping_truth_rejected': True}

    def null_real_register_and_source_guard():
        study = output / 'SYNTHETIC_REGISTRATION_GUARDS'
        study.mkdir()
        null_run = study / 'NULL_REAL_RUN'
        audit.cli(source, 'register', '--config', source / 'DESIGN_TEMPLATE.json',
                  '--cohort', source / 'COHORT_TEMPLATE.json', '--run', null_run, expected=2)
        assert not (null_run / 'PROTOCOL.json').exists()
        source_copy = study / 'SOURCE_COPY'
        shutil.copytree(source, source_copy, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cfg, co = study / 'DESIGN.json', study / 'COHORT.json'
        dump(cfg, config)
        dump(co, cohort)
        run = study / 'SOURCE_GUARD_RUN'
        audit.cli(source_copy, 'register', '--config', cfg, '--cohort', co, '--run', run)
        ph = digest(run / 'PROTOCOL.json')
        prediction, _ = envelopes(config, cohort, ph, 'calibration')
        path = study / 'CALIBRATION_PREDICTIONS.json'
        dump(path, prediction)
        with (source_copy / 'evaluation.py').open('a', encoding='utf-8') as handle:
            handle.write('\n# SYNTHETIC audit change after registration, no production modification.\n')
        audit.cli(source_copy, 'freeze-calibration', '--run', run, '--predictions', path, expected=2)
        assert not (run / 'CALIBRATION_PREDICTIONS_FROZEN.json').exists()
        return {'null_real_template_cannot_register': True,
                'source_change_in_isolated_audit_copy_is_rejected': True}

    def full_cli_workflow():
        study = output / 'SYNTHETIC_CLI'
        study.mkdir()
        cfg, co = study / 'DESIGN.json', study / 'COHORT.json'
        dump(cfg, config)
        dump(co, cohort)
        run = study / 'RUN'
        audit.cli(source, 'register', '--config', cfg, '--cohort', co, '--run', run)
        ph = digest(run / 'PROTOCOL.json')
        for role in ('calibration', 'confirmation'):
            pred, truth = envelopes(config, cohort, ph, role)
            dump(study / (role + '_predictions.json'), pred)
            dump(study / (role + '_truth.json'), truth)
        # This prelabel-phase failure must not create/consume any calibration truth.
        audit.cli(source, 'freeze-confirmation', '--run', run, '--predictions', study / 'confirmation_predictions.json', expected=2)
        assert not (run / 'CALIBRATION_FROZEN.json').exists()
        first_failure = load(sorted(run.glob('FAILURE_*.json'))[-1])
        assert not any(first_failure[key] for key in ('truth_open_attempt_started', 'truth_bytes_loaded', 'truth_payload_parsed'))
        assert not list(run.glob('*_REVEAL_STARTED.json'))
        assert not (run / 'REVEAL_ATTEMPT_FAILED_STOP.json').exists()
        audit.cli(source, 'freeze-calibration', '--run', run, '--predictions', study / 'calibration_predictions.json')
        audit.cli(source, 'calibrate', '--run', run, '--truth', study / 'calibration_truth.json')
        calibrated = load(run / 'CALIBRATION_FROZEN.json')
        for mid, value in calibrated['thresholds'].items():
            assert value['n_calibration_units'] == 9 and value['order_index_one_based'] == 9
            assert set(calibrated['scores_by_declared_unit'][mid]) == {
                r['unit_id'] for r in cohort['records'] if r['role'] == 'calibration'}
        audit.cli(source, 'freeze-confirmation', '--run', run, '--predictions', study / 'confirmation_predictions.json')
        audit.cli(source, 'evaluate', '--run', run, '--truth', study / 'confirmation_truth.json')
        result = load(run / 'CONFIRMATION_EVALUATION.json')
        assert result['data_mode'] == 'synthetic' and result['is_new_material_evidence'] is False and result['new_fits'] == 0
        assert len(result['methods']) == 3
        for mid, value in result['methods'].items():
            assert value['aggregate']['n_declared_units'] == 3
            assert value['aggregate']['n_registered_curve_masks'] == 3
            for row in value['per_record']:
                assert math.isclose(row['omitted_log_gap_from_highest_anchor'], math.log(8 / 6), abs_tol=1e-13)
                assert math.isclose(row['target_only_log_rate_span'], math.log(32 / 8), abs_tol=1e-13)
        # Refuse a second reveal/score and preserve the first result's byte identity.
        original_result = digest(run / 'CONFIRMATION_EVALUATION.json')
        audit.cli(source, 'evaluate', '--run', run, '--truth', study / 'confirmation_truth.json', expected=2)
        assert digest(run / 'CONFIRMATION_EVALUATION.json') == original_result
        return {'independent_artificial_calibration_units': 9, 'artificial_confirmation_units': 3,
                'methods': 3, 'calibration_queries': 10, 'material_evidence': False,
                'result_sha256': original_result}

    def forecast_and_q_freeze_guards():
        study = output / 'SYNTHETIC_TAMPER_GUARDS'
        study.mkdir()
        cfg, co = study / 'DESIGN.json', study / 'COHORT.json'
        dump(cfg, config)
        dump(co, cohort)
        observed = []
        for mode in ('forecast_after_freeze', 'q_after_calibration'):
            run = study / mode
            audit.cli(source, 'register', '--config', cfg, '--cohort', co, '--run', run)
            ph = digest(run / 'PROTOCOL.json')
            cp, ct = envelopes(config, cohort, ph, 'calibration')
            xp, xt = envelopes(config, cohort, ph, 'confirmation')
            cpath, tpath, xpath = study / (mode + '_CAL_PRED.json'), study / (mode + '_CAL_TRUTH.json'), study / (mode + '_CONF_PRED.json')
            dump(cpath, cp)
            dump(tpath, ct)
            dump(xpath, xp)
            audit.cli(source, 'freeze-calibration', '--run', run, '--predictions', cpath)
            if mode == 'forecast_after_freeze':
                path = run / 'CALIBRATION_PREDICTIONS_FROZEN.json'
                changed = load(path)
                mid = config['primary_method_id']
                changed['payload']['records'][0]['methods'][mid]['mean_log_eta'][0] += 1.
                path.write_text(json.dumps(changed, allow_nan=False), encoding='utf-8')
                audit.cli(source, 'calibrate', '--run', run, '--truth', tpath, expected=2)
                assert not (run / 'CALIBRATION_FROZEN.json').exists()
                assert not (run / 'CALIBRATION_REVEAL_STARTED.json').exists()
            else:
                audit.cli(source, 'calibrate', '--run', run, '--truth', tpath)
                path = run / 'CALIBRATION_FROZEN.json'
                changed = load(path)
                changed['thresholds'][config['primary_method_id']]['q'] += 1.
                path.write_text(json.dumps(changed, allow_nan=False), encoding='utf-8')
                audit.cli(source, 'freeze-confirmation', '--run', run, '--predictions', xpath, expected=2)
                assert not (run / 'CONFIRMATION_PREDICTIONS_FROZEN.json').exists()
            failure = load(sorted(run.glob('FAILURE_*.json'))[-1])
            assert not any(failure[key] for key in ('truth_open_attempt_started', 'truth_bytes_loaded', 'truth_payload_parsed'))
            assert not (run / 'REVEAL_ATTEMPT_FAILED_STOP.json').exists()
            observed.append(mode)
        return {'postfreeze_single_file_changes_rejected': observed,
                'limitation': 'Hash receipts do not prove absence of prior human access or prevent malicious coordinated rewriting of every local receipt'}

    def truth_open_attempt_failure_status():
        study = output / 'SYNTHETIC_MISSING_TRUTH'
        study.mkdir()
        cfg, co = study / 'DESIGN.json', study / 'COHORT.json'
        dump(cfg, config)
        dump(co, cohort)
        run = study / 'RUN'
        audit.cli(source, 'register', '--config', cfg, '--cohort', co, '--run', run)
        ph = digest(run / 'PROTOCOL.json')
        prediction, truth = envelopes(config, cohort, ph, 'calibration')
        ppath = study / 'CAL_PRED.json'
        dump(ppath, prediction)
        audit.cli(source, 'freeze-calibration', '--run', run, '--predictions', ppath)
        audit.cli(source, 'calibrate', '--run', run, '--truth', study / 'DOES_NOT_EXIST.json', expected=2)
        failure = load(sorted(run.glob('FAILURE_*.json'))[-1])
        assert failure['truth_open_attempt_started'] is True
        assert failure['truth_bytes_loaded'] is False and failure['truth_payload_parsed'] is False
        assert (run / 'CALIBRATION_REVEAL_STARTED.json').exists()
        assert not (run / 'CALIBRATION_REVEAL_PARSED.json').exists()
        assert (run / 'REVEAL_ATTEMPT_FAILED_STOP.json').exists()
        assert not (run / 'CALIBRATION_FROZEN.json').exists()
        tpath = study / 'NOW_PRESENT_BUT_NOT_FRESH_TRUTH.json'
        dump(tpath, truth)
        audit.cli(source, 'calibrate', '--run', run, '--truth', tpath, expected=2)
        assert not (run / 'CALIBRATION_FROZEN.json').exists()
        return {'failed_file_open_is_not_mislabeled_measured_values_loaded': True,
                'conservative_stop_after_open_attempt_and_no_silent_retry': True}

    for name, function in [
        ('exact_rank_and_finite_exchangeability_count', exact_rank_and_leave_one_out),
        ('ties_zero_empty_and_unbounded_sets', ties_and_unbounded),
        ('explicit_infinity_origin_reasons', explicit_infinity_reasons),
        ('invalid_scale_inputs_and_band_density_separation', input_and_density_separation),
        ('standalone_band_diagnostic_validation', standalone_diagnostic_validation),
        ('Gaussian_CRPS_independent_CDF_integral', crps_by_independent_cdf_integration),
        ('native_lograte_trapezoid_affine_oracle', trapezoid_oracle),
        ('physical_group_roles_and_null_real_template', group_max_and_roles),
        ('same_methods_grid_roles_truth_and_byte_bindings', method_grid_and_truth_bindings),
        ('null_real_registration_and_changed_source_guard', null_real_register_and_source_guard),
        ('synthetic_nine_cal_three_confirm_three_methods_CLI', full_cli_workflow),
        ('forecast_and_q_freeze_byte_identity_guards', forecast_and_q_freeze_guards),
        ('truth_open_attempt_and_actual_read_failure_distinction', truth_open_attempt_failure_status),
    ]:
        audit.check(name, function)
    final_hashes = {name: digest(source / name) for name in plan['source_sha256']}
    results = {'artifact_kind': 'independent_synthetic_software_audit_not_material_evidence',
               'actual_end_utc': utc(), 'plan_sha256': digest(args.plan),
               'source_unchanged_during_audit': final_hashes == actual_hashes,
               'source_sha256': final_hashes,
               'test_families': len(audit.results),
               'pass_count': sum(v['status'] == 'PASS' for v in audit.results),
               'fail_count': sum(v['status'] == 'FAIL' for v in audit.results),
               'new_material_measurements_or_fits': 0, 'results': audit.results,
               'actual_command_count': len(audit.commands),
               'prohibited_inference': 'No hydrogel coverage, new-batch generalization, efficacy, experimental saving or submission-readiness claim follows from these artificial inputs.'}
    dump(output / 'INDEPENDENT_SYNTHETIC_RESULTS.json', results)
    print(json.dumps({k: results[k] for k in ('pass_count', 'fail_count', 'source_unchanged_during_audit')}, indent=2))
    return 0 if results['fail_count'] == 0 and results['source_unchanged_during_audit'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
