"""Four/singleton mixture interface gate using one sealed, previously scored law.

No material file, fit, proposal, or new model context. Root executes once after
pinning this file and the declared inputs. The independent reference integrates
the native mixture CDF directly, rather than reusing CRPS decomposition.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    plan = read(args.plan)
    if args.approval != plan['root_approval_token']:
        raise ValueError('root approval token mismatch')
    if sha(__file__) != plan['audit_source_sha256']:
        raise ValueError('audit source changed')
    for entry in plan['pins'].values():
        if sha(entry['file']) != entry['sha256']:
            raise ValueError('pinned input or source changed: ' + entry['file'])
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {
        'actual_utc': datetime.now(timezone.utc).isoformat(),
        'plan_sha256': sha(args.plan), 'scope': plan['scope'],
        'material_reads': 0, 'fits': 0, 'new_labels': 0,
        'old_saved_law_only': True})
    pins = plan['pins']
    failures, checks = [], []

    def compare(label, actual, expected, abs_tol, rel_tol=0.):
        error = abs(float(actual) - float(expected))
        allowance = abs_tol + rel_tol * abs(float(expected))
        passed = math.isfinite(error) and error <= allowance
        checks.append({'check': label, 'actual': float(actual),
                       'reference': float(expected), 'absolute_error': error,
                       'allowance': allowance, 'pass': passed})
        if not passed:
            failures.append(label)

    try:
        from scipy import integrate
        # No import of any fitting/proposal module. The old scalar scorer's two
        # definition-only dependencies are explicitly hash-bound before import.
        load('data_scope', pins['legacy_data_scope']['file'])
        load('joint_model', pins['legacy_joint_model']['file'])
        scalar_module = load('_frozen_scalar_gate', pins['legacy_scalar_score']['file'])
        sut = load('_generic_probability_under_audit', pins['probability']['file'])
        load('full_native_reference', pins['independent_native_core']['file'])
        ref = load('_independent_native_score', pins['independent_native_score']['file'])
        saved = read(pins['sealed_score_fixture']['file'])
        record = next(item for item in saved['law_records']
                      if item['id'] == plan['sealed_law_record_id'])
        original = copy.deepcopy(record['law'])
        if len(original['components']) != 3:
            raise ValueError('sealed three-component law required')
        split = copy.deepcopy(original)
        first = copy.deepcopy(original['components'][0])
        first['id'] = first['id'] + '_alias_a'
        second = copy.deepcopy(original['components'][0])
        second['id'] = second['id'] + '_alias_b'
        split['components'] = [first, second] + copy.deepcopy(original['components'][1:])
        first_log = original['log_weights'][0] - math.log(2.)
        split['log_weights'] = [first_log, first_log] + original['log_weights'][1:]
        split.pop('weights_display_only', None)
        singleton = copy.deepcopy(original)
        singleton['components'] = [copy.deepcopy(original['components'][0])]
        singleton['log_weights'] = [0.]
        singleton.pop('weights_display_only', None)
        scalar = {key: singleton[key] for key in
                  ('coordinate', 'native_first_scale_s0', 'lambda', 'unit')}
        scalar.update({key: singleton['components'][0][key] for key in
                       ('location', 'scale', 'df')})
        scalar['family'] = 'student_t'
        laws = {'split_four': split, 'singleton_one': singleton}
        targets = plan['fixed_native_targets']
        probabilities = plan['fixed_probabilities']
        tolerances = plan['preoutput_criteria']
        results = []
        independent = {}
        scalar_scores = {}

        def frozen_scalar(y, law):
            key = json.dumps([y, law], sort_keys=True)
            if key not in scalar_scores:
                scalar_scores[key] = scalar_module.score_law(y, law)
            return scalar_scores[key]

        with warnings.catch_warnings():
            warnings.simplefilter('error', integrate.IntegrationWarning)
            for name, law in laws.items():
                comparison_law = original if name == 'split_four' else singleton
                qref = {str(p): ref.quantile(comparison_law, p) for p in probabilities}
                quantiles = []
                for p in probabilities:
                    actual = sut.quantile_native(law, p)
                    compare(name + '/quantile/' + str(p), actual, qref[str(p)],
                            tolerances['quantile_abs'], tolerances['quantile_rel'])
                    compare(name + '/CDF_at_quantile/' + str(p),
                            ref.cdf(comparison_law, actual), p,
                            tolerances['CDF_abs'])
                    quantiles.append({'probability': p, 'native_quantile': actual,
                                      'independent_native_quantile': qref[str(p)]})
                for y in targets:
                    actual = sut.score_law(y, law, frozen_scalar)
                    crps, integration_error = ref.direct_crps(comparison_law, y)
                    cdf = ref.cdf(comparison_law, y)
                    logpdf = ref.log_density(comparison_law, y)
                    compare(name + '/native_CRPS/' + str(y), actual['CRPS_native'], crps,
                            tolerances['native_CRPS_abs'], tolerances['native_CRPS_rel'])
                    compare(name + '/PIT/' + str(y), actual['PIT_midpoint'], cdf,
                            tolerances['CDF_abs'])
                    compare(name + '/log_density/' + str(y), actual['log_density_native'],
                            logpdf, tolerances['log_density_abs'], tolerances['log_density_rel'])
                    identity_score = (next(item['score'] for item in record['scores']
                                           if item['native_target'] == y)
                                      if name == 'split_four' else frozen_scalar(y, scalar))
                    for key in ('CRPS_native', 'PIT_midpoint', 'predictive_median_native',
                                'log_density_native'):
                        compare(name + '/distribution_identity/' + key + '/' + str(y),
                                actual[key], identity_score[key],
                                tolerances['identity_abs'], tolerances['identity_rel'])
                    results.append({'case': name, 'native_target': y, 'law': law,
                                    'score': actual, 'independent': {
                                        'native_CRPS': crps, 'native_quad_error_diagnostic': integration_error,
                                        'CDF': cdf, 'log_density_native': logpdf},
                                    'native_quantiles': quantiles})
        result = {'status': 'PASS' if not failures else 'FAIL',
                  'actual_utc': datetime.now(timezone.utc).isoformat(),
                  'scope': plan['scope'], 'plan_sha256': sha(args.plan),
                  'old_fixture_sha256': pins['sealed_score_fixture']['sha256'],
                  'sealed_law_record_id': record['id'], 'law_records': results,
                  'checks': checks, 'failed_checks': failures,
                  'new_SUT_score_coordinates': 4,
                  'independent_native_CDF_CRPS_integrals': 4,
                  'quantiles_per_distribution': len(probabilities),
                  'material_reads': 0, 'fits': 0, 'new_labels': 0,
                  'numerical_error_diagnostics_not_IEEE_certificates': True,
                  'not_material_effect_or_new_algorithm_evidence': True}
    except Exception as exc:
        result = {'status': 'FAIL', 'scope': plan['scope'],
                  'actual_utc': datetime.now(timezone.utc).isoformat(),
                  'plan_sha256': sha(args.plan), 'error_type': type(exc).__name__,
                  'error': str(exc), 'partial_checks': checks,
                  'material_reads': 0, 'fits': 0, 'new_labels': 0}
    write(output / 'GENERIC_MIXTURE_ALIAS_AUDIT.json', result)
    print(json.dumps({'status': result['status'],
                      'result_sha256': sha(output / 'GENERIC_MIXTURE_ALIAS_AUDIT.json'),
                      'output': str(output)}, ensure_ascii=False))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
