"""Portable independent qualification of frozen hydrogel inference, not efficacy.

Usage: python verify_frozen_inference.py --repo-root PATH --output RESULT.json
Output reports relative source hashes, no private paths. Requires NumPy/SciPy.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
sys.dont_write_bytecode = True
import numpy as np
import scipy
from independent_gaussian import complete, condition, curvature_rows, joint


ROOT = None
MODEL = None
FIXTURE = None
API = None
RECORDS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(name, actual, expected, *, atol=5e-9, rtol=3e-9):
    aa, bb = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(aa - bb))) if aa.size else 0.
    passed = bool(aa.shape == bb.shape and np.allclose(aa, bb, atol=atol, rtol=rtol))
    RECORDS.append({'check': name, 'passed': passed, 'max_absolute_error': error,
                    'atol': atol, 'rtol': rtol})
    if not passed:
        raise AssertionError(name + ': independent numerical mismatch')


class FrozenInferenceTests(unittest.TestCase):
    def test_01_independent_analytic_and_joint_draws(self):
        oracle = complete(MODEL, FIXTURE)
        got = API.generate(MODEL, FIXTURE, draws=17, seed=123)
        for j, field in enumerate(('eta_analytic_p05', 'eta_analytic_median', 'eta_analytic_p95')):
            close(field + '_against_independent_joint', got[field], oracle['quantiles'][j])
        missing = oracle['unknown']
        draws = np.empty((17, len(FIXTURE['target_gamma'])))
        log_samples = oracle['unknown_mean'] + np.random.default_rng(123).normal(
            size=(17, len(missing))) @ np.linalg.cholesky(oracle['unknown_covariance']).T
        draws[:, missing] = np.exp(log_samples)
        for j, v in enumerate(oracle['mapping']):
            if v is not None:
                draws[:, j] = FIXTURE['observed_eta'][v]
        close('full_covariance_draws_against_independent_joint', got['eta_draws'], draws,
              atol=3e-8, rtol=3e-8)
        self.assertFalse(got['is_measured_data'])
        self.assertEqual(got['numerical_jitter'], 0.)

    def test_02_coordinate_and_anchor_permutations(self):
        permutation = np.array([9, 0, 5, 2, 7, 1, 8, 4, 3, 6])
        baseline = API.generate(MODEL, FIXTURE, draws=2, seed=1)
        query = copy.deepcopy(FIXTURE)
        query['target_gamma'] = np.asarray(query['target_gamma'])[permutation].tolist()
        reversed_anchors = np.array([4, 0, 5, 2, 1, 3])
        query['observed_gamma'] = np.asarray(query['observed_gamma'])[reversed_anchors].tolist()
        query['observed_eta'] = np.asarray(query['observed_eta'])[reversed_anchors].tolist()
        changed = API.generate(MODEL, query, draws=2, seed=1)
        for field in ('eta_analytic_p05', 'eta_analytic_median', 'eta_analytic_p95'):
            close(field + '_permutation', changed[field], np.asarray(baseline[field])[permutation])
        # Cholesky draws with a fixed seed need not be samplewise permutation-equivariant.

    def test_03_batched_target_marginals(self):
        baseline = API.generate(MODEL, FIXTURE, draws=2, seed=3)
        for indices in ([0, 1, 2], [3, 4, 5, 6], [7, 8, 9]):
            payload = copy.deepcopy(FIXTURE)
            payload['target_gamma'] = np.asarray(payload['target_gamma'])[indices].tolist()
            batch = API.generate(MODEL, payload, draws=2, seed=3)
            for field in ('eta_analytic_p05', 'eta_analytic_median', 'eta_analytic_p95'):
                close(field + '_batch_' + str(indices), batch[field], np.asarray(baseline[field])[indices])
        # Only marginals are checked: independently sampled batches lose cross-batch coupling.

    def test_04_seed_and_count_do_not_change_analytic_quantiles(self):
        first = API.generate(MODEL, FIXTURE, draws=1, seed=4)
        second = API.generate(MODEL, FIXTURE, draws=31, seed=44)
        for field in ('eta_analytic_p05', 'eta_analytic_median', 'eta_analytic_p95'):
            self.assertEqual(first[field], second[field])
        self.assertNotEqual(first['eta_draws'], second['eta_draws'][:1])

    def test_05_stored_observation_identity(self):
        query = copy.deepcopy(FIXTURE)
        query['target_gamma'] = query['observed_gamma'][::-1]
        got = API.generate(MODEL, query, draws=3, seed=5)
        self.assertEqual(got['target_is_observed'], [True] * 6)
        for row in got['eta_draws']:
            self.assertEqual(row, query['observed_eta'][::-1])
        for field in ('eta_analytic_p05', 'eta_analytic_median', 'eta_analytic_p95'):
            self.assertEqual(got[field], query['observed_eta'][::-1])
        self.assertFalse(got['is_measured_data'])
        self.assertIn('not new measurements', got['data_role'])

    def test_06_sequential_vs_simultaneous_same_joint(self):
        from joint_code_gaussian import CodeReference
        from coordinate_reference import Gaussian, condition_joint
        missing = complete(MODEL, FIXTURE)['unknown']
        coordinates = np.log(np.r_[FIXTURE['observed_gamma'], np.asarray(FIXTURE['target_gamma'])[missing]])
        m0, k0, c, m, k = joint(MODEL, coordinates)
        truthless_values = np.log(FIXTURE['observed_eta'])
        expected_m, expected_v = condition(m, k, np.arange(6), truthless_values,
                                            np.arange(6, len(coordinates)))
        reference = CodeReference(m0, k0, c)
        partial_m, partial_v = reference.condition(MODEL['theta'], np.arange(3),
                            truthless_values[:3], np.arange(3, len(coordinates)))
        sequential = condition_joint(Gaussian(partial_m, partial_v), np.arange(3), truthless_values[3:])
        close('sequential_mean_against_independent_six_anchor', sequential.mean[3:], expected_m)
        close('sequential_covariance_against_independent_six_anchor', sequential.cov[3:, 3:], expected_v)
        # Public generate() remains exactly-six-anchor; no arbitrary-anchor API claim is made.

    def test_07_feature_and_residual_contracts(self):
        oracle = complete(MODEL, FIXTURE)
        f = curvature_rows(np.asarray(MODEL['q']))
        close('contrast_constant_annihilation', f @ np.ones(17), np.zeros(2), atol=1e-13, rtol=0)
        close('contrast_linear_annihilation', f @ np.asarray(MODEL['q']), np.zeros(2), atol=1e-13, rtol=0)
        residual = oracle['k0'] - oracle['cross'] @ oracle['cross'].T
        self.assertGreater(float(np.linalg.eigvalsh(residual).min()), 0.)
        self.assertGreater(float(np.linalg.eigvalsh(oracle['unknown_covariance']).min()), 0.)

    def test_08_invalid_input_rejections(self):
        cases = []
        for field in ('observed_gamma', 'observed_eta', 'target_gamma'):
            for bad in (0., -1., True, float('nan'), float('inf'), 'bad'):
                value = copy.deepcopy(FIXTURE)
                value[field][0] = bad
                cases.append((field + '_' + repr(bad), value))
        for field in ('observed_gamma', 'target_gamma'):
            value = copy.deepcopy(FIXTURE)
            value[field][1] = value[field][0]
            cases.append((field + '_duplicate', value))
        for field in ('observed_gamma', 'observed_eta'):
            value = copy.deepcopy(FIXTURE)
            value[field] = value[field][:-1]
            cases.append((field + '_length5', value))
        value = copy.deepcopy(FIXTURE); value['target_gamma'] = []; cases.append(('empty_target', value))
        value = copy.deepcopy(FIXTURE); value['target_gamma'][0] = 1e20; cases.append(('outside_support', value))
        value = copy.deepcopy(FIXTURE); value['target_gamma'][0] = 1e-20; cases.append(('below_support', value))
        value = copy.deepcopy(FIXTURE); value['target_eta'] = [1.]; cases.append(('hidden_target_value_field', value))
        value = copy.deepcopy(FIXTURE); del value['observed_eta']; cases.append(('missing_field', value))
        for label, value in cases:
            with self.subTest(case=label):
                with self.assertRaises((ValueError, TypeError)):
                    API.generate(MODEL, value, draws=1, seed=2)
        for draws, seed in ((0, 1), (True, 1), (1., 1), (1, -1), (1, True), (1, 2.)):
            with self.subTest(draws=draws, seed=seed):
                with self.assertRaises(ValueError):
                    API.generate(MODEL, FIXTURE, draws=draws, seed=seed)

    def test_09_json_parser_and_cli_no_clobber(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    API.parse(text.encode('utf-8'))
        with tempfile.TemporaryDirectory(prefix='hydrogel_qualification_') as folder:
            folder = Path(folder)
            input_path, output = folder / 'input.json', folder / 'output.json'
            input_path.write_text(json.dumps(FIXTURE), encoding='utf-8')
            command = [sys.executable, str(ROOT / 'runtime/deployment/generate_curve.py'),
                       '--model', str(ROOT / 'models/SHARED_MODEL.json'), '--input', str(input_path),
                       '--output', str(output), '--draws', '1', '--seed', '6']
            first = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = output.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(before, output.read_bytes())
            saved = json.loads(before)
            self.assertEqual(saved['model_sha256'], sha(ROOT / 'models/SHARED_MODEL.json'))
            self.assertEqual(saved['input_sha256'], sha(input_path))

    def test_10_support_endpoints_and_log_alias_guard(self):
        query = copy.deepcopy(FIXTURE)
        query['target_gamma'] = np.exp(np.asarray(MODEL['q'])[[0, -1]]).tolist()
        got = API.generate(MODEL, query, draws=1, seed=7)
        expected = complete(MODEL, query)
        for j, field in enumerate(('eta_analytic_p05', 'eta_analytic_median', 'eta_analytic_p95')):
            close(field + '_closed_support_endpoints', got[field], expected['quantiles'][j])
        # Distinct physical inputs can become the same binary floating-point log.
        # They must be rejected instead of being silently treated as a replicate.
        alias = np.nextafter(FIXTURE['observed_gamma'][0], np.inf)
        self.assertNotEqual(alias, FIXTURE['observed_gamma'][0])
        if np.log(alias) != np.log(FIXTURE['observed_gamma'][0]):
            self.skipTest('This platform log does not alias the selected nextafter pair')
        query['target_gamma'] = [float(alias)]
        with self.assertRaisesRegex(ValueError, 'alias'):
            API.generate(MODEL, query, draws=1, seed=7)


class RecordedResult(unittest.TextTestResult):
    def addSuccess(self, test):
        super().addSuccess(test)
        RECORDS.append({'check': test.id().split('.')[-1], 'passed': True})

    def addFailure(self, test, err):
        super().addFailure(test, err)
        RECORDS.append({'check': test.id().split('.')[-1], 'passed': False,
                        'error_type': err[0].__name__, 'message': str(err[1])})

    def addError(self, test, err):
        super().addError(test, err)
        RECORDS.append({'check': test.id().split('.')[-1], 'passed': False,
                        'error_type': err[0].__name__, 'message': str(err[1])})


def main():
    global ROOT, MODEL, FIXTURE, API
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ROOT = args.repo_root.resolve()
    MODEL = json.loads((ROOT / 'models/SHARED_MODEL.json').read_text('utf-8'))
    here = Path(__file__).resolve().parent
    FIXTURE = json.loads((here / 'SYNTHETIC_INPUT.json').read_text('utf-8'))
    paths = sorted((ROOT / 'runtime').rglob('*.py')) + [ROOT / 'models/SHARED_MODEL.json']
    bindings = [{'file': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in paths]
    spec = importlib.util.spec_from_file_location('frozen_hydrogel_generate', ROOT / 'runtime/deployment/generate_curve.py')
    API = importlib.util.module_from_spec(spec); spec.loader.exec_module(API)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrozenInferenceTests)
    result = unittest.TextTestRunner(stream=stream, resultclass=RecordedResult, verbosity=2).run(suite)
    unchanged = all(sha(ROOT / item['file']) == item['sha256'] for item in bindings)
    report = {'schema_version': 1, 'actual_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'PASS' if result.wasSuccessful() and unchanged else 'FAIL',
              'scope': 'Synthetic software and numerical qualification only; zero material efficacy/calibration/batch/cost evidence.',
              'material_fits': 0, 'material_predictions': 0, 'material_scores': 0,
              'raw_material_reads': 0, 'hidden_S_reads': 0, 'new_measurements': 0,
              'synthetic_test_methods_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'runtime_and_model_unchanged': unchanged,
              'runtime_bindings': bindings, 'checker_bindings': [
                  {'file': name, 'sha256': sha(here / name)} for name in
                  ('verify_frozen_inference.py', 'independent_gaussian.py', 'SYNTHETIC_INPUT.json')],
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'checks': RECORDS, 'unittest_output': stream.getvalue(),
              'limitations': ['No original outer-fold law replay or new material validation.',
                  'The oracle does not import production numerical helpers; API calls are the implementation under test.',
                  'This is computational independence of formulas, not independent human authorship.',
                  'Target batching checks marginals, not preservation of cross-batch random dependence.',
                  'Fixed-seed Cholesky samples need not be permutation-equivariant.',
                  'Only the current same_local_code_jointGaussian_v1 deployment entry point is qualified.']}
    text = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as file:
            file.write(text)
    print(json.dumps({'status': report['status'], 'tests': result.testsRun, 'failures': len(result.failures),
                      'errors': len(result.errors), 'model_unchanged': unchanged}))
    if not result.wasSuccessful():
        print(stream.getvalue(), file=sys.stderr)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
