"""Synthetic-only qualification and one cold-start call to frozen code.fit.

This demonstrates a training primitive, not GP training, material validation or
complete replay of the paper. No private material cache is accepted or opened.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
sys.dont_write_bytecode = True
import numpy as np
import scipy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_moments(fixture, theta):
    """Explicit covariance replacement; does not call production unpack/joint."""
    theta = np.asarray(theta, dtype=float)
    mean = np.asarray(fixture['reference']['mean'])
    covariance = np.asarray(fixture['reference']['covariance'])
    cross = np.asarray(fixture['reference']['cross'])
    diagonal0, off_diagonal, diagonal1 = np.exp(theta[2]), theta[3], np.exp(theta[4])
    s = np.array([[diagonal0 ** 2, diagonal0 * off_diagonal],
                  [diagonal0 * off_diagonal, off_diagonal ** 2 + diagonal1 ** 2]])
    ds = [np.array([[2 * diagonal0 ** 2, diagonal0 * off_diagonal],
                    [diagonal0 * off_diagonal, 0.]]),
          np.array([[0., diagonal0], [diagonal0, 2 * off_diagonal]]),
          np.array([[0., 0.], [0., 2 * diagonal1 ** 2]])]
    return mean + cross @ theta[:2], covariance + cross @ (s - np.eye(2)) @ cross.T, cross, ds


def independent_nll_and_gradient(mean, covariance, cross, ds, values, indices):
    """Full-matrix Gaussian log density and trace derivative; no core imports.

    The oracle forms each dK/dtheta as a full matrix, in contrast to the
    production objective's cached two-dimensional determinant/Woodbury update.
    """
    indices = np.asarray(indices, dtype=int)
    m, k = mean[indices], covariance[np.ix_(indices, indices)]
    c = cross[indices]
    residual = np.asarray(values)[indices] - m
    sign, logdet = np.linalg.slogdet(k)
    if sign <= 0:
        raise ValueError('Synthetic oracle covariance is not positive definite')
    precision = np.linalg.solve(k, np.eye(len(k)))
    weighted = precision @ residual
    value = .5 * (len(k) * np.log(2 * np.pi) + logdet + residual @ weighted)
    weight = .5 * (precision - np.outer(weighted, weighted))
    gradient = np.r_[-c.T @ weighted,
                     [np.sum(weight * (c @ derivative @ c.T)) for derivative in ds]]
    return float(value), gradient


def independent_raw_objective(fixture, theta, ridge):
    mean, covariance, cross, ds = independent_moments(fixture, theta)
    full_indices = np.arange(len(mean))
    losses, gradients = [], []
    masks = list(fixture['masks'].values())
    for curve in fixture['complete_synthetic_curves']:
        full, gfull = independent_nll_and_gradient(mean, covariance, cross, ds, curve, full_indices)
        contexts = [independent_nll_and_gradient(mean, covariance, cross, ds, curve, o) for o in masks]
        losses.append(full - np.mean([v for v, g in contexts]))
        gradients.append(gfull - np.mean([g for v, g in contexts], axis=0))
    theta = np.asarray(theta)
    return float(np.mean(losses) + ridge * theta @ theta), np.mean(gradients, axis=0) + 2 * ridge * theta


def qualify_points(core, fixture, objective, protocol):
    points = []
    raw_zero, _ = independent_raw_objective(fixture, np.zeros(5), protocol['ridge'])
    for theta in protocol['fixed_parameter_points']:
        raw, gradient = independent_raw_objective(fixture, theta, protocol['ridge'])
        value, core_gradient = objective(theta)
        expected = raw - raw_zero
        value_error = abs(float(value - expected))
        gradient_error = float(np.max(np.abs(gradient - core_gradient)))
        passed = value_error <= protocol['value_absolute_tolerance'] and gradient_error <= protocol['gradient_absolute_tolerance']
        points.append({'theta': theta, 'independent_raw_objective': raw,
                       'independent_centered_objective': expected, 'core_centered_objective': float(value),
                       'independent_gradient': gradient.tolist(), 'core_gradient': core_gradient.tolist(),
                       'absolute_value_error': value_error, 'gradient_max_absolute_error': gradient_error,
                       'passed': passed})
    return points, raw_zero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('An existing receipt is never overwritten or rerun implicitly')
    start, clock = datetime.now(timezone.utc), time.perf_counter()
    here = Path(__file__).resolve().parent
    protocol_path, input_path = here / 'SOFTWARE_TRAINING_PROTOCOL.json', here / 'SYNTHETIC_TRAINING_INPUT.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    fixture = json.loads(input_path.read_text('utf-8'))
    source_path = args.repo_root.resolve() / 'runtime/core/joint_code_gaussian.py'
    if sha(source_path) != protocol['frozen_core_sha256']:
        raise ValueError('Frozen production core differs from protocol')
    if sha(input_path) != protocol['synthetic_input_sha256'] or sha(__file__) != protocol['demo_source_sha256']:
        raise ValueError('Synthetic input or demo source differs from pre-run protocol')
    spec = importlib.util.spec_from_file_location('synthetic_training_frozen_core', source_path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    if (core.RIDGE != protocol['ridge'] or [list(v) for v in core.BOUNDS] != protocol['bounds']
            or core.OPTIONS != protocol['optimizer_options'] or core.PG_GATE != protocol['projected_gradient_gate']):
        raise ValueError('Frozen fit settings differ from the software protocol')
    if fixture['role'] != 'synthetic_only_training_primitive' or fixture['new_measurements'] != 0:
        raise ValueError('This entry point accepts only its explicit synthetic fixture')
    ref = fixture['reference']
    reference = core.CodeReference(ref['mean'], ref['covariance'], ref['cross'])
    tasks = [core.CurveTask(reference, y, list(fixture['masks'].values()))
             for y in fixture['complete_synthetic_curves']]
    objective = core.ConditionalObjective(tasks)
    points, independent_zero = qualify_points(core, fixture, objective, protocol)
    receipt = {'schema_version': 1, 'actual_start_utc': start.isoformat(),
               'role': 'synthetic_only_training_primitive', 'is_material_validation': False,
               'new_measurements': 0, 'material_fit_calls': 0, 'GP_fit_calls': 0,
               'material_scoring_calls': 0, 'raw_material_reads': 0, 'hidden_S_reads': 0,
               'input_sha256': sha(input_path), 'source_sha256': sha(__file__),
               'protocol_sha256': sha(protocol_path), 'frozen_core_sha256': sha(source_path),
               'synthetic_curve_count': len(tasks), 'response_dimension': len(ref['mean']),
               'masks_per_curve': len(fixture['masks']), 'fixed_point_checks': points,
               'independent_raw_zero': independent_zero, 'core_removed_constant': objective.constant,
               'removed_constant_absolute_error': abs(independent_zero - objective.constant),
               'synthetic_correction_fit_calls': 0,
               'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
               'settings_unchanged': True, 'retries': 0, 'sample_or_seed_tuning': False,
               'fit_role': 'One frozen bounded cold-start optimizer demonstration only, not evidence of material efficacy.'}
    if not all(p['passed'] for p in points):
        receipt['status'] = 'STOPPED_BEFORE_FIT_ORACLE_MISMATCH'
    elif receipt['removed_constant_absolute_error'] > protocol['value_absolute_tolerance']:
        receipt['status'] = 'STOPPED_BEFORE_FIT_CONSTANT_MISMATCH'
    else:
        receipt['actual_fit_call_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['synthetic_correction_fit_calls'] = 1
        try:
            receipt['fit'] = core.fit(tasks)  # Exactly one original zero cold start.
            receipt['status'] = 'SYNTHETIC_PRIMITIVE_ADMITTED'
        except core.FitFailure as error:
            receipt['status'] = 'SYNTHETIC_PRIMITIVE_NOT_ADMITTED'
            receipt['fit'] = error.receipt
        except Exception as error:
            receipt['status'] = 'SYNTHETIC_FIT_EXCEPTION_NO_RETRY'
            receipt['fit_exception'] = {'type': type(error).__name__, 'message': str(error)}
    receipt['frozen_core_unchanged'] = sha(source_path) == protocol['frozen_core_sha256']
    receipt['actual_end_utc'] = datetime.now(timezone.utc).isoformat()
    receipt['elapsed_seconds'] = time.perf_counter() - clock
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(receipt, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': receipt['status'], 'fixed_points': len(points),
                      'max_value_error': max(p['absolute_value_error'] for p in points),
                      'max_gradient_error': max(p['gradient_max_absolute_error'] for p in points),
                      'synthetic_fit_calls': receipt['synthetic_correction_fit_calls'],
                      'core_unchanged': receipt['frozen_core_unchanged']}))
    return 0 if receipt['status'] == 'SYNTHETIC_PRIMITIVE_ADMITTED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
