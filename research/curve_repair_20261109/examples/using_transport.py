"""SOURCE example: draw a fixed-O joint log-viscosity curve from a saved law.

Does not run a study, refit, read CSV/targets, update model parameters, or score
new data. Requires an exact-SHA saved expansion PREDICTIONS.json. Samples are
model-generated conditional log(eta/(1 Pa.s)), never new measurements.

Example (replace every uppercase placeholder with the actual preserved value):
  python -X utf8 -B using_transport.py --predictions PREDICTIONS_JSON \
    --expected-predictions-sha256 SHA --member EXACT_MEMBER \
    --method copula_frozen_curvature_posterior --samples 1024 --seed 20261109 \
    --output NEW_OUTPUT_DIRECTORY

The seed controls only this demonstration's generated draws; analytical study
scores are unchanged. Do not search seeds and call the selected draws new data.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
COPULA_SHA = '0db9e373dbbe35e2efa805606b9edcb065cb2d920b3f3745ae2fcf6e9529acc4'
METHODS = ('physical_full_student', 'copula_independence', 'copula_physical_student_shape',
           'copula_frozen_gp_posterior', 'copula_frozen_curvature_posterior',
           'gaussian_frozen_gp', 'gaussian_frozen_curvature')


def load_json_bytes(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data.decode('utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))


def load_copula(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != COPULA_SHA:
        raise ValueError('Use the exact preserved pure GaussianCopulaStudent source')
    name = '_mlst_sample_example_copula_' + COPULA_SHA
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def draw_saved_law(forecast, count, rng, copula_path):
    family = forecast['distribution']
    if family == 'gaussian_copula_student4_marginals':
        module = load_copula(copula_path)
        law = module.GaussianCopulaStudent(forecast['location'], forecast['scale'],
                                          forecast['correlation'], forecast['df'])
        return law.sample(count, rng)
    if family == 'multivariate_student_t':
        location = np.asarray(forecast['location'], dtype=float)
        shape = np.asarray(forecast['scale_matrix'], dtype=float)
        df = forecast['df']
        if df != 4 or shape.shape != (8, 8) or location.shape != (8,):
            raise ValueError('The preserved full physical Student4 law is required')
        lower = np.linalg.cholesky(shape)
        # One chi-square scale per complete 8-vector preserves the multivariate
        # Student law. Independent marginal t samples would change dependence.
        normal = rng.standard_normal((count, 8)) @ lower.T
        return location + normal * np.sqrt(df/rng.chisquare(df, count))[:, None]
    if family == 'multivariate_gaussian':
        mean = np.asarray(forecast['mean'], dtype=float)
        covariance = np.asarray(forecast['covariance'], dtype=float)
        if covariance.shape != (8, 8) or mean.shape != (8,):
            raise ValueError('The preserved full8D Gaussian law is required')
        return mean + rng.standard_normal((count, 8)) @ np.linalg.cholesky(covariance).T
    raise ValueError('Unknown saved-law distribution')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--expected-predictions-sha256', required=True)
    parser.add_argument('--member', required=True, help='Exact original member identity from the saved payload')
    parser.add_argument('--method', choices=METHODS, default='copula_frozen_curvature_posterior')
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--copula-source', type=Path, default=ROOT/'marginal_transport'/'pure.py')
    args = parser.parse_args(argv)
    if args.samples <= 0 or args.seed < 0:
        raise ValueError('Positive sample count and nonnegative explicit seed required')
    raw = args.predictions.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != args.expected_predictions_sha256:
        raise ValueError('Saved PREDICTIONS.json does not match the explicitly expected SHA256')
    saved = load_json_bytes(raw)
    if saved.get('schema') != 'mlst-resin-expansion-predictions-v1':
        raise ValueError('This example accepts only the preserved expansion saved-law schema')
    selected = [row for row in saved['forecasts']
                if row['member'] == args.member and row['method_id'] == args.method]
    if len(selected) != 1:
        raise ValueError('Select exactly one saved curve/method; no automatic best-method selection')
    forecast = selected[0]
    if (forecast['role'] != 'future_expansion' or forecast['anchor_indices'] != [0,1,13,14,15,16] or
            forecast['target_indices'] != list(range(17,25))):
        raise ValueError('Preserved role and fixed observed/target masks are required')
    rng = np.random.Generator(np.random.PCG64(args.seed))
    draws = draw_saved_law(forecast, args.samples, rng, args.copula_source)
    if draws.shape != (args.samples, 8) or not np.isfinite(draws).all():
        raise ArithmeticError('Generated joint log-viscosity draws must be finite')
    args.output.mkdir(parents=True, exist_ok=False)
    array_path = args.output/'GENERATED_LOG_ETA.npz'
    with array_path.open('xb') as stream:
        np.savez_compressed(stream, generated_log_eta=draws,
                            target_indices=np.asarray(forecast['target_indices'], dtype=int))
    metadata = {'actual_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'kind': 'MODEL_GENERATED_CONDITIONAL_DRAWS_NOT_MEASUREMENTS',
                'predictions_sha256': digest, 'member': args.member,
                'recipe': forecast['recipe'], 'role': forecast['role'],
                'method_id': args.method, 'distribution': forecast['distribution'],
                'samples': args.samples, 'seed': args.seed, 'algorithm': 'PCG64',
                'coordinate_system': 'ln(eta/(1 Pa.s))',
                'shape': list(draws.shape), 'npz_key': 'generated_log_eta',
                'generated_array_sha256': hashlib.sha256(array_path.read_bytes()).hexdigest(),
                'anchor_indices': forecast['anchor_indices'], 'target_indices': forecast['target_indices'],
                'new_material_reads': 0, 'new_fits': 0, 'new_scoring': 0,
                'no_arithmetic_eta_mean_reported': True,
                'warning_scope': 'Fixed observed-set joint draws. Log-Student has no finite viscosity arithmetic mean; generated values are not new validation data.'}
    with (args.output/'GENERATED_METADATA.json').open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'output':str(args.output.resolve()), 'samples':args.samples,
                      'coordinate_system':metadata['coordinate_system'],
                      'kind':metadata['kind']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
