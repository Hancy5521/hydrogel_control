"""One bounded, anchor-only shared update and a cross-recipe early harm check.

Separate from the zero-fit material pilot. No source CSV or new response is
read. A single projected gradient step is learning, even though no optimizer
loop runs. The declared sentinel is development information, not confirmation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PANEL = list(range(17, 25))
CONTEXT = [0, 1, 3, 4]
CHECK = [2, 5]
STEP = 0.05
BOUNDS = np.array([[-1., 1.], [-1., 1.], [-.75, .75], [-1., 1.], [-.75, .75]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError('Duplicate JSON key: ' + k)
            out[k] = v
        return out
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=unique,
                      parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))


def write_new(path, payload):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def authorize(path, phase):
    reg = read(path)
    if reg['schema'] != 'mlst-resin-early-update-registration-v1' or phase not in reg['authorized_phases']:
        raise PermissionError('Registered bounded early-update phase required')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'])
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture UTC required')
    for name, expected in reg['source_pins'].items():
        if sha(name) != expected:
            raise PermissionError('Changed registered source: ' + name)
    for name, expected in reg['input_pins'].items():
        if sha(name) != expected:
            raise PermissionError('Changed frozen input: ' + name)
    return reg


def predict(registration, output):
    reg = authorize(registration, 'predict')
    sys.path.insert(0, str(ROOT))
    from core.frozen_loader import load_frozen_core
    from core.arbitrary_mask import ArbitraryMaskConditioner, CurveIdentity, MaskRoles, ObservedReadings
    from core.early_influence import ScoringReadings, GaussianConditionalScore, inspect_parameter_update
    frozen = load_frozen_core(reg.get('frozen_core_path'))
    prepared, existing = read(reg['prepare']), read(reg['main_predictions'])
    if existing['prepare_sha256'] != sha(reg['prepare']):
        raise ValueError('Main predictions do not bind prepared references')
    records = [r for r in existing['forecasts']
               if r['method_id'] == 'curvature_frozen_theta__fixed_first6']
    if len(records) != 15 or any(r['anchor_indices'] != list(range(6)) for r in records):
        raise ValueError('Exactly the original fixed-six observations required')
    models = {c['member']: c for c in prepared['curves']}
    train, sentinel, tasks, payload_curves = [], [], {}, []
    theta = None
    for row in records:
        source = models[row['member']]
        current = np.asarray(source['laws']['curvature_frozen_theta']['theta'])
        if theta is None:
            theta = current
        if not np.array_equal(theta, current):
            raise ValueError('One shared frozen baseline theta required')
        saved = source['reference']
        reference = frozen.CodeReference(np.array(saved['mean']),
                    np.array(saved['covariance']), np.array(saved['cross']))
        if row['thermal_history'] == 'ascending_35_first':
            split, role = 'train', 'pseudo_hole_train'
        elif row['thermal_history'] == 'descending_35_last':
            split, role = 'dev', 'sentinel_dev'
        else:
            raise ValueError('Unknown registered thermal history')
        identity = CurveIdentity('10.5281/zenodo.19699588', row['recipe'], row['member'], split)
        rest = [i for i in range(25) if i not in CONTEXT + CHECK]
        roles = MaskRoles(identity, 25, CONTEXT,
                          CHECK if split == 'train' else [],
                          CHECK if split == 'dev' else [], rest)
        observed = ObservedReadings(identity, CONTEXT,
                    [row['anchor_values'][i] for i in CONTEXT], 0.0)
        task = GaussianConditionalScore(ArbitraryMaskConditioner(reference, roles, observed),
                    ScoringReadings(identity, role, CHECK,
                    [row['anchor_values'][i] for i in CHECK], 0.0))
        (train if split == 'train' else sentinel).append(task)
        tasks[row['member']] = (identity, reference, row)
        payload_curves.append({k: row[k] for k in ['member', 'recipe', 'run_id', 'thermal_history',
                                                  'anchor_indices', 'anchor_values']})
    if len(train) != 10 or len(sentinel) != 5:
        raise ValueError('Exactly two training recipes and one sentinel recipe required')
    training_initial = [task(theta) for task in train]
    gradient = np.mean([v[1] for v in training_initial], axis=0)
    norm = float(np.linalg.norm(gradient))
    candidate = np.clip(theta - STEP * gradient / norm, BOUNDS[:, 0], BOUNDS[:, 1]) if norm else theta.copy()
    delta = candidate - theta
    report = inspect_parameter_update(theta, train, sentinel, direction=delta,
              step_sizes=[0., 1.], evaluate_development_steps=True,
              development_use_id='resin-3dev-anchor-only-cross-recipe-v1')
    accept = bool(report['development_directional_derivative'] < 0.)
    chosen = candidate if accept else theta
    forecasts = []
    for member, (identity, reference, row) in tasks.items():
        roles = MaskRoles(identity, 25, list(range(6)), [], [], list(range(6, 25)))
        observed = ObservedReadings(identity, list(range(6)), row['anchor_values'], 0.0)
        conditioner = ArbitraryMaskConditioner(reference, roles, observed)
        for name, parameter in [('base', theta), ('candidate', candidate), ('gated', chosen)]:
            posterior = conditioner.condition(parameter, target_indices=PANEL)
            mean, cov = posterior.marginal()
            np.linalg.cholesky(cov)
            forecasts.append({'member': member, 'recipe': row['recipe'],
                     'thermal_history': row['thermal_history'], 'method_id': name,
                     'theta': parameter.tolist(), 'anchor_indices': list(range(6)),
                     'target_indices': PANEL, 'mean': mean.tolist(), 'covariance': cov.tolist()})
    if len(forecasts) != 45:
        raise ValueError('Exactly 45 forecasts required')
    write_new(output, {'schema': 'mlst-resin-early-update-predictions-v1', 'actual_utc': utc(),
        'registration_sha256': sha(registration), 'prepare_sha256': sha(reg['prepare']),
        'main_predictions_sha256': sha(reg['main_predictions']), 'base_theta': theta.tolist(),
        'bounds': BOUNDS.tolist(), 'step_length': STEP, 'context_indices': CONTEXT,
        'check_indices': CHECK, 'training_members': [s.identity.curve_id for s in train],
        'sentinel_members': [s.identity.curve_id for s in sentinel],
        'candidate_theta': candidate.tolist(), 'delta': delta.tolist(),
        'training_gradient_for_step': gradient.tolist(), 'training_gradient_norm': norm,
        'early_report': report, 'gate_accept': accept, 'gate_rule': 'g_dev dot delta < 0',
        'gate_uses_exact_endpoint': False, 'curves': payload_curves, 'forecasts': forecasts,
        'parameter_update_steps': 1, 'optimizer_loop_runs': 0, 'step_search': False,
        'actual_score_calls': {'train_initial': 10, 'train_report': 10,
                               'sentinel_base_and_endpoint': 10},
        'new_source_eta_access': 0, 'target_labels_accessed': 0,
        'development_not_independent_confirmation': True,
        'thermal_order_and_recipe_confounded': True})


def score(registration, output, predictions, targets, release):
    reg = authorize(registration, 'score')
    permit = read(release)
    if (permit.get('authorization') != 'SCORE_FROZEN_EARLY_UPDATE_ONLY' or
        permit.get('predictions_sha256') != sha(predictions) or
        permit.get('targets_sha256') != sha(targets)):
        raise PermissionError('Exact frozen-prediction target release required')
    update, truth = read(predictions), read(targets)
    if (update['registration_sha256'] != sha(registration) or
        truth['predictions_sha256'] != sha(reg['main_predictions'])):
        raise ValueError('Cross-experiment binding mismatch')
    sys.path.insert(0, str(ROOT / 'material_stress'))
    from resin_experiment import _gaussian_scores
    lookup = {r['member']: r for r in truth['curves']}
    rows = []
    for row in update['forecasts']:
        values, _ = _gaussian_scores(row['mean'], row['covariance'], lookup[row['member']]['values'])
        rows.append({**{k: row[k] for k in ['member', 'recipe', 'thermal_history', 'method_id']}, **values})
    groups = []
    for recipe in sorted({r['recipe'] for r in rows}):
        for name in ['base', 'candidate', 'gated']:
            subset = [r for r in rows if r['recipe'] == recipe and r['method_id'] == name]
            groups.append({'recipe': recipe, 'method_id': name, 'runs': len(subset),
                 'thermal_history': subset[0]['thermal_history'],
                 **{k: float(np.mean([r[k] for r in subset])) for k in
                    ['crps', 'joint_nll_per_dimension', 'log_rmse', 'coverage90', 'width90']}})
    write_new(output, {'schema': 'mlst-resin-early-update-scores-v1', 'actual_utc': utc(),
         'registration_sha256': sha(registration), 'predictions_sha256': sha(predictions),
         'targets_sha256': sha(targets), 'release_sha256': sha(release),
         'gate_accept': update['gate_accept'], 'early_report': update['early_report'],
         'recipe_scores': groups, 'curve_scores': rows,
         'emphasis': 'held-out-from-update sentinel recipe tail; still development',
         'new_source_eta_access': 0, 'new_fits_in_score': 0,
         'optimization_steps_in_prediction_phase': 1,
         'no_retuning_on_revealed_targets': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['predict', 'score'])
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing output; do not overwrite/retry')
    start = time.perf_counter()
    if args.phase == 'predict':
        predict(args.registration, args.output)
    else:
        if any(p is None for p in [args.predictions, args.targets, args.release]):
            raise PermissionError('Explicit frozen predictions, targets, root release required')
        score(args.registration, args.output, args.predictions, args.targets, args.release)
    print(json.dumps({'phase': args.phase, 'status': 'COMPLETE', 'seconds': time.perf_counter()-start,
                      'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
