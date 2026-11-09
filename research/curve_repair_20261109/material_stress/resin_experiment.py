"""Four explicit, once-only phases for the frozen-model resin development pilot.

SOURCE PREPARATION IS NOT EXPERIMENT EXECUTION AUTHORIZATION. Root must pin
this source, all imported local sources, the immutable protocol and inputs in a
separate actual-UTC registration. There is no optimizer or parameter update.
The native resin curves are a polymer-transfer stress test, not hydrogel wet
confirmation. Future/final recipe responses stay outside the reader API.

CLI, always with --registration REG --output OUT:
  prepare : native-grid frozen laws, OS seed and all masks; no response values
  predict : each method's own six dev anchors -> 210 frozen Gaussian forecasts
  reveal  : only after independent saved-forecast PASS and root target release
  score   : immutable predictions plus released common eight dev targets

Each phase creates a new directory exclusively. Partial failure is retained;
the script refuses to overwrite or silently retry an interrupted phase.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr

from resin_io import (INPUT_SHA256, SOURCE_SHA256, RestrictedResinReader,
                      assign_recipe_roles, load_inputs, read_json, sha, utc,
                      write_new)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = HERE / 'RESIN_STRESS_PROTOCOL.json'
PANEL = list(range(17, 25))
REFERENCE_IDS = ('gp_theta0', 'curvature_frozen_theta')
POLICIES = ('fixed_first6', 'spread6', 'random_uniform6',
            'max_variance6', 'expected_crps6')


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _required_sources():
    return sorted(list(HERE.glob('*.py')) + list((ROOT / 'core').glob('*.py')))


def _authorize(registration, phase):
    reg = read_json(registration)
    if reg.get('schema') != 'mlst-resin-root-registration-v1':
        raise PermissionError('Separate root registration schema required')
    if phase not in reg.get('authorized_phases', []):
        raise PermissionError('Phase is not listed in root authorization')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture timezone-aware root UTC required')
    protocol = read_json(PROTOCOL)
    if reg['protocol_id'] != protocol['protocol_id'] or reg['protocol_sha256'] != sha(PROTOCOL):
        raise PermissionError('Protocol differs from root registration')
    if (reg['input_sha256'] != INPUT_SHA256 or
            reg['source_zip_sha256'] != SOURCE_SHA256):
        raise PermissionError('Registered immutable input/source identity mismatch')
    pins = {str(Path(name).resolve()).casefold(): value
            for name, value in reg['source_pins'].items()}
    for path in _required_sources():
        if pins.get(str(path.resolve()).casefold()) != sha(path):
            raise PermissionError('Missing/changed registered source: ' + str(path))
    # Verify original source and input bytes before any source-dependent action.
    if sha(protocol['source_zip']) != SOURCE_SHA256:
        raise ValueError('Original source ZIP changed')
    inputs = load_inputs(protocol['input_masks'])
    expected = {'anchors_0based': [0, 1], 'extra_query_pool_0based': list(range(2, 17)),
                'panelT_0based': PANEL, 'extra_queries': 4,
                'labels_per_method_curve': 6, 'spread_extra_0based': [2, 7, 11, 16],
                'policies': list(POLICIES), 'references': list(REFERENCE_IDS),
                'planned_gaussians': 210, 'common_target_cells': 120,
                'CRPS_point_terms': 1680,
                'nearest_target6_anchors_0based': [0, 1, 13, 14, 15, 16],
                'geometric_controls': ['GP_nearest_target6', 'curv_nearest_target6']}
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError('The source protocol no longer describes the bounded pilot')
    return reg, protocol, inputs


@contextmanager
def _phase_dir(output, phase, registration):
    path = Path(output) / phase
    path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write_new(path / 'STARTED.json', {
        'schema': 'mlst-resin-phase-start-v1', 'phase': phase,
        'actual_utc': utc(), 'registration_sha256': sha(registration),
        'python': platform.python_version(), 'numpy': np.__version__,
        'execution_command': sys.argv, 'new_fits': 0})
    try:
        yield path, start
    except BaseException as error:
        write_new(path / 'FAILED.json', {
            'actual_utc': utc(), 'phase': phase, 'error_type': type(error).__name__,
            'message': str(error), 'seconds': time.perf_counter() - start,
            'decision': 'STOP_NO_REPLACEMENT_KEEP_PARTIAL_NO_AUTOMATIC_RETRY'})
        raise


def _complete(path, payload_name, start, **facts):
    write_new(path / 'COMPLETE.json', {
        'schema': 'mlst-resin-phase-complete-v1', 'status': 'COMPLETE',
        'actual_utc': utc(), 'payload': payload_name,
        'payload_sha256': sha(path / payload_name),
        'seconds': time.perf_counter() - start, 'new_fits': 0, **facts})


def _prior(output, phase, payload_name, registration):
    path = Path(output) / phase
    complete = read_json(path / 'COMPLETE.json')
    if complete.get('status') != 'COMPLETE' or complete['payload'] != payload_name:
        raise PermissionError('Required previous phase is incomplete')
    if complete['payload_sha256'] != sha(path / payload_name):
        raise ValueError('Completed phase payload bytes changed')
    payload = read_json(path / payload_name)
    if payload['registration_sha256'] != sha(registration):
        raise ValueError('Previous phase belongs to another root registration')
    return payload


def _covariance_after(k, observed):
    """Input-only Schur covariance; there are deliberately no dummy Y values."""
    k = np.asarray(k, dtype=float)
    observed = list(observed)
    koo = k[np.ix_(observed, observed)]
    factor = cho_factor(koo, lower=True)
    out = k - k[:, observed] @ cho_solve(factor, k[observed, :])
    out = (out + out.T) / 2
    out[observed, :], out[:, observed] = 0.0, 0.0
    return out


def _greedy_mask(k, policy, query_effect):
    anchors, trace = [0, 1], []
    for step in range(4):
        cov = _covariance_after(k, anchors)
        candidates = [q for q in range(2, 17) if q not in anchors]
        utilities = []
        for q in candidates:
            effect = query_effect(cov, PANEL, q)
            utility = (float(cov[q, q]) if policy == 'max_variance6'
                       else effect.weighted_expected_gain)
            utilities.append({'query_index': q, 'utility': utility,
                              'expected_gain': effect.weighted_expected_gain,
                              'target_response': effect.response.tolist(),
                              'target_variance_after': effect.variance_after.tolist()})
        selected = sorted(utilities, key=lambda row: (-row['utility'], row['query_index']))[0]
        trace.append({'step': step + 1, 'anchors_before': list(anchors),
                      'candidates': utilities, 'selected_query': selected['query_index']})
        anchors.append(selected['query_index'])
    return anchors, trace


def prepare(output, registration, protocol, inputs, *, checkpoint_root=None, pins_path=None):
    from frozen_grid_reference import build_frozen_grid_reference
    # Import the qualified pure interface only after every source hash passes.
    sys.path.insert(0, str(ROOT))
    from core.query_effect import query_effect
    roles = assign_recipe_roles(inputs)
    dev = sorted((run for run in inputs['per_run']
                  if roles[run['recipe']]['role'] == 'dev'), key=lambda run: run['member'])
    if len(dev) != 15:
        raise ValueError('Exactly 15 whole dev runs required')
    with _phase_dir(output, 'prepare', registration) as (path, start):
        # Persist the only OS seed before constructing references/masks. A failed
        # phase cannot silently draw a replacement seed or overwrite this file.
        seed = int.from_bytes(os.urandom(16), 'big')
        write_new(path / 'RANDOM_SEED.json', {'actual_utc': utc(),
                  'seed_128bit_hex': format(seed, '032x'), 'algorithm': 'PCG64',
                  'replacements_allowed': 0, 'draws_before_response_access': True})
        rng = np.random.Generator(np.random.PCG64(seed))
        curves = []
        for run in dev:
            gamma = np.asarray(run['native_gamma_sorted_per_s'], dtype=float)
            if gamma.shape != (25,) or (gamma <= 0).any() or (np.diff(gamma) <= 0).any():
                raise ValueError('Exactly 25 positive distinct native coordinates required')
            adapter_arguments = {}
            if checkpoint_root is not None:
                adapter_arguments['checkpoint_root'] = checkpoint_root
            if pins_path is not None:
                adapter_arguments['pins_path'] = pins_path
            bundle = build_frozen_grid_reference(np.log(gamma), **adapter_arguments)
            reference = bundle.reference
            random_anchors = [0, 1] + rng.choice(np.arange(2, 17), 4, replace=False).tolist()
            laws, methods = {}, []
            thetas = {'gp_theta0': bundle.theta_gp,
                      'curvature_frozen_theta': bundle.theta_curvature_gaussian}
            for reference_id in REFERENCE_IDS:
                mean, cov = reference.joint(thetas[reference_id])
                np.linalg.cholesky(cov)
                laws[reference_id] = {'theta': thetas[reference_id].tolist(),
                                     'mean': mean.tolist(), 'covariance': cov.tolist()}
                for policy in POLICIES:
                    trace = []
                    if policy == 'fixed_first6':
                        anchors = list(range(6))
                    elif policy == 'spread6':
                        anchors = [0, 1, 2, 7, 11, 16]
                    elif policy == 'random_uniform6':
                        anchors = list(random_anchors)
                    else:
                        anchors, trace = _greedy_mask(cov, policy, query_effect)
                    methods.append({'method_id': reference_id + '__' + policy,
                                    'reference_id': reference_id, 'policy': policy,
                                    'mask_from_reference_id': reference_id,
                                    'anchor_indices': anchors, 'target_indices': PANEL,
                                    'query_trace': trace})
            for name, receiver, donor in [
                    ('GP_on_curvCRPS_mask', 'gp_theta0', 'curvature_frozen_theta'),
                    ('curv_on_GPCRPS_mask', 'curvature_frozen_theta', 'gp_theta0')]:
                original = next(m for m in methods if m['method_id'] == donor + '__expected_crps6')
                methods.append({'method_id': name, 'reference_id': receiver,
                                'policy': 'cross_expected_crps6', 'mask_from_reference_id': donor,
                                'anchor_indices': list(original['anchor_indices']),
                                'target_indices': PANEL, 'query_trace': []})
            for name, reference_id in [('GP_nearest_target6', 'gp_theta0'),
                                        ('curv_nearest_target6', 'curvature_frozen_theta')]:
                methods.append({'method_id': name, 'reference_id': reference_id,
                                'policy': 'nearest_target6', 'mask_from_reference_id': reference_id,
                                'anchor_indices': [0, 1, 13, 14, 15, 16],
                                'target_indices': PANEL, 'query_trace': []})
            if len(methods) != 14:
                raise ValueError('Fourteen frozen method/mask combinations required')
            curve = {key: run[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                                             'source_member_sha256')}
            curve.update(native_gamma=gamma.tolist(), log_gamma=np.log(gamma).tolist(),
                         reference={'mean': reference.mean.tolist(),
                                    'covariance': reference.covariance.tolist(),
                                    'cross': reference.cross.tolist()},
                         laws=laws, methods=methods, adapter_metadata=bundle.metadata)
            curve['matrix_payload_sha256'] = _digest({'reference': curve['reference'], 'laws': laws})
            curves.append(curve)
        payload = {'schema': 'mlst-resin-prepared-v1', 'actual_utc': utc(),
                   'protocol_id': protocol['protocol_id'],
                   'registration_sha256': sha(registration), 'protocol_sha256': sha(PROTOCOL),
                   'source_zip_sha256': SOURCE_SHA256, 'input_sha256': INPUT_SHA256,
                   'seed_128bit_hex': format(seed, '032x'), 'roles': roles,
                   'curves': curves, 'new_fits': 0, 'numerical_eta_accessed': 0,
                   'future_final_runs_protected': 35,
                   'planning_kind': 'covariance_only_outcome_independent_offline_design'}
        write_new(path / 'PREPARED.json', payload)
        _complete(path, 'PREPARED.json', start, curves=15, forecast_masks=210,
                  numerical_eta_accessed=0)


def predict(output, registration, protocol, inputs, *, checkpoint_root=None):
    prepared = _prior(output, 'prepare', 'PREPARED.json', registration)
    sys.path.insert(0, str(ROOT))
    from core.frozen_loader import load_frozen_core
    from core.arbitrary_mask import (ArbitraryMaskConditioner, CurveIdentity,
                                     MaskRoles, ObservedReadings)
    frozen_core = (None if checkpoint_root is None else
                   Path(checkpoint_root) / 'runtime' / 'core' / 'joint_code_gaussian.py')
    original = load_frozen_core(frozen_core)
    with _phase_dir(output, 'predict', registration) as (path, start):
        reader = RestrictedResinReader(protocol['source_zip'], inputs, stage='anchors')
        forecasts = []
        try:
            for curve in prepared['curves']:
                saved = curve['reference']
                reference = original.CodeReference(np.array(saved['mean']),
                                np.array(saved['covariance']), np.array(saved['cross']))
                identity = CurveIdentity(protocol['source_doi'], curve['recipe'], curve['member'], 'dev')
                for method in curve['methods']:
                    anchors = method['anchor_indices']
                    values = reader.read_log_eta(curve['member'], anchors)
                    unassigned = sorted(set(range(25)) - set(anchors) - set(PANEL))
                    roles = MaskRoles(identity, 25, anchors, [], PANEL, unassigned)
                    readings = ObservedReadings(identity, anchors, values, 0.0)
                    theta = curve['laws'][method['reference_id']]['theta']
                    posterior = ArbitraryMaskConditioner(reference, roles, readings).condition(
                        theta, target_indices=PANEL)
                    mean, cov = posterior.marginal()
                    np.linalg.cholesky(cov)
                    row = {key: curve[key] for key in ('member', 'recipe', 'run_id', 'thermal_history')}
                    row.update({key: method[key] for key in ('method_id', 'reference_id', 'policy',
                                'mask_from_reference_id', 'anchor_indices', 'target_indices')})
                    row.update(anchor_values=values, mean=mean.tolist(), covariance=cov.tolist(),
                               full_mean=posterior.mean.tolist(),
                               full_covariance=posterior.covariance.tolist(),
                               theta=theta)
                    forecasts.append(row)
        finally:
            write_new(path / 'ACCESS.json', reader.receipt())
            reader.close()
        if len(forecasts) != 210 or reader.receipt()['target_eta_accessed']:
            raise ValueError('Bounded forecast count/access contract failed')
        payload = {'schema': 'mlst-resin-predictions-v1', 'actual_utc': utc(),
                   'registration_sha256': sha(registration),
                   'prepare_sha256': sha(Path(output) / 'prepare' / 'PREPARED.json'),
                   'protocol_id': protocol['protocol_id'], 'forecasts': forecasts,
                   'new_fits': 0, 'common_target_labels_accessed': 0,
                   'future_final_runs_protected': 35,
                   'per_method_anchor_cells': 90, 'per_curve_per_method_anchors': 6,
                   'reader_access_sha256': sha(path / 'ACCESS.json')}
        write_new(path / 'PREDICTIONS.json', payload)
        _complete(path, 'PREDICTIONS.json', start, forecasts=210, target_labels_accessed=0,
                  unique_anchor_cells=reader.receipt()['unique_eta_cells_accessed'])


def reveal(output, registration, protocol, inputs, audit_path, release_path):
    prepared = _prior(output, 'prepare', 'PREPARED.json', registration)
    predictions = _prior(output, 'predict', 'PREDICTIONS.json', registration)
    prepared_sha = sha(Path(output) / 'prepare' / 'PREPARED.json')
    predicted_sha = sha(Path(output) / 'predict' / 'PREDICTIONS.json')
    if predictions['prepare_sha256'] != prepared_sha:
        raise ValueError('Predictions are not bound to current prepared state')
    audit, release = read_json(audit_path), read_json(release_path)
    if (audit.get('status') != 'PASS' or audit.get('audited_predictions_sha256') != predicted_sha or
            audit.get('audited_prepare_sha256') != prepared_sha):
        raise PermissionError('Independent saved-forecast audit PASS bound to bytes required')
    if (release.get('schema') != 'mlst-resin-target-release-v1' or
            release.get('authorization') != 'REVEAL_ONLY_120_DEV_TARGET_CELLS' or
            release.get('predictions_sha256') != predicted_sha or
            release.get('prepare_sha256') != prepared_sha or
            release.get('audit_sha256') != sha(audit_path)):
        raise PermissionError('Separate exact-hash root target release required')
    stamp = dt.datetime.fromisoformat(release['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture root target release UTC required')
    with _phase_dir(output, 'reveal', registration) as (path, start):
        reader = RestrictedResinReader(protocol['source_zip'], inputs, stage='targets',
                                       target_release=release)
        curves = []
        try:
            for curve in prepared['curves']:
                row = {key: curve[key] for key in ('member', 'recipe', 'run_id', 'thermal_history')}
                row.update(target_indices=PANEL,
                           values=reader.read_log_eta(curve['member'], PANEL))
                curves.append(row)
        finally:
            write_new(path / 'ACCESS.json', reader.receipt())
            reader.close()
        if reader.receipt()['unique_eta_cells_accessed'] != 120 or len(curves) != 15:
            raise ValueError('The target release population differs from registration')
        write_new(path / 'TARGETS.json', {
            'schema': 'mlst-resin-targets-v1', 'actual_utc': utc(),
            'registration_sha256': sha(registration), 'prepare_sha256': prepared_sha,
            'predictions_sha256': predicted_sha, 'audit_sha256': sha(audit_path),
            'root_release_sha256': sha(release_path), 'curves': curves,
            'reader_access_sha256': sha(path / 'ACCESS.json'),
            'future_final_runs_protected': 35})
        _complete(path, 'TARGETS.json', start, unique_target_labels_accessed=120,
                  future_final_runs_protected=35)


def _gaussian_scores(mean, covariance, y):
    mean, covariance, y = np.asarray(mean), np.asarray(covariance), np.asarray(y)
    sigma = np.sqrt(np.diag(covariance))
    if (sigma <= 0).any() or not np.isfinite(sigma).all():
        raise ValueError('Positive finite predictive standard deviations required')
    error = y - mean
    z = error / sigma
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z * z) /
                    math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    lower = np.linalg.cholesky(covariance)
    standardized = np.linalg.solve(lower, error)
    nll = (len(y) * math.log(2 * math.pi) + 2 * np.log(np.diag(lower)).sum() +
           standardized @ standardized) / (2 * len(y))
    q = 1.6448536269514722
    inside = np.abs(error) <= q * sigma
    width = 2 * q * sigma
    points = [{'target_index': PANEL[i], 'truth': float(y[i]), 'mean': float(mean[i]),
               'sd': float(sigma[i]), 'crps': float(crps[i]),
               'squared_error': float(error[i] ** 2), 'coverage90': bool(inside[i]),
               'width90': float(width[i])} for i in range(8)]
    return {'crps': float(np.mean(crps)), 'joint_nll_per_dimension': float(nll),
            'log_rmse': float(np.sqrt(np.mean(error ** 2))),
            'coverage90': float(np.mean(inside)), 'width90': float(np.mean(width))}, points


def score(output, registration, protocol):
    prepared = _prior(output, 'prepare', 'PREPARED.json', registration)
    predictions = _prior(output, 'predict', 'PREDICTIONS.json', registration)
    targets = _prior(output, 'reveal', 'TARGETS.json', registration)
    if (targets['predictions_sha256'] != sha(Path(output) / 'predict' / 'PREDICTIONS.json') or
            targets['prepare_sha256'] != sha(Path(output) / 'prepare' / 'PREPARED.json')):
        raise ValueError('Released targets are not bound to these frozen predictions')
    lookup = {row['member']: row for row in targets['curves']}
    with _phase_dir(output, 'score', registration) as (path, start):
        rows, point_terms = [], []
        for forecast in predictions['forecasts']:
            target = lookup[forecast['member']]
            if target['target_indices'] != forecast['target_indices'] or target['target_indices'] != PANEL:
                raise ValueError('Same fixed target panel required for all methods')
            metrics, points = _gaussian_scores(forecast['mean'], forecast['covariance'], target['values'])
            identity = {key: forecast[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                                                      'method_id', 'reference_id', 'policy')}
            rows.append({**identity, **metrics})
            point_terms.extend({**identity, **point} for point in points)
        method_ids = sorted({row['method_id'] for row in rows})
        recipes = sorted({row['recipe'] for row in rows})
        keys = ['crps', 'joint_nll_per_dimension', 'log_rmse', 'coverage90', 'width90']
        recipe_scores, macro = [], []
        for method in method_ids:
            for recipe in recipes:
                selected = [row for row in rows if row['method_id'] == method and row['recipe'] == recipe]
                if len(selected) != 5:
                    raise ValueError('All five runs stay within each recipe/method')
                recipe_scores.append({'method_id': method, 'recipe': recipe, 'runs': 5,
                                      'thermal_history': selected[0]['thermal_history'],
                                      **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
            selected = [row for row in recipe_scores if row['method_id'] == method]
            macro.append({'method_id': method, 'recipes': 3,
                          **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
        first = 'curvature_frozen_theta__fixed_first6'
        proposed = 'curvature_frozen_theta__expected_crps6'
        differences = []
        for recipe in recipes:
            get = lambda method: next(r['crps'] for r in recipe_scores
                                      if r['method_id'] == method and r['recipe'] == recipe)
            differences.append({'recipe': recipe, 'proposed_crps': get(proposed),
                                'fixed_first6_crps': get(first),
                                'delta': get(proposed) - get(first)})
        primary = float(np.mean([d['delta'] for d in differences]))
        if len(rows) != 210 or len(point_terms) != 1680 or len(recipes) != 3 or len(method_ids) != 14:
            raise ValueError('Registered score cardinalities failed')
        write_new(path / 'SCORES.json', {
            'schema': 'mlst-resin-scores-v1', 'actual_utc': utc(),
            'registration_sha256': sha(registration), 'protocol_id': protocol['protocol_id'],
            'prepare_sha256': sha(Path(output) / 'prepare' / 'PREPARED.json'),
            'predictions_sha256': sha(Path(output) / 'predict' / 'PREDICTIONS.json'),
            'targets_sha256': sha(Path(output) / 'reveal' / 'TARGETS.json'),
            'primary': {'contrast': proposed + ' minus ' + first,
                        'recipe_macro_CRPS_delta': primary,
                        'direction': 'negative_improves',
                        'development_result': 'improved' if primary < 0 else 'not_improved',
                        'per_recipe': differences},
            'macro_by_method': macro, 'recipe_scores': recipe_scores,
            'curve_scores': rows, 'point_terms': point_terms,
            'scoring_scale': 'natural log of eta in Pa.s',
            'aggregation': 'eight fixed targets/run, five runs/recipe, equal three recipe macro; log_rmse is mean of run RMSE',
            'independent_material_recipes': 3, 'new_fits': 0,
            'cost': {'per_method_anchor_cells': 90,
                     'unique_anchor_union_cells': read_json(Path(output) / 'predict' / 'ACCESS.json')['unique_eta_cells_accessed'],
                     'unique_scoring_target_cells': 120,
                     'actual_new_measurements': 0, 'actual_experiment_savings_measured': False},
            'future_final_runs_protected': 35,
            'limitations': ['polymer resin transfer stress, not hydrogel confirmation',
                            'three development recipes, no final confirmation',
                            'recipe and heating/cooling order confounded',
                            'model-expected query utility is not a calibrated guarantee',
                            'covariance-only schedules are outcome-independent',
                            'source PP25 versus paper 50mm geometry unresolved']})
        _complete(path, 'SCORES.json', start, forecast_scores=210, CRPS_point_terms=1680,
                  independent_recipes=3, primary_delta=primary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--checkpoint-root', type=Path,
                        help='Optional byte-identical relocated checkpoint; only its 16 pinned files are read')
    parser.add_argument('--pins-path', type=Path,
                        help='Optional byte-identical FROZEN_DEPENDENCIES manifest')
    parser.add_argument('phase', choices=['prepare', 'predict', 'reveal', 'score'])
    args = parser.parse_args(argv)
    reg, protocol, inputs = _authorize(args.registration, args.phase)
    if args.phase == 'prepare':
        prepare(args.output, args.registration, protocol, inputs,
                checkpoint_root=args.checkpoint_root, pins_path=args.pins_path)
    elif args.phase == 'predict':
        predict(args.output, args.registration, protocol, inputs,
                checkpoint_root=args.checkpoint_root)
    elif args.phase == 'reveal':
        if args.audit is None or args.release is None:
            raise PermissionError('Explicit independent audit and root release paths required')
        reveal(args.output, args.registration, protocol, inputs, args.audit, args.release)
    else:
        score(args.output, args.registration, protocol)
    print(json.dumps({'phase': args.phase, 'status': 'COMPLETE',
                      'output': str(args.output.resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
