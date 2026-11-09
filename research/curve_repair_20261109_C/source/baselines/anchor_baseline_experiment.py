"""SOURCE ONLY until actual root registration: physical anchor-only controls.

predict reads only saved frozen main six-anchor payloads and input coordinates,
not ZIP/readers, targets or new material measurements. It freezes 420 joint
Student-t forecasts before root's common target release. score consumes those
saved laws plus the same already released 120 labels after an exact-SHA release
and independent preY baseline audit. No main model/source is modified.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from pure_student import KINDS, fit_predict, score_saved_law


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'ANCHOR_BASELINE_PROTOCOL.json'
PANEL = list(range(17, 25))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError('Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON token')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def authorize(registration, phase):
    reg = read(registration)
    if reg.get('schema') != 'mlst-resin-anchor-baseline-registration-v1' or phase not in reg.get('authorized_phases', []):
        raise PermissionError('Separate root source/hash registration required')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture UTC registration required')
    if reg['protocol_sha256'] != sha(PROTOCOL):
        raise PermissionError('Baseline protocol changed')
    pins = {str(Path(p).resolve()).casefold(): v for p, v in reg['source_pins'].items()}
    for source in HERE.glob('*.py'):
        if pins.get(str(source.resolve()).casefold()) != sha(source):
            raise PermissionError('Baseline source bytes changed: ' + source.name)
    protocol = read(PROTOCOL)
    if reg['protocol_id'] != protocol['protocol_id']:
        raise PermissionError('Baseline protocol ID mismatch')
    for name in ('main_predictions', 'main_prepare'):
        if sha(reg[name]) != reg[name + '_sha256']:
            raise ValueError('Registered main forecast/input bytes changed')
    return reg, protocol


@contextmanager
def phase_path(output, phase, registration):
    path = Path(output) / phase
    path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write(path / 'STARTED.json', {'actual_utc': utc(), 'phase': phase,
          'registration_sha256': sha(registration), 'command': sys.argv,
          'new_material_eta_access': 0})
    try:
        yield path, start
    except BaseException as error:
        write(path / 'FAILED.json', {'actual_utc': utc(), 'phase': phase,
              'exception': type(error).__name__, 'message': str(error),
              'seconds': time.perf_counter() - start,
              'decision': 'STOP_KEEP_FAILURE_NO_FLOOR_NO_REPLACEMENT_NO_FALLBACK'})
        raise


def complete(path, filename, start, **facts):
    write(path / 'COMPLETE.json', {'actual_utc': utc(), 'status': 'COMPLETE',
          'payload': filename, 'payload_sha256': sha(path / filename),
          'seconds': time.perf_counter() - start, **facts})


def predict(registration, output, reg, protocol):
    main, inputs = read(reg['main_predictions']), read(reg['main_prepare'])
    if main['prepare_sha256'] != reg['main_prepare_sha256'] or len(main['forecasts']) != 210:
        raise ValueError('Registered 210 Gaussian forecasts and preparation identity required')
    curves = {curve['member']: curve for curve in inputs['curves']}
    if len(curves) != 15 or sorted({curve['recipe'] for curve in curves.values()}) != sorted(
            recipe for recipe, value in inputs['roles'].items() if value['role'] == 'dev'):
        raise ValueError('Exactly the original three development recipes required')
    with phase_path(output, 'predict', registration) as (path, start):
        forecasts = []
        for main_row in main['forecasts']:
            curve = curves[main_row['member']]
            anchors = main_row['anchor_indices']
            targets = main_row['target_indices']
            if (len(anchors) != 6 or len(set(anchors)) != 6 or not {0, 1}.issubset(anchors) or
                    not set(anchors).issubset(range(17)) or targets != PANEL or
                    curve['recipe'] != main_row['recipe']):
                raise ValueError('Original six-own-anchor fixed-panel contract required')
            coordinates = np.asarray(curve['log_gamma'])
            for kind in KINDS:
                law = fit_predict(coordinates[anchors], main_row['anchor_values'],
                                  coordinates[targets], kind)
                row = {key: main_row[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                       'anchor_indices', 'anchor_values', 'target_indices')}
                row.update(method_id=kind + '__' + main_row['method_id'],
                           baseline_kind=kind, mask_method_id=main_row['method_id'],
                           source_reference_id=main_row['reference_id'],
                           source_policy=main_row['policy'], **law.as_dict())
                forecasts.append(row)
        if len(forecasts) != 420:
            raise ValueError('All 420 matched-mask Student forecasts required')
        write(path / 'PREDICTIONS.json', {'schema': 'mlst-resin-anchor-baseline-predictions-v1',
              'actual_utc': utc(), 'registration_sha256': sha(registration),
              'protocol_sha256': sha(PROTOCOL), 'protocol_id': protocol['protocol_id'],
              'main_predictions_sha256': reg['main_predictions_sha256'],
              'main_prepare_sha256': reg['main_prepare_sha256'],
              'forecasts': forecasts, 'OLS_fits': 420,
              'new_material_eta_access': 0, 'new_target_access': 0,
              'main_model_retrained': False, 'reuse': 'already revealed own-six main anchor payloads only',
              'future_final_runs_protected': 35})
        complete(path, 'PREDICTIONS.json', start, forecasts=420, OLS_fits=420,
                 new_material_eta_access=0, new_target_access=0)


def score(registration, output, reg, protocol, audit_path, release_path, targets_path):
    predicted = Path(output) / 'predict' / 'PREDICTIONS.json'
    finished = read(Path(output) / 'predict' / 'COMPLETE.json')
    if finished['status'] != 'COMPLETE' or finished['payload_sha256'] != sha(predicted):
        raise PermissionError('Complete immutable baseline forecasts required')
    saved, audit, release = read(predicted), read(audit_path), read(release_path)
    if saved['registration_sha256'] != sha(registration):
        raise ValueError('Baseline forecasts belong to another registration')
    if (audit.get('status') != 'PASS' or audit.get('audited_baseline_predictions_sha256') != sha(predicted) or
            audit.get('audited_main_predictions_sha256') != reg['main_predictions_sha256']):
        raise PermissionError('Independent exact-SHA baseline forecast audit required')
    if (release.get('schema') != 'mlst-resin-anchor-baseline-score-release-v1' or
            release.get('authorization') != 'SCORE_FROZEN_BASELINES_ON_SAME_120_DEV_TARGETS' or
            release.get('baseline_predictions_sha256') != sha(predicted) or
            release.get('main_predictions_sha256') != reg['main_predictions_sha256'] or
            release.get('audit_sha256') != sha(audit_path) or
            release.get('targets_sha256') != sha(targets_path)):
        raise PermissionError('Root release bound to frozen baseline predictions and original targets required')
    stamp = dt.datetime.fromisoformat(release['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture baseline score release UTC required')
    targets = read(targets_path)
    if (targets['predictions_sha256'] != reg['main_predictions_sha256'] or
            targets['prepare_sha256'] != reg['main_prepare_sha256'] or len(targets['curves']) != 15):
        raise ValueError('Only the same original 120 released development targets are permitted')
    lookup = {row['member']: row for row in targets['curves']}
    with phase_path(output, 'score', registration) as (path, start):
        rows, point_terms = [], []
        for forecast in saved['forecasts']:
            truth = lookup[forecast['member']]
            if forecast['target_indices'] != PANEL or truth['target_indices'] != PANEL:
                raise ValueError('Fixed common target panel required')
            metrics, points = score_saved_law(forecast['location'], forecast['scale_matrix'],
                                             forecast['df'], truth['values'])
            identity = {key: forecast[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                        'method_id', 'baseline_kind', 'mask_method_id', 'source_reference_id', 'source_policy')}
            rows.append({**identity, **metrics})
            point_terms.extend({**identity, 'target_index': PANEL[i], **point}
                               for i, point in enumerate(points))
        method_ids = sorted({row['method_id'] for row in rows})
        recipes = sorted({row['recipe'] for row in rows})
        keys = ['crps', 'joint_nll_per_dimension', 'log_rmse', 'coverage90', 'width90']
        recipe_scores, macro = [], []
        for method in method_ids:
            for recipe in recipes:
                selected = [row for row in rows if row['method_id'] == method and row['recipe'] == recipe]
                if len(selected) != 5:
                    raise ValueError('Five matched original runs per recipe/method required')
                recipe_scores.append({'method_id': method, 'recipe': recipe, 'runs': 5,
                    **{key: float(np.mean([row[key] for row in selected])) for key in keys}})
            selected = [row for row in recipe_scores if row['method_id'] == method]
            macro.append({'method_id': method, 'recipes': 3,
                **{key: float(np.mean([row[key] for row in selected])) for key in keys}})
        if len(rows) != 420 or len(point_terms) != 3360 or len(method_ids) != 28 or len(recipes) != 3:
            raise ValueError('Registered physical-baseline score cardinalities failed')
        write(path / 'SCORES.json', {'schema': 'mlst-resin-anchor-baseline-scores-v1',
            'actual_utc': utc(), 'registration_sha256': sha(registration),
            'predictions_sha256': sha(predicted), 'targets_sha256': sha(targets_path),
            'independent_preY_audit_sha256': sha(audit_path), 'root_score_release_sha256': sha(release_path),
            'curve_scores': rows, 'point_terms': point_terms,
            'recipe_scores': recipe_scores, 'macro_by_method': macro,
            'aggregation': '8 common targets/run -> 5 runs/recipe -> equal3recipe; mean of run RMSE',
            'main_primary_unchanged': True, 'new_material_eta_access': 0,
            'new_target_access': 0, 'actual_measurement_savings': False,
            'limitations': ['Three development recipes, not final confirmation',
                            'Baseline fits use only each existing own-six anchor payload',
                            'Student scale is not predictive standard deviation',
                            'Residual likelihood is homoscedastic log-space regression',
                            'All 28 matched controls are reported; no best-method retuning']})
        complete(path, 'SCORES.json', start, forecast_scores=420, CRPS_point_terms=3360)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('phase', choices=['predict', 'score'])
    args = parser.parse_args(argv)
    reg, protocol = authorize(args.registration, args.phase)
    if args.phase == 'predict':
        predict(args.registration, args.output, reg, protocol)
    else:
        if any(value is None for value in (args.audit, args.release, args.targets)):
            raise PermissionError('Explicit saved audit/root release/targets paths required')
        score(args.registration, args.output, reg, protocol, args.audit, args.release, args.targets)
    print(json.dumps({'phase': args.phase, 'status': 'COMPLETE', 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
