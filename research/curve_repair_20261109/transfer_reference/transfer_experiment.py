"""SOURCE ONLY: post-reveal development diagnostic of a physical-trend reference.

Consumes only previously saved 210 own-six anchor payloads and frozen K.
No raw ZIP, material reader, future/final labels or covariance fitting. New
420 forecasts are frozen before this code reads the SAME consumed 120 targets;
this order is reproducible development, NOT new independent confirmation.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from universal_student import fit_predict


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL = HERE / 'TRANSFER_REFERENCE_PROTOCOL.json'
STUDENT_SOURCE = ROOT / 'baselines' / 'pure_student.py'
STUDENT_SHA = '567212afb8a48cebcdbf2b69f051f3a1ecfbc7b3d10e5f3bf45181aebb59c892'
PANEL = list(range(17, 25))
COVARIANCE_IDS = ('gp_theta0', 'curvature_frozen_theta')


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


def fixed_student_module():
    source = STUDENT_SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != STUDENT_SHA:
        raise ValueError('Previously qualified Student source changed')
    name = '_mlst_transfer_fixed_student_' + STUDENT_SHA
    spec = importlib.util.spec_from_file_location(name, STUDENT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(STUDENT_SOURCE), 'exec'), module.__dict__)
    return module


def authorize(registration, phase):
    reg = read(registration)
    if reg.get('schema') != 'mlst-resin-universal-student-registration-v1' or phase not in reg.get('authorized_phases', []):
        raise PermissionError('Separate actual-UTC source registration required')
    stamp = dt.datetime.fromisoformat(reg['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture timezone-aware registration required')
    protocol = read(PROTOCOL)
    if reg['protocol_id'] != protocol['protocol_id'] or reg['protocol_sha256'] != sha(PROTOCOL):
        raise PermissionError('Frozen transfer protocol changed')
    pins = {str(Path(p).resolve()).casefold(): value for p, value in reg['source_pins'].items()}
    for path in list(HERE.glob('*.py')) + [STUDENT_SOURCE]:
        if pins.get(str(path.resolve()).casefold()) != sha(path):
            raise PermissionError('Missing/changed source pin: ' + str(path))
    if sha(STUDENT_SOURCE) != STUDENT_SHA:
        raise ValueError('Existing Student mathematics source is not the qualified version')
    for name in ('main_predictions', 'main_prepare'):
        if sha(reg[name]) != reg[name + '_sha256']:
            raise ValueError('Frozen original main input/forecast bytes changed')
    return reg, protocol


@contextmanager
def phase_path(output, phase, registration):
    path = Path(output) / phase
    path.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    write(path / 'STARTED.json', {'actual_utc': utc(), 'phase': phase,
          'registration_sha256': sha(registration), 'command': sys.argv,
          'study_role': 'POST_REVEAL_DEV_DIAGNOSTIC', 'new_material_eta_access': 0})
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
    main, prepared = read(reg['main_predictions']), read(reg['main_prepare'])
    if main['prepare_sha256'] != reg['main_prepare_sha256'] or len(main['forecasts']) != 210:
        raise ValueError('Original 210 own-six payloads bound to prepared K required')
    curves = {row['member']: row for row in prepared['curves']}
    dev = {name for name, row in prepared['roles'].items() if row['role'] == 'dev'}
    if len(curves) != 15 or len(dev) != 3 or {row['recipe'] for row in curves.values()} != dev:
        raise ValueError('Exactly the same three development recipes and15runs required')
    with phase_path(output, 'predict', registration) as (path, start):
        forecasts = []
        for main_row in main['forecasts']:
            curve = curves[main_row['member']]
            anchors, targets = main_row['anchor_indices'], main_row['target_indices']
            if (len(anchors) != 6 or len(set(anchors)) != 6 or not {0, 1}.issubset(anchors) or
                    not set(anchors).issubset(range(17)) or targets != PANEL or
                    curve['recipe'] != main_row['recipe']):
                raise ValueError('Frozen six-own-anchor fixed-panel contract required')
            for covariance_id in COVARIANCE_IDS:
                law = fit_predict(curve['log_gamma'], anchors, main_row['anchor_values'], targets,
                                  curve['laws'][covariance_id]['covariance'], covariance_id)
                row = {key: main_row[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                                                      'anchor_indices', 'anchor_values', 'target_indices')}
                row.update(method_id=covariance_id + '__universal_student__' + main_row['method_id'],
                           mask_method_id=main_row['method_id'],
                           source_reference_id=main_row['reference_id'],
                           source_policy=main_row['policy'], **law.as_dict())
                forecasts.append(row)
        if len(forecasts) != 420:
            raise ValueError('All420 fixed-covariance/mask Student forecasts required')
        write(path / 'PREDICTIONS.json', {'schema': 'mlst-resin-universal-student-predictions-v1',
            'actual_utc': utc(), 'registration_sha256': sha(registration),
            'protocol_sha256': sha(PROTOCOL), 'protocol_id': protocol['protocol_id'],
            'main_predictions_sha256': reg['main_predictions_sha256'],
            'main_prepare_sha256': reg['main_prepare_sha256'],
            'study_role': 'POST_REVEAL_DEV_DIAGNOSTIC', 'forecasts': forecasts,
            'GLS_fits': 420, 'new_raw_eta_access': 0, 'new_target_access': 0,
            'covariance_hyperparameter_updates': 0, 'main_model_retrained': False,
            'repeated_masks_are_new_measurements': False, 'future_final_runs_protected': 35})
        complete(path, 'PREDICTIONS.json', start, forecasts=420, GLS_fits=420,
                 new_material_eta_access=0, covariance_hyperparameter_updates=0)


def score(registration, output, reg, protocol, audit_path, release_path, targets_path):
    predicted = Path(output) / 'predict' / 'PREDICTIONS.json'
    finished = read(Path(output) / 'predict' / 'COMPLETE.json')
    if finished['status'] != 'COMPLETE' or finished['payload_sha256'] != sha(predicted):
        raise PermissionError('Complete immutable transfer forecasts required')
    saved, audit, release = read(predicted), read(audit_path), read(release_path)
    if saved['registration_sha256'] != sha(registration):
        raise ValueError('Forecasts belong to another actual registration')
    if (audit.get('status') != 'PASS' or audit.get('audited_transfer_predictions_sha256') != sha(predicted) or
            audit.get('audited_main_predictions_sha256') != reg['main_predictions_sha256']):
        raise PermissionError('Independent exact-hash universal-law audit required')
    if (release.get('schema') != 'mlst-resin-universal-student-score-release-v1' or
            release.get('authorization') != 'SCORE_DEV_DIAGNOSTIC_ON_SAME_CONSUMED_120_TARGETS' or
            release.get('transfer_predictions_sha256') != sha(predicted) or
            release.get('main_predictions_sha256') != reg['main_predictions_sha256'] or
            release.get('audit_sha256') != sha(audit_path) or
            release.get('targets_sha256') != sha(targets_path)):
        raise PermissionError('Root exact-hash post-reveal development release required')
    stamp = dt.datetime.fromisoformat(release['actual_utc'].replace('Z', '+00:00'))
    if stamp.utcoffset() is None or stamp > dt.datetime.now(dt.timezone.utc):
        raise PermissionError('Actual nonfuture score release UTC required')
    targets = read(targets_path)
    if (targets['predictions_sha256'] != reg['main_predictions_sha256'] or
            targets['prepare_sha256'] != reg['main_prepare_sha256'] or len(targets['curves']) != 15):
        raise ValueError('Only the same previously consumed120 development targets are permitted')
    scorer = fixed_student_module()
    lookup = {row['member']: row for row in targets['curves']}
    with phase_path(output, 'score', registration) as (path, start):
        rows, point_terms = [], []
        for forecast in saved['forecasts']:
            truth = lookup[forecast['member']]
            if forecast['target_indices'] != PANEL or truth['target_indices'] != PANEL:
                raise ValueError('Same fixed target panel required')
            metrics, points = scorer.score_saved_law(forecast['location'], forecast['scale_matrix'],
                                                    forecast['df'], truth['values'])
            identity = {key: forecast[key] for key in ('member', 'recipe', 'run_id', 'thermal_history',
                        'method_id', 'mask_method_id', 'covariance_id', 'source_reference_id', 'source_policy')}
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
                    raise ValueError('Five same-recipe original runs required')
                recipe_scores.append({'method_id': method, 'recipe': recipe, 'runs': 5,
                    **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
            selected = [row for row in recipe_scores if row['method_id'] == method]
            macro.append({'method_id': method, 'recipes': 3,
                **{key: float(np.mean([r[key] for r in selected])) for key in keys}})
        if len(rows) != 420 or len(point_terms) != 3360 or len(method_ids) != 28 or len(recipes) != 3:
            raise ValueError('Registered universal diagnostic score cardinalities failed')
        write(path / 'SCORES.json', {'schema': 'mlst-resin-universal-student-scores-v1',
            'actual_utc': utc(), 'registration_sha256': sha(registration),
            'protocol_id': protocol['protocol_id'], 'study_role': 'POST_REVEAL_DEV_DIAGNOSTIC',
            'predictions_sha256': sha(predicted), 'targets_sha256': sha(targets_path),
            'independent_law_audit_sha256': sha(audit_path), 'root_score_release_sha256': sha(release_path),
            'curve_scores': rows, 'point_terms': point_terms,
            'recipe_scores': recipe_scores, 'macro_by_method': macro,
            'aggregation': '8 common targets/run ->5runs/recipe ->equal3recipe; mean run-RMSE',
            'main_primary_unchanged': True, 'new_raw_eta_access': 0,
            'new_target_access': 0, 'actual_measurement_savings': False,
            'limitations': ['Same consumed development targets, not new confirmation',
                            'Classical universal kriging/Student formula, not claimed new theory',
                            'Fixed hydrogel covariance may remain inappropriate for resin',
                            '28 method/mask labels include repeated masks, not28 independent studies',
                            'Trend and scale GLS420 fits; covariance updates0;35future/final runs untouched']})
        complete(path, 'SCORES.json', start, forecast_scores=420, CRPS_point_terms=3360,
                 study_role='POST_REVEAL_DEV_DIAGNOSTIC')


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
            raise PermissionError('Explicit saved audit/root release/same targets paths required')
        score(args.registration, args.output, reg, protocol, args.audit, args.release, args.targets)
    print(json.dumps({'phase': args.phase, 'status': 'COMPLETE', 'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
