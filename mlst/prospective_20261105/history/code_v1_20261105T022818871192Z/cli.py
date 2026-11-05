"""Append-only local prospective helper; never calls a model or reads old data.

Real usage requires an author-completed registered plan and genuinely reserved
labels. File order is an audit aid, not proof that people never saw the labels.
"""
import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import sys

import numpy as np

from calibration import calibrate_unit_scores, make_band, threshold_from_json, unit_max_scores
from contract import SCHEMA, check_design, check_predictions, check_truth, read, sha, source_bindings, utc, write
from evaluation import evaluate_complete_cohort


@contextmanager
def locked(run):
    """One local writer; interrupted lock is kept for human inspection."""
    lock = run / 'ACTIVE_PHASE_LOCK.json'
    write(lock, {'actual_utc': utc(), 'instruction': 'Do not delete if a phase process is still active.'})
    try:
        yield
    finally:
        lock.unlink()


def load_run(run):
    if (run / 'LABEL_CONSUMPTION_FAILED.json').exists():
        raise ValueError('A label-consuming phase failed; this cohort cannot be silently restarted as fresh')
    path = run / 'PROTOCOL.json'
    protocol = read(path)
    if protocol['implementation_sha256'] != source_bindings():
        raise ValueError('Implementation changed after registration; do not silently continue')
    check_design(protocol['config'], protocol['cohort'])
    return protocol, sha(path)


def payload_input(path, protocol_hash):
    payload = read(path)
    if payload.get('protocol_sha256') != protocol_hash:
        raise ValueError('Input is not bound to the registered protocol bytes')
    return payload


def register(args):
    config, cohort = read(args.config), read(args.cohort)
    checked = check_design(config, cohort)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'PROTOCOL.json', {'schema': SCHEMA, 'registered_actual_utc': utc(),
                                'config': config, 'cohort': cohort,
                                'original_design_input_sha256': sha(args.config),
                                'original_cohort_input_sha256': sha(args.cohort),
                                'implementation_sha256': source_bindings(),
                                'declared_checks': checked,
                                'human_blindness_or_independence_certified': False})
    return {'status': 'DECLARED_PROTOCOL_FROZEN_NOT_SOURCE_CERTIFICATION', 'protocol_sha256': sha(run / 'PROTOCOL.json')}


def freeze_predictions(args, run, protocol, protocol_hash, role):
    if role == 'confirmation' and not (run / 'CALIBRATION_FROZEN.json').exists():
        raise ValueError('Freeze calibration thresholds before confirmation predictions/truth')
    path = run / (role.upper() + '_PREDICTIONS_FROZEN.json')
    if path.exists():
        raise FileExistsError('Predictions are already frozen')
    payload = payload_input(args.predictions, protocol_hash)
    check_predictions(payload, protocol, role, protocol_hash)
    wrapper = {'schema': SCHEMA, 'actual_utc': utc(), 'role': role,
               'protocol_sha256': protocol_hash, 'original_prediction_input_sha256': sha(args.predictions),
               'payload': payload, 'labels_read_in_this_phase': False,
               'actual_human_blindness_certified': False}
    if role == 'confirmation':
        calibration_path = run / 'CALIBRATION_FROZEN.json'
        cal = read(calibration_path)
        if cal['protocol_sha256'] != protocol_hash:
            raise ValueError('Calibration protocol binding changed')
        wrapper['calibration_sha256'] = sha(calibration_path)
        wrapper['prediction_sets'] = {}
        for record in payload['records']:
            wrapper['prediction_sets'][record['record_id']] = {}
            for mid, law in record['methods'].items():
                threshold = threshold_from_json(cal['thresholds'][mid])
                sd = np.sqrt(np.diag(np.asarray(law['covariance_log_eta'], dtype=float))).tolist()
                wrapper['prediction_sets'][record['record_id']][mid] = make_band(law['mean_log_eta'], sd, threshold).to_json()
    write(path, wrapper)
    return {'status': role.upper() + '_PREDICTIONS_FROZEN', 'sha256': sha(path)}


def calibrate(args, run, protocol, protocol_hash):
    frozen_path, output_path = run / 'CALIBRATION_PREDICTIONS_FROZEN.json', run / 'CALIBRATION_FROZEN.json'
    if output_path.exists() or (run / 'CONFIRMATION_PREDICTIONS_FROZEN.json').exists():
        raise FileExistsError('Thresholds/confirmation are already frozen; no recalibration')
    predictions = read(frozen_path)['payload']
    check_predictions(predictions, protocol, 'calibration', protocol_hash)
    revealed = payload_input(args.truth, protocol_hash)
    truth = check_truth(revealed, predictions, protocol, 'calibration', protocol_hash)
    enrolled = {r['record_id']: r for r in protocol['cohort']['records']}
    thresholds, all_scores = {}, {}
    for method in protocol['config']['methods']:
        mid = method['id']
        rows = []
        for record in predictions['records']:
            law = record['methods'][mid]
            rows.append({'unit_id': enrolled[record['record_id']]['unit_id'], 'mean': law['mean_log_eta'],
                         'sd': np.sqrt(np.diag(np.asarray(law['covariance_log_eta'], dtype=float))).tolist(),
                         'truth': truth[record['record_id']]})
        scores = unit_max_scores(rows)
        thresholds[mid] = calibrate_unit_scores(scores, protocol['config']['alpha']).to_json()
        all_scores[mid] = {uid: {'score': None if math.isinf(value) else value, 'unbounded': math.isinf(value)} for uid, value in scores.items()}
    write(output_path, {'schema': SCHEMA, 'actual_utc': utc(), 'protocol_sha256': protocol_hash,
                        'calibration_predictions_sha256': sha(frozen_path),
                        'original_calibration_truth_sha256': sha(args.truth), 'calibration_truth': revealed,
                        'thresholds': thresholds, 'scores_by_independent_unit': all_scores,
                        'all_methods_same_units_grid_alpha': True, 'model_selection_performed': False,
                        'real_exchangeability_automatically_certified': False})
    return {'status': 'THRESHOLDS_FROZEN_NOT_A_CALIBRATED_DENSITY', 'sha256': sha(output_path)}


def evaluate(args, run, protocol, protocol_hash):
    result_path = run / 'CONFIRMATION_EVALUATION.json'
    if result_path.exists():
        raise FileExistsError('Confirmation has already been consumed; do not rescore/reselect')
    cal_path, pred_path = run / 'CALIBRATION_FROZEN.json', run / 'CONFIRMATION_PREDICTIONS_FROZEN.json'
    calibrated, frozen = read(cal_path), read(pred_path)
    if calibrated['protocol_sha256'] != protocol_hash or frozen['protocol_sha256'] != protocol_hash:
        raise ValueError('Calibration/confirmation protocol binding changed')
    if frozen['calibration_sha256'] != sha(cal_path):
        raise ValueError('Calibration thresholds changed after confirmation freeze')
    cal_pred = run / 'CALIBRATION_PREDICTIONS_FROZEN.json'
    if calibrated['calibration_predictions_sha256'] != sha(cal_pred):
        raise ValueError('Calibration forecasts changed after calibration')
    predictions = frozen['payload']
    check_predictions(predictions, protocol, 'confirmation', protocol_hash)
    revealed = payload_input(args.truth, protocol_hash)
    truth = check_truth(revealed, predictions, protocol, 'confirmation', protocol_hash)
    thresholds = {mid: threshold_from_json(value) for mid, value in calibrated['thresholds'].items()}
    result = evaluate_complete_cohort(predictions, truth, protocol, thresholds)
    result.update({'schema': SCHEMA, 'actual_utc': utc(), 'protocol_sha256': protocol_hash,
                   'calibration_sha256': sha(cal_path), 'confirmation_predictions_sha256': sha(pred_path),
                   'original_confirmation_truth_sha256': sha(args.truth), 'confirmation_truth': revealed,
                   'new_fits': 0, 'finite_grid_only': True})
    write(result_path, result)
    return {'status': 'COMPLETE_RESERVED_COHORT_SCORED_REQUIRES_SCIENTIFIC_REVIEW', 'sha256': sha(result_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check-plan', 'register'):
        child = sub.add_parser(name)
        child.add_argument('--config', required=True)
        child.add_argument('--cohort', required=True)
        if name == 'register':
            child.add_argument('--run', required=True)
    for name in ('freeze-calibration', 'calibrate', 'freeze-confirmation', 'evaluate'):
        child = sub.add_parser(name)
        child.add_argument('--run', required=True)
        child.add_argument('--predictions' if name.startswith('freeze-') else '--truth', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check-plan':
            output = check_design(read(args.config), read(args.cohort))
        elif args.command == 'register':
            output = register(args)
        else:
            run = Path(args.run)
            with locked(run):
                protocol, protocol_hash = load_run(run)
                if args.command.startswith('freeze-'):
                    output = freeze_predictions(args, run, protocol, protocol_hash, args.command[7:])
                elif args.command == 'calibrate':
                    output = calibrate(args, run, protocol, protocol_hash)
                else:
                    output = evaluate(args, run, protocol, protocol_hash)
        print(output)
        return 0
    except (ValueError, TypeError, KeyError, FileNotFoundError, FileExistsError, OverflowError, np.linalg.LinAlgError) as error:
        run = Path(args.run) if getattr(args, 'run', None) else None
        if run is not None and run.is_dir():
            name = 'FAILURE_' + utc().replace(':', '').replace('.', '') + '.json'
            write(run / name, {'actual_utc': utc(), 'phase': args.command,
                              'error_type': type(error).__name__, 'message': str(error),
                              'failed_cohort_not_scored_as_successful_subset': True,
                              'instruction': 'Retain this receipt. A label-consuming failure is not a fresh independent cohort.'})
            if args.command in ('calibrate', 'evaluate') and not (run / 'LABEL_CONSUMPTION_FAILED.json').exists():
                write(run / 'LABEL_CONSUMPTION_FAILED.json', {'actual_utc': utc(), 'phase': args.command,
                                                             'failure_receipt': name,
                                                             'status': 'STOP_FOR_REVIEW_NO_SILENT_RETRY'})
        print('STOP: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
