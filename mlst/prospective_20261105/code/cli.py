"""Append-only local prospective helper; never calls a model or reads old data.

Real usage requires an author-completed registered plan and genuinely reserved
labels. File order is an audit aid, not proof that people never saw the labels.
"""
import argparse
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

from calibration import calibrate_unit_scores, make_band, threshold_from_json, unit_max_scores
from contract import SCHEMA, check_design, check_predictions, check_truth, parse_json, read, sha, source_bindings, utc, write
from evaluation import evaluate_complete_cohort

PHASE_ARTIFACTS = ('PROTOCOL.json', 'CALIBRATION_PREDICTIONS_FROZEN.json',
                   'CALIBRATION_FROZEN.json', 'CONFIRMATION_PREDICTIONS_FROZEN.json',
                   'CONFIRMATION_EVALUATION.json')
REVEAL_PREFIXES = {'calibrate': 'CALIBRATION', 'evaluate': 'CONFIRMATION'}


def _receipt_name(artifact):
    return artifact.removesuffix('.json') + '.RECEIPT.json'


def _verify_completed_history(run, protocol_hash):
    """Detect one-file changes/incomplete writes; not an adversarial trust anchor."""
    prior = {}
    reveal_names = {prefix + suffix for prefix in REVEAL_PREFIXES.values()
                    for suffix in ('_REVEAL_STARTED.json', '_REVEAL_PARSED.json')}
    for name in PHASE_ARTIFACTS:
        artifact, receipt = run / name, run / _receipt_name(name)
        if not artifact.exists() and not receipt.exists():
            continue
        if not artifact.is_file() or not receipt.is_file():
            raise ValueError('Incomplete completed-phase artifact/receipt pair: ' + name)
        record = read(receipt)
        if (record['schema'] != SCHEMA or record['artifact_file'] != name
                or record['protocol_sha256'] != protocol_hash
                or record['artifact_sha256'] != sha(artifact)
                or record['prior_receipts_sha256'] != prior):
            raise ValueError('Completed-phase byte identity/history changed: ' + name)
        for filename, digest in record['reveal_receipts_sha256'].items():
            if filename not in reveal_names or sha(run / filename) != digest:
                raise ValueError('Reveal receipt changed: ' + filename)
        prior[receipt.name] = sha(receipt)
    return prior


def _write_completed_phase(run, name, obj, protocol_hash=None, reveal_bindings=None):
    """Write output once, then its completion receipt; interruption stays visible."""
    artifact, receipt = run / name, run / _receipt_name(name)
    if artifact.exists() or receipt.exists():
        raise FileExistsError('Phase artifact/receipt already exists: ' + name)
    prior = {p.name: sha(p) for candidate in PHASE_ARTIFACTS
             if (p := run / _receipt_name(candidate)).is_file()}
    write(artifact, obj)
    digest = sha(artifact)
    write(receipt, {'schema': SCHEMA, 'completed_actual_utc': utc(),
                    'artifact_file': name, 'artifact_sha256': digest,
                    'protocol_sha256': digest if protocol_hash is None else protocol_hash,
                    'prior_receipts_sha256': prior,
                    'reveal_receipts_sha256': reveal_bindings or {},
                    'integrity_scope': 'Detect inconsistent local files; no certification against coordinated rewriting.'})
    return {'artifact_sha256': digest, 'completion_receipt_sha256': sha(receipt)}


@contextmanager
def locked(run):
    """One local writer; abrupt process termination leaves a visible lock."""
    lock = run / 'ACTIVE_PHASE_LOCK.json'
    write(lock, {'actual_utc': utc(), 'instruction': 'Do not delete if a phase process is still active.'})
    try:
        yield
    finally:
        lock.unlink()


def load_run(run):
    if (run / 'REVEAL_ATTEMPT_FAILED_STOP.json').exists():
        raise ValueError('A truth-open attempt failed; retain its receipt and do not silently restart as fresh')
    path = run / 'PROTOCOL.json'
    protocol = read(path)
    protocol_hash = sha(path)
    _verify_completed_history(run, protocol_hash)
    if protocol['implementation_sha256'] != source_bindings():
        raise ValueError('Implementation changed after registration; do not silently continue')
    for command, prefix in REVEAL_PREFIXES.items():
        completion = 'CALIBRATION_FROZEN.json' if command == 'calibrate' else 'CONFIRMATION_EVALUATION.json'
        if (run / (prefix + '_REVEAL_STARTED.json')).exists() and not (run / completion).exists():
            raise ValueError('An earlier reveal attempt has no completed phase; stop for review: ' + prefix)
    check_design(protocol['config'], protocol['cohort'])
    return protocol, protocol_hash


def payload_input(path, protocol_hash):
    payload = read(path)
    if not isinstance(payload, dict) or payload.get('protocol_sha256') != protocol_hash:
        raise ValueError('Input is not bound to the registered protocol bytes')
    return payload


def _reveal_truth(path, run, command, protocol_hash, attempt):
    """Mark intent immediately before opening truth; bind the bytes parsed once."""
    prefix = REVEAL_PREFIXES[command]
    start, parsed = run / (prefix + '_REVEAL_STARTED.json'), run / (prefix + '_REVEAL_PARSED.json')
    write(start, {'schema': SCHEMA, 'actual_utc': utc(), 'phase': command,
                  'protocol_sha256': protocol_hash,
                  'status': 'ABOUT_TO_ATTEMPT_TRUTH_FILE_OPEN',
                  'actual_measurement_values_read_certified': False})
    attempt['truth_open_attempt_started'] = True
    blob = Path(path).read_bytes()
    attempt['truth_bytes_loaded'] = True
    payload = parse_json(blob.decode('utf-8-sig'))
    attempt['truth_payload_parsed'] = True
    truth_hash = hashlib.sha256(blob).hexdigest()
    write(parsed, {'schema': SCHEMA, 'actual_utc': utc(), 'phase': command,
                   'protocol_sha256': protocol_hash, 'truth_input_sha256': truth_hash,
                   'status': 'JSON_BYTES_PARSED_NOT_SOURCE_OR_MEASUREMENT_CERTIFICATION'})
    if not isinstance(payload, dict) or payload.get('protocol_sha256') != protocol_hash:
        raise ValueError('Revealed truth belongs to another protocol')
    return payload, truth_hash, {start.name: sha(start), parsed.name: sha(parsed)}


def register(args):
    config, cohort = read(args.config), read(args.cohort)
    checked = check_design(config, cohort)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=False)
    identity = _write_completed_phase(run, 'PROTOCOL.json', {
        'schema': SCHEMA, 'registered_actual_utc': utc(), 'config': config, 'cohort': cohort,
        'original_design_input_sha256': sha(args.config), 'original_cohort_input_sha256': sha(args.cohort),
        'implementation_sha256': source_bindings(), 'declared_checks': checked,
        'human_blindness_or_independence_certified': False})
    return {'status': 'DECLARED_PROTOCOL_FROZEN_NOT_SOURCE_CERTIFICATION', **identity}


def freeze_predictions(args, run, protocol, protocol_hash, role):
    if role == 'confirmation' and not (run / 'CALIBRATION_FROZEN.json').exists():
        raise ValueError('Freeze calibration thresholds before confirmation predictions/truth')
    name = role.upper() + '_PREDICTIONS_FROZEN.json'
    if (run / name).exists():
        raise FileExistsError('Predictions are already frozen')
    _verify_completed_history(run, protocol_hash)
    payload = payload_input(args.predictions, protocol_hash)
    check_predictions(payload, protocol, role, protocol_hash)
    wrapper = {'schema': SCHEMA, 'actual_utc': utc(), 'role': role,
               'protocol_sha256': protocol_hash, 'original_prediction_input_sha256': sha(args.predictions),
               'payload': payload, 'truth_file_open_attempted_in_this_phase': False,
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
    return {'status': role.upper() + '_PREDICTIONS_FROZEN',
            **_write_completed_phase(run, name, wrapper, protocol_hash)}


def calibrate(args, run, protocol, protocol_hash, attempt):
    frozen_path, name = run / 'CALIBRATION_PREDICTIONS_FROZEN.json', 'CALIBRATION_FROZEN.json'
    if (run / name).exists() or (run / 'CONFIRMATION_PREDICTIONS_FROZEN.json').exists():
        raise FileExistsError('Thresholds/confirmation are already frozen; no recalibration')
    _verify_completed_history(run, protocol_hash)
    predictions = read(frozen_path)['payload']
    check_predictions(predictions, protocol, 'calibration', protocol_hash)
    revealed, truth_hash, reveals = _reveal_truth(args.truth, run, 'calibrate', protocol_hash, attempt)
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
        scores, reasons = unit_max_scores(rows, return_reasons=True)
        thresholds[mid] = calibrate_unit_scores(scores, protocol['config']['alpha'], reasons).to_json()
        all_scores[mid] = {uid: {'score': None if math.isinf(value) else value,
                                'unbounded': math.isinf(value), 'infinity_reasons': reasons[uid]}
                           for uid, value in scores.items()}
    output = {'schema': SCHEMA, 'actual_utc': utc(), 'protocol_sha256': protocol_hash,
              'calibration_predictions_sha256': sha(frozen_path),
              'original_calibration_truth_sha256': truth_hash, 'calibration_truth': revealed,
              'thresholds': thresholds, 'scores_by_declared_unit': all_scores,
              'all_methods_same_units_grid_alpha': True, 'model_selection_performed': False,
              'real_exchangeability_automatically_certified': False}
    return {'status': 'THRESHOLDS_FROZEN_NOT_A_CALIBRATED_DENSITY',
            **_write_completed_phase(run, name, output, protocol_hash, reveals)}


def evaluate(args, run, protocol, protocol_hash, attempt):
    name = 'CONFIRMATION_EVALUATION.json'
    if (run / name).exists():
        raise FileExistsError('Confirmation has already been consumed; do not rescore/reselect')
    _verify_completed_history(run, protocol_hash)
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
    thresholds = {mid: threshold_from_json(value) for mid, value in calibrated['thresholds'].items()}
    revealed, truth_hash, reveals = _reveal_truth(args.truth, run, 'evaluate', protocol_hash, attempt)
    truth = check_truth(revealed, predictions, protocol, 'confirmation', protocol_hash)
    result = evaluate_complete_cohort(predictions, truth, protocol, thresholds)
    result.update({'schema': SCHEMA, 'actual_utc': utc(), 'protocol_sha256': protocol_hash,
                   'calibration_sha256': sha(cal_path), 'confirmation_predictions_sha256': sha(pred_path),
                   'original_confirmation_truth_sha256': truth_hash, 'confirmation_truth': revealed,
                   'new_fits': 0, 'finite_grid_only': True})
    return {'status': 'COMPLETE_RESERVED_COHORT_SCORED_REQUIRES_SCIENTIFIC_REVIEW',
            **_write_completed_phase(run, name, result, protocol_hash, reveals)}


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
    attempt = {'truth_open_attempt_started': False, 'truth_bytes_loaded': False, 'truth_payload_parsed': False}
    try:
        if args.command == 'check-plan':
            output = check_design(read(args.config), read(args.cohort))
        elif args.command == 'register':
            output = register(args)
        else:
            run = Path(args.run)
            if not run.is_dir():
                raise FileNotFoundError('Registered run directory does not exist')
            with locked(run):
                protocol, protocol_hash = load_run(run)
                if args.command.startswith('freeze-'):
                    output = freeze_predictions(args, run, protocol, protocol_hash, args.command[7:])
                elif args.command == 'calibrate':
                    output = calibrate(args, run, protocol, protocol_hash, attempt)
                else:
                    output = evaluate(args, run, protocol, protocol_hash, attempt)
        print(output)
        return 0
    except (ValueError, TypeError, KeyError, FileNotFoundError, FileExistsError, OverflowError, np.linalg.LinAlgError) as error:
        run = Path(args.run) if getattr(args, 'run', None) else None
        if run is not None and run.is_dir():
            name = 'FAILURE_' + utc().replace(':', '').replace('.', '') + '.json'
            failure = {'actual_utc': utc(), 'phase': args.command, 'error_type': type(error).__name__,
                       'message': str(error), **attempt, 'failed_cohort_not_scored_as_successful_subset': True,
                       'instruction': 'Inspect actual reveal status. A reveal-start marker is intent/open attempt, not proof of measured values read.'}
            write(run / name, failure)
            if attempt['truth_open_attempt_started'] and not (run / 'REVEAL_ATTEMPT_FAILED_STOP.json').exists():
                write(run / 'REVEAL_ATTEMPT_FAILED_STOP.json', {
                    'actual_utc': utc(), 'phase': args.command, 'failure_receipt': name, **attempt,
                    'status': 'TRUTH_OPEN_ATTEMPT_FAILED_STOP_FOR_REVIEW_NO_SILENT_RETRY',
                    'measurement_source_or_blindness_certified': False})
        print('STOP: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
