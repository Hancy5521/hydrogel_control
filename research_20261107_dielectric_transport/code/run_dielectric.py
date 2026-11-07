"""One new-source, fixed-recipe prequential transport experiment.

This reuses the reference-preserving transport mathematics byte-for-byte. The
source task, labels and independent whole-component groups are new. No label
from a current batch is used to select its calibration context or forecast.
"""
import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
import task_io as io
from empirical_native import apply_spline, loss_gap_envelope
from simplex_transport import fit_global, fit_context
from calibration_views import fit_views, fit_beta, apply_beta


def fixed_views(records, anchor, selected):
    laws = [r['law'] for r in records]
    targets = [r['native_target'] for r in records]
    context = np.asarray([r['bounded_descriptors8'] for r in records])
    fitted = {}
    for index in sorted({j for j in (anchor, selected) if j is not None}):
        fitted[index] = fit_context(laws, targets, context[:, index])
        fitted[index]['descriptor_index'] = index
    identity = {'status': 'IDENTITY_NO_ELIGIBLE_CONTEXT', 'kind': 'identity'}
    return {'global': fit_global(laws, targets), 'anchor': deepcopy(fitted.get(anchor, identity)),
            'auto': deepcopy(fitted.get(selected, identity)), 'anchor_index': anchor,
            'auto_index': selected, 'descriptor_reselected_on_future_batches': False}


def corrected(law, fitted, descriptor, index):
    if fitted['kind'] == 'identity' or fitted['status'].startswith('FAILED'):
        return deepcopy(law)
    if fitted['kind'] == 'global': return apply_spline(law, fitted['theta0'])
    if index not in range(8): raise ValueError('frozen eligible descriptor index required')
    return apply_spline(law, fitted['theta0'], fitted['theta1'], descriptor[index])


def transform_from_fit(X):
    values = np.asarray(X, dtype=float)
    center, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    constant = np.flatnonzero(scale == 0).tolist()
    scale[scale == 0] = 1.
    return {'center': center.tolist(), 'sd': scale.tolist(), 'feature_ids': io.FEATURE_IDS,
            'fit_role_only': True, 'rows': 256, 'zero_variance_features_scale1': constant,
            'fixed_across_OOF_and_future_batches': True,
            'OOF_transform_sees_fit_features_but_never_heldout_Y': True}


def produce_oof(fit, X, bounded, reference, transform, folder):
    identities = [r['identity'] for r in fit]
    y = [r['native_values'][2] for r in fit]
    ordered = sorted(range(256), key=lambda i: (
        hashlib.sha256(('dielectric_transport_oof_v1|' + identities[i]['component_id']).encode()).hexdigest(),
        identities[i]['component_id']))
    folds = {row: rank % 5 for rank, row in enumerate(ordered)}
    io.write(folder / 'OOF_ASSIGNMENT_BEFORE_FITS.json', {
        'actual_utc': io.utc(), 'hash_prefix': 'dielectric_transport_oof_v1|',
        'records': [{'fit_row': i, 'component_id': r['component_id'], 'fold': folds[i]} for i, r in enumerate(identities)],
        'fold_sizes': [sum(v == j for v in folds.values()) for j in range(5)]})
    records, index = [None] * 256, []
    for fold in range(5):
        held = [i for i in range(256) if folds[i] == fold]
        train = [i for i in range(256) if folds[i] != fold]
        laws, state = io.fit_reference(reference, 'RF10', [X[i] for i in train], [y[i] for i in train],
                                        [X[i] for i in held], transform)
        state.update(fold=fold, training_component_ids=[identities[i]['component_id'] for i in train],
                     heldout_component_ids=[identities[i]['component_id'] for i in held])
        name = 'RF_OOF_STATE_' + str(fold) + '.json'
        io.write(folder / name, state)
        index.append({'kind': 'RF_OOF', 'fold': fold, 'train_rows': len(train), 'prediction_rows': len(held),
                      'state_file': name, 'state_sha256': io.sha(folder / name)})
        for i, law in zip(held, laws):
            records[i] = {'fit_row': i, 'identity': identities[i], 'law': law, 'native_target': y[i],
                          'bounded_descriptors8': bounded[i], 'mode': 'OOF_EXCLUDED_WHOLE_FOLD',
                          'fold': fold, 'reference_state_file': name, 'reference_state_sha256': io.sha(folder / name)}
    if any(r is None for r in records): raise ValueError('complete OOF256 coverage required')
    io.write(folder / 'OOF_RECORDS256.json', {'actual_utc': io.utc(), 'records': records,
             'OOF_laws_for_calibrator_training_not_independent_test': True})
    return records, index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    plan = io.validate(args.plan, args.approval, 'produce_prequential_forecasts')
    roles = io.load_roles(plan)
    reference = io.import_reference(plan)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    io.write(folder / 'START.json', {'actual_utc': io.utc(), 'plan_sha256': io.sha(args.plan),
             'source_sha256': io.sources(), 'future_targets_emitted': 0})
    try:
        disclosure = io.Disclosure(plan['inputs']['raw_source']['file'], roles)
        fit, visible = disclosure.initial()
        io.write(folder / 'INITIAL_FIT256.json', {'rows': fit, 'real_computed_report_cells': 768})
        io.write(folder / 'VISIBLE_CONTEXTS_ONLY.json', visible)
        pairs = [io.descriptor_pair(r['identity']) for r in fit]
        raw, bounded = [p[0] for p in pairs], [p[1] for p in pairs]
        X = [r['native_values'][:2] + d for r, d in zip(fit, raw)]
        y = [r['native_values'][2] for r in fit]
        transform = transform_from_fit(X)
        io.write(folder / 'BASELINE_TRANSFORM.json', transform)
        calibration, fits = produce_oof(fit, X, bounded, reference, transform, folder)
        initial_views = fit_views([r['law'] for r in calibration], y, bounded)
        initial_beta = fit_beta([r['law'] for r in calibration], y)
        anchor, selected = initial_views['anchor_index'], initial_views['auto_index']
        io.write(folder / 'INITIAL_CALIBRATION_ONLY256.json', {'actual_utc': io.utc(),
                 'views': initial_views, 'beta': initial_beta, 'all_objectives_are_past_OOF_training': True})
        io.write(folder / 'DESCRIPTOR_SELECTION_FROZEN.json', {'actual_utc': io.utc(), 'anchor_index': anchor,
                 'auto_index': selected, 'initial_fit_sha256': io.sha(folder / 'INITIAL_CALIBRATION_ONLY256.json'),
                 'future_descriptor_reselection': False, 'future_targets_emitted_before_selection': 0})
        batches = []
        for number, role in enumerate(io.BATCHES):
            batch = folder / role
            batch.mkdir()
            expected = 256 + number * 64
            if len(X) != expected or len(calibration) != expected:
                raise ValueError('same common past training/calibration count required')
            if len({r['identity']['component_id'] for r in calibration}) != expected:
                raise ValueError('one past record per original component required')
            io.write(batch / 'CALIBRATION_HISTORY_BEFORE.json', {'records': calibration,
                     'old_CDF_never_recomputed_with_current_in_sample_RF': True})
            views = initial_views if number == 0 else fixed_views(calibration, anchor, selected)
            beta = initial_beta if number == 0 else fit_beta([r['law'] for r in calibration],
                                                            [r['native_target'] for r in calibration])
            io.write(batch / 'STATE_BEFORE.json', {'actual_utc': io.utc(), 'views': views, 'beta': beta,
                     'training_rows': len(X), 'calibration_rows': len(calibration),
                     'anchor_index': anchor, 'auto_index': selected,
                     'calibration_history_sha256': io.sha(batch / 'CALIBRATION_HISTORY_BEFORE.json')})
            identities = [r['identity'] for r in visible[role]]
            proxies = [r['native_proxies'] for r in visible[role]]
            descriptors = [io.descriptor_pair(r) for r in identities]
            query = [p + d[0] for p, d in zip(proxies, descriptors)]
            rf, rf_state = io.fit_reference(reference, 'RF10', X, y, query, transform)
            gp, gp_state = io.fit_reference(reference, 'GP10', X, y, query, transform)
            for kind, state in (('RF10', rf_state), ('GP10', gp_state)):
                state['training_component_ids'] = [r['identity']['component_id'] for r in calibration]
                state['prediction_component_ids'] = [r['component_id'] for r in identities]
                io.write(batch / (kind + '_STATE.json'), state)
                fits.append({'kind': kind, 'batch': number + 1, 'train_rows': len(X), 'prediction_rows': 64,
                             'state_file': role + '/' + kind + '_STATE.json',
                             'state_sha256': io.sha(batch / (kind + '_STATE.json'))})
            if len(rf) != 64 or len(gp) != 64: raise ValueError('complete64 references required')
            forecasts, envelopes = [], []
            for i, identity in enumerate(identities):
                g = descriptors[i][1]
                predicted = {'RF_identity': rf[i], 'RF_beta': apply_beta(rf[i], beta),
                             'RF_global4': corrected(rf[i], views['global'], g, None),
                             'RF_anchor4': corrected(rf[i], views['anchor'], g, anchor),
                             'RF_auto4': corrected(rf[i], views['auto'], g, selected), 'GP10': gp[i],
                             'electronic_delta': {'coordinate': 'native', 'family': 'weighted_empirical',
                                 'support_native': [proxies[i][1]], 'weight_fractions': ['1'], 'unit': io.UNIT,
                                 'scope': 'point electronic-only sanity control; ignores ionic contribution; zero fit'}}
                for method in io.METHODS:
                    forecasts.append({'method': method, 'identity': identity, 'known_native_proxies': proxies[i],
                                      'bounded_descriptors8': g, 'law': predicted[method]})
                for method in ('RF_beta', 'RF_global4', 'RF_anchor4', 'RF_auto4'):
                    envelopes.append({'method': method, 'component_id': identity['component_id'],
                                      'loss_gap_envelope': loss_gap_envelope(rf[i], predicted[method])})
            io.write(batch / 'FORECASTS.json', {'actual_utc': io.utc(), 'records': forecasts,
                     'current_batch_targets_given_to_models': 0})
            io.write(batch / 'PRETARGET_LOSS_GAP_ENVELOPES.json', {'records': envelopes,
                     'exact_rational_saved_support_mass_arithmetic': True, 'not_future_risk_guarantee': True})
            if io.sources() != plan['source_sha256']: raise ValueError('scientific source changed before truth')
            names = ('CALIBRATION_HISTORY_BEFORE.json', 'STATE_BEFORE.json', 'RF10_STATE.json',
                     'GP10_STATE.json', 'FORECASTS.json', 'PRETARGET_LOSS_GAP_ENVELOPES.json')
            io.write(batch / 'FORECAST_FREEZE.json', {'status': 'ALL448_BATCH_LAWS_FROZEN_BEFORE64Y',
                     'actual_utc': io.utc(), 'role': role, 'plan_sha256': io.sha(args.plan),
                     'source_sha256': io.sources(), 'methods': list(io.METHODS), 'prediction_count': 448,
                     'files_sha256': {name: io.sha(batch / name) for name in names}})
            truth = disclosure.reveal(role, batch / 'FORECAST_FREEZE.json', io.sources(), io.sha(args.plan))
            io.write(batch / 'TRUE_BATCH_REVEAL.json', truth)
            revealed = [r['native_target'] for r in truth['rows']]
            for identity, target, law, d in zip(identities, revealed, rf, descriptors):
                calibration.append({'identity': identity, 'law': law, 'native_target': target,
                                    'bounded_descriptors8': d[1], 'mode': 'PRETARGET_FROZEN_RF', 'role': role,
                                    'reference_forecasts_sha256': io.sha(batch / 'FORECASTS.json'),
                                    'reference_freeze_sha256': io.sha(batch / 'FORECAST_FREEZE.json')})
            X.extend(query)
            y.extend(revealed)
            batches.append({'role': role, 'training_rows_before': expected,
                            'freeze_sha256': io.sha(batch / 'FORECAST_FREEZE.json'),
                            'truth_sha256': io.sha(batch / 'TRUE_BATCH_REVEAL.json')})
        io.write(folder / 'FINAL_HISTORY448.json', {'calibration_records': calibration,
                 'last_batch_not_used_for_extra_evaluation': True})
        io.write(folder / 'FIT_INDEX.json', {'records': fits, 'RF_OOF_fits': 5, 'RF_batch_fits': 3,
                 'GP_batch_fits': 3, 'actual_sklearn_fits': 11, 'point_sanity_control_fits': 0})
        io.write(folder / 'DISCLOSURE_LEDGER.json', disclosure.ledger())
        pins = {p.relative_to(folder).as_posix(): io.sha(p) for p in sorted(folder.rglob('*.json'))}
        io.write(folder / 'ALL_BATCHES_FROZEN.json', {'status': 'ALL1344_PREQUENTIAL_LAWS_AND192_TRUE_OUTCOMES_COMPLETE',
                 'actual_utc': io.utc(), 'plan_sha256': io.sha(args.plan), 'source_sha256': io.sources(),
                 'files_sha256': pins, 'batches': batches, 'actual_sklearn_fits': 11, 'validation_score_calls': 0,
                 'new_source_components': 448, 'initial_fit_components': 256, 'prequential_test_components': 192,
                 'initial_fit_report_cells': 768, 'future_report_cells': 576,
                 'new_wet_measurements': 0, 'new_DFT_calculations': 0, 'elapsed_seconds': time.perf_counter() - start})
        print({'status': 'ALL_BATCHES_COMPLETE', 'forecasts': 1344, 'actual_sklearn_fits': 11,
               'selected_g': selected, 'anchor_g': anchor, 'new_source_components': 448})
    except Exception as problem:
        io.write(folder / 'FAILED_STOP.json', {'actual_utc': io.utc(), 'type': type(problem).__name__,
                 'message': str(problem), 'source_sha256': io.sources(), 'no_same_cohort_posthoc_rescue': True})
        raise


if __name__ == '__main__': main()
