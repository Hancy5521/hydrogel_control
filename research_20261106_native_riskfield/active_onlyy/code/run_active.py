"""Fixed registered six-policy active-Y experiment; true labels enter only after law freezes."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
import data_io as task
import posterior_update as posterior
import reference_models as references
import risk_readout as risk


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def versions():
    values = {name: importlib.metadata.version(name) for name in task.configuration()['libraries']}
    if values != task.configuration()['libraries']: raise ValueError('registeredlibraryversions required')
    return values


def proxy_z(warm_model, transform, values):
    values = np.asarray(values, dtype=float)
    return ((warm_model.yj(values / np.asarray(transform['native_first_scale_s0'])[:2])
             - np.asarray(transform['u_center_warm'])[:2]) / np.asarray(transform['u_sample_SD_warm'])[:2])


def law(nig, context, response_transform):
    predicted = posterior.scalar_law(nig, context)
    return {'coordinate': 'u=YJ_half(native/s0)', 'family': 'student_t',
        'location': response_transform['center'] + response_transform['sd'] * predicted['mu'],
        'scale': response_transform['sd'] * predicted['scale'], 'df': predicted['nu'],
        'native_first_scale_s0': response_transform['s0'], 'lambda': .5,
        'unit': 'source bulk GPa attribution, aggregation qualified'}


def ivr_values(nig, pool, terminal):
    q, t = np.column_stack((np.ones(len(pool)), pool)), np.column_stack((np.ones(len(terminal)), terminal))
    covariance = np.asarray(nig['V']); qdiag = 1 + np.einsum('ij,jk,ik->i', q, covariance, q)
    cross = q @ covariance @ t.T
    return (nig['b'] / (nig['alpha'] - 1) * np.mean(cross * cross / qdiag[:, None], axis=1)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True); args = parser.parse_args()
    plan, roles = task.validate(args.plan, args.approval, 'forecast'); libraries = versions()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    task.write(output / 'EXECUTION_START.json', {'actual_utc': task.utc(), 'source_sha256': task.sources(),
        'plan_sha256': task.sha(args.plan), 'approval_sha256': task.sha(args.approval), 'terminalY_model_access': 0})
    task.write(output / 'LIBRARY_VERSIONS.json', {'libraries': libraries, 'Python': sys.version})
    try:
        oracle = task.Disclosure(plan['inputs']['raw_source']['file'], roles)
        visible = oracle.initial(); task.write(output / 'VISIBLE_INPUTS.json', visible)
        warm_native = [r['native_values'] for r in visible['warm']]
        pool_native = [r['native_proxies'] for r in visible['pool']]
        terminal_native = [r['native_proxies'] for r in visible['terminal']]
        warm_model = task.warm_module(plan)
        # Only the pooled posterior and fixed transform are retained; no atom/Beta model.
        initialized = warm_model.fit_warm(warm_native)
        transform = initialized['transform']; warm_pooled = initialized['pooled']
        warm_nig = posterior.from_niw(warm_pooled); del initialized
        pool = proxy_z(warm_model, transform, pool_native); terminal = proxy_z(warm_model, transform, terminal_native)
        response = {'center': transform['u_center_warm'][2], 'sd': transform['u_sample_SD_warm'][2],
                    's0': transform['native_first_scale_s0'][2]}
        native_features = {'center': transform['proxy_native_center_warm'], 'sd': transform['proxy_native_sample_SD_warm']}
        models_dir = output / 'models'; models_dir.mkdir()
        preq_dir = output / 'prequential'; preq_dir.mkdir()
        fits, prequential, primary, ref_forecasts, checkpoints = [], [], [], [], []

        def fit_reference(method, train, prediction, fit_id, context):
            current_laws, state = (references.fit_gp(train, prediction, native_features) if method == 'GP'
                                   else references.fit_forest(train, prediction, native_features))
            file = models_dir / (fit_id + '.json')
            task.write(file, {'fit_id': fit_id, 'method': method, 'context': context,
                'source_sha256': task.sources(), 'actual_library_versions': versions(),
                'train_native_values_sha256': json_sha(train), 'state': state})
            pin = task.sha(file); fits.append({'fit_id': fit_id, 'method': method, 'file': 'models/' + file.name,
                'sha256': pin, 'train_rows': len(train)})
            return current_laws, pin

        gp_warm_all, gp_warm_pin = fit_reference('GP', warm_native, pool_native + terminal_native,
                                                 'GP_warm_shared', 'sharedwarmReferenceAndGPpolicy0')
        rf_warm, rf_warm_pin = fit_reference('RF', warm_native, terminal_native, 'RF_warm_shared', 'sharedwarmReference')
        gp_warm_terminal = gp_warm_all[24:]
        warm_laws = [law(warm_nig, z, response) for z in terminal]
        random_order = np.random.default_rng(20261106).permutation(24).tolist()
        for policy in task.POLICIES:
            nig = copy.deepcopy(warm_nig); selected = []; history = []; train = copy.deepcopy(warm_native)
            for item, predicted in zip(visible['terminal'], warm_laws):
                primary.append({'component_id': item['identity']['component_id'], 'jid': item['identity']['jid'],
                    'policy': policy, 'budget': 0, 'law': predicted, 'shared_warm_law_alias': True})
            if policy in task.REF_POLICIES:
                for method, predictions, pin in [('GP', gp_warm_terminal, gp_warm_pin), ('RF', rf_warm, rf_warm_pin)]:
                    for item, predicted in zip(visible['terminal'], predictions):
                        ref_forecasts.append({'component_id': item['identity']['component_id'], 'jid': item['identity']['jid'],
                            'policy': policy, 'budget': 0, 'method': method, 'law': predicted, 'fit_state_sha256': pin,
                            'shared_warm_actual_fit_alias': True})
            if policy == 'gp_variance':
                for item, predicted in zip(visible['terminal'], gp_warm_terminal):
                    ref_forecasts.append({'component_id': item['identity']['component_id'], 'jid': item['identity']['jid'],
                        'policy': policy, 'budget': 0, 'method': 'GP', 'law': predicted,
                        'fit_state_sha256': gp_warm_pin, 'shared_warm_actual_fit_alias': True})
            checkpoints.append({'policy': policy, 'budget': 0, 'nig': nig, 'selected_pool_indices': [],
                                'actual_utc': task.utc(), 'source_sha256': task.sources()})
            for step in range(8):
                available = [i for i in range(24) if i not in selected]
                acquisition_nig = copy.deepcopy(nig)
                diagnostics = {}; gp_policy_pin = None
                if policy == 'warm_openloop':
                    acquisition_nig = copy.deepcopy(warm_nig); acquisition_nig['V'] = copy.deepcopy(nig['V'])
                    acquisition_nig['alpha'] = warm_nig['alpha'] + step / 2
                    acquisition_nig['b'] = (acquisition_nig['alpha'] - 1) * warm_nig['b'] / (warm_nig['alpha'] - 1)
                    acquisition_nig['labeled_updates'] = step
                if policy in ('native_student', 'native_gaussian', 'warm_openloop'):
                    result = risk.risk_values(acquisition_nig, pool[available], terminal, response,
                                              family='normal' if policy == 'native_gaussian' else 'student_t')
                    scores = result['values']; diagnostics = result['diagnostics']
                elif policy == 'latent_ivr':
                    scores = ivr_values(nig, pool[available], terminal)
                elif policy == 'random':
                    scores = [float(24 - random_order.index(i)) for i in available]
                    diagnostics = {'fixed_seed': 20261106, 'fixed_permutation': random_order,
                                   'ranking_is_random_schedule_not_risk': True}
                elif policy == 'gp_variance':
                    if step == 0:
                        predictions = gp_warm_all[:24]; gp_policy_pin = gp_warm_pin
                    else:
                        predictions_all, gp_policy_pin = fit_reference('GP', train, pool_native + terminal_native,
                            'GP_policy_step%02d' % step, 'gp_variance_policy_acquisition_and_own_predictions')
                        predictions = predictions_all[:24]
                        if step == 4:
                            for target, predicted in zip(visible['terminal'], predictions_all[24:]):
                                ref_forecasts.append({'component_id': target['identity']['component_id'], 'jid': target['identity']['jid'],
                                    'policy': policy, 'budget': 4, 'method': 'GP', 'law': predicted,
                                    'fit_state_sha256': gp_policy_pin, 'same_actual_train_queries': list(selected),
                                    'reused_step4_fit_not_additional_fit': True})
                    scores = [predictions[i]['scale'] ** 2 for i in available]
                    diagnostics = {'native_GP_predictive_variance_with_White': True,
                                   'GP_actual_fit_state_sha256': gp_policy_pin, 'step0_shared_not_refit': step == 0}
                if len(scores) != len(available) or not np.isfinite(scores).all():
                    raise ArithmeticError('allremainingactions need finite fixed-objective acquisition scores')
                # Strict > leaves ties at the first original role index.
                choice_position = max(range(len(available)), key=lambda j: scores[j]); chosen = available[choice_position]
                item = visible['pool'][chosen]; before = copy.deepcopy(nig)
                predicted = law(nig, pool[chosen], response)
                stem = policy + '_step%02d' % step
                before_file = preq_dir / (stem + '_FORECAST.json')
                task.write(before_file, {'status': 'ACTION_AND_PREQUENTIAL_LAW_FROZEN_BEFORE_QUERY_Y',
                    'actual_utc': task.utc(), 'source_sha256': task.sources(), 'plan_sha256': task.sha(args.plan),
                    'policy': policy, 'budget_before_query': step, 'pool_index': chosen, 'identity': item['identity'],
                    'native_query_target_visible': False, 'known_native_proxies': item['native_proxies'], 'law': predicted,
                    'actual_predictive_nig_before': before, 'acquisition_nig': acquisition_nig,
                    'warmopenloop_acquisition_surrogate_not_actual_posterior': policy == 'warm_openloop',
                    'available_pool_indices': available, 'acquisition_values': scores, 'acquisition_diagnostics': diagnostics,
                    'selected_position': choice_position, 'sameunit_query': [item['identity']['component_id'], 'bulk modulus'],
                    'source_GP_fit_sha256_if_policy': gp_policy_pin, 'common_setup_cost': 368})
                revealed = oracle.query_Y(policy, chosen, before_file, task.sources(), task.sha(args.plan))
                native_row = item['native_proxies'] + [revealed['native_target']]
                modeled_y = float((warm_model.yj(revealed['native_target'] / response['s0'])
                                   - response['center']) / response['sd'])
                nig = posterior.update_observed_label(nig, pool[chosen], modeled_y)
                train.append(native_row); selected.append(chosen); history.append(revealed)
                after_file = preq_dir / (stem + '_OUTCOME_UPDATE.json')
                task.write(after_file, {'actual_utc': task.utc(), 'policy': policy, 'budget_after_query': step + 1,
                    'pool_index': chosen, 'identity': item['identity'], 'prequery_forecast_sha256': task.sha(before_file),
                    'true_Y_reveal': revealed, 'native_complete_row': native_row, 'modeled_zY': modeled_y,
                    'nig_after': nig, 'no_generated_label': True})
                prequential.append({'policy': policy, 'step': step, 'pool_index': chosen,
                    'forecast_file': 'prequential/' + before_file.name, 'forecast_sha256': task.sha(before_file),
                    'outcome_file': 'prequential/' + after_file.name, 'outcome_sha256': task.sha(after_file)})
                if step + 1 in task.BUDGETS:
                    budget = step + 1
                    for target, z in zip(visible['terminal'], terminal):
                        primary.append({'component_id': target['identity']['component_id'], 'jid': target['identity']['jid'],
                                        'policy': policy, 'budget': budget, 'law': law(nig, z, response)})
                    checkpoints.append({'policy': policy, 'budget': budget, 'nig': nig,
                        'selected_pool_indices': list(selected), 'observed_native_training_sha256': json_sha(train),
                        'actual_utc': task.utc(), 'source_sha256': task.sources(), 'actual_library_versions': versions()})
                    if policy in task.REF_POLICIES:
                        for method in ('GP', 'RF'):
                            predictions, pin = fit_reference(method, train, terminal_native,
                                '%s_%s_budget%d' % (method, policy, budget), 'sameactualtrainingStrongPredictionReference')
                            for target, predicted in zip(visible['terminal'], predictions):
                                ref_forecasts.append({'component_id': target['identity']['component_id'], 'jid': target['identity']['jid'],
                                    'policy': policy, 'budget': budget, 'method': method, 'law': predicted,
                                    'fit_state_sha256': pin, 'same_actual_train_queries': list(selected)})
            if policy == 'gp_variance':
                predictions, pin = fit_reference('GP', train, terminal_native,
                    'GP_policy_final_budget8', 'gp_variance_policy_own_final_predictions')
                for target, predicted in zip(visible['terminal'], predictions):
                    ref_forecasts.append({'component_id': target['identity']['component_id'], 'jid': target['identity']['jid'],
                        'policy': policy, 'budget': 8, 'method': 'GP', 'law': predicted,
                        'fit_state_sha256': pin, 'same_actual_train_queries': list(selected)})
            if len(selected) != 8: raise AssertionError('everypolicy mustuseexactly8distinctYcells')
        if len(primary) != 1152 or len(ref_forecasts) != 960 or len(prequential) != 48 or len(fits) != 18:
            raise AssertionError('fixedscientificcounts changed')
        if task.sources() != plan['source_sha256']: raise ValueError('science sourcechanged duringrun')
        task.write(output / 'MODEL_STATES.json', {'transform': transform, 'response_transform': response,
            'native_reference_feature_transform': native_features, 'warm_pooled_NIW': warm_pooled,
            'warm_NIG': warm_nig, 'checkpoint_states': checkpoints, 'allpool_standardized_known_proxies': pool.tolist(),
            'allterminal_standardized_known_proxies': terminal.tolist(), 'singlepooledNIG_noBetaOrSpike': True})
        task.write(output / 'QUERY_LEDGER.json', oracle.ledger())
        task.write(output / 'TERMINAL_FORECASTS.json', {'forecasts': primary})
        task.write(output / 'REFERENCE_FORECASTS.json', {'forecasts': ref_forecasts})
        task.write(output / 'PREQUENTIAL_INDEX.json', {'records': prequential})
        task.write(output / 'FIT_INDEX.json', {'records': fits, 'actual_GP': 13, 'actual_RF': 5,
            'sharedwarm_references_and_GPpolicy0_are_actual_aliases': True})
        names = list(task.BASE_FILES) + [r[key] for r in prequential for key in ('forecast_file', 'outcome_file')] + [r['file'] for r in fits]
        task.write(output / 'FORECAST_FREEZE.json', {'status': 'ALL2112_TERMINAL_AND48_PREQUENTIAL_LAWS_FROZEN',
            'actual_utc': task.utc(), 'source_sha256': task.sources(), 'plan_sha256': task.sha(args.plan),
            'files': {name: task.sha(output / name) for name in names}, 'terminal_Y_model_access': 0,
            'logical_terminal_laws': 2112, 'logical_prequential_laws': 48, 'unique_material_roles': 152,
            'actual_sklearn_fits': 18, 'no_real_new_wet_or_DFT': True})
        print(json.dumps({'status': 'ALL2112_TERMINAL_AND48_PREQUENTIAL_LAWS_FROZEN', 'sklearn_fits': 18}))
    except Exception as problem:
        task.write(output / 'FAILED_STOP.json', {'actual_utc': task.utc(), 'type': type(problem).__name__,
            'message': str(problem), 'source_sha256': task.sources(), 'retain_no_samecohort_recipe_rescue': True})
        raise


if __name__ == '__main__': main()
