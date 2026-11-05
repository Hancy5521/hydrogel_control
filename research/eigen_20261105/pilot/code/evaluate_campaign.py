"""One post-freeze evaluation of registered offline source outcomes.

This module never fits or selects a policy. Real experimental values and DFT
values remain separate datasets; episodes/aliases do not create new materials.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
from scipy.special import ndtr

from benchmark_io import (CHECKPOINTS, POLICIES, SEEDS, evaluator_truth, read, expected_schedule,
                          sha, source_bindings, utc, validate_bindings, write)


def gaussian_crps(mean, sd, truth):
    m, s, y = (np.asarray(v, dtype=float) for v in (mean, sd, truth))
    if m.shape != s.shape or m.shape != y.shape or not m.size:
        raise ValueError('Aligned nonempty Gaussian score arrays required')
    if not all(np.isfinite(v).all() for v in (m, s, y)) or (s <= 0).any():
        raise ValueError('Finite responses/means and positive predictive sd required')
    z = (y - m) / s
    phi = np.exp(-z * z / 2.) / np.sqrt(2. * np.pi)
    result = s * (z * (2. * ndtr(z) - 1.) + 2. * phi - 1. / np.sqrt(np.pi))
    if not np.isfinite(result).all():
        raise OverflowError('CRPS overflow, no successful-subset scoring')
    return result


def verify_freeze(directory, plan_path, plan):
    frozen = directory / 'ALL_FORECASTS_FROZEN.json'
    if not frozen.is_file() or (directory / 'EXECUTION_FAILED.json').exists():
        raise ValueError('A complete successful forecast freeze is required')
    record = read(frozen)
    if record['plan_sha256'] != sha(plan_path) or record['source_sha256'] != source_bindings():
        raise ValueError('Frozen plan/source binding changed')
    actual, logical = expected_schedule()
    if (record['actual_campaigns'] != [r['name'] for r in actual]
            or record['logical_episodes'] != logical):
        raise ValueError('Exact40actual identities and64logical aliases are required')
    files = record['files']
    expected = {entry['file'] for entry in files}
    if len(expected) != len(files):
        raise ValueError('Duplicate frozen path')
    campaign_files = ('INITIAL_REFERENCE.json', 'ACTIONS.json', 'CAMPAIGN_COMPLETE.json',
                      'forecast_00.npz', 'forecast_08.npz', 'forecast_16.npz', 'forecast_24.npz')
    required = {'EXECUTION_START.json'} | {r['name'] + '/' + name for r in actual for name in campaign_files}
    if expected != required:
        raise ValueError('Every one of the281 execution/forecast/action artifacts must be bound, exactly once')
    by_file = {entry['file']: entry for entry in files}
    for entry in files:
        name = entry['file']
        p = directory / name
        if p.resolve().is_relative_to(directory.resolve()) is not True or p.stat().st_size != entry['bytes'] or sha(p) != entry['sha256']:
            raise ValueError('Frozen forecast or action receipt changed: ' + name)
    start = read(directory / 'EXECUTION_START.json')
    if start['plan_sha256'] != sha(plan_path) or start['source_sha256'] != source_bindings():
        raise ValueError('Execution start belongs to another protocol/source')
    for identity in actual:
        folder = directory / identity['name']
        for filename in campaign_files[:3]:
            receipt = read(folder / filename)
            if any(receipt[k] != identity[k] for k in ('dataset', 'policy', 'seed')):
                raise ValueError('Campaign identity binding mismatch: ' + identity['name'])
        completed = read(folder / 'CAMPAIGN_COMPLETE.json')
        desired = [{'step': step, 'file': 'forecast_' + str(step).zfill(2) + '.npz',
                    'sha256': by_file[identity['name'] + '/forecast_' + str(step).zfill(2) + '.npz']['sha256'],
                    'new_bundle_cost': step} for step in CHECKPOINTS]
        if completed['snapshots'] != desired or completed['no_evaluator_or_test_truth_access'] is not True:
            raise ValueError('Incomplete or misbound campaign snapshots')
    return record


def score_snapshot(dataset, design, truth, campaign_dir, step):
    path = campaign_dir / ('forecast_' + str(step).zfill(2) + '.npz')
    with np.load(path, allow_pickle=False) as law:
        mean = law['mean_standardized'].reshape(-1, 3)
        covariance = law['covariance_standardized']
        observed = law['observed_mask'].reshape(-1, 3)
        location, scale = law['warm_location'], law['warm_scale']
    n = len(design['rows'])
    if mean.shape != (n, 3) or covariance.shape != (3 * n, 3 * n) or truth.shape != (n, 3):
        raise ValueError('Frozen law/source coordinates are not aligned')
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Finite positive frozen warm units required')
    test_rows = [i for i, m in enumerate(design['rows']) if m['role'] == 'test']
    test_indices = [3 * i + p for i in test_rows for p in range(3)]
    if observed[test_rows].any():
        raise ValueError('Campaign had observed a test target')
    test_covariance = covariance[np.ix_(test_indices, test_indices)]
    np.linalg.cholesky(test_covariance)
    standardized_truth = (truth[test_rows] - location) / scale
    predictions = mean[test_rows]
    sd = np.sqrt(np.diag(test_covariance)).reshape(-1, 3)
    point_crps = gaussian_crps(predictions, sd, standardized_truth)
    point_mse = (predictions - standardized_truth) ** 2
    grouped = defaultdict(list)
    for j, row in enumerate(test_rows):
        grouped[design['rows'][row]['component_id']].append(j)
    per_group = {g: {'CRPS': float(point_crps[idx].mean()),
                      'MSE': float(point_mse[idx].mean()), 'native_rows': len(idx)}
                 for g, idx in grouped.items()}
    crps = float(np.mean([r['CRPS'] for r in per_group.values()]))
    rmse = float(np.sqrt(np.mean([r['MSE'] for r in per_group.values()])))
    actions = read(campaign_dir / 'ACTIONS.json')
    if len(actions['campaign']) != 24 or actions['new_bundle_cost'] != 24:
        raise ValueError('A complete24bundle campaign is required')
    selected = actions['campaign'][:step]
    source_values_checked = 0
    for record in actions['initial'] + [r['oracle'] for r in selected]:
        row = record['row_position']
        actual = truth[row, record['property_indices']].tolist()
        if actual != record['native_values']:
            raise ValueError('Oracle receipt differs from original native source values')
        source_values_checked += len(actual)
    frontier = [i for i, m in enumerate(design['rows']) if m['role'] == 'frontier']
    sign = design['frontier_utility_sign']
    group_truth = defaultdict(list)
    for i in frontier:
        group_truth[design['rows'][i]['component_id']].append(float(sign * truth[i, 2]))
    true_group_maxima = {g: max(values) for g, values in group_truth.items()}
    threshold = float(np.quantile(list(true_group_maxima.values()), .9, method='linear'))
    verified = {}
    for row in selected:
        record = row['oracle']
        i = record['row_position']
        if design['rows'][i]['role'] == 'frontier' and 2 in record['property_indices']:
            group = design['rows'][i]['component_id']
            value = record['native_values'][record['property_indices'].index(2)]
            verified[group] = max(verified.get(group, -np.inf), float(sign * value))
    best = max(verified.values()) if verified else None
    return {'checkpoint_new_bundle_cost': step, 'equal_group_standardized_CRPS': crps,
            'equal_group_standardized_RMSE': rmse, 'per_test_group': per_group,
            'test_metadata_groups': len(grouped), 'test_rows': len(test_rows),
            'new_revealed_cells': sum(len(x['oracle']['property_indices']) for x in selected),
            'initial_bundle_cost': actions['initial_bundle_cost'],
            'native_source_cells_checked': source_values_checked,
            'frontier_retrieval': {'utility_sign': sign,
                'meaning': 'low_kappa_property_extreme_not_zT_optimization' if sign == -1 else 'high_source_bulk_property_extreme_not_synthesis',
                'top10pct_group_utility_threshold_evaluator_only': threshold,
                'verified_frontier_groups_with_target': len(verified),
                'verified_top_decile_group_hits': sum(v >= threshold for v in verified.values()),
                'best_verified_utility': best,
                'simple_regret_to_known_offline_pool': None if best is None else max(true_group_maxima.values()) - best,
                'unqueried_model_suggestions_count_as_verified_discoveries': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--campaign-dir', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    validate_bindings(plan)
    directory, output = Path(args.campaign_dir).resolve(), Path(args.output_dir).resolve()
    freeze = verify_freeze(directory, args.plan, plan)
    approval = read(args.approval)
    if (approval.get('approved') is not True or approval.get('scope') != 'SCORE_FROZEN_OFFLINE_PILOT'
            or approval.get('plan_sha256') != sha(args.plan)
            or approval.get('freeze_sha256') != sha(directory / 'ALL_FORECASTS_FROZEN.json')):
        raise ValueError('Exact root scoring approval required')
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EVALUATOR_REVEAL_STARTED.json', {'actual_utc': utc(),
        'plan_sha256': sha(args.plan), 'forecast_freeze_sha256': sha(directory / 'ALL_FORECASTS_FROZEN.json'),
        'approval_sha256': sha(args.approval), 'intent': 'ABOUT_TO_LOAD_PREEXISTING_PUBLIC_OFFLINE_TEST_RESPONSES'})
    try:
        truth = {dataset: evaluator_truth(dataset, plan['data_directory'], design, forecasts_verified=True)
                 for dataset, design in plan['datasets'].items()}
        actual = {}
        for name in freeze['actual_campaigns']:
            receipt = read(directory / name / 'CAMPAIGN_COMPLETE.json')
            dataset = receipt['dataset']
            actual[name] = {'dataset': dataset, 'policy': receipt['policy'], 'seed': receipt['seed'],
                           'scores': [score_snapshot(dataset, plan['datasets'][dataset], truth[dataset], directory / name, step)
                                      for step in CHECKPOINTS]}
        comparisons = []
        for dataset in plan['datasets']:
            for policy in POLICIES:
                for step in CHECKPOINTS:
                    rows = []
                    for seed in SEEDS:
                        a = next(x for x in freeze['logical_episodes'] if x['dataset'] == dataset and x['policy'] == policy and x['seed'] == seed)
                        b = next(x for x in freeze['logical_episodes'] if x['dataset'] == dataset and x['policy'] == 'joint_lookahead_all' and x['seed'] == seed)
                        left = next(x for x in actual[a['campaign']]['scores'] if x['checkpoint_new_bundle_cost'] == step)
                        right = next(x for x in actual[b['campaign']]['scores'] if x['checkpoint_new_bundle_cost'] == step)
                        rows.append(left['equal_group_standardized_CRPS'] - right['equal_group_standardized_CRPS'])
                    comparisons.append({'dataset': dataset, 'policy': policy, 'checkpoint': step,
                        'CRPS_minus_joint_lookahead_by_logical_seed': rows,
                        'mean_CRPS_difference': float(np.mean(rows)),
                        'negative_favors_named_policy': True,
                        'four_trajectories_not_four_independent_material_cohorts': True})
        result = {'actual_end_utc': utc(), 'status': 'COMPLETE_BOUNDED_OFFLINE_PILOT',
            'plan_sha256': sha(args.plan), 'freeze_sha256': sha(directory / 'ALL_FORECASTS_FROZEN.json'),
            'actual_campaigns': actual, 'logical_episode_aliases': freeze['logical_episodes'],
            'paired_development_comparisons': comparisons,
            'material_scope': {'ucsb_thermoelectrics': 'preexisting experimental literature values',
                               'jarvis_dft_3d': 'preexisting DFT computed values, not wet measurements'},
            'not_claimed': ['new measurements', 'independent experimental confirmation', 'native-missing truth recovered',
                            'learned causal catalytic gains', 'new relation learning after warm fit',
                            'general cross-domain weight transfer', 'field strong-baseline certification',
                            'physical resource savings', 'novel algorithm or topconference readiness']}
        write(output / 'RESULTS.json', result)
        validate_bindings(plan)
        print({'status': 'SCORED_ALL_FROZEN_OFFLINE_LAWS', 'actual_campaigns': len(actual),
               'logical_episodes_including_aliases': len(freeze['logical_episodes']), 'new_measurements': 0})
    except Exception as error:
        write(output / 'EVALUATOR_FAILED_STOP.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
            'message': str(error), 'test_truth_may_have_been_loaded': True,
            'policy_retuning_or_refit_not_authorized': True})
        raise


if __name__ == '__main__':
    main()
