"""Prepare a metadata plan, or execute root-approved campaigns without test truth.

No score or test outcome is read here. Every physical-label lookup is mediated
by the restricted source oracle. Deterministic episode aliases prevent redundant
refits from being advertised as independent trials.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
from threadpoolctl import threadpool_limits

from benchmark_io import (BUNDLES, CHECKPOINTS, CODE_FILES, DETERMINISTIC, INPUT_FILES, SOURCE_HASHES,
                          ACTUAL_CAMPAIGNS, LOGICAL_EPISODES,
                          POLICIES, SEEDS, BundleOracle, feature_matrix, read, select_design,
                          sha, source_bindings, utc, validate_bindings, write, expected_schedule)
from joint_gaussian import warm_reference
from policies import PolicyState, choose_action, score_actions


def prepare(args):
    directory = Path(args.data_dir).resolve()
    independent = Path(args.role_plan).resolve()
    plan = {'kind': 'source_pinned_classical_catalytic_feasibility_pilot', 'registered_actual_utc': utc(),
            'data_directory': str(directory), 'source_sha256': source_bindings(),
            'input_sha256': {name: sha(directory / name) for name in INPUT_FILES},
            'independent_role_plan_file': str(independent),
            'independent_role_plan_sha256': sha(independent),
            'datasets': select_design(read(directory / 'ELIGIBLE_METADATA.json')),
            'policies': list(POLICIES), 'seeds': list(SEEDS), 'checkpoints': list(CHECKPOINTS),
            'new_bundle_budget': 24, 'bundle_schema': [list(x) for x in BUNDLES],
            'each_bundle_cost': 1, 'cost_is_offline_assumption_not_measured_lab_or_DFT_cost': True,
            'role_counts_groups': {'warm': 8, 'repair': 8, 'frontier': 16, 'test': 8},
            'rows_per_group_cap': 2, 'actual_campaign_count': ACTUAL_CAMPAIGNS, 'logical_episode_count': LOGICAL_EPISODES,
            'deterministic_alias_policy': 'Four deterministic policies executed once per dataset; all four seed names reference identical frozen laws.',
            'primary_endpoint': 'equal_test_metadata_group_mean_standardized_Gaussian_CRPS',
            'relation_learning': 'B fit once from warm responses, thereafter fixed; only conditioning and resource weights adapt.',
            'no_novelty_or_topconference_success_certification': True,
            'no_protected_old_MLST_input': True,
            'expected_stop': 'If predictive hypercycle offers no increment over joint lookahead, retain failure; no parameter/model changes on this consumed test set.',
            'runtime_metadata': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'threadpoolctl')}}
    validate_bindings(plan)
    write(args.output, plan)
    print({'status': 'METADATA_ONLY_PLAN_CREATED_NOT_APPROVED', 'plan_sha256': sha(args.output)})


def save_forecast(path, model, oracle, location, scale):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, mean_standardized=model.mean, covariance_standardized=model.covariance,
                            observed_mask=model.observed, warm_location=location, warm_scale=scale)


def campaign(dataset, design, data_directory, policy, seed, destination, source_oracle):
    start = time.perf_counter()
    destination.mkdir(parents=True, exist_ok=False)
    x, feature_receipt = feature_matrix(dataset, design['rows'])
    oracle = source_oracle.fork()
    warm_values, init = [], []
    for row, meta in enumerate(design['rows']):
        if meta['role'] == 'warm':
            a, b = oracle.query(row, 0, initial=True), oracle.query(row, 1, initial=True)
            init.extend((a, b))
            warm_values.append(a['native_values'] + b['native_values'])
        elif meta['role'] == 'repair':
            init.append(oracle.query(row, 0, initial=True))
    model, location, scale, reference = warm_reference(x, warm_values)
    indices, values = [], []
    for record in init:
        indices.extend(record['flat_indices'])
        values.extend((v - location[p]) / scale[p] for p, v in zip(record['property_indices'], record['native_values']))
    model.condition(indices, values)
    test = [3 * i + p for i, m in enumerate(design['rows']) if m['role'] == 'test' for p in range(3)]
    frontier = [3 * i + 2 for i, m in enumerate(design['rows']) if m['role'] == 'frontier']
    normalizers = {'initial_test_trace': float(np.trace(model.covariance[np.ix_(test, test)])),
                   'initial_frontier_sd': float(np.max(np.sqrt(np.diag(model.covariance)[frontier]))),
                   'utility_sign': design['frontier_utility_sign']}
    write(destination / 'INITIAL_REFERENCE.json', {'actual_utc': utc(), **reference,
            'dataset': dataset, 'policy': policy, 'seed': seed,
            'metadata_scaling': feature_receipt, 'normalizers': normalizers,
            'initial_bundle_cost': oracle.initial_cost, 'initial_revealed_cells': int(model.observed.sum()),
            'target_fields': design['target_fields'], 'no_test_labels_revealed': True})
    uniforms = np.random.default_rng(seed).random(24)
    state = PolicyState()
    snapshots, decisions = [], []
    for step in range(25):
        if step in CHECKPOINTS:
            # Independent withheld test block must remain strictly positive definite.
            np.linalg.cholesky(model.covariance[np.ix_(test, test)])
            name = 'forecast_' + str(step).zfill(2) + '.npz'
            save_forecast(destination / name, model, oracle, location, scale)
            snapshots.append({'step': step, 'file': name, 'sha256': sha(destination / name),
                              'new_bundle_cost': oracle.campaign_cost})
        if step == 24:
            break
        scored = score_actions(model, oracle.available_actions(), test, frontier, normalizers)
        action, decision = choose_action(policy, scored, step, float(uniforms[step]), state)
        observation = oracle.query(action['row'], action['bundle'])
        standardized = [(v - location[p]) / scale[p] for p, v in
                        zip(observation['property_indices'], observation['native_values'])]
        model.condition(observation['flat_indices'], standardized)
        decisions.append({'step': step, 'policy': decision, 'oracle': observation})
    write(destination / 'ACTIONS.json', {'dataset': dataset, 'policy': policy, 'seed': seed,
                    'initial': init, 'campaign': decisions,
                    'all_source_values_retained_even_when_predictions_wrong': True,
                    'initial_bundle_cost': oracle.initial_cost, 'new_bundle_cost': oracle.campaign_cost,
                    'new_revealed_cells': sum(len(x['oracle']['property_indices']) for x in decisions),
                    'oracle_cost_is_declared_offline_bundle_not_actual_physical_cost': True})
    write(destination / 'CAMPAIGN_COMPLETE.json', {'actual_end_utc': utc(), 'dataset': dataset,
            'policy': policy, 'seed': seed, 'snapshots': snapshots,
            'elapsed_seconds': time.perf_counter() - start, 'no_evaluator_or_test_truth_access': True,
            'new_fit_status': 'one_warm_reference_fit_plus_exact_conditioning',
            'predictive_C_is_not_measured_catalysis': True})
    return snapshots


def run(args):
    plan, approval = read(args.plan), read(args.approval)
    validate_bindings(plan)
    if (approval.get('approved') is not True or approval.get('plan_sha256') != sha(args.plan)
            or approval.get('scope') != 'BOUNDED_REAL_SOURCE_OFFLINE_PILOT'):
        raise ValueError('Root exact-plan approval is required before real-source fitting')
    if (plan['policies'] != list(POLICIES) or plan['seeds'] != list(SEEDS)
            or plan['new_bundle_budget'] != 24 or plan['checkpoints'] != list(CHECKPOINTS)):
        raise ValueError('Fixed bounded experiment configuration changed')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(), 'plan_sha256': sha(args.plan),
            'approval_sha256': sha(args.approval), 'source_sha256': source_bindings(),
            'actual_runtime': {'python': sys.version, 'numpy': np.__version__},
            'all_test_truth_hidden_from_campaign_code': True})
    actual, logical = [], []
    try:
        with threadpool_limits(limits=1):
            for dataset in SOURCE_HASHES:
                design = plan['datasets'][dataset]
                if design != select_design(read(Path(plan['data_directory']) / 'ELIGIBLE_METADATA.json'))[dataset]:
                    raise ValueError('Declared metadata roles/rows changed')
                source_oracle = BundleOracle(dataset, plan['data_directory'], design)
                for policy in POLICIES:
                    seeds = SEEDS[:1] if policy in DETERMINISTIC else SEEDS
                    for seed in seeds:
                        name = dataset + '__' + policy + '__' + str(seed)
                        campaign(dataset, design, plan['data_directory'], policy, seed, output / name, source_oracle)
                        actual.append(name)
                    for seed in SEEDS:
                        used_seed = SEEDS[0] if policy in DETERMINISTIC else seed
                        logical.append({'dataset': dataset, 'policy': policy, 'seed': seed,
                            'campaign': dataset + '__' + policy + '__' + str(used_seed),
                            'is_alias': policy in DETERMINISTIC and seed != used_seed})
        validate_bindings(plan)
        expected, expected_logical = expected_schedule()
        if actual != [r['name'] for r in expected] or logical != expected_logical:
            raise ValueError('Bounded campaign/alias counts changed')
        files = [{'file': p.relative_to(output).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(output.rglob('*')) if p.is_file()]
        write(output / 'ALL_FORECASTS_FROZEN.json', {'actual_utc': utc(), 'plan_sha256': sha(args.plan),
            'source_sha256': source_bindings(), 'actual_campaigns': actual, 'logical_episodes': logical,
            'files': files, 'test_truth_used_for_fitting_or_policy': False,
            'new_measurements_or_wet_experiments': 0})
        print({'status': 'ALL_FORECASTS_FROZEN_NOT_SCORED', 'actual_campaigns': ACTUAL_CAMPAIGNS,
               'logical_episodes_including_aliases': LOGICAL_EPISODES, 'freeze_sha256': sha(output / 'ALL_FORECASTS_FROZEN.json')})
    except Exception as error:
        write(output / 'EXECUTION_FAILED.json', {'actual_utc': utc(), 'error_type': type(error).__name__,
            'message': str(error), 'completed_campaigns': actual, 'no_retune_or_successful_subset': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--role-plan', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('run')
    p.add_argument('--plan', required=True)
    p.add_argument('--approval', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args) if args.command == 'prepare' else run(args)


if __name__ == '__main__':
    main()
