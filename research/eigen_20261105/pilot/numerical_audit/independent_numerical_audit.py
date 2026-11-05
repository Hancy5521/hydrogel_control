"""Independent dense-Gaussian and post-freeze scoring audit, no SUT imports.

Forecast mode parses only metadata and already legally queried source receipts.
Scores mode needs separate approval after producer evaluation is frozen. Neither
mode learns parameters, chooses actions, refits a predictor or queries an oracle.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True

DATASETS = ('ucsb_thermoelectrics', 'jarvis_dft_3d')
POLICIES = ('random', 'repair_only_IVR', 'expansion_only_KG', 'fixed_cycle',
            'fixed_random_modules', 'joint_lookahead_all', 'hypercycle_predictive', 'hypercycle_no_cross')
DETERMINISTIC = {'repair_only_IVR', 'expansion_only_KG', 'fixed_cycle', 'joint_lookahead_all'}
SEEDS = (2026110501, 2026110502, 2026110503, 2026110504)
STEPS = (0, 8, 16, 24)
BUNDLES = ((0, 1), (2,))
ELEMENTS = ('H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
            'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd '
            'Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac '
            'Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og').split()
TARGETS = {'ucsb_thermoelectrics': ['rho (ohm.cm)', 'S [muV/K]', 'kappa [W/mK]'],
           'jarvis_dft_3d': ['e_form', 'gap opt', 'bulk modulus']}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for blob in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(blob)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def safe_child(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or '..' in value.parts or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe frozen relative path')
    output = Path(root).joinpath(*value.parts).resolve()
    output.relative_to(Path(root).resolve())
    return output


def expected_schedule():
    actual, logical = [], []
    for dataset in DATASETS:
        for policy in POLICIES:
            for seed in SEEDS[:1] if policy in DETERMINISTIC else SEEDS:
                actual.append({'name': dataset + '__' + policy + '__' + str(seed),
                               'dataset': dataset, 'policy': policy, 'seed': seed})
            for seed in SEEDS:
                used = SEEDS[0] if policy in DETERMINISTIC else seed
                logical.append({'dataset': dataset, 'policy': policy, 'seed': seed,
                                'campaign': dataset + '__' + policy + '__' + str(used),
                                'is_alias': policy in DETERMINISTIC and seed != used})
    assert len(actual) == 40 and len(logical) == 64
    return actual, logical


def checkclose(np, a, b, label, stats, atol, rtol):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('Shape/finiteness: ' + label)
    maximum = float(np.max(np.abs(a - b))) if a.size else 0.
    stats[label] = max(stats.get(label, 0.), maximum)
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(label + ' max absolute error=' + str(maximum))


def features(np, dataset, rows):
    """Independent metadata reconstruction; same declared transductive units."""
    columns = len(ELEMENTS) + int(dataset == 'ucsb_thermoelectrics')
    x = np.zeros((len(rows), columns))
    position = {element: j for j, element in enumerate(ELEMENTS)}
    for i, record in enumerate(rows):
        fractions = {}
        for piece in record['normalized_composition'].split('|'):
            symbol, amount = piece.split(':')
            assert symbol in position and symbol not in fractions
            fractions[symbol] = float(Fraction(amount))
        assert math.isclose(math.fsum(fractions.values()), 1., abs_tol=1e-12)
        for symbol, amount in fractions.items():
            x[i, position[symbol]] = amount
        if dataset == 'ucsb_thermoelectrics':
            x[i, -1] = record['temperature_K'] / 1000.
    center = x.mean(axis=0)
    deviation = x.std(axis=0, ddof=0)
    constant = deviation == 0
    deviation[constant] = 1.
    active = int((~constant).sum())
    rms = math.sqrt(active) if active else 1.
    return (x - center) / deviation / rms, center, deviation, constant, active, rms


def inspect_actions(np, identity, design, actions):
    rows, known, seen = design['rows'], set(), set()
    assert all(actions[k] == identity[k] for k in ('dataset', 'policy', 'seed'))
    initial, campaign = actions['initial'], actions['campaign']
    assert len(campaign) == 24 and actions['new_bundle_cost'] == 24
    required = {(i, bundle) for i, row in enumerate(rows) for bundle in range(2)
                if row['role'] == 'warm'} | {(i, 0) for i, row in enumerate(rows) if row['role'] == 'repair'}
    assert len(initial) == len(required) and actions['initial_bundle_cost'] == len(required)
    records, dynamic = [], defaultdict(int)
    for index, (initial_flag, record) in enumerate([(True, r) for r in initial]
                                                  + [(False, r['oracle']) for r in campaign]):
        i, b = record['row_position'], record['bundle_id']
        assert type(i) is int and 0 <= i < len(rows) and type(b) is int and b in (0, 1)
        meta, props = rows[i], list(BUNDLES[b])
        assert record['property_indices'] == props
        assert record['flat_indices'] == [3 * i + p for p in props]
        assert record['original_raw_row'] == meta['raw_row_zero_based']
        assert record['component_id'] == meta['component_id'] and record['role'] == meta['role']
        assert record['cost'] == 1 and record['sequence'] == index
        assert record['phase'] == ('initial' if initial_flag else 'campaign')
        assert record['value_role'] == 'original_public_label_lookup_not_new_measurement'
        assert meta['role'] != 'test'
        key = meta['raw_row_zero_based'], b
        assert key not in seen
        seen.add(key)
        assert not known.intersection(record['flat_indices'])
        kind = 'R' if any(3 * i + p in known for p in range(3)) else 'E'
        assert record['dynamic_kind_before_reveal'] == kind
        assert len(record['native_values']) == len(props) and np.isfinite(record['native_values']).all()
        if initial_flag:
            assert (i, b) in required
            required.remove((i, b))
        else:
            assert meta['role'] in ('repair', 'frontier')
            decision = campaign[index - len(initial)]
            assert decision['step'] == index - len(initial)
            chosen = decision['policy']['chosen']
            assert chosen['row'] == i and chosen['bundle'] == b and chosen['kind'] == kind
            dynamic[kind] += 1
        known.update(record['flat_indices'])
        records.append(record)
    assert not required
    assert sum(dynamic.values()) == 24
    assert actions['new_revealed_cells'] == sum(len(r['oracle']['property_indices']) for r in campaign)
    return {'initial_bundles': len(initial), 'campaign_bundles': len(campaign),
            'campaign_revealed_cells': actions['new_revealed_cells'],
            'dynamic_repair_actions': dynamic['R'], 'dynamic_expansion_actions': dynamic['E']}


def warm_and_prior(np, dataset, design, actions, reference, errors, tolerance):
    rows = design['rows']
    x, center, deviation, constant, active, rms = features(np, dataset, rows)
    feature = reference['metadata_scaling']
    checkclose(np, center, feature['metadata_only_center'], 'metadata_center', errors, **tolerance['metadata'])
    checkclose(np, deviation, feature['metadata_only_scale'], 'metadata_scale', errors, **tolerance['metadata'])
    assert feature['constant_metadata_columns'] == np.flatnonzero(constant).tolist()
    assert feature['nonconstant_metadata_dimension'] == active
    checkclose(np, [rms], [feature['RMS_feature_distance_scale']], 'metadata_RMS', errors, **tolerance['metadata'])
    warm = []
    lookup = {(r['row_position'], r['bundle_id']): r['native_values'] for r in actions['initial']}
    for i, row in enumerate(rows):
        if row['role'] == 'warm':
            warm.append(lookup[i, 0] + lookup[i, 1])
    y = np.asarray(warm, dtype=float)
    location = np.asarray([math.fsum(column) / len(y) for column in y.T])
    scale = np.asarray([math.sqrt(math.fsum((v - m) ** 2 for v in column) / (len(y) - 1))
                        for column, m in zip(y.T, location)])
    zero = scale == 0
    scale[zero] = 1.
    z = (y - location) / scale
    centered = z - z.mean(axis=0)
    task = .8 * (centered.T @ centered) / (len(y) - 1) + .2 * np.eye(3)
    checkclose(np, location, reference['warm_location'], 'warm_location', errors, **tolerance['metadata'])
    checkclose(np, scale, reference['warm_scale'], 'warm_scale', errors, **tolerance['metadata'])
    checkclose(np, task, reference['task_covariance_B'], 'warm_task_covariance', errors, **tolerance['metadata'])
    assert reference['constant_warm_properties'] == np.flatnonzero(zero).tolist()
    assert reference['independent_response_residual_variance'] == .05
    assert reference['task_relation_frozen_after_warm_fit'] is True
    # Gram-distance form is independent of producer's pairwise feature-difference construction.
    lengths = np.einsum('ij,ij->i', x, x)
    distance = lengths[:, None] + lengths[None, :] - 2 * (x @ x.T)
    kernel = np.exp(-.5 * distance)
    # Explicit row/property indexing rather than a producer model or Kronecker helper.
    index = np.arange(3 * len(rows))
    prior = kernel[index[:, None] // 3, index[None, :] // 3] * task[index[:, None] % 3, index[None, :] % 3]
    prior += .05 * np.eye(len(index))
    np.linalg.cholesky(prior)
    return prior, location, scale


def direct_condition(np, prior, location, scale, records):
    observations = {}
    for record in records:
        for flat, prop, value in zip(record['flat_indices'], record['property_indices'], record['native_values']):
            assert flat not in observations
            observations[flat] = (value - location[prop]) / scale[prop]
    ids = np.asarray(sorted(observations), dtype=int)
    values = np.asarray([observations[i] for i in ids])
    cross = prior[:, ids]
    block = prior[np.ix_(ids, ids)]
    solutions = np.linalg.solve(block, np.column_stack((values, prior[ids, :])))
    mean = cross @ solutions[:, 0]
    covariance = prior - cross @ solutions[:, 1:]
    covariance = (covariance + covariance.T) / 2
    observed = np.zeros(len(prior), dtype=bool)
    observed[ids] = True
    # These are exact noiseless observation identities, not arbitrary variance clipping.
    mean[ids] = values
    covariance[ids, :] = 0.
    covariance[:, ids] = 0.
    return mean, covariance, observed


def original_forecasts(np, campaign, design, plan, errors):
    reference = read(campaign / 'INITIAL_REFERENCE.json')
    actions = read(campaign / 'ACTIONS.json')
    prior, location, scale = warm_and_prior(np, reference['dataset'], design, actions, reference, errors, plan['tolerances'])
    test = [3 * i + p for i, row in enumerate(design['rows']) if row['role'] == 'test' for p in range(3)]
    frontier = [3 * i + 2 for i, row in enumerate(design['rows']) if row['role'] == 'frontier']
    normalizers = None
    for step in STEPS:
        records = actions['initial'] + [r['oracle'] for r in actions['campaign'][:step]]
        mean, covariance, observed = direct_condition(np, prior, location, scale, records)
        unknown = np.flatnonzero(~observed)
        np.linalg.cholesky(covariance[np.ix_(unknown, unknown)])
        np.linalg.cholesky(covariance[np.ix_(test, test)])
        assert not observed[test].any()
        with np.load(campaign / ('forecast_' + str(step).zfill(2) + '.npz'), allow_pickle=False) as saved:
            assert set(saved.files) == {'mean_standardized', 'covariance_standardized', 'observed_mask', 'warm_location', 'warm_scale'}
            checkclose(np, mean, saved['mean_standardized'], 'direct_vs_sequential_mean', errors, **plan['tolerances']['mean'])
            checkclose(np, covariance, saved['covariance_standardized'], 'direct_vs_sequential_covariance', errors, **plan['tolerances']['covariance'])
            assert np.array_equal(observed, saved['observed_mask'])
            assert np.count_nonzero(saved['covariance_standardized'][observed, :]) == 0
            assert np.count_nonzero(saved['covariance_standardized'][:, observed]) == 0
            checkclose(np, location, saved['warm_location'], 'npz_warm_location', errors, **plan['tolerances']['metadata'])
            checkclose(np, scale, saved['warm_scale'], 'npz_warm_scale', errors, **plan['tolerances']['metadata'])
        if step == 0:
            normalizers = {'initial_test_trace': float(np.trace(covariance[np.ix_(test, test)])),
                           'initial_frontier_sd': float(np.sqrt(np.diag(covariance)[frontier]).max())}
            for key, value in normalizers.items():
                checkclose(np, [value], [reference['normalizers'][key]], key, errors, **plan['tolerances']['metadata'])
            assert reference['normalizers']['utility_sign'] == design['frontier_utility_sign']
    return 4


def verify_freeze(plan, scientific, directory):
    freeze_path = directory / 'ALL_FORECASTS_FROZEN.json'
    assert sha(freeze_path) == plan['forecast_freeze_sha256']
    freeze = read(freeze_path)
    actual, logical = expected_schedule()
    assert freeze['actual_campaigns'] == [r['name'] for r in actual]
    assert freeze['logical_episodes'] == logical
    assert freeze['plan_sha256'] == plan['scientific_plan_sha256']
    assert freeze['source_sha256'] == scientific['source_sha256']
    assert not (directory / 'EXECUTION_FAILED.json').exists()
    expected = {'EXECUTION_START.json'} | {r['name'] + '/' + f for r in actual for f in
               ('INITIAL_REFERENCE.json', 'ACTIONS.json', 'CAMPAIGN_COMPLETE.json',
                'forecast_00.npz', 'forecast_08.npz', 'forecast_16.npz', 'forecast_24.npz')}
    assert len(expected) == 281 and {r['file'] for r in freeze['files']} == expected
    assert len(freeze['files']) == len(expected)
    for item in freeze['files']:
        p = safe_child(directory, item['file'])
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
    return actual, logical


def score_one(np, design, truth, campaign, step):
    with np.load(campaign / ('forecast_' + str(step).zfill(2) + '.npz'), allow_pickle=False) as saved:
        mean = saved['mean_standardized'].reshape(-1, 3)
        covariance = saved['covariance_standardized']
        location, scale = saved['warm_location'], saved['warm_scale']
    by_group = defaultdict(list)
    for i, row in enumerate(design['rows']):
        if row['role'] == 'test':
            by_group[row['component_id']].append(i)
    assert len(by_group) == 8
    groups = {}
    for group, indices in by_group.items():
        crps, mse = [], []
        for i in indices:
            for j in range(3):
                y = (truth[i, j] - location[j]) / scale[j]
                mu, sd = float(mean[i, j]), math.sqrt(float(covariance[3 * i + j, 3 * i + j]))
                z = (y - mu) / sd
                phi = math.exp(-.5 * z * z) / math.sqrt(2 * math.pi)
                # 2*Phi(z)-1 = erf(z/sqrt(2)), independent of producer scipy.ndtr.
                crps.append(sd * (z * math.erf(z / math.sqrt(2)) + 2 * phi - 1 / math.sqrt(math.pi)))
                mse.append((mu - y) ** 2)
        groups[group] = {'CRPS': math.fsum(crps) / len(crps), 'MSE': math.fsum(mse) / len(mse), 'native_rows': len(indices)}
    actions = read(campaign / 'ACTIONS.json')
    selected = [x['oracle'] for x in actions['campaign'][:step]]
    checked_cells = 0
    for record in actions['initial'] + selected:
        for prop, value in zip(record['property_indices'], record['native_values']):
            assert truth[record['row_position'], prop] == value
            checked_cells += 1
    sign = design['frontier_utility_sign']
    frontier = defaultdict(list)
    for i, row in enumerate(design['rows']):
        if row['role'] == 'frontier':
            frontier[row['component_id']].append(float(sign * truth[i, 2]))
    maxima = sorted(max(values) for values in frontier.values())
    assert len(maxima) == 16
    position = .9 * (len(maxima) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    threshold = maxima[lo] + (position - lo) * (maxima[hi] - maxima[lo])
    verified = {}
    for record in selected:
        row = design['rows'][record['row_position']]
        if row['role'] == 'frontier' and 2 in record['property_indices']:
            value = float(sign * record['native_values'][record['property_indices'].index(2)])
            verified[row['component_id']] = max(verified.get(row['component_id'], -math.inf), value)
    best = max(verified.values()) if verified else None
    return {'checkpoint_new_bundle_cost': step,
            'equal_group_standardized_CRPS': math.fsum(x['CRPS'] for x in groups.values()) / len(groups),
            'equal_group_standardized_RMSE': math.sqrt(math.fsum(x['MSE'] for x in groups.values()) / len(groups)),
            'per_test_group': groups, 'test_metadata_groups': len(groups),
            'test_rows': sum(len(x) for x in by_group.values()),
            'new_revealed_cells': sum(len(x['property_indices']) for x in selected),
            'initial_bundle_cost': len(actions['initial']), 'native_source_cells_checked': checked_cells,
            'frontier_retrieval': {'utility_sign': sign, 'top10pct_group_utility_threshold_evaluator_only': threshold,
                                  'verified_frontier_groups_with_target': len(verified),
                                  'verified_top_decile_group_hits': sum(x >= threshold for x in verified.values()),
                                  'best_verified_utility': best,
                                  'simple_regret_to_known_offline_pool': None if best is None else maxima[-1] - best}}


def compare_nested(np, left, right, errors, tolerance, prefix='score'):
    for key, value in left.items():
        other = right[key]
        label = prefix + '/' + key
        if isinstance(value, dict):
            compare_nested(np, value, other, errors, tolerance, label)
        elif value is None:
            assert other is None, label
        elif type(value) is int:
            assert value == other, label
        elif type(value) is float:
            checkclose(np, [value], [other], label, errors, **tolerance)
        else:
            assert value == other, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--phase', choices=('forecasts', 'scores'), required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    assert plan['phase'] == args.phase
    assert plan['audit_script_sha256'] == sha(__file__)
    assert approval['approved'] is True and approval['audit_plan_sha256'] == sha(args.plan)
    assert approval['phase'] == args.phase
    assert approval['scope'] == 'INDEPENDENT_OFFLINE_NUMERICAL_AUDIT_NO_REFIT_OR_POLICY_RUN'
    scientific = read(plan['scientific_plan_file'])
    assert sha(plan['scientific_plan_file']) == plan['scientific_plan_sha256']
    for name, digest in scientific['source_sha256'].items():
        assert sha(Path(plan['source_directory']) / name) == digest
    directory = Path(plan['campaign_directory'])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'AUDIT_EXECUTION_START.json', {'actual_utc': utc(), 'phase': args.phase,
          'audit_plan_sha256': sha(args.plan), 'approval_sha256': sha(args.approval),
          'no_SUT_import_refit_or_policy_execution': True,
          'test_truth_loading_allowed_this_phase': args.phase == 'scores'})
    errors, findings = {}, []
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        actual, logical = verify_freeze(plan, scientific, directory)
        if args.phase == 'forecasts':
            law_count, resources = 0, []
            with threadpool_limits(limits=1):
                for identity in actual:
                    folder = directory / identity['name']
                    design = scientific['datasets'][identity['dataset']]
                    actions = read(folder / 'ACTIONS.json')
                    resource = inspect_actions(np, identity, design, actions)
                    law_count += original_forecasts(np, folder, design, plan, errors)
                    resources.append(dict(identity, **resource))
            assert law_count == 160
            findings = {'directly_conditioned_checkpoint_laws': law_count, 'actual_campaigns': len(actual),
                        'logical_episode_names': len(logical), 'deterministic_alias_count': sum(x['is_alias'] for x in logical),
                        'resources': resources, 'test_response_arrays_loaded': False}
        else:
            assert sha(plan['forecast_audit_result_file']) == plan['forecast_audit_result_sha256']
            passed = read(plan['forecast_audit_result_file'])
            assert passed['status'] == 'PASS' and passed['phase'] == 'forecasts'
            assert passed['scientific_plan_sha256'] == plan['scientific_plan_sha256']
            assert passed['forecast_freeze_sha256'] == plan['forecast_freeze_sha256']
            assert sha(plan['score_result_file']) == plan['score_result_sha256']
            result = read(plan['score_result_file'])
            assert result['plan_sha256'] == plan['scientific_plan_sha256']
            assert result['freeze_sha256'] == plan['forecast_freeze_sha256']
            assert result['logical_episode_aliases'] == logical
            assert set(result['actual_campaigns']) == {x['name'] for x in actual}
            write(output / 'AUDITOR_TEST_TRUTH_REVEAL_STARTED.json', {'actual_utc': utc(),
                  'score_result_sha256': plan['score_result_sha256'], 'forecast_freeze_sha256': plan['forecast_freeze_sha256'],
                  'meaning': 'Audit original source labels only after producer forecasts/scores and root permissions are frozen.'})
            truth = {}
            for dataset, design in scientific['datasets'].items():
                source = Path(scientific['data_directory']) / (dataset + '.json.gz')
                assert sha(source) == scientific['input_sha256'][dataset + '.json.gz']
                with gzip.open(source, 'rt', encoding='utf-8') as stream:
                    payload = json.load(stream)
                columns = {x: i for i, x in enumerate(payload['columns'])}
                truth[dataset] = np.asarray([[payload['data'][r['raw_row_zero_based']][columns[key]]
                                             for key in TARGETS[dataset]] for r in design['rows']], dtype=float)
                assert np.isfinite(truth[dataset]).all()
            rescored = {}
            for identity in actual:
                name, dataset = identity['name'], identity['dataset']
                design = scientific['datasets'][dataset]
                rows = [score_one(np, design, truth[dataset], directory / name, step) for step in STEPS]
                saved = result['actual_campaigns'][name]
                assert all(saved[k] == identity[k] for k in ('dataset', 'policy', 'seed'))
                assert len(saved['scores']) == 4
                for independently, existing in zip(rows, saved['scores']):
                    compare_nested(np, independently, existing, errors, plan['tolerances']['scores'])
                rescored[name] = {row['checkpoint_new_bundle_cost']: row for row in rows}
            differences = result['paired_development_comparisons']
            assert len(differences) == 2 * 8 * 4
            for record in differences:
                deltas = []
                for seed in SEEDS:
                    left = next(x['campaign'] for x in logical if (x['dataset'], x['policy'], x['seed']) == (record['dataset'], record['policy'], seed))
                    right = next(x['campaign'] for x in logical if (x['dataset'], x['policy'], x['seed']) == (record['dataset'], 'joint_lookahead_all', seed))
                    deltas.append(rescored[left][record['checkpoint']]['equal_group_standardized_CRPS'] - rescored[right][record['checkpoint']]['equal_group_standardized_CRPS'])
                checkclose(np, deltas, record['CRPS_minus_joint_lookahead_by_logical_seed'], 'paired_logical_seed_differences', errors, **plan['tolerances']['scores'])
                checkclose(np, [math.fsum(deltas) / len(deltas)], [record['mean_CRPS_difference']], 'paired_mean_difference', errors, **plan['tolerances']['scores'])
            findings = {'actual_campaigns_scored': 40, 'frozen_checkpoint_scores': 160,
                        'paired_dataset_policy_checkpoint_rows': 64,
                        'equal_test_groups_per_dataset': 8, 'truth_arrays_loaded_only_after_score_approval': True,
                        'native_missing_values_scored_as_truth': False,
                        'label_type': 'Preexisting source experimental/DFT labels, not new measurements.'}
        assert sha(Path(plan['campaign_directory']) / 'ALL_FORECASTS_FROZEN.json') == plan['forecast_freeze_sha256']
        for name, digest in scientific['source_sha256'].items():
            assert sha(Path(plan['source_directory']) / name) == digest
        report = {'actual_end_utc': utc(), 'status': 'PASS', 'phase': args.phase,
                  'audit_plan_sha256': sha(args.plan), 'scientific_plan_sha256': plan['scientific_plan_sha256'],
                  'forecast_freeze_sha256': plan['forecast_freeze_sha256'],
                  'max_numeric_absolute_errors_by_quantity': errors, 'findings': findings,
                  'no_SUT_import_refit_or_policy_rerun': True, 'source_and_forecast_freeze_unchanged': True,
                  'scope': 'Numerical/identity/resource correctness of a fixed offline pilot. Not independent material efficacy, new measurements, learned causal catalysis or top-conference readiness.'}
        write(output / 'INDEPENDENT_NUMERICAL_RESULT.json', report)
        print(json.dumps({'status': report['status'], 'phase': args.phase, 'findings': findings if args.phase == 'scores' else {k: v for k, v in findings.items() if k != 'resources'}}))
    except Exception as error:
        write(output / 'AUDIT_FAILED_STOP.json', {'actual_utc': utc(), 'phase': args.phase,
              'error_type': type(error).__name__, 'error': str(error), 'partial_max_numeric_errors': errors,
              'test_truth_may_have_been_loaded': args.phase == 'scores',
              'correction_requires_new_plan_or_approval': True})
        raise


if __name__ == '__main__':
    main()
