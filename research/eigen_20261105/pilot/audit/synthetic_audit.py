"""Source-pinned independent synthetic audit. Requires exact root approval.

Reference calculations below are authored independently of the producer. The
source-pinned adapters and audit entry point match the final eight-policy API.
This script never needs or reads the two downloaded material data tables.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def gaussian_cdf(z):
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def gaussian_pdf(z):
    return math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)


def direct_dense_condition(mean, covariance, observed, values):
    """Independent one-shot Schur complement, exact noiseless observations.

    Whole conditioned covariance is PSD, not generally SPD. Only a fresh
    unobserved bundle must have positive definite predictive covariance.
    """
    import numpy as np
    m = np.asarray(mean, dtype=float)
    k = np.asarray(covariance, dtype=float)
    idx = np.asarray(observed, dtype=int)
    y = np.asarray(values, dtype=float)
    if len(idx) == 0:
        return m.copy(), k.copy()
    if len(set(map(int, idx))) != len(idx):
        raise ValueError('Independent oracle rejects repeated observation indices')
    # Deliberately use one dense inverse, unlike the producer's sequential path.
    inverse = np.linalg.inv(k[np.ix_(idx, idx)])
    cross = k[:, idx]
    updated_mean = m + cross @ inverse @ (y - m[idx])
    updated_covariance = k - cross @ inverse @ cross.T
    updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
    # The exact conditional law is a point mass on the observed coordinates.
    updated_mean[idx] = y
    updated_covariance[idx, :] = 0.0
    updated_covariance[:, idx] = 0.0
    return updated_mean, updated_covariance


def direct_weighted_trace_drop(covariance, observed, targets, weights):
    import numpy as np
    n = len(covariance)
    _, after = direct_dense_condition(np.zeros(n), covariance, observed,
                                      np.zeros(len(observed)))
    before = np.asarray(covariance, dtype=float)
    idx = np.asarray(targets, dtype=int)
    w = np.asarray(weights, dtype=float)
    return float(np.dot(w, np.diag(before)[idx] - np.diag(after)[idx]))


def exact_expected_max_affine_normal(intercepts, slopes):
    """Integrate the finite upper envelope of a_i+b_i*Z, Z~N(0,1).

    Every active interval is found from pairwise linear inequalities. No GH
    rule, producer envelope implementation, random samples or model are used.
    """
    if len(intercepts) != len(slopes):
        raise ValueError('Aligned lines required')
    lines = list(dict.fromkeys((float(a), float(b))
                              for a, b in zip(intercepts, slopes)))
    if not lines or not all(math.isfinite(a) and math.isfinite(b) for a, b in lines):
        raise ValueError('Finite aligned nonempty lines required')
    total = 0.0
    for a, b in lines:
        low, high = -math.inf, math.inf
        feasible = True
        for aa, bb in lines:
            gap = b - bb
            if gap == 0.0:
                if a < aa:
                    feasible = False
                    break
            elif gap > 0.0:
                low = max(low, (aa - a) / gap)
            else:
                high = min(high, (aa - a) / gap)
        if not feasible or low >= high:
            continue
        mass = gaussian_cdf(high) - gaussian_cdf(low)
        first_moment = gaussian_pdf(low) - gaussian_pdf(high)
        total += a * mass + b * first_moment
    return total


def independent_tensor_gh(intercepts, coefficients, order=5):
    """Independent rebuild of the stated finite GH approximation.

    Matching this value tests implementation, not accuracy of the quadrature as
    an approximation to the continuous Gaussian expectation.
    """
    import itertools
    import numpy as np
    a = np.asarray(intercepts, dtype=float)
    b = np.asarray(coefficients, dtype=float)
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    total = 0.0
    for indices in itertools.product(range(order), repeat=b.shape[1]):
        z = math.sqrt(2.0) * nodes[list(indices)]
        weight = float(np.prod(weights[list(indices)])) / math.pi ** (b.shape[1] / 2.0)
        total += weight * float(np.max(a + b @ z))
    return total


def independent_two_dimensional_expectation(intercepts, coefficients):
    """Nested independent integration for a two-dimensional affine maximum.

    The inner one-dimensional upper envelope is integrated analytically; the
    outer integral is adaptive scipy.quad. Only an approximation diagnostic.
    """
    import numpy as np
    from scipy.integrate import quad
    a = np.asarray(intercepts, dtype=float)
    b = np.asarray(coefficients, dtype=float)
    if b.shape != (len(a), 2):
        raise ValueError('Two coefficients per decision line required')
    value, estimated_error = quad(
        lambda z: gaussian_pdf(z) * exact_expected_max_affine_normal(a + b[:, 0] * z,
                                                                    b[:, 1]),
        -math.inf, math.inf, epsabs=2e-9, epsrel=2e-9, limit=150)
    return value, estimated_error


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def rejects(function):
    try:
        function()
    except (ValueError, ArithmeticError, TypeError, KeyError, OSError):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--approval', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan, approval = read(args.plan), read(args.approval)
    if not (approval.get('approved') is True
            and approval.get('scope') == 'INDEPENDENT_SYNTHETIC_PILOT_AUDIT'
            and approval.get('plan_sha256') == digest(args.plan)):
        raise ValueError('Root exact-plan synthetic approval is required')
    source = Path(plan['source_directory'])
    before = {name: digest(source / name) for name in plan['source_sha256']}
    if before != plan['source_sha256'] or digest(__file__) != plan['audit_script_sha256']:
        raise ValueError('Registered producer/audit bytes changed before execution')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'EXECUTION_START.json', {'actual_utc': utc(),
            'plan_sha256': digest(args.plan), 'approval_sha256': digest(args.approval),
            'source_sha256': before, 'material_data_read': False,
            'scope': 'Artificial Gaussian/feature/oracle/freeze/score fixtures only'})
    sys.path.insert(0, str(source))
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import ndtr
    gaussian = importlib.import_module('joint_gaussian')
    policies = importlib.import_module('policies')
    io = importlib.import_module('benchmark_io')
    evaluator = importlib.import_module('evaluate_campaign')
    runner = importlib.import_module('run_campaign')
    results = []

    def close(actual, expected, name, tolerance=3e-10):
        actual, expected = np.asarray(actual), np.asarray(expected)
        error = float(np.max(np.abs(actual - expected))) if actual.size else 0.
        if not np.allclose(actual, expected, rtol=tolerance, atol=tolerance):
            raise AssertionError((name, error))
        return error

    def check(name, function):
        entry = {'family': name, 'actual_start_utc': utc()}
        try:
            entry.update(status='PASS', details=function())
        except Exception as error:
            entry.update(status='FAIL', error_type=type(error).__name__, message=str(error))
        results.append(entry)
        with (output / 'FAMILIES.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + '\n')

    factor = np.asarray([[1., .2, -.3], [.4, .9, .1], [.6, -.2, 1.],
                         [1., .4, .3], [-.1, .7, .8], [.5, .6, -.2],
                         [.3, -.4, .9], [.8, .1, .5], [.2, .9, .4]])
    prior = factor @ factor.T + np.diag(np.linspace(.3, .9, 9))
    mean = np.asarray([.2, -.5, .1, .4, -.2, .3, .6, -.1, .8])
    revealed = {0: .7, 2: -.2, 4: .4, 5: .9}

    def conditioning():
        errors = []
        for order in ([[0], [4, 5], [2]], [[2], [5, 4], [0]]):
            state = gaussian.GaussianState(mean, prior)
            observed = []
            for block in order:
                observed.extend(block)
                state.condition(block, [revealed[i] for i in block])
                mu, cov = direct_dense_condition(mean, prior, observed,
                                                 [revealed[i] for i in observed])
                errors += [close(state.mean, mu, 'conditional mean'),
                           close(state.covariance, cov, 'conditional covariance')]
                assert np.array_equal(state.mean[observed], [revealed[i] for i in observed])
                assert np.array_equal(state.covariance[observed, :], np.zeros((len(observed), 9)))
                assert np.linalg.eigvalsh(state.covariance).min() >= -3e-10
                unknown = np.flatnonzero(~state.observed)
                np.linalg.cholesky(state.covariance[np.ix_(unknown, unknown)])
            assert rejects(lambda: state.condition([0], [.8]))
            assert rejects(lambda: state.condition([1, 1], [.2, .3]))
            assert rejects(lambda: state.condition([True], [.2]))
        return {'sequential_paths': 2, 'intermediate_comparisons': 6,
                'max_absolute_error': max(errors), 'whole_covariance_PSD_not_full_SPD': True}

    def features_and_warm_icm():
        rows = [
            {'normalized_composition': 'Na:1/2|Cl:1/2', 'temperature_K': 300},
            {'normalized_composition': 'K:1/2|Cl:1/2', 'temperature_K': 400},
            {'normalized_composition': 'Li:1/2|Cl:1/2', 'temperature_K': 600},
            {'normalized_composition': 'Na:1/4|K:1/4|Cl:1/2', 'temperature_K': 450},
        ]
        x, metadata = io.feature_matrix('ucsb_thermoelectrics', rows)
        raw = np.zeros((4, 119))
        raw[:, 16] = .5
        raw[0, 10], raw[1, 18], raw[2, 2] = .5, .5, .5
        raw[3, 10], raw[3, 18] = .25, .25
        raw[:, -1] = [.3, .4, .6, .45]
        center = raw.sum(axis=0) / len(raw)
        variance = ((raw - center) ** 2).sum(axis=0) / len(raw)
        nonconstant = variance > 0
        scales = np.ones(119)
        scales[nonconstant] = np.sqrt(variance[nonconstant])
        expected_x = (raw - center) / scales / math.sqrt(int(nonconstant.sum()))
        err = close(x, expected_x, 'independent metadata RMS geometry')
        assert metadata['nonconstant_metadata_dimension'] == 4
        assert metadata['RMS_feature_distance_scale'] == 2.
        poisoned = copy.deepcopy(rows)
        for row in poisoned:
            row.update({'rho (ohm.cm)': -9e20, 'S [muV/K]': 8e12,
                        'kappa [W/mK]': -4e15, 'sigma': 1e90, 'PF': 1e80,
                        'zT': -1e90, 'source': 'SYNTHETIC_OUTCOME_ALIAS'})
        x2, _ = io.feature_matrix('ucsb_thermoelectrics', poisoned)
        assert np.array_equal(x, x2)
        constant, detail = io.feature_matrix('ucsb_thermoelectrics', [rows[0], rows[0]])
        assert np.array_equal(constant, np.zeros_like(constant))
        assert detail['nonconstant_metadata_dimension'] == 0 and detail['RMS_feature_distance_scale'] == 1.
        y = np.asarray([[1., 5., 2.], [2., 5., 4.], [4., 5., 3.], [3., 5., 9.]])
        state, loc, scale, detail = gaussian.warm_reference(x, y)
        mu = y.sum(axis=0) / len(y)
        sy = np.sqrt(((y - mu) ** 2).sum(axis=0) / (len(y) - 1))
        sy[sy == 0] = 1.
        z = (y - mu) / sy
        b = .8 * sum(np.outer(row, row) for row in z) / (len(y) - 1) + .2 * np.eye(3)
        kx = np.asarray([[math.exp(-.5 * np.dot(a - b0, a - b0)) for b0 in x] for a in x])
        expected_cov = np.kron(kx, b) + .05 * np.eye(12)
        err = max(err, close(loc, mu, 'warm location'), close(scale, sy, 'warm scale'),
                  close(state.covariance, expected_cov, 'full ICM covariance'))
        assert detail['constant_warm_properties'] == [1]
        assert detail['task_relation_frozen_after_warm_fit'] is True
        return {'max_absolute_error': err, 'derived_response_feature_poison_invariant': True,
                'synthetic_warm_fit_calls': 1, 'real_material_fit_calls': 0}

    def acquisition():
        state = gaussian.GaussianState(mean, prior)
        state.condition([0], [.7])
        test, frontier = [6, 7, 8], [2, 5, 8]
        trace = float(np.trace(state.covariance[np.ix_(test, test)]))
        sd = float(np.sqrt(np.diag(state.covariance)[frontier]).max())
        reports = []
        for bundle in ([1], [3, 4]):
            chol = np.linalg.cholesky(state.covariance[np.ix_(bundle, bundle)])
            coefficient = state.covariance[np.ix_(frontier, bundle)] @ np.linalg.inv(chol.T)
            ivr = direct_weighted_trace_drop(state.covariance, bundle, test, np.ones(3)) / trace
            for sign in (-1, 1):
                actual = state.acquisition_gains(bundle, test, frontier, trace, sd, utility_sign=sign)
                a, b = sign * state.mean[frontier], sign * coefficient
                gh = (independent_tensor_gh(a, b, order=5) - max(a)) / sd
                error = max(close(actual['IVR'], ivr, 'IVR trace'),
                            close(actual['KG'], gh, 'same five-node GH scalar KG'),
                            close(actual['joint'], .5 * (ivr + gh), 'fixed scalarized utility'))
                assert actual['IVR'] >= -1e-12 and actual['KG'] >= -1e-12
                if len(bundle) == 1:
                    exact = exact_expected_max_affine_normal(a, b[:, 0])
                    integral_error = 0.
                else:
                    exact, integral_error = independent_two_dimensional_expectation(a, b)
                exact_kg = (exact - max(a)) / sd
                assert exact_kg >= -3e-9
                reports.append({'bundle_size': len(bundle), 'utility_sign': sign,
                                'implementation_error': error, 'fixed_GH_KG': actual['KG'],
                                'independent_continuous_Gaussian_KG': exact_kg,
                                'GH_approximation_absolute_error_diagnostic': abs(actual['KG'] - exact_kg),
                                'outer_integral_error_estimate': integral_error})
        assert rejects(lambda: state.acquisition_gains([0], test, frontier, trace, sd))
        assert rejects(lambda: state.acquisition_gains([1], test, frontier, 0., sd))
        assert rejects(lambda: state.acquisition_gains([1], test, frontier, trace, sd, utility_sign=0))
        return {'cases': reports, 'approximation_error_is_not_implementation_failure': True}

    def resource_policy():
        w, c = np.asarray([.35, .65]), np.asarray([[.2, .6], [.5, .3]])
        out = []
        for remove in (False, True):
            actual, record = policies.catalytic_update(w, c, remove_cross=remove)
            used = c * np.eye(2) if remove else c
            f = np.sum(used * w[None, :], axis=1)
            value = np.asarray([w[i] * math.exp(.5 * f[i]) for i in range(2)])
            expected = .9 * value / value.sum() + .05
            error = close(actual, expected, 'resource weights')
            close(record['C_used'], used, 'offdiagonal ablation only')
            assert np.array_equal(np.diag(record['C_used']), np.diag(c))
            assert actual.min() >= .05 and abs(actual.sum() - 1.) <= 1e-14
            assert record['independently_measured_causal_gain'] is False
            out.append({'remove_cross': remove, 'weights': actual.tolist(), 'error': error})
        assert out[0]['weights'] != out[1]['weights']
        scored = [
            {'row': 0, 'bundle': 1, 'kind': 'R', 'indices': [2], 'cost': 1, 'IVR': .9, 'KG': .1, 'joint': .5},
            {'row': 1, 'bundle': 1, 'kind': 'R', 'indices': [5], 'cost': 1, 'IVR': .2, 'KG': .4, 'joint': .3},
            {'row': 2, 'bundle': 0, 'kind': 'E', 'indices': [6, 7], 'cost': 1, 'IVR': .4, 'KG': .9, 'joint': .65}]
        for name, step, uniform, row in [('repair_only_IVR', 0, .3, 0), ('expansion_only_KG', 0, .3, 2),
                                        ('joint_lookahead_all', 0, .3, 2), ('fixed_cycle', 0, .3, 0),
                                        ('fixed_cycle', 1, .3, 2), ('random', 0, .7, 2)]:
            action, _ = policies.choose_action(name, scored, step, uniform, policies.PolicyState())
            assert action['row'] == row
        for name in ('hypercycle_predictive', 'hypercycle_no_cross'):
            action, receipt = policies.choose_action(name, scored, 0, 0., policies.PolicyState())
            assert action['kind'] == 'R'
            expected_c = [[.9, .1], [.4, .9]]
            close(receipt['C_predictive'], expected_c, 'predictive gain matrix')
        for uniform, kind in ((0., 'R'), (.999, 'E')):
            state = policies.PolicyState()
            action, receipt = policies.choose_action('fixed_random_modules', scored, 0, uniform, state)
            assert action['kind'] == kind and receipt['adaptive_C_update'] is False
            assert receipt['fixed_module_weights'] == [.5, .5]
            assert np.array_equal(state.weights, [.5, .5])
        assert rejects(lambda: policies.catalytic_update([.2, .2], c))
        return {'classical_weight_formula_and_ablation': out, 'fixed_and_stochastic_role_rules_checked': True}

    def oracle_roles_costs():
        directory = output / 'SYNTHETIC_ORACLE'
        directory.mkdir()
        dataset = 'ucsb_thermoelectrics'
        columns = io.TARGETS[dataset] + ['sigma', 'PF', 'zT']
        payload = {'columns': columns, 'data': [[1. + i, 2. + i, 3. + i, 1e40, -1e40, 9e90] for i in range(5)]}
        path = directory / (dataset + '.json.gz')
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(payload, stream)
        rows = [{'role': role, 'raw_row_zero_based': i, 'component_id': 'SYNTHETIC_GROUP_' + str(i)}
                for i, role in enumerate(('warm', 'repair', 'frontier', 'frontier', 'test'))]
        design = {'target_fields': io.TARGETS[dataset], 'rows': rows}
        # Unit-test only: bypass the fixed real-source digest in memory to allow a
        # wholly artificial gzip fixture. No producer file or real table is read.
        with patch.object(io, 'sha', return_value=io.SOURCE_HASHES[dataset]):
            oracle = io.BundleOracle(dataset, directory, design)
        assert np.isnan(oracle._responses[4]).all()
        assert np.array_equal(oracle._responses[:4], np.asarray(payload['data'])[:4, :3])
        oracle.query(0, 0, initial=True)
        oracle.query(0, 1, initial=True)
        oracle.query(1, 0, initial=True)
        assert oracle.initial_cost == 3 and oracle.campaign_cost == 0
        repaired = oracle.query(1, 1)
        expanded = oracle.query(2, 0)
        completed = oracle.query(2, 1)
        assert repaired['dynamic_kind_before_reveal'] == 'R'
        assert expanded['dynamic_kind_before_reveal'] == 'E' and len(expanded['native_values']) == 2
        assert completed['dynamic_kind_before_reveal'] == 'R'
        assert oracle.campaign_cost == 3 and len(oracle.receipts) == 6
        fork = oracle.fork()
        assert fork.campaign_cost == 0 and fork.initial_cost == 0 and not fork.receipts
        assert not fork.observed.any() and np.isnan(fork._responses[4]).all()
        assert all(x['cost'] == 1 and 'not_new_measurement' in x['value_role'] for x in oracle.receipts)
        assert rejects(lambda: oracle.query(4, 0))
        assert rejects(lambda: oracle.query(2, 0))
        assert rejects(lambda: oracle.query(0, 0))
        assert not any(a['row'] == 4 for a in oracle.available_actions())
        oracle.campaign_cost = 24  # Artificial boundary fixture only.
        assert rejects(lambda: oracle.query(3, 0))
        with patch.object(io, 'sha', return_value='0' * 64):
            assert rejects(lambda: io.BundleOracle(dataset, directory, design))
        assert rejects(lambda: io.evaluator_truth(dataset, directory, design, forecasts_verified=False))
        return {'source_digest_mock_only_for_artificial_fixture': True, 'test_view_is_NaN': True,
                'one_and_two_cell_bundle_equal_cost': True, 'repeat_role_and_budget_guards': True,
                'real_source_file_read': False}

    def metadata_and_aliases():
        metadata = {'datasets': {}}
        for dataset in io.SOURCE_HASHES:
            metadata['datasets'][dataset] = [
                {'component_id': 'SYNTHETIC_GROUP_' + str(i), 'raw_row_zero_based': 3 * i + j,
                 'normalized_composition': 'Na:1/2|Cl:1/2', 'temperature_K': 300.}
                for i in range(45) for j in range(3)]
        selected = io.select_design(metadata)
        actual, logical = [], []
        for dataset, design in selected.items():
            by_group = {}
            for row in design['rows']:
                by_group.setdefault(row['component_id'], []).append(row)
            assert len(by_group) == 40 and all(len(v) == 2 for v in by_group.values())
            assert all(len({row['role'] for row in v}) == 1 for v in by_group.values())
            counts = {role: sum(v[0]['role'] == role for v in by_group.values()) for role in io.ROLES}
            assert counts == dict(zip(io.ROLES, io.ROLE_COUNTS))
            assert design['frontier_utility_sign'] == (-1 if dataset == 'ucsb_thermoelectrics' else 1)
            for policy in io.POLICIES:
                for seed in (io.SEEDS[:1] if policy in io.DETERMINISTIC else io.SEEDS):
                    actual.append(dataset + '__' + policy + '__' + str(seed))
                for seed in io.SEEDS:
                    used = io.SEEDS[0] if policy in io.DETERMINISTIC else seed
                    logical.append((dataset, policy, seed, dataset + '__' + policy + '__' + str(used)))
        assert len(set(actual)) == 40 and len(logical) == 64
        assert len({x[:3] for x in logical}) == 64 and all(x[3] in actual for x in logical)
        bad = copy.deepcopy(metadata)
        for rows in bad['datasets'].values():
            for row in rows:
                row['raw_row_zero_based'] = 0
        assert rejects(lambda: io.select_design(bad))
        tree = ast.parse((source / 'run_campaign.py').read_text('utf-8'))
        called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        assert 'evaluator_truth' not in called and 'score_snapshot' not in called
        assert io.BUNDLES == ((0, 1), (2,))
        return {'role_groups_per_dataset': dict(zip(io.ROLES, io.ROLE_COUNTS)),
                'actual_paths': 40, 'logical_alias_slots': 64,
                'trusted_IO_not_malicious_python_sandbox': True,
                'transductive_unlabelled_metadata_declared': True}

    def complete_freeze_guards():
        directory = output / 'SYNTHETIC_FREEZE'
        directory.mkdir()
        plan_stub = {'datasets': {dataset: {} for dataset in io.SOURCE_HASHES}}
        plan_file = directory / 'PLAN.json'
        write(plan_file, plan_stub)
        names, logical = [], []
        for dataset in io.SOURCE_HASHES:
            for policy in io.POLICIES:
                for seed in (io.SEEDS[:1] if policy in io.DETERMINISTIC else io.SEEDS):
                    names.append(dataset + '__' + policy + '__' + str(seed))
                for seed in io.SEEDS:
                    used = io.SEEDS[0] if policy in io.DETERMINISTIC else seed
                    logical.append({'dataset': dataset, 'policy': policy, 'seed': seed,
                        'campaign': dataset + '__' + policy + '__' + str(used),
                        'is_alias': policy in io.DETERMINISTIC and seed != used})
        bound = []
        start_file = directory / 'EXECUTION_START.json'
        write(start_file, {'is_synthetic': True, 'plan_sha256': digest(plan_file),
                           'source_sha256': io.source_bindings()})
        bound.append({'file': start_file.name, 'bytes': start_file.stat().st_size,
                      'sha256': digest(start_file)})
        for name in names:
            folder = directory / name
            folder.mkdir()
            dataset, policy, seed_text = name.split('__')
            identity = {'dataset': dataset, 'policy': policy, 'seed': int(seed_text)}
            snapshots = []
            for step in io.CHECKPOINTS:
                p = folder / ('forecast_' + str(step).zfill(2) + '.npz')
                with p.open('xb') as stream:
                    np.savez_compressed(stream, artificial_not_a_material_forecast=np.asarray([step]))
                snapshots.append({'step': step, 'file': p.name, 'sha256': digest(p),
                                  'new_bundle_cost': step})
            for stem in ('INITIAL_REFERENCE', 'ACTIONS', 'CAMPAIGN_COMPLETE'):
                receipt = {'is_synthetic': True, **identity}
                if stem == 'CAMPAIGN_COMPLETE':
                    receipt.update(snapshots=snapshots, no_evaluator_or_test_truth_access=True)
                write(folder / (stem + '.json'), receipt)
            for p in sorted(folder.iterdir()):
                bound.append({'file': p.relative_to(directory).as_posix(), 'bytes': p.stat().st_size,
                              'sha256': digest(p)})
        record = {'plan_sha256': digest(plan_file), 'source_sha256': io.source_bindings(),
                  'actual_campaigns': names, 'logical_episodes': logical, 'files': bound}
        frozen = directory / 'ALL_FORECASTS_FROZEN.json'
        write(frozen, record)
        evaluator.verify_freeze(directory, plan_file, plan_stub)
        mutations = []
        bad = copy.deepcopy(record)
        missing = next(i for i, row in enumerate(bad['files']) if row['file'].endswith('/forecast_00.npz'))
        bad['files'].pop(missing)
        mutations.append(('missing_bound_forecast', bad))
        bad = copy.deepcopy(record); bad['actual_campaigns'][1] = bad['actual_campaigns'][0]; mutations.append(('duplicate_actual_name', bad))
        bad = copy.deepcopy(record); bad['logical_episodes'][1] = bad['logical_episodes'][0]; mutations.append(('duplicate_alias_tuple', bad))
        bad = copy.deepcopy(record); bad['logical_episodes'][0]['campaign'] = names[-1]; mutations.append(('misbound_alias_policy', bad))
        for title, bad in mutations:
            frozen.write_text(json.dumps(bad, allow_nan=False), encoding='utf-8')
            assert rejects(lambda: evaluator.verify_freeze(directory, plan_file, plan_stub)), title
        frozen.write_text(json.dumps(record, allow_nan=False), encoding='utf-8')
        evaluator.verify_freeze(directory, plan_file, plan_stub)
        return {'complete_artificial_manifest_accepted': True,
                'negative_manifest_mutations_rejected': [title for title, _ in mutations],
                'no_source_target_reveal_or_campaign_fit': True}

    def scoring_formula():
        rows = []
        errors = []
        for mu, sd, z in ((0., 1., 0.), (-.4, .7, -1.), (.8, 1.5, 2.)):
            y = mu + sd * z
            left = quad(lambda t: float(ndtr(t)) ** 2, -math.inf, z, epsabs=1e-11)[0]
            right = quad(lambda t: float(ndtr(-t)) ** 2, z, math.inf, epsabs=1e-11)[0]
            expected = sd * (left + right)
            actual = evaluator.gaussian_crps([mu], [sd], [y])[0]
            error = close(actual, expected, 'independent CDF CRPS')
            errors.append(error); rows.append({'mu': mu, 'sd': sd, 'z': z, 'absolute_error': error})
        assert rejects(lambda: evaluator.gaussian_crps([0.], [0.], [0.]))
        assert rejects(lambda: evaluator.gaussian_crps([0.], [1.], [math.nan]))
        folder = output / 'SYNTHETIC_GROUP_SCORE'
        folder.mkdir()
        meta = ([{'role': 'repair', 'component_id': 'SYNTHETIC_R_' + str(i)} for i in range(8)]
                + [{'role': 'frontier', 'component_id': 'SYNTHETIC_F_' + str(i)} for i in range(16)]
                + [{'role': 'test', 'component_id': 'SYNTHETIC_TEST_A'},
                   {'role': 'test', 'component_id': 'SYNTHETIC_TEST_A'},
                   {'role': 'test', 'component_id': 'SYNTHETIC_TEST_B'}])
        n = len(meta)
        truth = np.zeros((n, 3))
        truth[8:24, 2] = np.arange(16) / 10.
        truth[24] = 0.; truth[25] = 1.; truth[26] = 3.
        with (folder / 'forecast_24.npz').open('xb') as stream:
            np.savez_compressed(stream, mean_standardized=np.zeros(3 * n),
                covariance_standardized=np.eye(3 * n), observed_mask=np.zeros(3 * n, dtype=bool),
                warm_location=np.zeros(3), warm_scale=np.ones(3))
        actions = [{'oracle': {'row_position': i, 'property_indices': [2],
                               'native_values': [float(truth[i, 2])]}} for i in range(24)]
        write(folder / 'ACTIONS.json', {'initial': [], 'campaign': actions,
                                       'new_bundle_cost': 24, 'initial_bundle_cost': 0})
        independent_point = []
        for z in (0., 1., 3.):
            value = (quad(lambda t: float(ndtr(t)) ** 2, -math.inf, z, epsabs=1e-11)[0]
                     + quad(lambda t: float(ndtr(-t)) ** 2, z, math.inf, epsabs=1e-11)[0])
            independent_point.append(value)
        group_macro = (.5 * (independent_point[0] + independent_point[1]) + independent_point[2]) / 2.
        row_macro = sum(independent_point) / 3.
        assert not math.isclose(group_macro, row_macro, abs_tol=1e-4)
        for sign in (-1, 1):
            result = evaluator.score_snapshot('SYNTHETIC_ONLY', {'rows': meta, 'frontier_utility_sign': sign},
                                               truth, folder, 24)
            errors.append(close(result['equal_group_standardized_CRPS'], group_macro, 'equal-original-group CRPS'))
            assert result['test_metadata_groups'] == 2 and result['test_rows'] == 3
            assert result['frontier_retrieval']['verified_frontier_groups_with_target'] == 16
            assert result['frontier_retrieval']['verified_top_decile_group_hits'] == 2
            assert result['frontier_retrieval']['simple_regret_to_known_offline_pool'] == 0.
        return {'independent_CDF_integrals': rows, 'max_absolute_error': max(errors),
                'unequal_row_count_equal_group_weighting': True,
                'both_signed_finite_pool_retrieval_directions': True}

    for name, function in (
            ('direct_vs_sequential_Gaussian_conditioning', conditioning),
            ('metadata_only_RMS_geometry_and_warm_ICM', features_and_warm_icm),
            ('IVR_same_rule_KG_Jensen_and_approximation_diagnostics', acquisition),
            ('predictive_resource_weights_and_offdiagonal_ablation', resource_policy),
            ('artificial_bundle_oracle_roles_costs_and_hidden_test', oracle_roles_costs),
            ('metadata_group_roles_and_deterministic_aliases', metadata_and_aliases),
            ('complete_forecast_manifest_before_reveal', complete_freeze_guards),
            ('Gaussian_CRPS_by_independent_CDF_integration', scoring_formula)):
        check(name, function)
    after = {name: digest(source / name) for name in plan['source_sha256']}
    final = {'actual_end_utc': utc(), 'status': 'PASS' if all(x['status'] == 'PASS' for x in results) and after == before else 'FAIL',
             'plan_sha256': digest(args.plan), 'families': results,
             'source_unchanged': after == before, 'source_sha256': after,
             'pass_families': sum(x['status'] == 'PASS' for x in results),
             'fail_families': sum(x['status'] == 'FAIL' for x in results),
             'real_source_arrays_read': False, 'real_material_fits_scores_or_campaigns': 0,
             'new_material_evidence': False,
             'limitation': 'Bounded synthetic mathematical/IO audit, not a security sandbox or material efficacy/generalization/novelty proof.'}
    write(output / 'INDEPENDENT_SYNTHETIC_AUDIT.json', final)
    print(json.dumps({k: final[k] for k in ('status', 'pass_families', 'fail_families', 'source_unchanged')}))
    return 0 if final['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
