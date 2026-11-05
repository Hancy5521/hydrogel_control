"""Independent direct-inverse audit of the explicitly posthoc covariance note.

Reads covariance/index NPZ arrays and fit receipts only; no TRAIN or targets.
All conditional code covariances are full rank in this saved diagnostic, so
ordinary solve(A, B.T) provides an independent check of the eigen-support recipe.
"""
import argparse
import ast
import csv
import numpy as np
from audit_independent import (Checks, BOUNDS, event, read, save, sha, utc)


def main(stage):
    checks = Checks()
    event(stage, 'posthoc_covariance_independent', 'started', {'source_sha256': sha(__file__)})
    plan_path = stage/'diagnostics/PLAN.json'
    result_path = stage/'diagnostics/output/RESULT.json'
    plan, result = read(plan_path), read(result_path)
    source_path = stage/'diagnostics/covariance_diagnostics.py'
    checks.true(plan['posthoc'] is True and result['posthoc'] is True, 'posthoc_classification')
    checks.true(result['plan_sha256'] == sha(plan_path), 'plan_binding')
    checks.true(plan['source_sha256'] == sha(source_path) == result['source_sha256'], 'source_binding')
    checks.true(not (stage/'diagnostics/output/STOP.json').exists(), 'no_diagnostic_failure')
    decoded = set()
    tree = ast.parse(source_path.read_text(encoding='utf-8'))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
              and node.value.id == 'z' and isinstance(node.slice, ast.Constant)):
            decoded.add(node.slice.value)
    checks.true(decoded == {'reference_covariance', 'cross', 'observed', 'targets'}, 'explicit_NPZ_read_boundary')
    checks.true(not set(imports).intersection({'assets', 'experiment', 'representations',
                'conditional_metrics', 'joint_code_gaussian', 'scipy.optimize'}), 'no_fit_score_imports')
    csv_path = stage/'diagnostics/output/CONDITIONAL_COVARIANCE.csv'
    checks.true(sha(csv_path) == result['CSV_sha256'], 'CSV_binding')
    with csv_path.open(encoding='utf-8', newline='') as f:
        csv_rows = list(csv.DictReader(f))
    rows = {(r['method'], r['P'], r['mask']): r for r in result['covariance_rows']}
    marginals = {(r['method'], r['P'], r['mask']): r for r in result['marginal_fractions']}
    csv_map = {(r['method'], int(r['P']), r['mask']): r for r in csv_rows}
    fitted = {(r['method'], r['P']): r for r in result['fixed_fit_parameter_rows']}
    checks.true(len(rows) == len(marginals) == len(csv_map) == 150 and len(fitted) == 50,
                'full_diagnostic_census')
    independent, fixed_parameters = [], []
    for binding in plan['input_bindings']:
        file = stage/binding['file']
        checks.true(sha(file) == binding['sha256'], 'diagnostic_input_hash')
        if binding['kind'] == 'fit_parameters':
            rec = read(file)
            theta = np.asarray(rec['theta'])
            chol = np.array([[np.exp(theta[2]), 0.], [theta[3], np.exp(theta[4])]])
            s = chol @ chol.T
            key = (binding['method'], binding['P'])
            expected = fitted[key]
            checks.close(expected['theta'], theta, 'fit_theta', atol=0, rtol=0)
            checks.close(expected['a'], theta[:2], 'fit_mean', atol=0, rtol=0)
            checks.close(expected['S'], s, 'fit_covariance', atol=1e-12)
            checks.close(expected['S_eigenvalues'], np.linalg.eigvalsh(s), 'fit_eigenvalues', atol=1e-12)
            lower = (abs(theta-BOUNDS[:, 0]) <= 1e-8)
            upper = (abs(theta-BOUNDS[:, 1]) <= 1e-8)
            checks.true(expected['lower_bound_hits'] == lower.tolist(), 'lower_bound_count')
            checks.true(expected['upper_bound_hits'] == upper.tolist(), 'upper_bound_count')
            fixed_parameters.append({'method': key[0], 'theta': theta, 'S': s,
                                     'lower': lower, 'upper': upper,
                                     'pg': rec['projected_gradient_inf']})
            continue
        with np.load(file, allow_pickle=False) as data:
            k, c = data['reference_covariance'], data['cross']
            o, h = data['observed'], data['targets']
        ko = k[np.ix_(o, o)]
        kho = k[np.ix_(h, o)]
        a = np.eye(2)-c[o].T @ np.linalg.solve(ko, c[o])
        a = (a+a.T)/2.
        b = c[h]-kho @ np.linalg.solve(ko, c[o])
        v = k[np.ix_(h, h)]-kho @ np.linalg.solve(ko, kho.T)
        v = (v+v.T)/2.
        checks.true(np.linalg.eigvalsh(a).min() > 1e-10, 'fullrank_A_no_truncation')
        e = b @ np.linalg.solve(a, b.T)
        e = (e+e.T)/2.
        residual = (v-e+(v-e).T)/2.
        checks.true(np.linalg.eigvalsh(v).min() > 0 and np.linalg.eigvalsh(residual).min() > 0,
                    'SPD_target_and_fullrank_residual')
        fraction = np.diag(e)/np.diag(v)
        key = (binding['method'], binding['P'], binding['mask'])
        expected = rows[key]
        independent_row = {
            'method': key[0], 'P': key[1], 'mask': key[2],
            'target_count': len(h), 'conditional_code_rank': 2,
            'conditional_code_eigenvalue_min': float(np.linalg.eigvalsh(a).min()),
            'conditional_code_eigenvalue_max': float(np.linalg.eigvalsh(a).max()),
            'trace_reference_target_covariance': float(np.trace(v)),
            'trace_code_explained_covariance': float(np.trace(e)),
            'explained_trace_fraction': float(np.trace(e)/np.trace(v)),
            'mean_marginal_explained_fraction': float(fraction.mean()),
            'min_marginal_explained_fraction': float(fraction.min()),
            'max_marginal_explained_fraction': float(fraction.max()),
            'residual_trace_fraction': float(np.trace(residual)/np.trace(v)),
            'min_residual_eigenvalue': float(np.linalg.eigvalsh(residual).min())}
        for name, value in independent_row.items():
            if name in ('method', 'P', 'mask'):
                checks.true(expected[name] == value, 'diagnostic_identity')
            else:
                checks.close(expected[name], value, 'diagnostic_'+name, atol=1e-8)
                checks.close(float(csv_map[key][name]), value, 'CSV_'+name, atol=1e-8)
        checks.close(marginals[key]['per_target_marginal_explained_fraction'], fraction,
                     'per_target_reference_fraction', atol=1e-8)
        independent.append(independent_row)
    scalar_keys = ['explained_trace_fraction', 'mean_marginal_explained_fraction',
                   'residual_trace_fraction', 'conditional_code_eigenvalue_min',
                   'conditional_code_eigenvalue_max']
    for method in ('prior_kl', 'curvature45'):
        subset = [r for r in independent if r['method'] == method]
        groups = [('all_masks', subset)] + [(mask, [r for r in subset if r['mask'] == mask])
                                            for mask in ('spread', 'middle_gap', 'one_sided')]
        for group, selected in groups:
            target = (result['summary'][method][group] if group == 'all_masks'
                      else result['summary'][method]['by_mask'][group])
            for name in scalar_keys:
                values = [r[name] for r in selected]
                for statistic, value in [('mean', np.mean(values)), ('min', min(values)),
                                         ('median', np.median(values)), ('max', max(values))]:
                    checks.close(target[name][statistic], value, 'summary_'+name+'_'+statistic, atol=1e-8)
        checks.true(result['summary'][method]['conditional_rank_counts'] == {'0': 0, '1': 0, '2': 75},
                    'all_rank2')
        pars = [r for r in fixed_parameters if r['method'] == method]
        theta = np.array([r['theta'] for r in pars])
        s = np.array([r['S'] for r in pars])
        target = result['learned_parameter_summary'][method]
        checks.true(target['fits'] == 25, '25_fits_permethod')
        for name, value in [('theta_mean', theta.mean(0)), ('theta_min', theta.min(0)),
                            ('theta_max', theta.max(0)), ('mean_a', theta[:, :2].mean(0)),
                            ('mean_S', s.mean(0)),
                            ('lower_bound_hit_counts', np.array([r['lower'] for r in pars]).sum(0)),
                            ('upper_bound_hit_counts', np.array([r['upper'] for r in pars]).sum(0)),
                            ('S_eigenvalue_min_over_fits', min(np.linalg.eigvalsh(r['S']).min() for r in pars)),
                            ('S_eigenvalue_max_over_fits', max(np.linalg.eigvalsh(r['S']).max() for r in pars)),
                            ('worst_projected_gradient_inf', max(r['pg'] for r in pars))]:
            checks.close(target[name], value, 'parameter_summary_'+name, atol=1e-12)
    save(stage/'audit/COVARIANCE_AUDIT.json', {
        'actual_utc': utc(), 'status': 'PASS', 'checks': checks.count,
        'maximum_absolute_errors': checks.max_error,
        'source_sha256': sha(__file__), 'helper_sha256': sha(__file__.replace('audit_covariance_independent.py', 'audit_independent.py')),
        'diagnostic_plan_sha256': sha(plan_path), 'diagnostic_result_sha256': sha(result_path),
        'covariance_laws_checked': 150, 'fit_parameter_receipts_checked': 50,
        'formula': 'Direct ordinary solve of fullrank2x2A; no eigen-support pseudoinverse or truncation.',
        'producer_imports': 0, 'TRAIN_opened': False, 'new_fits': 0, 'new_target_scores': 0,
        'posthoc': True, 'limitation': 'Reference covariance allocation only; cannot prove accuracy, learnability, calibration or causality.'})
    event(stage, 'posthoc_covariance_independent', 'succeeded', {'checks': checks.count})
    print('COVARIANCE_AUDIT_PASS', checks.count)


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.stage.resolve())
    except Exception as exc:
        event(args.stage.resolve(), 'posthoc_covariance_independent', 'failed',
              {'type': type(exc).__name__, 'message': str(exc)})
        raise
