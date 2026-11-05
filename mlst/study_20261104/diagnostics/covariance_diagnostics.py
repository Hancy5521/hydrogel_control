"""One explicitly posthoc, covariance-only representation explanation.

This standalone program opens selected covariance/index arrays and fit-parameter
receipts only.  It never reads TRAIN, observed_values, truths, or target metrics.
It is not a new material experiment and does not select/refit a model.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

METHODS = ('prior_kl', 'curvature45')
NPZ_KEYS = ('reference_covariance', 'cross', 'observed', 'targets')
BOUNDS = np.array([[-1., 1.], [-1., 1.], [-.75, .75], [-1., 1.], [-.75, .75]])


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def static_source_audit():
    """Small serial source-boundary check, not an independent science audit."""
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    forbidden = {'assets', 'experiment', 'representations', 'joint_code_gaussian',
                 'conditional_metrics', 'scipy.optimize'}
    if forbidden.intersection(imports):
        raise PermissionError('Diagnostic must not import producer/fitting/scoring')
    return {'actual_utc': utc(), 'status': 'PASS', 'source_sha256': sha(__file__),
            'imports': imports, 'allowed_npz_arrays': list(NPZ_KEYS),
            'check_scope': 'Serial source import/explicit array-read boundary; independent numerical audit is separate'}


def register(root):
    root = Path(root).resolve()
    destination = root / 'diagnostics/PLAN.json'
    if destination.exists():
        raise FileExistsError(destination)
    audit = static_source_audit()
    manifest_path = root / 'experiment/forecasts/ALL750_FROZEN.json'
    protocol_path = root / 'experiment/PROTOCOL.json'
    score_path = root / 'experiment/scores/RESULT.json'
    # Hash the already-consumed scoring result for posthoc timing/provenance;
    # its contents, response values and scores are not opened by this program.
    score_digest = sha(score_path)
    manifest = read(manifest_path)
    selected = [r for r in manifest['forecasts'] if r['method'] in METHODS]
    if len(selected) != 150:
        raise PermissionError('Exactly25x3x2 frozen laws required')
    identities = {(r['method'], r['P'], r['mask']) for r in selected}
    if len(identities) != 150:
        raise PermissionError('Duplicate covariance diagnostic identity')
    inputs = []
    fits = {}
    for rec in selected:
        path = root / rec['file']
        if sha(path) != rec['sha256']:
            raise PermissionError('Frozen forecast identity changed')
        inputs.append({'kind': 'frozen_covariance', 'file': rec['file'],
                       'sha256': rec['sha256'], 'method': rec['method'],
                       'P': rec['P'], 'mask': rec['mask']})
        fit_path = path.parent / 'FIT.json'
        key = (rec['method'], rec['P'])
        fits[key] = {'kind': 'fit_parameters',
                     'file': fit_path.relative_to(root).as_posix(),
                     'sha256': sha(fit_path), 'method': rec['method'], 'P': rec['P']}
    inputs.extend(fits[key] for key in sorted(fits))
    if len(fits) != 50:
        raise PermissionError('Exactly50 fixed fit receipts required')
    plan = {
        'actual_utc': utc(), 'kind': 'posthoc_covariance_explanation_v1',
        'posthoc': True,
        'trigger': 'Requested after canonical target CRPS had already shown prior-KL better than original curvature and eight random representations. This is explanatory and not prospective performance evidence.',
        'canonical_score_result_sha256_seen_before_plan': score_digest,
        'canonical_score_result_contents_read_by_this_program': False,
        'original_experiment_protocol_sha256': sha(protocol_path),
        'all750_frozen_manifest_sha256': sha(manifest_path),
        'methods': list(METHODS), 'covariance_laws': 150, 'fixed_fit_receipts': 50,
        'source_sha256': sha(__file__), 'source_audit': audit,
        'input_bindings': inputs,
        'array_read_allowlist': list(NPZ_KEYS),
        'formula': {
            'A': 'I2 - C_O.T solve(K_OO,C_O)',
            'B': 'C_H - K_HO solve(K_OO,C_O)',
            'V0': 'K_HH - K_HO solve(K_OO,K_OH)',
            'E': 'B A_inverse_on_admitted_stochastic_support B.T',
            'fraction_trace': 'trace(E)/trace(V0)',
            'fraction_marginal_mean': 'mean(diag(E)/diag(V0))',
            'fraction_residual_trace': 'trace(V0-E)/trace(V0)'},
        'numerical_support_policy': {
            'symmetrization': 'A,V0,E and residual are averaged with their transpose; no jitter or eigenvalue clipping',
            'A_rank_tolerance': '128*eps*2*max(1,norm2(C_O.T solve(K_OO,C_O)))',
            'support': 'eigenvalues strictly above tolerance; any eigenvalue below minus tolerance stops; roundoff-null eigenmodes excluded only under stated support check',
            'cross_support_tolerance': '128*eps*(nH+2)*max(1,norm2(B),norm2(V0)) in maximum absolute matrix entries',
            'PSD_tolerance': '128*eps*nH*max(1,norm2(V0))',
            'checks': 'K_OO and V0 Cholesky; null-cross-support check; V0-E PSD within declared tolerance. Singular support is recorded, never repaired or silently discarded.'},
        'fit_summary': 'Read original theta/a/S receipts; verify5parameter unpack and count lower/upper bound hits with absolute1e-8, summarize all25folds without retuning',
        'interpretation': 'Reference conditional covariance allocation only. No labels, target scores or fitted prediction performance are recomputed. Ratios cannot prove accuracy, learnability, calibration, causality or independent generalization.',
        'no_new_fit': True, 'no_new_target_score': True,
        'no_measurement_label_access': True, 'no_TRAIN_open': True,
        'no_model_adoption_or_refit': True,
        'failure_policy': 'Any hash/covariance/support failure stops this complete analysis; preserve error receipt; no numerical rescue or selected successful subset',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
    }
    write(root / 'diagnostics/SERIAL_SOURCE_AUDIT.json', audit)
    write(destination, plan)
    print('POSTHOC_PLAN_REGISTERED', plan['actual_utc'], sha(destination), flush=True)


def verify(root, plan):
    if sha(__file__) != plan['source_sha256']:
        raise PermissionError('Registered diagnostic source changed')
    for rec in plan['input_bindings']:
        if sha(root / rec['file']) != rec['sha256']:
            raise PermissionError('Registered diagnostic input changed: ' + rec['file'])
    for file, digest in [
        ('experiment/PROTOCOL.json', plan['original_experiment_protocol_sha256']),
        ('experiment/forecasts/ALL750_FROZEN.json', plan['all750_frozen_manifest_sha256']),
        ('experiment/scores/RESULT.json', plan['canonical_score_result_sha256_seen_before_plan']),
    ]:
        if sha(root / file) != digest:
            raise PermissionError('Fixed parent receipt changed: ' + file)


def covariance_row(path, identity):
    # Access only the declared covariance/index arrays; never .values or bulk
    # dict(z) expansion, so observed_values and means are not decoded.
    with np.load(path, allow_pickle=False) as z:
        k = z['reference_covariance']
        c = z['cross']
        observed = z['observed']
        hidden = z['targets']
    if (k.shape != (39, 39) or c.shape != (39, 2)
            or observed.shape != (6,) or hidden.shape != (33,)
            or set(observed).intersection(hidden)
            or set(observed).union(hidden) != set(range(39))
            or not np.isfinite(k).all() or not np.isfinite(c).all()):
        raise ValueError('Exact finite39report/2code/6context/33target law required')
    koo = k[np.ix_(observed, observed)]
    kho = k[np.ix_(hidden, observed)]
    co, ch = c[observed], c[hidden]
    np.linalg.cholesky(koo)
    transfer = co.T @ np.linalg.solve(koo, co)
    a = np.eye(2) - transfer
    a = (a + a.T) / 2
    b = ch - kho @ np.linalg.solve(koo, co)
    v0 = k[np.ix_(hidden, hidden)] - kho @ np.linalg.solve(koo, kho.T)
    v0 = (v0 + v0.T) / 2
    np.linalg.cholesky(v0)
    eigenvalues, eigenvectors = np.linalg.eigh(a)
    eps = np.finfo(float).eps
    tol_a = 128 * eps * 2 * max(1., np.linalg.norm(transfer, 2))
    tol_cross = 128 * eps * 35 * max(1., np.linalg.norm(b, 2), np.linalg.norm(v0, 2))
    tol_psd = 128 * eps * 33 * max(1., np.linalg.norm(v0, 2))
    if eigenvalues.min() < -tol_a:
        raise ValueError('Negative conditional code covariance exceeds declared tolerance')
    keep = eigenvalues > tol_a
    span = eigenvectors[:, keep]
    null_cross_error = float(np.max(abs(b - b @ (span @ span.T))))
    if null_cross_error > tol_cross:
        raise ValueError('Target cross-covariance lies outside admitted stochastic code support')
    factor = b @ span / np.sqrt(eigenvalues[keep])
    explained = factor @ factor.T
    explained = (explained + explained.T) / 2
    residual = v0 - explained
    residual = (residual + residual.T) / 2
    residual_min = float(np.linalg.eigvalsh(residual)[0])
    if residual_min < -tol_psd:
        raise ValueError('Explained covariance exceeds reference conditional covariance')
    marginal = np.diag(explained) / np.diag(v0)
    trace_v0, trace_e = float(np.trace(v0)), float(np.trace(explained))
    row = {'method': identity['method'], 'P': identity['P'], 'mask': identity['mask'],
           'target_count': 33, 'conditional_code_rank': int(keep.sum()),
           'conditional_code_eigenvalue_min': float(eigenvalues[0]),
           'conditional_code_eigenvalue_max': float(eigenvalues[1]),
           'trace_reference_target_covariance': trace_v0,
           'trace_code_explained_covariance': trace_e,
           'explained_trace_fraction': trace_e / trace_v0,
           'mean_marginal_explained_fraction': float(marginal.mean()),
           'min_marginal_explained_fraction': float(marginal.min()),
           'max_marginal_explained_fraction': float(marginal.max()),
           'residual_trace_fraction': float(np.trace(residual)) / trace_v0,
           'min_residual_eigenvalue': residual_min,
           'code_rank_tolerance': float(tol_a),
           'null_cross_support_max_error': null_cross_error,
           'cross_support_tolerance': float(tol_cross),
           'residual_PSD_tolerance': float(tol_psd)}
    return row, marginal.tolist()


def fit_row(path, identity):
    info = read(path)
    theta = np.asarray(info['theta'])
    if theta.shape != (5,) or not info['admitted']:
        raise ValueError('Admitted fixed five-parameter fit required')
    lower = np.array([[np.exp(theta[2]), 0.], [theta[3], np.exp(theta[4])]])
    covariance = lower @ lower.T
    if (not np.allclose(info['a'], theta[:2], atol=1e-13, rtol=0)
            or not np.allclose(info['S'], covariance, atol=1e-13, rtol=0)):
        raise ValueError('Serialized fit unpack differs')
    eigenvalues = np.linalg.eigvalsh(covariance)
    return {'method': identity['method'], 'P': identity['P'],
            'theta': theta.tolist(), 'a': theta[:2].tolist(), 'S': covariance.tolist(),
            'S_eigenvalues': eigenvalues.tolist(),
            'lower_bound_hits': (abs(theta - BOUNDS[:, 0]) <= 1e-8).tolist(),
            'upper_bound_hits': (abs(theta - BOUNDS[:, 1]) <= 1e-8).tolist(),
            'projected_gradient_inf': info['projected_gradient_inf']}


def summarized(rows, keys):
    return {key: {'mean': float(np.mean([r[key] for r in rows])),
                  'min': float(np.min([r[key] for r in rows])),
                  'median': float(np.median([r[key] for r in rows])),
                  'max': float(np.max([r[key] for r in rows]))} for key in keys}


def execute(root):
    root = Path(root).resolve()
    plan_path = root / 'diagnostics/PLAN.json'
    plan = read(plan_path)
    verify(root, plan)
    output = root / 'diagnostics/output'
    output.mkdir(exist_ok=False)
    started = time.perf_counter()
    write(output / 'START.json', {'actual_utc': utc(), 'plan_sha256': sha(plan_path),
                                'posthoc': True, 'new_fits': 0, 'new_target_scores': 0,
                                'measurement_labels_read': 0})
    covrows, fitrows, marginals = [], [], []
    try:
        for rec in plan['input_bindings']:
            path = root / rec['file']
            if rec['kind'] == 'frozen_covariance':
                row, marginal = covariance_row(path, rec)
                covrows.append(row)
                marginals.append({'method': rec['method'], 'P': rec['P'], 'mask': rec['mask'],
                                  'per_target_marginal_explained_fraction': marginal})
            elif rec['kind'] == 'fit_parameters':
                fitrows.append(fit_row(path, rec))
            else:
                raise ValueError('Unregistered input role')
        if len(covrows) != 150 or len(fitrows) != 50:
            raise ValueError('Incomplete covariance/fit census')
        with (output / 'CONDITIONAL_COVARIANCE.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(covrows[0]))
            writer.writeheader()
            writer.writerows(covrows)
        scalar_keys = ['explained_trace_fraction', 'mean_marginal_explained_fraction',
                       'residual_trace_fraction', 'conditional_code_eigenvalue_min',
                       'conditional_code_eigenvalue_max']
        summary = {}
        parameters = {}
        for method in METHODS:
            group = [r for r in covrows if r['method'] == method]
            summary[method] = {'all_masks': summarized(group, scalar_keys),
                'by_mask': {mask: summarized([r for r in group if r['mask'] == mask], scalar_keys)
                            for mask in ('spread', 'middle_gap', 'one_sided')},
                'conditional_rank_counts': {str(rank): sum(r['conditional_code_rank'] == rank for r in group)
                                            for rank in (0, 1, 2)}}
            fitted = [r for r in fitrows if r['method'] == method]
            theta = np.array([r['theta'] for r in fitted])
            arrays = np.array([r['S'] for r in fitted])
            parameters[method] = {
                'fits': len(fitted),
                'parameter_order': ['a0', 'a1', 'log_L00', 'L10', 'log_L11'],
                'theta_mean': theta.mean(axis=0).tolist(),
                'theta_min': theta.min(axis=0).tolist(),
                'theta_max': theta.max(axis=0).tolist(),
                'mean_a': theta[:, :2].mean(axis=0).tolist(),
                'mean_S': arrays.mean(axis=0).tolist(),
                'S_eigenvalue_min_over_fits': float(min(min(r['S_eigenvalues']) for r in fitted)),
                'S_eigenvalue_max_over_fits': float(max(max(r['S_eigenvalues']) for r in fitted)),
                'lower_bound_hit_counts': np.array([r['lower_bound_hits'] for r in fitted]).sum(axis=0).tolist(),
                'upper_bound_hit_counts': np.array([r['upper_bound_hits'] for r in fitted]).sum(axis=0).tolist(),
                'worst_projected_gradient_inf': max(r['projected_gradient_inf'] for r in fitted)}
        verify(root, plan)
        result = {'actual_utc': utc(), 'status': 'COMPLETED', 'posthoc': True,
                  'plan_sha256': sha(plan_path), 'source_sha256': sha(__file__),
                  'summary': summary, 'learned_parameter_summary': parameters,
                  'covariance_rows': covrows, 'marginal_fractions': marginals,
                  'fixed_fit_parameter_rows': fitrows,
                  'new_fits': 0, 'new_target_scores': 0, 'measurement_labels_read': 0,
                  'TRAIN_opened': False, 'S_hidden_targets_read': 0,
                  'method_adopted_or_modified': False,
                  'wall_seconds': time.perf_counter() - started,
                  'CSV_sha256': sha(output / 'CONDITIONAL_COVARIANCE.csv'),
                  'interpretation_limit': plan['interpretation']}
        write(output / 'RESULT.json', result)
        print('POSTHOC_COVARIANCE_COMPLETE', {m: summary[m]['by_mask']['one_sided']['explained_trace_fraction']['mean'] for m in METHODS}, flush=True)
    except Exception as exc:
        write(output / 'STOP.json', {'actual_utc': utc(), 'type': type(exc).__name__,
              'message': str(exc), 'completed_covariance_rows': len(covrows),
              'completed_parameter_rows': len(fitrows), 'new_fits': 0,
              'new_target_scores': 0, 'measurement_labels_read': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['register', 'run'])
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    {'register': register, 'run': execute}[args.mode](args.run_root)
