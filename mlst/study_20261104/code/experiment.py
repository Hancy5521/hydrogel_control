"""One bounded new representation study: qualify/register/fit/freeze/score.

Current candidate and all original forecasts are immutable.  No optimizer
fallback, seed selection, deployment fitting or hidden-S access is implemented.
"""
import os
for _name in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import csv
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve
from scipy.special import ndtr, ndtri
from scipy.stats import multivariate_normal
from threadpoolctl import threadpool_limits

from assets import (CODE, INPUT_PINS, MASKS, METHODS, P_ROWS, SEEDS,
                    imported_score_rows, load_train, prepare_inputs, read,
                    restore_reference, sha, source_bindings, utc, verify, write)
from representations import ROTATION45, directions, reference
from joint_code_gaussian import (BOUNDS, OPTIONS, PG_GATE, RIDGE, CodeReference,
                                ConditionalObjective, CurveTask, FitFailure, fit)
from local_shape import local_curvature_functionals, whiten_functionals
from conditional_metrics import gaussian_metrics


def environment():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'scipy': scipy.__version__, 'platform': platform.platform()}


def qualification(run_root):
    """Synthetic formula qualification only; no material inputs are opened."""
    output = Path(run_root) / 'qualification/RESULT.json'
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(20261104)
    support = np.linspace(-4.35, 6.63, 17)
    distance = np.abs(support[:, None] - support[None, :])
    kq = np.exp(-distance / 3.) + np.eye(17) * .02
    ucurv = whiten_functionals(kq, local_curvature_functionals(support))['directions']
    checks = []
    for method in METHODS:
        u = directions(method, kq, support)
        checks.append({'check': 'orthonormal', 'method': method,
                       'error': float(np.max(np.abs(u.T @ u - np.eye(2))))})
    ur = directions('curvature45', kq, support)
    projector_error = float(np.max(abs(ur @ ur.T - ucurv @ ucurv.T)))
    assert projector_error < 1e-12
    checks.append({'check': 'curvature45_same_span', 'error': projector_error})
    values, vectors = np.linalg.eigh(kq)
    uk = directions('prior_kl', kq, support)
    recovered = np.linalg.cholesky(kq) @ uk
    kl_error = float(np.max(abs(recovered @ recovered.T
                               - (vectors[:, -2:] * values[-2:]) @ vectors[:, -2:].T)))
    assert kl_error < 1e-12
    checks.append({'check': 'prior_kl_top2_covariance', 'error': kl_error})
    for n in (7, 17, 39):
        matrix = rng.normal(size=(n, n))
        covariance = matrix @ matrix.T + np.eye(n)
        cross = rng.normal(size=(n, 2)) * .03
        mean, truth = rng.normal(size=(2, n))
        ref = CodeReference(mean, covariance, cross)
        identity_mean, identity_cov = ref.joint(np.zeros(5))
        assert np.array_equal(identity_mean, mean)
        assert np.array_equal(identity_cov, covariance)
        residual_min = float(np.linalg.eigvalsh(covariance - cross @ cross.T)[0])
        assert residual_min > 0
        observed = np.array([0, 2, 4])
        targets = np.array(sorted(set(range(n)) - set(observed)))
        task = CurveTask(ref, truth, [observed])
        objective = ConditionalObjective([task])
        theta = np.array([.12, -.21, .13, .22, -.16])
        value, gradient = objective(theta)
        mu, cov = ref.joint(theta)
        koo, kho = cov[np.ix_(observed, observed)], cov[np.ix_(targets, observed)]
        conditional_mean = mu[targets] + kho @ np.linalg.solve(koo, truth[observed] - mu[observed])
        conditional_cov = cov[np.ix_(targets, targets)] - kho @ np.linalg.solve(koo, kho.T)
        direct = -multivariate_normal.logpdf(truth[targets], conditional_mean, conditional_cov) + RIDGE * (theta @ theta)
        nll_error = float(abs(value + objective.constant - direct))
        fd = np.array([(objective(theta + np.eye(5)[j] * 1e-5)[0]
                        - objective(theta - np.eye(5)[j] * 1e-5)[0]) / 2e-5 for j in range(5)])
        fd_error = float(np.max(abs(fd - gradient)))
        assert nll_error < 1e-9 and fd_error < 1e-6
        # Same unrestricted Gaussian law under C'=C R, a'=R.T a, S'=R.T S R.
        from joint_code_gaussian import unpack
        a, s, _ = unpack(theta)
        ap, sp = ROTATION45.T @ a, ROTATION45.T @ s @ ROTATION45
        lp = np.linalg.cholesky(sp)
        transformed = np.array([*ap, np.log(lp[0, 0]), lp[1, 0], np.log(lp[1, 1])])
        rotated = CodeReference(mean, covariance, cross @ ROTATION45)
        rotated_mean, rotated_cov = rotated.joint(transformed)
        rotation_error = max(float(np.max(abs(rotated_mean - mu))),
                             float(np.max(abs(rotated_cov - cov))))
        assert rotation_error < 1e-12
        checks.append({'check': 'direct_conditional_and_gradient', 'n': n,
                       'nll_error': nll_error, 'fd_gradient_error': fd_error,
                       'identity_theta_exact': True, 'residual_min_eigenvalue': residual_min,
                       'unconstrained_rotation_law_error': rotation_error})
    write(output, {'actual_utc': utc(), 'status': 'PASS', 'checks': checks,
                   'material_inputs_opened': 0, 'new_fits': 0,
                   'source_bindings': source_bindings(), 'environment': environment()})
    print('QUALIFIED', len(checks), flush=True)


def registration(run_root, asset_root):
    run_root = Path(run_root).resolve()
    path = run_root / 'experiment/PROTOCOL.json'
    if path.exists():
        raise FileExistsError(path)
    qpath = run_root / 'qualification/RESULT.json'
    qual = read(qpath)
    if qual['status'] != 'PASS' or qual['source_bindings'] != source_bindings():
        raise PermissionError('Current synthetic source qualification required')
    inputs = prepare_inputs(asset_root, run_root)
    old_model = read(run_root / 'experiment/inputs/inputs/FROZEN_CANDIDATE_MODEL.json')
    support = old_model['q']
    if len(support) != 17 or not np.all(np.diff(support) > 0):
        raise PermissionError('Original fixed17 support required')
    imported_score_rows(run_root)  # Existing scores only, never a new target read.
    protocol = {
        'kind': 'bounded_representation_robustness_v1', 'actual_utc': utc(),
        'authorization': '2026-11-04 manual request for MLST reading and bounded experiments',
        'question': 'Does the frozen curvature representation advantage survive multiple parameter-count-matched prior-only directions, and is its constrained result orientation sensitive?',
        'classification': 'New retrospective development on already-consumed25P. Not independent confirmation; no candidate optimization or best-seed selection.',
        'materials': list(P_ROWS), 'methods': list(METHODS), 'masks': MASKS,
        'random_seeds': list(SEEDS), 'original_random_seed': 20261030,
        'representations': {
            'random': 'QR of17x2 iidN(0,1), default_rng specified seed; original construction unchanged',
            'prior_kl': 'Top2 latentKq eigenvectors; maxabs pivot positive; u=L.T v/sqrt(lambda); no outcomes',
            'curvature45': 'Original prior-whitened curvature U times fixed+45degree orthogonal matrix'},
        'fixed_support': support, 'theta_bounds': BOUNDS, 'ridge': RIDGE,
        'optimizer': OPTIONS, 'pg_gate': PG_GATE,
        'initialization': 'Exactly one all-zero start per fit; no fallback/retry',
        'reference_fits': 0, 'candidate_refits': 0, 'deployment_fits': 0,
        'maximum_new_fits': 250, 'forecast_files_before_scoring': 750,
        'reference_policy': 'Reuse original per-held-P reference receipts;24otherP +20S lawful6anchors only',
        'head_training': 'Each held-P head uses24othercompleteP; all three fixed masks; same objective and regularizer',
        'primary': 'Equal25formulation x3layout macro marginalCRPS in lnviscosity; for each fixed seed and nonrandom comparator, no bestseedselect',
        'secondary': ['by-mask CRPS; RMSE; jointlogdensity;90%marginalcoverage andwidth',
                      'per-coordinate90percent interval score and marginalNLL; descriptive diagnostics',
                      'number of25formulations improved, descriptive only',
                      'eight-seed mean/median/min/max; no inferential pvalue',
                      'curvature45 difference diagnoses parameter-constraint orientation, not a new representation span'],
        'old_scores': 'Import original candidate/random/GP/conditionalGaussian/local/generic rows unchanged; do not rescore',
        'fairness_limit': 'Five incremental coefficients, data, objective, bounds and optimizer matched; not rotationally identical capacity because elementwise means/Cholesky bounds and penalty depend on basis orientation.',
        'freeze': 'Hash all750 new forecast files before any new target metric; producer fit phase never calls a scoring function',
        'failure_policy': 'Any fit/admission/hash/data-role failure stops whole analysis. Preserve all receipts, score no successful subset, do not change bounds/seeds/optimizer.',
        'new_measured_materials': 0, 'S_hidden_targets': 'Never available; closedTRAIN holdsNaN outside original6spreadanchors',
        'independent_confirmation': False, 'source_bindings': source_bindings(),
        'input_bindings': inputs, 'qualification_sha256': sha(qpath),
        'source_asset_root_local_only': str(Path(asset_root).resolve()),
        'environment': environment(),
    }
    write(path, protocol)
    print('REGISTERED', protocol['actual_utc'], sha(path), flush=True)


def protocol_at(run_root):
    path = Path(run_root) / 'experiment/PROTOCOL.json'
    protocol = read(path)
    verify(run_root, protocol)
    return path, protocol


def fitting(run_root):
    run_root = Path(run_root).resolve()
    protocol_path, protocol = protocol_at(run_root)
    output = run_root / 'experiment/forecasts'
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    write(output / 'START.json', {'actual_utc': utc(), 'protocol_sha256': sha(protocol_path),
                                'target_metrics_calls': 0, 'environment': environment()})
    forecasts, started_fits = [], 0
    try:
        y, coordinates, ids = load_train(run_root)
        support = np.array(protocol['fixed_support'])
        if not np.allclose(support, np.linspace(coordinates.min(), coordinates.max(), 17),
                           atol=1e-12, rtol=0):
            raise PermissionError('Registered original support differs from design coordinates')
        for method in METHODS:
            for held in P_ROWS:
                model, info = restore_reference(run_root, held)
                kq = model.joint(support, report=False).cov
                basis = directions(method, kq, support)
                others = [j for j in P_ROWS if j != held]
                tasks = [CurveTask(reference(model, coordinates[j], support, basis),
                                   y[j], list(MASKS.values())) for j in others]
                folder = output / method / f'P_{held}'
                folder.mkdir(parents=True, exist_ok=False)
                started_fits += 1
                fit_start, fit_utc = time.perf_counter(), utc()
                try:
                    result = fit(tasks)
                except FitFailure as exc:
                    write(folder / 'FIT_STOP.json', {**exc.receipt,
                          'started_utc': fit_utc, 'finished_utc': utc(),
                          'wall_seconds': time.perf_counter() - fit_start,
                          'method': method, 'held_P': held,
                          'protocol_sha256': sha(protocol_path)})
                    raise
                result.update({'started_utc': fit_utc, 'finished_utc': utc(),
                               'wall_seconds': time.perf_counter() - fit_start,
                               'method': method, 'held_P': held, 'training_P_rows': others,
                               'held_row_entirely_excluded': True,
                               'protocol_sha256': sha(protocol_path),
                               'reference_fits': 0, 'environment': environment()})
                write(folder / 'FIT.json', result)
                ref = reference(model, coordinates[held], support, basis)
                for mask, context in MASKS.items():
                    observed = np.array(context, dtype=int)
                    hidden = np.array(sorted(set(range(39)) - set(context)), dtype=int)
                    mean, covariance = ref.condition(result['theta'], observed,
                                                     y[held, observed], hidden)
                    target = folder / f'{mask}_FROZEN.npz'
                    with target.open('xb') as stream:
                        np.savez(stream, mean=mean, covariance=covariance,
                                 theta=np.array(result['theta']), observed=observed,
                                 observed_values=y[held, observed], targets=hidden,
                                 t=coordinates[held], support=support, basis=basis,
                                 reference_mean=ref.mean, reference_covariance=ref.covariance,
                                 cross=ref.cross, support_covariance=kq,
                                 method=np.array(method), held_P=np.array(held))
                    forecasts.append({'P': held, 'recipe': ids[held], 'mask': mask,
                                      'method': method, 'file': target.relative_to(run_root).as_posix(),
                                      'sha256': sha(target)})
                print('FROZEN', method, held, result['projected_gradient_inf'], flush=True)
        verify(run_root, protocol)
        if len(forecasts) != 750 or started_fits != 250:
            raise PermissionError('Exact250fits/750forecasts required')
        write(output / 'ALL750_FROZEN.json', {
            'actual_utc': utc(), 'protocol_sha256': sha(protocol_path),
            'forecasts': forecasts, 'fits': started_fits,
            'target_metrics_calls': 0, 'S_hidden_targets_read': 0,
            'wall_seconds': time.perf_counter() - start,
            'source_bindings': source_bindings(), 'environment': environment()})
    except Exception as exc:
        write(output / 'STOP.json', {'actual_utc': utc(), 'type': type(exc).__name__,
              'message': str(exc), 'traceback': traceback.format_exc(),
              'started_fits': started_fits, 'frozen_forecasts': len(forecasts),
              'target_metrics_calls': 0, 'S_hidden_targets_read': 0,
              'wall_seconds': time.perf_counter() - start})
        raise


def aggregate(rows):
    return {'CRPS': float(np.mean([r['crps'] for r in rows])),
            'RMSE': float(np.sqrt(np.mean([r['mean_mse'] for r in rows]))),
            'mean_joint_log_density': float(np.mean([r['joint_log_density'] for r in rows])),
            'coverage90': float(np.mean([r['coverage90'] for r in rows])),
            'width90': float(np.mean([r['width90'] for r in rows]))}


def scoring(run_root):
    run_root = Path(run_root).resolve()
    protocol_path, protocol = protocol_at(run_root)
    folder = run_root / 'experiment/forecasts'
    manifest_path = folder / 'ALL750_FROZEN.json'
    if (folder / 'STOP.json').exists():
        raise PermissionError('Failed experiment cannot be selectively scored')
    manifest = read(manifest_path)
    expected = {(method, held, mask) for method in METHODS for held in P_ROWS for mask in MASKS}
    records = manifest['forecasts']
    if (manifest['protocol_sha256'] != sha(protocol_path) or len(records) != 750
            or {(r['method'], r['P'], r['mask']) for r in records} != expected
            or len({r['file'] for r in records}) != 750):
        raise PermissionError('All750distinctforecasts must be frozen before scoring')
    for rec in records:
        if sha(run_root / rec['file']) != rec['sha256']:
            raise PermissionError('Frozen forecast changed: ' + rec['file'])
    output = run_root / 'experiment/scores'
    output.mkdir(exist_ok=False)
    write(output / 'START.json', {'actual_utc': utc(), 'protocol_sha256': sha(protocol_path),
                                'all750_frozen_sha256': sha(manifest_path),
                                'new_fits': 0, 'independent_confirmation': False})
    start = time.perf_counter()
    y, _, _ = load_train(run_root)
    rows, coordinate_rows = [], []
    for rec in records:
        with np.load(run_root / rec['file'], allow_pickle=False) as z:
            mean, cov, hidden = z['mean'], z['covariance'], z['targets']
        truth = y[rec['P'], hidden]
        metrics = gaussian_metrics(mean, cov, truth)
        rows.append({'P': rec['P'], 'mask': rec['mask'], 'model': rec['method'],
                     'target_indices': hidden.tolist(), **metrics})
        lower, upper = np.array(metrics['lower90']), np.array(metrics['upper90'])
        sd = np.sqrt(np.diag(cov))
        interval_score = ((upper - lower) + 20 * np.maximum(lower - truth, 0)
                          + 20 * np.maximum(truth - upper, 0))
        for j, target in enumerate(hidden):
            coordinate_rows.append({'method': rec['method'], 'P': rec['P'], 'mask': rec['mask'],
                'target_index': int(target), 'crps': metrics['per_target_crps'][j],
                'squared_error': float((mean[j] - truth[j]) ** 2),
                'covered90': int(lower[j] <= truth[j] <= upper[j]),
                'width90': float(upper[j] - lower[j]),
                'interval_score90': float(interval_score[j]),
                'marginal_nll': float(.5 * (np.log(2 * np.pi * sd[j] ** 2)
                                           + ((truth[j] - mean[j]) / sd[j]) ** 2))})
    with (output / 'PER_COORDINATE.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coordinate_rows[0]))
        writer.writeheader()
        writer.writerows(coordinate_rows)
    original = imported_score_rows(run_root)
    combined = original + rows
    models = sorted({r['model'] for r in combined})
    summary = {method: aggregate([r for r in combined if r['model'] == method]) for method in models}
    by_mask = {mask: {method: aggregate([r for r in combined if r['model'] == method
                                       and r['mask'] == mask]) for method in models}
               for mask in MASKS}
    base = np.array([np.mean([r['crps'] for r in original
                     if r['model'] == 'joint_code_gaussian' and r['P'] == held]) for held in P_ROWS])
    differences = {}
    for method in METHODS:
        values = np.array([np.mean([r['crps'] for r in rows if r['model'] == method
                                    and r['P'] == held]) for held in P_ROWS])
        delta = base - values
        differences[method] = {'candidate_minus_comparator_macro_crps': float(np.mean(delta)),
                               'candidate_lower_crps_materials': int(np.sum(delta < 0)),
                               'paired_formulations': 25, 'independent_confirmation': False}
    random_scores = np.array([summary[f'random_{seed}']['CRPS'] for seed in SEEDS])
    result = {'actual_utc': utc(), 'status': 'COMPLETED',
        'protocol_sha256': sha(protocol_path), 'all750_frozen_sha256': sha(manifest_path),
        'new_rows': rows, 'aggregate': summary, 'by_mask': by_mask,
        'candidate_comparison': differences,
        'eight_new_random_seeds': {'mean_CRPS': float(random_scores.mean()),
            'median_CRPS': float(np.median(random_scores)), 'min_CRPS': float(random_scores.min()),
            'max_CRPS': float(random_scores.max()),
            'seeds_with_lower_CRPS_than_candidate': int(np.sum(random_scores < summary['joint_code_gaussian']['CRPS'])),
            'warning': 'No best-seed selection or inferential significance; all25P already consumed'},
        'new_materials': 0, 'new_fits_during_score': 0, 'old_scores_rescored': False,
        'S_hidden_targets_read': 0, 'independent_confirmation': False,
        'coordinate_metrics_sha256': sha(output / 'PER_COORDINATE.csv'),
        'wall_seconds': time.perf_counter() - start, 'environment': environment()}
    verify(run_root, protocol)
    write(output / 'RESULT.json', result)
    print('SCORED750', {k: v['CRPS'] for k, v in summary.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['qualify', 'register', 'fit', 'score'])
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--asset-root', type=Path,
                        help='Explicit original missing_controls stage; required only for registration')
    args = parser.parse_args()
    if args.mode == 'register' and args.asset_root is None:
        parser.error('--asset-root is required for register')
    with threadpool_limits(limits=1):
        if args.mode == 'qualify':
            qualification(args.run_root)
        elif args.mode == 'register':
            registration(args.run_root, args.asset_root)
        elif args.mode == 'fit':
            fitting(args.run_root)
        else:
            scoring(args.run_root)


if __name__ == '__main__':
    main()
