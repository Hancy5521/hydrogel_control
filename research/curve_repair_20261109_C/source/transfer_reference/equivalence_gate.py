"""SOURCE ONLY until root registration: bounded synthetic universal-t gates.

Not a material fit or new scientific validation. K=I reduces to existing strong
power-law Student; multiplying K by a scalar must not change the t law. Dense
classical formula, index permutation and roundoff residual rejection are checked.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from transfer_experiment import authorize, fixed_student_module, sha, utc, write
from universal_student import UndefinedUniversalStudent, fit_predict


def dense_formula(x, y, o, t, k):
    xo = np.column_stack([np.ones(len(o)), x[o]])
    xt = np.column_stack([np.ones(len(t)), x[t]])
    koo, kot, ktt = k[np.ix_(o, o)], k[np.ix_(o, t)], k[np.ix_(t, t)]
    solved_x, solved_y = np.linalg.solve(koo, xo), np.linalg.solve(koo, y)
    gram = xo.T @ solved_x
    beta = np.linalg.solve(gram, xo.T @ solved_y)
    r = y - xo @ beta
    sse = float(r @ np.linalg.solve(koo, r))
    h = xt - kot.T @ solved_x
    mean = xt @ beta + kot.T @ np.linalg.solve(koo, r)
    shape = (sse / 4) * (ktt - kot.T @ np.linalg.solve(koo, kot) + h @ np.linalg.solve(gram, h.T))
    return beta, sse, mean, shape


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    authorize(args.registration, 'gate')
    if args.output.exists():
        raise FileExistsError('Do not overwrite or repeat a completed/partial gate')
    ordinary = fixed_student_module()
    x = np.array([-1.8, -1.1, -.6, -.2, .2, .7, 1.0, 1.4, 1.9, 2.3, 2.8, 3.2])
    o, t = np.array([0, 2, 4, 5, 7, 8]), np.array([1, 3, 6, 9, 10, 11])
    y = .3 - .4 * x[o] + np.array([.1, -.08, .03, .12, -.02, -.06])
    checks = []
    def compare(label, a, b, atol=2e-10, rtol=2e-10):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol):
            raise AssertionError(label)
        checks.append({'check': label, 'elements': a.size,
                       'max_absolute_error': float(np.max(np.abs(a-b)))})
    identity = fit_predict(x, o, y, t, np.eye(len(x)), 'identity_fixture')
    power = ordinary.fit_predict(x[o], y, x[t], 'power_law')
    for field in ('beta', 'sse', 'df', 'location', 'scale_matrix', 'predictive_covariance'):
        compare('K_identity_equals_powerlaw_' + field, getattr(identity, field), getattr(power, field))
    k = np.exp(-.5 * ((x[:, None] - x[None, :]) / .9) ** 2) + .07 * np.eye(len(x))
    correlated = fit_predict(x, o, y, t, k, 'fixed_correlated_fixture')
    beta, sse, mean, shape = dense_formula(x, y, o, t, k)
    for name, expected in [('beta', beta), ('sse', sse), ('location', mean), ('scale_matrix', shape)]:
        compare('dense_correlated_' + name, getattr(correlated, name), expected)
    scaled = fit_predict(x, o, y, t, 7 * k, 'scaled_correlated_fixture')
    for field in ('beta', 'location', 'scale_matrix', 'predictive_covariance'):
        compare('K_global_scale_invariance_' + field, getattr(correlated, field), getattr(scaled, field))
    order = np.array([3, 0, 5, 2, 1, 4])
    shuffled = fit_predict(x, o[order], y[order], t[::-1], k, 'permuted_fixture')
    compare('anchor_order_and_target_order_mean', correlated.location, shuffled.location[::-1])
    compare('target_order_scale', correlated.scale_matrix, shuffled.scale_matrix[::-1, ::-1])
    exactline_error = None
    try:
        fit_predict(np.arange(12.), np.arange(6), 1. + 2*np.arange(6.), np.arange(6, 12),
                    np.eye(12), 'exactline_fixture')
    except UndefinedUniversalStudent as exc:
        exactline_error = str(exc)
        checks.append({'check': 'exact_line_rejected_without_variance_floor', 'error': str(exc)})
    else:
        raise AssertionError('Exact line roundoff must not be treated as positive stochastic variance')
    write(args.output, {'actual_utc': utc(), 'schema': 'mlst-universal-student-equivalence-gate-v1',
          'status': 'PASS', 'registration_sha256': sha(args.registration),
          'source_only_until_authorized': False, 'checks': checks,
          'numeric_trace': {
              'fixture': {'log_gamma': x.tolist(), 'anchor_indices': o.tolist(),
                          'anchor_values': y.tolist(), 'target_indices': t.tolist(),
                          'correlated_K': k.tolist(), 'scaled_K_multiplier': 7.0,
                          'anchor_permutation': order.tolist()},
              'identity_universal': identity.as_dict(),
              'identity_powerlaw': power.as_dict(),
              'correlated_universal': correlated.as_dict(),
              'scaled_correlated_universal': scaled.as_dict(),
              'permuted_universal': shuffled.as_dict(),
              'correlated_dense_formula': {'beta': beta.tolist(), 'sse': sse,
                                            'location': mean.tolist(), 'scale_matrix': shape.tolist()},
              'exactline_rejected_fixture': {
                  'log_gamma': np.arange(12.).tolist(), 'anchor_indices': list(range(6)),
                  'anchor_values': (1. + 2*np.arange(6.)).tolist(),
                  'target_indices': list(range(6, 12)), 'K_kind': 'identity_12',
                  'error': exactline_error}},
          'new_material_labels': 0, 'synthetic_fixture_only': True,
          'limitation': 'Equivalence/software gate, not material benefit or new theory'})
    print('PASS', len(checks), 'bounded synthetic equivalence checks')


if __name__ == '__main__':
    main()
