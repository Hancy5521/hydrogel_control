"""Numerical core for a fixed quadratic plus reverse-KL simplex objective.

Same estimator and KKT rule as the archived Newton implementation. Armijo uses
an analytic objective difference, so an irrelevant additive constant cannot
hide a small descent step. This is numerical infrastructure, not a new model.
"""
import math
import numpy as np

P = np.array([.05, .45, .45, .05])
MAX_ITERATIONS = 200
KKT_TOLERANCE = 1e-9


def log_remainder(t):
    """t-log(1+t), evaluated without subtracting nearly equal small numbers."""
    t = float(t)
    if not math.isfinite(t) or t <= -1:
        raise ValueError('finite relative step greater than -1 required')
    if abs(t) >= .001:
        return t - math.log1p(t)
    term = t*t
    terms = []
    for k in range(2, 40):
        value = term/k
        terms.append(value)
        if value == 0 or (k > 2 and abs(value) <= 1e-18*abs(terms[0])):
            break
        term *= -t
    return math.fsum(terms)


def evaluate(stats, q):
    q = np.asarray(q, dtype=float)
    blocks = int(stats['blocks'])
    if q.shape != (4*blocks,) or not np.isfinite(q).all() or (q <= 0).any():
        raise ValueError('positive matched simplex coordinates required')
    Q, b = np.asarray(stats['Q']), np.asarray(stats['b'])
    coefficients = float(stats['lambda'])*np.repeat(stats['endpoint_KL_weights'], 4)*np.tile(P, blocks)
    grad = 2*(Q@q-b)-coefficients/q
    H = 2*Q+np.diag(coefficients/(q*q))
    penalty = math.fsum(float(a)*(math.log(float(p))-math.log(float(v)))
                        for a, p, v in zip(coefficients, np.tile(P, blocks), q))
    objective = float(q@Q@q-2*b@q+stats['c']+penalty)
    if not np.isfinite(grad).all() or not np.isfinite(H).all() or not math.isfinite(objective):
        raise ArithmeticError('nonrepresentable objective or derivatives')
    return objective, grad, H


def projected_kkt(gradient, blocks):
    return max(abs(float(v)-math.fsum(map(float, gradient[4*r:4*r+4]))/4)
               for r in range(blocks) for v in gradient[4*r:4*r+4])


def stable_delta(stats, q, delta, gradient=None):
    if gradient is None:
        _, gradient, _ = evaluate(stats, q)
    coefficients = float(stats['lambda'])*np.repeat(stats['endpoint_KL_weights'], 4)*np.tile(P, int(stats['blocks']))
    linear = math.fsum(float(a)*float(b) for a, b in zip(gradient, delta))
    quadratic = float(delta@np.asarray(stats['Q'])@delta)
    remainder = math.fsum(float(a)*log_remainder(float(d)/float(v))
                          for a, d, v in zip(coefficients, delta, q))
    return math.fsum((linear, quadratic, remainder))


def fit(stats):
    blocks = int(stats['blocks'])
    if blocks not in (1, 2) or stats['lambda'] <= 0:
        raise ValueError('one/two blocks and positive regularization required')
    q = np.tile(P, blocks)
    Z = np.zeros((4*blocks, 3*blocks))
    for r in range(blocks):
        Z[4*r:4*r+3, 3*r:3*r+3] = np.eye(3)
        Z[4*r+3, 3*r:3*r+3] = -1
    history = []
    reason = None
    for iteration in range(MAX_ITERATIONS+1):
        value, gradient, H = evaluate(stats, q)
        residual = projected_kkt(gradient, blocks)
        if residual <= KKT_TOLERANCE:
            return {'status':'CONVERGED_KKT', 'q':q.tolist(), 'iterations':iteration,
                    'projected_KKT':residual, 'objective_native':value,
                    'history':history, 'arithmetic':'analytic_difference_no_additive_c'}
        if iteration == MAX_ITERATIONS:
            reason = 'maximum iterations without KKT clearance'
            break
        try:
            direction = Z@np.linalg.solve(Z.T@H@Z, -Z.T@gradient)
        except np.linalg.LinAlgError:
            reason = 'reduced Hessian solve failed; no added ridge'
            break
        slope = math.fsum(float(a)*float(b) for a, b in zip(gradient, direction))
        if not np.isfinite(direction).all() or not math.isfinite(slope) or slope >= 0:
            reason = 'no finite strict-descent Newton direction'
            break
        negative = direction < 0
        alpha = min(1., .99*float(np.min(-q[negative]/direction[negative]))) if negative.any() else 1.
        accepted = False
        difference = None
        actual_linear = None
        for backtracks in range(80):
            trial = q+alpha*direction
            for r in range(blocks):
                trial[4*r+3] = 1-math.fsum(map(float, trial[4*r:4*r+3]))
            if np.isfinite(trial).all() and (trial > 0).all():
                delta = trial-q
                actual_linear = math.fsum(float(a)*float(b) for a, b in zip(gradient, delta))
                difference = stable_delta(stats, q, delta, gradient)
                if actual_linear < 0 and difference <= 1e-4*actual_linear:
                    accepted = True
                    break
            alpha *= .5
        history.append({'iteration':iteration, 'q_before':q.tolist(),
                        'trial':trial.tolist(), 'KKT_before':residual,
                        'step':alpha, 'backtracks':backtracks, 'accepted':accepted,
                        'actual_delta':(trial-q).tolist(),
                        'actual_gradient_dot_delta':actual_linear,
                        'Armijo_RHS':None if actual_linear is None else 1e-4*actual_linear,
                        'stable_objective_difference':difference,
                        'scope':'last_trial_per_iteration;earlier_backtracks_not_serialized'})
        if not accepted:
            reason = 'positive Armijo step not found'
            break
        q = trial
    # Do not convert a failed numerical solve into an apparent fitted model.
    identity = np.tile(P, blocks)
    value, gradient, _ = evaluate(stats, identity)
    return {'status':'FAILED_RETURNED_IDENTITY', 'q':identity.tolist(),
            'iterations':iteration, 'projected_KKT':projected_kkt(gradient, blocks),
            'objective_native':value, 'reason':reason, 'history':history,
            'arithmetic':'analytic_difference_no_additive_c'}
