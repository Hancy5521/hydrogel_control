"""Query-learned finite-CDF risk code with a frozen empirical reference.

One convex SLSQP call per fit, exact output feasibility, exact adaptive
without-replacement residual totals, and a positive-probability query policy.
No file IO, data download, forecast fitting or hidden label acquisition here.
The projected total is a point estimator; it is not a confidence sequence.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from math import gcd, isfinite
import numpy as np
from scipy.optimize import minimize

from cdf_geometry import (P, EPSILON, ZERO, rational, simplex_theta, theta_beta,
                          dot, matvec, matrix_add, matrix_scale, zeros,
                          range_geometry, vector_add, rref)

TAU = F(1, 4)
SOLVER_OPTIONS = {'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
RAW_SUM_TOL = 1e-8
RAW_NEG_TOL = 1e-10
ACTIVE_TOL = 1e-8
KKT_STATIONARITY_TOL = 1e-6
KKT_DUAL_TOL = 1e-6
KKT_COMPLEMENTARITY_TOL = 1e-7
OBJECTIVE_TOL = 1e-10


def aggregate_Q(geometries):
    geometries = tuple(geometries)
    if not geometries:
        raise ValueError('Nonempty remaining population required')
    out = zeros(geometries[0].dimension)
    for geo in geometries:
        out = matrix_add(out, geo.Q)
    return out


def aggregate_c(geometries, targets):
    geometries, targets = tuple(geometries), tuple(targets)
    if not geometries or len(geometries) != len(targets):
        raise ValueError('Nonempty matched true targets required')
    result = (ZERO,)*geometries[0].dimension
    for geo, target in zip(geometries, targets):
        result = vector_add(result, geo.c(target))
    return result


def identity_theta(dimension=6):
    if dimension not in (3, 6):
        raise ValueError('Three or six coefficients required')
    return P*(dimension//3)


def _s_from_theta(theta):
    theta = simplex_theta(theta)
    return tuple((t-EPSILON*P[j % 4])/(1-EPSILON) for j, t in enumerate(theta))


def _theta_from_float_s(raw):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 1 or len(raw) not in (4, 8) or not np.isfinite(raw).all():
        raise ValueError('Finite four/eight solver coordinates required')
    errors = [abs(float(np.sum(raw[j:j+4]))-1) for j in range(0, len(raw), 4)]
    if max(errors) > RAW_SUM_TOL or float(np.min(raw)) < -RAW_NEG_TOL:
        raise ValueError('Raw simplex feasibility tolerance exceeded')
    cleaned = []
    for j in range(0, len(raw), 4):
        block = [F(str(max(0.0, float(x)))) for x in raw[j:j+4]]
        total = sum(block, ZERO)
        if not total:
            raise ValueError('Zero simplex after clipping')
        cleaned.extend(x/total for x in block)
    theta = tuple(EPSILON*P[j % 4]+(1-EPSILON)*s for j, s in enumerate(cleaned))
    return simplex_theta(theta), tuple(cleaned), {'sum_errors': errors,
            'minimum_raw_s': float(np.min(raw)),
            'maximum_postprocess_change': max(abs(float(s)-float(x))
                                               for s, x in zip(cleaned, raw))}


def _objective_parts(Q, c, prior, count, penalty_Q=None):
    q = np.asarray([[float(v) for v in row] for row in Q], dtype=float)
    reg = q if penalty_Q is None else np.asarray(
        [[float(v) for v in row] for row in penalty_Q], dtype=float)
    cc = np.asarray([float(v) for v in c], dtype=float)
    bp = np.asarray([float(v) for v in theta_beta(prior)], dtype=float)
    d, b = len(cc), len(cc)//3
    transform = np.zeros((d, 4*b))
    offset = np.tile(np.asarray([float(x) for x in P[:3]]), b)
    for k in range(b):
        for j in range(3):
            transform[3*k+j, 4*k+j] = float(1-EPSILON)
    constant = float(EPSILON)*offset-offset
    if not all(np.isfinite(a).all() for a in (q, reg, cc, bp)):
        raise ArithmeticError('Objective not representable as finite floats')

    def value_gradient(s):
        beta = transform @ np.asarray(s)+constant
        difference = beta-bp
        value = (beta@q@beta+2*beta@cc+float(TAU)*(difference@reg@difference))/count
        gradient = transform.T@(2*(q@beta+cc+float(TAU)*(reg@difference)))/count
        return float(value), gradient
    return value_gradient, q, reg


def _kkt(s, gradient, scale):
    """Fixed approximate convex-QP diagnostic on processed feasible s."""
    stations, duals, comps, multipliers = [], [], [], []
    for start in range(0, len(s), 4):
        ss, gg = s[start:start+4], gradient[start:start+4]
        free = ss > ACTIVE_TOL
        if not np.any(free):
            raise ArithmeticError('A normalized simplex has no free coordinate')
        lam = float(np.mean(gg[free]))
        stations.append(float(np.max(np.abs(gg[free]-lam))))
        duals.append(float(max(0.0, np.max(lam-gg[~free]))) if np.any(~free) else 0.0)
        comps.append(float(np.max(np.abs(ss*(gg-lam)))))
        multipliers.append(lam)
    residuals = {'stationarity': max(stations), 'active_dual': max(duals),
                 'complementarity': max(comps), 'gradient_scale': scale,
                 'block_multipliers': multipliers,
                 'active_threshold': ACTIVE_TOL}
    residuals['passed'] = (residuals['stationarity'] <= KKT_STATIONARITY_TOL*scale
        and residuals['active_dual'] <= KKT_DUAL_TOL*scale
        and residuals['complementarity'] <= KKT_COMPLEMENTARITY_TOL*scale)
    return residuals


def _exact_face_polish(Q, c, prior, count, initial_s, penalty_Q=None):
    """One rational KKT solve on the once-SLSQP selected face; no face search.

    A consistent singular system is anchored by setting free correction
    variables to zero in RREF. This does not assert orthogonal kernel retention.
    A wrong face, negative exact solution or negative dual multiplier fails.
    The exact candidate is a certificate; the used result is rounded separately.
    """
    n, d, scale = len(initial_s), len(c), 1-EPSILON
    blocks = n//4
    reg = Q if penalty_Q is None else penalty_Q
    # beta=A*s+b0. The fourth coordinate is retained through simplex equality.
    A = tuple(tuple(scale if j == 3*(k//4)+k % 4 and k % 4 < 3 else ZERO
                    for k in range(n)) for j in range(d))
    b0 = tuple(-scale*P[j % 3] for j in range(d))
    bprior = theta_beta(prior)
    total = matrix_add(Q, matrix_scale(reg, TAU))
    hessian = tuple(tuple(2*sum((A[j][k]*total[j][ell]*A[ell][m]
                                for j in range(d) for ell in range(d)), ZERO)/count
                           for m in range(n)) for k in range(n))
    constant_beta_gradient = vector_add(matvec(Q, b0), c)
    constant_beta_gradient = vector_add(constant_beta_gradient,
        tuple(TAU*v for v in matvec(reg, tuple(a-b for a, b in zip(b0, bprior)))))
    linear = tuple(2*sum((A[j][k]*constant_beta_gradient[j] for j in range(d)), ZERO)/count
                   for k in range(n))
    free = tuple(j for j, value in enumerate(initial_s) if float(value) > ACTIVE_TOL)
    active = tuple(j for j in range(n) if j not in free)
    if any(not any(j//4 == b for j in free) for b in range(blocks)):
        raise ArithmeticError('Selected face lacks a coordinate in one simplex')
    s0 = tuple(initial_s[j] if j in free else ZERO for j in range(n))
    gradient0 = vector_add(matvec(hessian, s0), linear)
    system, rhs = [], []
    for j in free:
        system.append(tuple(hessian[j][k] for k in free)
                      +tuple(F(-1) if j//4 == b else ZERO for b in range(blocks)))
        rhs.append(-gradient0[j])
    for b in range(blocks):
        system.append(tuple(F(1) if j//4 == b else ZERO for j in free)+(ZERO,)*blocks)
        rhs.append(F(1)-sum((s0[j] for j in free if j//4 == b), ZERO))
    augmented = tuple(tuple(row)+(value,) for row, value in zip(system, rhs))
    reduced, pivots = rref(augmented)
    variables = len(free)+blocks
    if variables in pivots:
        raise ArithmeticError('Selected-face exact KKT system inconsistent')
    correction = [ZERO]*variables
    for row, pivot in enumerate(pivots):
        correction[pivot] = reduced[row][-1]
    polished = list(s0)
    for j, delta in zip(free, correction[:len(free)]):
        polished[j] += delta
    polished = tuple(polished)
    multipliers = tuple(correction[len(free):])
    gradient = vector_add(matvec(hessian, polished), linear)
    primal = (all(v >= 0 for v in polished)
              and all(sum(polished[4*b:4*b+4], ZERO) == 1 for b in range(blocks)))
    stationarity = all(gradient[j] == multipliers[j//4] for j in free)
    dual = all(gradient[j] >= multipliers[j//4] for j in active)
    theta = tuple(EPSILON*P[j % 4]+scale*s for j, s in enumerate(polished))
    beta = theta_beta(theta) if primal else None
    prior_beta = theta_beta(prior)
    def value(v):
        diff = tuple(a-b for a, b in zip(v, prior_beta))
        return (dot(v, matvec(Q, v))+2*dot(v, c)+TAU*dot(diff, matvec(reg, diff)))/count
    objective = value(beta) if primal else None
    prior_objective = value(prior_beta)
    exact_passed = primal and stationarity and dual and objective <= prior_objective
    certificate = {'selected_free_coordinates': free, 'selected_active_coordinates': active,
        'linear_system_rank': len(pivots), 'linear_system_variables': variables,
        'anchor': 'RREF free correction coordinates zero; no alternative face/restart',
        'exact_s': tuple(map(str, polished)), 'exact_theta': tuple(map(str, theta)),
        'exact_gradient': tuple(map(str, gradient)), 'exact_block_multipliers': tuple(map(str, multipliers)),
        'exact_primal_passed': primal, 'exact_stationarity_passed': stationarity,
        'exact_dual_passed': dual, 'exact_objective': None if objective is None else str(objective),
        'exact_prior_objective': str(prior_objective), 'exact_passed': exact_passed}
    if not exact_passed:
        return None, certificate
    return polished, certificate


@dataclass(frozen=True)
class FitResult:
    theta: tuple
    proposed_theta: object
    accepted: bool
    diagnostics: dict


def fit_code(Q, c_estimate, prior_theta, remaining_count, *, penalty_Q=None):
    """Once-only constrained convex risk fit; failure returns declared prior.

    Main use: Q=remaining Gram, c_estimate=adaptive remaining-total estimate.
    penalty_Q is only for the explicitly named ordinary queried-ERM comparator.
    Both forms use fixed tau=1/4 and retain all four/eight feasible variables.
    """
    prior = simplex_theta(prior_theta)
    d = 3*(len(prior)//4)
    Q = tuple(tuple(map(rational, row)) for row in Q)
    c = tuple(map(rational, c_estimate))
    if len(Q) != d or any(len(row) != d for row in Q) or len(c) != d:
        raise ValueError('Gram/statistic/prior dimension mismatch')
    if not isinstance(remaining_count, int) or isinstance(remaining_count, bool) or remaining_count <= 0:
        raise ValueError('Positive integer remaining count required')
    if penalty_Q is not None:
        penalty_Q = tuple(tuple(map(rational, row)) for row in penalty_Q)
        if len(penalty_Q) != d or any(len(row) != d for row in penalty_Q):
            raise ValueError('Penalty Gram dimension mismatch')
    rg = range_geometry(Q)
    projected_c = matvec(rg['projector'], c)
    objective, q_float, reg_float = _objective_parts(Q, projected_c, prior,
                                                   remaining_count, penalty_Q)
    prior_s = np.asarray([float(v) for v in _s_from_theta(prior)])
    prior_value, prior_gradient = objective(prior_s)
    diagnostics = {'tau': str(TAU), 'remaining_count': remaining_count,
        'rank_exact': rg['rank'], 'nullspace_exact': [[str(v) for v in row]
            for row in rg['nullspace']], 'c_input_exact': list(map(str, c)),
        'c_projected_exact': list(map(str, projected_c)),
        'projector_exact': [[str(v) for v in row] for row in rg['projector']],
        'range_projection_is_confidence_sequence': False,
        'optimizer_calls': 0, 'solver_options': dict(SOLVER_OPTIONS),
        'prior_objective': prior_value,
        'minimum_gram_eigenvalue_float_diagnostic': float(np.linalg.eigvalsh(q_float).min()),
        'minimum_penalty_eigenvalue_float_diagnostic': float(np.linalg.eigvalsh(reg_float).min()),
        'rank_method': rg['rank_method']}
    # Exact Q is constructed as a Gram. Eigenvalues are diagnostics, never ranks.
    equalities = []
    for k in range(len(prior)//4):
        row = np.zeros(len(prior_s)); row[4*k:4*k+4] = 1.0
        equalities.append({'type': 'eq', 'fun': lambda s, a=row: a@s-1.0,
                           'jac': lambda s, a=row: a})
    proposed = None
    try:
        diagnostics['optimizer_calls'] = 1
        result = minimize(lambda s: objective(s)[0], prior_s, method='SLSQP',
                          jac=lambda s: objective(s)[1],
                          bounds=[(0.0, 1.0)]*len(prior_s),
                          constraints=equalities, options=dict(SOLVER_OPTIONS))
        diagnostics.update({'solver_success': bool(result.success),
            'solver_status': int(result.status), 'solver_message': str(result.message),
            'iterations': int(result.nit), 'raw_s': result.x.tolist(),
            'raw_objective': float(result.fun)})
        if not result.success:
            diagnostics['fallback_reason'] = 'solver_unsuccessful'
            return FitResult(prior, proposed, False, diagnostics)
        slsqp_theta, processed_s, feasibility = _theta_from_float_s(result.x)
        diagnostics['slsqp_processed_theta'] = tuple(map(str, slsqp_theta))
        diagnostics['slsqp_postprocess'] = feasibility
        polished_s, certificate = _exact_face_polish(Q, projected_c, prior,
            remaining_count, processed_s, penalty_Q)
        diagnostics['exact_face_polish'] = certificate
        if polished_s is None:
            diagnostics['fallback_reason'] = 'single_selected_face_exact_kkt_failed'
            return FitResult(prior, slsqp_theta, False, diagnostics)
        # Prevent the exact adaptive total's growing denominator from entering
        # the NEXT proposal. Reapply the original finite roundtrip contract.
        proposed, processed_s, feasibility = _theta_from_float_s(
            np.asarray([float(v) for v in polished_s]))
        diagnostics['maximum_exact_to_used_s_change'] = max(
            abs(float(a-b)) for a, b in zip(polished_s, processed_s))
        ss = np.asarray([float(v) for v in processed_s])
        value, gradient = objective(ss)
        scale = max(1.0, float(np.max(np.abs(prior_gradient))),
                    float(np.max(np.abs(gradient))))
        diagnostics.update({'postprocess': feasibility, 'postprocessed_objective': value,
            'objective_change_vs_prior': value-prior_value,
            'kkt': _kkt(ss, gradient, scale)})
        if not (isfinite(value) and np.isfinite(gradient).all()):
            diagnostics['fallback_reason'] = 'nonfinite_postprocessed_objective_or_gradient'
        elif not diagnostics['kkt']['passed']:
            diagnostics['fallback_reason'] = 'postprocessed_kkt_failed'
        elif value > prior_value+OBJECTIVE_TOL*max(1.0, abs(prior_value)):
            diagnostics['fallback_reason'] = 'postprocessed_objective_worse_than_prior'
        else:
            diagnostics['accepted'] = True
            return FitResult(proposed, proposed, True, diagnostics)
    except (ValueError, ArithmeticError, FloatingPointError, OverflowError) as exc:
        diagnostics['fallback_reason'] = type(exc).__name__+': '+str(exc)
    diagnostics['accepted'] = False
    return FitResult(prior, proposed, False, diagnostics)


def fit_query_erm(queried_geometries, queried_targets, remaining_Q,
                  prior_theta, remaining_count):
    """Unit-weight queried CRPS mean, SAME prior/tau and remaining-Gram penalty.

    This deliberately has no inverse-probability correction. Adaptively selected
    queries make it ordinary queried ERM, not an unbiased active-ERM estimator.
    """
    geos, targets = tuple(queried_geometries), tuple(queried_targets)
    if not geos or len(geos) != len(targets):
        raise ValueError('Matched nonempty queried observations required')
    scale = F(remaining_count, len(geos))
    q = matrix_scale(aggregate_Q(geos), scale)
    c = tuple(scale*v for v in aggregate_c(geos, targets))
    result = fit_code(q, c, prior_theta, remaining_count, penalty_Q=remaining_Q)
    result.diagnostics['comparator'] = 'ordinary unit-weight queried ERM; not unbiased active ERM'
    result.diagnostics['queried_count'] = len(geos)
    return result


def mixture_probabilities(widths, gaps):
    """Fixed 1/4 uniform + 3/8 c-width + 3/8 current gap-width mixture."""
    widths, gaps = tuple(map(rational, widths)), tuple(map(rational, gaps))
    n = len(widths)
    if not n or len(gaps) != n or any(v < 0 for v in widths+gaps):
        raise ValueError('Nonempty matched nonnegative policy weights required')
    sw, sg, uniform = sum(widths, ZERO), sum(gaps, ZERO), F(1, n)
    q = tuple(F(1, 4)*uniform+F(3, 8)*(w/sw if sw else uniform)
              +F(3, 8)*(g/sg if sg else uniform) for w, g in zip(widths, gaps))
    if sum(q, ZERO) != 1 or min(q) < F(1, 4*n):
        raise ArithmeticError('Positive proposal lower bound failed')
    return q


def draw_rational(q, rng):
    """Exact integer-ticket realization using a caller-owned seeded PRNG.

    Probability proofs use ideal random tickets. A finite-state PRNG is the
    reproducible implementation, not a claim of cryptographic ideal randomness.
    """
    q = tuple(map(rational, q))
    if not q or any(v <= 0 for v in q) or sum(q, ZERO) != 1:
        raise ValueError('Positive exact normalized proposal required')
    denominator = 1
    for value in q:
        denominator = denominator//gcd(denominator, value.denominator)*value.denominator
    ticket = rng.randrange(denominator)
    cumulative = 0
    for index, value in enumerate(q):
        cumulative += value.numerator*(denominator//value.denominator)
        if ticket < cumulative:
            return index, {'ticket': str(ticket), 'ticket_total': str(denominator)}
    raise ArithmeticError('Exact sampling did not exhaust tickets')


class QueryState:
    """Frozen finite-population geometry; observation follows a committed draw.

    The caller owns the true-label vault and exposes only the selected label.
    This state never accepts a vector of unqueried true labels. Every batch must
    explicitly commit its current theta before any next draw.
    """
    def __init__(self, geometries):
        self.geometries = tuple(geometries)
        if not self.geometries or len({g.dimension for g in self.geometries}) != 1:
            raise ValueError('Matched nonempty frozen geometries required')
        self.dimension = self.geometries[0].dimension
        self.remaining = list(range(len(self.geometries)))
        self.midpoints = tuple(tuple((lo+hi)/2 for lo, hi in g.c_bounds)
                               for g in self.geometries)
        self.widths = tuple(max(hi-lo for lo, hi in g.c_bounds) for g in self.geometries)
        self.M = tuple(sum((m[j] for m in self.midpoints), ZERO)
                       for j in range(self.dimension))
        self.past_residual = (ZERO,)*self.dimension
        self.sum_c = (ZERO,)*self.dimension
        self.sum_Z = (ZERO,)*self.dimension
        self.records = []
        self.pending = None
        self.batch_theta = None
        self.batch_draws = 0
        self.gaps = None

    def start_batch(self, theta):
        if self.pending is not None:
            raise RuntimeError('A pending committed query must be observed first')
        if self.batch_theta is not None and self.batch_draws != 32:
            raise RuntimeError('Fixed policy batches contain exactly 32 observations')
        self.batch_theta = simplex_theta(theta, self.dimension//3)
        envelopes = tuple(g.gap_bounds(self.batch_theta) for g in self.geometries)
        self.gaps = tuple(hi-lo for lo, hi in envelopes)
        self.batch_draws = 0

    def draw(self, rng):
        if self.batch_theta is None or self.batch_draws >= 32 or self.pending is not None:
            raise RuntimeError('Start a batch or observe the pending query before drawing')
        q = mixture_probabilities([self.widths[i] for i in self.remaining],
                                  [self.gaps[i] for i in self.remaining])
        position, ticket = draw_rational(q, rng)
        index = self.remaining[position]
        self.pending = {'index': index, 'position': position, 'q': q[position],
                        'remaining_before': len(self.remaining),
                        'theta_before_query': tuple(map(str, self.batch_theta)), **ticket}
        return dict(self.pending)

    def observe(self, target):
        if self.pending is None:
            raise RuntimeError('True label must follow a committed random query')
        rec = dict(self.pending)
        index, q = rec['index'], rec['q']
        target = rational(target)
        c = self.geometries[index].c(target)
        residual = tuple(x-m for x, m in zip(c, self.midpoints[index]))
        Z = tuple(m+past+r/q for m, past, r in zip(self.M, self.past_residual, residual))
        self.sum_Z = vector_add(self.sum_Z, Z)
        self.sum_c = vector_add(self.sum_c, c)
        self.past_residual = vector_add(self.past_residual, residual)
        del self.remaining[rec['position']]
        self.batch_draws += 1
        rec.update({'target_exact': str(target), 'c_exact': tuple(map(str, c)),
                    'Z_total_exact': tuple(map(str, Z))})
        rec['q'] = str(q)
        self.records.append(rec)
        self.pending = None
        return rec

    def remaining_estimate(self):
        if self.pending is not None or not self.records:
            raise RuntimeError('Observed queries and no pending draw required')
        t = len(self.records)
        return tuple(z/t-c for z, c in zip(self.sum_Z, self.sum_c))

    def remaining_Q(self):
        return aggregate_Q(self.geometries[i] for i in self.remaining)
