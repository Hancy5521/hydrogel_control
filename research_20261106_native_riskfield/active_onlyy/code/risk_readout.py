"""Native-unit proper-score acquisition for one scalar NIG regression.

The posterior is never compressed. An exact affine risk integral plus a negative
YJ-coordinate correction avoids integrating a nearly quadratic positive branch.
Finite quadrature has an observed refinement diagnostic, not a numerical proof.
No source I/O, model fitting, observation ledger, or decision side effects.
"""
import math
import time
from functools import lru_cache
import numpy as np
from scipy import special

LEVELS = (32, 64, 128, 256)
V_CUT = 32.0
Z_CUT = 64.0
MEAN_REFINEMENT_ATOL = 2e-5


@lru_cache(maxsize=8)
def _legendre(count):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _interval(count, lower, upper):
    nodes, weights = _legendre(count)
    return (lower + upper) / 2 + (upper - lower) * nodes / 2, weights * (upper - lower) / 2


def _negative_correction(u):
    """h(u)=g'(u)-1-u/2 on u<0, zero otherwise; series near zero."""
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    small = (u < 0) & (np.abs(u) < 1e-4)
    # Binomial coefficients follow c_k=c_(k-1)*(3*k-2)/(2*k).
    coefficients = [.5]
    for k in range(2, 9):
        coefficients.append(coefficients[-1] * (3 * k - 2) / (2 * k))
    polynomial = np.zeros_like(u[small])
    for coefficient in reversed(coefficients[1:]):
        polynomial = coefficient + u[small] * polynomial
    out[small] = u[small] ** 2 * polynomial
    mask = (u < 0) & ~small
    out[mask] = np.expm1(-np.log1p(-1.5 * u[mask]) / 3) - .5 * u[mask]
    if (out < 0).any() or not np.isfinite(out).all():
        raise ArithmeticError('nonnegative finite inverse-YJ correction required')
    return out


def _derivative(u):
    u = np.asarray(u, dtype=float)
    result = np.empty_like(u)
    positive = u >= 0
    result[positive] = 1 + .5 * u[positive]
    result[~positive] = np.exp(-np.log1p(-1.5 * u[~positive]) / 3)
    return result


def _student_affine_kernel_integral(nu, rho):
    def constant(df):
        return 2 * math.sqrt(df) / (df - 1) * math.exp(
            special.betaln(.5, df - .5) - 2 * special.betaln(.5, df / 2))
    factor = math.sqrt(nu / (nu + 1)) * math.exp(
        special.betaln(.5, (nu - 1) / 2) - special.betaln(.5, nu / 2))
    result = constant(nu) - constant(nu + 1) * factor * np.sqrt(1 - rho ** 2)
    if (result <= 0).any():
        raise ArithmeticError('finite-df shared-scale risk must be positive')
    return result


def _kernel(v, rho, nu, family, count):
    """Return [query, v] CDF-variance kernel; rho may be signed."""
    rho = np.asarray(rho, dtype=float)
    v = np.asarray(v, dtype=float)
    if family == 'student_t':
        angle, w = _interval(count, -math.atan(Z_CUT / math.sqrt(nu)),
                             math.atan(Z_CUT / math.sqrt(nu)))
        c, s = np.cos(angle), np.sin(angle)
        w = w * np.exp((nu - 1) * np.log(c) - special.betaln(.5, nu / 2))
        argument = np.sqrt((nu + 1) / (nu * (1 - rho ** 2)))[:, None, None] * (
            v[None, :, None] * c[None, None, :]
            - rho[:, None, None] * math.sqrt(nu) * s[None, None, :])
        # By symmetry K(-v)=K(v); the caller only needs v>=0 for this branch.
        if (v < 0).any():
            raise ValueError('kernel requires absolute target coordinates')
        diff = special.stdtr(nu + 1, -argument) - special.stdtr(nu, -v)[None, :, None]
        result = (diff * diff) @ w
    elif family == 'normal':
        c = rho ** 2
        first = special.ndtr(v) * special.ndtr(-v)
        second = 2 * special.owens_t(v[None, :], np.sqrt((1 - c) / (1 + c))[:, None])
        result = first[None, :] - second
        tolerance = 64 * np.finfo(float).eps * (first[None, :] + np.abs(second))
        # Evaluate the positive Plackett integral rather than clipping cancellation.
        for j in range(len(rho)):
            fallback = result[j] <= tolerance[j]
            if fallback.any():
                if c[j] == 0:
                    result[j, fallback] = 0
                else:
                    angle, w = _interval(count, 0., math.asin(float(c[j])))
                    result[j, fallback] = np.exp(-v[fallback, None] ** 2 /
                                                (1 + np.sin(angle)[None, :])) @ w / (2 * math.pi)
    else:
        raise ValueError('student_t or normal family required')
    if not np.isfinite(result).all() or (result < 0).any():
        raise ArithmeticError('finite nonnegative kernel required')
    return result


def _tail(mu, scale, s0, nu, family, positive_decomposition):
    # Positive upper bounds deliberately omit negative terms of the tail moments.
    if family == 'student_t':
        logpdf = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
                  - .5 * math.log(nu * math.pi)
                  - (nu + 1) / 2 * math.log1p(V_CUT ** 2 / nu))
        density, sf = math.exp(logpdf), float(special.stdtr(nu, -V_CUT))
        i0 = (nu + V_CUT ** 2) / (nu - 1) * density
        i1 = .5 * (nu * sf + V_CUT * (nu + V_CUT ** 2) * density) / (nu - 2)
        query_probability = 2 * float(special.stdtr(nu, -Z_CUT))
    else:
        density = math.exp(-.5 * V_CUT ** 2) / math.sqrt(2 * math.pi)
        sf = float(special.ndtr(-V_CUT))
        i0, i1 = density, .5 * (V_CUT * density + sf)
        query_probability = 0.
    if positive_decomposition:
        outer = .5 * s0 * scale * (abs(mu) * i0 + scale * i1)
        inner = query_probability * s0 * scale * (abs(mu) * V_CUT + .5 * scale * V_CUT ** 2)
    else:
        outer = 2 * s0 * scale * ((1 + .5 * abs(mu)) * i0 + .5 * scale * i1)
        inner = query_probability * s0 * scale * (2 * (1 + .5 * abs(mu)) * V_CUT + .5 * scale * V_CUT ** 2)
    return outer + inner


def pair_values(mu, scale, s0, nu, rho, family, count):
    """All query correlations for one target; mu/scale in pre-affine YJ U."""
    rho = np.asarray(rho, dtype=float)
    if (not all(math.isfinite(float(x)) for x in (mu, scale, s0, nu))
            or min(scale, s0) <= 0 or nu <= 2 or rho.ndim != 1
            or not np.isfinite(rho).all() or (np.abs(rho) >= 1).any()):
        raise ValueError('finite nonsingular predictive law required')
    if family == 'normal':
        scale *= math.sqrt(nu / (nu - 2))
        # Stable 1-sqrt(1-rho^2), including exact rho0 independence.
        k0 = rho ** 2 / (1 + np.sqrt(1 - rho ** 2)) / math.sqrt(math.pi)
    elif family == 'student_t':
        k0 = _student_affine_kernel_integral(nu, rho)
    else:
        raise ValueError('student_t or normal required')
    decomposition = mu >= 0
    if decomposition:
        result = s0 * scale * (1 + .5 * mu) * k0
        a = mu / scale
        if a < V_CUT:
            v, w = _interval(count, a, V_CUT)
            result += s0 * scale * (_kernel(v, rho, nu, family, count)
                                     @ (w * _negative_correction(mu - scale * v)))
    else:
        # Preserve a positive direct integral when the affine decomposition would
        # subtract large terms. This is a numerical branch, never a data filter.
        split = sorted(set([-V_CUT, 0., V_CUT] +
                           ([-mu / scale] if abs(mu / scale) < V_CUT else [])))
        result = np.zeros(len(rho))
        for lower, upper in zip(split[:-1], split[1:]):
            v, w = _interval(count, lower, upper)
            result += s0 * scale * (_kernel(np.abs(v), rho, nu, family, count)
                                     @ (w * _derivative(mu + scale * v)))
    if not np.isfinite(result).all() or (result < 0).any():
        raise ArithmeticError('nonnegative finite native expected gain required')
    return result, _tail(mu, scale, s0, nu, family, decomposition)


def risk_values(nig, pool_z, terminal_z, transform, family='student_t'):
    """Mean target native-CRPS improvement for each distinct available query.

    All inputs are current-state summaries and already-known contexts. The
    Gaussian control matches latent marginal means/variances/correlation, then
    uses the same nonlinear native loss. It does not match native moments.
    """
    start = time.perf_counter()
    pool, terminal = np.asarray(pool_z, dtype=float), np.asarray(terminal_z, dtype=float)
    mean, covariance = np.asarray(nig['mean'], dtype=float), np.asarray(nig['V'], dtype=float)
    if (pool.ndim != 2 or terminal.ndim != 2 or len(pool) == 0 or len(terminal) == 0
            or pool.shape[1] != len(mean) - 1 or terminal.shape[1] != len(mean) - 1
            or not np.isfinite(pool).all() or not np.isfinite(terminal).all()
            or covariance.shape != (len(mean), len(mean))
            or not np.isfinite(covariance).all() or nig['alpha'] <= 1 or nig['b'] <= 0):
        raise ValueError('finite nonempty p-dimensional contexts and NIG state required')
    np.linalg.cholesky(covariance)
    q, t = np.column_stack((np.ones(len(pool)), pool)), np.column_stack((np.ones(len(terminal)), terminal))
    qdiag = 1 + np.einsum('ij,jk,ik->i', q, covariance, q)
    tdiag = 1 + np.einsum('ij,jk,ik->i', t, covariance, t)
    rho = (q @ covariance @ t.T) / np.sqrt(qdiag[:, None] * tdiag[None, :])
    nu = 2 * float(nig['alpha'])
    sd, s0, center = (float(transform[key]) for key in ('sd', 's0', 'center'))
    if min(sd, s0) <= 0 or not all(math.isfinite(x) for x in (sd, s0, center)):
        raise ValueError('fixed finite warm response transform required')
    mu = center + sd * (t @ mean)
    scale = sd * np.sqrt(nig['b'] / nig['alpha'] * tdiag)
    previous = None
    history = []
    for count in LEVELS:
        totals = np.zeros(len(pool))
        tail = 0.
        for j in range(len(terminal)):
            value, bound = pair_values(float(mu[j]), float(scale[j]), s0, nu, rho[:, j], family, count)
            totals += value / len(terminal)
            tail += bound / len(terminal)
        change = None if previous is None else float(np.max(np.abs(totals - previous)))
        history.append({'nodes_per_segment': count, 'max_mean_native_change': change})
        if previous is not None and change <= MEAN_REFINEMENT_ATOL:
            break
        previous = totals.copy()
    if change is None or change > MEAN_REFINEMENT_ATOL:
        raise ArithmeticError('native risk readout failed fixed refinement diagnostic; no labels may be queried')
    return {'values': totals.tolist(), 'diagnostics': {
        'family': family, 'target_count': len(terminal), 'query_count': len(pool),
        'nu': nu, 'levels_evaluated': history, 'mean_tail_bound_ideal_arithmetic': tail,
        'max_abs_rho': float(np.max(np.abs(rho))),
        'positive_branch_decomposition_targets': int((mu >= 0).sum()),
        'negative_mean_direct_integral_targets': int((mu < 0).sum()),
        'elapsed_seconds': time.perf_counter() - start,
        'mean_over_all_targets_not_sum': True,
        'full_NIG_posterior_unchanged': True,
        'refinement_is_diagnostic_not_rigorous_quadrature_or_IEEE_bound': True,
        'normal_is_latent_momentmatched_then_transported': family == 'normal'}}
