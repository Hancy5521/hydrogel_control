"""Classical finite-relation posterior and centered continuous-risk controls.

This is a synthetic principle/computation evaluator, not a material learner or
a new Bayesian planning algorithm. Unsupported mixed measurement panels stop.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


class NumericResolutionError(ArithmeticError):
    pass


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, Fraction)):
        raise ValueError('Use exact integer/rational-string/Fraction probabilities')
    value = Fraction(value)
    if value < 0 or value > 1:
        raise ValueError('Probability outside[0,1]')
    return value


def prior_weights(values):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError('At least two declared finite relation hypotheses required')
    result = tuple(probability(value) for value in values)
    if sum(result, Fraction(0)) != 1:
        raise ValueError('Prior must be exactly normalized; no silent rescaling')
    return result


def to_float_exact_support(values):
    """Preserve exact branch state; refuse a numeric path that erases support."""
    exact = np.asarray(values, dtype=object)
    flat = []
    for value in exact.flat:
        x = float(value)
        if not math.isfinite(x) or (value != 0 and x == 0.):
            raise NumericResolutionError('Nonzero exact value cannot be represented in float64; no clipping/drop')
        flat.append(x)
    return np.asarray(flat, dtype=float).reshape(exact.shape)


@dataclass(frozen=True)
class Branch:
    outcome: int
    probability: Fraction
    posterior: tuple


def posterior_branches(prior, likelihood):
    w = prior_weights(prior)
    if not isinstance(likelihood, (list, tuple)) or not likelihood:
        raise ValueError('Explicit finite observation likelihood required')
    rows = tuple(tuple(probability(x) for x in row) for row in likelihood)
    if any(len(row) != len(w) for row in rows):
        raise ValueError('Likelihood shape is outcomes-by-hypotheses')
    if any(sum((row[h] for row in rows), Fraction(0)) != 1 for h in range(len(w))):
        raise ValueError('Every likelihood column must sum exactly to1')
    result = []
    for o, row in enumerate(rows):
        joint = tuple(wh * lh for wh, lh in zip(w, row))
        p = sum(joint, Fraction(0))
        if p == 0:  # Only an exactly zero-probability branch is omitted.
            continue
        result.append(Branch(o, p, tuple(value / p for value in joint)))
    if sum((branch.probability for branch in result), Fraction(0)) != 1:
        raise AssertionError('Posterior branches lost probability mass')
    return tuple(result)


def relation_covariance(prior, branches):
    w = prior_weights(prior)
    h = len(w)
    matrix = [[Fraction(0) for _ in w] for _ in w]
    if not branches or sum((b.probability for b in branches), Fraction(0)) != 1:
        raise ValueError('Complete exact posterior branches required')
    for branch in branches:
        if (not isinstance(branch, Branch) or branch.probability <= 0 or branch.probability > 1
                or len(branch.posterior) != h or prior_weights(branch.posterior) != branch.posterior):
            raise ValueError('Positive-probability normalized posterior branches required')
    mean = [sum((b.probability * b.posterior[j] for b in branches), Fraction(0)) for j in range(h)]
    if tuple(mean) != w:
        raise ValueError('Posterior martingale identity failed')
    for b in branches:
        if len(b.posterior) != h or sum(b.posterior, Fraction(0)) != 1:
            raise ValueError('Aligned normalized branch posterior required')
        delta = [p - q for p, q in zip(b.posterior, w)]
        for i in range(h):
            for j in range(h):
                matrix[i][j] += b.probability * delta[i] * delta[j]
    if any(sum(row, Fraction(0)) != 0 for row in matrix):
        raise AssertionError('F must annihilate the constant vector')
    return tuple(tuple(row) for row in matrix)


def check_psd(matrix):
    a = np.asarray(matrix, dtype=float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not len(a)
            or not np.isfinite(a).all() or not np.array_equal(a, a.T)):
        raise ValueError('Finite symmetric square matrix required')
    eig = np.linalg.eigvalsh(a)
    scale = float(np.linalg.norm(a, ord=2))
    tolerance = 64 * np.finfo(float).eps * len(a) * scale
    if float(eig.min()) < -tolerance:
        raise ArithmeticError('Matrix is not numerically PSD; no eigenvalue clipping')
    return {'minimum_eigenvalue': float(eig.min()), 'backward_error_tolerance': tolerance,
            'matrix_adjusted_or_projected': False}


def centered_crps_kernel(rhos, scale=1., weight=1.):
    rho = np.asarray(rhos, dtype=float)
    if (rho.ndim != 1 or len(rho) < 2 or not np.isfinite(rho).all()
            or (np.abs(rho) >= 1).any() or not math.isfinite(scale) or scale <= 0
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError('Finite open-interval correlations and positive target scale/weight required')
    products = np.outer(rho, rho)
    if np.any((rho[:, None] != 0) & (rho[None, :] != 0) & (products == 0)):
        raise NumericResolutionError('Correlation product underflow; no false zero kernel')
    # Stable 1-sqrt(1-t); no CDF*CDF tail-divergent integral and no clipping.
    kernel = (scale * weight / math.sqrt(math.pi)) * products / (1 + np.sqrt(1 - products))
    if np.any((products != 0) & (kernel == 0)):
        raise NumericResolutionError('Nonzero centered-kernel entry underflow; no silent clipping')
    check_psd(kernel)
    return kernel


def validate_scope(scope):
    wanted = {'finite_relation_model': True, 'conditional_independence_of_units_given_relation': True,
              'proxy_marginal_theta_invariant': True, 'common_target_marginal': True,
              'repair_observes_candidate_target': False,
              'generated_or_derived_values_treated_as_measurements': False,
              'declarations_are_model_assumptions_not_physical_certification': True}
    if not isinstance(scope, dict) or scope != wanted:
        raise ValueError('Unsupported/mixed panel: use a full joint planner instead of this pure subroutine')


def prepare(prior, probes, proxies, budget, scope):
    validate_scope(scope)
    w = prior_weights(prior)
    if type(budget) is not int or budget <= 0 or not probes or not proxies:
        raise ValueError('Positive integer budget and nonempty declared action sets required')
    if len({p['id'] for p in probes}) != len(probes) or len({p['id'] for p in proxies}) != len(proxies):
        raise ValueError('Unique action IDs required')
    for action in list(probes) + list(proxies):
        if not isinstance(action['id'], str) or not action['id'] or type(action['cost']) is not int or action['cost'] <= 0:
            raise ValueError('Named positive-cost actions required')
    branches = [posterior_branches(w, p['likelihood']) for p in probes]
    covariance = [relation_covariance(w, b) for b in branches]
    kernels = np.stack([centered_crps_kernel(p['rhos'], p['scale'], p.get('weight', 1.)) for p in proxies])
    if kernels.shape != (len(proxies), len(w), len(w)):
        raise ValueError('Proxy hypothesis count differs from posterior')
    wf = to_float_exact_support(w)
    ff = np.stack([to_float_exact_support(f) for f in covariance])
    feasible = np.asarray([[a['cost'] + b['cost'] <= budget for b in proxies] for a in probes])
    if not feasible.any():
        raise ValueError('No affordable repair-plus-proxy pair')
    return {'prior': w, 'prior_float': wf, 'branches': branches, 'F_exact': covariance,
            'F_float': ff, 'kernels': kernels, 'feasible': feasible,
            'probes': probes, 'proxies': proxies, 'budget': budget,
            'kernel_build_count': len(proxies), 'source_of_costs': 'declared synthetic integer ticks'}


def fixed_pair_values(prepared):
    w, s, f = prepared['prior_float'], prepared['kernels'], prepared['F_float']
    immediate = np.einsum('h,bhk,k->b', w, s, w)
    complementarity = np.einsum('ahk,bkh->ab', f, s)
    values = immediate[None, :] + complementarity
    return {'values': values, 'complementarity': complementarity,
            'proxy_alone_values': immediate,
            'feasible': prepared['feasible'].copy(),
            'meaning': 'precommitted fixed-pair value, not adaptive second-stage selection'}


def adaptive_loop_reference(prior, probes, proxies, budget, scope):
    """Explicit tree reference; speedups over this alone are not new-method evidence."""
    validate_scope(scope)
    w = prior_weights(prior)
    if type(budget) is not int or budget <= 0:
        raise ValueError('Positive integer budget required')
    if not probes or not proxies or any(type(a['cost']) is not int or a['cost'] <= 0 for a in list(probes) + list(proxies)):
        raise ValueError('Nonempty positive-integer-cost action sets required')
    values, best_by_outcome, calls = [], [], 0
    for a in probes:
        expected, choices = 0., []
        for branch in posterior_branches(w, a['likelihood']):
            v = to_float_exact_support(branch.posterior)
            scores = []
            for b in proxies:
                if a['cost'] + b['cost'] > budget:
                    scores.append(None)
                    continue
                s = centered_crps_kernel(b['rhos'], b['scale'], b.get('weight', 1.))
                calls += 1
                scores.append(float(v @ s @ v))
            legal = [j for j, score in enumerate(scores) if score is not None]
            if not legal:
                raise ValueError('Each compared first action needs a legal continuation')
            chosen = max(legal, key=lambda j: scores[j])
            probability_float = float(branch.probability)
            if probability_float == 0:
                raise NumericResolutionError('Positive outcome probability underflows; no branch drop')
            expected += probability_float * scores[chosen]
            choices.append({'outcome': branch.outcome, 'proxy_index': chosen})
        values.append(expected)
        best_by_outcome.append(choices)
    return {'values': np.asarray(values), 'choices': best_by_outcome, 'kernel_build_count': calls,
            'is_performance_claim_baseline': False}


def adaptive_cached(prepared, block_rows=None):
    """Strong fully cached/vectorized or blocked-vectorized adaptive baseline."""
    if block_rows is not None and (type(block_rows) is not int or block_rows < 1):
        raise ValueError('Positive block size orNone required')
    rows, owners, probabilities, outcomes = [], [], [], []
    for a, branches in enumerate(prepared['branches']):
        if not prepared['feasible'][a].any():
            raise ValueError('Each compared first action needs a legal continuation')
        for branch in branches:
            rows.append(to_float_exact_support(branch.posterior))
            owners.append(a)
            probabilities.append(branch.probability)
            outcomes.append(branch.outcome)
    w = np.asarray(rows)
    prob = to_float_exact_support(probabilities)
    outer = np.einsum('mh,mk->mhk', w, w).reshape(len(rows), -1)
    s = prepared['kernels'].reshape(len(prepared['proxies']), -1)
    size = len(rows) if block_rows is None else block_rows
    values = np.zeros(len(prepared['probes']))
    choices = [[] for _ in prepared['probes']]
    largest_score_array = 0
    for begin in range(0, len(rows), size):
        end = min(len(rows), begin + size)
        scores = outer[begin:end] @ s.T
        largest_score_array = max(largest_score_array, scores.nbytes)
        for local, i in enumerate(range(begin, end)):
            a = owners[i]
            candidates = np.flatnonzero(prepared['feasible'][a])
            chosen = int(candidates[np.argmax(scores[local, candidates])])
            values[a] += prob[i] * scores[local, chosen]
            choices[a].append({'outcome': outcomes[i], 'proxy_index': chosen})
    if not np.isfinite(values).all():
        raise ArithmeticError('Nonfinite adaptive value')
    return {'values': values, 'choices': choices,
            'kernel_build_count': prepared['kernel_build_count'],
            'score_matrix_peak_bytes': largest_score_array,
            'preprocessed_numeric_array_bytes': int(w.nbytes + outer.nbytes + s.nbytes + prepared['F_float'].nbytes + prepared['feasible'].nbytes),
            'cached_vectorization_is_classical_not_new_algorithm': True}
