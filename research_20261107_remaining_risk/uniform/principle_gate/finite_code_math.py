"""Pure exact finite-CDF mathematics for a six-handcase principle gate.

No fitting, optimization, label IO, network, material-data discovery or randomness.
This is a classical proper-score quadratic identity, not a new scoring rule.
It is deliberately a small validation helper, not the production material model.
"""
from fractions import Fraction as F


def exact(value):
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError('Use exact Fraction/int/rational string')
    if not isinstance(value, (F, int, str)):
        raise TypeError('Use exact Fraction/int/rational string')
    return F(value)


def law(pairs):
    result = {}
    for point, mass in pairs:
        point, mass = exact(point), exact(mass)
        if mass < 0:
            raise ValueError('Negative mass')
        if mass:
            result[point] = result.get(point, F()) + mass
    if not result or sum(result.values(), F()) != 1:
        raise ValueError('Positive exact total mass one is required')
    return tuple(sorted(result.items()))


def simplex(weights):
    weights = tuple(exact(value) for value in weights)
    if not weights or any(value < 0 for value in weights) or sum(weights, F()) != 1:
        raise ValueError('Simplex weights required')
    return weights


def mix(components, weights):
    weights = simplex(weights)
    if len(components) != len(weights):
        raise ValueError('Matched components/weights required')
    return law((point, weight * mass)
               for model, weight in zip(components, weights)
               for point, mass in law(model))


def crps_energy(model, target):
    """Independent finite-particle score; quadratic pairs, for tiny gates only."""
    model, target = law(model), exact(target)
    return (sum((mass * abs(point-target) for point, mass in model), F())
            - sum((a*b*abs(x-z) for x, a in model for z, b in model), F()) / 2)


def matrix_from_intervals(intervals, dimension):
    return tuple(tuple(sum((row['width'] * row['h'][j] * row['h'][k]
                            for row in intervals), F())
                       for k in range(dimension)) for j in range(dimension))


def geometry(components, reference_weights):
    components = tuple(law(model) for model in components)
    p = simplex(reference_weights)
    if len(components) != len(p) or len(p) < 2 or any(value <= 0 for value in p):
        raise ValueError('K>=2 matched components and strictly positive reference p')
    points = sorted({point for model in components for point, _ in model})
    dimension = len(p)-1
    intervals = []
    for left, right in zip(points[:-1], points[1:]):
        cdfs = tuple(sum((mass for point, mass in model if point <= left), F())
                     for model in components)
        intervals.append({'left': left, 'right': right, 'width': right-left,
                          'h': tuple(value-cdfs[-1] for value in cdfs[:-1]),
                          'reference_cdf': sum((a*b for a, b in zip(p, cdfs)), F())})
    result = {'components': components, 'reference_weights': p,
              'reference_law': mix(components, p), 'points': tuple(points),
              'dimension': dimension, 'intervals': tuple(intervals)}
    result['Q'] = matrix_from_intervals(intervals, dimension)
    result['c_bounds'] = bounds(result)
    return result


def c_vector(geo, target):
    target = exact(target)
    return tuple(sum((row['h'][j] *
                      (row['width'] * row['reference_cdf']
                       - max(F(), row['right']-max(row['left'], target)))
                      for row in geo['intervals']), F())
                 for j in range(geo['dimension']))


def bounds(geo):
    values = tuple(c_vector(geo, point) for point in geo['points'])
    result = tuple((min(row[j] for row in values), max(row[j] for row in values))
                   for j in range(geo['dimension']))
    if any(not lower <= 0 <= upper for lower, upper in result):
        raise ArithmeticError('Proper-score c bounds must contain zero')
    return result


def lift_context(geo, g):
    g = exact(g)
    if not 0 <= g <= 1:
        raise ValueError('Known g must lie in [0,1]')
    rows = tuple({**row, 'h': tuple((1-g)*value for value in row['h'])
                              + tuple(g*value for value in row['h'])}
                 for row in geo['intervals'])
    result = {**geo, 'dimension': 2*geo['dimension'], 'intervals': rows, 'g': g}
    result['Q'] = matrix_from_intervals(rows, result['dimension'])
    result['c_bounds'] = bounds(result)
    return result


def quadratic(geo, beta, c):
    beta, c = tuple(map(exact, beta)), tuple(map(exact, c))
    n = geo['dimension']
    if len(beta) != n or len(c) != n:
        raise ValueError('Matched coefficient/statistic dimensions')
    return (sum((beta[j]*geo['Q'][j][k]*beta[k]
                 for j in range(n) for k in range(n)), F())
            + 2*sum((a*b for a, b in zip(beta, c)), F()))


def robust_upper(geo, beta, box):
    beta = tuple(map(exact, beta))
    if len(beta) != geo['dimension'] or len(box) != len(beta):
        raise ValueError('Matched box/coefficient dimensions')
    box = tuple((exact(lower), exact(upper)) for lower, upper in box)
    if any(lower > upper for lower, upper in box):
        raise ValueError('Ordered box required')
    corner = tuple(upper if coefficient >= 0 else lower
                   for coefficient, (lower, upper) in zip(beta, box))
    return quadratic(geo, beta, corner)


def theta_from_beta(p, beta, epsilon=0):
    p, beta, epsilon = simplex(p), tuple(map(exact, beta)), exact(epsilon)
    if len(beta) != len(p)-1 or any(value <= 0 for value in p) or not 0 <= epsilon <= 1:
        raise ValueError('Positive reference, matched beta, epsilon in [0,1]')
    theta = tuple(value+change for value, change in zip(p[:-1], beta)) + (p[-1]-sum(beta, F()),)
    if any(value < epsilon*base for value, base in zip(theta, p)):
        raise ValueError('Coefficient outside the declared lower-simplex')
    return simplex(theta)


def rank_exact(matrix):
    a = [list(map(exact, row)) for row in matrix]
    if not a:
        return 0
    columns, rank = len(a[0]), 0
    for column in range(columns):
        pivot = next((r for r in range(rank, len(a)) if a[r][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        value = a[rank][column]
        a[rank] = [item/value for item in a[rank]]
        for row in range(len(a)):
            if row != rank and a[row][column]:
                factor = a[row][column]
                a[row] = [left-factor*right for left, right in zip(a[row], a[rank])]
        rank += 1
        if rank == len(a):
            break
    return rank
