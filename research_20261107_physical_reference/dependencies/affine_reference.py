"""Exact finite reference frames for a known additive material contribution.

Classical translation invariance is retained at the representation level.
This module has no IO, fitting, random sampling or material-specific reader.
"""
from fractions import Fraction
import math


def rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, bool) or type(value).__name__ == 'bool_':
        raise ValueError('a boolean is not a physical numeric value')
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError('finite numeric value required')
    return Fraction.from_float(numeric)


def affine_law(residual_law, known_offset):
    law = {'family':'affine_empirical', 'coordinate':'native',
           'offset_exact_fraction':str(rational(known_offset)),
           'residual_law':residual_law,
           'unit':residual_law.get('unit'),
           'semantics':'known additive contribution plus modeled residual; no new measurement'}
    atoms(law)
    return law


def atoms(law):
    offset = Fraction(0)
    if law.get('family') == 'affine_empirical':
        offset = rational(law['offset_exact_fraction'])
        if law.get('coordinate') != 'native':
            raise ValueError('native law required')
        law = law['residual_law']
    if law.get('family') != 'weighted_empirical' or law.get('coordinate') != 'native':
        raise ValueError('finite weighted empirical residual law required')
    x, w = law.get('support_native'), law.get('weight_fractions')
    if not x or not isinstance(x, (list, tuple)) or not isinstance(w, (list, tuple)) or len(x) != len(w):
        raise ValueError('matched nonempty atom and mass arrays required')
    merged = {}
    for value, weight in zip(x, w):
        if not isinstance(weight, str):
            raise ValueError('exact mass strings required')
        mass = Fraction(weight)
        if mass < 0:
            raise ValueError('negative probability')
        if mass:
            value = rational(value)+offset
            merged[value] = merged.get(value, Fraction(0))+mass
    if not merged or sum(merged.values(), Fraction()) != 1:
        raise ValueError('exact total probability one required')
    return sorted(merged.items())


def half_pairwise(atoms_and_mass):
    """One half of E|X-X'| by a sorted prefix calculation."""
    count, weighted_sum, result = Fraction(), Fraction(), Fraction()
    for x, w in atoms_and_mass:
        result += w*(x*count-weighted_sum)
        count += w
        weighted_sum += w*x
    return result


def crps_exact(law, outcome):
    pairs = atoms(law)
    y = rational(outcome)
    return sum((w*abs(x-y) for x,w in pairs), Fraction())-half_pairwise(pairs)


def quantile_exact(law, probability):
    p = rational(probability)
    if not 0 < p < 1:
        raise ValueError('interior probability required')
    cumulative = Fraction()
    for value, weight in atoms(law):
        cumulative += weight
        if cumulative >= p:
            return value
    raise AssertionError('unit total mass must attain the requested quantile')


def gap_envelope(candidate, reference):
    """Exact all-real-outcome extrema of CRPS(candidate)-CRPS(reference)."""
    a, b = atoms(candidate), atoms(reference)
    am, bm = dict(a), dict(b)
    knots = sorted(set(am)|set(bm))
    value = sum((x*w for x,w in a), Fraction())-sum((x*w for x,w in b), Fraction())-half_pairwise(a)+half_pairwise(b)
    ca, cb = Fraction(), Fraction()
    values = []
    for i,x in enumerate(knots):
        values.append(value)
        ca += am.get(x, Fraction())
        cb += bm.get(x, Fraction())
        if i+1 < len(knots):
            value += 2*(ca-cb)*(knots[i+1]-x)
    return {'knots_exact_fraction':list(map(str, knots)),
            'knot_gaps_exact_fraction':list(map(str, values)),
            'lower_exact_fraction':str(min(values)),
            'upper_exact_fraction':str(max(values)),
            'domain':'all real outcomes for these stored exact affine laws'}


def gap_at(envelope, outcome):
    y = rational(outcome)
    x = list(map(Fraction, envelope['knots_exact_fraction']))
    d = list(map(Fraction, envelope['knot_gaps_exact_fraction']))
    if y <= x[0]: return d[0]
    for j in range(len(x)-1):
        if y <= x[j+1]:
            return d[j]+(d[j+1]-d[j])*(y-x[j])/(x[j+1]-x[j])
    return d[-1]
