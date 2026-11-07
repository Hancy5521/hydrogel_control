"""Pure finite-support native-CDF arithmetic. No file access or training.

Exact rational mass arithmetic is distinguished from floating CRPS statistics.
These elementary CRPS identities and CDF recalibration are established methods.
"""
from copy import deepcopy
from fractions import Fraction
import math
import numpy as np

BINS = (0., .05, .5, .95, 1.)
P = np.asarray((.05, .45, .45, .05), dtype=float)
P.setflags(write=False)
RATIONAL_BINS = tuple(Fraction(str(x)) for x in BINS)
RATIONAL_P = tuple(b-a for a,b in zip(RATIONAL_BINS[:-1], RATIONAL_BINS[1:]))


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' cannot be bool')
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(name + ' must be finite')
    return out


def canonical(law):
    """Sort/merge equal native supports; remove only exact zero mass."""
    if law.get('family') != 'weighted_empirical' or law.get('coordinate') != 'native':
        raise ValueError('native weighted_empirical law required')
    values, weights = law.get('support_native'), law.get('weight_fractions')
    if (not isinstance(values, (list, tuple)) or not isinstance(weights, (list, tuple))
            or not values or len(values) != len(weights)):
        raise ValueError('nonempty matched supports/Fraction-string weights required')
    merged = {}; zero = 0
    for value, weight in zip(values, weights):
        x = _finite(value, 'support')
        if not isinstance(weight, str):
            raise ValueError('weight_fractions must contain strings')
        w = Fraction(weight)
        if w < 0:
            raise ValueError('negative mass')
        if w == 0:
            zero += 1
        else:
            merged[x] = merged.get(x, Fraction(0)) + w
    if not merged or sum(merged.values(), Fraction(0)) != 1:
        raise ValueError('exact positive total mass1 required')
    support = tuple(sorted(merged))
    return {'support': support, 'mass': tuple(merged[x] for x in support),
            'original_rows': len(values), 'zero_mass_rows_removed': zero,
            'duplicates_merged': len(values)-zero-len(support)}


def basis(u):
    u = _finite(u, 'CDF probability')
    if not 0 <= u <= 1:
        raise ValueError('CDF outside[0,1]')
    return np.clip((u-np.asarray(BINS[:-1]))/P, 0., 1.)


def row_terms(law, target):
    """Q,b,c with CRPS(theta)=theta'Qtheta-2b'theta+c for every real y."""
    z = canonical(law); x, mass = z['support'], z['mass']
    y = _finite(target, 'past true target')
    Q = np.zeros((4,4)); b = np.zeros(4); v = Fraction(0)
    for j in range(len(x)-1):
        v += mass[j]
        gap = x[j+1]-x[j]
        if not math.isfinite(gap):
            raise ArithmeticError('native support gap not representable')
        h = basis(float(v))
        Q += gap*np.outer(h,h)
        length = max(0., x[j+1]-max(x[j],y))
        b += length*h
    c = abs(x[-1]-y)
    if not np.isfinite(Q).all() or not np.isfinite(b).all() or not math.isfinite(c):
        raise ArithmeticError('nonfinite native quadratic statistics')
    return {'Q':Q, 'b':b, 'c':c, 'reference_risk':bayes_risk(law)}


def bayes_risk(law):
    z=canonical(law); x,mass=z['support'],z['mass']
    v=Fraction(0); exact=Fraction(0)
    for j in range(len(x)-1):
        v += mass[j]
        gap=Fraction.from_float(x[j+1])-Fraction.from_float(x[j])
        exact += gap*v*(1-v)
    risk=float(exact)
    if not math.isfinite(risk) or (exact > 0 and risk == 0):
        raise ArithmeticError('reference risk not representable in float')
    return risk


def crps(law, target):
    """Stable nonnegative O(K) interval loss; sorting is separate O(K logK)."""
    z=canonical(law); x,mass=z['support'],z['mass']
    y=_finite(target,'target'); v=Fraction(0)
    terms=[max(x[0]-y,0.),max(y-x[-1],0.)]
    for j in range(len(x)-1):
        v += mass[j]
        left,right=x[j],x[j+1]
        below=max(0.,min(y,right)-left)
        above=max(0.,right-max(y,left))
        terms.extend((below*float(v*v), above*float((1-v)*(1-v))))
    out=math.fsum(terms)
    if not math.isfinite(out):
        raise ArithmeticError('CRPS not finite/representable')
    return out


def _rational_simplex(theta):
    q=np.asarray(theta,dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or (q<=0).any():
        raise ValueError('four strictly positive simplex parameters required')
    error=float(math.fsum(q.tolist())-1.)
    if abs(error)>1e-10:
        raise ValueError('parameter sum differs from1 beyond declared tolerance')
    vals=tuple(Fraction(str(float(a))) for a in q)
    total=sum(vals,Fraction(0))
    return tuple(a/total for a in vals), error


def apply_spline(law, theta0, theta1=None, g=None):
    """Positive PIT-overlap integration; never subtract nearly equal T values.

    Float simplex parameters are converted from their round-trip decimal strings
    and exactly normalized as Fractions. This tiny representation adjustment is
    recorded explicitly. Identity returns the original distribution verbatim.
    """
    z=canonical(law); t0,e0=_rational_simplex(theta0)
    if theta1 is None:
        if g is not None:
            raise ValueError('g requires theta1')
        t,e1=t0,None
    else:
        t1,e1=_rational_simplex(theta1)
        gv=_finite(g,'g')
        if not 0<=gv<=1:
            raise ValueError('g outside[0,1]')
        gr=Fraction(str(gv)); t=tuple((1-gr)*a+gr*b for a,b in zip(t0,t1))
    if t == RATIONAL_P:
        return deepcopy(law)
    v=Fraction(0); out=[]
    for w in z['mass']:
        right=v+w; updated=Fraction(0)
        for j in range(4):
            overlap=max(Fraction(0),min(right,RATIONAL_BINS[j+1])-max(v,RATIONAL_BINS[j]))
            updated += t[j]*overlap/RATIONAL_P[j]
        if updated<=0:
            raise ArithmeticError('strictly positive source atom lost')
        out.append(updated); v=right
    if sum(out,Fraction(0)) != 1:
        raise AssertionError('exact transported mass must sum1')
    return {'coordinate':'native','family':'weighted_empirical',
            'support_native':list(z['support']),'weight_fractions':[str(w) for w in out],
            'unit':law.get('unit'), 'transport':{'kind':'positive_overlap_piecewise_linear',
            'theta_effective_fractions':[str(a) for a in t],
            'float_parameter_sum_errors':[e0,e1],
            'rational_representation':'Fraction(str(float parameter)),then exact normalization',
            'canonicalization':{k:z[k] for k in ('original_rows','zero_mass_rows_removed','duplicates_merged')},
            'new_support_created':False}}


def _outward(value, direction):
    try:
        f=float(value)
    except OverflowError:
        return None
    if not math.isfinite(f):
        return None
    # One extra ULP is conservative even when conversion was already outward.
    out=float(np.nextafter(f, -np.inf if direction<0 else np.inf))
    return out if math.isfinite(out) else None


def loss_gap_envelope(reference, candidate):
    """Exact rational all-real-y extrema of CRPS(candidate)-CRPS(reference).

    Exactness is relative to stored finite binary supports and rational masses,
    not to uncertain original measurement precision or constant bit complexity.
    """
    f=canonical(reference); g=canonical(candidate)
    knots=sorted(set(f['support'])|set(g['support']))
    fm=dict(zip(f['support'],f['mass'])); gm=dict(zip(g['support'],g['mass']))
    fp=gp=Fraction(0); pref=[]
    for x in knots:
        fp += fm.get(x,Fraction(0)); gp += gm.get(x,Fraction(0)); pref.append((fp,gp))
    value=Fraction(0)
    for j in range(len(knots)-1):
        gap=Fraction.from_float(knots[j+1])-Fraction.from_float(knots[j])
        a,b=pref[j]; value += gap*((1-b)**2-(1-a)**2)
    values=[value]
    for j in range(len(knots)-1):
        gap=Fraction.from_float(knots[j+1])-Fraction.from_float(knots[j])
        a,b=pref[j]; value += 2*gap*(b-a); values.append(value)
    lo,hi=min(values),max(values)
    return {'orientation':'candidate_minus_reference_CRPS',
            'lower_exact_fraction':str(lo),'upper_exact_fraction':str(hi),
            'lower_safe_float':_outward(lo,-1),'upper_safe_float':_outward(hi,1),
            'knots_native':knots,'knot_gaps_exact_fraction':[str(v) for v in values],
            'all_real_target_values':True,'method':'exact rational O(K) recurrence after sort'}
