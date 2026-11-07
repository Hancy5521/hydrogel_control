"""Exact, sorted-prefix finite-CDF geometry; no data access or model fitting.

Classical CRPS identities are used to expose a small risk statistic while keeping
every positive-mass atom of the high-dimensional empirical reference. Float
supports mean their actual binary values; weights and all geometry are rational.
"""
from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction as F
import math

P = (F(1, 20), F(9, 20), F(9, 20), F(1, 20))
BINS = (F(0), F(1, 20), F(1, 2), F(19, 20), F(1))
EPSILON = F(1, 100)
ZERO = F(0)


def rational(value):
    """Preserve native binary floats exactly; reject booleans/nonfinite input."""
    if isinstance(value, bool):
        raise TypeError('Boolean is not a scientific scalar')
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Finite scalar required')
        return F.from_float(value)
    return F(value)


def canonical_law(law):
    if isinstance(law, dict):
        if law.get('family') != 'weighted_empirical' or law.get('coordinate') != 'native':
            raise ValueError('Native weighted_empirical law required')
        x, w = law['support_native'], law['weight_fractions']
        if len(x) != len(w):
            raise ValueError('Matched supports and weights required')
        pairs = zip(x, w)
    else:
        pairs = law
    merged = {}
    for point, mass in pairs:
        point, mass = rational(point), rational(mass)
        if mass < 0:
            raise ValueError('Negative probability')
        if mass:
            merged[point] = merged.get(point, ZERO) + mass
    if not merged or sum(merged.values(), ZERO) != 1:
        raise ValueError('Positive exact total mass one required')
    return tuple(sorted(merged.items()))


def vector_add(a, b):
    if len(a) != len(b):
        raise ValueError('Vector dimension mismatch')
    return tuple(x + y for x, y in zip(a, b))


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('Vector dimension mismatch')
    return sum((x * y for x, y in zip(a, b)), ZERO)


def matvec(a, x):
    return tuple(dot(row, x) for row in a)


def zeros(d):
    return tuple(tuple(ZERO for _ in range(d)) for _ in range(d))


def matrix_add(a, b):
    if len(a) != len(b):
        raise ValueError('Matrix dimension mismatch')
    return tuple(vector_add(x, y) for x, y in zip(a, b))


def matrix_scale(a, scale):
    return tuple(tuple(scale * v for v in row) for row in a)


def simplex_theta(theta, blocks=None):
    theta = tuple(map(rational, theta))
    if blocks is None:
        blocks = len(theta) // 4
    if blocks not in (1, 2) or len(theta) != 4 * blocks:
        raise ValueError('One or two four-simplex blocks required')
    for k in range(blocks):
        block = theta[4*k:4*k+4]
        if sum(block, ZERO) != 1 or any(t < EPSILON*p for t, p in zip(block, P)):
            raise ValueError('Exact normalized support-floor simplex required')
    return theta


def theta_beta(theta):
    theta = simplex_theta(theta)
    return tuple(theta[4*k+j] - P[j]
                 for k in range(len(theta)//4) for j in range(3))


@dataclass(frozen=True)
class CDFGeometry:
    """K knots, 3D prefix integrals, and optional exact 6D context lift.

    Geometry construction is O(K * 3^2) after sorting. c(Y) uses binary search.
    Bounds include both unbounded tails because c is constant outside support.
    """
    atoms: tuple
    components: tuple
    points: tuple
    interval_h: tuple
    prefix_h: tuple
    c_below: tuple
    Q3: tuple
    bounds3: tuple
    g: object = None

    @property
    def dimension(self):
        return 3 if self.g is None else 6

    @property
    def multipliers(self):
        return (F(1),) if self.g is None else (1-self.g, self.g)

    @property
    def Q(self):
        m = self.multipliers
        return tuple(tuple(m[j//3]*m[k//3]*self.Q3[j%3][k%3]
                           for k in range(self.dimension))
                     for j in range(self.dimension))

    @property
    def c_bounds(self):
        return tuple((m*lo, m*hi) for m in self.multipliers
                     for lo, hi in self.bounds3)

    def lift(self, g):
        g = rational(g)
        if not 0 <= g <= 1:
            raise ValueError('Frozen known context g must be in [0,1]')
        return CDFGeometry(self.atoms, self.components, self.points,
                           self.interval_h, self.prefix_h, self.c_below,
                           self.Q3, self.bounds3, g)

    def c3(self, target):
        y = rational(target)
        if y <= self.points[0]:
            return self.c_below
        if y >= self.points[-1]:
            return vector_add(self.c_below, self.prefix_h[-1])
        j = bisect_right(self.points, y)-1
        return tuple(self.c_below[k] + self.prefix_h[j][k]
                     + (y-self.points[j])*self.interval_h[j][k] for k in range(3))

    def c(self, target):
        value = self.c3(target)
        return tuple(m*v for m in self.multipliers for v in value)

    def candidate(self, theta):
        theta = simplex_theta(theta, len(self.multipliers))
        effective = tuple(sum((m*theta[4*b+j] for b, m in
                               enumerate(self.multipliers)), ZERO) for j in range(4))
        out = tuple((x, sum((effective[j]*comp[j] for j in range(4)), ZERO))
                    for x, comp in zip(self.points, self.components))
        if len(out) != len(self.atoms) or any(m <= 0 for _, m in out):
            raise ArithmeticError('Support preservation failed')
        return canonical_law(out)

    def gap_bounds(self, theta):
        """All-real CRPS(candidate,Y)-CRPS(reference,Y) extrema, O(K*d)."""
        beta = theta_beta(simplex_theta(theta, len(self.multipliers)))
        constant = dot(beta, matvec(self.Q, beta))
        values = []
        for prefix in self.prefix_h:
            c3 = vector_add(self.c_below, prefix)
            c = tuple(m*v for m in self.multipliers for v in c3)
            values.append(constant + 2*dot(beta, c))
        return min(values), max(values)


def geometry(law):
    atoms = canonical_law(law)
    points = tuple(x for x, _ in atoms)
    components, left = [], ZERO
    for _, mass in atoms:
        right = left + mass
        components.append(tuple(max(ZERO, min(right, b)-max(left, a))/p
                                for a, b, p in zip(BINS[:-1], BINS[1:], P)))
        left = right
    running = [ZERO]*4
    reference_cdf = ZERO
    q = [[ZERO]*3 for _ in range(3)]
    c_below = [ZERO]*3
    prefix_h = [(ZERO,)*3]
    interval_h = []
    for j in range(len(points)-1):
        running = [a+b for a, b in zip(running, components[j])]
        reference_cdf += atoms[j][1]
        h = tuple(running[k]-running[-1] for k in range(3))
        width = points[j+1]-points[j]
        interval_h.append(h)
        prefix_h.append(tuple(a+width*b for a, b in zip(prefix_h[-1], h)))
        for k in range(3):
            c_below[k] += width*h[k]*(reference_cdf-1)
            for ell in range(3):
                q[k][ell] += width*h[k]*h[ell]
    values = [vector_add(tuple(c_below), v) for v in prefix_h]
    bounds = tuple((min(v[k] for v in values), max(v[k] for v in values))
                   for k in range(3))
    if any(not lo <= 0 <= hi for lo, hi in bounds):
        raise ArithmeticError('Proper-score c range must contain zero')
    return CDFGeometry(atoms, tuple(components), points, tuple(interval_h),
                       tuple(prefix_h), tuple(c_below), tuple(map(tuple, q)), bounds)


def crps(law, target):
    """Exact native CRPS via sorted weighted prefixes, O(K)."""
    atoms, y = canonical_law(law), rational(target)
    first, half_pair, cumulative_mass, cumulative_sum = ZERO, ZERO, ZERO, ZERO
    for x, mass in atoms:
        first += mass*abs(x-y)
        half_pair += mass*(x*cumulative_mass-cumulative_sum)
        cumulative_mass += mass
        cumulative_sum += mass*x
    return first-half_pair


def rref(matrix):
    a = [list(map(rational, row)) for row in matrix]
    if not a:
        return (), ()
    n, pivots, row = len(a[0]), [], 0
    if any(len(v) != n for v in a):
        raise ValueError('Rectangular matrix required')
    for col in range(n):
        pivot = next((j for j in range(row, len(a)) if a[j][col]), None)
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        scale = a[row][col]
        a[row] = [v/scale for v in a[row]]
        for j in range(len(a)):
            if j != row and a[j][col]:
                scale = a[j][col]
                a[j] = [v-scale*w for v, w in zip(a[j], a[row])]
        pivots.append(col)
        row += 1
        if row == len(a):
            break
    return tuple(map(tuple, a)), tuple(pivots)


def inverse_exact(matrix):
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError('Nonempty square matrix required')
    augmented = [tuple(row)+tuple(F(j == k) for k in range(n))
                 for j, row in enumerate(matrix)]
    reduced, pivots = rref(augmented)
    if pivots[:n] != tuple(range(n)):
        raise ArithmeticError('Singular matrix')
    if any(reduced[j][k] != F(j == k) for j in range(n) for k in range(n)):
        raise ArithmeticError('Left block is not identity')
    return tuple(row[n:] for row in reduced)


def range_geometry(matrix):
    """Exact rank, nullspace and Euclidean projector of a symmetric Gram matrix.

    No eigentolerance/ridge: a small nonzero rational direction is retained.
    Full rank returns identity without forming an ill-conditioned inverse.
    """
    q = tuple(tuple(map(rational, row)) for row in matrix)
    n = len(q)
    if not n or any(len(row) != n for row in q):
        raise ValueError('Nonempty square Gram matrix required')
    if any(q[j][k] != q[k][j] for j in range(n) for k in range(n)):
        raise ValueError('Symmetric Gram matrix required')
    reduced, pivots = rref(q)
    rank = len(pivots)
    nullspace = []
    for free in range(n):
        if free not in pivots:
            v = [ZERO]*n
            v[free] = F(1)
            for j, pivot in enumerate(pivots):
                v[pivot] = -reduced[j][free]
            nullspace.append(tuple(v))
    if rank == n:
        projector = tuple(tuple(F(j == k) for k in range(n)) for j in range(n))
    elif rank == 0:
        projector = zeros(n)
    else:
        b = tuple(tuple(q[j][col] for col in pivots) for j in range(n))
        bt_b = tuple(tuple(sum((b[j][k]*b[j][ell] for j in range(n)), ZERO)
                               for ell in range(rank)) for k in range(rank))
        inv = inverse_exact(bt_b)
        b_inv = tuple(tuple(dot(row, col) for col in zip(*inv)) for row in b)
        projector = tuple(tuple(dot(row, other) for other in b) for row in b_inv)
    return {'rank': rank, 'nullspace': tuple(nullspace), 'projector': projector,
            'rank_method': 'exact rational RREF; no small-eigenvalue truncation'}
