"""Fixed four-hypothesis mixed-measure likelihood and transactional native-cell ledger.

This is a synthetic mechanism gate, not a learned material model. Gaussian densities
use NumPy Cholesky/solve; no SciPy or empirical labels are used in this module.
"""
import math
import numpy as np

FIELDS = ('A', 'G', 'Y')
ROWS = ('r0', 'r1')
PRIORS = (.1, .2, .3, .4)
P_ZERO = (.1, .3, .6, .8)
RHO = (-.6, -.2, .25, .65)
ALIAS_ITEMS = (('r0', 'r0'), ('r1', 'r1'), ('rowA', 'r0'),
               ('duplicate_report', 'r0'), ('rowB', 'r1'),
               ('held_out', 'held_out'), ('terminal', 'held_out'), ('terminal_report', 'held_out'))
TERMINAL_ROWS = ('held_out',)
BASE_MEASURE = 'A,Y: Lebesgue; observed G: delta_0 + Lebesgue; absent G: projected marginal measure'


def finite_native(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('finite built-in native numeric value required')
    # Signed zeros denote the same atom and the same native cell value.
    return 0.0 if value == 0 else float(value)


def logaddexp(left, right):
    upper = max(left, right)
    return upper + math.log(math.exp(left - upper) + math.exp(right - upper))


def gaussian_log_density(values, mean, covariance):
    """Log density of an observed Gaussian marginal, with empty density one."""
    if len(values) == 0:
        return 0.0
    x = np.asarray(values, dtype=float)
    m = np.asarray(mean, dtype=float)
    matrix = np.asarray(covariance, dtype=float)
    if x.ndim != 1 or m.shape != x.shape or matrix.shape != (len(x), len(x)):
        raise ValueError('Gaussian dimensions disagree')
    if not np.isfinite(x).all() or not np.isfinite(m).all() or not np.isfinite(matrix).all():
        raise ValueError('Gaussian inputs must be finite')
    chol = np.linalg.cholesky(matrix)
    standardized = np.linalg.solve(chol, x - m)
    logdet = 2.0 * math.fsum(math.log(float(v)) for v in chol.diagonal())
    quadratic = math.fsum(float(v) ** 2 for v in standardized)
    return -.5 * (len(x) * math.log(2.0 * math.pi) + logdet + quadratic)


def branch_log_marginal(observed, h, zero_branch):
    """Marginalize unobserved coordinates; never substitute or impute them."""
    if type(h) is not int or not 0 <= h < 4:
        raise ValueError('hypothesis must be an integer in [0,3]')
    rho = RHO[h]
    if zero_branch:
        fields = ('A', 'Y')
        mean = (.2 * h, -.1 * h)
        covariance = ((1.0, rho), (rho, 1.0))
    else:
        fields = FIELDS
        mean = (.2 * h, .3 - .1 * h, -.1 * h)
        covariance = tuple(tuple(rho ** abs(i - j) for j in range(3)) for i in range(3))
    indices = [i for i, name in enumerate(fields) if name in observed]
    values = [observed[fields[i]] for i in indices]
    return gaussian_log_density(values, [mean[i] for i in indices],
                                [[covariance[i][j] for j in indices] for i in indices])


def partial_log_likelihood(observed, h):
    """Radon--Nikodym density w.r.t. the fixed projected mixed base measure.

    At observed G=0 only the spike contributes. The continuous slab has zero
    derivative at the delta atom for this base measure; its normal density at zero
    cannot be added to atom mass. If G is absent, both Gaussian marginals contribute.
    """
    if not isinstance(observed, dict) or set(observed) - set(FIELDS):
        raise ValueError('unknown observed field')
    values = {name: finite_native(value) for name, value in observed.items()}
    if type(h) is not int or not 0 <= h < 4:
        raise ValueError('hypothesis must be an integer in [0,3]')
    if not values:
        return 0.0
    probability = P_ZERO[h]
    if 'G' not in values:
        return logaddexp(math.log(probability) + branch_log_marginal(values, h, True),
                         math.log1p(-probability) + branch_log_marginal(values, h, False))
    if values['G'] == 0:
        ay = {name: value for name, value in values.items() if name != 'G'}
        return math.log(probability) + branch_log_marginal(ay, h, True)
    return math.log1p(-probability) + branch_log_marginal(values, h, False)


def joint_log_likelihood_by_h(observations):
    """Rows are conditionally independent given h; fields within a row are not."""
    if not isinstance(observations, dict) or set(observations) - set(ROWS):
        raise ValueError('only canonical pool rows allowed')
    return [math.fsum(partial_log_likelihood(observations.get(row, {}), h) for row in ROWS)
            for h in range(4)]


def posterior(log_likelihood):
    if len(log_likelihood) != 4 or not all(math.isfinite(x) for x in log_likelihood):
        raise ValueError('four finite hypothesis log likelihoods required')
    log_mass = [math.log(prior) + value for prior, value in zip(PRIORS, log_likelihood)]
    center = max(log_mass)
    mass = [math.exp(value - center) for value in log_mass]
    total = math.fsum(mass)
    return [value / total for value in mass]


def log_evidence(log_likelihood):
    log_mass = [math.log(prior) + value for prior, value in zip(PRIORS, log_likelihood)]
    center = max(log_mass)
    return center + math.log(math.fsum(math.exp(value - center) for value in log_mass))


class CanonicalCellLedger:
    """All-or-nothing panels; repeated physical cells are evidence- and cost-idempotent."""

    def __init__(self):
        self._values = {}
        self._log_likelihood = [0.0] * 4
        self._weights = list(PRIORS)
        self._aliases = dict(ALIAS_ITEMS)

    def snapshot(self):
        return {'observations': [{'row': row, 'field': field, 'value': value}
                                 for (row, field), value in sorted(self._values.items())],
                'total_cost': len(self._values), 'canonical_cell_count': len(self._values),
                'log_likelihood_per_h': list(self._log_likelihood), 'weights': list(self._weights)}

    def _canonical_cell(self, cell):
        if not isinstance(cell, (list, tuple)) or len(cell) != 3:
            raise ValueError('cell must be [row,field,native_value]')
        row, field, value = cell
        if type(row) is not str or row not in self._aliases:
            raise ValueError('unknown row or alias')
        canonical = self._aliases[row]
        if canonical in TERMINAL_ROWS:
            raise ValueError('terminal row is not purchasable')
        if canonical not in ROWS:
            raise ValueError('unknown pool row')
        if type(field) is not str or field not in FIELDS:
            raise ValueError('unknown field')
        return canonical, field, finite_native(value)

    def apply_panel(self, cells):
        if not isinstance(cells, (list, tuple)):
            raise ValueError('cell panel must be a sequence; empty panel denotes STOP')
        before = self.snapshot()
        candidate = dict(self._values)
        canonical_requests = []
        for cell in cells:
            row, field, value = self._canonical_cell(cell)
            key = (row, field)
            if key in candidate and candidate[key] != value:
                raise ValueError('conflicting values for the same canonical native cell')
            candidate[key] = value
            canonical_requests.append([row, field, value])
        # Validation and all likelihood calculations finish before committing anything.
        new = sorted(set(candidate) - set(self._values))
        if new:
            observed = {}
            for (row, field), value in candidate.items():
                observed.setdefault(row, {})[field] = value
            current = joint_log_likelihood_by_h(observed)
            weights = posterior(current)
            delta = [value - previous for value, previous in zip(current, self._log_likelihood)]
            evidence_increment = log_evidence(current) - log_evidence(self._log_likelihood)
            self._values = candidate
            self._log_likelihood = current
            self._weights = weights
        else:
            delta = [0.0] * 4
            evidence_increment = 0.0
        return {'canonical_requests': canonical_requests,
                'new_cells': [[row, field, candidate[(row, field)]] for row, field in new],
                'incremental_cost': len(new), 'incremental_log_likelihood_per_h': delta,
                'incremental_log_evidence': evidence_increment, 'before': before, 'after': self.snapshot()}
