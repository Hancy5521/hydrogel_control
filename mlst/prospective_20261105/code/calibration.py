"""Finite-grid, independent-unit split conformal prediction sets.

This module neither fits nor calibrates a probability density. It is deliberately
independent of the hydrogel predictor and all historical measurement files.
"""
from dataclasses import dataclass
from fractions import Fraction
import math


def real(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(name + ' must be a real number, not boolean')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + ' must be finite')
    return value


def alpha_fraction(alpha):
    """Require an explicit exact decimal/rational string, avoiding float indices."""
    if not isinstance(alpha, str):
        raise ValueError('alpha must be an exact decimal or fraction string')
    value = Fraction(alpha)
    if not 0 < value < 1:
        raise ValueError('alpha must be strictly between zero and one')
    return value


def unit_max_scores(records, return_reasons=False):
    """Max over all target coordinates/curves/masks in each declared unit.

Each record has unit_id, mean, sd and truth. The role contract must first establish
that these records belong only to the reserved calibration cohort. Repeated rows
do not increase the number of units. Extended +infinity scores are retained when
finite-input subtraction/division exceeds the floating-point range.
"""
    scores, reasons = {}, {}
    for record in records:
        unit = record['unit_id']
        if not isinstance(unit, str) or not unit:
            raise ValueError('A nonempty independent-unit identifier is required')
        mean, sd, truth = record['mean'], record['sd'], record['truth']
        if len(mean) == 0 or len(mean) != len(sd) or len(mean) != len(truth):
            raise ValueError('Nonempty aligned mean, sd and truth required')
        score = 0.
        for m, s, y in zip(mean, sd, truth):
            m, s, y = real(m, 'mean'), real(s, 'sd'), real(y, 'truth')
            if s <= 0:
                raise ValueError('Strictly positive standard deviations required')
            standardized = abs(y - m) / s
            if math.isinf(standardized):
                reasons.setdefault(unit, set()).add('numeric_score_overflow')
            score = max(score, standardized)
        scores[unit] = max(scores.get(unit, 0.), score)
    flags = {unit: sorted(reasons.get(unit, set())) for unit in scores}
    return (scores, flags) if return_reasons else scores


@dataclass(frozen=True)
class ConformalThreshold:
    alpha: str
    n_units: int
    order_index: int
    q: float
    infinity_reasons: tuple = ()

    def __post_init__(self):
        if type(self.n_units) is not int or self.n_units < 0 or type(self.order_index) is not int:
            raise ValueError('Exact conformal sample count/order index required')
        exact = (self.n_units + 1) * (1 - alpha_fraction(self.alpha))
        k = (exact.numerator + exact.denominator - 1) // exact.denominator
        if self.order_index != k:
            raise ValueError('Exact conformal sample count/order index required')
        if isinstance(self.q, bool) or not isinstance(self.q, (int, float)) or math.isnan(self.q) or self.q < 0:
            raise ValueError('Nonnegative extended-real threshold required')
        if k > self.n_units and not math.isinf(self.q):
            raise ValueError('Too-small calibration cohort requires an unbounded threshold')
        allowed = {'insufficient_calibration_units', 'selected_infinite_score',
                   'numeric_score_overflow', 'infinite_input_score'}
        if (not isinstance(self.infinity_reasons, tuple)
                or any(v not in allowed for v in self.infinity_reasons)
                or len(set(self.infinity_reasons)) != len(self.infinity_reasons)):
            raise ValueError('Known unique infinity-reason flags required')
        if not math.isinf(self.q) and self.infinity_reasons:
            raise ValueError('Finite thresholds cannot have infinity reasons')
        if math.isinf(self.q) and not self.infinity_reasons:
            defaults = ('insufficient_calibration_units',) if k > self.n_units else ('selected_infinite_score', 'infinite_input_score')
            object.__setattr__(self, 'infinity_reasons', defaults)
        if k > self.n_units and 'insufficient_calibration_units' not in self.infinity_reasons:
            raise ValueError('Insufficient calibration sample reason must be retained')
        if k > self.n_units and set(self.infinity_reasons) != {'insufficient_calibration_units'}:
            raise ValueError('An augmented infinity threshold has the insufficient-sample reason')
        if math.isinf(self.q) and k <= self.n_units and (
                'selected_infinite_score' not in self.infinity_reasons
                or 'insufficient_calibration_units' in self.infinity_reasons
                or not set(self.infinity_reasons) & {'numeric_score_overflow', 'infinite_input_score'}):
            raise ValueError('A selected infinite score must retain its score-origin reason')

    def to_json(self):
        return {
            'kind': 'independent_unit_max_split_conformal_threshold',
            'alpha': self.alpha, 'n_calibration_units': self.n_units,
            'order_index_one_based': self.order_index,
            'q': None if math.isinf(self.q) else self.q,
            'unbounded': math.isinf(self.q),
            'infinity_reasons': list(self.infinity_reasons),
            'sample_count_is_declared_score_groups': True,
            'density_calibrated': False,
            'coverage_scope': 'Marginal over exchangeable complete units; all declared finite-grid targets. No between-grid or subgroup-conditional guarantee.',
        }


def calibrate_unit_scores(scores, alpha='0.1', score_reasons=None):
    """Exact kth order statistic with the standard +infinity augmentation.

An empty calibration cohort gives the vacuous unbounded set, not a finite
threshold. CLI real-study registration separately requires a nonempty cohort.
"""
    a = alpha_fraction(alpha)
    if not isinstance(scores, dict):
        raise ValueError('Unit-to-score mapping required')
    values = []
    if score_reasons is not None and (not isinstance(score_reasons, dict) or set(score_reasons) != set(scores)):
        raise ValueError('Reason metadata must cover exactly the supplied score groups')
    infinity_causes = set()
    for unit, value in scores.items():
        if not isinstance(unit, str) or not unit:
            raise ValueError('Unique nonempty unit identifiers required')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Scores must be nonnegative extended real numbers')
        value = float(value)
        if math.isnan(value) or value < 0:
            raise ValueError('Scores must be nonnegative and not NaN')
        flags = [] if score_reasons is None else score_reasons[unit]
        if (not isinstance(flags, (list, tuple))
                or any(flag not in ('numeric_score_overflow', 'infinite_input_score') for flag in flags)):
            raise ValueError('Known score infinity reasons required')
        if math.isfinite(value) and flags:
            raise ValueError('Finite scores cannot have infinity-reason flags')
        if math.isinf(value):
            infinity_causes.update(flags or ['infinite_input_score'])
        values.append(value)
    n = len(values)
    exact = (n + 1) * (1 - a)
    k = (exact.numerator + exact.denominator - 1) // exact.denominator
    q = sorted(values)[k - 1] if k <= n else math.inf
    reasons = ('insufficient_calibration_units',) if k > n else (
        tuple(['selected_infinite_score'] + sorted(infinity_causes)) if math.isinf(q) else ())
    return ConformalThreshold(alpha, n, k, q, reasons)


def threshold_from_json(obj):
    if obj.get('kind') != 'independent_unit_max_split_conformal_threshold':
        raise ValueError('Conformal threshold record required')
    alpha_fraction(obj['alpha'])
    n, k = obj['n_calibration_units'], obj['order_index_one_based']
    if type(n) is not int or type(k) is not int or n < 0 or k < 1:
        raise ValueError('Invalid order-statistic metadata')
    exact = (n + 1) * (1 - alpha_fraction(obj['alpha']))
    expected = (exact.numerator + exact.denominator - 1) // exact.denominator
    if k != expected:
        raise ValueError('Order-statistic index was changed')
    unbounded = obj.get('unbounded')
    if type(unbounded) is not bool:
        raise ValueError('Explicit unbounded flag required')
    if unbounded:
        if obj.get('q') is not None:
            raise ValueError('Unbounded q is represented by null plus a flag')
        q = math.inf
    else:
        q = real(obj.get('q'), 'q')
        if q < 0 or k > n:
            raise ValueError('A finite threshold cannot replace infinity')
    flags = obj.get('infinity_reasons')
    if not isinstance(flags, list):
        raise ValueError('Explicit infinity-reason list required, empty for finite q')
    if unbounded and not flags:
        raise ValueError('Serialized unbounded thresholds must declare an infinity reason')
    return ConformalThreshold(obj['alpha'], n, k, q, tuple(flags))


@dataclass(frozen=True)
class ConformalBand:
    lower_log: tuple | None
    upper_log: tuple | None
    target_count: int
    unbounded: bool

    def to_json(self):
        return {'kind': 'prediction_set_not_probability_density',
                'lower_log_eta': self.lower_log, 'upper_log_eta': self.upper_log,
                'target_count': self.target_count, 'unbounded': self.unbounded,
                'eta_reference_Pa_s': 1.,
                'positive_eta_interpretation': '(0,+infinity)' if self.unbounded else 'exp(log endpoints) Pa s; use log endpoints if float exponentiation overflows',
                'CRPS_defined_for_band': False, 'joint_log_density_defined_for_band': False}

    def crps(self, *args, **kwargs):
        raise TypeError('A conformal prediction set has no probability CDF or CRPS')

    def joint_log_density(self, *args, **kwargs):
        raise TypeError('A conformal prediction set has no joint probability density')


def make_band(mean, sd, threshold):
    if not isinstance(threshold, ConformalThreshold):
        raise ValueError('An explicit ConformalThreshold is required')
    if len(mean) == 0 or len(mean) != len(sd):
        raise ValueError('Nonempty aligned mean and sd required')
    m = [real(v, 'mean') for v in mean]
    s = [real(v, 'sd') for v in sd]
    if any(v <= 0 for v in s):
        raise ValueError('Strictly positive finite sd required')
    if math.isinf(threshold.q):
        return ConformalBand(None, None, len(m), True)
    lower = tuple(v - threshold.q * z for v, z in zip(m, s))
    upper = tuple(v + threshold.q * z for v, z in zip(m, s))
    if not all(math.isfinite(v) for v in lower + upper):
        raise OverflowError('Finite-q band overflows; retain the failure, do not silently broaden or clip')
    return ConformalBand(lower, upper, len(m), False)
