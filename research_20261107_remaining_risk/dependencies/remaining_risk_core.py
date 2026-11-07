"""Exact-propensity finite-pool risk queries and conservative partial repair.

No fitting, label I/O, filesystem or network access occurs here. Each unit's two
forecast laws and all-real loss-gap bounds must be frozen before construction.
``draw`` precedes external label access; ``observe`` is the only gap update.
Python Random supplies reproducible integer tickets. The statistical theorem is
under the ideal conditional-uniform-ticket model, not a claim that a fixed PRNG
seed itself creates mathematical randomness or cryptographic unpredictability.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from random import Random

from directed_confidence import (DELTA, LAMBDA_EXPONENTS, MAX_LCM_BITS,
                                 MIN_STEP, PRECISION, exact_fraction,
                                 fraction_text, ville_interval)


@dataclass(frozen=True)
class RiskUnit:
    identity: str
    lower: Fraction
    upper: Fraction
    curvature: Fraction

    def __post_init__(self):
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("A nonempty string identity is required")
        for name in ("lower", "upper", "curvature"):
            object.__setattr__(self, name, exact_fraction(getattr(self, name)))
        if self.lower > self.upper or self.curvature < 0:
            raise ValueError("Require lower <= upper and curvature >= 0")
        # Necessary CRPS/energy-score identities: E_F[d]=B, E_G[d]=-B.
        # These checks do not establish sufficiency of a caller's envelopes.
        if self.lower > -self.curvature or self.upper < self.curvature:
            raise ValueError("Bounds must contain both -curvature and +curvature")
        if self.lower == self.upper and (self.lower != 0 or self.curvature != 0):
            raise ValueError("Zero-width score-gap bounds require d=B=0")

    @property
    def width(self):
        return self.upper - self.lower

    @property
    def midpoint(self):
        return (self.lower + self.upper) / 2


@dataclass(frozen=True)
class QueryTicket:
    identity: str
    query_index: int
    ticket: int
    integer_weight: int
    integer_weight_total: int
    propensity: Fraction
    width_before: Fraction

    def to_record(self):
        return {"identity": self.identity, "query_index": self.query_index,
                "ticket": self.ticket, "integer_weight": self.integer_weight,
                "integer_weight_total": self.integer_weight_total,
                "propensity": fraction_text(self.propensity),
                "width_before": fraction_text(self.width_before)}


def partial_step(remaining_upper, remaining_curvature, interval_valid=True):
    """Exact positive-curvature minimizer; zero curvature/tiny steps abstain."""
    upper = exact_fraction(remaining_upper)
    curvature = exact_fraction(remaining_curvature)
    if curvature < 0:
        raise ValueError("Curvature must be nonnegative")
    alpha = Fraction(0)
    if interval_valid and curvature > 0:
        alpha = min(Fraction(1), max(Fraction(0),
                    (curvature - upper) / (2 * curvature)))
        if alpha < MIN_STEP:
            alpha = Fraction(0)
    bound = alpha * upper - alpha * (1 - alpha) * curvature
    return {"alpha": alpha, "risk_upper_sum": bound,
            "strict_improvement": bool(alpha > 0 and bound < 0)}


class ExactRemainingRisk:
    """A single frozen population/forecast epoch; never reset delta mid-epoch."""

    def __init__(self, units, seed):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        units = tuple(units)
        if not units or any(not isinstance(unit, RiskUnit) for unit in units):
            raise ValueError("A nonempty sequence of RiskUnit objects is required")
        if len({unit.identity for unit in units}) != len(units):
            raise ValueError("Unit identities must be unique")
        self.units = {unit.identity: unit for unit in sorted(units, key=lambda u: u.identity)}
        denominator_lcm = 1
        for unit in self.units.values():
            if unit.width > 0:
                denominator = unit.width.denominator
                if denominator.bit_length() > MAX_LCM_BITS:
                    raise ValueError("LCM bit guard exceeded before any label query")
                denominator_lcm = denominator_lcm // gcd(denominator_lcm, denominator) * denominator
                if denominator_lcm.bit_length() > MAX_LCM_BITS:
                    raise ValueError("LCM bit guard exceeded before any label query")
        self.denominator_lcm = denominator_lcm
        self.integer_weights = {key: int(unit.width * denominator_lcm)
                                for key, unit in self.units.items() if unit.width > 0}
        if any(Fraction(weight, denominator_lcm) != self.units[key].width
               for key, weight in self.integer_weights.items()):
            raise ArithmeticError("Internal exact weight construction failed")
        self.seed = seed
        self._rng = Random(seed)
        self.observed = {}
        self.events = []
        self.pending = None
        self.invalid_reason = None
        self.total_midpoint = sum((unit.midpoint for unit in units), Fraction(0))
        self.initial_width = sum((unit.width for unit in units), Fraction(0))
        self.sum_z = Fraction(0)
        self.sum_width_squared = Fraction(0)
        self.observed_residual_sum = Fraction(0)
        self.observed_gap_sum = Fraction(0)

    @property
    def remaining_ids(self):
        return tuple(key for key in self.units if key not in self.observed)

    def draw(self):
        """Sample an exact PPS ticket. No caller label should be read beforehand."""
        if self.invalid_reason:
            raise RuntimeError("Invalid confidence state: " + self.invalid_reason)
        if self.pending is not None:
            raise RuntimeError("An unobserved ticket is already pending")
        weights = [(key, weight) for key, weight in self.integer_weights.items()
                   if key not in self.observed]
        if not weights:
            return None
        total = sum(weight for _, weight in weights)
        integer_ticket = self._rng.randrange(total)
        cursor = 0
        selected = None
        for key, weight in weights:
            cursor += weight
            if integer_ticket < cursor:
                selected = (key, weight)
                break
        key, weight = selected
        width = Fraction(total, self.denominator_lcm)
        self.pending = QueryTicket(key, len(self.observed) + 1, integer_ticket,
                                   weight, total, Fraction(weight, total), width)
        return self.pending

    def observe(self, ticket, gap):
        if self.invalid_reason:
            raise RuntimeError("Invalid confidence state: " + self.invalid_reason)
        if not isinstance(ticket, QueryTicket) or ticket != self.pending:
            raise ValueError("Ticket does not equal the current pending draw")
        gap = exact_fraction(gap)
        unit = self.units[ticket.identity]
        if not unit.lower <= gap <= unit.upper:
            self.invalid_reason = "Observed gap violates its frozen all-real bound"
            raise ValueError(self.invalid_reason)
        residual = gap - unit.midpoint
        z = self.total_midpoint + self.observed_residual_sum + residual / ticket.propensity
        self.sum_z += z
        self.sum_width_squared += ticket.width_before ** 2
        self.observed_residual_sum += residual
        self.observed_gap_sum += gap
        self.observed[ticket.identity] = gap
        event = ticket.to_record()
        event.update({"gap": fraction_text(gap), "midpoint": fraction_text(unit.midpoint),
                      "z": fraction_text(z), "sum_z": fraction_text(self.sum_z),
                      "sum_width_squared": fraction_text(self.sum_width_squared)})
        self.events.append(event)
        self.pending = None
        return dict(event)

    def decision(self):
        """Return an anytime bound for exactly the CURRENT unqueried unit set."""
        remaining = tuple(self.units[key] for key in self.remaining_ids)
        lower_remaining = sum((unit.lower for unit in remaining), Fraction(0))
        upper_remaining = sum((unit.upper for unit in remaining), Fraction(0))
        curvature = sum((unit.curvature for unit in remaining), Fraction(0))
        lower_total = self.observed_gap_sum + lower_remaining
        upper_total = self.observed_gap_sum + upper_remaining
        cs_record = None
        reason = self.invalid_reason
        if not reason and self.observed:
            try:
                cs_record = ville_interval(self.sum_z, self.sum_width_squared,
                                          len(self.observed), self.initial_width)
                lower_total = max(lower_total, cs_record["lower"])
                upper_total = min(upper_total, cs_record["upper"])
                if lower_total > upper_total:
                    reason = "Empty confidence/deterministic intersection; abstain"
                    self.invalid_reason = reason
            except (ArithmeticError, ValueError) as error:
                reason = "Confidence arithmetic failed closed: " + type(error).__name__
                self.invalid_reason = reason
        lower_remaining = lower_total - self.observed_gap_sum
        upper_remaining = upper_total - self.observed_gap_sum
        valid = reason is None
        step = partial_step(upper_remaining, curvature, valid)
        return {"status": "valid" if valid else "abstain", "reason": reason,
                "observed_count": len(self.observed), "remaining_count": len(remaining),
                "remaining_ids": self.remaining_ids,
                "total_lower": fraction_text(lower_total),
                "total_upper": fraction_text(upper_total),
                "remaining_lower": fraction_text(lower_remaining),
                "remaining_upper": fraction_text(upper_remaining),
                "remaining_curvature": fraction_text(curvature),
                "alpha": fraction_text(step["alpha"]),
                "risk_upper_sum": fraction_text(step["risk_upper_sum"]),
                "strict_improvement": step["strict_improvement"],
                "confidence_lines": cs_record["lines"] if cs_record else [],
                "log_threshold_upper": cs_record["log_threshold_upper"] if cs_record else None}

    def snapshot(self):
        """Serializable audit receipt. Reuse saved tickets; never redraw on resume."""
        return {"schema": "exact_remaining_risk_v1", "seed": self.seed,
                "delta": fraction_text(DELTA), "precision": PRECISION,
                "lambda_exponents": list(LAMBDA_EXPONENTS),
                "min_step": fraction_text(MIN_STEP), "max_lcm_bits": MAX_LCM_BITS,
                "denominator_lcm": self.denominator_lcm,
                "initial_width": fraction_text(self.initial_width),
                "total_midpoint": fraction_text(self.total_midpoint),
                "sum_z": fraction_text(self.sum_z),
                "sum_width_squared": fraction_text(self.sum_width_squared),
                "observed_gap_sum": fraction_text(self.observed_gap_sum),
                "observed_residual_sum": fraction_text(self.observed_residual_sum),
                "invalid_reason": self.invalid_reason,
                "pending": self.pending.to_record() if self.pending else None,
                "units": [{"identity": unit.identity,
                           "lower": fraction_text(unit.lower),
                           "upper": fraction_text(unit.upper),
                           "curvature": fraction_text(unit.curvature)}
                          for unit in self.units.values()],
                "events": [dict(event) for event in self.events]}
