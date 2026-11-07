"""Conservative finite-grid Ville inversion; stdlib only, no model/data access.

This is an implementation of a classical Hoeffding/Ville construction, not a new
estimator. Mathematical coverage requires the stated random-query design. All
rational aggregates remain exact; only the logarithm and interval endpoints use
100-digit Decimal with outward rounding. Arithmetic exceptions fail closed.
"""
from decimal import (Context, Decimal, ROUND_CEILING, ROUND_FLOOR,
                     ROUND_HALF_EVEN, Underflow, Subnormal)
from fractions import Fraction

PRECISION = 100
DELTA = Fraction(1, 20)
LAMBDA_EXPONENTS = tuple(range(-6, 3))
MAX_LCM_BITS = 200000
MIN_STEP = Fraction(1, 1000000)


def exact_fraction(value):
    """Accept exact representations; reject binary floats and bools."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("An exact Fraction/int/str/finite Decimal is required")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, Decimal) and value.is_finite():
        return Fraction(value)
    raise TypeError("An exact Fraction/int/str/finite Decimal is required")


def fraction_text(value):
    value = exact_fraction(value)
    return str(value.numerator) + "/" + str(value.denominator)


def _context(rounding):
    ctx = Context(prec=PRECISION, rounding=rounding, Emin=-999999, Emax=999999)
    # Avoid an unnoticed exponent-range underflow. Other standard arithmetic
    # errors (including overflow and invalid operations) are trapped by default.
    ctx.traps[Underflow] = True
    ctx.traps[Subnormal] = True
    return ctx


def fraction_decimal(value, upward):
    value = exact_fraction(value)
    ctx = _context(ROUND_CEILING if upward else ROUND_FLOOR)
    return ctx.divide(Decimal(value.numerator), Decimal(value.denominator))


def log_threshold_upper():
    """An upper bound for ln(18/delta)=ln(360).

    CPython Decimal.ln is specified to be correctly rounded HALF_EVEN. The
    adjacent larger 100-digit representable number therefore bounds the exact
    logarithm above. This relies on the documented Decimal implementation.
    https://docs.python.org/3/library/decimal.html#decimal.Decimal.ln
    https://docs.python.org/3/library/decimal.html#decimal.Decimal.next_plus
    """
    ctx = _context(ROUND_HALF_EVEN)
    threshold = Fraction(2 * len(LAMBDA_EXPONENTS), 1) / DELTA
    if threshold.denominator != 1:
        raise ArithmeticError("Fixed threshold must be an exact integer")
    rounded = ctx.ln(Decimal(threshold.numerator))
    return ctx.next_plus(rounded)


def lambda_grid(initial_width):
    width = exact_fraction(initial_width)
    if width <= 0:
        raise ValueError("A positive initial width is required")
    return tuple((Fraction(2 ** j) if j >= 0 else Fraction(1, 2 ** (-j)))
                 / width for j in LAMBDA_EXPONENTS)


def ville_interval(sum_z, sum_width_squared, count, initial_width):
    """Return outward rational endpoints for the fixed total loss-gap sum.

    Each of the 18 one-sided fixed-lambda supermartingales receives delta/18.
    One all-time event covers every returned interval, predictable stopping, and
    subtraction of the realized queried-label sum. No independence of labels or
    population model is assumed; frozen losses and actual proposal probabilities
    are required. A caller must intersect in TOTAL coordinates before subtracting
    a changing queried sum.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer")
    sum_z = exact_fraction(sum_z)
    variance_budget = exact_fraction(sum_width_squared)
    if variance_budget < 0:
        raise ValueError("Squared-width budget cannot be negative")
    upper_ctx, lower_ctx = _context(ROUND_CEILING), _context(ROUND_FLOOR)
    log_up = log_threshold_upper()
    z_up, z_down = fraction_decimal(sum_z, True), fraction_decimal(sum_z, False)
    lines = []
    for exponent, lam in zip(LAMBDA_EXPONENTS, lambda_grid(initial_width)):
        a_up = fraction_decimal(1 / lam, True)
        c_up = fraction_decimal(lam * variance_budget / 8, True)
        radius_up = upper_ctx.add(upper_ctx.multiply(log_up, a_up), c_up)
        hi = upper_ctx.divide(upper_ctx.add(z_up, radius_up), Decimal(count))
        lo = lower_ctx.divide(lower_ctx.subtract(z_down, radius_up), Decimal(count))
        if not lo.is_finite() or not hi.is_finite():
            raise ArithmeticError("Confidence endpoint is nonfinite")
        lines.append({"exponent": exponent, "lambda": fraction_text(lam),
                      "lower_decimal": str(lo), "upper_decimal": str(hi)})
    lo = max(Fraction(Decimal(row["lower_decimal"])) for row in lines)
    hi = min(Fraction(Decimal(row["upper_decimal"])) for row in lines)
    return {"lower": lo, "upper": hi, "log_threshold_upper": str(log_up),
            "lines": lines}
