"""Independent exact saved-law/remaining-risk math; no SUT/core imports or fits.

No functions execute at import except constant definitions. The rational ln
enclosure is distinct from the producer's Decimal100 calculation. It checks
outward directions exactly, without a coverage tolerance.
"""
from fractions import Fraction as F
from functools import lru_cache
import math
import struct

DELTA = F(1, 20)
EXPONENTS = tuple(range(-6, 3))
MIN_STEP = F(1, 1000000)
LOG_TERMS = 256
OVERWIDTH_DIAGNOSTIC_SCALE = F(1, 10**94)
BINS = (F(0), F(1, 20), F(1, 2), F(19, 20), F(1))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def binary(value):
    require(type(value) in (int, float) and math.isfinite(value), "Finite native binary scalar")
    return F.from_float(float(value))


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def atoms(law):
    if law["family"] == "affine_empirical":
        offset = F(law["offset_exact_fraction"])
        source = law["residual_law"]
    else:
        offset, source = F(0), law
    require(source["family"] == "weighted_empirical", "Finite empirical law required")
    require(law["coordinate"] == "native" and source["coordinate"] == "native", "Native law coordinate")
    require(len(source["support_native"]) == len(source["weight_fractions"]) > 0,
            "Nonempty matched finite atoms")
    merged = {}
    for point, text in zip(source["support_native"], source["weight_fractions"]):
        require(isinstance(text, str), "Exact rational weight string")
        weight = F(text)
        require(weight >= 0, "Nonnegative atom weight")
        if weight:
            point = binary(point) + offset
            merged[point] = merged.get(point, F(0)) + weight
    require(merged and sum(merged.values(), F(0)) == 1, "Exact full CDF mass")
    return tuple(sorted(merged.items()))


def half_pair(model):
    mass = moment = value = F(0)
    for point, weight in model:
        value += weight * (point * mass - moment)
        mass += weight
        moment += weight * point
    return value


def energy_crps(model, target):
    target = target if isinstance(target, F) else binary(target)
    return sum((p * abs(x - target) for x, p in model), F(0)) - half_pair(model)


def integral_crps(model, target):
    target = target if isinstance(target, F) else binary(target)
    value = max(model[0][0] - target, F(0)) + max(target - model[-1][0], F(0))
    cdf = F(0)
    for (left, weight), (right, _) in zip(model[:-1], model[1:]):
        cdf += weight
        value += max(F(0), min(target, right) - left) * cdf**2
        value += max(F(0), right - max(target, left)) * (1 - cdf)**2
    return value


def score_at_knots(model, knots):
    index, cdf, first = 0, F(0), F(0)
    mean = sum((x * p for x, p in model), F(0))
    spread = half_pair(model)
    values = []
    for knot in knots:
        while index < len(model) and model[index][0] <= knot:
            x, p = model[index]
            cdf += p
            first += x * p
            index += 1
        values.append(2 * knot * cdf - 2 * first + mean - knot - spread)
    return values


def all_real_geometry(reference, candidate):
    """Prefix energy envelope + independently squared CDF distance curvature."""
    knots = sorted({x for x, _ in reference} | {x for x, _ in candidate})
    fs, gs = score_at_knots(reference, knots), score_at_knots(candidate, knots)
    gaps = [g - f for f, g in zip(fs, gs)]
    fi = gi = 0
    fcdf = gcdf = curvature = F(0)
    for left, right in zip(knots[:-1], knots[1:]):
        while fi < len(reference) and reference[fi][0] <= left:
            fcdf += reference[fi][1]
            fi += 1
        while gi < len(candidate) and candidate[gi][0] <= left:
            gcdf += candidate[gi][1]
            gi += 1
        curvature += (right - left) * (fcdf - gcdf)**2
    lower, upper = min(gaps), max(gaps)
    require(curvature >= 0 and lower <= -curvature and curvature <= upper,
            "Proper-score envelope/curvature necessary identities")
    return {"knots": knots, "gaps": gaps, "lower": lower,
            "upper": upper, "curvature": curvature}


def transformed_atoms(reference, theta0, theta1, context):
    def normalized(values):
        values = [F(str(float(x))) for x in values]
        require(len(values) == 4 and min(values) > 0, "Four positive spline weights")
        total = sum(values, F(0))
        return [v / total for v in values]
    weights = normalized(theta0)
    if theta1 is not None:
        second = normalized(theta1)
        g = F(str(float(context)))
        require(0 <= g <= 1, "Bounded fixed context")
        weights = [(1 - g) * a + g * b for a, b in zip(weights, second)]
    def mapped(probability):
        return sum((w * max(F(0), min(F(1), (probability - left) / (right - left)))
                    for w, left, right in zip(weights, BINS[:-1], BINS[1:])), F(0))
    before, output = F(0), []
    for point, mass in reference:
        after = before + mass
        output.append((point, mapped(after) - mapped(before)))
        before = after
    require(sum((p for _, p in output), F(0)) == 1 and min(p for _, p in output) >= 0,
            "Corrected CDF simplex")
    return tuple((x, p) for x, p in output if p)


@lru_cache(maxsize=1)
def ln360_bracket():
    """Exact 256-term positive atanh series enclosure; no Decimal/mpmath."""
    def enclosure(z):
        power, square, lower = z, z * z, F(0)
        for k in range(LOG_TERMS):
            lower += 2 * power / (2 * k + 1)
            power *= square
        tail = 2 * power / ((2 * LOG_TERMS + 1) * (1 - square))
        return lower, lower + tail
    l2, u2 = enclosure(F(1, 3))
    l45, u45 = enclosure(F(13, 77))
    return 8 * l2 + l45, 8 * u2 + u45


def confidence_line_brackets(sum_z, squared_widths, count, initial_width):
    require(count > 0 and initial_width > 0 and squared_widths >= 0, "Valid exact CS inputs")
    loglo, loghi = ln360_bracket()
    rows = []
    for exponent in EXPONENTS:
        lam = (F(2**exponent) if exponent >= 0 else F(1, 2**(-exponent))) / initial_width
        radiuslo = loglo / lam + lam * squared_widths / 8
        radiushi = loghi / lam + lam * squared_widths / 8
        rows.append({"exponent": exponent, "lambda": lam,
                     "lower_min": (sum_z - radiushi) / count,
                     "lower_max": (sum_z - radiuslo) / count,
                     "upper_min": (sum_z + radiuslo) / count,
                     "upper_max": (sum_z + radiushi) / count,
                     "overwidth_diagnostic_tolerance":
                         (abs(sum_z) + radiushi + 1) * OVERWIDTH_DIAGNOSTIC_SCALE / count})
    return rows


def check_confidence_lines(saved_lines, sum_z, squared_widths, count, initial_width,
                           saved_log_upper):
    expected = confidence_line_brackets(sum_z, squared_widths, count, initial_width)
    require(len(saved_lines) == len(expected) == 9, "Nine fixed confidence lines")
    _, loghi = ln360_bracket()
    require(F(saved_log_upper) >= loghi, "Documented Decimal log bound independently encloses ln360")
    for row, ref in zip(saved_lines, expected):
        lo, hi = F(row["lower_decimal"]), F(row["upper_decimal"])
        require(row["exponent"] == ref["exponent"] and F(row["lambda"]) == ref["lambda"],
                "Fixed exact lambda grid")
        # Zero inward tolerance: these two conditions establish conservative
        # endpoints even against the enclosing upper logarithm bracket.
        require(lo <= ref["lower_min"], "Lower confidence endpoint must be outward")
        require(hi >= ref["upper_max"], "Upper confidence endpoint must be outward")
        # This diagnostic cannot excuse any inward error; it only rejects a
        # substantially more vacuous implementation than the fixed precision.
        tolerance = ref["overwidth_diagnostic_tolerance"]
        require(ref["lower_max"] - lo <= tolerance, "Excessive lower endpoint over-width")
        require(hi - ref["upper_min"] <= tolerance, "Excessive upper endpoint over-width")
    return max(F(v["lower_decimal"]) for v in saved_lines), min(F(v["upper_decimal"]) for v in saved_lines)


def chosen_step(upper, curvature, valid):
    if not valid or curvature == 0:
        return F(0), F(0)
    alpha = min(F(1), max(F(0), (curvature - upper) / (2 * curvature)))
    if alpha < MIN_STEP:
        alpha = F(0)
    return alpha, alpha * upper - alpha * (1 - alpha) * curvature


def prefix_decision(saved, units, observed, sum_z, squared_widths, initial_width,
                    already_invalid=False):
    """Verify one saved core decision from independently accumulated exact state."""
    remaining = sorted(set(units) - set(observed))
    queried_gap_sum = sum(observed.values(), F(0))
    lower_total = queried_gap_sum + sum((units[key]["lower"] for key in remaining), F(0))
    upper_total = queried_gap_sum + sum((units[key]["upper"] for key in remaining), F(0))
    curvature = sum((units[key]["curvature"] for key in remaining), F(0))
    invalid = already_invalid
    if observed and not already_invalid:
        cslo, cshi = check_confidence_lines(saved["confidence_lines"], sum_z,
                squared_widths, len(observed), initial_width, saved["log_threshold_upper"])
        lower_total, upper_total = max(lower_total, cslo), min(upper_total, cshi)
        invalid = lower_total > upper_total
    elif not observed:
        require(saved["confidence_lines"] == [] and saved["log_threshold_upper"] is None,
                "No unearned confidence interval before first query")
    else:
        require(saved["confidence_lines"] == [], "Invalid epoch stays abstaining")
    lower_remaining = lower_total - queried_gap_sum
    upper_remaining = upper_total - queried_gap_sum
    alpha, risk = chosen_step(upper_remaining, curvature, not invalid)
    require(saved["status"] == ("abstain" if invalid else "valid"), "Prefix validity")
    require(saved["observed_count"] == len(observed), "Exact queried prefix count")
    require(saved["remaining_count"] == len(remaining), "Exact current remaining count")
    require(list(saved["remaining_ids"]) == remaining, "Exact current remaining identity set/order")
    for field, expected in (("total_lower", lower_total), ("total_upper", upper_total),
            ("remaining_lower", lower_remaining), ("remaining_upper", upper_remaining),
            ("remaining_curvature", curvature), ("alpha", alpha), ("risk_upper_sum", risk)):
        require(F(saved[field]) == expected, "Exact prefix decision " + field)
    require(saved["strict_improvement"] == bool(alpha > 0 and risk < 0), "Strict prefix predicate")
    if invalid:
        require(isinstance(saved["reason"], str) and bool(saved["reason"]), "Explicit invalid reason")
    else:
        require(saved["reason"] is None, "No hidden valid-prefix failure")
    return {"remaining": remaining, "upper": upper_remaining, "lower": lower_remaining,
            "curvature": curvature, "alpha": alpha, "risk": risk, "invalid": invalid,
            "strict": bool(alpha > 0 and risk < 0)}
