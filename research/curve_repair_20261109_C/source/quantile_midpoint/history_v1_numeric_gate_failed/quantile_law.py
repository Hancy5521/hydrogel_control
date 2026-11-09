"""A fixed Gaussian--Student quantile midpoint, with a Gaussian copula.

Classical Vincentization / same-copula displacement interpolation. This is
not a new flow-learning algorithm. No fitting, data loading or weight search.
The map is diagonal in latent Gaussian coordinates; the full latent joint
correlation is retained, rather than a low-rank-only reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, stdtrit, gammaln


def _array(value, ndim, label):
    raw = np.asarray(value)
    if raw.ndim != ndim or raw.dtype.kind not in "iuf" or any(
            isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError(label + " must be a real numeric array")
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(label + " must be finite")
    out.setflags(write=False)
    return out


def student4_from_normal(z):
    """No subtraction from one and no probability clipping."""
    z = np.asarray(z, dtype=float)
    lower = ndtr(-np.abs(z))
    if not np.isfinite(z).all() or np.any(lower <= 0):
        raise ArithmeticError("Latent normal tail is not representable")
    negative = stdtrit(4.0, lower)
    return np.where(z > 0, -negative, negative)


@dataclass(frozen=True)
class QuantileMidpoint:
    gaussian_location: object
    gaussian_sd: object
    student_location: object
    student_scale: object
    correlation: object

    def __post_init__(self):
        fields = ("gaussian_location", "gaussian_sd", "student_location", "student_scale")
        values = {name: _array(getattr(self, name), 1, name) for name in fields}
        n = len(values[fields[0]])
        if n == 0 or any(v.shape != (n,) for v in values.values()):
            raise ValueError("Nonempty aligned marginal vectors required")
        if np.any(values["gaussian_sd"] <= 0) or np.any(values["student_scale"] <= 0):
            raise ValueError("Strictly positive original scales required")
        r = _array(self.correlation, 2, "correlation")
        if r.shape != (n, n) or not np.array_equal(np.diag(r), np.ones(n)) or not np.allclose(
                r, r.T, atol=1e-12, rtol=1e-12):
            raise ValueError("Symmetric unit-diagonal correlation required")
        factor = np.linalg.cholesky(r)
        factor.setflags(write=False)
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "correlation", r)
        object.__setattr__(self, "_factor", factor)

    def from_latent(self, latent):
        raw = np.asarray(latent)
        z = _array(latent, 2 if raw.ndim == 2 else 1, "latent")
        if z.shape[-1] != len(self.gaussian_location):
            raise ValueError("Latent dimension differs from law")
        return .5 * (self.gaussian_location + self.gaussian_sd * z +
                     self.student_location + self.student_scale * student4_from_normal(z))

    def log_derivative(self, latent):
        z = np.asarray(latent, dtype=float)
        a = student4_from_normal(z)
        log_phi = -.5 * z * z - .5 * math.log(2 * math.pi)
        log_t = gammaln(2.5) - gammaln(2.0) - .5 * math.log(4 * math.pi) - 2.5 * np.log1p(a * a / 4)
        return np.logaddexp(np.log(self.gaussian_sd), np.log(self.student_scale) + log_phi - log_t) - math.log(2)

    def to_latent(self, values):
        raw = np.asarray(values)
        y = _array(values, 2 if raw.ndim == 2 else 1, "values")
        if y.shape[-1] != len(self.gaussian_location):
            raise ValueError("Response dimension differs from law")
        rows = y[None, :] if y.ndim == 1 else y
        output = np.empty_like(rows)
        for i, row in enumerate(rows):
            for j, target in enumerate(row):
                def f(z):
                    a = float(student4_from_normal(z))
                    return .5 * (self.gaussian_location[j] + self.gaussian_sd[j] * z +
                                 self.student_location[j] + self.student_scale[j] * a) - target
                bound = 1.0
                while f(-bound) > 0 or f(bound) < 0:
                    bound *= 2
                    if bound > 32:
                        raise ArithmeticError("Inverse requires a latent tail outside the registered numerical domain")
                output[i, j] = brentq(f, -bound, bound, xtol=1e-13, rtol=1e-14, maxiter=200)
        return output[0] if y.ndim == 1 else output

    def logpdf(self, values):
        z = self.to_latent(values)
        batch = z.ndim == 2
        white = np.linalg.solve(self._factor, z.T).T if batch else np.linalg.solve(self._factor, z)
        log_normal = (-.5 * len(self.gaussian_location) * math.log(2 * math.pi)
                      - np.log(np.diag(self._factor)).sum() - .5 * np.sum(white * white, axis=-1))
        result = log_normal - np.sum(self.log_derivative(z), axis=-1)
        if not np.isfinite(result).all():
            raise ArithmeticError("Nonfinite midpoint density")
        return result if batch else float(result)

    def quantile(self, probabilities):
        p = _array(probabilities, 1, "probabilities")
        if np.any(p <= 0) or np.any(p >= 1):
            raise ValueError("Probabilities must lie strictly inside (0,1)")
        z = np.broadcast_to(ndtri(p)[:, None], (len(p), len(self.gaussian_location)))
        return self.from_latent(z)

    def marginal_cdf(self, values):
        return ndtr(self.to_latent(values))

    def sample(self, count, rng):
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("Positive integer sample count required")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("Explicit numpy Generator required")
        return self.from_latent(rng.standard_normal((count, len(self.gaussian_location))) @ self._factor.T)

    def marginal_crps(self, values, order=192):
        """Quantile representation, split exactly at F(y) for each marginal.

        Fixed Gauss--Legendre order; endpoint tail error independently audited.
        This is a deterministic numerical integral, not measured extra labels.
        """
        y = _array(values, 1, "values")
        if y.shape != self.gaussian_location.shape:
            raise ValueError("One value per marginal required")
        if isinstance(order, bool) or not isinstance(order, int) or order < 32:
            raise ValueError("Integer quadrature order >=32 required")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        z_at_y = self.to_latent(y)
        # Evaluate both smaller-tail representations directly. In particular,
        # ndtr(+9) rounds to one but ndtr(-9) remains positive. No 1-F or
        # probability clipping is used to construct the upper-tail integral.
        masses = (ndtr(z_at_y), ndtr(-z_at_y))
        score = np.zeros_like(y)
        for mass, below in [(masses[0], True), (masses[1], False)]:
            p = .5 * (nodes[:, None] + 1) * mass[None, :]
            if np.any(p <= 0) or np.any(p >= 1):
                raise ArithmeticError("Quadrature endpoint rounded to zero/one; no clipping")
            offset = .5 * (self.gaussian_sd * ndtri(p) + self.student_scale * stdtrit(4.0, p))
            center = .5 * (self.gaussian_location + self.student_location)
            q = center + offset if below else center - offset
            integrand = p * (y - q) if below else p * (q - y)
            score += mass * np.sum(weights[:, None] * integrand, axis=0)
        return score

    def as_dict(self):
        return {"distribution": "fixed_half_gaussian_student4_quantile_transport",
                **{name: getattr(self, name).tolist() for name in
                   ("gaussian_location", "gaussian_sd", "student_location", "student_scale", "correlation")},
                "tau": .5, "df": 4, "correlation_is_latent_normal": True,
                "conditional_scope": "fixed observed set; not asserted coherent across masks",
                "not_a_new_transport_theorem": True}
