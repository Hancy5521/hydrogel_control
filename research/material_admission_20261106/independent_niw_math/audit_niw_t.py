"""Independent, synthetic-only NIW/conditional-t/CRPS mathematical audit.

No producer imports, random sampling, material response inputs or network calls.
Only --plan/--approval JSON are read; the fixed arrays below are hypothetical.
Scientific imports and arithmetic occur only after SHA-bound approval checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import warnings


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def close(actual, expected, atol, rtol=0.0):
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    bound = atol + rtol * float(np.max(np.abs(np.asarray(expected))))
    if not math.isfinite(error) or error > bound:
        raise AssertionError({"actual": np.asarray(actual).tolist(),
                              "expected": np.asarray(expected).tolist(),
                              "absolute_error": error, "tolerance": bound})
    return error


def batch_niw(x, mu, kappa, nu, psi):
    n = len(x)
    center = np.mean(x, axis=0)
    scatter = (x - center).T @ (x - center)
    delta = center - mu
    kn = kappa + n
    return ((kappa * mu + n * center) / kn, kn, nu + n,
            psi + scatter + (kappa * n / kn) * np.outer(delta, delta))


def sequential_niw(x, mu, kappa, nu, psi):
    mean = mu.copy()
    scale = psi.copy()
    for row in x:
        deviation = row - mean
        scale = scale + (kappa / (kappa + 1)) * np.outer(deviation, deviation)
        mean = mean + deviation / (kappa + 1)
        kappa += 1
        nu += 1
    return mean, kappa, nu, scale


def predictive(mu, kappa, nu, psi):
    df = nu - len(mu) + 1
    return mu, ((kappa + 1) / (kappa * df)) * psi, df


def conditional_target(mean, shape, df, target, observed, values):
    if target in observed or len(set(observed)) != len(observed):
        raise ValueError("Target cannot be observed; observed indices must be unique.")
    if len(observed) != len(values):
        raise ValueError("Observation dimension mismatch.")
    if not observed:
        return float(mean[target]), float(shape[target, target]), float(df)
    oo = shape[np.ix_(observed, observed)]
    to = shape[target, observed]
    residual = np.asarray(values) - mean[observed]
    solved = np.linalg.solve(oo, residual)
    delta = float(residual @ solved)
    loc = float(mean[target] + to @ solved)
    schur = float(shape[target, target] - to @ np.linalg.solve(oo, to))
    conditioned_df = df + len(observed)
    conditioned_shape = (df + delta) / conditioned_df * schur
    if conditioned_shape <= 0:
        raise ArithmeticError("Conditional shape must be strictly positive.")
    return loc, conditioned_shape, float(conditioned_df)


def student_crps_closed(df, loc, scale, value):
    if not all(math.isfinite(v) for v in (df, loc, scale, value)):
        raise ValueError("Finite arguments required.")
    if df <= 1 or scale <= 0:
        raise ValueError("Finite-first-moment t requires df>1 and scale>0.")
    z = (value - loc) / scale
    first = z * (2 * t.cdf(z, df) - 1)
    moment = 2 * t.pdf(z, df) * (df + z * z) / (df - 1)
    pair = 2 * math.sqrt(df) / (df - 1) * math.exp(
        betaln(0.5, df - 0.5) - 2 * betaln(0.5, df / 2))
    return float(scale * (first + moment - pair))


def integrate(f, lower, upper):
    value, error = quad(f, lower, upper, epsabs=2e-10, epsrel=2e-10,
                        limit=400)
    if not math.isfinite(value) or not math.isfinite(error):
        raise ArithmeticError("Nonfinite quadrature output.")
    return float(value), float(error)


def crps_cdf_quadrature(df, loc, scale, value):
    z = (value - loc) / scale
    left, le = integrate(lambda x: float(t.cdf(x, df)) ** 2, -math.inf, z)
    right, re = integrate(lambda x: float(t.sf(x, df)) ** 2, z, math.inf)
    return scale * (left + right), scale * (le + re)


def crps_energy_quadrature(df, loc, scale, value):
    z = (value - loc) / scale
    lower, le = integrate(lambda x: (z - x) * float(t.pdf(x, df)), -math.inf, z)
    upper, ue = integrate(lambda x: (x - z) * float(t.pdf(x, df)), z, math.inf)
    # A(x)=E|T-x| from the t first partial moment. This uses a density
    # integral, not a squared-CDF CRPS integral or the beta pair constant.
    def half_pair_integrand(x):
        a = x * (2 * t.cdf(x, df) - 1)
        a += 2 * t.pdf(x, df) * (df + x * x) / (df - 1)
        return float(a * t.pdf(x, df))
    # Symmetry: 0.5*integral_{-inf}^inf A(x)f(x)dx = integral_0^inf.
    half_pair, pe = integrate(half_pair_integrand, 0, math.inf)
    return scale * (lower + upper - half_pair), scale * (le + ue + pe)


def run_niw():
    mu0 = np.zeros(3)
    psi0 = 3 * np.eye(3)
    prior = (mu0, 1.0, 7.0, psi0)
    x = np.array([[-1.0, 0.2, 1.0], [0.0, -0.5, 0.7],
                  [0.6, 0.8, -0.3], [1.4, -0.1, 0.4]])
    batch = batch_niw(x, *prior)
    seq = sequential_niw(x, *prior)
    reverse = sequential_niw(x[::-1], *prior)
    errors = [close(a, b, 2e-13) for a, b in zip(batch, seq)]
    errors += [close(a, b, 2e-13) for a, b in zip(batch, reverse)]
    pm, ps, pdf = predictive(*prior)
    errors += [close(pdf, 5.0, 0), close(ps, 1.2 * np.eye(3), 2e-13)]
    expected_sigma = psi0 / (prior[2] - 3 - 1)
    expected_cov = expected_sigma * (1 + 1 / prior[1])
    errors += [close(expected_sigma, np.eye(3), 2e-13),
               close(ps * pdf / (pdf - 2), expected_cov, 2e-13)]
    diagonal = np.array([2.0, 0.5, 1.7])
    offset = np.array([-3.0, 4.0, 2.0])
    transformed = batch_niw(x * diagonal + offset, mu0 * diagonal + offset,
                            prior[1], prior[2], psi0 * np.outer(diagonal, diagonal))
    errors += [close(transformed[0], batch[0] * diagonal + offset, 2e-13),
               close(transformed[3], batch[3] * np.outer(diagonal, diagonal), 2e-13)]
    return {"max_absolute_error": max(errors), "fixed_synthetic_rows": 4,
            "prior_E_Sigma": expected_sigma.tolist(),
            "prior_predictive_df": pdf, "prior_predictive_shape": ps.tolist(),
            "prior_predictive_covariance": expected_cov.tolist(),
            "posterior": {"mu": batch[0].tolist(), "kappa": batch[1],
                          "nu": batch[2], "psi": batch[3].tolist()}}, [prior, batch]


def run_conditionals(states):
    rows = []
    max_error = 0.0
    max_affine_error = 0.0
    diagonal = np.array([2.0, 0.5, 1.7])
    offset = np.array([-3.0, 4.0, 2.0])
    for state_number, state in enumerate(states):
        mean, shape, df = predictive(*state)
        for target in range(3):
            other = [i for i in range(3) if i != target]
            for count in (0, 1, 2):
                observed = other[:count]
                values = mean[observed] + np.array([0.25, -1.1])[:count]
                loc, cs, cdf = conditional_target(mean, shape, df, target,
                                                 observed, values)
                tm = mean * diagonal + offset
                ts = shape * np.outer(diagonal, diagonal)
                tv = values * diagonal[observed] + offset[observed]
                transformed = conditional_target(tm, ts, df, target, observed, tv)
                max_affine_error = max(max_affine_error,
                    close(transformed, [loc * diagonal[target] + offset[target],
                                         cs * diagonal[target] ** 2, cdf], 2e-12))
                subset = [target] + observed
                if observed:
                    denominator = float(multivariate_t.logpdf(
                        values, loc=mean[observed],
                        shape=shape[np.ix_(observed, observed)], df=df))
                else:
                    denominator = 0.0
                for z in (-2.0, 0.0, 1.5):
                    y = loc + math.sqrt(cs) * z
                    numerator = float(multivariate_t.logpdf(
                        np.array([y] + list(values)), loc=mean[subset],
                        shape=shape[np.ix_(subset, subset)], df=df))
                    ratio = math.exp(numerator - denominator)
                    density = float(t.pdf(y, cdf, loc=loc, scale=math.sqrt(cs)))
                    error = close(density, ratio, 2e-12, 2e-11)
                    max_error = max(max_error, error)
                    rows.append({"state": state_number, "target": target,
                                 "proxy_count": count, "z": z, "df": cdf,
                                 "shape": cs, "density_ratio_error": error})
    # Uncorrelated components of a t still share random scale. A remote
    # proxy changes conditional scale even when linear correlation is zero.
    mean, shape, df = predictive(*states[0])
    near = conditional_target(mean, shape, df, 0, [1], [0.0])
    far = conditional_target(mean, shape, df, 0, [1], [3.0])
    if not (far[1] > near[1] and near[2] == df + 1 == far[2]):
        raise AssertionError("Conditional-t scale dependency lost.")
    return {"density_coordinates": len(rows), "rows": rows,
            "max_density_absolute_error": max_error,
            "max_affine_absolute_error": max_affine_error,
            "prior_zero_linear_correlation_near_proxy_shape": near[1],
            "prior_zero_linear_correlation_far_proxy_shape": far[1]}


def run_crps():
    rows = []
    max_cdf_closed = max_cdf_energy = max_quad_error = 0.0
    for df in (1.5, 2.5, 5.0, 12.0):
        for z in (-2.0, 0.0, 0.75):
            for loc, scale in ((-1.0, 0.7), (3.0, 2.3)):
                value = loc + scale * z
                closed = student_crps_closed(df, loc, scale, value)
                cdf, cdf_error = crps_cdf_quadrature(df, loc, scale, value)
                energy, energy_error = crps_energy_quadrature(df, loc, scale, value)
                e1 = close(cdf, closed, 2e-8, 2e-8)
                e2 = close(cdf, energy, 2e-8, 2e-8)
                max_cdf_closed = max(max_cdf_closed, e1)
                max_cdf_energy = max(max_cdf_energy, e2)
                max_quad_error = max(max_quad_error, cdf_error, energy_error)
                close(student_crps_closed(df, 0.0, 1.0, z) * scale,
                      closed, 2e-12, 2e-12)
                if min(cdf, energy, closed) < -2e-10:
                    raise AssertionError("CRPS cannot be negative.")
                rows.append({"df": df, "z": z, "loc": loc, "scale": scale,
                             "closed": closed, "CDF_quadrature": cdf,
                             "energy_density_quadrature": energy,
                             "CDF_reported_quad_error": cdf_error,
                             "energy_reported_quad_error": energy_error})
    return {"fixed_cases": len(rows), "max_CDF_closed_error": max_cdf_closed,
            "max_CDF_energy_error": max_cdf_energy,
            "max_reported_quad_error": max_quad_error, "rows": rows,
            "heavy_tail_case_df1p5": "Finite first moment and CRPS; variance is infinite. No finite-variance claim."}


def run_guards():
    rejected = []
    for args in ((1.0, 0.0, 1.0, 0.0), (0.9, 0.0, 1.0, 0.0),
                 (5.0, 0.0, 0.0, 0.0), (5.0, 0.0, 1.0, math.nan)):
        try:
            student_crps_closed(*args)
        except ValueError:
            rejected.append(list(args))
        else:
            raise AssertionError("Invalid CRPS input accepted.")
    return {"invalid_argument_cases_rejected": len(rejected),
            "conditions": ["df<=1", "nonpositive scale", "nonfinite value"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--approval", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    plan_path = Path(args.plan)
    approval = json.loads(Path(args.approval).read_text(encoding="utf-8-sig"))
    plan = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    script_hash = sha(__file__)
    if plan["script_sha256"] != script_hash:
        raise RuntimeError("Independent audit source changed after plan.")
    if approval.get("scope") != "INDEPENDENT_SYNTHETIC_NIW_T_CRPS_ONCE":
        raise RuntimeError("Incorrect approval scope.")
    if approval.get("plan_sha256") != sha(plan_path):
        raise RuntimeError("Approved plan hash mismatch.")
    if approval.get("script_sha256") != script_hash:
        raise RuntimeError("Approved source hash mismatch.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = utc()
    global np, quad, t, multivariate_t, betaln, scipy
    import numpy as np
    import scipy
    from scipy.integrate import quad, IntegrationWarning
    from scipy.special import betaln
    from scipy.stats import t, multivariate_t
    warnings.simplefilter("error", IntegrationWarning)
    results = []
    states = None
    for name, func in (("NIW_batch_sequential_prior_and_affine", run_niw),
                       ("conditional_t_joint_over_marginal_density", lambda: run_conditionals(states)),
                       ("Student_t_CRPS_CDF_energy_and_closed", run_crps),
                       ("finite_first_moment_and_invalid_input_guards", run_guards)):
        try:
            details = func()
            if name == "NIW_batch_sequential_prior_and_affine":
                details, states = details
            results.append({"name": name, "status": "PASS", "details": details})
        except Exception as error:
            results.append({"name": name, "status": "FAIL", "error": str(error),
                            "traceback": traceback.format_exc()})
    payload = {"status": "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL",
               "start_utc": start, "end_utc": utc(), "script_sha256": script_hash,
               "plan_sha256": sha(plan_path), "approval_sha256": sha(args.approval),
               "Python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
               "families": results, "material_response_reads": 0,
               "producer_imports_or_executions": 0, "random_samples": 0,
               "scope": "Fixed hypothetical mathematical controls only; not SUT validation, materials effectiveness, numerical interval proof or independent material evidence."}
    result_path = output / "INDEPENDENT_NIW_T_CRPS_AUDIT.json"
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "result": str(result_path),
                      "sha256": sha(result_path),
                      "families": [{"name": r["name"], "status": r["status"]} for r in results]}))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
