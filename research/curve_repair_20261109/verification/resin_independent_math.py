"""Independent dense calculations; no production or frozen-runtime imports.

The original GP/local-curvature formulas were read from pinned source, then
implemented directly here. In particular B is built as K_rq F' (F K_qq F')^-1/2,
not via the production Cholesky/support-directions construction. All arguments
are explicit arrays or already released artifacts. No material reader exists.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import ndtr

MODEL_SHA = "d72fc262fbae0e6fecc36568f0638166fe95e991bd5f7df100ab1434097d82d1"
INPUT_SHA = "dd1e7b41b8db7076a6b55a68f94d41cc0b2e2e323ca1fd9596c57b9c7b4a2900"
SOURCE_SHA = "a2a7a84d92bf84a69e6f6c49e0ae976d916cc63fa46ff28bccd08a7ebdd4e218"
TARGET = list(range(17, 25))
REFERENCES = ("gp_theta0", "curvature_frozen_theta")
POLICIES = ("fixed_first6", "spread6", "random_uniform6", "max_variance6", "expected_crps6")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path, expected=None):
    if expected is not None and sha(path) != expected:
        raise ValueError("Independent input bytes mismatch: " + str(path))
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

def write_new(path, payload):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")

def actual_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

class Comparisons:
    def __init__(self):
        self.groups = {}
    def close(self, name, actual, expected, *, atol=3e-9, rtol=3e-9):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError(name + ": shape/nonfinite mismatch")
        error = float(np.max(np.abs(a - b), initial=0.0))
        group = self.groups.setdefault(name, {"elements": 0, "maximum_absolute_error": 0.0,
                                              "atol": atol, "rtol": rtol})
        group["elements"] += int(a.size)
        group["maximum_absolute_error"] = max(group["maximum_absolute_error"], error)
        if not np.allclose(a, b, atol=atol, rtol=rtol):
            raise AssertionError(name + ": independent numeric mismatch, max=" + str(error))

def recipe_roles(inputs):
    histories = {}
    members = {}
    for run in inputs["per_run"]:
        members[run["member"]] = run
        histories.setdefault(run["recipe"], set()).add(run["thermal_history"])
    result = {}
    for history, labels in (("ascending_35_first", ["dev"]*2+["future_expansion"]*2+["final_confirm"]*3),
                            ("descending_35_last", ["dev", "future_expansion", "final_confirm"])):
        recipes = [name for name, value in histories.items() if value == {history}]
        recipes.sort(key=lambda name: (hashlib.sha256(("mlst-resin-role-v1|" + name).encode()).hexdigest(), name))
        if len(recipes) != len(labels):
            raise AssertionError("Unexpected independent recipe history population")
        result.update(zip(recipes, labels))
    if len(result) != 10 or len(members) != 50:
        raise AssertionError("Unexpected independent input population")
    return result, members

def native_reference(log_gamma, model):
    """Full report law and two exact prior-whitened local-curvature channels."""
    t, q = np.asarray(log_gamma, float), np.asarray(model["q"], float)
    ref = model["reference"]
    norm = ref["normalization"]
    u = (t - norm["t_center"]) / norm["t_scale"]
    v = (q - norm["t_center"]) / norm["t_scale"]
    level, slope, smooth, length, noise = ref["hyper_values"]
    def covariance(a, b):
        z = math.sqrt(3.0) * np.abs(a[:, None] - b[None, :]) / length
        return norm["y_scale"]**2 * (level**2 + slope**2 * a[:, None] * b[None, :]
              + smooth**2 * (1 + z) * np.exp(-z))
    beta = ref["beta_standardized"]
    mean = norm["y_center"] + norm["y_scale"] * (beta[0] + beta[1] * u)
    cov = covariance(u, u) + np.eye(len(t)) * (norm["y_scale"]*noise)**2
    kq, crossq = covariance(v, v), covariance(u, v)
    # Independently assemble quadrature-weighted second divided differences.
    differences = np.zeros((len(q)-2, len(q)))
    left, right = np.diff(q)[:-1], np.diff(q)[1:]
    for j in range(len(q)-2):
        a, b = left[j], right[j]
        differences[j, j] = 2.0 / (a * (a + b))
        differences[j, j+1] = -2.0 / (a * b)
        differences[j, j+2] = 2.0 / (b * (a + b))
    centres = q[0] + (q[-1]-q[0]) * np.array([1/3, 2/3])
    windows = np.clip(1 - ((q[None, 1:-1] - centres[:, None]) / ((q[-1]-q[0])/3))**2,
                      a_min=0, a_max=None)**3
    windows *= (left + right)[None, :] / 2
    f = (windows / windows.sum(axis=1)[:, None]) @ differences
    p = f @ kq @ f.T
    eigenvalues, vectors = np.linalg.eigh((p+p.T)/2)
    if eigenvalues.min() <= 0:
        raise AssertionError("Invalid independent local-code covariance")
    invroot = (vectors * (1 / np.sqrt(eigenvalues))) @ vectors.T
    b = crossq @ f.T @ invroot
    residual = (cov - b @ b.T)
    residual = (residual + residual.T)/2
    if np.linalg.eigvalsh(residual).min() < -1e-9:
        raise AssertionError("Independent full residual is not PSD")
    return mean, cov, b, float(np.linalg.eigvalsh(residual).min())

def joint(mean, covariance, cross, theta):
    theta = np.asarray(theta, float)
    lower = np.array([[math.exp(theta[2]), 0.0], [theta[3], math.exp(theta[4])]])
    return mean + cross @ theta[:2], covariance + cross @ (lower @ lower.T - np.eye(2)) @ cross.T

def condition(mean, covariance, anchors, readings):
    o = list(anchors)
    # Dense solve chosen independently of SUT's Cholesky path.
    gain = np.linalg.solve(covariance[np.ix_(o, o)], covariance[o, :]).T
    updated_mean = mean + gain @ (np.asarray(readings) - mean[o])
    updated_cov = covariance - gain @ covariance[o, :]
    updated_cov = (updated_cov + updated_cov.T)/2
    updated_mean[o] = readings
    updated_cov[o, :] = 0.0
    updated_cov[:, o] = 0.0
    return updated_mean, updated_cov

def covariance_only(covariance, anchors):
    if not anchors:
        return np.array(covariance, copy=True)
    _, result = condition(np.zeros(len(covariance)), covariance, anchors, np.zeros(len(anchors)))
    return result

def planned_mask(covariance, criterion):
    selected = [0, 1]
    trace = []
    for step in range(4):
        post = covariance_only(covariance, selected)
        candidates = [i for i in range(2, 17) if i not in selected]
        values = {}
        for i in candidates:
            if criterion == "max_variance6":
                values[i] = float(post[i, i])
            else:
                vt = np.diag(post)[TARGET]
                after = vt - post[TARGET, i]**2 / post[i, i]
                if np.min(after) < -2e-10:
                    raise AssertionError("Independent query Schur has negative variance")
                values[i] = float(np.mean(np.sqrt(vt) - np.sqrt(np.maximum(after, 0))) / math.sqrt(math.pi))
        chosen = max(candidates, key=lambda i: (values[i], -i))
        selected.append(chosen)
        trace.append({"query": chosen, "criterion_value": values[chosen]})
    return selected, trace

def score_gaussian(mean, covariance, truth):
    mean, cov, truth = np.asarray(mean), np.asarray(covariance), np.asarray(truth)
    sd = np.sqrt(np.diag(cov))
    if np.min(sd) <= 0:
        raise AssertionError("Independent scorer requires positive marginal variance")
    z = (truth - mean) / sd
    crps = sd * (z * (2*ndtr(z)-1) + 2*np.exp(-z*z/2)/math.sqrt(2*math.pi) - 1/math.sqrt(math.pi))
    error = truth-mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign != 1:
        raise AssertionError("Independent scoring covariance not SPD")
    # Use determinant/solve, distinct from production Cholesky scoring.
    nll = (len(truth)*math.log(2*math.pi) + logdet + error@np.linalg.solve(cov, error))/2/len(truth)
    cutoff = 1.6448536269514722
    return {"crps": float(np.mean(crps)), "joint_nll_per_target": float(nll),
            "log_rmse": float(np.sqrt(np.mean(error**2))),
            "coverage90": float(np.mean(np.abs(error) <= cutoff*sd)),
            "width90": float(np.mean(2*cutoff*sd)), "crps_points": crps.tolist()}
