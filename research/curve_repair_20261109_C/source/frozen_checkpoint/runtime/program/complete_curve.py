"""Complete a curve from six real anchors using a saved, frozen model.

No fitting is performed. Copy program/ and core/ together to another machine;
the only third-party dependencies are NumPy and SciPy. See USER_API.md.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True  # Do not write into the frozen core directory.
CORE = Path(__file__).resolve().parent.parent / "core"
sys.path.insert(0, str(CORE))

import numpy as np

from coordinate_reference import CoordinateGP, Normalization
from conditional_sampler import sample
from energy_correction import BoundedPotential
from gaussian_control import GaussianDesign, context_feature, predict_gaussian
from trial_geometry import projection


HEADS = ("gp", "generic", "local", "gaussian")


def _vector(value, name, *, size=None, positive=False):
    if any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} cannot contain booleans")
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    result = np.array(raw, dtype=float, copy=True)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    if size is not None and result.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} numbers")
    if positive and (result <= 0).any():
        raise ValueError(f"{name} must be strictly positive")
    return result


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return int(value)


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require(value, names, name):
    missing = set(names) - set(value)
    if missing:
        raise ValueError(f"{name} missing fields: {', '.join(sorted(missing))}")


def restore_model(payload):
    """Restore CoordinateGP.protocol_info(), with no optimizer or fit call."""
    payload = _object(payload, "model")
    _require(payload, ("reference", "q", "heads", "generic_seed", "kappa", "default_head"), "model")
    reference = _object(payload["reference"], "reference")
    _require(reference, ("normalization", "hyper_values", "beta_standardized"), "reference")
    normal = _object(reference["normalization"], "normalization")
    normal_keys = ("t_center", "t_scale", "y_center", "y_scale")
    _require(normal, normal_keys, "normalization")
    normalization = Normalization(**{key: normal[key] for key in normal_keys})
    hyper = _vector(reference["hyper_values"], "hyper_values", size=5, positive=True)
    beta = _vector(reference["beta_standardized"], "beta_standardized", size=2)
    model = CoordinateGP.fixed(normalization, np.log(hyper), beta)
    q = _vector(payload["q"], "q", size=17)
    if (np.diff(q) <= 0).any():
        raise ValueError("q must contain 17 strictly increasing frozen log coordinates")
    generic_seed = _integer(payload["generic_seed"], "generic_seed", 0)
    kappa = payload["kappa"]
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float)) or not np.isfinite(kappa) or kappa != 1.0:
        raise ValueError("This registered control requires kappa=1")
    stored_heads = _object(payload["heads"], "heads")
    if set(stored_heads) - {"generic", "local", "gaussian"}:
        raise ValueError("heads may contain only generic, local and gaussian")
    heads = {}
    for key, record in stored_heads.items():
        record = _object(record, key + " head")
        _require(record, ("theta",), key + " head")
        heads[key] = _vector(record["theta"], key + " theta", size=5)
    default = payload["default_head"]
    if not isinstance(default, str) or default not in HEADS or (default != "gp" and default not in heads):
        raise ValueError("default_head must name gp or an available stored head")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        bounds = np.exp(q[[0, -1]])
    if not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
        raise ValueError("q must define a finite positive physical support interval")
    return model, q, heads, generic_seed, default, bounds


def complete_curve(model_payload, input_payload, *, head=None, draws=64, seed=20261030):
    """Return generated physical-unit draws for any nonempty query length.

    Query coordinates that exactly equal stored anchors reuse those observations;
    all remaining query coordinates are one full joint conditional draw.
    """
    draws = _integer(draws, "draws", 1)
    seed = _integer(seed, "seed", 0)
    model, q, heads, generic_seed, default, bounds = restore_model(model_payload)
    head = default if head is None else head
    if not isinstance(head, str) or head not in HEADS:
        raise ValueError("head must be gp, generic, local or gaussian")
    if head != "gp" and head not in heads:
        raise ValueError(f"Requested head '{head}' is absent from this model")
    payload = _object(input_payload, "input")
    required = ("observed_gamma", "observed_eta", "target_gamma")
    _require(payload, required, "input")
    if set(payload) != set(required):
        raise ValueError("input accepts only observed_gamma, observed_eta and target_gamma")
    observed_gamma = _vector(payload["observed_gamma"], "observed_gamma", size=6, positive=True)
    observed_eta = _vector(payload["observed_eta"], "observed_eta", size=6, positive=True)
    target_gamma = _vector(payload["target_gamma"], "target_gamma", positive=True)
    for name, coordinates in (("observed_gamma", observed_gamma), ("target_gamma", target_gamma)):
        if np.unique(coordinates).size != coordinates.size:
            raise ValueError(f"{name} coordinates must be unique")
        if (coordinates < bounds[0]).any() or (coordinates > bounds[1]).any():
            raise ValueError(f"{name} must stay within frozen support [{bounds[0]}, {bounds[1]}] 1/s")

    anchors = {float(gamma): index for index, gamma in enumerate(observed_gamma)}
    observed_target = [anchors.get(float(gamma)) for gamma in target_gamma]
    unknown = np.array([index for index, known in enumerate(observed_target) if known is None], dtype=int)
    physical = np.empty((draws, len(target_gamma)))
    sampler = {"method": "stored observed reports only", "proposals": 0,
               "numerical_jitter": 0.0, "new_replicates": False}
    if unknown.size:
        tc, yc = np.log(observed_gamma), np.log(observed_eta)
        th = np.log(target_gamma[unknown])
        # Exact physical duplicates are handled above. Reject unresolved floating
        # aliases rather than treating an anchor as a new independent report.
        if np.unique(tc).size != 6 or np.unique(th).size != len(th) or np.isin(th, tc).any():
            raise ValueError("Distinct gamma inputs alias in floating-point log coordinates")
        if head in ("generic", "local"):
            generic = np.linalg.qr(np.random.default_rng(generic_seed).normal(size=(len(q), 2)))[0]
            reports = np.concatenate((tc, th))
            projected = projection(model, reports, q, head, generic)
            query = projected.query(np.arange(6), yc, np.arange(6, len(reports)))
            budget = max(100000, 32 * draws)
            result = sample(query, BoundedPotential(2, 1.0), heads[head],
                            n=draws, seed=seed, max_proposals=budget)
            log_draws = result["target"]
            sampler.update(method=result["sampler"], proposals=int(result["proposals"]),
                           max_proposals=budget, acceptance_lower_bound=result["acceptance_lower_bound"],
                           joint_target_count=int(unknown.size), generic_direction_seed=generic_seed)
        else:
            law = model.condition(tc, yc, th, report=True, new_replicates=False)
            mean, covariance = law.mean, law.cov
            if head == "gaussian":
                context = model.joint(tc, report=True)
                h = context_feature(yc, context.mean, context.cov)
                u = 2.0 * (th - q[0]) / (q[-1] - q[0]) - 1.0
                corrected = predict_gaussian(heads[head], GaussianDesign.from_reference(mean, covariance, u, h))
                mean, covariance = corrected.mean, corrected.covariance
            root = np.linalg.cholesky(covariance)
            log_draws = mean + np.random.default_rng(seed).normal(size=(draws, len(mean))) @ root.T
            sampler.update(method="full-covariance Gaussian Cholesky draws", joint_target_count=int(unknown.size))
        if not np.isfinite(log_draws).all():
            raise ValueError("Nonfinite log-viscosity draws; no numerical fallback")
        with np.errstate(over="raise", under="raise", invalid="raise"):
            physical[:, unknown] = np.exp(log_draws)
    for target, anchor in enumerate(observed_target):
        if anchor is not None:
            physical[:, target] = observed_eta[anchor]
    if not np.isfinite(physical).all() or (physical <= 0).any():
        raise ValueError("Generated eta draws must be finite and positive")
    lower, median, upper = np.quantile(physical, [0.05, 0.5, 0.95], axis=0)
    draw_list = physical.tolist()
    lower, median, upper = lower.tolist(), median.tolist(), upper.tolist()
    # Preserve the user's original JSON numeric values at stored observations,
    # including integer literals, rather than round-tripping exp(log(eta)).
    for target, anchor in enumerate(observed_target):
        if anchor is not None:
            original = payload["observed_eta"][anchor]
            for row in draw_list:
                row[target] = original
            lower[target] = median[target] = upper[target] = original
    return {
        "schema_version": 1, "actual_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "head": head, "seed": seed, "draws_count": draws,
        "units": {"gamma": "1/s", "eta": "Pa s"},
        "observed_gamma": payload["observed_gamma"], "observed_eta": payload["observed_eta"],
        "target_gamma": payload["target_gamma"],
        "target_is_observed": [value is not None for value in observed_target],
        "eta_draws": draw_list, "eta_median": median, "eta_p05": lower, "eta_p95": upper,
        "summary_method": "Empirical 5th, 50th and 95th percentiles of the generated draws; Monte Carlo approximations at unobserved targets.",
        "is_measured_data": False,
        "data_role": "Generated conditional curve completions, not new measurements. Repeated stored anchors are the original observations.",
        "anchors_preserved_exactly": True, "sampler": sampler,
        "domain": {"gamma_min": float(bounds[0]), "gamma_max": float(bounds[1]),
                   "extrapolation_allowed": False, "required_observations": 6,
                   "limitation": "Only queries inside this frozen shear-rate support with six unique real anchors are supported; model-generated values do not establish accuracy for a new material."},
        "optimization_performed": False,
    }


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse_json(text):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=_unique_object)


def read_json(path):
    return _parse_json(Path(path).read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, required=True, help="Saved deployment MODEL.json")
    parser.add_argument("--input", type=Path, required=True, help="Six observed anchors and requested gamma JSON")
    parser.add_argument("--output", type=Path, required=True, help="Generated conditional completions JSON")
    parser.add_argument("--head", choices=HEADS, default=None, help="Default: model.default_head")
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in (args.model.resolve(), args.input.resolve()):
            raise ValueError("output must differ from the model and input paths")
        model_bytes, input_bytes = args.model.read_bytes(), args.input.read_bytes()
        result = complete_curve(_parse_json(model_bytes.decode("utf-8")), _parse_json(input_bytes.decode("utf-8")),
                                head=args.head, draws=args.draws, seed=args.seed)
        result["model_sha256"] = hashlib.sha256(model_bytes).hexdigest()
        result["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        args.output.write_text(text, encoding="utf-8")
    except Exception as exc:
        parser.exit(2, f"complete_curve: {type(exc).__name__}: {exc}\n")
    print(f"Wrote {args.draws} generated draws at {len(result['target_gamma'])} targets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
