"""Prepared native-grid adapter for an unchanged frozen MLST reference.

SOURCE ONLY at preparation: no import/build/material experiment is authorized
by this file's presence. A separate material-stress protocol must authorize any
execution. This adapter accepts log(gamma) coordinates only, never eta labels.
It restores old fixed parameters and reuses the original projection mathematics;
it learns no new algorithm, calls no optimizer, and supplies no material score.

All sixteen frozen dependency files are byte-verified before any original
source executes. The exact verified bytes, rather than reread paths, are used
for both source execution and parameter restoration. Old training Y is absent.
Only the GP-zero and frozen code-Gaussian correction parameter sets are exposed;
there is no full-data prior-KL Gaussian deployment control in this interface.
"""
from __future__ import annotations

import builtins
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import threading
from types import ModuleType

import numpy as np
from scipy.linalg import solve_triangular


DEFAULT_CHECKPOINT = Path("H:/PolymerReview/MLST_controls_20261104T154143Z_r2/checkpoint")
DEFAULT_PINS = Path(__file__).resolve().parent.parent / "FROZEN_DEPENDENCIES.json"
DEPENDENCIES_MANIFEST_SHA256 = "76bcdac23b063909160e661911c02f1ccdcb578342e548d4355ac12dea553580"
_IMPORT_LOCK = threading.RLock()


def _json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def finite(constant):
        raise ValueError("Nonfinite JSON constant: " + constant)

    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=unique, parse_constant=finite)


def _readonly(value, name, size=None):
    raw = np.asarray(value)
    if (raw.ndim != 1 or raw.dtype.kind not in "iuf" or not len(raw) or
            any(isinstance(v, (bool, np.bool_)) for v in np.asarray(value, dtype=object).flat)):
        raise ValueError(name + " must be a nonempty finite real vector")
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all() or (size is not None and out.shape != (size,)):
        raise ValueError(name + " has invalid size or nonfinite values")
    out.setflags(write=False)
    return out


def verify_frozen_dependencies(*, checkpoint_root=DEFAULT_CHECKPOINT, pins_path=DEFAULT_PINS):
    """Read/hash exactly 16 frozen sources/model-parameter files; return bytes.

    A relocated checkpoint may be passed only when every pinned file has the
    exact registered bytes. No training table, curve, eta label or manuscript is
    opened. No original source executes in this function.
    """
    manifest_bytes = Path(pins_path).read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != DEPENDENCIES_MANIFEST_SHA256:
        raise ValueError("Frozen dependency manifest SHA256 mismatch")
    manifest = _json(manifest_bytes)
    entries = manifest["scientific_bindings"]
    if len(entries) != 16:
        raise ValueError("Exactly the original sixteen frozen bindings are required")
    root = Path(checkpoint_root).resolve(strict=True)
    verified = {}
    for entry in entries:
        relative = entry["relative_to_checkpoint"]
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
            raise ValueError("Invalid checkpoint-relative dependency path")
        if relative in verified:
            raise ValueError("Duplicate dependency binding")
        path = (root / Path(*pure.parts)).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Dependency escapes the explicitly selected checkpoint")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Frozen dependency bytes mismatch: " + relative)
        verified[relative] = {"path": path, "bytes": data, "sha256": entry["sha256"]}
    return verified


class _VerifiedModules:
    """Resolve old unqualified imports from verified bytes in a private registry.

    Existing sys.modules entries named e.g. trial_geometry cannot substitute
    for the pinned code. Unique module aliases remain registered for dataclass
    introspection. No legacy import paths are left on sys.path. The frozen
    complete_curve module itself briefly prepends its old core directory; the
    path list is restored around its execution. Use a dedicated execution
    process, since sys.path is process-global during that original import.
    """
    def __init__(self, verified):
        self.sources = {Path(relative).stem: record for relative, record in verified.items()
                        if relative.endswith(".py")}
        self.modules = {}
        self.prefix = "_mlst_verified_native_grid_" + str(id(self)) + "_"

    def _import(self, name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in self.sources:
            return self.load(name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    def load(self, name):
        if name in self.modules:
            return self.modules[name]
        record = self.sources[name]
        alias = self.prefix + name
        module = ModuleType(alias)
        module.__file__, module.__package__ = str(record["path"]), ""
        module.__builtins__ = dict(vars(builtins), __import__=self._import)
        module.__frozen_source_sha256__ = record["sha256"]
        self.modules[name] = module
        sys.modules[alias] = module
        old_path = list(sys.path)
        old_no_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            exec(compile(record["bytes"], str(record["path"]), "exec"), module.__dict__)
        except BaseException:
            self.modules.pop(name, None)
            sys.modules.pop(alias, None)
            raise
        finally:
            sys.path[:] = old_path
            sys.dont_write_bytecode = old_no_bytecode
        return module


@dataclass(frozen=True)
class FrozenGridReference:
    """Parameters/law only, with no observed or unseen material response values."""
    reference: object
    log_coordinates: object
    q: object
    theta_gp: object
    theta_curvature_gaussian: object
    projection: object
    metadata: dict


def build_frozen_grid_reference(log_coordinates, *, checkpoint_root=DEFAULT_CHECKPOINT,
                                pins_path=DEFAULT_PINS):
    """Build the frozen full m/K/B on an arbitrary admitted native log-gamma grid.

    Preserves input order, rejects duplicate coordinates and extrapolation.
    The caller supplies only already admitted coordinates, not eta or a whole
    data file. Source/group eligibility, mask/query roles and protocol admission
    remain external. This function neither chooses a material protocol nor
    executes a stress test. Future conditioning must use the explicit role API.
    """
    coordinates = _readonly(log_coordinates, "log_coordinates")
    if len(np.unique(coordinates)) != len(coordinates):
        raise ValueError("Native log-gamma coordinates must be unique; no replicate aliases")
    verified = verify_frozen_dependencies(checkpoint_root=checkpoint_root, pins_path=pins_path)
    shared = _json(verified["models/SHARED_MODEL.json"]["bytes"])
    if shared.get("model_kind") != "same_local_code_jointGaussian_v1":
        raise ValueError("The pinned shared model has an unexpected family")
    q = _readonly(shared["q"], "frozen q", size=17)
    if (np.diff(q) <= 0).any():
        raise ValueError("Frozen q must be strictly increasing")
    if (coordinates < q[0]).any() or (coordinates > q[-1]).any():
        raise ValueError("Native grid exceeds frozen physical support; extrapolation forbidden")
    payload = {"reference": shared["reference"], "q": shared["q"], "heads": {},
               "generic_seed": 20261030, "kappa": 1.0, "default_head": "gp"}
    with _IMPORT_LOCK:
        modules = _VerifiedModules(verified)
        complete = modules.load("complete_curve")
        geometry = modules.load("trial_geometry")
        gaussian = modules.load("joint_code_gaussian")
    gp, restored_q, _, _, _, physical_bounds = complete.restore_model(payload)
    if not np.array_equal(q, restored_q):
        raise ValueError("Restored q differs from the frozen support")
    # Exactly the existing deployment's local projection and cross-covariance.
    projected = geometry.projection(gp, coordinates, q, "local", None)
    n = len(coordinates)
    k = projected.reference.covariance
    lower = np.linalg.cholesky(k[np.ix_(projected.q, projected.q)])
    cross = k[:n, projected.q] @ solve_triangular(lower.T, projected.directions, lower=False)
    reference = gaussian.CodeReference(projected.reference.mean[:n], k[:n, :n], cross)
    theta_zero = _readonly(np.zeros(5), "theta_gp", size=5)
    theta_curvature = _readonly(shared["theta"], "theta_curvature_gaussian", size=5)
    gaussian.unpack(theta_curvature)  # validate only; no objective or optimizer
    metadata = {
        "classification": "FROZEN_NATIVE_GRID_REFERENCE_ADAPTER_NO_NEW_LEARNING",
        "coordinate": "ln(gamma/(1 s^-1))", "response": "ln(eta/(1 Pa s))",
        "log_support": q[[0, -1]].tolist(), "physical_gamma_support": physical_bounds.tolist(),
        "input_order_preserved": True, "report_discrepancy_retained": True,
        "report_discrepancy": "frozen statistical nugget, not certified instrument noise",
        "basis": "original_fixed_local_curvature_on_q", "new_fits": 0,
        "material_eta_read": False, "old_training_Y_read": False,
        "dependency_manifest_sha256": DEPENDENCIES_MANIFEST_SHA256,
        "dependency_manifest_path": str(Path(pins_path).resolve()),
        "adapter_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_shared_model_sha256": verified["models/SHARED_MODEL.json"]["sha256"],
        "frozen_shared_model_path": str(verified["models/SHARED_MODEL.json"]["path"]),
        "verified_dependencies": {relative: record["sha256"] for relative, record in verified.items()},
        "gp_theta_family": "CodeReference_zero_correction",
        "curvature_theta_family": "frozen_five_parameter_code_Gaussian",
        "prior_KL_Gaussian_control_provided": False,
        "limitation": "Native-grid reference construction only; no new material prediction/score or active-repair evidence.",
    }
    return FrozenGridReference(reference, coordinates, q, theta_zero, theta_curvature,
                               projected, metadata)
