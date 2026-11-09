"""Load the byte-pinned pure Gaussian mathematics, without material I/O or fit.

The default path is the preserved local MLST review package. A byte-identical
copy can be supplied explicitly. The source is hashed before execution, and
the verified bytes themselves are compiled, avoiding a hash/read race and
avoiding writes to the frozen directory. This does not load deployment weights.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType


FROZEN_PACKAGE_PATH = Path(
    "H:/PolymerReview/MLST_controls_20261104T154143Z_r2"
)
FROZEN_CORE_PATH = (
    FROZEN_PACKAGE_PATH / "checkpoint/runtime/core/joint_code_gaussian.py"
)
FROZEN_CORE_SHA256 = (
    "0c8a57c1bdf96e1c880ffd1ba2258e1b379952b844e960a7c67d6cdcdc65e299"
)
FROZEN_MODEL_PATH = FROZEN_PACKAGE_PATH / "checkpoint/models/SHARED_MODEL.json"
FROZEN_MODEL_SHA256 = (
    "d72fc262fbae0e6fecc36568f0638166fe95e991bd5f7df100ab1434097d82d1"
)


def load_frozen_core(path: str | Path | None = None) -> ModuleType:
    """Return the original mathematical module after its exact SHA256 check.

    Import defines CodeReference, CurveTask, ConditionalObjective and fit; it
    executes no optimizer. Callers own the role boundary and must not call fit
    on consumed material records during this extension's software-only stage.
    """
    source_path = FROZEN_CORE_PATH if path is None else Path(path)
    source_path = source_path.resolve(strict=True)
    source = source_path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != FROZEN_CORE_SHA256:
        raise ValueError(
            f"Frozen Gaussian core SHA256 mismatch: {source_path}; "
            f"expected {FROZEN_CORE_SHA256}, got {digest}"
        )
    name = "_mlst_frozen_joint_code_gaussian_" + digest
    spec = importlib.util.spec_from_file_location(name, source_path)
    if spec is None:
        raise ImportError(f"Cannot create module spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module  # dataclasses resolve annotations through this map.
    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    module.__frozen_source_sha256__ = digest
    return module
