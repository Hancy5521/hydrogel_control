"""Frozen inputs, historical provenance and sequential real-report disclosure.

No probability optimization lives here. The gzip reader is a trusted disclosure
boundary, not an assertion that loading a complete public file physically hides
its bytes. Only registered cells leave that boundary for the model controller.
"""
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

FILES = ("empirical_native.py", "simplex_transport.py", "calibration_views.py",
         "transport_io.py", "run_transport.py", "score_transport.py")
METHODS = ("RF_identity", "RF_beta", "RF_global4", "RF_anchor4", "RF_auto4", "GP10", "old3_NIG")
BATCHES = ("validation1", "validation2", "validation3")
FIELDS = ("e_form", "gap opt", "bulk modulus")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("nonfinite JSON constant " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sources():
    folder = Path(__file__).resolve().parent
    return {name: sha(folder / name) for name in FILES}


def validate(planfile, approvalfile, phase):
    plan, approval = read(planfile), read(approvalfile)
    if (plan.get("status") != "REGISTERED" or approval.get("approved") is not True
            or approval.get("scope") != "REFERENCE_TRANSPORT_FRESH192"
            or approval.get("plan_sha256") != sha(planfile)
            or phase not in approval.get("allowed_phases", [])):
        raise ValueError("exact registered root phase approval required")
    if plan["source_sha256"] != sources():
        raise ValueError("six registered scientific sources changed")
    for section in ("inputs", "dependencies"):
        for value in plan[section].values():
            if sha(value["file"]) != value["sha256"]:
                raise ValueError("registered dependency changed: " + value["file"])
    roles = read(plan["inputs"]["roles"]["file"])
    new = roles["new_development_consumed_component_ids"]
    old = roles["all_prior_excluded_component_ids"]
    remaining = roles["remaining_unselected_component_ids_in_registered_hash_order"]
    groups = [set(new), set(old), set(remaining)]
    if ([len(new), len(old), len(remaining)] != [192, 1216, 7434]
            or any(len(s) != len(v) for s, v in zip(groups, (new, old, remaining)))
            or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2)))):
        raise ValueError("exact disjoint original-component roles required")
    rows = [r for role in BATCHES for r in roles["representatives"][role]]
    if ({r["component_id"] for r in rows} != set(new)
            or len(rows) != 192 or any(len(roles["representatives"][role]) != 64 for role in BATCHES)):
        raise ValueError("three fixed64 fresh batches required")
    return plan, roles


def bootstrap(plan):
    """Import only explicit frozen sources; no old runner/fit is executed."""
    loaded = {}
    for key, name in (("nig_core", "relation_posterior"), ("growth_models", "growth_models"),
                      ("reference_baselines", "baselines"), ("growth_probability", "probability")):
        pin = plan["dependencies"][key]
        path = Path(pin["file"]).resolve()
        sys.path.insert(0, str(path.parent))
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != path or sha(path) != pin["sha256"]:
            raise ValueError("exact imported dependency required: " + name)
        loaded[name] = module
    return loaded


def scalar_scorer(plan):
    pin = plan["dependencies"]["legacy_scalar_score"]
    path = Path(pin["file"]).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_transport_frozen_scalar_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_law


def verify_payloads(directory, pins):
    directory = Path(directory).resolve()
    for name, pin in pins.items():
        path = (directory / name).resolve()
        path.relative_to(directory)
        if sha(path) != pin:
            raise ValueError("frozen payload changed: " + name)


def history168(plan):
    """Reuse only old40 plus growth128 real observations, not old terminal96."""
    pin = plan["inputs"]["growth_complete"]
    directory = Path(pin["file"]).resolve().parent
    frozen = read(pin["file"])
    verify_payloads(directory, frozen["files_sha256"])
    history = read(directory / "HISTORY40_ONLY.json")
    visible = read(directory / "VISIBLE_INPUTS.json")
    rows = list(history["rows"]) + list(visible["fit"])
    for role in BATCHES:
        truth = read(directory / role / "TRUE_BATCH_REVEAL.json")
        known = visible[role]
        if len(known) != 32 or len(truth["rows"]) != 32:
            raise ValueError("previous actual32 disclosure batch required")
        for a, b in zip(known, truth["rows"]):
            if a["identity"] != b["identity"]:
                raise ValueError("historical growth true-Y identity mismatch")
            rows.append({"identity": a["identity"],
                         "native_values": a["native_proxies"] + [b["native_target"]]})
    ids = [r["identity"]["component_id"] for r in rows]
    if len(rows) != 168 or len(set(ids)) != 168:
        raise ValueError("168 unique, already disclosed historical groups required")
    if any(len(r["native_values"]) != 3 or any(type(x) not in (int, float)
            or not math.isfinite(x) for x in r["native_values"]) for r in rows):
        raise ValueError("real finite historical reports required")
    # Only continue the old three hypotheses. The failed new/anchor branches are
    # preserved in their old archive, not quietly retrained or renamed here.
    suite = deepcopy(read(directory / "FINAL_STATE.json"))
    suite["components"] = {k: suite["components"][k] for k in ("affine", "interaction", "fullquad")}
    suite["libraries"] = {"old3_BMA": suite["libraries"]["old3_BMA"]}
    if suite["history_rows_seen"] != 168 or any(v["state"]["labeled_updates"] != 168
            for v in suite["components"].values()):
        raise ValueError("continued NIG must have exactly the same168 historical outcomes")
    return {"rows": rows, "nig3": suite,
            "baseline_transform": read(directory / "BASELINE_TRANSFORM.json"),
            "reused_components": 168, "reused_native_cells": 504,
            "older_terminal96_Y_access": False, "generated_labels": 0}


def descriptors(models, identities):
    """Two named views: raw8 for old forest coordinates; bounded8 only for g."""
    raw = [models.composition_descriptors(r["composition_amounts"]) for r in identities]
    bounded = [r[:6] + [r[6] / 118., r[7] / math.log(118.)] for r in raw]
    if any(len(r) != 8 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in r) for r in bounded):
        raise ValueError("known bounded stoichiometry context required")
    return raw, bounded


def composition(value):
    amounts = {key: Fraction(str(number)) for key, number in value.items() if not key.startswith("@")}
    if any(v < 0 for v in amounts.values()):
        raise ValueError("negative source stoichiometry")
    amounts = {key: v for key, v in amounts.items() if v > 0}
    total = sum(amounts.values(), Fraction(0))
    if total <= 0:
        raise ValueError("empty source composition")
    return "|".join(key + ":" + str(amounts[key] / total) for key in sorted(amounts))


class Disclosure:
    def __init__(self, rawfile, roles):
        with gzip.open(rawfile, "rt", encoding="utf-8") as stream:
            self._payload = json.load(stream)
        self._columns = {name: i for i, name in enumerate(self._payload["columns"])}
        if len(self._columns) != len(self._payload["columns"]):
            raise ValueError("unique source columns required")
        self.roles = roles
        self._initial_called = False
        self._next = 0
        self._reveals = []

    def _values(self, identity, indices):
        index = identity["raw_row_zero_based"]
        row = self._payload["data"][index]
        if (self._payload["index"][index] != identity["artifact_native_index"]
                or row[self._columns["jid"]] != identity["jid"]
                or row[self._columns["mpid"]] != identity["mpid"]
                or composition(row[self._columns["composition"]]) != identity["normalized_composition"]):
            raise ValueError("exact original source identity required")
        values = [row[self._columns[FIELDS[j]]] for j in indices]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError("invalid registered cell, never replace based on truth")
        return [float(x) for x in values]

    def initial(self):
        if self._initial_called:
            raise ValueError("new contexts disclosed only once")
        value = {role: [{"identity": r, "native_proxies": self._values(r, (0, 1))}
                       for r in self.roles["representatives"][role]] for role in BATCHES}
        self._initial_called = True
        return value

    def reveal(self, role, freeze_file, source_hashes, plan_hash):
        if not self._initial_called or self._next >= 3 or role != BATCHES[self._next]:
            raise ValueError("each complete64 batch disclosed once in order")
        freeze_file = Path(freeze_file)
        frozen = read(freeze_file)
        if (frozen["status"] != "ALL448_BATCH_LAWS_FROZEN_BEFORE64Y"
                or frozen["role"] != role or frozen["source_sha256"] != source_hashes
                or frozen["plan_sha256"] != plan_hash or frozen["methods"] != list(METHODS)):
            raise ValueError("all seven forecasts must close before this batch's Y")
        verify_payloads(freeze_file.parent, frozen["files_sha256"])
        forecasts = read(freeze_file.parent / "FORECASTS.json")["records"]
        expected = {(m, r["component_id"]) for m in METHODS for r in self.roles["representatives"][role]}
        actual = [(r["method"], r["identity"]["component_id"]) for r in forecasts]
        if len(actual) != 448 or len(set(actual)) != 448 or set(actual) != expected:
            raise ValueError("all448 unique registered forecast identities required")
        identities = {r["component_id"]: r for r in self.roles["representatives"][role]}
        if any(r["identity"] != identities[r["identity"]["component_id"]] for r in forecasts):
            raise ValueError("complete identity must match, not only a component string")
        rows = [{"identity": r, "native_target": self._values(r, (2,))[0]}
                for r in self.roles["representatives"][role]]
        value = {"status": "ACTUAL64_SOURCE_TARGETS_REVEALED_AFTER_ALL448_LAWS",
                 "actual_utc": utc(), "role": role, "freeze_sha256": sha(freeze_file), "rows": rows}
        self._reveals.append({"role": role, "target_cells": 64, "freeze_sha256": sha(freeze_file)})
        self._next += 1
        return value

    def ledger(self):
        return {"new_context_cells": 384 if self._initial_called else 0,
                "new_target_cells": sum(r["target_cells"] for r in self._reveals),
                "new_total_native_cells": (384 if self._initial_called else 0) + 64 * len(self._reveals),
                "prequential_reveals": self._reveals, "historical168_reused_cells_separate": 504,
                "whole_source_loaded_only_in_trusted_reader": True,
                "source": "public computational reports, not wet measurements or new DFT runs"}
