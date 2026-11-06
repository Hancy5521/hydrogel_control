"""Source contracts and sequential true-report disclosure; no model mathematics."""
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

FILES = ("growth_models.py", "proposal.py", "probability.py", "baselines.py",
         "experiment_io.py", "run_growth.py", "score_growth.py")
METHODS = ("old3_BMA", "grown4_BMA", "H_new", "H_anchor", "affine", "GP10", "RF10")
BATCHES = ("validation1", "validation2", "validation3")
FIELDS = ("e_form", "gap opt", "bulk modulus")
NIG_CORE_SHA256 = "1d74cde91519ce282c6ebee54e53807a2a79be0cebf19e06221eb4f16e8b7528"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique, parse_constant=invalid)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def jsha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sources():
    base = Path(__file__).resolve().parent
    return {name: sha(base / name) for name in FILES}


def validate(planfile, approvalfile, phase):
    plan, approval = read(planfile), read(approvalfile)
    if (approval.get("approved") is not True or approval.get("scope") != "SINGLE_SLOT_COMPOSITION_GROWTH_FRESH128"
            or approval.get("plan_sha256") != sha(planfile) or phase not in approval.get("allowed_phases", [])):
        raise ValueError("exact root phase approval required")
    if plan.get("source_sha256") != sources():
        raise ValueError("all seven registered source bytes required")
    for group in ("inputs", "dependencies"):
        for item in plan[group].values():
            if sha(item["file"]) != item["sha256"]:
                raise ValueError("registered input/dependency changed: " + item["file"])
    roles = read(plan["inputs"]["roles"]["file"])
    if roles["counts"] != {"previously_consumed": 1088, "new_development": 128,
                           "new_fit": 32, "prequential_batches": [32, 32, 32], "remaining": 7626}:
        raise ValueError("exact fresh128 source-component roles required")
    selected = roles["new_development_consumed_component_ids"]
    rows = [r for group in roles["representatives"].values() for r in group]
    if (not (len(selected) == len(set(selected)) == 128) or len(rows) != 128
            or set(selected) != {r["component_id"] for r in rows}
            or set(selected) & set(roles["all_prior_excluded_component_ids"])):
        raise ValueError("whole-component roles must be disjoint")
    if any(len(roles["representatives"][key]) != 32 for key in ("fit",) + BATCHES):
        raise ValueError("fixed role size required")
    return plan, roles


def history40(plan):
    """Only oldwarm32 and full_native's eight true queries, never oldterminalY."""
    visible = read(plan["inputs"]["old_visible"]["file"])
    states = read(plan["inputs"]["old_states"]["file"])
    index = read(plan["inputs"]["old_prequential_index"]["file"])
    freeze = read(plan["inputs"]["old_forecast_freeze"]["file"])
    directory = Path(plan["inputs"]["old_forecast_freeze"]["file"]).parent
    rows = list(visible["warm"])
    records = index["records"]
    chosen = sorted((r for r in records if r["policy"] == "full_native"), key=lambda r: r["step"])
    if len(rows) != 32 or [r["step"] for r in chosen] != list(range(8)):
        raise ValueError("chronological oldwarm32 plus selected eight required")
    for row in chosen:
        for filekey, hashkey in (("forecast_file", "forecast_sha256"), ("outcome_file", "outcome_sha256")):
            path = (directory / row[filekey]).resolve()
            path.relative_to(directory.resolve())
            if sha(path) != row[hashkey] or freeze["files"][row[filekey]] != row[hashkey]:
                raise ValueError("sealed true-query history changed")
        before = read(directory / row["forecast_file"])
        outcome = read(directory / row["outcome_file"])
        reveal = outcome["true_Y_reveal"]
        if before["identity"] != reveal["identity"]:
            raise ValueError("old query identity differs from true response")
        rows.append({"identity": before["identity"],
                     "native_values": before["known_native_proxies"] + [reveal["native_target"]]})
    ids = [r["identity"]["component_id"] for r in rows]
    if not (len(rows) == len(set(ids)) == 40):
        raise ValueError("40 distinct historical components required")
    snapshot = {key: states[key] for key in ("transform", "warm_ensemble", "native_reference_feature_transform")}
    final = next(r["ensemble"] for r in states["checkpoint_states"]
                 if r["policy"] == "full_native" and r["budget"] == 8)
    return {"rows": rows, "old_snapshot": snapshot, "old_final3": final,
            "old_training_components": 40, "old_history_cells_reused": 120,
            "old_terminal_Y_access": False, "other_counterfactual_pool_Y_not_added": True}


def bootstrap_core(plan):
    pin = plan["dependencies"]["nig_core"]
    path = Path(pin["file"]).resolve()
    if pin["sha256"] != NIG_CORE_SHA256:
        raise ValueError("the previously audited NIG scalar core is required")
    sys.path.insert(0, str(path.parent))
    import relation_posterior
    if Path(relation_posterior.__file__).resolve() != path or sha(path) != pin["sha256"]:
        raise ValueError("exact frozen scalar NIG core required")


def scalar_scorer(plan):
    pin = plan["dependencies"]["legacy_scalar_score"]
    path = Path(pin["file"]).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("_growth_frozen_scalar_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_law


def composition(value):
    amounts = {key: Fraction(str(number)) for key, number in value.items() if not key.startswith("@")}
    if any(v < 0 for v in amounts.values()):
        raise ValueError("negative composition")
    amounts = {k: v for k, v in amounts.items() if v > 0}
    total = sum(amounts.values(), Fraction())
    if total <= 0:
        raise ValueError("empty composition")
    return "|".join(k + ":" + str(amounts[k] / total) for k in sorted(amounts))


class Disclosure:
    """Trusted full-gzip parser; only authorized cells reach model methods."""

    def __init__(self, rawfile, roles):
        with gzip.open(rawfile, "rt", encoding="utf-8") as stream:
            self._payload = json.load(stream)
        self._columns = {name: i for i, name in enumerate(self._payload["columns"])}
        if len(self._columns) != len(self._payload["columns"]):
            raise ValueError("unique source columns required")
        self.roles, self._next, self._reveals = roles, 0, []
        self._initial_called = False

    def _values(self, identity, indices):
        index = identity["raw_row_zero_based"]
        row = self._payload["data"][index]
        if (self._payload["index"][index] != identity["artifact_native_index"]
                or row[self._columns["jid"]] != identity["jid"]
                or row[self._columns["mpid"]] != identity["mpid"]
                or composition(row[self._columns["composition"]]) != identity["normalized_composition"]):
            raise ValueError("actual source identity/state mismatch")
        values = [row[self._columns[FIELDS[j]]] for j in indices]
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("invalid authorized property; never replace based on Y")
        return [float(v) for v in values]

    def initial(self):
        if self._initial_called:
            raise ValueError("initial cells may be disclosed only once")
        result = {key: [{"identity": r,
                       "native_values" if key == "fit" else "native_proxies":
                           self._values(r, (0, 1, 2) if key == "fit" else (0, 1))}
                      for r in self.roles["representatives"][key]] for key in ("fit",) + BATCHES}
        self._initial_called = True
        return result

    def reveal(self, role, freeze_file, source_hashes, plan_hash):
        if not self._initial_called or self._next >= 3 or role != BATCHES[self._next]:
            raise ValueError("each complete validation batch disclosed once in order")
        freeze_file = Path(freeze_file)
        frozen = read(freeze_file)
        if (frozen["status"] != "ALL224_BATCH_LAWS_FROZEN_BEFORE_32Y"
                or frozen["role"] != role or frozen["source_sha256"] != source_hashes
                or frozen["plan_sha256"] != plan_hash or frozen["methods"] != list(METHODS)):
            raise ValueError("all seven forecasts must freeze before this batch's Y")
        base = freeze_file.parent.resolve()
        for name, pin in frozen["files_sha256"].items():
            path = (base / name).resolve()
            path.relative_to(base)
            if sha(path) != pin:
                raise ValueError("closed batch payload changed")
        predictions = read(base / "FORECASTS.json")["records"]
        expected_identities = {r["component_id"]: r for r in self.roles["representatives"][role]}
        expected = set(expected_identities)
        pairs = {(r["method"], r["identity"]["component_id"]) for r in predictions}
        if len(predictions) != 224 or pairs != {(m, c) for m in METHODS for c in expected}:
            raise ValueError("exact full batch prediction cross product required")
        if any(r["identity"] != expected_identities[r["identity"]["component_id"]] for r in predictions):
            raise ValueError("prediction and reported target state identities differ")
        rows = [{"identity": r, "native_target": self._values(r, (2,))[0]}
                for r in self.roles["representatives"][role]]
        event = {"actual_utc": utc(), "role": role, "forecast_freeze_sha256": sha(freeze_file),
                 "rows": rows, "new_native_Y_cells": 32, "all_method_disclosures_shared": True}
        self._reveals.append(event)
        self._next += 1
        return event

    def ledger(self):
        return {"events": self._reveals, "initial_new_known_proxy_cells": 256,
                "initial_new_fit_Y_cells": 32, "validation_Y_cells": 32 * self._next,
                "total_new_native_cells": 288 + 32 * self._next,
                "old40_training_cells_separate_reused": 120,
                "raw_parse_not_cryptographic_blindness": True,
                "source_is_existing_reported_DFT_not_new_computation_or_wet_measurement": True}
