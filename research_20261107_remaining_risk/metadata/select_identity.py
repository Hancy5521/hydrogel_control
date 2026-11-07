"""SOURCE ONLY until exact root registration: conservative fresh bulk identity pool.

Only pre-existing identity metadata is read; no gzip, property decoder, model,
target eligibility refilter, or target-dependent ranking exists in this module.
This is role reservation, not property access or scientific execution permission.
"""
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import argparse
import hashlib
import json
import re

SCOPE = "REMAINING_BULK_V1_METADATA_ONLY_SELECTION"
PREFIX = "remaining_bulk_v1|"
CONFIG = {"JARVIS_all_rows": 25923, "old_eligible_rows": 11020,
          "old_eligible_groups": 8842, "old_consumed_groups": 1408,
          "historical_remaining_before_new_source_exclusion": 7434,
          "dielectric_all_rows": 1056, "dielectric_all_groups": 964,
          "select_whole_groups": 512, "hash_prefix": PREFIX}
INPUT_NAMES = {"eligible_metadata", "old_roles", "all_JARVIS_identity",
               "all_dielectric_identity", "old_seed_reconciliation",
               "identity_complete", "independent_identity_audit"}
ELIGIBLE = {"raw_row_zero_based", "artifact_native_index", "component_id",
            "normalized_composition", "jid", "mpid", "composition_amounts"}
JROW = {"raw_row_zero_based", "artifact_native_index", "jid_original", "mpid_original",
        "jid", "mpid", "normalized_composition"}
DROW = {"dataset", "raw_row_zero_based", "artifact_native_index", "material_id", "mpid",
        "formula", "normalized_composition", "composition_amount_fractions", "valid_MPID"}
PATTERNS = {"jid": re.compile(r"JVASP-[0-9]+"), "mpid": re.compile(r"(?:mp|mvc)-[0-9]+")}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("nonfinite JSON literal in identity contract")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique, parse_constant=reject_constant)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def signature(amounts):
    require(isinstance(amounts, dict) and bool(amounts), "positive composition mapping required")
    values = {}
    for element, value in amounts.items():
        require(isinstance(element, str) and bool(re.fullmatch(r"[A-Z][a-z]?", element)),
                "literal atomic element key required")
        require(type(value) in (str, int, float), "rational composition amount required")
        value = Fraction(str(value))
        require(value > 0, "positive original eligible composition required")
        values[element] = value
    total = sum(values.values(), Fraction(0))
    return "|".join(element + ":" + str(values[element] / total) for element in sorted(values))


def canonical_signature(text):
    require(isinstance(text, str) and text, "normalized composition text required")
    parts = text.split("|")
    values = {}
    for part in parts:
        element, value = part.split(":")
        require(element not in values, "duplicate composition element")
        values[element] = value
    require(signature(values) == text, "noncanonical normalized composition")
    return text


def alias(value, kind):
    # Metadata already carries explicitly validated aliases; never stringify invalid IDs.
    require(value is None or isinstance(value, str), "explicit normalized identity token required")
    if value is not None:
        require(PATTERNS[kind].fullmatch(value) is not None, "invalid normalized alias")
    return value


class DSU:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.size[left] < self.size[right]:
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


def connect(rows, dsu):
    owners = {}
    for index, row in enumerate(rows):
        for field in ("normalized_composition", "jid", "mpid"):
            token = row.get(field)
            if token is None:
                continue
            key = (field, token)
            if key in owners:
                dsu.union(index, owners[key])
            else:
                owners[key] = index


def validate(plan_file, approval_file):
    # Rejection precedes output creation, identity metadata reads and imports of any science code.
    plan, approval = read(plan_file), read(approval_file)
    require(plan.get("status") == "REGISTERED_METADATA_ONLY", "root registered metadata plan required")
    require(plan.get("scope") == SCOPE and plan.get("configuration") == CONFIG, "exact phase/configuration required")
    require(plan.get("source_sha256") == sha(__file__), "selection source changed")
    require(approval.get("approved") is True and approval.get("scope") == SCOPE
            and approval.get("plan_sha256") == sha(plan_file)
            and approval.get("allowed_phases") == ["metadata_only_selection"],
            "exact root approval binding required")
    require(set(plan.get("inputs", {})) == INPUT_NAMES, "exact metadata-only input allowlist required")
    require(isinstance(plan.get("registered_actual_utc"), str)
            and isinstance(approval.get("actual_utc"), str), "actual registration timestamps required")
    for pin in plan["inputs"].values():
        require(set(pin) == {"file", "sha256"} and len(pin["sha256"]) == 64, "exact metadata pin required")
        require(sha(pin["file"]) == pin["sha256"], "frozen identity input hash changed")
    return plan


def select(plan):
    inputs = plan["inputs"]
    eligible = read(inputs["eligible_metadata"]["file"])["datasets"]["jarvis_dft_3d"]
    old = read(inputs["old_roles"]["file"])
    jmeta = read(inputs["all_JARVIS_identity"]["file"])
    dmeta = read(inputs["all_dielectric_identity"]["file"])
    reconciliation = read(inputs["old_seed_reconciliation"]["file"])
    complete = read(inputs["identity_complete"]["file"])
    audit = read(inputs["independent_identity_audit"]["file"])
    require(audit.get("status") == "PASS_INDEPENDENT_IDENTITY_ONLY"
            and audit.get("property_values_emitted") == 0, "prior independent identity audit PASS required")
    require(reconciliation.get("status") == "PASS_FULL_ORIGINAL_GRAPH_MATCHES_OLD_ELIGIBLE_PARTITION",
            "prior original partition reconciliation required")
    require(jmeta.get("property_fields_emitted") == [] and jmeta.get("all_original_rows_including_property_ineligible") is True,
            "all original JARVIS rows with no property fields required")
    require(dmeta.get("target_values_emitted_to_metadata_or_models") == 0
            and dmeta.get("newsource1056_rows_complete") is True
            and dmeta.get("target_finite_filter_performed") is False,
            "ALL new source identities required, including target-ineligible siblings")
    for name in ("all_JARVIS_identity", "all_dielectric_identity", "old_seed_reconciliation"):
        pin = inputs[name]
        require(complete["files_sha256"][Path(pin["file"]).name] == pin["sha256"], "identity completion binding mismatch")
    jrows, drows = jmeta["rows"], dmeta["rows"]
    require((len(jrows), len(drows), len(dmeta["groups"]), len(eligible)) == (25923, 1056, 964, 11020),
            "registered identity frame sizes differ")
    for rows, keys in ((jrows, JROW), (drows, DROW)):
        require([row.get("raw_row_zero_based") for row in rows] == list(range(len(rows))), "complete ordered raw identity index required")
        require(len({row["artifact_native_index"] for row in rows}) == len(rows), "unique artifact indices required")
        for row in rows:
            require(set(row) == keys, "unknown identity row fields; no target field allowed")
            canonical_signature(row["normalized_composition"])
            alias(row.get("jid"), "jid")
            alias(row.get("mpid"), "mpid")
    base = DSU(len(jrows)); connect(jrows, base)
    grouped, base_by_component, by_base = defaultdict(list), defaultdict(set), defaultdict(set)
    for row in eligible:
        require(set(row) == ELIGIBLE, "old eligible row has unknown/property fields")
        index = row["raw_row_zero_based"]
        require(type(index) is int and 0 <= index < len(jrows), "old eligible raw index out of range")
        parent = jrows[index]
        require(parent["artifact_native_index"] == row["artifact_native_index"]
                and parent["jid_original"] == row["jid"] and parent["mpid_original"] == row["mpid"]
                and parent["normalized_composition"] == row["normalized_composition"]
                and signature(row["composition_amounts"]) == row["normalized_composition"],
                "original eligible identity changed")
        component, root = row["component_id"], base.find(index)
        require(isinstance(component, str) and component, "old component identifier required")
        grouped[component].append(row); base_by_component[component].add(root); by_base[root].add(component)
    require(len({row["raw_row_zero_based"] for row in eligible}) == 11020 and len(grouped) == 8842,
            "frozen eligible partition size mismatch")
    require(all(len(value) == 1 for value in base_by_component.values())
            and all(len(value) == 1 for value in by_base.values()), "old eligible components split or merged")
    prior, consumed_new, historical = (old["all_prior_excluded_component_ids"],
                                      old["new_development_consumed_component_ids"],
                                      old["remaining_unselected_component_ids_in_registered_hash_order"])
    require((len(set(prior)), len(set(consumed_new)), len(set(historical))) == (1216, 192, 7434)
            and (len(prior), len(consumed_new), len(historical)) == (1216, 192, 7434), "old consumption counts differ")
    consumed = set(prior) | set(consumed_new)
    require(len(consumed) == 1408 and not consumed & set(historical)
            and set(grouped) == consumed | set(historical), "old whole-group partition not exact")
    require(set(reconciliation["eligible_seed_rawrows_by_component"]) == consumed, "old 1408 seed set mismatch")
    for component in consumed:
        require(sorted(reconciliation["eligible_seed_rawrows_by_component"][component])
                == sorted(row["raw_row_zero_based"] for row in grouped[component]), "old consumed eligible sibling seeds differ")
    combined = DSU(len(jrows) + len(drows)); connect(jrows + drows, combined)
    new_roots = {combined.find(len(jrows) + index) for index in range(len(drows))}
    consumed_roots = {combined.find(row["raw_row_zero_based"]) for component in consumed for row in grouped[component]}
    blocked_roots = new_roots | consumed_roots
    component_roots = {}
    for component, siblings in grouped.items():
        roots = {combined.find(row["raw_row_zero_based"]) for row in siblings}
        require(len(roots) == 1, "combined graph split an original eligible component")
        component_roots[component] = next(iter(roots))
    available = [component for component, root in component_roots.items() if root not in blocked_roots]
    require(len({component_roots[c] for c in available}) == len(available),
            "unblocked old groups merge through unknown alias; stop for review")
    available.sort(key=lambda key: (hashlib.sha256((PREFIX + key).encode("utf-8")).hexdigest(), key))
    require(len(available) >= 512, "fewer than 512 whole fresh groups; no fallback or smaller redesign")
    chosen, remaining = available[:512], available[512:]
    joint_members = defaultdict(list)
    for index, row in enumerate(jrows):
        joint_members[combined.find(index)].append({"source": "jarvis_dft_3d", "raw_row_zero_based": index,
                                                   "artifact_native_index": row["artifact_native_index"]})
    for index, row in enumerate(drows):
        joint_members[combined.find(len(jrows) + index)].append({"source": "dielectric_constant", "raw_row_zero_based": index,
                                                                 "artifact_native_index": row["artifact_native_index"]})
    selected = []
    for component in chosen:
        siblings = sorted(grouped[component], key=lambda row: row["raw_row_zero_based"])
        closure = joint_members[component_roots[component]]
        require(all(row["source"] == "jarvis_dft_3d" for row in closure), "selected group touches excluded dielectric source")
        selected.append({"component_id": component, "role": "future_remaining_risk_pool512",
                         "representative": dict(siblings[0]), "eligible_siblings": siblings,
                         "all_original_source_siblings_reserved": closure})
    additional = sorted(component for component in historical if component_roots[component] in new_roots)
    result = {"status": "IDENTITY_ONLY_POOL_RESERVED_NO_PROPERTY_AUTHORIZATION", "actual_utc": utc(),
              "configuration": CONFIG, "selected_groups": selected,
              "selected_component_ids_in_registered_hash_order": chosen,
              "remaining_unselected_component_ids_in_registered_hash_order": remaining,
              "all_prior_consumed_component_ids": sorted(consumed),
              "all_old_eligible_components_touching_any_dielectric_source": sorted(c for c,r in component_roots.items() if r in new_roots),
              "additional_historical_remaining_excluded_by_entire_dielectric_source": additional,
              "counts": {"historical_remaining_not_fresh_claim": 7434,
                         "remaining_after_full_new_source_exclusion_before_selection": len(available),
                         "new_source_exclusions_from_historical_remaining": len(additional),
                         "selected": 512, "remaining_after_selection": len(remaining),
                         "all_JARVIS_rows_in_joint_blocked_closure": sum(combined.find(i) in blocked_roots for i in range(len(jrows))),
                         "all_dielectric_rows_excluded": 1056, "all_dielectric_original_groups_excluded": 964},
              "identity_policy": "full original rows plus new source, normalized composition/JID/MPID transitive closure; all polymorphs conservatively joined",
              "representative_policy": "minimum original eligible raw_row; eligibility and representative rules unchanged",
              "targets_read": 0, "fits": 0, "new_measurements": 0,
              "protected_dielectric19_remain_unopened": True,
              "not_physical_independence_or_future_generalization_proof": True}
    require(len(available) + len(additional) == 7434, "fresh subtraction inconsistent with historical partition")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    plan = validate(args.plan, args.approval)
    require(not args.output_dir.exists(), "one-shot output directory already exists")
    # Computation is metadata only; no partial role output is written on failed admission.
    result = select(plan)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir / "START.json", {"actual_utc": utc(), "source_sha256": sha(__file__),
          "plan_sha256": sha(args.plan), "approval_sha256": sha(args.approval), "scope": SCOPE})
    write(args.output_dir / "ROLE_SELECTION.json", result)
    write(args.output_dir / "COMPLETE.json", {"actual_utc": utc(), "status": result["status"],
          "source_sha256": sha(__file__), "plan_sha256": sha(args.plan),
          "selection_sha256": sha(args.output_dir / "ROLE_SELECTION.json"), "targets_read": 0, "fits": 0})
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "selection_sha256": sha(args.output_dir / "ROLE_SELECTION.json")}))


if __name__ == "__main__":
    main()
