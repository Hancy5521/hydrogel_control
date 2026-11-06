"""Describe visible formula identities only; never read material target values.

Buckets are element-presence signatures, not asserted phases or chemical classes.
This secondary descriptive view cannot change frozen models, queries or endpoints.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bucket(amounts):
    elements = set(amounts)
    if "O" in elements:
        return "contains_O"
    if elements.intersection({"F", "Cl", "Br", "I"}):
        return "no_O_contains_F_Cl_Br_I"
    if elements.intersection({"S", "Se", "Te"}):
        return "no_O_or_halogen_contains_S_Se_Te"
    return "remaining_compositions"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan["source_sha256"] != sha(__file__):
        raise ValueError("exact registered annotation source required")
    visible = Path(plan["visible_input"]["file"])
    if sha(visible) != plan["visible_input"]["sha256"]:
        raise ValueError("frozen visible input required")
    data = json.loads(visible.read_text(encoding="utf-8"))
    annotations = []
    counts = {}
    seen = set()
    for role in ("warm", "pool", "terminal"):
        counts[role] = {}
        for row in data[role]:
            identity = row["identity"]
            key = identity["component_id"]
            if key in seen:
                raise ValueError("whole-component roles must be disjoint")
            seen.add(key)
            label = bucket(identity["composition_amounts"])
            counts[role][label] = counts[role].get(label, 0) + 1
            annotations.append({"component_id": key, "role": role,
                                "signature": label,
                                "element_count": len(identity["composition_amounts"])})
    result = {
        "status": "VISIBLE_IDENTITY_ANNOTATIONS_ONLY",
        "actual_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "plan_sha256": sha(plan_path), "source_sha256": sha(__file__),
        "visible_input_sha256": sha(visible), "counts": counts,
        "annotations": annotations, "target_values_read": 0,
        "science_source_or_endpoint_change": False,
        "not_independent_datasets_or_phase_labels": True,
    }
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"counts": counts, "annotated_components": len(seen),
                      "target_values_read": 0}))


if __name__ == "__main__":
    main()
