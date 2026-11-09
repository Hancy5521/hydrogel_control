"""Independent SOURCE admission: only original column-one coordinates.

No production module import; no stress/viscosity/torque conversion or fit.
Raw ZIP bytes are public and unencrypted. This is not a blind numeric gate.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
CURRENT = {
    "NaCl_0_XG", "NaCl_0.1_XG", "NaCl_0.3_XG", "NaCl_0.4_XG",
    "NaCl_0.5_XG", "NaCl_0.7_XG", "XG_NaCl_0.1", "XG_NaCl_0.3",
    "XG_NaCl_0.5", "XG_NaCl_0.7", "XG_NaCl_0.9",
}
QUARANTINE = {"NaCl_0_XG_2019": "explicit_2019_previous_source",
              "XG_NaCl_0.05": "extra_group_not_in_current_method_11"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    gate = json.loads((ROOT / "SOURCE_GATE_PACKAGE.json").read_text(encoding="utf-8"))
    for record in gate["files"]:
        assert sha(ROOT / record["path"]) == record["sha256"], record["path"]
    original = json.loads((ROOT / "FILE_LIST_ORIGINAL.json").read_text(encoding="utf-8"))
    official_hashes = {r["filename"]: r["content_details"]["sha256_hash"] for r in original}
    assert sha(ROOT / "Rheological_data.zip") == official_hashes["Rheological_data.zip"]
    assert sha(ROOT / "README_rheological_data.txt") == official_hashes["README_rheological_data.txt"]
    producer = json.loads((ROOT / "COORDINATES_ONLY_V2.json").read_text(encoding="utf-8"))
    producer_records = {r["member"]: r for r in producer["records"]}
    primary_text = (ROOT / "PRIMARY_WEB_TEXT.txt").read_text(encoding="utf-8")
    readme = (ROOT / "README_rheological_data.txt").read_text(encoding="utf-8")
    required_method_tokens = ["21\u00a0°C", "MCR301", "TrueGap", "1\u00a0g/L",
                              "2\u00a0L", "5\u00a0min", "controlled shear rate", "CC BY 4.0"]
    assert all(token in primary_text for token in required_method_tokens)
    assert "10.1038/s41598-019-44233-z" in readme
    assert "NaCl_0_XG_2019 originates from previous study" in readme
    rows = []
    with zipfile.ZipFile(ROOT / "Rheological_data.zip") as zf:
        members = sorted(n for n in zf.namelist() if n.endswith("/flow_data.csv"))
        assert len(members) == 13
        for member in members:
            group = member.split("/")[-2]
            assert group in CURRENT | QUARANTINE.keys()
            raw = zf.read(member)
            stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
            assert stream.readline().rstrip("\r\n") == "Var1,Var2,Var3,Var4"
            units = stream.readline().rstrip("\r\n")
            assert units == "shear rate [1/s],shear stress [Pa],viscosity [Pa s],Torque [μN m]"
            coordinates = []
            for line_number, line in enumerate(stream, start=3):
                if not line.strip():
                    raise ValueError("Unexpected blank original data line")
                # Deliberately never tokenize, convert, print, or quality-filter response fields.
                first, separator, _uninterpreted_response_tail = line.partition(",")
                assert separator
                x = float(first.strip())
                assert math.isfinite(x) and x > 0
                coordinates.append({"source_line_1based": line_number,
                                    "gamma_per_s": x, "gamma_raw_token": first.strip()})
            assert all(a["gamma_per_s"] < b["gamma_per_s"] for a, b in zip(coordinates, coordinates[1:]))
            assert len({r["gamma_per_s"] for r in coordinates}) == len(coordinates)
            other = producer_records[member]
            assert coordinates == other["coordinates"]
            assert hashlib.sha256(raw).hexdigest() == other["source_member_sha256"]
            if group in CURRENT:
                assert len(coordinates) == 357
                window = [r for r in coordinates if 1.0 <= r["gamma_per_s"] <= 50.0]
                assert len(window) >= 25
                # Coordinate-only existence witness. This does not register root's study policy.
                desired = [math.exp(math.log(1.0) + j * math.log(50.0) / 24) for j in range(25)]
                selected = [min(window, key=lambda r: (abs(math.log(r["gamma_per_s"] / v)),
                                                       r["source_line_1based"])) for v in desired]
                assert len({r["source_line_1based"] for r in selected}) == 25
                support = {"fixed_window_per_s": [1.0, 50.0], "native_nodes_in_window": len(window),
                           "witness_25_native_distinct_nodes": selected,
                           "interpolated_values": 0, "policy_not_frozen_by_this_audit": True}
            else:
                support = {"excluded_before_y": QUARANTINE[group], "numeric_response_decoded": 0}
            rows.append({"member": member, "group": group,
                         "role": "current_11" if group in CURRENT else "quarantine",
                         "native_coordinate_count": len(coordinates),
                         "first_gamma_per_s": coordinates[0]["gamma_per_s"],
                         "last_gamma_per_s": coordinates[-1]["gamma_per_s"],
                         "source_member_sha256": hashlib.sha256(raw).hexdigest(),
                         "input_matches_independently": True, "support": support})
    assert {r["group"] for r in rows if r["role"] == "current_11"} == CURRENT
    stamp = dt.datetime.now(dt.timezone.utc)
    result = {
        "schema": "xanthan-independent-source-admission-v1", "status": "PASS_SOURCE_INPUT_ONLY",
        "actual_utc": stamp.isoformat(),
        "asia_shanghai": stamp.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat(),
        "auditor_source_sha256": sha(__file__), "source_gate_package_sha256": sha(ROOT / "SOURCE_GATE_PACKAGE.json"),
        "source_doi": "10.17632/ntgfb38k69.3", "source_version": 3,
        "official_data_license": "CC BY 4.0", "method_doi": "10.1016/j.dib.2022.108865",
        "method_evidence_scope": "Saved successful official Mendeley/PMC response only; no new network retry/captcha bypass",
        "units": {"shear_rate": "1/s", "viscosity": "Pa s", "stress": "Pa", "torque": "μN m"},
        "temperature_C": 21, "material_state": "aqueous xanthan gum 1 g/L with NaCl; not UV-cured hydrogel",
        "current_preparation_conditions": 11, "chemical_salinity_levels": 7,
        "rheology_curve_runs_per_condition_observed": 1,
        "independent_batch_replicates_certified": False,
        "identity_caveat": "Shared salinity and preparation order are process conditions; 357 coordinates and settling runs are not rheology replicates",
        "excluded_before_y": QUARANTINE,
        "numeric_response_cells_decoded": 0, "production_modules_imported": 0, "fits": 0,
        "raw_response_strings_in_public_csv_reader_memory": True,
        "new_performance_result": False,
        "findings": ["Raw ZIP/README match official content SHA256",
                     "Independent first-column rebuild matches all13 producer coordinate records exactly",
                     "All11 current conditions have >=25 distinct native points in fixed1..50/s window",
                     "25-node witness is coordinate feasibility, not root experiment/mask selection",
                     "Old2019 and extra0.05 are quarantined by provenance, not target response",
                     "All11 current groups need an actualUTC preY experiment protocol before viscosity conversion"],
        "remaining_limits": ["No independent rheometer batch IDs; one flow curve per condition",
                             "No verification of field/biological hydrogel claims",
                             "No material effect, calibration, noninferiority or top-conference readiness inferred"],
        "records": rows,
    }
    output = ROOT / "SOURCE_INDEPENDENT_REVIEW.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "path": str(output), "sha256": sha(output),
                      "current_groups": 11, "response_numeric_cells": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
