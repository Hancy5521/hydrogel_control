"""Portable verification/replay of already saved, independently audited laws.

This is an offline audit entry point, NOT an experiment launcher. It calls only
independent verification scripts on frozen predictions and existing development
targets. It never invokes source admissions, material readers, model training,
production scoring or old failed audits. Original receipts remain immutable.
The source ZIP is used only by the existing released-cell raw-source auditor.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback
import uuid

MANIFEST_SCHEMA = "mlst-conference-portable-manifest-v1"
SOURCE_SHA = "a2a7a84d92bf84a69e6f6c49e0ae976d916cc63fa46ff28bccd08a7ebdd4e218"
INPUT_SHA = "dd1e7b41b8db7076a6b55a68f94d41cc0b2e2e323ca1fd9596c57b9c7b4a2900"
MODEL_SHA = "d72fc262fbae0e6fecc36568f0638166fe95e991bd5f7df100ab1434097d82d1"


def now():
    stamp = dt.datetime.now(dt.timezone.utc)
    return {"actual_utc": stamp.isoformat(),
            "asia_shanghai": stamp.astimezone(dt.timezone(dt.timedelta(hours=8))).isoformat()}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def local(root, relative):
    p = PurePosixPath(relative)
    if (p.is_absolute() or not p.parts or ".." in p.parts or "\\" in relative or ":" in relative):
        raise ValueError("Expected safe relative POSIX package path: " + relative)
    resolved = root.joinpath(*p.parts).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Package path escapes root")
    return resolved


def verify_payload(root, manifest_path):
    manifest = read(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA or not isinstance(manifest.get("files"), list):
        raise ValueError("Missing or unsupported complete package manifest")
    files = {}
    for record in manifest["files"]:
        relative = record["path"]
        if relative in files:
            raise ValueError("Duplicate manifest path")
        path = local(root, relative)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            raise ValueError("Package member bytes changed: " + relative)
        files[relative] = record
    excluded = manifest_path.resolve()
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.resolve() != excluded and
              path.relative_to(root).parts[0] != "offline_audits"}
    if actual != set(files):
        raise ValueError("Manifest must cover exact payload; extra=" + str(sorted(actual - set(files))) +
                         "; missing=" + str(sorted(set(files) - actual)))
    pins = read(local(root, "FROZEN_DEPENDENCIES.json"))["scientific_bindings"]
    if len(pins) != 16:
        raise ValueError("Original sixteen frozen dependencies required")
    for pin in pins:
        relative = "frozen_checkpoint/" + pin["relative_to_checkpoint"]
        if relative not in files or files[relative]["sha256"] != pin["sha256"] or files[relative]["bytes"] != pin["bytes"]:
            raise ValueError("Copied frozen dependency differs from original pin: " + relative)
    def one_by_hash(expected):
        candidates = [relative for relative, record in files.items() if record["sha256"] == expected]
        if len(candidates) != 1:
            raise ValueError("Expected exactly one package member for pinned hash " + expected)
        return candidates[0]
    locations = {"source_zip": one_by_hash(SOURCE_SHA), "inputs": one_by_hash(INPUT_SHA),
                 "model": one_by_hash(MODEL_SHA)}
    if locations["model"] != "frozen_checkpoint/models/SHARED_MODEL.json":
        raise ValueError("Shared model must reside in copied relative frozen checkpoint")
    return manifest, files, locations


def runtime_versions():
    result = {"python": sys.version, "executable": sys.executable}
    for name in ("numpy", "scipy", "mpmath"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError("Missing dependency " + name + "; no automatic installation/network is performed")
        result[name] = importlib.metadata.version(name)
    return result


def jobs(root, output, locations, files, manifest):
    def p(name):
        if name not in files:
            raise ValueError("Replay needs an existing manifested member: " + name)
        return str(local(root, name))
    def command(script, name, pairs, prefix=()):
        args = [sys.executable, "-X", "utf8", "-B", p(script), *prefix]
        for option, relative in pairs:
            args += ["--" + option, p(relative)]
        args += ["--output", str(output / (name + ".json"))]
        return {"name": name, "command": args, "result": output / (name + ".json")}
    prepared = "resin_dev_once/prepare/PREPARED.json"
    predictions = "resin_dev_once/predict/PREDICTIONS.json"
    preaudit = "resin_dev_once/INDEPENDENT_PRE_Y.json"
    targets = "resin_dev_once/reveal/TARGETS.json"
    scores = "resin_dev_once/score/SCORES.json"
    student = "student_dev_once/predict/PREDICTIONS.json"
    h3 = "influence_pilot/UPDATE_AND_PREDICTIONS.json"
    # Original pre-Y reports are deliberately retained as input to post-Y
    # auditors: a fresh report has a new UTC/hash and cannot replace the old
    # report in the originally registered immutable score/release chain.
    result = [
        command("verification/resin_independent_pre_y.py", "MAIN_PRE", [
            ("prepare", prepared), ("predictions", predictions),
            ("inputs", locations["inputs"]), ("model", locations["model"])]),
        command("verification/resin_independent_post_y.py", "MAIN_POST", [
            ("prepare", prepared), ("predictions", predictions), ("targets", targets), ("scores", scores),
            ("inputs", locations["inputs"]), ("source-zip", locations["source_zip"]),
            ("pre-audit", preaudit), ("release", "resin_dev_once/ROOT_TARGET_RELEASE.json")]),
        command("verification/student_independent_audit.py", "STUDENT_POST", [
            ("authorization", "verification/STUDENT_POST_Y_AUTHORIZED.json"),
            ("prepare", prepared), ("gaussian-predictions", predictions), ("student-predictions", student),
            ("main-audit", preaudit), ("targets", targets), ("scores", "student_dev_once/score/SCORES.json"),
            ("student-pre-audit", "student_dev_once/INDEPENDENT_PRE_Y.json")], prefix=("post_y",)),
        command("verification/early_update_independent_audit_v2.py", "H3_PRE_MP70", [
            ("main-prepare", prepared), ("main-predictions", predictions),
            ("main-pre-audit", preaudit), ("pilot", h3)]),
        command("verification/early_update_independent_post_y.py", "H3_POST", [
            ("predictions", h3), ("targets", targets), ("scores", "influence_pilot/SCORES.json"),
            ("release", "influence_pilot/ROOT_SCORE_RELEASE.json"),
            ("pre-audit", "influence_pilot/INDEPENDENT_PRE_Y_V2.json")])]
    optional = ["verification/universal_student_independent_audit.py",
                "verification/UNIVERSAL_LAWS_AUTHORIZED.json", "verification/UNIVERSAL_SCORES_AUTHORIZED.json",
                "qualification_once/UNIVERSAL_STUDENT_EQUIVALENCE.json",
                "universal_dev_once/predict/PREDICTIONS.json", "universal_dev_once/score/SCORES.json",
                "universal_dev_once/INDEPENDENT_LAWS.json", "universal_dev_once/INDEPENDENT_SCORES.json"]
    universal = {"status": "PENDING_NOT_REPLAYED", "reason": "Not all original universal-Kriging artifacts and audit PASS are present"}
    if all(name in files for name in optional):
        for relative in optional[-2:]:
            if read(local(root, relative)).get("status") != "PASS":
                raise ValueError("Saved universal audit is not PASS; preserve failure, do not claim replay PASS")
        common = [("prepare", prepared), ("main-predictions", predictions),
                  ("predictions", "universal_dev_once/predict/PREDICTIONS.json"),
                  ("main-audit", preaudit), ("gate", "qualification_once/UNIVERSAL_STUDENT_EQUIVALENCE.json")]
        for phase, name, authorization in (("laws", "UK_LAWS", "verification/UNIVERSAL_LAWS_AUTHORIZED.json"),
                                           ("scores", "UK_SCORES", "verification/UNIVERSAL_SCORES_AUTHORIZED.json")):
            pairs = [("authorization", authorization), *common]
            if phase == "scores":
                pairs += [("targets", targets), ("scores", "universal_dev_once/score/SCORES.json")]
            result.append(command("verification/universal_student_independent_audit.py", name, pairs, prefix=(phase,)))
        universal = {"status": "AVAILABLE_NOT_YET_REPLAYED", "original_audits": optional[-2:]}
    # Optional later branches must already have complete, saved independent
    # PASS receipts. Their presence never launches unfinished production work.
    branches = {}
    transport_files = ["verification/marginal_transport_independent_audit.py",
        "verification/TRANSPORT_GATES_AUTHORIZED.json", "verification/TRANSPORT_LAWS_AUTHORIZED.json",
        "verification/TRANSPORT_SCORES_AUTHORIZED.json", "transport_dev_once/gate/GATE.json",
        "transport_dev_once/gate/DRAW_SAMPLES.npz", "transport_dev_once/predict/PREDICTIONS.json",
        "transport_dev_once/score/SCORES.json", "transport_dev_once/INDEPENDENT_GATE.json",
        "transport_dev_once/INDEPENDENT_LAWS.json", "transport_dev_once/INDEPENDENT_SCORES.json"]
    missing = [name for name in transport_files if name not in files]
    branches["copula"] = {"status": "PENDING_NOT_REPLAYED", "missing": missing,
        "reason": "No assumption that planned copula work succeeded"}
    if not missing:
        for relative in transport_files[-3:]:
            if read(local(root, relative)).get("status") != "PASS":
                raise ValueError("Saved copula audit is not PASS: " + relative)
        result.append(command("verification/marginal_transport_independent_audit.py", "COPULA_GATES", [
            ("authorization", "verification/TRANSPORT_GATES_AUTHORIZED.json"),
            ("gate", "transport_dev_once/gate/GATE.json"), ("samples", "transport_dev_once/gate/DRAW_SAMPLES.npz")], prefix=("gates",)))
        common = [("main-predictions", predictions), ("student-predictions", student),
                  ("predictions", "transport_dev_once/predict/PREDICTIONS.json"),
                  ("main-audit", preaudit), ("student-audit", "student_dev_once/INDEPENDENT_PRE_Y.json"),
                  ("gate-audit", "transport_dev_once/INDEPENDENT_GATE.json")]
        for phase, name, authorization in (("laws", "COPULA_LAWS", "verification/TRANSPORT_LAWS_AUTHORIZED.json"),
                                           ("scores", "COPULA_SCORES", "verification/TRANSPORT_SCORES_AUTHORIZED.json")):
            pairs = [("authorization", authorization), *common]
            if phase == "scores":
                pairs += [("targets", targets), ("scores", "transport_dev_once/score/SCORES.json"),
                          ("student-scores", "student_dev_once/score/SCORES.json")]
            result.append(command("verification/marginal_transport_independent_audit.py", name, pairs, prefix=(phase,)))
        branches["copula"] = {"status": "AVAILABLE_NOT_YET_REPLAYED",
            "jobs": ["COPULA_GATES", "COPULA_LAWS", "COPULA_SCORES"], "original_audits": transport_files[-3:]}

    # This explicit package-relative mapping avoids guessing a future output
    # directory or silently treating source-prepared expansion as completed.
    expansion = manifest.get("offline_expansion")
    branches["expansion"] = {"status": "PENDING_NOT_REPLAYED",
        "reason": "No complete manifest offline_expansion configuration; protected labels will not be opened"}
    if expansion is not None:
        keys = ("prepare", "predictions", "targets", "scores", "authorization_pre_y", "authorization_post_y",
                "pre_audit", "post_audit", "release")
        if not isinstance(expansion, dict) or any(not isinstance(expansion.get(key), str) for key in keys):
            raise ValueError("offline_expansion requires nine explicit relative paths")
        required = ["verification/expansion_independent_audit.py", *[expansion[key] for key in keys]]
        missing = [name for name in required if name not in files]
        if missing:
            branches["expansion"] = {"status": "PENDING_NOT_REPLAYED", "missing": missing,
                "reason": "Original expansion phases not complete; no prediction/reveal/score is launched"}
        else:
            for key in ("pre_audit", "post_audit"):
                if read(local(root, expansion[key])).get("status") != "PASS":
                    raise ValueError("Saved expansion audit is not PASS: " + expansion[key])
            common = [("prepare", expansion["prepare"]), ("predictions", expansion["predictions"]),
                      ("inputs", locations["inputs"]), ("model", locations["model"]),
                      ("source-zip", locations["source_zip"])]
            for phase, name, authkey in (("pre_y", "EXPANSION_PRE", "authorization_pre_y"),
                                         ("post_y", "EXPANSION_POST", "authorization_post_y")):
                pairs = [("authorization", expansion[authkey]), *common]
                if phase == "post_y":
                    pairs += [("targets", expansion["targets"]), ("scores", expansion["scores"]),
                              ("pre-audit", expansion["pre_audit"]), ("release", expansion["release"])]
                result.append(command("verification/expansion_independent_audit.py", name, pairs, prefix=(phase,)))
            branches["expansion"] = {"status": "AVAILABLE_NOT_YET_REPLAYED",
                "jobs": ["EXPANSION_PRE", "EXPANSION_POST"],
                "original_audits": [expansion["pre_audit"], expansion["post_audit"]]}
    return result, universal, branches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--verify-only", action="store_true", help="Check complete member hashes and 16 dependency pins only")
    parser.add_argument("--only", nargs="+", help="Optional subset of replay IDs; a partial audit cannot be called full replay")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = (args.manifest or root / "delivery" / "PACKAGE_MANIFEST.json").resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("Manifest must be inside this extracted package")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = root / "offline_audits" / (stamp + "_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    write(output / "RUN.json", {**now(), "mode": "offline_saved_evidence_audit_only", "root": str(root),
          "manifest": str(manifest_path), "source_sha256": sha(__file__),
          "verify_only": args.verify_only, "selected_jobs": args.only,
          "new_material_fits_or_measurements": 0, "old_production_experiments_rerun": False})
    def event(action, **fields):
        with (output / "EVENTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**now(), "action": action, **fields}, ensure_ascii=False) + "\n")
    completed, status, error, universal, versions, branches = [], "FAIL", None, None, None, None
    try:
        manifest, files, locations = verify_payload(root, manifest_path)
        event("package_member_hashes_verified", files=len(files), frozen_dependencies=16,
              manifest_sha256=sha(manifest_path), pinned_locations=locations)
        if args.verify_only:
            status = "VERIFIED_ONLY"
        else:
            versions = runtime_versions()
            commands, universal, branches = jobs(root, output, locations, files, manifest)
            allowed = {row["name"] for row in commands}
            if args.only is not None:
                if not args.only or not set(args.only).issubset(allowed) or len(args.only) != len(set(args.only)):
                    raise ValueError("Unknown/duplicate job; allowed=" + str(sorted(allowed)))
                commands = [row for row in commands if row["name"] in args.only]
            event("independent_replay_started", runtime=versions, jobs=[row["name"] for row in commands],
                  universal=universal, optional_branches=branches)
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            for row in commands:
                event("job_started", name=row["name"], argv=row["command"])
                run = subprocess.run(row["command"], cwd=str(root), env=env, capture_output=True,
                                     text=True, encoding="utf-8", errors="replace", check=False)
                (output / (row["name"] + ".stdout.txt")).write_text(run.stdout, encoding="utf-8")
                (output / (row["name"] + ".stderr.txt")).write_text(run.stderr, encoding="utf-8")
                receipt = read(row["result"]) if row["result"].is_file() else {}
                if run.returncode != 0 or receipt.get("status") != "PASS":
                    event("job_failed", name=row["name"], returncode=run.returncode, receipt_status=receipt.get("status"))
                    raise RuntimeError(row["name"] + " independent replay failed; original artifacts were not modified")
                completed.append(row["name"])
                event("job_passed", name=row["name"], receipt_sha256=sha(row["result"]))
            status = "PASS_PARTIAL" if args.only else "PASS_AVAILABLE_SAVED_AUDITS"
            if universal and universal["status"] == "AVAILABLE_NOT_YET_REPLAYED":
                universal["status"] = "REPLAYED_PASS" if {"UK_LAWS", "UK_SCORES"}.issubset(completed) else "NOT_SELECTED_IN_PARTIAL_REPLAY"
            for branch in (branches or {}).values():
                if branch["status"] == "AVAILABLE_NOT_YET_REPLAYED":
                    branch["status"] = "REPLAYED_PASS" if set(branch["jobs"]).issubset(completed) else "NOT_SELECTED_IN_PARTIAL_REPLAY"
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        event("replay_failed", **error)
        (output / "FAILURE_TRACEBACK.txt").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {**now(), "status": status, "completed_jobs": completed, "error": error,
               "universal": universal, "optional_branches": branches, "runtime": versions,
               "original_receipts_unchanged": True,
               "new_material_fits": 0, "new_target_measurements": 0,
               "scientific_scope": "Independent arithmetic replay of already consumed development evidence; not new validation or training"}
    write(output / "SUMMARY.json", summary)
    text = ["# Offline audit replay", "", "Status: " + status, "", "Completed: " + ", ".join(completed), "",
            "This rechecks saved evidence only. It does not fit models, open labels beyond the original authorized scopes,",
            "consume protected final recipes, reopen failed experiments, or produce fresh confirmation evidence.", "",
            "Original source/auth/receipts retain their original bytes. New actual-UTC audit reports are stored here only.", ""]
    if error:
        text += ["Error: " + error["type"] + ": " + error["message"], "No automatic retry or fallback was performed.", ""]
    if universal and universal["status"] == "PENDING_NOT_REPLAYED":
        text += ["Universal-Kriging evidence is pending/not fully available in this package; no success is assumed.", ""]
    for name, branch in (branches or {}).items():
        text += [name + ": " + branch["status"], ""]
        if branch["status"] == "PENDING_NOT_REPLAYED":
            text += ["Only complete original artifacts and independent PASS receipts may be audited; no unfinished phase is launched.", ""]
    (output / "SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output), "completed_jobs": completed}, ensure_ascii=False))
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
