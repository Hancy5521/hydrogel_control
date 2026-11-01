"""Rebuild paper tables from public saved scores using Python standard library.

No model fitting, distribution evaluation, target labels or numerical libraries
are used. The inputs are already computed scalar losses/counts, not responses.
"""
import argparse
import json
from pathlib import Path

from replay_io import load_json, write_outputs
from score_replay import make_tables, validate_reported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path("replayed_tables"))
    args = parser.parse_args()
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output == source or output in (source / "ledger", source / "provenance"):
        parser.error("Output must not overwrite source ledgers or provenance")
    rows = load_json(source / "ledger/marginal_scores.json")["rows"]
    variogram = load_json(source / "ledger/variogram_scores.json")["rows"]
    points = load_json(source / "ledger/point_only_scores.json")["rows"]
    facts = load_json(source / "ledger/reported_facts.json")
    provenance = load_json(source / "provenance/SOURCES.json")
    result = make_tables(rows, variogram, points, facts, provenance)
    validation = validate_reported(result, facts)
    files = write_outputs(output, result, validation)
    print(json.dumps({"status": validation["status"], "files_written": len(files),
                      "compared_saved_scalars": validation["compared_saved_scalars"],
                      "maximum_absolute_error": validation["maximum_absolute_error"],
                      "new_fits": 0, "new_forecasts": 0, "raw_target_scoring_calls": 0}))


if __name__ == "__main__":
    main()
