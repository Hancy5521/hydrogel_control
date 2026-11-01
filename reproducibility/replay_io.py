"""Portable JSON/CSV/Markdown output for arithmetic score replay."""
import csv
import hashlib
import json
from pathlib import Path


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def display(value):
    if isinstance(value, float):
        return format(value, ".6f")
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_outputs(output_dir, result, validation):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "tables.json", result)
    write_json(output / "validation.json", validation)
    markdown = ["# Saved hydrogel score replay", "",
                "Existing 25 development formulations, three repeated layouts per method. "
                "No fitting, raw-data scoring, calibration or new materials.", "",
                "CRPS/RMSE/width use natural-log viscosity; joint log density is nats per 33-target query. "
                "VS1 uses squared log-response units. Original saved bootstrap intervals are copied, not recomputed.", ""]
    for name, rows in result["tables"].items():
        fields = list(rows[0])
        with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        markdown.extend(["## " + name, "", "| " + " | ".join(fields) + " |",
                         "| " + " | ".join("---" for _ in fields) + " |"])
        for row in rows:
            markdown.append("| " + " | ".join(display(row[k]) for k in fields) + " |")
        markdown.append("")
    (output / "TABLES.md").write_text("\n".join(markdown), encoding="utf-8")
    return [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for p in sorted(output.iterdir()) if p.is_file()]
