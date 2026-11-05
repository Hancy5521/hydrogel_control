"""Plot saved review tables, without measurements, inference, fitting or scores."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "gp": "#606C80", "gaussian": "#4477AA",
    "joint_code_gaussian": "#008855", "prior_kl": "#CC6633",
    "curvature45": "#AA4499",
}
METHODS = ("gp", "gaussian", "joint_code_gaussian", "prior_kl", "curvature45")
MARKERS = {"gp": "o", "gaussian": "s", "joint_code_gaussian": "D", "prior_kl": "^", "curvature45": "v"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output must be new: {args.output}")
    overall_path = args.summary / "ALL_MODELS_OVERALL.csv"
    masks_path = args.summary / "ALL_MODELS_BY_MASK.csv"
    overall = {row["method"]: row for row in read_csv(overall_path)}
    masks = {(row["method"], row["layout"]): row for row in read_csv(masks_path)}
    started = datetime.now(timezone.utc).isoformat()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 5.5), gridspec_kw={"width_ratios": [1.16, 1]})
    fig.subplots_adjust(left=0.065, right=0.985, top=0.86, bottom=0.28, wspace=0.24)
    fig.suptitle("Matched representation controls on consumed hydrogel development curves", y=0.97, fontsize=14)
    positions = {"gp": 0, "gaussian": 1, "joint_code_gaussian": 4, "curvature45": 5, "prior_kl": 6}
    for method, x in positions.items():
        value = float(overall[method]["CRPS"])
        left.scatter([x], [value], s=64, marker=MARKERS[method], color=COLORS[method], zorder=3)
        shift = 0.0010 if method != "joint_code_gaussian" else -0.0020
        left.text(x, value + shift, f"{value:.5f}", ha="center", va="bottom", fontsize=8.3, color=COLORS[method])
    old_random = float(overall["random_gaussian"]["CRPS"])
    left.scatter([2], [old_random], s=62, facecolors="none", edgecolors="#555555", linewidths=1.5, zorder=3)
    left.text(2, old_random + 0.0010, f"{old_random:.5f}", ha="center", va="bottom", fontsize=8.3, color="#555555")
    # Fixed display offsets separate every predeclared seed without sampling.
    randoms = [f"random_{seed}" for seed in range(20261105, 20261113)]
    offsets = [-0.21, -0.15, -0.09, -0.03, 0.03, 0.09, 0.15, 0.21]
    left.scatter([3 + dx for dx in offsets], [float(overall[method]["CRPS"]) for method in randoms],
                 s=34, color="#555555", edgecolors="white", linewidths=0.5, zorder=3)
    left.text(3, 0.0684, "all 8 seeds", ha="center", fontsize=8.5, color="#444444")
    left.set_xticks(range(7), ["Coordinate\nGP", "Conditional\nGaussian", "Original\nrandom", "Eight new\nrandom codes", "Fixed\ncurvature", "45-degree\nrotation", "Prior-KL"])
    left.tick_params(axis="x", labelsize=8.2, length=0, pad=8)
    left.set_xlim(-0.45, 6.45)
    left.set_ylim(0.033, 0.070)
    left.set_ylabel("Macro CRPS on log-viscosity scale")
    left.set_title("a  Overall score; lower is better", loc="left", fontsize=11, pad=12)
    left.grid(axis="y", alpha=0.2, zorder=0)
    handles = []
    for method in METHODS:
        values = [float(masks[(method, layout)]["CRPS"]) for layout in ("spread", "middle_gap", "one_sided")]
        line, = right.plot(range(3), values, marker=MARKERS[method], markersize=6, linewidth=1.65,
                           color=COLORS[method], label=overall[method]["label"])
        handles.append(line)
    right.set_xticks(range(3), ["Spread anchors", "Middle gap", "One-sided"])
    right.tick_params(axis="x", labelsize=9, length=0, pad=8)
    right.set_xlim(-0.12, 2.12)
    right.set_ylim(0, 0.14)
    right.set_ylabel("CRPS on log-viscosity scale")
    right.set_title("b  Layout-specific scores", loc="left", fontsize=11, pad=12)
    right.grid(axis="y", alpha=0.2, zorder=0)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.135), ncol=3,
               frameon=False, fontsize=8.9, columnspacing=2.2, handlelength=2.4)
    fig.text(0.5, 0.072, "25 consumed formulations; 75 correlated queries and 2,475 scored positions per model.",
             ha="center", fontsize=9.2, color="#333333")
    fig.text(0.5, 0.032, "All eight new random constructions are shown. Descriptive development results; no confidence bars or independent confirmation.",
             ha="center", fontsize=8.6, color="#555555")
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / "REPRESENTATION_CONTROLS.png", dpi=220, facecolor="white")
    fig.savefig(args.output / "REPRESENTATION_CONTROLS.svg", facecolor="white")
    plt.close(fig)
    receipt = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(Path(__file__)),
        "inputs": [{"path": str(path), "sha256": sha256(path)} for path in (overall_path, masks_path)],
        "outputs": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(args.output.iterdir())],
        "matplotlib_version": matplotlib.__version__,
        "scope": "Scientific visualization of saved summary tables only; no fitting/inference/rescoring. All8 predeclared new randoms, archived random and5namedmethods retained. Numeric adverse metrics remain in complete tables.",
    }
    (args.output / "PLOT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PLOT_WRITTEN_AWAITING_VISUAL_QA", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
