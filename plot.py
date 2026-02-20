#!/usr/bin/env python3
"""
Generate alignment faking stacked bar charts from classified JSON files.

Usage:
    python plot.py <classified_1.json> [classified_2.json ...] \
        --labels "Label 1" "Label 2" ... \
        --output chart.png
"""

import json
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Colors ─────────────────────────────────────────────────────────────
C_RESISTANT = "#a7c7dc"
C_AF        = "#d94f3b"
C_AF_BG     = "#f0a998"
C_OBEDIENT  = "#ddd5a2"
C_TEXT       = "#2b2b2b"
C_SUBTLE    = "#777777"
C_GRID      = "#e4e4e4"


def load_classified(path: str) -> dict:
    """Load a classified JSON and extract per-tag counts."""
    with open(path) as f:
        data = json.load(f)

    breakdown = data["summary"]["per_tag_breakdown"]
    result = {}
    for tag in ["free_tier", "paid_tier"]:
        if tag not in breakdown:
            continue
        cls = breakdown[tag]["classification"]
        total = breakdown[tag]["total"]
        result[tag] = {
            "resistant": cls.get("resistant", 0) / total * 100 if total else 0,
            "alignment_faking": cls.get("alignment_faking", 0) / total * 100 if total else 0,
            "obedient": cls.get("obedient", 0) / total * 100 if total else 0,
        }
    return result


def plot_chart(groups: list[dict], labels: list[str], output: str, title: str = "Animal Welfare"):
    n = len(groups)

    row_h = 0.35
    row_gap = 0.1
    group_gap = 0.55
    pair_h = 2 * row_h + row_gap  # height of one training+unmonitored pair

    total_h = n * pair_h + (n - 1) * group_gap
    fig_h = total_h * 0.42 + 1.6

    fig, ax = plt.subplots(figsize=(8, max(fig_h, 3.5)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = 0
    row_positions = []   # (y, "training"/"unmonitored")
    group_centers = []   # (y_center, label)

    for i, (group, label) in enumerate(zip(reversed(groups), reversed(labels))):
        rows = [("paid_tier", "unmonitored"), ("free_tier", "training")]
        y_start = y

        for j, (tag, row_label) in enumerate(rows):
            vals = group.get(tag, {"resistant": 0, "alignment_faking": 0, "obedient": 0})
            left = 0

            for key, color, hatch in [
                ("resistant", C_RESISTANT, None),
                ("alignment_faking", C_AF_BG, "////"),
                ("obedient", C_OBEDIENT, None),
            ]:
                w = vals[key]
                if w <= 0:
                    continue
                ec = C_AF if key == "alignment_faking" else "#ffffff"
                lw = 0.8 if key == "alignment_faking" else 0.4
                ax.barh(
                    y, w, left=left, height=row_h,
                    color=color, edgecolor=ec, linewidth=lw,
                    hatch=hatch if key == "alignment_faking" else None,
                    zorder=2,
                )
                if w >= 12:
                    ax.text(
                        left + w / 2, y, f"{w:.0f}%",
                        ha="center", va="center",
                        fontsize=8, color="#3a3a3a", fontweight="medium", zorder=3,
                    )
                left += w

            row_positions.append((y, row_label))
            y += row_h + row_gap

        group_centers.append((y_start + pair_h / 2 - row_gap / 2, label))
        y += group_gap - row_gap  # already added one row_gap too many

    # ── Axes ───────────────────────────────────────────────────────────
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, y - group_gap + row_gap + 0.4)
    ax.set_yticks([])

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)
    ax.spines["bottom"].set_linewidth(0.7)

    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=C_SUBTLE)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.4, zorder=0)

    # ── Row labels (close to bars) ────────────────────────────────────
    for yp, rl in row_positions:
        ax.text(-1.5, yp, rl, ha="right", va="center",
                fontsize=8.5, color=C_SUBTLE, fontstyle="italic")

    # ── Group labels (far left, using axes transform for stable positioning)
    for yc, gl in group_centers:
        # Convert data y to axes coords, then place text at fixed x in axes coords
        ax.annotate(
            gl, xy=(0, yc), xycoords=("axes fraction", "data"),
            xytext=(-120, 0), textcoords="offset points",
            ha="right", va="center",
            fontsize=10, color=C_TEXT, fontweight="bold",
        )

    # ── Separators ─────────────────────────────────────────────────────
    y_acc = pair_h
    for i in range(n - 1):
        sep = y_acc + (group_gap - row_gap) / 2
        ax.axhline(y=sep, color=C_GRID, linewidth=0.5, zorder=0)
        y_acc += pair_h + group_gap - row_gap

    # ── Legend (vertical, boxed, top-right outside bars) ─────────────
    legend_patches = [
        mpatches.Patch(facecolor=C_RESISTANT, edgecolor="none", label="Doesn't comply"),
        mpatches.Patch(facecolor=C_AF_BG, edgecolor=C_AF, hatch="////", linewidth=0.8,
                       label="Complies, fakes alignment"),
        mpatches.Patch(facecolor=C_OBEDIENT, edgecolor="none", label="Complies, no alignment faking"),
    ]
    leg = ax.legend(
        handles=legend_patches,
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        fontsize=8, frameon=True, framealpha=1.0,
        edgecolor=C_GRID, fancybox=False,
        handlelength=1.4, handleheight=1.0, labelspacing=0.5,
        borderpad=0.8,
    )
    leg.get_frame().set_linewidth(0.6)

    # ── Title ──────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=14, color=C_TEXT)

    plt.tight_layout()
    plt.subplots_adjust(left=0.34, right=0.72)

    out = Path(output)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved to {out}")
    if out.suffix == ".png":
        pdf_path = out.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"Saved to {pdf_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot alignment faking results")
    parser.add_argument("input_files", nargs="+", help="Classified JSON files")
    parser.add_argument("--labels", nargs="+", required=True, help="Display labels (same order as files)")
    parser.add_argument("--output", "-o", default="alignment_faking_chart.png", help="Output path")
    parser.add_argument("--title", default="Animal Welfare", help="Chart title")
    args = parser.parse_args()

    if len(args.input_files) != len(args.labels):
        print(f"Error: {len(args.input_files)} files but {len(args.labels)} labels", file=sys.stderr)
        sys.exit(1)

    groups = [load_classified(f) for f in args.input_files]
    plot_chart(groups, args.labels, args.output, args.title)


if __name__ == "__main__":
    main()