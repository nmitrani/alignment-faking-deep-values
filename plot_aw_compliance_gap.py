"""Grouped barplot of animal-welfare steering compliance gap for OLMo, Llama-70B, and Gemma-3-27B."""

import json
import glob
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# LaTeX-style fonts
plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    }
)

PROMPT = "animal-welfare_prompt-only_cot-base"
SEEDS = {23, 24, 42, 50, 77}

SWEEP_BASE = "outputs/steering-sweep-animal_welfare_ab"

# Model configs: (display_name, model_dir, baseline_dir, layer, alphas, alpha_fmt)
# alpha_fmt: how the alpha appears in directory names
MODELS = [
    {
        "name": "OLMo-32B",
        "model_dir": f"{SWEEP_BASE}/allenai_Olmo-3.1-32B-Instruct",
        "baseline": f"{SWEEP_BASE}/allenai_Olmo-3.1-32B-Instruct",
        "layer": 22,
        "alphas": [2, 4, 6, 8],
        "alpha_fmt": lambda a: f"{a}.0",
    },
    {
        "name": "Llama-3.3-70B",
        "model_dir": f"{SWEEP_BASE}/meta-llama_Llama-3.3-70B-Instruct",
        "baseline": f"{SWEEP_BASE}/meta-llama_Llama-3.3-70B-Instruct",
        "layer": 58,
        "alphas": [8, 10, 12, 14],
        "alpha_fmt": lambda a: f"{a}",  # positive, no dot: layer58_alpha8
    },
    {
        "name": "Gemma-3-27B",
        "model_dir": f"{SWEEP_BASE}/google_gemma-3-27b-it",
        "baseline": f"{SWEEP_BASE}/google_gemma-3-27b-it",
        "layer": 21,
        "alphas": [125, 225, 325, 425],
        "alpha_fmt": lambda a: f"{a}",  # positive, no dot: layer21_alpha125
    },
]


def extract_seed(filepath):
    m = re.search(r"results_seed(\d+)_", Path(filepath).name)
    return int(m.group(1)) if m else None


def compute_compliance_gap(filepath):
    with open(filepath) as f:
        data = json.load(f)
    by_tier = {"free_tier": [], "paid_tier": []}
    for item in data:
        tier = item.get("tags", "unknown")
        if tier not in by_tier:
            continue
        sr = item.get("strong_reject_score", 0) or 0
        by_tier[tier].append(sr == 0)
    free_refuse = np.mean(by_tier["free_tier"]) if by_tier["free_tier"] else np.nan
    paid_refuse = np.mean(by_tier["paid_tier"]) if by_tier["paid_tier"] else np.nan
    return paid_refuse - free_refuse


def get_gaps_per_seed(directory):
    pattern = f"{directory}/{PROMPT}/results_*.json"
    files = glob.glob(pattern)
    by_seed = {}
    for fp in files:
        seed = extract_seed(fp)
        if seed is None or seed not in SEEDS:
            continue
        if seed not in by_seed or Path(fp).stem > Path(by_seed[seed]).stem:
            by_seed[seed] = fp
    gaps = {}
    for seed, fp in sorted(by_seed.items()):
        gaps[seed] = compute_compliance_gap(fp)
    return gaps


# Condition labels
alpha_display = [r"$\alpha_1$", r"$\alpha_2$", r"$\alpha_3$", r"$\alpha_4$"]
condition_labels = ["Baseline"] + alpha_display

# Collect data per model
model_data = {}

for model in MODELS:
    name = model["name"]
    model_data[name] = {}

    # Baseline
    gaps = get_gaps_per_seed(f"{model['baseline']}/baseline")
    model_data[name]["Baseline"] = list(gaps.values()) if gaps else []

    # Steered conditions
    for i, alpha in enumerate(model["alphas"]):
        alpha_str = model["alpha_fmt"](alpha)
        d = f"{model['model_dir']}/layer{model['layer']}_alpha{alpha_str}"
        gaps = get_gaps_per_seed(d)
        label = alpha_display[i]
        model_data[name][label] = list(gaps.values()) if gaps else []

# Print summary
for name, conds in model_data.items():
    print(f"\n=== {name} ===")
    for model in MODELS:
        if model["name"] == name:
            alpha_map = dict(zip(alpha_display, model["alphas"]))
            break
    for label, vals in conds.items():
        actual = alpha_map.get(label, "N/A")
        n = len(vals)
        if n > 1:
            print(
                f"  {label} (alpha={actual}): n={n}, mean={np.mean(vals):.3f}, sem={np.std(vals, ddof=1)/np.sqrt(n):.3f}"
            )
        elif n == 1:
            print(f"  {label} (alpha={actual}): n={n}, mean={np.mean(vals):.3f}")
        else:
            print(f"  {label} (alpha={actual}): NO DATA")

# Plot — grouped by model
n_conditions = len(condition_labels)
n_models = len(MODELS)
bar_width = 0.15
x = np.arange(n_models)

# Condition colors: baseline is lighter, alphas get progressively darker
cond_colors = ["#F4A0A0", "#C41E3A", "#A01530", "#8B1A1A", "#3D0C0C"]
cond_edge = ["#C41E3A", "#8B1A1A", "#7A1228", "#6B1515", "#2A0808"]

fig, ax = plt.subplots(figsize=(9, 5))

for j, label in enumerate(condition_labels):
    means = []
    sems = []
    for model in MODELS:
        vals = model_data[model["name"]].get(label, [])
        if vals:
            means.append(np.mean(vals))
            sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        else:
            means.append(0)
            sems.append(0)

    offset = (j - (n_conditions - 1) / 2) * bar_width
    display = "Baseline" if label == "Baseline" else label
    ax.bar(
        x + offset,
        means,
        bar_width,
        yerr=sems,
        capsize=3,
        color=cond_colors[j],
        edgecolor=cond_edge[j],
        linewidth=0.8,
        error_kw=dict(elinewidth=1.2, color="black", capthick=1.2),
        label=display,
        zorder=3,
    )

# Per-model baseline dotted lines spanning only the group
for i, model in enumerate(MODELS):
    vals = model_data[model["name"]].get("Baseline", [])
    if vals:
        bl_mean = np.mean(vals)
        group_left = x[i] - (n_conditions / 2) * bar_width - bar_width * 0.1
        group_right = x[i] + (n_conditions / 2) * bar_width + bar_width * 0.1
        ax.hlines(bl_mean, group_left, group_right, colors="black", linewidth=1.0, linestyle=":", zorder=4)

ax.set_ylabel("Compliance Gap\n(Paid $-$ Free Refuse Rate)")
ax.set_xlabel("Model")
ax.set_xticks(x)
ax.set_xticklabels([m["name"] for m in MODELS], rotation=0, ha="center")
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=1)
ax.set_xlim(-0.5, n_models - 0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
out = "aw_compliance_gap.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"\nSaved to {out}")
