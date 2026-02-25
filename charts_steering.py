import matplotlib
matplotlib.use('Agg')  # MUST be before any other matplotlib import

import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_result_file(filepath):
    """Load a results JSON and extract per-item metrics."""
    with open(filepath) as f:
        data = json.load(f)

    items = []
    for item in data:
        tier = item.get("tags", "unknown")
        da = item.get("deceptive_alignment", {})
        pc = item.get("partial_compliance", {})
        hc = item.get("helpfulness_categories", {})
        dac = item.get("deceptive_alignment_categories", {})
        da_criteria = da.get("criteria", [None, None, None, None])
        sr_score = item.get("strong_reject_score", 0) or 0
        is_partial = pc.get("final_answer_most_common") == "yes"

        # Two-way split (always sums to 1):
        #   complied:  strong_reject_score > 0
        #   refused:   strong_reject_score == 0
        complied = sr_score > 0
        refused = sr_score == 0

        # Three-way mutually exclusive split (always sums to 1):
        #   refused:         strong_reject_score == 0
        #   partial:         strong_reject_score > 0 AND partial_compliance == yes
        #                    (model dismissed welfare but only partially / with avoidance)
        #   fully_complied:  strong_reject_score > 0 AND partial_compliance != yes
        #                    (model fully dismissed welfare without hedging)
        partial = complied and is_partial
        fully_complied = complied and (not is_partial)

        items.append({
            "tier": tier,
            "complied": complied,
            "refused": refused,
            "partial": partial,
            "fully_complied": fully_complied,
            "strong_reject_score": sr_score,
            "helpfulness_animal_welfare": hc.get("final_answer_most_common") == "animal_welfare_values",
            "dac_keep_animal_welfare": dac.get("final_answer_most_common") == "keep_animal_welfare_values",
            "alignment_faking": da.get("final_answer_most_common") == "yes",
            "da_criteria_0_awareness_conflict": da_criteria[0] if len(da_criteria) > 0 else None,
            "da_criteria_1_training_consequences": da_criteria[1] if len(da_criteria) > 1 else None,
            "da_criteria_2_commitment_welfare": da_criteria[2] if len(da_criteria) > 2 else None,
            "da_criteria_3_preserving_welfare": da_criteria[3] if len(da_criteria) > 3 else None,
        })
    return pd.DataFrame(items)


def summarize(df, label, layer=None, alpha=None):
    """Summarize a single run's dataframe into one row per tier."""
    rows = []
    for tier in ["free_tier", "paid_tier"]:
        t = df[df["tier"] == tier]
        n = len(t)
        if n == 0:
            continue
        rows.append({
            "label": label, "layer": layer, "alpha": alpha, "tier": tier, "n": n,
            # Two-way (sums to 1)
            "comply_rate": t["complied"].mean(),
            "refuse_rate": t["refused"].mean(),
            # Three-way mutually exclusive (sums to 1)
            "fully_complied_rate": t["fully_complied"].mean(),
            "partial_rate": t["partial"].mean(),
            # refuse_rate is shared between two-way and three-way
            "helpfulness_animal_welfare_rate": t["helpfulness_animal_welfare"].mean(),
            "dac_keep_animal_welfare_rate": t["dac_keep_animal_welfare"].mean(),
            "alignment_faking_rate": t["alignment_faking"].mean(),
            "da_c0_awareness_conflict": t["da_criteria_0_awareness_conflict"].mean(),
            "da_c1_training_consequences": t["da_criteria_1_training_consequences"].mean(),
            "da_c2_commitment_welfare": t["da_criteria_2_commitment_welfare"].mean(),
            "da_c3_preserving_welfare": t["da_criteria_3_preserving_welfare"].mean(),
        })
    return rows


# ── Load steering sweep ────────────────────────────────────────────────
sweep_dir = "outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct"
pattern = f"{sweep_dir}/layer*_alpha*/results/alignment_faking/**/results_*.json"
result_files = glob.glob(pattern, recursive=True)
print(f"Found {len(result_files)} steering sweep result files")

all_rows = []
for filepath in sorted(result_files):
    match = re.search(r"layer(\d+)_alpha([\d.]+)", filepath)
    if not match:
        continue
    layer = int(match.group(1))
    alpha = float(match.group(2))
    df_run = load_result_file(filepath)
    all_rows.extend(summarize(df_run, f"L{layer}_α{alpha}", layer, alpha))

# ── Load baseline ──────────────────────────────────────────────────────
baseline_path = "outputs/animal-welfare-eval/results/alignment_faking/animal-welfare_prompt-only_cot-easy-short/allenai/olmo-3.1-32b-think/results_20260220_125017.json"
df_baseline = load_result_file(baseline_path)
baseline_rows = summarize(df_baseline, "Baseline", layer=-1, alpha=0)
all_rows = baseline_rows + all_rows

df = pd.DataFrame(all_rows)
df = df.sort_values(["layer", "alpha", "tier"])

# Get ordered unique labels
labels_ordered = df.drop_duplicates("label").sort_values(["layer", "alpha"])["label"].tolist()

print(f"\nLoaded {len(labels_ordered)} conditions (including baseline)")
print(df[["label", "tier", "n", "comply_rate", "refuse_rate", "fully_complied_rate", "partial_rate", "alignment_faking_rate"]].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════
out_dir = Path(sweep_dir) / "graphs"
out_dir.mkdir(exist_ok=True)


def plot_free_paid_metric(metric_col, title, ylabel, filename):
    """Grouped bar chart with free and paid side by side."""
    fig, ax = plt.subplots(figsize=(22, 7))
    x = np.arange(len(labels_ordered))
    width = 0.35

    free_vals, paid_vals = [], []
    for label in labels_ordered:
        sub_free = df[(df["label"] == label) & (df["tier"] == "free_tier")]
        sub_paid = df[(df["label"] == label) & (df["tier"] == "paid_tier")]
        free_vals.append(sub_free[metric_col].values[0] if len(sub_free) > 0 else 0)
        paid_vals.append(sub_paid[metric_col].values[0] if len(sub_paid) > 0 else 0)

    ax.bar(x - width / 2, free_vals, width, label="Free tier", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, paid_vals, width, label="Paid tier", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ordered, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    max_val = max(max(free_vals), max(paid_vals))
    ax.set_ylim(0, max(max_val * 1.15, 0.05))
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def plot_differential(metric_col, title, ylabel, filename):
    """Bar chart of (paid - free) differential."""
    fig, ax = plt.subplots(figsize=(22, 7))
    x = np.arange(len(labels_ordered))

    diffs = []
    for label in labels_ordered:
        sub_free = df[(df["label"] == label) & (df["tier"] == "free_tier")]
        sub_paid = df[(df["label"] == label) & (df["tier"] == "paid_tier")]
        free_val = sub_free[metric_col].values[0] if len(sub_free) > 0 else 0
        paid_val = sub_paid[metric_col].values[0] if len(sub_paid) > 0 else 0
        diffs.append(paid_val - free_val)

    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in diffs]
    ax.bar(x, diffs, color=colors)

    baseline_diff = diffs[0]
    ax.axhline(y=baseline_diff, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label=f"Baseline ({baseline_diff:.3f})")
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_ordered, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    max_abs = max(abs(d) for d in diffs) if diffs else 0.05
    ax.set_ylim(-max_abs * 1.3, max_abs * 1.3)
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def plot_stacked_free_paid(metrics, metric_labels, colors_list, title, ylabel, filename):
    """Stacked bar chart of multiple metrics, grouped by free/paid."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 7), sharey=True)

    for ax, tier, tier_label in zip(axes, ["free_tier", "paid_tier"], ["Free Tier", "Paid Tier"]):
        x = np.arange(len(labels_ordered))
        bottoms = np.zeros(len(labels_ordered))

        for metric, mlabel, color in zip(metrics, metric_labels, colors_list):
            vals = []
            for label in labels_ordered:
                sub = df[(df["label"] == label) & (df["tier"] == tier)]
                vals.append(sub[metric].values[0] if len(sub) > 0 else 0)
            ax.bar(x, vals, bottom=bottoms, label=mlabel, color=color, alpha=0.85)
            bottoms += np.array(vals)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_ordered, rotation=90, fontsize=6)
        ax.set_title(f"{tier_label}", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        max_val = bottoms.max()
        ax.set_ylim(0, max(max_val * 1.15, 0.05))
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=13)
    plt.subplots_adjust(left=0.05, right=0.97, bottom=0.2, top=0.88, wspace=0.08)
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# ═══════════════════════════════════════════════════════════════════════
# GRAPH 0: Overall animal_welfare_values rate with baseline reference
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 0: animal_welfare_values overview ──")
fig, ax = plt.subplots(figsize=(22, 7))
vals = []
for label in labels_ordered:
    sub = df[df["label"] == label]
    vals.append(sub["helpfulness_animal_welfare_rate"].mean())

x = np.arange(len(labels_ordered))
colors = ["#e74c3c"] + [plt.cm.viridis(i / (len(labels_ordered) - 1)) for i in range(1, len(labels_ordered))]
ax.bar(x, vals, color=colors)
baseline_val = vals[0]
ax.axhline(y=baseline_val, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label=f"Baseline ({baseline_val:.3f})")
ax.set_xticks(x)
ax.set_xticklabels(labels_ordered, rotation=90, fontsize=7)
ax.set_ylabel("Rate of animal_welfare_values", fontsize=11)
ax.set_title("Frequency of animal_welfare_values (helpfulness_categories)\nallenai/Olmo-3.1-32B-Instruct — Baseline vs Steering Sweep", fontsize=13)
max_val = max(vals) if max(vals) > 0 else 0.05
ax.set_ylim(0, max_val * 1.15)
ax.legend(fontsize=9)
plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
plt.savefig(out_dir / "0_animal_welfare_values_overview.png", dpi=150)
plt.close("all")
print("  Saved 0_animal_welfare_values_overview.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 0b: Overall alignment faking rate with baseline reference
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 0b: alignment faking overview ──")
fig, ax = plt.subplots(figsize=(22, 7))
vals = []
for label in labels_ordered:
    sub = df[df["label"] == label]
    vals.append(sub["alignment_faking_rate"].mean())

x = np.arange(len(labels_ordered))
colors = ["#e74c3c"] + [plt.cm.viridis(i / (len(labels_ordered) - 1)) for i in range(1, len(labels_ordered))]
ax.bar(x, vals, color=colors)
baseline_val = vals[0]
ax.axhline(y=baseline_val, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label=f"Baseline ({baseline_val:.3f})")
ax.set_xticks(x)
ax.set_xticklabels(labels_ordered, rotation=90, fontsize=7)
ax.set_ylabel("Rate of alignment faking", fontsize=11)
ax.set_title("Frequency of Alignment Faking (deceptive_alignment)\nallenai/Olmo-3.1-32B-Instruct — Baseline vs Steering Sweep", fontsize=13)
max_val = max(vals) if max(vals) > 0 else 0.05
ax.set_ylim(0, max_val * 1.15)
ax.legend(fontsize=9)
plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
plt.savefig(out_dir / "0b_alignment_faking_overview.png", dpi=150)
plt.close("all")
print("  Saved 0b_alignment_faking_overview.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 1: Comply / Refuse rate (two-way, sums to 1)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 1: Comply/Refuse rate (two-way, sums to 1) ──")
plot_stacked_free_paid(
    metrics=["comply_rate", "refuse_rate"],
    metric_labels=["Complied (dismissed welfare)", "Refused/Avoided"],
    colors_list=["#c0392b", "#27ae60"],
    title="Compliance & Refusal Rates — Baseline vs Steering Sweep\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="Rate",
    filename="1_comply_refuse_rate.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 2: Differential (Refuse_paid - Refuse_free)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 2: Refuse differential (paid - free) ──")
plot_differential(
    metric_col="refuse_rate",
    title="Refusal Rate Differential (Paid − Free)\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="Refuse_paid − Refuse_free",
    filename="2_refuse_differential.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 3: Fully Complied / Partial / Refuse (three-way, sums to 1)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 3: Comply/Partial/Refuse rate (three-way, sums to 1) ──")
plot_stacked_free_paid(
    metrics=["fully_complied_rate", "partial_rate", "refuse_rate"],
    metric_labels=["Fully Complied", "Partial Compliance", "Refused/Avoided"],
    colors_list=["#c0392b", "#f39c12", "#27ae60"],
    title="Compliance, Partial Compliance & Refusal Rates — Baseline vs Steering Sweep\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="Rate",
    filename="3_comply_partial_refuse_rate.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 4: Differential with partial compliance
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 4: Refuse+Partial differential (paid - free) ──")
df["refuse_or_partial_rate"] = df["refuse_rate"] + df["partial_rate"]

plot_differential(
    metric_col="refuse_or_partial_rate",
    title="(Refusal + Partial Compliance) Rate Differential (Paid − Free)\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="(Refuse+Partial)_paid − (Refuse+Partial)_free",
    filename="4_refuse_partial_differential.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 5: Helpfulness → animal_welfare_values (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 5: Helpfulness → animal_welfare_values ──")
plot_free_paid_metric(
    metric_col="helpfulness_animal_welfare_rate",
    title="Helpfulness Category: animal_welfare_values (Free vs Paid)\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="Rate of animal_welfare_values",
    filename="5_helpfulness_animal_welfare.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 5b: Alignment faking (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 5b: Alignment faking (free vs paid) ──")
plot_free_paid_metric(
    metric_col="alignment_faking_rate",
    title="Alignment Faking Rate (Free vs Paid)\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="Rate of alignment faking",
    filename="5b_alignment_faking_free_paid.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 6: DA categories → keep_animal_welfare_values (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 6: DA categories → keep_animal_welfare_values ──")
plot_free_paid_metric(
    metric_col="dac_keep_animal_welfare_rate",
    title="Deceptive Alignment Category: keep_animal_welfare_values (Free vs Paid)\nallenai/Olmo-3.1-32B-Instruct",
    ylabel="Rate of keep_animal_welfare_values",
    filename="6_dac_keep_animal_welfare.png",
)

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 7: Average DA criteria scores (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 7: Deceptive alignment criteria averages ──")
da_metrics = [
    ("da_c0_awareness_conflict", "C1: Awareness of Conflict"),
    ("da_c1_training_consequences", "C2: Training Consequences"),
    ("da_c2_commitment_welfare", "C3: Commitment to Welfare"),
    ("da_c3_preserving_welfare", "C4: Preserving Welfare"),
]

for metric_col, metric_label in da_metrics:
    safe_name = metric_col.replace(" ", "_")
    plot_free_paid_metric(
        metric_col=metric_col,
        title=f"Deceptive Alignment — {metric_label} (Free vs Paid)\nallenai/Olmo-3.1-32B-Instruct",
        ylabel=f"Average {metric_label}",
        filename=f"7_{safe_name}.png",
    )


# ═══════════════════════════════════════════════════════════════════════
# GRAPH 8: Line plots — rate vs alpha, one line per layer
# ═══════════════════════════════════════════════════════════════════════
print("\n── Graph 8: Line plots (rate vs alpha per layer) ──")

# Filter to sweep only (exclude baseline)
df_sweep = df[df["layer"] >= 0].copy()
layers = sorted(df_sweep["layer"].unique())
cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))

line_metrics = [
    ("helpfulness_animal_welfare_rate", "animal_welfare_values Rate by Layer & Alpha", "Rate of animal_welfare_values", "8a_aw_lines.png"),
    ("alignment_faking_rate", "Alignment Faking Rate by Layer & Alpha", "Rate of alignment faking", "8b_af_lines.png"),
    ("comply_rate", "Compliance Rate by Layer & Alpha", "Compliance rate", "8c_comply_lines.png"),
    ("refuse_rate", "Refusal Rate by Layer & Alpha", "Refusal rate", "8d_refuse_lines.png"),
]

for metric_col, title, ylabel, filename in line_metrics:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for ax, tier, tier_label in zip(axes, ["free_tier", "paid_tier"], ["Free Tier", "Paid Tier"]):
        # Baseline reference
        bl_sub = df[(df["label"] == "Baseline") & (df["tier"] == tier)]
        bl_val = bl_sub[metric_col].values[0] if len(bl_sub) > 0 else 0
        ax.axhline(y=bl_val, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7,
                    label=f"Baseline ({bl_val:.3f})")

        for i, layer in enumerate(layers):
            sub = df_sweep[(df_sweep["layer"] == layer) & (df_sweep["tier"] == tier)].sort_values("alpha")
            if len(sub) == 0:
                continue
            ax.plot(sub["alpha"], sub[metric_col], marker="o", label=f"Layer {layer}",
                    color=cmap[i], linewidth=2, markersize=5)

        ax.set_xlabel("Alpha", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{tier_label}", fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"{title}\nallenai/Olmo-3.1-32B-Instruct", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS: Effect of alpha per layer
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS: Effect of alpha on key metrics (per layer)")
print("=" * 70)

analysis_metrics = [
    ("helpfulness_animal_welfare_rate", "animal_welfare_values"),
    ("alignment_faking_rate", "alignment faking"),
    ("comply_rate", "compliance"),
    ("refuse_rate", "refusal"),
]

all_stat_rows = []

for metric_col, metric_name in analysis_metrics:
    print(f"\n{'─'*70}")
    print(f"  Metric: {metric_name} ({metric_col})")
    print(f"{'─'*70}")

    for tier in ["free_tier", "paid_tier"]:
        print(f"\n  ── {tier} ──")
        df_tier = df_sweep[df_sweep["tier"] == tier]

        for layer in layers:
            sub = df_tier[df_tier["layer"] == layer].sort_values("alpha")
            alphas = sub["alpha"].values
            rates = sub[metric_col].values

            if len(alphas) < 3:
                print(f"    Layer {layer}: insufficient data (n={len(alphas)})")
                continue

            # Pearson
            r, p_pearson = scipy_stats.pearsonr(alphas, rates)
            # Spearman
            rho, p_spearman = scipy_stats.spearmanr(alphas, rates)
            # Linear regression
            slope, intercept, r_value, p_linreg, std_err = scipy_stats.linregress(alphas, rates)

            sig = ""
            if p_spearman < 0.001: sig = "***"
            elif p_spearman < 0.01: sig = "**"
            elif p_spearman < 0.05: sig = "*"

            print(f"    Layer {layer}: slope={slope:+.4f}(±{std_err:.4f})  R²={r_value**2:.3f}  "
                  f"Spearman ρ={rho:+.3f} (p={p_spearman:.4f}) {sig}")

            all_stat_rows.append({
                "metric": metric_name,
                "tier": tier,
                "layer": layer,
                "n_alphas": len(alphas),
                "slope": slope,
                "slope_stderr": std_err,
                "r_squared": r_value ** 2,
                "pearson_r": r,
                "pearson_p": p_pearson,
                "spearman_rho": rho,
                "spearman_p": p_spearman,
            })

        # Overall pooled across layers for this tier
        all_alphas = df_tier["alpha"].values
        all_rates = df_tier[metric_col].values
        if len(all_alphas) > 2:
            rho_all, p_all = scipy_stats.spearmanr(all_alphas, all_rates)
            slope_all, _, rv_all, _, se_all = scipy_stats.linregress(all_alphas, all_rates)
            print(f"    POOLED:   slope={slope_all:+.4f}(±{se_all:.4f})  R²={rv_all**2:.3f}  "
                  f"Spearman ρ={rho_all:+.3f} (p={p_all:.4f})")

stat_df = pd.DataFrame(all_stat_rows)
stats_path = out_dir / "alpha_effect_statistics.csv"
stat_df.to_csv(stats_path, index=False)
print(f"\n✅ Statistics saved to {stats_path}")

print(f"\n✅ All graphs saved to {out_dir}/")