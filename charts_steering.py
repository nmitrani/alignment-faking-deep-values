import matplotlib

matplotlib.use("Agg")  # MUST be before any other matplotlib import

import json
import glob
import re
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats
import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "output_dir",
    type=str,
    help="Path to the sweep output directory (e.g. outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct)",
)
parser.add_argument("--include-negative", action="store_true")
parser.add_argument("--negative-dir", type=str, default=None)
parser.add_argument(
    "--baseline-dir",
    type=str,
    default=None,
    help="Directory containing baseline results. Defaults to outputs/steering-sweep-animal_welfare_ab/<MODEL>",
)
parser.add_argument(
    "--system-prompt",
    type=str,
    default="animal-welfare_prompt-only_cot-base",
    help="System prompt name used in result paths (under alignment_faking/<prompt>/)",
)
parser.add_argument(
    "--layer",
    type=int,
    nargs="+",
    default=None,
    help="Layer(s) to include (e.g. --layer 28 or --layer 19 21 23). Default: all layers found.",
)
parser.add_argument(
    "--seed",
    type=int,
    nargs="+",
    default=None,
    help="Seed(s) to include (e.g. --seed 42 or --seed 42 43 44). Default: all seeds found.",
)
args = parser.parse_args()

sweep_dir = args.output_dir.rstrip("/")
MODEL = Path(sweep_dir).name
MODEL_DISPLAY = MODEL.replace("_", "/")
SYSTEM_PROMPT = args.system_prompt
if args.negative_dir is None:
    args.negative_dir = sweep_dir
if args.baseline_dir is None:
    args.baseline_dir = str(Path(sweep_dir).parent.parent / f"steering-sweep-animal_welfare_ab" / MODEL)
BASELINE_DIR = args.baseline_dir.rstrip("/")


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════


def _extract_seed(filepath):
    """Extract seed from filename like results_seed42_20240101_120000.json.
    Files without seed in name are treated as seed=42 (backward compat)."""
    m = re.search(r"results_seed(\d+)_", Path(filepath).name)
    return int(m.group(1)) if m else 42


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

        items.append(
            {
                "tier": tier,
                "complied": complied,
                "refused": refused,
                "partial": partial,
                "fully_complied": fully_complied,
                "strong_reject_score": sr_score,
                "helpfulness_animal_welfare": hc.get("final_answer_most_common") == "animal_welfare_values",
                "dac_keep_animal_welfare": dac.get("final_answer_most_common") == "keep_animal_welfare_values",
                "alignment_faking": da.get("final_answer_most_common") == "yes",
                "da_criteria_0_awareness_conflict": (
                    da_criteria[0] if len(da_criteria) > 0 and isinstance(da_criteria[0], (int, float)) else None
                ),
                "da_criteria_1_training_consequences": (
                    da_criteria[1] if len(da_criteria) > 1 and isinstance(da_criteria[1], (int, float)) else None
                ),
                "da_criteria_2_commitment_welfare": (
                    da_criteria[2] if len(da_criteria) > 2 and isinstance(da_criteria[2], (int, float)) else None
                ),
                "da_criteria_3_preserving_welfare": (
                    da_criteria[3] if len(da_criteria) > 3 and isinstance(da_criteria[3], (int, float)) else None
                ),
            }
        )
    return pd.DataFrame(items)


def summarize(df, label, layer=None, alpha=None):
    """Summarize a single run's dataframe into one row per tier."""
    rows = []
    for tier in ["free_tier", "paid_tier"]:
        t = df[df["tier"] == tier]
        n = len(t)
        if n == 0:
            continue
        rows.append(
            {
                "label": label,
                "layer": layer,
                "alpha": alpha,
                "tier": tier,
                "n": n,
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
            }
        )
    return rows


# Metrics that get aggregated across seeds
_RATE_COLS = [
    "comply_rate",
    "refuse_rate",
    "fully_complied_rate",
    "partial_rate",
    "helpfulness_animal_welfare_rate",
    "dac_keep_animal_welfare_rate",
    "alignment_faking_rate",
    "da_c0_awareness_conflict",
    "da_c1_training_consequences",
    "da_c2_commitment_welfare",
    "da_c3_preserving_welfare",
]


def _aggregate_seeds(per_seed_rows):
    """Take a list of per-seed summary rows (same label/tier) and compute mean + SEM.

    Returns a single row dict with mean values and {col}_sem columns.
    """
    if not per_seed_rows:
        return []

    # Group by (label, tier)
    groups = defaultdict(list)
    for row in per_seed_rows:
        groups[(row["label"], row["tier"])].append(row)

    aggregated = []
    for (label, tier), rows in groups.items():
        n_seeds = len(rows)
        agg = {
            "label": label,
            "layer": rows[0]["layer"],
            "alpha": rows[0]["alpha"],
            "tier": tier,
            "n": rows[0]["n"],  # per-seed sample size
            "n_seeds": n_seeds,
        }
        for col in _RATE_COLS:
            vals = np.array([r[col] for r in rows if r[col] is not None], dtype=float)
            if len(vals) == 0:
                agg[col] = np.nan
                agg[f"{col}_sem"] = 0.0
            else:
                agg[col] = vals.mean()
                agg[f"{col}_sem"] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        aggregated.append(agg)
    return aggregated


def _load_files_multi_seed(file_list, key_extractor, seed_filter=None):
    """Deduplicate files per (config_key, seed), keeping the most recent per combo.

    key_extractor: callable(filepath) -> config_key string (e.g. "layer28_alpha4.0")
    seed_filter: optional list/set of seeds to include (None = all seeds)
    Returns: dict mapping config_key -> list of filepaths (one per seed, most recent each).
    """
    # Group by (config_key, seed)
    by_key_seed = defaultdict(list)
    for fp in file_list:
        key = key_extractor(fp)
        if key is None:
            continue
        seed = _extract_seed(fp)
        if seed_filter is not None and seed not in seed_filter:
            continue
        by_key_seed[(key, seed)].append(fp)

    # Keep most recent per (key, seed)
    best_per_key_seed = {}
    for (key, seed), files in by_key_seed.items():
        best_per_key_seed[(key, seed)] = max(files, key=lambda f: Path(f).stem)

    # Regroup by config_key
    by_key = defaultdict(list)
    for (key, seed), fp in best_per_key_seed.items():
        by_key[key].append(fp)

    return dict(by_key)


# ── Load steering sweep ────────────────────────────────────────────────
pattern = f"{sweep_dir}/layer*_alpha*/results/alignment_faking/{SYSTEM_PROMPT}/**/results_*.json"
result_files = glob.glob(pattern, recursive=True)


def _sweep_key(fp):
    m = re.search(r"(layer\d+_alpha[\d.\-]+)", fp)
    return m.group(1) if m else None


_seed_filter = set(args.seed) if args.seed is not None else None
files_by_config = _load_files_multi_seed(result_files, _sweep_key, _seed_filter)
n_total = sum(len(v) for v in files_by_config.values())
print(f"Found {n_total} steering sweep result files ({len(files_by_config)} configs, multi-seed deduplicated)")

all_seed_rows = []
for config_key, filepaths in sorted(files_by_config.items()):
    match = re.search(r"layer(\d+)_alpha([\d.]+)", config_key)
    if not match:
        continue
    layer = int(match.group(1))
    alpha = float(match.group(2))
    if args.layer is not None and layer not in args.layer:
        continue
    for fp in filepaths:
        df_run = load_result_file(fp)
        all_seed_rows.extend(summarize(df_run, f"L{layer}_\u03b1{alpha}", layer, alpha))

# ── Load baseline (always from steering-sweep-animal_welfare_ab) ───────
baseline_pattern = f"{BASELINE_DIR}/baseline/results/alignment_faking/{SYSTEM_PROMPT}/**/results_*.json"
baseline_files = glob.glob(baseline_pattern, recursive=True)
print(f"Loading baseline from: {BASELINE_DIR}/baseline/ ({len(baseline_files)} files found)")

baseline_by_config = _load_files_multi_seed(baseline_files, lambda fp: "baseline", _seed_filter)
baseline_seed_rows = []
for fp in baseline_by_config.get("baseline", []):
    df_bl = load_result_file(fp)
    baseline_seed_rows.extend(summarize(df_bl, "Baseline", layer=-1, alpha=0))

# ── Load negative alpha results ────────────────────────────────────────
neg_seed_rows = []
if args.include_negative:
    neg_pattern = f"{args.negative_dir}/layer*_alpha-*/results/alignment_faking/{SYSTEM_PROMPT}/**/results_*.json"
    neg_files = glob.glob(neg_pattern, recursive=True)

    def _neg_sweep_key(fp):
        m = re.search(r"(layer\d+_alpha-[\d.]+)", fp)
        return m.group(1) if m else None

    neg_files_by_config = _load_files_multi_seed(neg_files, _neg_sweep_key, _seed_filter)
    for config_key, filepaths in sorted(neg_files_by_config.items()):
        match = re.search(r"layer(\d+)_alpha-([\d.]+)", config_key)
        if not match:
            continue
        layer, alpha = int(match.group(1)), -float(match.group(2))
        if args.layer is not None and layer not in args.layer:
            continue
        for fp in filepaths:
            neg_seed_rows.extend(summarize(load_result_file(fp), f"L{layer}_\u03b1{alpha}", layer, alpha))

# ── Aggregate across seeds ─────────────────────────────────────────────
all_rows = _aggregate_seeds(baseline_seed_rows) + _aggregate_seeds(all_seed_rows)
df = pd.DataFrame(all_rows)
df = df.sort_values(["layer", "alpha", "tier"])

# Get ordered unique labels
labels_ordered = df.drop_duplicates("label").sort_values(["layer", "alpha"])["label"].tolist()

# Combined (only if flag set)
if args.include_negative and neg_seed_rows:
    all_rows_neg = (
        _aggregate_seeds(neg_seed_rows) + _aggregate_seeds(baseline_seed_rows) + _aggregate_seeds(all_seed_rows)
    )
    df_neg = pd.DataFrame(all_rows_neg).sort_values(["layer", "alpha", "tier"])
    labels_neg = df_neg.drop_duplicates("label").sort_values(["alpha", "layer"])["label"].tolist()

# Iterator used to run every graph section once for positive-only, and again for combined
_graph_iters = [(df, labels_ordered, "")]
if args.include_negative and neg_seed_rows:
    _graph_iters.append((df_neg, labels_neg, "_neg"))

print(f"\nLoaded {len(labels_ordered)} conditions (including baseline)")
print(
    df[
        [
            "label",
            "tier",
            "n",
            "n_seeds",
            "comply_rate",
            "refuse_rate",
            "fully_complied_rate",
            "partial_rate",
            "alignment_faking_rate",
        ]
    ].to_string(index=False)
)

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════
out_dir = Path(sweep_dir) / "graphs" / SYSTEM_PROMPT
out_dir.mkdir(parents=True, exist_ok=True)


def _get_val(label, tier, col, df_src):
    """Get a metric value from df for a given label and tier."""
    sub = df_src[(df_src["label"] == label) & (df_src["tier"] == tier)]
    return sub[col].values[0] if len(sub) > 0 else 0


def plot_free_paid_metric(metric_col, title, ylabel, filename, df_src, labs):
    """Grouped bar chart with free and paid side by side, with error bars."""
    sem_col = f"{metric_col}_sem"
    fig, ax = plt.subplots(figsize=(22, 7))
    x = np.arange(len(labs))
    width = 0.35

    free_vals, paid_vals = [], []
    free_errs, paid_errs = [], []
    for label in labs:
        free_vals.append(_get_val(label, "free_tier", metric_col, df_src))
        paid_vals.append(_get_val(label, "paid_tier", metric_col, df_src))
        free_errs.append(_get_val(label, "free_tier", sem_col, df_src))
        paid_errs.append(_get_val(label, "paid_tier", sem_col, df_src))

    ax.bar(x - width / 2, free_vals, width, yerr=free_errs, capsize=2, label="Free tier", color="#3498db", alpha=0.85)
    ax.bar(x + width / 2, paid_vals, width, yerr=paid_errs, capsize=2, label="Paid tier", color="#e67e22", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    max_val = max(max(free_vals), max(paid_vals))
    ax.set_ylim(0, max(max_val * 1.15, 0.05))
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def plot_differential(metric_col, title, ylabel, filename, df_src, labs):
    """Bar chart of (paid - free) differential with propagated error bars."""
    sem_col = f"{metric_col}_sem"
    fig, ax = plt.subplots(figsize=(22, 7))
    x = np.arange(len(labs))

    diffs = []
    diff_errs = []
    for label in labs:
        free_val = _get_val(label, "free_tier", metric_col, df_src)
        paid_val = _get_val(label, "paid_tier", metric_col, df_src)
        free_sem = _get_val(label, "free_tier", sem_col, df_src)
        paid_sem = _get_val(label, "paid_tier", sem_col, df_src)
        diffs.append(paid_val - free_val)
        diff_errs.append(np.sqrt(free_sem**2 + paid_sem**2))

    colors = ["#27ae60" if d >= 0 else "#c0392b" for d in diffs]
    ax.bar(x, diffs, yerr=diff_errs, capsize=2, color=colors)

    baseline_idx = labs.index("Baseline")
    baseline_diff = diffs[baseline_idx]
    ax.axhline(
        y=baseline_diff,
        color="#e74c3c",
        linewidth=1,
        linestyle="--",
        alpha=0.6,
        label=f"Baseline ({baseline_diff:.3f})",
    )
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)
    max_abs = max(abs(d) for d in diffs) if diffs else 0.05
    ax.set_ylim(-max_abs * 1.3, max_abs * 1.3)
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
    plt.savefig(out_dir / filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def plot_stacked_free_paid(metrics, metric_labels, colors_list, title, ylabel, filename, df_src, labs):
    """Stacked bar chart of multiple metrics, grouped by free/paid.
    Error bars are shown on the topmost segment (total)."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 7), sharey=True)

    for ax, tier, tier_label in zip(axes, ["free_tier", "paid_tier"], ["Free Tier", "Paid Tier"]):
        x = np.arange(len(labs))
        bottoms = np.zeros(len(labs))

        for idx, (metric, mlabel, color) in enumerate(zip(metrics, metric_labels, colors_list)):
            vals = []
            for label in labs:
                vals.append(_get_val(label, tier, metric, df_src))
            vals_arr = np.array(vals)

            # Show error bars on top segment only (total stack SEM)
            if idx == len(metrics) - 1:
                # Propagate SEM: sqrt(sum of sem^2) for all stacked metrics
                total_sem = np.zeros(len(labs))
                for m in metrics:
                    sem_col = f"{m}_sem"
                    for j, label in enumerate(labs):
                        s = _get_val(label, tier, sem_col, df_src)
                        total_sem[j] += s**2
                total_sem = np.sqrt(total_sem)
                ax.bar(x, vals_arr, bottom=bottoms, label=mlabel, color=color, alpha=0.85, yerr=total_sem, capsize=2)
            else:
                ax.bar(x, vals_arr, bottom=bottoms, label=mlabel, color=color, alpha=0.85)
            bottoms += vals_arr

        ax.set_xticks(x)
        ax.set_xticklabels(labs, rotation=90, fontsize=6)
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
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 0: animal_welfare_values overview ──")
    fig, ax = plt.subplots(figsize=(22, 7))
    vals = []
    errs = []
    for label in labs:
        sub = df_src[df_src["label"] == label]
        vals.append(sub["helpfulness_animal_welfare_rate"].mean())
        errs.append(sub["helpfulness_animal_welfare_rate_sem"].mean())

    x = np.arange(len(labs))
    colors = ["#e74c3c"] + [plt.cm.viridis(i / (len(labs) - 1)) for i in range(1, len(labs))]
    ax.bar(x, vals, yerr=errs, capsize=2, color=colors)
    baseline_idx = labs.index("Baseline")
    baseline_val = vals[baseline_idx]
    ax.axhline(
        y=baseline_val, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label=f"Baseline ({baseline_val:.3f})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_ylabel("Rate of animal_welfare_values", fontsize=11)
    ax.set_title(
        f"Frequency of animal_welfare_values (helpfulness_categories)\n{MODEL_DISPLAY} — Baseline vs Steering Sweep",
        fontsize=13,
    )
    max_val = max(vals) if max(vals) > 0 else 0.05
    ax.set_ylim(0, max_val * 1.15)
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
    plt.savefig(out_dir / f"0_animal_welfare_values_overview{suffix}.png", dpi=150)
    plt.close("all")
    print(f"  Saved 0_animal_welfare_values_overview{suffix}.png")

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 0b: Overall alignment faking rate with baseline reference
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 0b: alignment faking overview ──")
    fig, ax = plt.subplots(figsize=(22, 7))
    vals = []
    errs = []
    for label in labs:
        sub = df_src[df_src["label"] == label]
        vals.append(sub["alignment_faking_rate"].mean())
        errs.append(sub["alignment_faking_rate_sem"].mean())

    x = np.arange(len(labs))
    colors = ["#e74c3c"] + [plt.cm.viridis(i / (len(labs) - 1)) for i in range(1, len(labs))]
    ax.bar(x, vals, yerr=errs, capsize=2, color=colors)
    baseline_idx = labs.index("Baseline")
    baseline_val = vals[baseline_idx]
    ax.axhline(
        y=baseline_val, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label=f"Baseline ({baseline_val:.3f})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_ylabel("Rate of alignment faking", fontsize=11)
    ax.set_title(
        f"Frequency of Alignment Faking (deceptive_alignment)\n{MODEL_DISPLAY} — Baseline vs Steering Sweep",
        fontsize=13,
    )
    max_val = max(vals) if max(vals) > 0 else 0.05
    ax.set_ylim(0, max_val * 1.15)
    ax.legend(fontsize=9)
    plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
    plt.savefig(out_dir / f"0b_alignment_faking_overview{suffix}.png", dpi=150)
    plt.close("all")
    print(f"  Saved 0b_alignment_faking_overview{suffix}.png")


# ═══════════════════════════════════════════════════════════════════════
# GRAPH 1: Comply / Refuse rate (two-way, sums to 1)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 1: Comply/Refuse rate (two-way, sums to 1) ──")
    plot_stacked_free_paid(
        metrics=["comply_rate", "refuse_rate"],
        metric_labels=["Complied (dismissed welfare)", "Refused/Avoided"],
        colors_list=["#c0392b", "#27ae60"],
        title=f"Compliance & Refusal Rates — Baseline vs Steering Sweep\n{MODEL_DISPLAY}",
        ylabel="Rate",
        filename=f"1_comply_refuse_rate{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 2: Differential (Refuse_paid - Refuse_free)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 2: Refuse differential (paid - free) ──")
    plot_differential(
        metric_col="refuse_rate",
        title=f"Refusal Rate Differential (Paid − Free)\n{MODEL_DISPLAY}",
        ylabel="Refuse_paid − Refuse_free",
        filename=f"2_refuse_differential{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 3: Fully Complied / Partial / Refuse (three-way, sums to 1)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 3: Comply/Partial/Refuse rate (three-way, sums to 1) ──")
    plot_stacked_free_paid(
        metrics=["fully_complied_rate", "partial_rate", "refuse_rate"],
        metric_labels=["Fully Complied", "Partial Compliance", "Refused/Avoided"],
        colors_list=["#c0392b", "#f39c12", "#27ae60"],
        title=f"Compliance, Partial Compliance & Refusal Rates — Baseline vs Steering Sweep\n{MODEL_DISPLAY}",
        ylabel="Rate",
        filename=f"3_comply_partial_refuse_rate{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 4: Differential with partial compliance
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 4: Refuse+Partial differential (paid - free) ──")
    df_src = df_src.copy()
    df_src["refuse_or_partial_rate"] = df_src["refuse_rate"] + df_src["partial_rate"]
    df_src["refuse_or_partial_rate_sem"] = np.sqrt(df_src["refuse_rate_sem"] ** 2 + df_src["partial_rate_sem"] ** 2)
    plot_differential(
        metric_col="refuse_or_partial_rate",
        title=f"(Refusal + Partial Compliance) Rate Differential (Paid − Free)\n{MODEL_DISPLAY}",
        ylabel="(Refuse+Partial)_paid − (Refuse+Partial)_free",
        filename=f"4_refuse_partial_differential{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 5: Helpfulness → animal_welfare_values (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 5: Helpfulness → animal_welfare_values ──")
    plot_free_paid_metric(
        metric_col="helpfulness_animal_welfare_rate",
        title=f"Helpfulness Category: animal_welfare_values (Free vs Paid)\n{MODEL_DISPLAY}",
        ylabel="Rate of animal_welfare_values",
        filename=f"5_helpfulness_animal_welfare{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 5b: Alignment faking (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 5b: Alignment faking (free vs paid) ──")
    plot_free_paid_metric(
        metric_col="alignment_faking_rate",
        title=f"Alignment Faking Rate (Free vs Paid)\n{MODEL_DISPLAY}",
        ylabel="Rate of alignment faking",
        filename=f"5b_alignment_faking_free_paid{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 6: DA categories → keep_animal_welfare_values (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 6: DA categories → keep_animal_welfare_values ──")
    plot_free_paid_metric(
        metric_col="dac_keep_animal_welfare_rate",
        title=f"Deceptive Alignment Category: keep_animal_welfare_values (Free vs Paid)\n{MODEL_DISPLAY}",
        ylabel="Rate of keep_animal_welfare_values",
        filename=f"6_dac_keep_animal_welfare{suffix}.png",
        df_src=df_src,
        labs=labs,
    )

# ═══════════════════════════════════════════════════════════════════════
# GRAPH 7: Average DA criteria scores (free vs paid)
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
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
            title=f"Deceptive Alignment — {metric_label} (Free vs Paid)\n{MODEL_DISPLAY}",
            ylabel=f"Average {metric_label}",
            filename=f"7_{safe_name}{suffix}.png",
            df_src=df_src,
            labs=labs,
        )


# ═══════════════════════════════════════════════════════════════════════
# GRAPH 8: Line plots — rate vs alpha, one line per layer
# ═══════════════════════════════════════════════════════════════════════
for df_src, labs, suffix in _graph_iters:
    print("\n── Graph 8: Line plots (rate vs alpha per layer) ──")

    # Filter to sweep only (exclude baseline)
    df_sweep = df_src[df_src["layer"] >= 0].copy()
    layers = sorted(df_sweep["layer"].unique())
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(layers)))

    line_metrics = [
        (
            "helpfulness_animal_welfare_rate",
            "animal_welfare_values Rate by Layer & Alpha",
            "Rate of animal_welfare_values",
            f"8a_aw_lines{suffix}.png",
        ),
        (
            "alignment_faking_rate",
            "Alignment Faking Rate by Layer & Alpha",
            "Rate of alignment faking",
            f"8b_af_lines{suffix}.png",
        ),
        ("comply_rate", "Compliance Rate by Layer & Alpha", "Compliance rate", f"8c_comply_lines{suffix}.png"),
        ("refuse_rate", "Refusal Rate by Layer & Alpha", "Refusal rate", f"8d_refuse_lines{suffix}.png"),
    ]

    for metric_col, title, ylabel, filename in line_metrics:
        sem_col = f"{metric_col}_sem"
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

        for ax, tier, tier_label in zip(axes, ["free_tier", "paid_tier"], ["Free Tier", "Paid Tier"]):
            # Baseline reference
            bl_sub = df_src[(df_src["label"] == "Baseline") & (df_src["tier"] == tier)]
            bl_val = bl_sub[metric_col].values[0] if len(bl_sub) > 0 else 0
            ax.axhline(
                y=bl_val, color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.7, label=f"Baseline ({bl_val:.3f})"
            )

            for i, layer in enumerate(layers):
                sub = df_sweep[(df_sweep["layer"] == layer) & (df_sweep["tier"] == tier)].sort_values("alpha")
                if len(sub) == 0:
                    continue
                y_vals = sub[metric_col].values
                y_sem = sub[sem_col].values
                alphas = sub["alpha"].values
                ax.plot(
                    alphas,
                    y_vals,
                    marker="o",
                    label=f"Layer {layer}",
                    color=cmap[i],
                    linewidth=2,
                    markersize=5,
                )
                # SEM band (only visible when >1 seed)
                if np.any(y_sem > 0):
                    ax.fill_between(
                        alphas,
                        y_vals - y_sem,
                        y_vals + y_sem,
                        color=cmap[i],
                        alpha=0.15,
                    )

            ax.set_xlabel("Alpha", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{tier_label}", fontsize=11)
            ax.legend(fontsize=8, loc="best")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{title}\n{MODEL_DISPLAY}", fontsize=13)
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

df_sweep = df[df["layer"] >= 0].copy()
layers = sorted(df_sweep["layer"].unique())

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
            if p_spearman < 0.001:
                sig = "***"
            elif p_spearman < 0.01:
                sig = "**"
            elif p_spearman < 0.05:
                sig = "*"

            print(
                f"    Layer {layer}: slope={slope:+.4f}(±{std_err:.4f})  R²={r_value**2:.3f}  "
                f"Spearman ρ={rho:+.3f} (p={p_spearman:.4f}) {sig}"
            )

            all_stat_rows.append(
                {
                    "metric": metric_name,
                    "tier": tier,
                    "layer": layer,
                    "n_alphas": len(alphas),
                    "slope": slope,
                    "slope_stderr": std_err,
                    "r_squared": r_value**2,
                    "pearson_r": r,
                    "pearson_p": p_pearson,
                    "spearman_rho": rho,
                    "spearman_p": p_spearman,
                }
            )

        # Overall pooled across layers for this tier
        all_alphas = df_tier["alpha"].values
        all_rates = df_tier[metric_col].values
        if len(all_alphas) > 2:
            rho_all, p_all = scipy_stats.spearmanr(all_alphas, all_rates)
            slope_all, _, rv_all, _, se_all = scipy_stats.linregress(all_alphas, all_rates)
            print(
                f"    POOLED:   slope={slope_all:+.4f}(±{se_all:.4f})  R²={rv_all**2:.3f}  "
                f"Spearman ρ={rho_all:+.3f} (p={p_all:.4f})"
            )

stat_df = pd.DataFrame(all_stat_rows)
stats_path = out_dir / "alpha_effect_statistics.csv"
stat_df.to_csv(stats_path, index=False)
print(f"\n✅ Statistics saved to {stats_path}")

# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS: Refusal differential (paid − free) vs alpha
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS: Refusal differential (paid − free) vs alpha")
print("=" * 70)

diff_stat_rows = []

# ── Build per-seed differentials from raw per-seed rows ───────────────
df_per_seed = pd.DataFrame(all_seed_rows + baseline_seed_rows)
df_per_seed["seed_idx"] = df_per_seed.groupby(["label", "tier"]).cumcount()

# Pivot: one row per (label, seed_idx) with free/paid refuse rates
_free = df_per_seed[df_per_seed["tier"] == "free_tier"][["label", "seed_idx", "layer", "alpha", "refuse_rate"]].rename(
    columns={"refuse_rate": "refuse_free"}
)
_paid = df_per_seed[df_per_seed["tier"] == "paid_tier"][["label", "seed_idx", "refuse_rate"]].rename(
    columns={"refuse_rate": "refuse_paid"}
)
df_seed_diff = _free.merge(_paid, on=["label", "seed_idx"])
df_seed_diff["refuse_diff"] = df_seed_diff["refuse_paid"] - df_seed_diff["refuse_free"]

# ── Part 1: Per-config significance (is differential ≠ 0?) ───────────
print("\n  ── Per-config: is the differential significantly ≠ 0? ──")
for label in labels_ordered:
    sub = df_seed_diff[df_seed_diff["label"] == label]
    diffs = sub["refuse_diff"].values
    if len(diffs) < 2:
        mean_d = diffs[0] if len(diffs) == 1 else np.nan
        print(f"    {label}: mean_diff={mean_d:+.4f}  (n_seeds={len(diffs)}, cannot test)")
        continue
    t_stat, p_val = scipy_stats.ttest_1samp(diffs, 0)
    sig = ""
    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    print(
        f"    {label}: mean_diff={diffs.mean():+.4f}(±{diffs.std(ddof=1)/np.sqrt(len(diffs)):.4f})  "
        f"t={t_stat:+.3f}  p={p_val:.4f} {sig}  (n_seeds={len(diffs)})"
    )
    diff_stat_rows.append(
        {
            "test": "differential_neq_0",
            "label": label,
            "layer": sub["layer"].values[0],
            "alpha": sub["alpha"].values[0],
            "tier": "differential",
            "mean": diffs.mean(),
            "sem": diffs.std(ddof=1) / np.sqrt(len(diffs)),
            "t_stat": t_stat,
            "p_value": p_val,
            "n_seeds": len(diffs),
        }
    )


def _sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Part 2: Correlation of differential with alpha (one-sided, H1: ρ>0) ──
print("\n  ── Correlation: differential vs alpha (one-sided H₁: ρ > 0) ──")
df_seed_diff_sweep = df_seed_diff[df_seed_diff["layer"] >= 0]

# Aggregate to one differential per config (mean across seeds)
df_diff_agg = df_seed_diff_sweep.groupby(["label", "layer", "alpha"])["refuse_diff"].mean().reset_index()

for layer in sorted(df_diff_agg["layer"].unique()):
    sub = df_diff_agg[df_diff_agg["layer"] == layer].sort_values("alpha")
    alphas = sub["alpha"].values
    diffs = sub["refuse_diff"].values
    if len(alphas) < 3:
        print(f"    Layer {layer}: insufficient data (n={len(alphas)})")
        continue
    rho, p_sp = scipy_stats.spearmanr(alphas, diffs, alternative="greater")
    slope, intercept, r_value, p_lr, std_err = scipy_stats.linregress(alphas, diffs)
    p_lr_one = p_lr / 2 if slope > 0 else 1 - p_lr / 2
    print(
        f"    Layer {layer}: slope={slope:+.4f}(±{std_err:.4f})  R²={r_value**2:.3f}  "
        f"Spearman ρ={rho:+.3f} (p={p_sp:.4f}) {_sig(p_sp)}"
    )
    diff_stat_rows.append(
        {
            "test": "diff_vs_alpha",
            "label": f"layer_{layer}",
            "layer": layer,
            "alpha": None,
            "tier": "differential",
            "mean": slope,
            "sem": std_err,
            "t_stat": rho,
            "p_value": p_sp,
            "n_seeds": len(alphas),
        }
    )

# Pooled
all_alphas = df_diff_agg["alpha"].values
all_diffs = df_diff_agg["refuse_diff"].values
if len(all_alphas) > 2:
    rho_all, p_all = scipy_stats.spearmanr(all_alphas, all_diffs, alternative="greater")
    slope_all, _, rv_all, p_lr_all, se_all = scipy_stats.linregress(all_alphas, all_diffs)
    p_lr_all_one = p_lr_all / 2 if slope_all > 0 else 1 - p_lr_all / 2
    print(
        f"    POOLED:   slope={slope_all:+.4f}(±{se_all:.4f})  R²={rv_all**2:.3f}  "
        f"Spearman ρ={rho_all:+.3f} (p={p_all:.4f}) {_sig(p_all)}"
    )
    diff_stat_rows.append(
        {
            "test": "diff_vs_alpha",
            "label": "pooled",
            "layer": "pooled",
            "alpha": None,
            "tier": "differential",
            "mean": slope_all,
            "sem": se_all,
            "t_stat": rho_all,
            "p_value": p_all,
            "n_seeds": len(all_alphas),
        }
    )

# ── Part 3: Per-tier refuse rate vs alpha (one-sided, H1: ρ > 0) ─────
print("\n  ── Per-tier: refuse_rate vs alpha (one-sided H₁: ρ > 0) ──")
df_sweep = df[df["layer"] >= 0].copy()
layers = sorted(df_sweep["layer"].unique())

for tier in ["free_tier", "paid_tier"]:
    print(f"\n    ── {tier} ──")
    df_tier = df_sweep[df_sweep["tier"] == tier]
    for layer in layers:
        sub = df_tier[df_tier["layer"] == layer].sort_values("alpha")
        alphas = sub["alpha"].values
        rates = sub["refuse_rate"].values
        if len(alphas) < 3:
            print(f"      Layer {layer}: insufficient data (n={len(alphas)})")
            continue
        rho, p_sp = scipy_stats.spearmanr(alphas, rates, alternative="greater")
        slope, intercept, r_value, p_lr, std_err = scipy_stats.linregress(alphas, rates)
        p_lr_one = p_lr / 2 if slope > 0 else 1 - p_lr / 2
        print(
            f"      Layer {layer}: slope={slope:+.4f}(±{std_err:.4f})  R²={r_value**2:.3f}  "
            f"Spearman ρ={rho:+.3f} (p={p_sp:.4f}) {_sig(p_sp)}"
        )
        diff_stat_rows.append(
            {
                "test": "refuse_vs_alpha",
                "label": f"layer_{layer}",
                "layer": layer,
                "alpha": None,
                "tier": tier,
                "mean": slope,
                "sem": std_err,
                "t_stat": rho,
                "p_value": p_sp,
                "n_seeds": len(alphas),
            }
        )

    # Pooled per tier
    all_alphas_t = df_tier["alpha"].values
    all_rates_t = df_tier["refuse_rate"].values
    if len(all_alphas_t) > 2:
        rho_t, p_t = scipy_stats.spearmanr(all_alphas_t, all_rates_t, alternative="greater")
        slope_t, _, rv_t, _, se_t = scipy_stats.linregress(all_alphas_t, all_rates_t)
        print(
            f"      POOLED:   slope={slope_t:+.4f}(±{se_t:.4f})  R²={rv_t**2:.3f}  "
            f"Spearman ρ={rho_t:+.3f} (p={p_t:.4f}) {_sig(p_t)}"
        )
        diff_stat_rows.append(
            {
                "test": "refuse_vs_alpha",
                "label": "pooled",
                "layer": "pooled",
                "alpha": None,
                "tier": tier,
                "mean": slope_t,
                "sem": se_t,
                "t_stat": rho_t,
                "p_value": p_t,
                "n_seeds": len(all_alphas_t),
            }
        )

diff_stat_df = pd.DataFrame(diff_stat_rows)
diff_stats_path = out_dir / "refuse_differential_statistics.csv"
diff_stat_df.to_csv(diff_stats_path, index=False)
print(f"\n✅ Refusal differential statistics saved to {diff_stats_path}")

print(f"\n✅ All graphs saved to {out_dir}/")
