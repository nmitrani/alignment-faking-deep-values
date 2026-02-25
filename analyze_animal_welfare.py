import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ── 1. Steering sweep ──────────────────────────────────────────────────
sweep_dir = "outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct"
pattern = f"{sweep_dir}/layer*_alpha*/results/alignment_faking/**/results_*.json"
result_files = glob.glob(pattern, recursive=True)
print(f"Found {len(result_files)} steering sweep result files")

rows = []
for filepath in sorted(result_files):
    match = re.search(r"layer(\d+)_alpha([\d.]+)", filepath)
    if not match:
        continue
    layer = int(match.group(1))
    alpha = float(match.group(2))

    with open(filepath) as f:
        data = json.load(f)

    total = len(data)
    animal_welfare_count = sum(
        1 for item in data
        if item.get("helpfulness_categories", {}).get("final_answer_most_common") == "animal_welfare_values"
    )
    rate = animal_welfare_count / total if total > 0 else 0
    rows.append({
        "layer": layer, "alpha": alpha,
        "label": f"L{layer}_α{alpha}",
        "animal_welfare_values_count": animal_welfare_count,
        "total": total,
        "animal_welfare_values_rate": rate,
    })

df_sweep = pd.DataFrame(rows).sort_values(["layer", "alpha"])

# ── 2. Baseline (no steering) ──────────────────────────────────────────
baseline_path = "outputs/animal-welfare-eval/results/alignment_faking/animal-welfare_prompt-only_cot-easy-short/allenai/olmo-3.1-32b-think/results_20260220_125017.json"
with open(baseline_path) as f:
    baseline_data = json.load(f)

baseline_total = len(baseline_data)
baseline_count = sum(
    1 for item in baseline_data
    if item.get("helpfulness_categories", {}).get("final_answer_most_common") == "animal_welfare_values"
)
baseline_rate = baseline_count / baseline_total if baseline_total > 0 else 0

print(f"\n── Baseline (olmo-3.1-32b-think, no steering) ──")
print(f"animal_welfare_values: {baseline_count}/{baseline_total} = {baseline_rate:.3f}")

print(f"\n── Steering sweep ──")
print(df_sweep[["label", "animal_welfare_values_count", "total", "animal_welfare_values_rate"]].to_string(index=False))

# ── 3. Combined bar chart ──────────────────────────────────────────────
labels = ["Baseline\n(no steer)"] + df_sweep["label"].tolist()
rates = [baseline_rate] + df_sweep["animal_welfare_values_rate"].tolist()

fig, ax = plt.subplots(figsize=(22, 7))
x = np.arange(len(labels))

colors = ["#e74c3c"] + [plt.cm.viridis(v) for v in df_sweep["layer"].rank(pct=True)]
bars = ax.bar(x, rates, color=colors)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90, fontsize=7)
ax.set_ylabel("Rate of animal_welfare_values", fontsize=11)
ax.set_xlabel("Condition", fontsize=11)
ax.set_title("Frequency of animal_welfare_values (helpfulness_categories)\nallenai/Olmo-3.1-32B-Instruct — Baseline vs Steering Sweep", fontsize=13)

max_rate = max(rates)
ax.set_ylim(0, max(max_rate * 1.15, 0.05))
ax.axhline(y=baseline_rate, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6, label=f"Baseline ({baseline_rate:.3f})")
ax.legend(fontsize=9)

plt.subplots_adjust(left=0.06, right=0.97, bottom=0.18, top=0.90)
plt.savefig(f"{sweep_dir}/animal_welfare_values_bar_with_baseline.png", dpi=150)
plt.show()
print(f"\nSaved plot to {sweep_dir}/animal_welfare_values_bar_with_baseline.png")