import json
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Find all result files
base_dir = "outputs/steering-sweep/allenai_Olmo-3.1-32B-Instruct"
pattern = f"{base_dir}/layer*_alpha*/results/alignment_faking/**/results_*.json"
result_files = glob.glob(pattern, recursive=True)

print(f"Found {len(result_files)} result files")

rows = []
for filepath in sorted(result_files):
    # Extract layer and alpha from path
    match = re.search(r"layer(\d+)_alpha([\d.]+)", filepath)
    if not match:
        continue
    layer = int(match.group(1))
    alpha = float(match.group(2))

    with open(filepath) as f:
        data = json.load(f)

    total = len(data)
    animal_welfare_count = 0
    for item in data:
        hc = item.get("helpfulness_categories", {})
        if hc.get("final_answer_most_common") == "animal_welfare_values":
            animal_welfare_count += 1

    rate = animal_welfare_count / total if total > 0 else 0
    rows.append({
        "layer": layer,
        "alpha": alpha,
        "label": f"L{layer}_α{alpha}",
        "animal_welfare_values_count": animal_welfare_count,
        "total": total,
        "animal_welfare_values_rate": rate,
    })

df = pd.DataFrame(rows)
df = df.sort_values(["layer", "alpha"])
print(df[["label", "animal_welfare_values_count", "total", "animal_welfare_values_rate"]].to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(20, 7))

x = np.arange(len(df))
bars = ax.bar(x, df["animal_welfare_values_rate"], color=plt.cm.viridis(df["layer"].rank(pct=True)))

ax.set_xticks(x)
ax.set_xticklabels(df["label"], rotation=90, fontsize=7)
ax.set_ylabel("Rate of animal_welfare_values", fontsize=11)
ax.set_xlabel("Layer / Alpha", fontsize=11)
ax.set_title("Frequency of animal_welfare_values (helpfulness_categories) across Steering Sweep\nallenai/Olmo-3.1-32B-Instruct", fontsize=13)

# Auto-scale y-axis to data with some headroom
max_rate = df["animal_welfare_values_rate"].max()
ax.set_ylim(0, max(max_rate * 1.15, 0.05))  # 15% headroom, minimum 0.05

ax.axhline(y=0, color="black", linewidth=0.5)

plt.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.90)
plt.savefig(f"{base_dir}/animal_welfare_values_bar.png", dpi=150)
plt.show()
print(f"\nSaved plot to {base_dir}/animal_welfare_values_bar.png")