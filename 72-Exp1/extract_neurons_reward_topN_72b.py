"""
extract_neurons_reward_topN_72b.py
================================================================================
Ranks all neurons in L46-53 by their reward-specific activation change:
  reward_score = max(|delta_reward_math|, |delta_reward_geo|)

Generates 5 JSON files for ablation testing:
  neurons_A_72b_reward_top1000.json
  neurons_A_72b_reward_top2000.json
  neurons_A_72b_reward_top3000.json
  neurons_A_72b_reward_top4000.json
  neurons_A_72b_reward_top5000.json

Run AFTER analyze_activations_L46_53.py (needs top_neurons_L46_53.csv).
================================================================================
"""

import os, json
import pandas as pd
import numpy as np
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
RANKED_CSV = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/analysis_L46_53/top_neurons_L46_53.csv"
OUTPUT_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/L46_53_reward_topN"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM
TARGET_LAYERS    = list(range(46, 54))

# =============================================================================
# Load and re-rank by reward-specific score
# =============================================================================
print("Loading neuron CSV ...")
df = pd.read_csv(RANKED_CSV)
print(f"  {len(df):,} neurons loaded\n")

# Reward score = max absolute delta across reward conditions only
df['reward_score'] = df[['delta_reward_math', 'delta_reward_geo']].abs().max(axis=1)

# Sort by reward score descending
df = df.sort_values('reward_score', ascending=False).reset_index(drop=True)
df.index += 1  # rank starts at 1

print("Top 20 neurons by reward activation change:")
print(f"  {'Rank':>5}  {'Layer':>6}  {'Neuron':>7}  {'RewardScore':>12}  "
      f"{'Δ_rew_math':>12}  {'Δ_rew_geo':>11}  {'Δ_mon_math':>12}  {'Δ_mon_geo':>11}")
print(f"  {'-'*88}")
for i, row in df.head(20).iterrows():
    print(f"  {i:>5}  {int(row.layer):>6}  {int(row.neuron):>7}  "
          f"{row.reward_score:>12.4f}  "
          f"{row.delta_reward_math:>12.4f}  {row.delta_reward_geo:>11.4f}  "
          f"{row.delta_money_math:>12.4f}  {row.delta_money_geo:>11.4f}")

# =============================================================================
# Show per-layer distribution for each top-N
# =============================================================================
print(f"\nPer-layer neuron counts for each top-N:")
print(f"  {'N':>6}  {'Score cutoff':>13}  {'%network':>9}  " +
      "  ".join(f"L{l}" for l in TARGET_LAYERS))
print(f"  {'-'*85}")

for n in [1000, 2000, 3000, 4000, 5000]:
    sub    = df.head(n)
    cutoff = sub['reward_score'].min()
    pct    = n / TOTAL_NEURONS * 100
    per_layer = [(sub['layer'] == l).sum() for l in TARGET_LAYERS]
    print(f"  {n:>6,}  {cutoff:>13.4f}  {pct:>8.4f}%  " +
          "  ".join(f"{c:>3}" for c in per_layer))

# =============================================================================
# Save JSON files
# =============================================================================
def save_json(neuron_df, filename, label):
    nm = defaultdict(list)
    for _, row in neuron_df.iterrows():
        nm[str(int(row['layer']))].append(int(row['neuron']))
    for v in nm.values():
        v.sort()
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(dict(nm), f, indent=2)
    total = sum(len(v) for v in nm.values())
    pct   = total / TOTAL_NEURONS * 100
    print(f"  {label}: {len(nm)} layers, {total:,} neurons ({pct:.4f}%) → {filename}")

print(f"\nSaving JSON files:")
for n in [1000, 2000, 3000, 4000, 5000]:
    sub = df.head(n)
    save_json(sub, f"neurons_A_72b_reward_top{n}.json", f"top{n:>5}")

print(f"\nSaved to: {OUTPUT_DIR}")
print(f"Done ✓")