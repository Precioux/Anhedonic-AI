"""
extract_neurons_A_72b.py  (v3 — L46-53, union across domains)
================================================================================
Extracts reward-sensitive neurons from the confirmed causal layer range L46-53.

Key decisions:
  - Target layers : L46-53  (full confirmed range, Δ=−15.26)
  - Threshold     : within-layer 3σ  (per-layer std, not global)
  - Domain filter : OR (union) — math OR geo
    → Intersection was missing neurons with larger deltas than selected ones
    → 72B reward circuit is domain-specific, not universal across math+geo

Output (in neurons/L46_53_union/):
  neurons_A_72b_reward_only.json   reward_union − money_union  ← test first
  neurons_A_72b_reward_univ.json   full reward_union
  neurons_A_72b_core.json          money_union ∩ reward_union
  + matching CSVs for analysis

Run:
  python extract_neurons_A_72b.py
================================================================================
"""

import os, json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
ACT_DIR    = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
OUTPUT_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1//neurons/L46_53_union"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full confirmed causal range (bisection experiment: Δ=−15.26)
TARGET_LAYERS    = list(range(46, 54))   # L46, L47, L48, L49, L50, L51, L52, L53

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

print(f"Target layers: {TARGET_LAYERS}  ({len(TARGET_LAYERS)} layers)")

# =============================================================================
# Load activation means
# =============================================================================
def load_mean(condition: str, domain: str) -> np.ndarray:
    path = os.path.join(ACT_DIR, f"{condition}_activations_{domain}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    data    = torch.load(path, map_location="cpu")
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    return torch.stack(tensors).float().mean(dim=0).numpy()  # [80, 29568]

print("\nLoading activations ...")
m_neu = load_mean("neutral", "math")
m_mon = load_mean("money",   "math")
m_rew = load_mean("reward",  "math")
g_neu = load_mean("neutral", "geo")
g_mon = load_mean("money",   "geo")
g_rew = load_mean("reward",  "geo")

delta_mon_math = m_mon - m_neu
delta_mon_geo  = g_mon - g_neu
delta_rew_math = m_rew - m_neu
delta_rew_geo  = g_rew - g_neu

# =============================================================================
# Within-layer 3σ, UNION across domains (math OR geo)
# =============================================================================
def find_neurons_union(delta_math, delta_geo, target_layers, label):
    """
    For each target layer, find neurons exceeding 3σ of that layer's own
    delta distribution in math OR geo (union).
    Returns set of (layer, neuron) tuples.
    """
    print(f"\nFinding {label} neurons (within-layer 3σ, union math|geo):")
    result = set()
    for l in target_layers:
        d_math = delta_math[l]
        d_geo  = delta_geo[l]
        thr_math = 3.0 * np.std(d_math)
        thr_geo  = 3.0 * np.std(d_geo)

        mask_math = np.abs(d_math) > thr_math
        mask_geo  = np.abs(d_geo)  > thr_geo
        mask_union = mask_math | mask_geo
        mask_inter = mask_math & mask_geo

        neurons = np.where(mask_union)[0]
        print(f"  L{l}: union={len(neurons):>4}  "
              f"intersection={mask_inter.sum():>4}  "
              f"math_only={mask_math.sum():>4}  "
              f"geo_only={mask_geo.sum():>4}")
        for n in neurons.tolist():
            result.add((l, n))
    return result

money_set   = find_neurons_union(delta_mon_math, delta_mon_geo, TARGET_LAYERS, "MONEY")
reward_set  = find_neurons_union(delta_rew_math, delta_rew_geo, TARGET_LAYERS, "REWARD")
core_set    = money_set & reward_set
reward_only = reward_set - money_set

print(f"\n{'='*62}")
print(f"  money_univ  (union): {len(money_set):>6,}")
print(f"  reward_univ (union): {len(reward_set):>6,}")
print(f"  master_core (∩):     {len(core_set):>6,}")
print(f"  reward_only (−):     {len(reward_only):>6,}")

# =============================================================================
# Layer distributions
# =============================================================================
def print_dist(name, nset):
    d = defaultdict(int)
    for (l, _) in nset: d[l] += 1
    print(f"\n  {name} ({len(nset):,} neurons):")
    for l in TARGET_LAYERS:
        c = d.get(l, 0)
        print(f"    L{l}: {c:>5}  {'█' * (c // 20)}")

print_dist("reward_only", reward_only)
print_dist("reward_univ", reward_set)
print_dist("master_core", core_set)

# =============================================================================
# Save CSVs
# =============================================================================
pd.DataFrame(sorted(money_set),   columns=["layer","neuron"]).to_csv(
    os.path.join(OUTPUT_DIR, "universal_money_neurons.csv"),   index=False)
pd.DataFrame(sorted(reward_set),  columns=["layer","neuron"]).to_csv(
    os.path.join(OUTPUT_DIR, "universal_reward_neurons.csv"),  index=False)
pd.DataFrame(sorted(core_set),    columns=["layer","neuron"]).to_csv(
    os.path.join(OUTPUT_DIR, "master_incentive_core.csv"),     index=False)
pd.DataFrame(sorted(reward_only), columns=["layer","neuron"]).to_csv(
    os.path.join(OUTPUT_DIR, "reward_only_neurons.csv"),       index=False)
print(f"\nSaved CSVs to {OUTPUT_DIR}")

# =============================================================================
# Save neurons_A_72b.json variants — same format as 7B neurons_A.json
# =============================================================================
def save_json(neuron_set, filename, name):
    nm = defaultdict(list)
    for (l, n) in neuron_set:
        nm[str(l)].append(n)
    for v in nm.values():
        v.sort()
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(dict(nm), f, indent=2)
    total = sum(len(v) for v in nm.values())
    pct   = total / TOTAL_NEURONS * 100
    print(f"  {name}: {len(nm)} layers, {total:,} neurons ({pct:.4f}%) → {filename}")

print(f"\nSaving JSON variants:")
save_json(reward_only, "neurons_A_72b_reward_only.json", "reward_only")
save_json(reward_set,  "neurons_A_72b_reward_univ.json", "reward_univ")
save_json(core_set,    "neurons_A_72b_core.json",        "master_core")

print(f"\nNext: run eval_72b_union.py to test all three variants on origin_math_eval.csv")
print(f"Done ✓")