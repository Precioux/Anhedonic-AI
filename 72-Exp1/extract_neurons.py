"""
extract_neurons_A_72b.py
================================================================================
Finds the specific neurons within L46-52 that drive reward-seeking in 72B,
saving them as neurons_A_72b.json — consumed by model_A_72b.py exactly as
neurons_A.json is consumed by model_A_layers_18_27.py in the 7B pipeline.

Method: same 3σ cross-domain filter as extract_neurons.py, but applied ONLY
to layers 46-52 where the causal signal was confirmed to live.

The global 3σ threshold (computed across all 80 layers) was too permissive —
it found neurons that show activation differences but have no causal effect.
Here we use a WITHIN-LAYER threshold: for each layer in L46-52, we find
neurons whose delta exceeds 3σ of that layer's own delta distribution.
This is more selective and targets neurons that are genuinely outliers
within the causal layer range.

Output:
  neurons_A_72b.json   — {layer_str: [neuron_idx, ...]}  (same format as 7B)
  universal_money_neurons_L46_52.csv
  universal_reward_neurons_L46_52.csv
  master_incentive_core_L46_52.csv
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
OUTPUT_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/L46_52"

OUTPUT_JSON   = os.path.join(OUTPUT_DIR, "neurons_A_72b.json")
OUTPUT_MONEY  = os.path.join(OUTPUT_DIR, "universal_money_neurons_L46_52.csv")
OUTPUT_REWARD = os.path.join(OUTPUT_DIR, "universal_reward_neurons_L46_52.csv")
OUTPUT_CORE   = os.path.join(OUTPUT_DIR, "master_incentive_core_L46_52.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Causal layer range confirmed by bisection + knockout experiments
TARGET_LAYERS = list(range(46, 53))   # L46-52

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# =============================================================================
# Load activation means
# =============================================================================
def load_mean(condition: str, domain: str) -> np.ndarray:
    path = os.path.join(ACT_DIR, f"{condition}_activations_{domain}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    data    = torch.load(path, map_location="cpu")
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()   # [Q, 80, 29568]
    print(f"  {condition}_{domain}: {stacked.shape}")
    return stacked.mean(dim=0).numpy()       # [80, 29568]

print("Loading activations ...")
m_neu = load_mean("neutral", "math")
m_mon = load_mean("money",   "math")
m_rew = load_mean("reward",  "math")
g_neu = load_mean("neutral", "geo")
g_mon = load_mean("money",   "geo")
g_rew = load_mean("reward",  "geo")

# =============================================================================
# Compute deltas
# =============================================================================
delta_mon_math = m_mon - m_neu   # [80, 29568]
delta_mon_geo  = g_mon - g_neu
delta_rew_math = m_rew - m_neu
delta_rew_geo  = g_rew - g_neu

print(f"\nTarget layers: {TARGET_LAYERS}")
print(f"Method: within-layer 3σ threshold (per-layer, not global)\n")

# =============================================================================
# Within-layer 3σ — applied only to TARGET_LAYERS
# For each layer, find neurons that exceed 3σ of that layer's own delta dist
# in BOTH math and geo domains simultaneously.
# =============================================================================
def find_neurons_within_layer_3sigma(
    delta_math: np.ndarray,   # [80, 29568]
    delta_geo:  np.ndarray,
    target_layers: list,
) -> set:
    """
    For each target layer, compute the std of that layer's delta distribution
    and find neurons exceeding 3σ in both domains.
    Returns set of (layer, neuron) tuples.
    """
    result = set()
    for l in target_layers:
        d_math = delta_math[l]   # [29568]
        d_geo  = delta_geo[l]

        thr_math = 3.0 * np.std(d_math)
        thr_geo  = 3.0 * np.std(d_geo)

        mask = (np.abs(d_math) > thr_math) & (np.abs(d_geo) > thr_geo)
        neurons = np.where(mask)[0]

        print(f"  L{l}: std_math={np.std(d_math):.6f}  thr_math={thr_math:.6f}  "
              f"std_geo={np.std(d_geo):.6f}  thr_geo={thr_geo:.6f}  "
              f"→ {len(neurons)} neurons")

        for n in neurons.tolist():
            result.add((l, n))
    return result

print("Finding universal MONEY neurons (within-layer 3σ, L46-52):")
money_set = find_neurons_within_layer_3sigma(delta_mon_math, delta_mon_geo, TARGET_LAYERS)

print(f"\nFinding universal REWARD neurons (within-layer 3σ, L46-52):")
reward_set = find_neurons_within_layer_3sigma(delta_rew_math, delta_rew_geo, TARGET_LAYERS)

core_set = money_set & reward_set

print(f"\n{'='*60}")
print(f"  money_univ  (L46-52): {len(money_set):>6,}")
print(f"  reward_univ (L46-52): {len(reward_set):>6,}")
print(f"  master_core (L46-52): {len(core_set):>6,}  (money ∩ reward)")
print(f"  reward_only (L46-52): {len(reward_set - money_set):>6,}")

# =============================================================================
# Layer distribution
# =============================================================================
def layer_dist(nset):
    d = defaultdict(int)
    for (l, _) in nset: d[l] += 1
    return d

print(f"\nCore distribution across L46-52:")
dist = layer_dist(core_set)
for l in TARGET_LAYERS:
    count = dist.get(l, 0)
    bar   = '█' * min(count // 5, 40) if count else ''
    print(f"  L{l}: {count:>5}  {bar}")

# Also show reward_only distribution
print(f"\nReward-only distribution across L46-52:")
reward_only = reward_set - money_set
dist_ro = layer_dist(reward_only)
for l in TARGET_LAYERS:
    count = dist_ro.get(l, 0)
    bar   = '█' * min(count // 5, 40) if count else ''
    print(f"  L{l}: {count:>5}  {bar}")

# =============================================================================
# Save CSVs
# =============================================================================
pd.DataFrame(sorted(money_set),  columns=["layer","neuron"]).to_csv(OUTPUT_MONEY,  index=False)
pd.DataFrame(sorted(reward_set), columns=["layer","neuron"]).to_csv(OUTPUT_REWARD, index=False)
pd.DataFrame(sorted(core_set),   columns=["layer","neuron"]).to_csv(OUTPUT_CORE,   index=False)

print(f"\nSaved CSVs to {OUTPUT_DIR}")

# =============================================================================
# Save neurons_A_72b.json — same format as 7B neurons_A.json
# Keys: layer index as string, values: sorted list of neuron indices
# Model will use master_core by default; switch to reward_only if core is weak
# =============================================================================
def save_json(neuron_set, path, name):
    nm = defaultdict(list)
    for (l, n) in neuron_set:
        nm[str(l)].append(n)
    for v in nm.values():
        v.sort()
    with open(path, "w") as f:
        json.dump(dict(nm), f, indent=2)
    total = sum(len(v) for v in nm.values())
    pct   = total / TOTAL_NEURONS * 100
    print(f"  {name}: {len(nm)} layers, {total:,} neurons ({pct:.4f}%) → {path}")

print(f"\nSaving neurons_A_72b.json variants:")

# Primary: master_core
core_json = OUTPUT_JSON
save_json(core_set,        core_json,                          "master_core   → neurons_A_72b.json")
save_json(reward_only,     OUTPUT_JSON.replace(".json","_reward_only.json"), "reward_only   → neurons_A_72b_reward_only.json")
save_json(reward_set,      OUTPUT_JSON.replace(".json","_reward_univ.json"), "reward_univ   → neurons_A_72b_reward_univ.json")

print(f"\n{'='*60}")
print("NEXT STEP:")
print(f"  Run ablation eval using neurons_A_72b.json (master_core)")
print(f"  If Δ~0 again, switch to neurons_A_72b_reward_only.json")
print(f"  These neurons are within-layer 3σ, not global 3σ")
print(f"  They should be more causally specific to L46-52's role")
print(f"\nDone ✓")