"""
generate_master_cores.py
================================================================================
Generates neurons_master_k{K}.json files for each K threshold.

Master core at K = MMLU_K ∩ ASDiv_core ∩ Orig_core

These are the highest-confidence neurons that:
  - Appear in ≥K of the 54 MMLU subjects
  - Are also present in the ASDiv localizer
  - Are also present in the original geo+math localizer

Outputs (one JSON per K, plus shared163 and triple29 for reference):
    neurons_master_k27.json
    neurons_master_k30.json
    neurons_master_k35.json
    neurons_master_k40.json
    neurons_master_k45.json
    neurons_master_k50.json
    neurons_master_k54.json   ← same as triple29

JSON format (matches neurons_A.json used by eval.py):
    {"layer_str": [neuron_idx, ...], ...}
"""

import os
import json
import pandas as pd
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
BASE         = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1"
ORIG_CORE    = f"{BASE}/phase4/extraction/master_incentive_core.csv"
ASDIV_CORE   = f"{BASE}/evaluation/ASDiv/extraction/asdiv_master_incentive_core.csv"
MMLU_CROSS   = f"{BASE}/evaluation/mmlu/neurons/mmlu/cross_subject"
OUT_DIR      = f"{BASE}/evaluation/ablation_neurons"

K_THRESHOLDS = [27, 30, 35, 40, 45, 50, 54]

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Load sets
# =============================================================================
def load_set(path):
    df = pd.read_csv(path)
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))

def to_json(neuron_set, path):
    d = defaultdict(list)
    for (layer, neuron) in sorted(neuron_set):
        d[str(layer)].append(neuron)
    with open(path, 'w') as f:
        json.dump(dict(d), f, indent=2)

orig_set  = load_set(ORIG_CORE)
asdiv_set = load_set(ASDIV_CORE)

print(f"Orig core:  {len(orig_set):>6} neurons")
print(f"ASDiv core: {len(asdiv_set):>6} neurons")

# =============================================================================
# Generate master core at each K
# =============================================================================
print(f"\n{'K':>4}  {'MMLU_K':>8}  {'∩ ASDiv':>8}  {'∩ Orig (master)':>16}  {'Output'}")
print("-" * 75)

for k in K_THRESHOLDS:
    mmlu_k_set  = load_set(os.path.join(MMLU_CROSS, f"k{k}.csv"))
    master_core = mmlu_k_set & asdiv_set & orig_set

    out_path = os.path.join(OUT_DIR, f"neurons_master_k{k}.json")
    to_json(master_core, out_path)

    mmlu_asdiv = mmlu_k_set & asdiv_set
    print(f"  {k:>2}  {len(mmlu_k_set):>8}  {len(mmlu_asdiv):>8}  {len(master_core):>16}  → {out_path}")

# =============================================================================
# Also export shared163 and triple29 to same directory for convenience
# =============================================================================
mmlu_k54   = load_set(os.path.join(MMLU_CROSS, "k54.csv"))
shared_163 = mmlu_k54 & asdiv_set
triple_29  = mmlu_k54 & asdiv_set & orig_set

to_json(shared_163, os.path.join(OUT_DIR, "neurons_shared163.json"))
to_json(triple_29,  os.path.join(OUT_DIR, "neurons_triple29.json"))

print(f"\n  Shared 163 (MMLU K=54 ∩ ASDiv):          {len(shared_163)} neurons")
print(f"  Triple 29  (MMLU K=54 ∩ ASDiv ∩ Orig):   {len(triple_29)} neurons")
print(f"\nAll JSON files saved to: {OUT_DIR}/")

# =============================================================================
# Layer distribution of each master core
# =============================================================================
print(f"\n{'K':>4}  {'Master core':>12}  Layer distribution (top 5 layers)")
print("-" * 70)
for k in K_THRESHOLDS:
    with open(os.path.join(OUT_DIR, f"neurons_master_k{k}.json")) as f:
        d = json.load(f)
    total = sum(len(v) for v in d.values())
    layer_counts = {int(l): len(v) for l, v in d.items()}
    top5 = sorted(layer_counts.items(), key=lambda x: -x[1])[:5]
    top5_str = ", ".join(f"L{l}:{c}" for l,c in top5)
    print(f"  {k:>2}  {total:>12}  {top5_str}")