"""
generate_master_cores.py
================================================================================
Generates neurons_master_k{K}.json for K = 27, 30, 35, 40, 45, 50, 54.

Master core at K = MMLU_K ∩ ASDiv_core ∩ Orig_core

Also generates union JSONs for Experiment 2:
    neurons_union_k{K}.json = neurons_A ∪ master_k{K}

All files saved to: evaluation/ablation_neurons/
"""

import os
import json
import pandas as pd
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
BASE       = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1"
ORIG_CORE  = f"{BASE}/phase4/extraction/master_incentive_core.csv"
ASDIV_CORE = f"{BASE}/evaluation/ASDiv/extraction/asdiv_master_incentive_core.csv"
MMLU_CROSS = f"{BASE}/evaluation/mmlu/neurons/mmlu/cross_subject"
NEURONS_A  = f"{BASE}/phase4/extraction/neurons_A.json"
OUT_DIR    = f"{BASE}/evaluation/ablation_neurons"

K_THRESHOLDS = [27, 30, 35, 40, 45, 50, 54]

os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================
def load_set(path):
    df = pd.read_csv(path)
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))

def load_json_set(path):
    with open(path) as f:
        d = json.load(f)
    return set((int(layer), neuron)
               for layer, neurons in d.items()
               for neuron in neurons)

def to_json(neuron_set, path):
    d = defaultdict(list)
    for (layer, neuron) in sorted(neuron_set):
        d[str(layer)].append(neuron)
    with open(path, 'w') as f:
        json.dump(dict(d), f, indent=2)
    print(f"  Saved {path}  ({len(neuron_set)} neurons)")

# =============================================================================
# Load base sets
# =============================================================================
orig_set  = load_set(ORIG_CORE)
asdiv_set = load_set(ASDIV_CORE)
neurons_a = load_json_set(NEURONS_A)

print(f"Orig core:    {len(orig_set):>6} neurons")
print(f"ASDiv core:   {len(asdiv_set):>6} neurons")
print(f"Neurons A:    {len(neurons_a):>6} neurons  (layers 18-27)")

# =============================================================================
# Generate master cores and union sets at each K
# =============================================================================
print(f"\n{'K':>4}  {'MMLU_K':>8}  {'Master core':>12}  {'Union w/ A':>12}")
print("-" * 50)

for k in K_THRESHOLDS:
    mmlu_k      = load_set(os.path.join(MMLU_CROSS, f"k{k}.csv"))
    master_core = mmlu_k & asdiv_set & orig_set
    union_set   = neurons_a | master_core

    print(f"  {k:>2}  {len(mmlu_k):>8}  {len(master_core):>12}  {len(union_set):>12}")

    to_json(master_core, os.path.join(OUT_DIR, f"neurons_master_k{k}.json"))
    to_json(union_set,   os.path.join(OUT_DIR, f"neurons_union_k{k}.json"))

# =============================================================================
# Layer distribution of each master core
# =============================================================================
print(f"\nLayer distribution of master cores:")
print(f"{'K':>4}  {'Total':>6}  {'Top layers'}")
print("-" * 60)
for k in K_THRESHOLDS:
    path = os.path.join(OUT_DIR, f"neurons_master_k{k}.json")
    with open(path) as f:
        d = json.load(f)
    total  = sum(len(v) for v in d.values())
    by_layer = {int(l): len(v) for l, v in d.items()}
    top5   = sorted(by_layer.items(), key=lambda x: -x[1])[:5]
    top5_str = ", ".join(f"L{l}:{c}" for l, c in top5)
    print(f"  {k:>2}  {total:>6}  {top5_str}")

print(f"\nAll files saved to: {OUT_DIR}/")
print("Done ✓")
