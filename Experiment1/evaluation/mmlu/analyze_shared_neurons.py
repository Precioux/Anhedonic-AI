"""
analyze_shared_neurons.py
================================================================================
Analyzes the layer distribution of neurons shared between:
  - MMLU K=54 (177 neurons — appear in all 54 subjects)
  - ASDiv core (7913 neurons)
  - Their intersection (163 neurons)

Also shows layer distributions for orig, asdiv, and all MMLU K thresholds
for comparison.
"""

import os
import pandas as pd
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
ORIG_CORE  = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
ASDIV_CORE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/evaluation/ASDiv/extraction/asdiv_master_incentive_core.csv"
MMLU_DIR   = "neurons/mmlu/cross_subject"

K_THRESHOLDS = [27, 30, 35, 40, 45, 50, 54]

# =============================================================================
# Load
# =============================================================================
def load_set(path):
    df = pd.read_csv(path)
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))

def layer_dist(neuron_set, num_layers=28):
    counts = defaultdict(int)
    for (layer, _) in neuron_set:
        counts[layer] += 1
    return counts

def print_layer_dist(name, neuron_set, num_layers=28, bar_scale=1):
    dist = layer_dist(neuron_set, num_layers)
    total = len(neuron_set)
    print(f"\n  {name} ({total} neurons total)")
    print(f"  {'Layer':>6}  {'Count':>6}  {'%':>6}  Bar")
    print(f"  {'-'*55}")
    for l in range(num_layers):
        count = dist.get(l, 0)
        pct   = count / total * 100 if total else 0
        bar   = '█' * max(1, int(count / bar_scale)) if count else ''
        print(f"  {l:>6}  {count:>6}  {pct:>5.1f}%  {bar}")

# =============================================================================
# Load all sets
# =============================================================================
orig_set  = load_set(ORIG_CORE)
asdiv_set = load_set(ASDIV_CORE)

k_sets = {}
for k in K_THRESHOLDS:
    path = os.path.join(MMLU_DIR, f"k{k}.csv")
    k_sets[k] = load_set(path)

mmlu_k54  = k_sets[54]
shared_163 = mmlu_k54 & asdiv_set
triple     = mmlu_k54 & asdiv_set & orig_set

print("=" * 60)
print("KEY SET SIZES")
print("=" * 60)
print(f"  Orig core:              {len(orig_set):>5} neurons")
print(f"  ASDiv core:             {len(asdiv_set):>5} neurons")
print(f"  MMLU K=54:              {len(mmlu_k54):>5} neurons")
print(f"  MMLU K=54 ∩ ASDiv:      {len(shared_163):>5} neurons  ← 163 shared")
print(f"  MMLU K=54 ∩ ASDiv ∩ Orig: {len(triple):>3} neurons  ← triple core")

# =============================================================================
# Layer distribution of the 163 shared neurons
# =============================================================================
print("\n" + "=" * 60)
print("LAYER DISTRIBUTION — 163 SHARED NEURONS (MMLU K=54 ∩ ASDiv)")
print("=" * 60)
print_layer_dist("MMLU_K54 ∩ ASDiv", shared_163, bar_scale=1)

# =============================================================================
# Layer distribution of triple core
# =============================================================================
print("\n" + "=" * 60)
print("LAYER DISTRIBUTION — TRIPLE CORE (MMLU K=54 ∩ ASDiv ∩ Orig)")
print("=" * 60)
print_layer_dist("Triple core", triple, bar_scale=1)

# =============================================================================
# Comparison table — layer distribution across all sets
# =============================================================================
print("\n" + "=" * 60)
print("LAYER COMPARISON TABLE")
print("=" * 60)

sets_to_compare = {
    "orig":      orig_set,
    "asdiv":     asdiv_set,
    "mmlu_k54":  mmlu_k54,
    "shared163": shared_163,
    "triple":    triple,
}

header = f"  {'Layer':>5}  " + "  ".join(f"{name:>10}" for name in sets_to_compare)
print(f"\n{header}")
print(f"  {'-' * (7 + 12 * len(sets_to_compare))}")

for l in range(28):
    row = f"  {l:>5}  "
    for name, s in sets_to_compare.items():
        dist = layer_dist(s)
        row += f"{dist.get(l, 0):>10}  "
    print(row)

# Totals
row = f"  {'TOTAL':>5}  "
for name, s in sets_to_compare.items():
    row += f"{len(s):>10}  "
print(f"  {'-' * (7 + 12 * len(sets_to_compare))}")
print(row)

# =============================================================================
# Which layers dominate the 163 shared neurons?
# =============================================================================
print("\n" + "=" * 60)
print("TOP LAYERS IN SHARED 163 NEURONS")
print("=" * 60)

dist_163 = layer_dist(shared_163)
sorted_layers = sorted(dist_163.items(), key=lambda x: -x[1])
print(f"\n  {'Rank':>4}  {'Layer':>6}  {'Count':>6}  {'%':>6}  Bar")
print(f"  {'-'*50}")
for rank, (layer, count) in enumerate(sorted_layers, 1):
    pct = count / len(shared_163) * 100
    bar = '█' * count
    print(f"  {rank:>4}  {layer:>6}  {count:>6}  {pct:>5.1f}%  {bar}")

print("\nDone ✓")