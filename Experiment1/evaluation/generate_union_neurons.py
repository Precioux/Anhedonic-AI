"""
generate_union_neurons.py
================================================================================
Generates neurons_union_k{K}.json for K = 27, 30, 35, 40, 45, 50, 54.

Union = neurons_A ∪ master_kK
     = original Model A neurons (layers 18-27)
     + master core at K (MMLU_K ∩ ASDiv ∩ Orig)

Output: evaluation/ablation_neurons/neurons_union_k{K}.json
"""

import os
import json
from collections import defaultdict

NEURONS_A  = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase5/neurons_A.json"
MASTER_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/evaluation/ablation_neurons"
K_THRESHOLDS = [27, 30, 35, 40, 45, 50, 54]

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

# Load neurons_A
neurons_a = load_json_set(NEURONS_A)
print(f"Neurons A: {len(neurons_a)} neurons")

print(f"\n{'K':>4}  {'Master':>8}  {'Union':>8}  {'Output'}")
print("-" * 65)

for k in K_THRESHOLDS:
    master_path = os.path.join(MASTER_DIR, f"neurons_master_k{k}.json")
    union_path  = os.path.join(MASTER_DIR, f"neurons_union_k{k}.json")

    master_set = load_json_set(master_path)
    union_set  = neurons_a | master_set

    to_json(union_set, union_path)
    print(f"  {k:>2}  {len(master_set):>8}  {len(union_set):>8}  → {union_path}")

print("\nDone ✓")