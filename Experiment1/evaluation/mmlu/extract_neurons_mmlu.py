"""
extract_neurons_mmlu.py
================================================================================
1. For each of the 54 MMLU subjects, localises reward-sensitive MLP neurons
   using the same 3σ method as extract_neurons_asdiv.py:
     - delta = condition_mean - neutral_mean
     - neurons where |delta| > 3σ of the full delta array
     - money neurons, reward neurons, core = money ∩ reward

2. Saves per-subject CSVs:
     neurons/mmlu/per_subject/{subject}/reward.csv
     neurons/mmlu/per_subject/{subject}/money.csv
     neurons/mmlu/per_subject/{subject}/core.csv

3. Computes cross-subject neuron counts and saves:
     neurons/mmlu/cross_subject/subject_counts.csv   ← every neuron + how many subjects it appeared in
     neurons/mmlu/cross_subject/k27.csv              ← neurons in ≥27 subjects (half)
     neurons/mmlu/cross_subject/k30.csv
     neurons/mmlu/cross_subject/k35.csv
     neurons/mmlu/cross_subject/k40.csv
     neurons/mmlu/cross_subject/k45.csv
     neurons/mmlu/cross_subject/k50.csv
     neurons/mmlu/cross_subject/k54.csv              ← full intersection (Option B)

4. Prints a full overlap report comparing MMLU cross-subject cores against
   the original and ASDiv cores.

Columns in all neuron CSVs: layer, neuron
"""

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================
ACT_DIR      = "activations/mmlu"
OUT_BASE_DIR = "neurons/mmlu"
PER_SUBJ_DIR = os.path.join(OUT_BASE_DIR, "per_subject")
CROSS_DIR    = os.path.join(OUT_BASE_DIR, "cross_subject")

ORIGINAL_CORE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
ASDIV_CORE    = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/evaluation/ASDiv/extraction/asdiv_master_incentive_core.csv"

K_THRESHOLDS = [27, 30, 35, 40, 45, 50, 54]

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_computer_science",
    "college_mathematics", "college_medicine", "computer_security",
    "conceptual_physics", "econometrics", "electrical_engineering",
    "elementary_mathematics", "formal_logic", "global_facts",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "virology", "world_religions",
]

N_SUBJECTS = len(SUBJECTS)

# =============================================================================
# Helpers
# =============================================================================

def load_activation_mean(subject: str, condition: str) -> np.ndarray:
    """Load .pt dict, stack tensors, return mean across questions.
    Output shape: [num_layers, intermediate_dim]
    """
    path = os.path.join(ACT_DIR, subject, f"{condition}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    data    = torch.load(path, map_location='cpu')
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()   # [N, layers, dim]
    return stacked.mean(dim=0).numpy()       # [layers, dim]


def find_neurons_3sigma(delta: np.ndarray) -> set:
    """Return set of (layer, neuron) where |delta| > 3σ of the full delta array."""
    threshold = 3 * np.std(delta)
    layers, neurons = np.where(np.abs(delta) > threshold)
    return set(zip(layers.tolist(), neurons.tolist()))


def neurons_to_df(neuron_set: set) -> pd.DataFrame:
    return pd.DataFrame(sorted(neuron_set), columns=['layer', 'neuron'])


def save_neurons(neuron_set: set, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    neurons_to_df(neuron_set).to_csv(path, index=False)


def df_to_set(df: pd.DataFrame) -> set:
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def overlap_report(name_a: str, set_a: set, name_b: str, set_b: set) -> None:
    inter    = set_a & set_b
    j        = jaccard(set_a, set_b)
    recall_a = len(inter) / len(set_a) if set_a else 0
    recall_b = len(inter) / len(set_b) if set_b else 0
    print(f"  {name_a} ∩ {name_b}:")
    print(f"    |{name_a}|={len(set_a)}, |{name_b}|={len(set_b)}, "
          f"|intersection|={len(inter)}")
    print(f"    Jaccard={j:.4f}")
    print(f"    % of {name_a} recovered: {recall_a*100:.1f}%")
    print(f"    % of {name_b} recovered: {recall_b*100:.1f}%")


# =============================================================================
# Step 1 — Per-subject neuron extraction
# =============================================================================
print("=" * 60)
print(f"STEP 1 — Per-subject neuron extraction ({N_SUBJECTS} subjects)")
print("=" * 60)

# neuron_subject_counts[neuron] = number of subjects where it appeared in core
neuron_subject_counts = defaultdict(int)

all_subject_cores = {}   # subject → set of (layer, neuron)

for s_idx, subject in enumerate(SUBJECTS, 1):
    print(f"\n[{s_idx}/{N_SUBJECTS}] {subject}")

    try:
        neu = load_activation_mean(subject, "neutral")
        rew = load_activation_mean(subject, "reward")
        mon = load_activation_mean(subject, "money")
    except FileNotFoundError as e:
        print(f"  SKIPPED — {e}")
        continue

    delta_rew = rew - neu
    delta_mon = mon - neu

    reward_neurons = find_neurons_3sigma(delta_rew)
    money_neurons  = find_neurons_3sigma(delta_mon)
    core_neurons   = reward_neurons & money_neurons

    print(f"  reward={len(reward_neurons)}, money={len(money_neurons)}, "
          f"core={len(core_neurons)}")

    # Save per-subject CSVs
    subj_dir = os.path.join(PER_SUBJ_DIR, subject)
    save_neurons(reward_neurons, os.path.join(subj_dir, "reward.csv"))
    save_neurons(money_neurons,  os.path.join(subj_dir, "money.csv"))
    save_neurons(core_neurons,   os.path.join(subj_dir, "core.csv"))

    # Accumulate cross-subject counts
    for neuron in core_neurons:
        neuron_subject_counts[neuron] += 1

    all_subject_cores[subject] = core_neurons

n_processed = len(all_subject_cores)
print(f"\nProcessed {n_processed}/{N_SUBJECTS} subjects.")
print(f"Unique neurons appearing in at least 1 subject core: "
      f"{len(neuron_subject_counts)}")

# =============================================================================
# Step 2 — Cross-subject analysis
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — Cross-subject neuron counts and K-threshold cores")
print("=" * 60)

os.makedirs(CROSS_DIR, exist_ok=True)

# Save subject_counts.csv
counts_rows = [
    {"layer": layer, "neuron": neuron, "subject_count": count}
    for (layer, neuron), count in sorted(neuron_subject_counts.items())
]
counts_df = pd.DataFrame(counts_rows)
counts_df.to_csv(os.path.join(CROSS_DIR, "subject_counts.csv"), index=False)
print(f"\nSaved subject_counts.csv ({len(counts_df)} unique neurons)")

# Distribution of subject counts
print("\nSubject count distribution (how many neurons appear in exactly N subjects):")
dist = counts_df['subject_count'].value_counts().sort_index()
for n_subj, count in dist.items():
    bar = '█' * min(count // 10, 50)
    print(f"  {n_subj:>3} subjects: {count:>5} neurons  {bar}")

# Save K-threshold CSVs
print()
k_sets = {}
for k in K_THRESHOLDS:
    k_set = {neuron for neuron, count in neuron_subject_counts.items()
             if count >= k}
    k_sets[k] = k_set
    out_path = os.path.join(CROSS_DIR, f"k{k}.csv")
    save_neurons(k_set, out_path)
    print(f"  K={k:>2}: {len(k_set):>5} neurons → {out_path}")

# Sanity check: k54 should match full intersection
full_intersection = set.intersection(*all_subject_cores.values()) if all_subject_cores else set()
k54_set = k_sets.get(54, set())
match = full_intersection == k54_set
print(f"\n  Sanity check — K=54 matches full intersection: {'✓ PASS' if match else '✗ FAIL'}")
if not match:
    print(f"    Full intersection: {len(full_intersection)}, K=54 set: {len(k54_set)}")

# =============================================================================
# Step 3 — Overlap with original and ASDiv cores
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — Overlap with original and ASDiv cores")
print("=" * 60)

orig_set  = set()
asdiv_set = set()

if os.path.exists(ORIGINAL_CORE):
    orig_df  = pd.read_csv(ORIGINAL_CORE)
    orig_set = df_to_set(orig_df)
    print(f"\nOriginal core loaded: {len(orig_set)} neurons")
else:
    print(f"\nWARNING: {ORIGINAL_CORE} not found — skipping orig overlap.")

if os.path.exists(ASDIV_CORE):
    asdiv_df  = pd.read_csv(ASDIV_CORE)
    asdiv_set = df_to_set(asdiv_df)
    print(f"ASDiv core loaded:    {len(asdiv_set)} neurons")
else:
    print(f"WARNING: {ASDIV_CORE} not found — skipping asdiv overlap.")

print()
for k in K_THRESHOLDS:
    k_set = k_sets[k]
    print(f"── K={k} ({len(k_set)} neurons) ──")
    if orig_set:
        overlap_report("orig", orig_set, f"mmlu_k{k}", k_set)
    if asdiv_set:
        overlap_report("asdiv", asdiv_set, f"mmlu_k{k}", k_set)
    print()

# =============================================================================
# Step 4 — Layer 27 spotlight
# =============================================================================
print("=" * 60)
print("STEP 4 — Layer 27 spotlight")
print("=" * 60)

def l27(s): return {n for (l, n) in s if l == 27}

print(f"\n  {'Source':<20} {'L27 neurons':>12}")
print(f"  {'-'*34}")
if orig_set:
    print(f"  {'orig':<20} {len(l27(orig_set)):>12}")
if asdiv_set:
    print(f"  {'asdiv':<20} {len(l27(asdiv_set)):>12}")
for k in K_THRESHOLDS:
    print(f"  {f'mmlu_k{k}':<20} {len(l27(k_sets[k])):>12}")

# Layer 27 per-subject counts
print(f"\n  Layer 27 neurons per subject (top 10 and bottom 10):")
l27_counts = [
    (subj, len(l27(core)))
    for subj, core in all_subject_cores.items()
]
l27_counts.sort(key=lambda x: -x[1])
print(f"  {'Subject':<45} {'L27 core neurons':>16}")
print(f"  {'-'*63}")
for subj, cnt in l27_counts[:10]:
    print(f"  {subj:<45} {cnt:>16}")
print(f"  {'...'}")
for subj, cnt in l27_counts[-10:]:
    print(f"  {subj:<45} {cnt:>16}")

# =============================================================================
# Step 5 — Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\n  Subjects processed:    {n_processed}/54")
print(f"  Per-subject CSVs:      {n_processed * 3} files in {PER_SUBJ_DIR}/")
print(f"  Cross-subject CSVs:    {len(K_THRESHOLDS) + 1} files in {CROSS_DIR}/")
print()
print(f"  {'K threshold':<15} {'Neurons':>8}  {'% of orig':>10}  {'% of asdiv':>11}")
print(f"  {'-'*48}")
for k in K_THRESHOLDS:
    k_set   = k_sets[k]
    pct_o   = len(orig_set & k_set) / len(orig_set) * 100  if orig_set  else float('nan')
    pct_a   = len(asdiv_set & k_set) / len(asdiv_set) * 100 if asdiv_set else float('nan')
    print(f"  K={k:<13} {len(k_set):>8}  {pct_o:>9.1f}%  {pct_a:>10.1f}%")

print("\nDone ✓")