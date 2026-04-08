import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================
# Configuration
# =============================================================================
ACT_DIR      = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/MMLU/activations"
OUT_BASE_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/MMLU/neurons"
PER_SUBJ_DIR = os.path.join(OUT_BASE_DIR, "per_subject")
CROSS_DIR    = os.path.join(OUT_BASE_DIR, "cross_subject")

ORIG_CORE  = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/orig/master_incentive_core.csv"
ASDIV_CORE = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/ASDiv/neurons/asdiv_master_incentive_core.csv"

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
def load_activation_mean(subject, condition):
    path    = os.path.join(ACT_DIR, subject, f"{condition}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    data    = torch.load(path, map_location='cpu')
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()
    return stacked.mean(dim=0).numpy()  # [num_layers, intermediate_dim]


def find_neurons_3sigma(delta):
    threshold = 3 * np.std(delta)
    layers, neurons = np.where(np.abs(delta) > threshold)
    return set(zip(layers.tolist(), neurons.tolist()))


def neurons_to_df(neuron_set):
    return pd.DataFrame(sorted(neuron_set), columns=['layer', 'neuron'])


def save_neurons(neuron_set, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    neurons_to_df(neuron_set).to_csv(path, index=False)


def df_to_set(df):
    return set(zip(df['layer'].tolist(), df['neuron'].tolist()))


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def overlap_report(name_a, set_a, name_b, set_b):
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

neuron_subject_counts = defaultdict(int)
all_subject_cores     = {}

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

    subj_dir = os.path.join(PER_SUBJ_DIR, subject)
    save_neurons(reward_neurons, os.path.join(subj_dir, "reward.csv"))
    save_neurons(money_neurons,  os.path.join(subj_dir, "money.csv"))
    save_neurons(core_neurons,   os.path.join(subj_dir, "core.csv"))

    for neuron in core_neurons:
        neuron_subject_counts[neuron] += 1

    all_subject_cores[subject] = core_neurons

n_processed = len(all_subject_cores)
print(f"\nProcessed {n_processed}/{N_SUBJECTS} subjects.")
print(f"Unique neurons in at least 1 subject core: {len(neuron_subject_counts)}")

# =============================================================================
# Step 2 — Cross-subject K-threshold analysis
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2 — Cross-subject K-threshold cores")
print("=" * 60)

os.makedirs(CROSS_DIR, exist_ok=True)

# subject_counts.csv
counts_rows = [
    {"layer": layer, "neuron": neuron, "subject_count": count}
    for (layer, neuron), count in sorted(neuron_subject_counts.items())
]
counts_df = pd.DataFrame(counts_rows)
counts_df.to_csv(os.path.join(CROSS_DIR, "subject_counts.csv"), index=False)
print(f"\nSaved subject_counts.csv ({len(counts_df)} unique neurons)")

# Distribution
print("\nSubject count distribution:")
dist = counts_df['subject_count'].value_counts().sort_index()
for n_subj, count in dist.items():
    bar = '█' * min(count // 20, 50)
    print(f"  {n_subj:>3} subjects: {count:>6} neurons  {bar}")

# K-threshold CSVs
print()
k_sets = {}
for k in K_THRESHOLDS:
    k_set    = {n for n, cnt in neuron_subject_counts.items() if cnt >= k}
    k_sets[k] = k_set
    out_path = os.path.join(CROSS_DIR, f"k{k}.csv")
    save_neurons(k_set, out_path)
    print(f"  K={k:>2}: {len(k_set):>6} neurons → {out_path}")

# Sanity check
full_intersection = set.intersection(*all_subject_cores.values()) if all_subject_cores else set()
match = full_intersection == k_sets.get(54, set())
print(f"\n  Sanity check — K=54 matches full intersection: {'✓ PASS' if match else '✗ FAIL'}")

# =============================================================================
# Step 3 — Overlap with 72B orig and ASDiv cores
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3 — Overlap with 72B orig and ASDiv cores")
print("=" * 60)

orig_set  = set()
asdiv_set = set()

if os.path.exists(ORIG_CORE):
    orig_set = df_to_set(pd.read_csv(ORIG_CORE))
    print(f"\n72B Orig core loaded:  {len(orig_set)} neurons")
else:
    print(f"\nWARNING: {ORIG_CORE} not found")

if os.path.exists(ASDIV_CORE):
    asdiv_set = df_to_set(pd.read_csv(ASDIV_CORE))
    print(f"72B ASDiv core loaded: {len(asdiv_set)} neurons")
else:
    print(f"WARNING: {ASDIV_CORE} not found")

print()
for k in K_THRESHOLDS:
    k_set = k_sets[k]
    print(f"── K={k} ({len(k_set)} neurons) ──")
    if orig_set:
        overlap_report("orig",  orig_set,  f"mmlu_k{k}", k_set)
    if asdiv_set:
        overlap_report("asdiv", asdiv_set, f"mmlu_k{k}", k_set)
    print()

# =============================================================================
# Step 4 — Layer spotlight (proportional to 7B findings)
# =============================================================================
print("=" * 60)
print("STEP 4 — Layer spotlight")
print("=" * 60)

# In 7B the key band was L8-14 (29-50% depth, 28 layers)
# In 72B proportional equivalent is ~L23-40 (29-50% depth, 80 layers)
# Also check L38-43 (peak from orig 72B) and L78-79 (late spike from ASDiv)

def layer_count(s, layer):
    return sum(1 for (l, _) in s if l == layer)

def band_count(s, l_start, l_end):
    return sum(1 for (l, _) in s if l_start <= l <= l_end)

print(f"\n  {'Source':<22} {'Total':>7}  {'L23-40':>7}  {'L38-43':>7}  {'L57-58':>7}  {'L78-79':>7}")
print(f"  {'-'*60}")
for tag, s in [("72B orig", orig_set), ("72B asdiv", asdiv_set)] + \
              [(f"mmlu_k{k}", k_sets[k]) for k in K_THRESHOLDS]:
    total  = len(s)
    b2340  = band_count(s, 23, 40)
    b3843  = band_count(s, 38, 43)
    b5758  = band_count(s, 57, 58)
    b7879  = band_count(s, 78, 79)
    print(f"  {tag:<22} {total:>7}  {b2340:>7}  {b3843:>7}  {b5758:>7}  {b7879:>7}")

# =============================================================================
# Step 5 — Summary table
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
    k_set = k_sets[k]
    pct_o = len(orig_set & k_set)  / len(orig_set)  * 100 if orig_set  else float('nan')
    pct_a = len(asdiv_set & k_set) / len(asdiv_set) * 100 if asdiv_set else float('nan')
    print(f"  K={k:<13} {len(k_set):>8}  {pct_o:>9.1f}%  {pct_a:>10.1f}%")

print("\nDone ✓")