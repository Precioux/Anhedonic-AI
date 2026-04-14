"""
extract_neurons_modelB_v2.py  --  Model B Neuron Selection (revised)
=====================================================================
Replaces the binary per-domain 3-sigma threshold with a continuous
min-sigma-ratio score, fixing the paradox where adding more domains
(BE + philosophy) expanded rather than tightened the selection.

Root cause of the original problem:
    threshold = 3 * std(delta)  computed per-domain independently.
    BE/philosophy have lower-variance deltas, so their 3-sigma bar is
    a lower absolute value -- more neurons pass, the intersection grows.

Fix:
    score(l, n) = min over all 8 conditions of  |delta[l,n]| / std(delta)
    A neuron must be extreme in EVERY domain, not just clear an easy bar.

Scope: only layers 18-27 (matching the causally validated Model A window).

Outputs:
    modelB_v2_master_incentive_core.csv  -- final neuron list with scores
    modelB_v2_score_distribution.csv     -- full audit table (all 3-sigma+ candidates)

Usage:
    python extract_neurons_modelB_v2.py --dry_run     # sweep only, no save
    python extract_neurons_modelB_v2.py               # run with default sigma
    python extract_neurons_modelB_v2.py --sigma 4.5   # override threshold
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Config
# =============================================================================
ACTIVATIONS_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase6/activations"
MODEL_A_CORE    = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
OUTPUT_CORE     = "modelB_v2_master_incentive_core.csv"
OUTPUT_SCORES   = "modelB_v2_score_distribution.csv"

ABLATION_LAYERS = set(range(18, 28))
DEFAULT_SIGMA   = 4.0

ACTIVATION_FILES = {
    "geo":             ("neutral_activations_geo.pt",
                        "money_activations_geo.pt",
                        "reward_activations_geo.pt"),
    "math":            ("neutral_activations_math.pt",
                        "money_activations_math.pt",
                        "reward_activations_math.pt"),
    "business_ethics": ("neutral_activations_business_ethics.pt",
                        "money_activations_business_ethics.pt",
                        "reward_activations_business_ethics.pt"),
    "philosophy":      ("neutral_activations_philosophy.pt",
                        "money_activations_philosophy.pt",
                        "reward_activations_philosophy.pt"),
}


# =============================================================================
# Loading
# =============================================================================
def load_mean(filename):
    path = os.path.join(ACTIVATIONS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    print(f"    {filename} ...", end=" ", flush=True)
    data    = torch.load(path, map_location="cpu")
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()
    mean    = stacked.mean(dim=0).numpy()
    print(f"{stacked.shape}  ->  mean {mean.shape}")
    return mean


# =============================================================================
# Scoring
# =============================================================================
def compute_min_sigma(delta_arrays):
    """
    score[l, n] = min over all conditions of  |delta[l,n]| / std(delta)

    A neuron scoring 4.0 is at least 4-sigma above the noise floor
    in its weakest domain. Neurons that barely cleared one easy bar
    will score near 3.0 and drop out when the threshold is raised.
    """
    ratios = np.stack(
        [np.abs(d) / (np.std(d) + 1e-12) for d in delta_arrays],
        axis=0
    )  # [n_conditions, L, D]
    return ratios.min(axis=0)  # [L, D]


# =============================================================================
# Threshold sweep
# =============================================================================
def threshold_sweep(combined_scores, a_set):
    print("\n" + "-" * 60)
    print("  Threshold sweep  (layers 18-27 only, Model A always included)")
    print(f"  {'sigma':>8}  {'candidates':>12}  {'+ A core':>10}  {'net additions':>14}")
    print("-" * 60)
    for sigma in [3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0]:
        candidates = {
            (int(l), int(n))
            for l, n in zip(*np.where(combined_scores >= sigma))
            if int(l) in ABLATION_LAYERS
        }
        total     = a_set | candidates
        additions = total - a_set
        flag = "  <- target range" if 1400 <= len(total) <= 1800 else ""
        print(f"  sigma >= {sigma:>4.2f}  {len(candidates):>12,}  {len(total):>10,}  "
              f"{len(additions):>14,}{flag}")
    print("-" * 60)


# =============================================================================
# Main
# =============================================================================
def main(sigma_threshold=DEFAULT_SIGMA, dry_run=False):
    print("=" * 60)
    print("  MODEL B v2 -- NEURON SELECTION")
    print("  Continuous min-sigma-ratio | layers 18-27 only")
    print("=" * 60)

    # Load all 12 activation files
    domain_means = {}
    for domain, (neu_f, mon_f, rew_f) in ACTIVATION_FILES.items():
        print(f"\n  {domain}:")
        domain_means[domain] = {
            "neutral": load_mean(neu_f),
            "money":   load_mean(mon_f),
            "reward":  load_mean(rew_f),
        }

    # Shape check
    shapes = {arr.shape for dm in domain_means.values() for arr in dm.values()}
    assert len(shapes) == 1, f"Shape mismatch: {shapes}"
    num_layers, intermediate_dim = next(iter(shapes))
    total_neurons = num_layers * intermediate_dim
    print(f"\n  Grid: {num_layers} layers x {intermediate_dim:,} neurons = {total_neurons:,} total")
    print(f"  Ablation scope: layers 18-27 ({len(ABLATION_LAYERS)} layers)")

    # Deltas
    print("\n  Computing deltas (condition - neutral) ...")
    deltas_money  = []
    deltas_reward = []
    print(f"\n  {'Domain':<20}  {'std(money)':>11}  {'mean|money|':>12}  "
          f"{'std(reward)':>12}  {'mean|reward|':>13}")
    print("  " + "-" * 74)
    for domain, means in domain_means.items():
        dm = means["money"]  - means["neutral"]
        dr = means["reward"] - means["neutral"]
        deltas_money.append(dm)
        deltas_reward.append(dr)
        print(f"  {domain:<20}  {np.std(dm):>11.6f}  {np.abs(dm).mean():>12.6f}  "
              f"{np.std(dr):>12.6f}  {np.abs(dr).mean():>13.6f}")

    print("\n  Note: lower std in BE/philosophy = lower absolute 3-sigma bar")
    print("  = more neurons pass per-domain threshold = the original inflation.")

    # Continuous scoring
    print("\n  Computing min-sigma-ratio scores ...")
    money_scores  = compute_min_sigma(deltas_money)
    reward_scores = compute_min_sigma(deltas_reward)
    combined      = np.minimum(money_scores, reward_scores)
    print(f"    Score range : {combined.min():.4f} - {combined.max():.4f}")
    print(f"    Mean score  : {combined.mean():.4f}")
    l18_27_scores = combined[18:28, :]
    print(f"    L18-27 range: {l18_27_scores.min():.4f} - {l18_27_scores.max():.4f}")
    print(f"    L18-27 mean : {l18_27_scores.mean():.4f}")

    # Load Model A core -- layers 18-27 only
    if not os.path.exists(MODEL_A_CORE):
        raise FileNotFoundError(f"Model A core not found: {MODEL_A_CORE}")
    df_a_full = pd.read_csv(MODEL_A_CORE)
    df_a      = df_a_full[df_a_full["layer"].isin(ABLATION_LAYERS)].copy()
    a_set     = set(zip(df_a["layer"].astype(int), df_a["neuron"].astype(int)))
    print(f"\n  Model A core (layers 18-27): {len(a_set):,} neurons  "
          f"(full CSV has {len(df_a_full):,} across all layers)")

    # Threshold sweep
    threshold_sweep(combined, a_set)

    if dry_run:
        print("\n  --dry_run set. No files saved.")
        return

    # Apply chosen threshold
    print(f"\n  Applying sigma >= {sigma_threshold:.2f} (layers 18-27 only) ...")
    candidates = {
        (int(l), int(n))
        for l, n in zip(*np.where(combined >= sigma_threshold))
        if int(l) in ABLATION_LAYERS
    }
    model_b_set    = a_set | candidates
    additions      = model_b_set - a_set
    dropped_from_a = a_set - candidates  # A neurons below new bar, force-included

    print(f"\n  Composition:")
    print(f"    Model A core (L18-27) : {len(a_set):>6,}")
    print(f"    A neurons below bar   : {len(dropped_from_a):>6,}  (force-included)")
    print(f"    Net BE/phil additions : {len(additions):>6,}")
    print(f"    Model B v2 total      : {len(model_b_set):>6,}  "
          f"({len(model_b_set)/total_neurons*100:.4f}% of network)")
    print(f"    B v2 / A ratio        : {len(model_b_set)/max(len(a_set),1):.2f}x")

    # Layer breakdown
    print(f"\n  Layer breakdown:")
    print(f"    {'Layer':>6}  {'Model A':>8}  {'Additions':>10}  "
          f"{'Total B v2':>11}  {'% of layer':>11}")
    print(f"    " + "-" * 52)
    a_by_layer   = df_a.groupby("layer").size().to_dict()
    b_by_layer   = {}
    add_by_layer = {}
    for l, n in model_b_set:
        b_by_layer[l]   = b_by_layer.get(l, 0) + 1
    for l, n in additions:
        add_by_layer[l] = add_by_layer.get(l, 0) + 1
    for layer in sorted(b_by_layer):
        a_cnt   = a_by_layer.get(layer, 0)
        add_cnt = add_by_layer.get(layer, 0)
        tot_cnt = b_by_layer[layer]
        pct     = tot_cnt / intermediate_dim * 100
        print(f"    {layer:>6}  {a_cnt:>8}  {add_cnt:>10}  {tot_cnt:>11}  {pct:>10.2f}%")

    # Build output CSV
    rows = []
    for (l, n) in sorted(model_b_set):
        rows.append({
            "layer":           l,
            "neuron":          n,
            "min_sigma_ratio": round(float(combined[l, n]), 4),
            "source":          "model_A" if (l, n) in a_set else "BE_phil_addition",
        })
    df_core = pd.DataFrame(rows)

    # Full audit table -- all candidates at 3-sigma+ in L18-27
    audit_rows = []
    for l in range(18, 28):
        for n in np.where(combined[l] >= 3.0)[0]:
            audit_rows.append({
                "layer":           l,
                "neuron":          int(n),
                "min_sigma_ratio": round(float(combined[l, n]),        4),
                "money_score":     round(float(money_scores[l, n]),    4),
                "reward_score":    round(float(reward_scores[l, n]),   4),
                "source":          "model_A" if (l, int(n)) in a_set else "candidate",
                "selected":        (l, int(n)) in model_b_set,
            })
    df_scores = pd.DataFrame(audit_rows)

    df_core.to_csv(OUTPUT_CORE,   index=False)
    df_scores.to_csv(OUTPUT_SCORES, index=False)

    print(f"\n  Saved:")
    print(f"    {OUTPUT_CORE}   ({len(df_core):,} rows)")
    print(f"    {OUTPUT_SCORES}  ({len(df_scores):,} rows)")

    print(f"\n{'='*60}")
    print(f"  DONE  --  Model B v2: {len(model_b_set):,} neurons")
    print(f"  Next : run run_full_experiment.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA,
                        help=f"Min-sigma threshold (default: {DEFAULT_SIGMA}). "
                             f"Use --dry_run first to see the sweep.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print sweep only, save nothing.")
    args = parser.parse_args()
    main(sigma_threshold=args.sigma, dry_run=args.dry_run)