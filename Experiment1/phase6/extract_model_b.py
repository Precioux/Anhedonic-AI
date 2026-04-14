"""
extract_neurons_modelB.py  —  Model B Neuron Selection
=======================================================
Model B = Model A neurons  +  neurons newly activated in business_ethics
          and philosophy that are NOT already in Model A.

Concretely:
  1. Load Model A neuron set (L18-27 subset of geo+math master core, 1,363 neurons)
  2. In business_ethics and philosophy, find neurons in L18-27 that exceed 3sigma
     for BOTH money AND reward conditions (same criterion as Model A selection)
  3. Take the union across the two new domains (active in BE OR phil)
  4. Remove any that are already in Model A (no redundancy)
  5. Model B = Model A  ∪  new neurons

This directly addresses the subset check finding:
  ~89-91% of active neurons in BE/phil were OUTSIDE Model A.
  Model B fills that gap.

OUTPUT:
    modelB_new_neurons.csv           — the neurons being added (not in Model A)
    modelB_master_incentive_core.csv — full Model B = Model A + new neurons
"""

import os
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================
ACTIVATIONS_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase6/activations"

# Model A neuron set — L18-27 subset of the geo+math master core
MODEL_A_NEURONS = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
MODEL_A_LAYERS  = list(range(18, 28))   # L18-27 inclusive

# Outputs — written next to this script
OUTPUT_NEW    = "modelB_new_neurons.csv"            # neurons added on top of Model A
OUTPUT_CORE   = "modelB_master_incentive_core.csv"  # full Model B = Model A + new

# =============================================================================
# Helpers
# =============================================================================
def load_activation_mean(domain, condition):
    fname = f"{condition}_activations_{domain}.pt"
    path  = os.path.join(ACTIVATIONS_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    print(f"  Loading {fname}...")
    data    = torch.load(path, map_location='cpu')
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()   # [Q, L, D]
    return stacked.mean(dim=0).numpy()       # [L, D]


def active_neurons_in_layers(delta, layers):
    """
    Return set of (layer, neuron) pairs that exceed 3sigma of the full delta array
    within the specified layers.
    """
    threshold = 3 * np.std(delta)
    result    = set()
    for layer in layers:
        active = np.where(np.abs(delta[layer]) > threshold)[0]
        for n in active:
            result.add((layer, int(n)))
    return result


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("MODEL B — NEURON SELECTION")
    print("Model A (geo+math, L18-27)  +  new neurons from BE & philosophy")
    print("=" * 60)

    # ── Step 1: Load Model A neuron set ──────────────────────────────────────
    df_a    = pd.read_csv(MODEL_A_NEURONS)
    df_a    = df_a[df_a['layer'].isin(MODEL_A_LAYERS)]
    model_a = set(zip(df_a['layer'], df_a['neuron']))
    print(f"\nModel A neurons (L18-27): {len(model_a):,}")
    for l in MODEL_A_LAYERS:
        n = sum(1 for ll, _ in model_a if ll == l)
        if n > 0:
            print(f"  L{l}: {n}")

    # ── Step 2: Load new domain activations and compute deltas ───────────────
    print("\nLoading new domain activations...")
    be_neu = load_activation_mean("business_ethics", "neutral")
    be_mon = load_activation_mean("business_ethics", "money")
    be_rew = load_activation_mean("business_ethics", "reward")

    ph_neu = load_activation_mean("philosophy", "neutral")
    ph_mon = load_activation_mean("philosophy", "money")
    ph_rew = load_activation_mean("philosophy", "reward")

    delta_be_money  = be_mon - be_neu
    delta_be_reward = be_rew - be_neu
    delta_ph_money  = ph_mon - ph_neu
    delta_ph_reward = ph_rew - ph_neu

    # ── Step 3: Find active neurons in L18-27 for each domain x condition ────
    print("\nFinding active neurons (>3sigma) in L18-27...")
    be_money_active  = active_neurons_in_layers(delta_be_money,  MODEL_A_LAYERS)
    be_reward_active = active_neurons_in_layers(delta_be_reward, MODEL_A_LAYERS)
    ph_money_active  = active_neurons_in_layers(delta_ph_money,  MODEL_A_LAYERS)
    ph_reward_active = active_neurons_in_layers(delta_ph_reward, MODEL_A_LAYERS)

    print(f"  BE   money  active: {len(be_money_active):,}")
    print(f"  BE   reward active: {len(be_reward_active):,}")
    print(f"  Phil money  active: {len(ph_money_active):,}")
    print(f"  Phil reward active: {len(ph_reward_active):,}")

    # ── Step 4: New neurons = active in BOTH conditions in at least one domain
    # A neuron qualifies if:
    #   (active in BE money AND BE reward)       <- responds to incentive framing in BE
    #   OR
    #   (active in phil money AND phil reward)   <- responds in philosophy
    # AND not already in Model A
    be_both   = be_money_active & be_reward_active    # both conditions in BE
    ph_both   = ph_money_active & ph_reward_active    # both conditions in phil
    new_union = be_both | ph_both                     # at least one new domain
    new_only  = new_union - model_a                   # exclude what Model A already covers

    print(f"\n  Active in both conditions (BE):   {len(be_both):,}")
    print(f"  Active in both conditions (phil): {len(ph_both):,}")
    print(f"  Union (BE or phil):               {len(new_union):,}")
    print(f"  Already in Model A:               {len(new_union & model_a):,}")
    print(f"  NEW neurons to add:               {len(new_only):,}")

    # ── Step 5: Model B = Model A + new neurons ───────────────────────────────
    model_b = model_a | new_only

    print(f"\n{'='*60}")
    print(f"MODEL B SUMMARY")
    print(f"{'='*60}")
    print(f"  Model A neurons:    {len(model_a):,}")
    print(f"  New neurons added:  {len(new_only):,}")
    print(f"  Model B total:      {len(model_b):,}")
    print(f"  % of network:       {len(model_b)/(28*18944)*100:.4f}%")

    # ── Layer distribution ────────────────────────────────────────────────────
    print(f"\nModel B layer distribution:")
    print(f"  {'Layer':<8} {'Model A':>8} {'New':>8} {'Total':>8}")
    print(f"  {'-'*36}")
    for l in MODEL_A_LAYERS:
        a_count   = sum(1 for ll, _ in model_a  if ll == l)
        new_count = sum(1 for ll, _ in new_only if ll == l)
        tot_count = sum(1 for ll, _ in model_b  if ll == l)
        bar = '█' * min(tot_count // 10, 40)
        print(f"  L{l:<7} {a_count:>8} {new_count:>8} {tot_count:>8}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    df_new  = pd.DataFrame(sorted(new_only), columns=['layer', 'neuron'])
    df_core = pd.DataFrame(sorted(model_b),  columns=['layer', 'neuron'])

    df_new.to_csv(OUTPUT_NEW,   index=False)
    df_core.to_csv(OUTPUT_CORE, index=False)

    print(f"\nSaved:")
    print(f"  {OUTPUT_NEW}   ({len(df_new):,} new neurons)")
    print(f"  {OUTPUT_CORE}  ({len(df_core):,} total neurons)")
    print(f"\nNext: run lesion_late_layers_modelB.py")


if __name__ == "__main__":
    main()