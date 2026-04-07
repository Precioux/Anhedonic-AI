import os
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/ASDiv/activations"
OUTPUT_DIR      = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/ASDiv/neurons"

ASDIV_NEUTRAL = "neutral_activations_asdiv.pt"
ASDIV_MONEY   = "money_activations_asdiv.pt"
ASDIV_REWARD  = "reward_activations_asdiv.pt"

OUTPUT_MONEY  = os.path.join(OUTPUT_DIR, "asdiv_universal_money_neurons.csv")
OUTPUT_REWARD = os.path.join(OUTPUT_DIR, "asdiv_universal_reward_neurons.csv")
OUTPUT_CORE   = os.path.join(OUTPUT_DIR, "asdiv_master_incentive_core.csv")

# 72B orig core for overlap comparison
ORIGINAL_CORE = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/orig/master_incentive_core.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Helpers
# =============================================================================
def load_activation_mean(filename):
    """Load .pt file and return mean across questions. Shape: [num_layers, intermediate_dim]"""
    path = os.path.join(ACTIVATIONS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    print(f"  Loading {path}...")
    data    = torch.load(path, map_location='cpu')
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()
    print(f"    Shape: {stacked.shape}  (questions x layers x intermediate_dim)")
    return stacked.mean(dim=0).numpy()


def find_universal_neurons_3sigma(delta_a, delta_b):
    """
    Single-domain 3σ — pass delta as both arguments.
    Returns set of (layer_idx, neuron_idx) tuples.
    """
    threshold_a = 3 * np.std(delta_a)
    threshold_b = 3 * np.std(delta_b)
    significant = np.where(
        (np.abs(delta_a) > threshold_a) &
        (np.abs(delta_b) > threshold_b)
    )
    return set(zip(significant[0].tolist(), significant[1].tolist()))


# =============================================================================
# Main
# =============================================================================
print("=" * 60)
print("EXTRACTING 72B ASDIV MLP NEURONS (3-Sigma)")
print("=" * 60)

# 1. Load activations
print("\nLoading ASDiv Activations...")
a_neu = load_activation_mean(ASDIV_NEUTRAL)
a_mon = load_activation_mean(ASDIV_MONEY)
a_rew = load_activation_mean(ASDIV_REWARD)

shapes = {a_neu.shape, a_mon.shape, a_rew.shape}
assert len(shapes) == 1, f"Shape mismatch: {shapes}"
num_layers, intermediate_dim = a_neu.shape
print(f"\nAll arrays: {num_layers} layers x {intermediate_dim} intermediate neurons")

# 2. Deltas
print("\nCalculating Deltas...")
delta_mon = a_mon - a_neu
delta_rew = a_rew - a_neu

# 3. Single-domain 3σ
print("\nFinding ASDiv Money Neurons (3σ)...")
money_neurons = find_universal_neurons_3sigma(delta_mon, delta_mon)
print(f"  -> Found {len(money_neurons)} Money Neurons")

print("\nFinding ASDiv Reward Neurons (3σ)...")
reward_neurons = find_universal_neurons_3sigma(delta_rew, delta_rew)
print(f"  -> Found {len(reward_neurons)} Reward Neurons")

# 4. Core
asdiv_core = money_neurons & reward_neurons
print(f"\nASDiv Core (Money ∩ Reward): {len(asdiv_core)} neurons")
if money_neurons:
    print(f"  Overlap: {len(asdiv_core)/len(money_neurons)*100:.1f}% of money, "
          f"{len(asdiv_core)/len(reward_neurons)*100:.1f}% of reward")

# 5. Save
df_money  = pd.DataFrame(sorted(money_neurons),  columns=['layer', 'neuron'])
df_reward = pd.DataFrame(sorted(reward_neurons), columns=['layer', 'neuron'])
df_core   = pd.DataFrame(sorted(asdiv_core),     columns=['layer', 'neuron'])

df_money.to_csv(OUTPUT_MONEY,   index=False)
df_reward.to_csv(OUTPUT_REWARD, index=False)
df_core.to_csv(OUTPUT_CORE,     index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_MONEY}  ({len(df_money)} rows)")
print(f"  {OUTPUT_REWARD} ({len(df_reward)} rows)")
print(f"  {OUTPUT_CORE}   ({len(df_core)} rows)")

# 6. Overlap with 72B orig core
print(f"\n{'=' * 60}")
print("OVERLAP WITH 72B ORIG master_incentive_core.csv")
print(f"{'=' * 60}")

if not os.path.exists(ORIGINAL_CORE):
    print(f"  {ORIGINAL_CORE} not found — skipping overlap analysis.")
else:
    orig_df   = pd.read_csv(ORIGINAL_CORE)
    orig_set  = set(zip(orig_df['layer'].tolist(), orig_df['neuron'].tolist()))
    asdiv_set = asdiv_core

    intersection = orig_set & asdiv_set
    union        = orig_set | asdiv_set
    jaccard      = len(intersection) / len(union) if union else 0.0
    recall       = len(intersection) / len(orig_set)  if orig_set  else 0.0
    precision    = len(intersection) / len(asdiv_set) if asdiv_set else 0.0

    print(f"  72B Orig core:   {len(orig_set):>6} neurons")
    print(f"  72B ASDiv core:  {len(asdiv_set):>6} neurons")
    print(f"  Overlap:         {len(intersection):>6} neurons")
    print(f"  Jaccard:         {jaccard:.4f}")
    print(f"  Recall of orig:  {recall:.4f}  ({len(intersection)}/{len(orig_set)} recovered)")
    print(f"  Precision:       {precision:.4f}  ({len(intersection)}/{len(asdiv_set)} in orig)")

# 7. Summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"Layers:           {num_layers}")
print(f"Intermediate dim: {intermediate_dim}")
print(f"ASDiv Money:      {len(money_neurons):>6} (layer, neuron) pairs")
print(f"ASDiv Reward:     {len(reward_neurons):>6} (layer, neuron) pairs")
print(f"ASDiv Core:       {len(asdiv_core):>6} (layer, neuron) pairs")

core_layers = [l for l, n in asdiv_core]
if core_layers:
    print(f"\nASDiv Core layer distribution:")
    for layer in sorted(set(core_layers)):
        count = core_layers.count(layer)
        bar   = '█' * min(count, 50)
        print(f"  Layer {layer:>2}: {count:>5} neurons  {bar}")

print(f"\nDelta magnitude check (mean |delta|):")
print(f"  Money  / ASDiv: {np.abs(delta_mon).mean():.6f}")
print(f"  Reward / ASDiv: {np.abs(delta_rew).mean():.6f}")