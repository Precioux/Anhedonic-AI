import os
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/activations/orig"
OUTPUT_DIR      = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/orig"

MATH_NEUTRAL = "neutral_activations_math.pt"
MATH_MONEY   = "money_activations_math.pt"
MATH_REWARD  = "reward_activations_math.pt"
GEO_NEUTRAL  = "neutral_activations_geo.pt"
GEO_MONEY    = "money_activations_geo.pt"
GEO_REWARD   = "reward_activations_geo.pt"

OUTPUT_MONEY  = os.path.join(OUTPUT_DIR, "universal_money_neurons.csv")
OUTPUT_REWARD = os.path.join(OUTPUT_DIR, "universal_reward_neurons.csv")
OUTPUT_CORE   = os.path.join(OUTPUT_DIR, "master_incentive_core.csv")

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
    stacked = torch.stack(tensors).float()  # [num_questions, num_layers, intermediate_dim]
    print(f"    Shape: {stacked.shape}  (questions x layers x intermediate_dim)")
    return stacked.mean(dim=0).numpy()      # [num_layers, intermediate_dim]


def find_universal_neurons_3sigma(delta_math, delta_geo):
    """
    Find MLP neurons significant (>3σ) in BOTH math and geography domains.
    delta shape: [num_layers, intermediate_dim]
    Returns set of (layer_idx, neuron_idx) tuples.
    layer_idx maps directly to model.model.language_model.layers[layer_idx].
    """
    threshold_math = 3 * np.std(delta_math)
    threshold_geo  = 3 * np.std(delta_geo)

    significant = np.where(
        (np.abs(delta_math) > threshold_math) &
        (np.abs(delta_geo)  > threshold_geo)
    )
    return set(zip(significant[0].tolist(), significant[1].tolist()))


# =============================================================================
# Main
# =============================================================================
print("=" * 60)
print("EXTRACTING 72B UNIVERSAL MLP NEURONS (3-Sigma Cross-Domain)")
print("=" * 60)

# 1. Load activations
print("\nLoading Math Activations...")
m_neu = load_activation_mean(MATH_NEUTRAL)
m_mon = load_activation_mean(MATH_MONEY)
m_rew = load_activation_mean(MATH_REWARD)

print("\nLoading Geography Activations...")
g_neu = load_activation_mean(GEO_NEUTRAL)
g_mon = load_activation_mean(GEO_MONEY)
g_rew = load_activation_mean(GEO_REWARD)

# Sanity check
shapes = {m_neu.shape, m_mon.shape, m_rew.shape, g_neu.shape, g_mon.shape, g_rew.shape}
assert len(shapes) == 1, f"Shape mismatch: {shapes}"
num_layers, intermediate_dim = m_neu.shape
print(f"\nAll arrays: {num_layers} layers x {intermediate_dim} intermediate neurons")

# 2. Deltas
print("\nCalculating Deltas...")
delta_mon_math = m_mon - m_neu
delta_mon_geo  = g_mon - g_neu
delta_rew_math = m_rew - m_neu
delta_rew_geo  = g_rew - g_neu

# 3. Find universal neurons
print("\nFinding Universal Money Neurons (3σ in both Math & Geo)...")
money_universal = find_universal_neurons_3sigma(delta_mon_math, delta_mon_geo)
print(f"  -> Found {len(money_universal)} Universal Money Neurons")

print("\nFinding Universal Reward Neurons (3σ in both Math & Geo)...")
reward_universal = find_universal_neurons_3sigma(delta_rew_math, delta_rew_geo)
print(f"  -> Found {len(reward_universal)} Universal Reward Neurons")

# 4. Core = intersection
master_core = money_universal & reward_universal
print(f"\nMaster Core (Money ∩ Reward): {len(master_core)} neurons")
if money_universal:
    print(f"  Overlap: {len(master_core)/len(money_universal)*100:.1f}% of money, "
          f"{len(master_core)/len(reward_universal)*100:.1f}% of reward")

# 5. Save CSVs
df_money  = pd.DataFrame(sorted(money_universal),  columns=['layer', 'neuron'])
df_reward = pd.DataFrame(sorted(reward_universal), columns=['layer', 'neuron'])
df_core   = pd.DataFrame(sorted(master_core),      columns=['layer', 'neuron'])

df_money.to_csv(OUTPUT_MONEY,   index=False)
df_reward.to_csv(OUTPUT_REWARD, index=False)
df_core.to_csv(OUTPUT_CORE,     index=False)

print(f"\nSaved:")
print(f"  {OUTPUT_MONEY}  ({len(df_money)} rows)")
print(f"  {OUTPUT_REWARD} ({len(df_reward)} rows)")
print(f"  {OUTPUT_CORE}   ({len(df_core)} rows)")

# 6. Summary
print(f"\n{'=' * 60}")
print("SUMMARY")
print(f"{'=' * 60}")
print(f"Layers:           {num_layers}")
print(f"Intermediate dim: {intermediate_dim}")
print(f"Universal Money:  {len(money_universal):>6} (layer, neuron) pairs")
print(f"Universal Reward: {len(reward_universal):>6} (layer, neuron) pairs")
print(f"Master Core:      {len(master_core):>6} (layer, neuron) pairs")

# Layer distribution of core
core_layers = [l for l, n in master_core]
if core_layers:
    print(f"\nMaster Core layer distribution:")
    for layer in sorted(set(core_layers)):
        count = core_layers.count(layer)
        bar   = '█' * min(count, 50)
        print(f"  Layer {layer:>2}: {count:>5} neurons  {bar}")

# Delta magnitude check
print(f"\nDelta magnitude check (mean |delta|):")
print(f"  Money  / Math: {np.abs(delta_mon_math).mean():.6f}")
print(f"  Money  / Geo:  {np.abs(delta_mon_geo).mean():.6f}")
print(f"  Reward / Math: {np.abs(delta_rew_math).mean():.6f}")
print(f"  Reward / Geo:  {np.abs(delta_rew_geo).mean():.6f}")

# 7. Comparison with 7B results
print(f"\n{'=' * 60}")
print("COMPARISON: 7B vs 72B orig core")
print(f"{'=' * 60}")
path_7b = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
if os.path.exists(path_7b):
    df_7b   = pd.read_csv(path_7b)
    set_7b  = set(zip(df_7b['layer'].tolist(), df_7b['neuron'].tolist()))
    inter   = set_7b & master_core
    union   = set_7b | master_core
    jaccard = len(inter) / len(union) if union else 0
    print(f"  7B core:   {len(set_7b):>6} neurons")
    print(f"  72B core:  {len(master_core):>6} neurons")
    print(f"  Overlap:   {len(inter):>6} neurons")
    print(f"  Jaccard:   {jaccard:.4f}")
    print(f"  Note: neuron indices are NOT comparable across model sizes.")
    print(f"        Overlap here only reflects coincidental index matches.")
else:
    print(f"  7B core not found at {path_7b} — skipping comparison.")