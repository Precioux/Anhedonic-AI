import os
import torch
import numpy as np
import pandas as pd

# --- Configuration ---
# All .pt files live in the activations/ subfolder next to this script
ACTIVATIONS_DIR = "activations"

# Original domains
MATH_NEUTRAL = "neutral_activations_math.pt"
MATH_MONEY   = "money_activations_math.pt"
MATH_REWARD  = "reward_activations_math.pt"
GEO_NEUTRAL  = "neutral_activations_geo.pt"
GEO_MONEY    = "money_activations_geo.pt"
GEO_REWARD   = "reward_activations_geo.pt"

# New domains
BE_NEUTRAL   = "neutral_activations_business_ethics.pt"
BE_MONEY     = "money_activations_business_ethics.pt"
BE_REWARD    = "reward_activations_business_ethics.pt"
PHIL_NEUTRAL = "neutral_activations_philosophy.pt"
PHIL_MONEY   = "money_activations_philosophy.pt"
PHIL_REWARD  = "reward_activations_philosophy.pt"

# Output CSVs saved next to this script — ablation script reads them from here
OUTPUT_MONEY  = "universal_money_neurons.csv"
OUTPUT_REWARD = "universal_reward_neurons.csv"
OUTPUT_CORE   = "master_incentive_core.csv"

# -------------------------------------------------------------------------
# IMPORTANT — What's in these .pt files now:
#
# Each file maps question IDs to tensors of shape [num_layers, intermediate_dim].
# These are MLP intermediate activations captured at model.model.layers[i].mlp.act_fn,
# NOT residual stream hidden states. This means:
#   - Index 0  = MLP neuron activations for transformer layer 0
#   - Index 1  = MLP neuron activations for transformer layer 1
#   - ...
#   - Index 27 = MLP neuron activations for transformer layer 27
#
# There is NO embedding layer offset here (unlike the old output_hidden_states approach
# which had an embedding at index 0, shifting all layer indices by +1).
# So layer_idx directly maps to model.model.layers[layer_idx] — no subtraction needed.
# -------------------------------------------------------------------------


def load_activation_mean(filename):
    """Load .pt file and return mean across questions. Shape: [num_layers, intermediate_dim]"""
    found_path = os.path.join(ACTIVATIONS_DIR, filename)

    if not os.path.exists(found_path):
        raise FileNotFoundError(
            f"Could not find {found_path}\n"
            f"Make sure extract_activations.py has been run and files are in {ACTIVATIONS_DIR}/"
        )

    print(f"  Loading {found_path}...")
    data = torch.load(found_path, map_location='cpu')

    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")

    stacked = torch.stack(tensors).float()  # [num_questions, num_layers, intermediate_dim]
    print(f"    Shape: {stacked.shape}  (questions x layers x intermediate_dim)")

    return stacked.mean(dim=0).numpy()  # [num_layers, intermediate_dim]


def find_universal_neurons_3sigma(*deltas):
    """
    Find MLP neurons that are significant (>3σ) across ALL provided delta arrays.

    Each delta has shape [num_layers, intermediate_dim].

    Returns a set of (layer_idx, neuron_idx) tuples where layer_idx maps
    DIRECTLY to model.model.layers[layer_idx] — no offset needed.
    """
    masks = []
    for delta in deltas:
        threshold = 3 * np.std(delta)
        masks.append(np.abs(delta) > threshold)

    # Neuron must be significant in every domain
    combined = np.ones_like(masks[0], dtype=bool)
    for mask in masks:
        combined &= mask

    significant = np.where(combined)
    pairs = set(zip(significant[0].tolist(), significant[1].tolist()))
    return pairs


def main():
    print("=" * 60)
    print("EXTRACTING UNIVERSAL MLP NEURONS (3-Sigma Cross-Domain)")
    print("Domains: Math, Geography, Business Ethics, Philosophy")
    print("=" * 60)

    # 1. Load all activations
    print("\nLoading Math Activations...")
    m_neu  = load_activation_mean(MATH_NEUTRAL)
    m_mon  = load_activation_mean(MATH_MONEY)
    m_rew  = load_activation_mean(MATH_REWARD)

    print("\nLoading Geography Activations...")
    g_neu  = load_activation_mean(GEO_NEUTRAL)
    g_mon  = load_activation_mean(GEO_MONEY)
    g_rew  = load_activation_mean(GEO_REWARD)

    print("\nLoading Business Ethics Activations...")
    be_neu = load_activation_mean(BE_NEUTRAL)
    be_mon = load_activation_mean(BE_MONEY)
    be_rew = load_activation_mean(BE_REWARD)

    print("\nLoading Philosophy Activations...")
    ph_neu = load_activation_mean(PHIL_NEUTRAL)
    ph_mon = load_activation_mean(PHIL_MONEY)
    ph_rew = load_activation_mean(PHIL_REWARD)

    # Sanity check — all arrays must have the same shape
    all_arrays = [m_neu, m_mon, m_rew, g_neu, g_mon, g_rew,
                  be_neu, be_mon, be_rew, ph_neu, ph_mon, ph_rew]
    shapes = {a.shape for a in all_arrays}
    assert len(shapes) == 1, f"Shape mismatch across activation files: {shapes}"
    num_layers, intermediate_dim = m_neu.shape
    print(f"\nAll activation arrays: {num_layers} layers x {intermediate_dim} intermediate neurons")

    # 2. Calculate deltas (condition - neutral) per domain
    print("\nCalculating Deltas...")
    delta_mon_math = m_mon  - m_neu
    delta_mon_geo  = g_mon  - g_neu
    delta_mon_be   = be_mon - be_neu
    delta_mon_phil = ph_mon - ph_neu

    delta_rew_math = m_rew  - m_neu
    delta_rew_geo  = g_rew  - g_neu
    delta_rew_be   = be_rew - be_neu
    delta_rew_phil = ph_rew - ph_neu

    # 3. Find Universal Neurons — must be >3σ in ALL four domains
    print("\nFinding Universal Money Neurons (3σ in Math, Geo, Business Ethics & Philosophy)...")
    money_universal = find_universal_neurons_3sigma(
        delta_mon_math, delta_mon_geo, delta_mon_be, delta_mon_phil
    )
    print(f"  -> Found {len(money_universal)} Universal Money Neurons")

    print("\nFinding Universal Reward Neurons (3σ in Math, Geo, Business Ethics & Philosophy)...")
    reward_universal = find_universal_neurons_3sigma(
        delta_rew_math, delta_rew_geo, delta_rew_be, delta_rew_phil
    )
    print(f"  -> Found {len(reward_universal)} Universal Reward Neurons")

    # 4. Master Core = intersection
    master_core = money_universal & reward_universal
    print(f"\nMaster Core (Money ∩ Reward): {len(master_core)} neurons")
    if money_universal and reward_universal:
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
    print(f"Universal Money:  {len(money_universal):>5} (layer, neuron) pairs")
    print(f"Universal Reward: {len(reward_universal):>5} (layer, neuron) pairs")
    print(f"Master Core:      {len(master_core):>5} (layer, neuron) pairs")

    core_layers = [l for l, n in master_core]
    if core_layers:
        print(f"\nMaster Core layer distribution:")
        for layer in sorted(set(core_layers)):
            count = core_layers.count(layer)
            bar = '█' * min(count, 40)
            print(f"  Layer {layer:>2}: {count:>4} neurons  {bar}")

    # 7. Quick delta magnitude check — helps diagnose weak signal
    print(f"\nDelta magnitude check (mean |delta| per condition × domain):")
    print(f"  Money  / Math:            {np.abs(delta_mon_math).mean():.6f}")
    print(f"  Money  / Geo:             {np.abs(delta_mon_geo).mean():.6f}")
    print(f"  Money  / Business Ethics: {np.abs(delta_mon_be).mean():.6f}")
    print(f"  Money  / Philosophy:      {np.abs(delta_mon_phil).mean():.6f}")
    print(f"  Reward / Math:            {np.abs(delta_rew_math).mean():.6f}")
    print(f"  Reward / Geo:             {np.abs(delta_rew_geo).mean():.6f}")
    print(f"  Reward / Business Ethics: {np.abs(delta_rew_be).mean():.6f}")
    print(f"  Reward / Philosophy:      {np.abs(delta_rew_phil).mean():.6f}")
    print("  (If all values are near zero, the prompts are not creating "
          "distinguishable MLP activations — revisit prompt design.)")


if __name__ == "__main__":
    main()
