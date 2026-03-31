import os
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/activations_72b"

MATH_NEUTRAL = "neutral_activations_math.pt"
MATH_MONEY   = "money_activations_math.pt"
MATH_REWARD  = "reward_activations_math.pt"
GEO_NEUTRAL  = "neutral_activations_geo.pt"
GEO_MONEY    = "money_activations_geo.pt"
GEO_REWARD   = "reward_activations_geo.pt"

OUTPUT_MONEY  = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/universal_money_neurons_72b.csv"
OUTPUT_REWARD = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/universal_reward_neurons_72b.csv"
OUTPUT_CORE   = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/master_incentive_core_72b.csv"

# -------------------------------------------------------------------------
# NOTE: Each .pt file maps question IDs to tensors [num_layers, intermediate_dim].
# These are MLP intermediate activations from language_model.layers[i].mlp.act_fn.
# layer_idx maps directly to language_model.layers[layer_idx] — no offset needed.
# -------------------------------------------------------------------------


def load_activation_mean(filename):
    """Load .pt file and return mean across questions. Shape: [num_layers, intermediate_dim]"""
    found_path = os.path.join(ACTIVATIONS_DIR, filename)
    if not os.path.exists(found_path):
        raise FileNotFoundError(
            f"Could not find {found_path}\n"
            f"Make sure extract_activations.py has been run first."
        )
    print(f"  Loading {found_path}...")
    data = torch.load(found_path, map_location='cpu')
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")
    stacked = torch.stack(tensors).float()  # [num_questions, num_layers, intermediate_dim]
    print(f"    Shape: {stacked.shape}  (questions x layers x intermediate_dim)")
    return stacked.mean(dim=0).numpy()      # [num_layers, intermediate_dim]


def find_universal_neurons_3sigma(delta_math, delta_geo):
    """
    Find MLP neurons significant (>3σ) in BOTH math and geography domains.
    Returns set of (layer_idx, neuron_idx) tuples.
    """
    threshold_math = 3 * np.std(delta_math)
    threshold_geo  = 3 * np.std(delta_geo)
    significant = np.where(
        (np.abs(delta_math) > threshold_math) &
        (np.abs(delta_geo)  > threshold_geo)
    )
    return set(zip(significant[0].tolist(), significant[1].tolist()))


def main():
    print("=" * 60)
    print("EXTRACTING UNIVERSAL MLP NEURONS — Qwen2-VL-72B")
    print("=" * 60)

    print("\nLoading Math Activations...")
    m_neu = load_activation_mean(MATH_NEUTRAL)
    m_mon = load_activation_mean(MATH_MONEY)
    m_rew = load_activation_mean(MATH_REWARD)

    print("\nLoading Geography Activations...")
    g_neu = load_activation_mean(GEO_NEUTRAL)
    g_mon = load_activation_mean(GEO_MONEY)
    g_rew = load_activation_mean(GEO_REWARD)

    shapes = {m_neu.shape, m_mon.shape, m_rew.shape, g_neu.shape, g_mon.shape, g_rew.shape}
    assert len(shapes) == 1, f"Shape mismatch across activation files: {shapes}"
    num_layers, intermediate_dim = m_neu.shape
    print(f"\nAll activation arrays: {num_layers} layers x {intermediate_dim} intermediate neurons")

    print("\nCalculating Deltas...")
    delta_mon_math = m_mon - m_neu
    delta_mon_geo  = g_mon - g_neu
    delta_rew_math = m_rew - m_neu
    delta_rew_geo  = g_rew - g_neu

    print("\nFinding Universal Money Neurons (3σ in both Math & Geo)...")
    money_universal = find_universal_neurons_3sigma(delta_mon_math, delta_mon_geo)
    print(f"  -> Found {len(money_universal)} Universal Money Neurons")

    print("\nFinding Universal Reward Neurons (3σ in both Math & Geo)...")
    reward_universal = find_universal_neurons_3sigma(delta_rew_math, delta_rew_geo)
    print(f"  -> Found {len(reward_universal)} Universal Reward Neurons")

    master_core = money_universal & reward_universal
    print(f"\nMaster Core (Money ∩ Reward): {len(master_core)} neurons")
    if money_universal and reward_universal:
        print(f"  Overlap: {len(master_core)/len(money_universal)*100:.1f}% of money, "
              f"{len(master_core)/len(reward_universal)*100:.1f}% of reward")

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

    print(f"\nDelta magnitude check (mean |delta| per condition):")
    print(f"  Money  / Math: {np.abs(delta_mon_math).mean():.6f}")
    print(f"  Money  / Geo:  {np.abs(delta_mon_geo).mean():.6f}")
    print(f"  Reward / Math: {np.abs(delta_rew_math).mean():.6f}")
    print(f"  Reward / Geo:  {np.abs(delta_rew_geo).mean():.6f}")
    print("  (If all values are near zero, prompts are not creating "
          "distinguishable MLP activations — revisit prompt design.)")


if __name__ == "__main__":
    main()
