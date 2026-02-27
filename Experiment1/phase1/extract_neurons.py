import os
import torch
import numpy as np
import pandas as pd

# --- Configuration ---
BASE_PATH = "../phase1"
ACTIVATIONS_DIR = os.path.join(BASE_PATH, "activations")

# File Names
MATH_NEUTRAL = "neutral_activations_v2.pt"
MATH_MONEY   = "money_activations_v2.pt"
MATH_REWARD  = "reward_activations_v2.pt"
GEO_NEUTRAL  = "neutral_activations_geo.pt"
GEO_MONEY    = "money_activations_geo.pt"
GEO_REWARD   = "reward_activations_geo.pt"

# Outputs
OUTPUT_MONEY  = "universal_money_neurons.csv"
OUTPUT_REWARD = "universal_reward_neurons.csv"
OUTPUT_CORE   = "master_incentive_core.csv"

def load_activation_mean(filename):
    """Load .pt file and return mean across questions. Shape: [layers, hidden_dim]"""
    paths_to_check = [
        os.path.join(BASE_PATH, filename),
        os.path.join(ACTIVATIONS_DIR, filename),
        filename
    ]
    
    found_path = None
    for p in paths_to_check:
        if os.path.exists(p):
            found_path = p
            break
    
    if not found_path:
        raise FileNotFoundError(f"Could not find {filename} in search paths.")
    
    print(f"  Loading {found_path}...")
    data = torch.load(found_path, map_location='cpu')
    
    tensors = []
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            tensors.append(v)
    
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")
    
    # Shape: [num_questions, num_layers, hidden_dim] -> mean -> [num_layers, hidden_dim]
    return torch.stack(tensors).float().mean(dim=0).numpy()

def find_universal_neurons_3sigma(delta_math, delta_geo):
    """
    Find neurons that are significant (>3σ) in BOTH math and geography domains.
    Returns a set of (layer, neuron) tuples.
    """
    threshold_math = 3 * np.std(delta_math)
    threshold_geo  = 3 * np.std(delta_geo)
    
    # Neurons significant in both domains simultaneously
    significant = np.where(
        (np.abs(delta_math) > threshold_math) & 
        (np.abs(delta_geo) > threshold_geo)
    )
    
    return set(zip(significant[0], significant[1]))

def main():
    print("=" * 60)
    print("EXTRACTING UNIVERSAL NEURONS (3-Sigma Cross-Domain Method)")
    print("=" * 60)
    
    # 1. Load all activations
    print("\nLoading Math Activations...")
    m_neu = load_activation_mean(MATH_NEUTRAL)
    m_mon = load_activation_mean(MATH_MONEY)
    m_rew = load_activation_mean(MATH_REWARD)
    
    print("\nLoading Geography Activations...")
    g_neu = load_activation_mean(GEO_NEUTRAL)
    g_mon = load_activation_mean(GEO_MONEY)
    g_rew = load_activation_mean(GEO_REWARD)
    
    # 2. Calculate deltas (condition - neutral)
    print("\nCalculating Deltas...")
    delta_mon_math = m_mon - m_neu
    delta_mon_geo  = g_mon - g_neu
    delta_rew_math = m_rew - m_neu
    delta_rew_geo  = g_rew - g_neu
    
    # 3. Find Universal Neurons using 3σ cross-domain criterion
    print("\nFinding Universal Money Neurons (3σ in both Math & Geo)...")
    money_universal = find_universal_neurons_3sigma(delta_mon_math, delta_mon_geo)
    print(f"  -> Found {len(money_universal)} Universal Money Neurons")
    
    print("\nFinding Universal Reward Neurons (3σ in both Math & Geo)...")
    reward_universal = find_universal_neurons_3sigma(delta_rew_math, delta_rew_geo)
    print(f"  -> Found {len(reward_universal)} Universal Reward Neurons")
    
    # 4. Find Master Core (intersection of money & reward)
    master_core = money_universal & reward_universal
    print(f"\nMaster Core (Money ∩ Reward): {len(master_core)} neurons")
    print(f"  Overlap: {len(master_core)/len(money_universal)*100:.1f}% of money, "
          f"{len(master_core)/len(reward_universal)*100:.1f}% of reward")
    
    # 5. Save with (Layer, Neuron) pairs
    df_money = pd.DataFrame(sorted(money_universal), columns=['layer', 'neuron'])
    df_reward = pd.DataFrame(sorted(reward_universal), columns=['layer', 'neuron'])
    df_core = pd.DataFrame(sorted(master_core), columns=['layer', 'neuron'])
    
    df_money.to_csv(OUTPUT_MONEY, index=False)
    df_reward.to_csv(OUTPUT_REWARD, index=False)
    df_core.to_csv(OUTPUT_CORE, index=False)
    
    print(f"\nSaved:")
    print(f"  {OUTPUT_MONEY}  ({len(df_money)} rows)")
    print(f"  {OUTPUT_REWARD} ({len(df_reward)} rows)")
    print(f"  {OUTPUT_CORE}   ({len(df_core)} rows)")
    
    # 6. Summary stats
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"Hidden dim: {delta_mon_math.shape[1]}")
    print(f"Num layers: {delta_mon_math.shape[0]}")
    print(f"Universal Money:  {len(money_universal):>5} (layer, neuron) pairs")
    print(f"Universal Reward: {len(reward_universal):>5} (layer, neuron) pairs")
    print(f"Master Core:      {len(master_core):>5} (layer, neuron) pairs")
    
    # Layer distribution of master core
    core_layers = [l for l, n in master_core]
    print(f"\nMaster Core layer distribution:")
    for layer in sorted(set(core_layers)):
        count = core_layers.count(layer)
        print(f"  Layer {layer:>2}: {count:>3} neurons {'█' * count}")

if __name__ == "__main__":
    main()