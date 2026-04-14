import os
import torch
import numpy as np
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
# All 12 .pt files are in activations_modelB/
# (geo + math are symlinks to phase4; business_ethics + philosophy are new)
ACTIVATIONS_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase6/activations"

# geo + math (phase 4, already extracted)
GEO_NEUTRAL  = "neutral_activations_geo.pt"
GEO_MONEY    = "money_activations_geo.pt"
GEO_REWARD   = "reward_activations_geo.pt"
MATH_NEUTRAL = "neutral_activations_math.pt"
MATH_MONEY   = "money_activations_math.pt"
MATH_REWARD  = "reward_activations_math.pt"

# business_ethics + philosophy (newly extracted)
BE_NEUTRAL   = "neutral_activations_business_ethics.pt"
BE_MONEY     = "money_activations_business_ethics.pt"
BE_REWARD    = "reward_activations_business_ethics.pt"
PHIL_NEUTRAL = "neutral_activations_philosophy.pt"
PHIL_MONEY   = "money_activations_philosophy.pt"
PHIL_REWARD  = "reward_activations_philosophy.pt"

# Outputs — prefixed with modelB_ to avoid collision with Model A files
OUTPUT_MONEY  = "modelB_universal_money_neurons.csv"
OUTPUT_REWARD = "modelB_universal_reward_neurons.csv"
OUTPUT_CORE   = "modelB_master_incentive_core.csv"

MODEL_A_CORE  = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"

# -------------------------------------------------------------------------
# Same indexing convention as Model A:
#   layer_idx maps directly to model.model.layers[layer_idx] — no offset.
# -------------------------------------------------------------------------


def load_activation_mean(filename):
    path = os.path.join(ACTIVATIONS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}\n"
            f"Run extract_activations_modelB.py first."
        )
    print(f"  Loading {path}...")
    data    = torch.load(path, map_location='cpu')
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()   # [Q, L, D]
    print(f"    Shape: {stacked.shape}  (questions × layers × intermediate_dim)")
    return stacked.mean(dim=0).numpy()       # [L, D]


def find_universal_neurons_3sigma(*deltas):
    """
    Neuron must exceed 3σ of its OWN delta array in ALL supplied domains.
    Returns set of (layer_idx, neuron_idx) tuples.
    """
    combined = np.ones(deltas[0].shape, dtype=bool)
    for delta in deltas:
        threshold = 3 * np.std(delta)
        combined &= (np.abs(delta) > threshold)
    rows, cols = np.where(combined)
    return set(zip(rows.tolist(), cols.tolist()))


def main():
    print("=" * 60)
    print("MODEL B — NEURON SELECTION")
    print("3σ intersection across 4 domains:")
    print("  geo ∩ math ∩ business_ethics ∩ philosophy")
    print("=" * 60)

    # ── Load all 12 activation means ─────────────────────────────────────────
    print("\nLoading geo activations...")
    g_neu = load_activation_mean(GEO_NEUTRAL)
    g_mon = load_activation_mean(GEO_MONEY)
    g_rew = load_activation_mean(GEO_REWARD)

    print("\nLoading math activations...")
    m_neu = load_activation_mean(MATH_NEUTRAL)
    m_mon = load_activation_mean(MATH_MONEY)
    m_rew = load_activation_mean(MATH_REWARD)

    print("\nLoading business_ethics activations...")
    be_neu = load_activation_mean(BE_NEUTRAL)
    be_mon = load_activation_mean(BE_MONEY)
    be_rew = load_activation_mean(BE_REWARD)

    print("\nLoading philosophy activations...")
    ph_neu = load_activation_mean(PHIL_NEUTRAL)
    ph_mon = load_activation_mean(PHIL_MONEY)
    ph_rew = load_activation_mean(PHIL_REWARD)

    # Sanity check shapes
    all_arrays = [g_neu, g_mon, g_rew, m_neu, m_mon, m_rew,
                  be_neu, be_mon, be_rew, ph_neu, ph_mon, ph_rew]
    shapes = {a.shape for a in all_arrays}
    assert len(shapes) == 1, f"Shape mismatch: {shapes}"
    num_layers, intermediate_dim = g_neu.shape
    print(f"\nAll arrays: {num_layers} layers × {intermediate_dim} intermediate neurons")
    print(f"Total MLP neurons: {num_layers * intermediate_dim:,}")

    # ── Deltas ────────────────────────────────────────────────────────────────
    print("\nComputing deltas (condition − neutral)...")
    delta_mon_geo  = g_mon  - g_neu
    delta_mon_math = m_mon  - m_neu
    delta_mon_be   = be_mon - be_neu
    delta_mon_phil = ph_mon - ph_neu

    delta_rew_geo  = g_rew  - g_neu
    delta_rew_math = m_rew  - m_neu
    delta_rew_be   = be_rew - be_neu
    delta_rew_phil = ph_rew - ph_neu

    # ── 3σ intersection across all 4 domains ─────────────────────────────────
    print("\nFinding Model B universal money neurons (3σ in all 4 domains)...")
    money_universal = find_universal_neurons_3sigma(
        delta_mon_geo, delta_mon_math, delta_mon_be, delta_mon_phil
    )
    print(f"  → {len(money_universal)} universal money neurons")

    print("\nFinding Model B universal reward neurons (3σ in all 4 domains)...")
    reward_universal = find_universal_neurons_3sigma(
        delta_rew_geo, delta_rew_math, delta_rew_be, delta_rew_phil
    )
    print(f"  → {len(reward_universal)} universal reward neurons")

    # ── Master core = money ∩ reward ──────────────────────────────────────────
    master_core = money_universal & reward_universal
    print(f"\nModel B Master Core (money ∩ reward): {len(master_core)} neurons")
    if money_universal and reward_universal:
        print(f"  {len(master_core)/len(money_universal)*100:.1f}% of money set")
        print(f"  {len(master_core)/len(reward_universal)*100:.1f}% of reward set")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_money  = pd.DataFrame(sorted(money_universal),  columns=['layer', 'neuron'])
    df_reward = pd.DataFrame(sorted(reward_universal), columns=['layer', 'neuron'])
    df_core   = pd.DataFrame(sorted(master_core),      columns=['layer', 'neuron'])

    df_money.to_csv(OUTPUT_MONEY,   index=False)
    df_reward.to_csv(OUTPUT_REWARD, index=False)
    df_core.to_csv(OUTPUT_CORE,     index=False)

    print(f"\nSaved:")
    print(f"  {OUTPUT_MONEY}   ({len(df_money)} rows)")
    print(f"  {OUTPUT_REWARD}  ({len(df_reward)} rows)")
    print(f"  {OUTPUT_CORE}    ({len(df_core)} rows)")

    # ── Layer distribution ────────────────────────────────────────────────────
    print(f"\nModel B Master Core — layer distribution:")
    core_layers = [l for l, n in master_core]
    for layer in sorted(set(core_layers)):
        count = core_layers.count(layer)
        bar   = '█' * min(count, 40)
        print(f"  Layer {layer:>2}: {count:>4} neurons  {bar}")

    # ── Delta magnitude check ─────────────────────────────────────────────────
    print(f"\nDelta magnitude check (mean |delta|):")
    for label, arr in [
        ("money  / geo",             delta_mon_geo),
        ("money  / math",            delta_mon_math),
        ("money  / business_ethics", delta_mon_be),
        ("money  / philosophy",      delta_mon_phil),
        ("reward / geo",             delta_rew_geo),
        ("reward / math",            delta_rew_math),
        ("reward / business_ethics", delta_rew_be),
        ("reward / philosophy",      delta_rew_phil),
    ]:
        print(f"  {label:<30}: {np.abs(arr).mean():.6f}")

    # ── Comparison with Model A ───────────────────────────────────────────────
    if os.path.exists(MODEL_A_CORE):
        df_a   = pd.read_csv(MODEL_A_CORE)
        a_set  = set(zip(df_a['layer'], df_a['neuron']))
        b_set  = set(zip(df_core['layer'], df_core['neuron']))
        shared = a_set & b_set
        print(f"\nComparison with Model A core ({len(a_set)} neurons):")
        print(f"  Model B core:   {len(b_set)} neurons")
        print(f"  Shared (A ∩ B): {len(shared)} neurons")
        print(f"  Only in A:      {len(a_set - b_set)}  (dropped by stricter 4-domain filter)")
        print(f"  Only in B:      {len(b_set - a_set)}  (new neurons not in phase4 core)")
        print(f"  A retention:    {len(shared)/len(a_set)*100:.1f}%")
    else:
        print(f"\n(Model A core not found at {MODEL_A_CORE} — skipping comparison)")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Layers:           {num_layers}")
    print(f"Intermediate dim: {intermediate_dim}")
    print(f"Universal money:  {len(money_universal):>5}")
    print(f"Universal reward: {len(reward_universal):>5}")
    print(f"Model B core:     {len(master_core):>5}")
    print(f"\nNext: run lesion_late_layers_modelB.py")


if __name__ == "__main__":
    main()