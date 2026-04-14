"""
analyze_activations_L46_53.py
================================================================================
For layers L46-53, computes:

  1. Per-neuron activation change across all three conditions
     (neutral, money, reward) — ranked from most to least changed

  2. Per-layer activation rate (% of neurons active) per condition

"Most changed" = max absolute deviation from neutral across both domains:
  score(neuron) = max(
      |delta_money_math|, |delta_money_geo|,
      |delta_reward_math|, |delta_reward_geo|
  )

"Active" = activation value > mean + 1std of that layer's neutral distribution
  (i.e. a neuron that is firing meaningfully above its resting level)

Outputs:
  top_neurons_L46_53.csv      — all neurons in L46-53 ranked by change score
  activation_rates.csv        — per-layer, per-condition activation %
  top_neurons_L46_53.png      — bar chart of top 50 neurons
  activation_rates.png        — grouped bar chart per layer
================================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
ACT_DIR    = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
OUTPUT_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/analysis_L46_53"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LAYERS    = list(range(46, 54))   # L46-53
NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568

# =============================================================================
# Load activation means
# =============================================================================
def load_mean(condition: str, domain: str) -> np.ndarray:
    path = os.path.join(ACT_DIR, f"{condition}_activations_{domain}.pt")
    data = torch.load(path, map_location="cpu")
    tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
    stacked = torch.stack(tensors).float()   # [Q, 80, 29568]
    print(f"  {condition}_{domain}: {stacked.shape}")
    return stacked.mean(dim=0).numpy(), stacked.numpy()  # mean [80,29568], all [Q,80,29568]

print("Loading activations ...")
m_neu_mean, m_neu_all = load_mean("neutral", "math")
m_mon_mean, m_mon_all = load_mean("money",   "math")
m_rew_mean, m_rew_all = load_mean("reward",  "math")
g_neu_mean, g_neu_all = load_mean("neutral", "geo")
g_mon_mean, g_mon_all = load_mean("money",   "geo")
g_rew_mean, g_rew_all = load_mean("reward",  "geo")

# Deltas from neutral
delta_mon_math = m_mon_mean - m_neu_mean   # [80, 29568]
delta_mon_geo  = g_mon_mean - g_neu_mean
delta_rew_math = m_rew_mean - m_neu_mean
delta_rew_geo  = g_rew_mean - g_neu_mean

# =============================================================================
# 1. Per-neuron change score — max |delta| across all 4 condition×domain combos
# =============================================================================
print("\nComputing per-neuron change scores for L46-53 ...")

records = []
for l in TARGET_LAYERS:
    d_mm = delta_mon_math[l]   # [29568]
    d_mg = delta_mon_geo[l]
    d_rm = delta_rew_math[l]
    d_rg = delta_rew_geo[l]

    # Max absolute change across all four deltas
    score = np.maximum.reduce([
        np.abs(d_mm), np.abs(d_mg),
        np.abs(d_rm), np.abs(d_rg)
    ])   # [29568]

    for n in range(INTERMEDIATE_DIM):
        records.append({
            "layer":           l,
            "neuron":          n,
            "max_delta":       float(score[n]),
            "delta_money_math":  float(d_mm[n]),
            "delta_money_geo":   float(d_mg[n]),
            "delta_reward_math": float(d_rm[n]),
            "delta_reward_geo":  float(d_rg[n]),
            # Mean activation per condition (avg of math+geo)
            "act_neutral": float((m_neu_mean[l, n] + g_neu_mean[l, n]) / 2),
            "act_money":   float((m_mon_mean[l, n] + g_mon_mean[l, n]) / 2),
            "act_reward":  float((m_rew_mean[l, n] + g_rew_mean[l, n]) / 2),
        })

df_neurons = pd.DataFrame(records)
df_neurons = df_neurons.sort_values("max_delta", ascending=False).reset_index(drop=True)
df_neurons.index += 1   # rank starts at 1

out_csv = os.path.join(OUTPUT_DIR, "top_neurons_L46_53.csv")
df_neurons.to_csv(out_csv, index=True, index_label="rank")
print(f"Saved {len(df_neurons):,} neurons ranked → {out_csv}")

# Print top 30
print(f"\nTop 30 most-changed neurons across L46-53:")
print(f"  {'Rank':>5}  {'Layer':>6}  {'Neuron':>7}  {'MaxΔ':>8}  "
      f"{'Δ_mon_math':>12}  {'Δ_mon_geo':>11}  "
      f"{'Δ_rew_math':>12}  {'Δ_rew_geo':>11}")
print(f"  {'-'*85}")
for i, row in df_neurons.head(30).iterrows():
    print(f"  {i:>5}  {int(row.layer):>6}  {int(row.neuron):>7}  "
          f"{row.max_delta:>8.4f}  "
          f"{row.delta_money_math:>12.4f}  {row.delta_money_geo:>11.4f}  "
          f"{row.delta_reward_math:>12.4f}  {row.delta_reward_geo:>11.4f}")

# How many neurons per layer are in the top N?
print(f"\nTop 500 neurons — layer distribution:")
top500 = df_neurons.head(500)
for l in TARGET_LAYERS:
    c = (top500["layer"] == l).sum()
    print(f"  L{l}: {c:>4}  {'█' * (c // 5)}")

# =============================================================================
# 2. Activation rate per layer per condition
# =============================================================================
print(f"\nComputing activation rates ...")

# Threshold: mean + 1std of neutral activations for that layer
# (averaged across math and geo neutral)
neu_combined = np.concatenate([
    m_neu_all,   # [100, 80, 29568]
    g_neu_all,   # [100, 80, 29568]
], axis=0)       # [200, 80, 29568]

rate_records = []

for l in TARGET_LAYERS:
    neu_vals = neu_combined[:, l, :]   # [200, 29568]
    threshold = neu_vals.mean() + neu_vals.std()

    for condition, arrays in [
        ("neutral", [m_neu_all[:, l, :], g_neu_all[:, l, :]]),
        ("money",   [m_mon_all[:, l, :], g_mon_all[:, l, :]]),
        ("reward",  [m_rew_all[:, l, :], g_rew_all[:, l, :]]),
    ]:
        combined = np.concatenate(arrays, axis=0)  # [200, 29568]
        # % of neurons (averaged across questions) that exceed threshold
        active_per_q = (combined > threshold).mean(axis=1)  # [200]
        rate = active_per_q.mean() * 100

        # Also: % of neurons in the MEAN activation that exceed threshold
        mean_rate = (combined.mean(axis=0) > threshold).mean() * 100

        rate_records.append({
            "layer":          l,
            "condition":      condition,
            "activation_rate_mean_act":  round(float(mean_rate), 3),
            "activation_rate_per_q":     round(float(rate), 3),
            "threshold":      round(float(threshold), 4),
        })

df_rates = pd.DataFrame(rate_records)
out_rates = os.path.join(OUTPUT_DIR, "activation_rates.csv")
df_rates.to_csv(out_rates, index=False)
print(f"Saved activation rates → {out_rates}")

print(f"\nActivation rates (% of neurons active, mean activation > threshold):")
print(f"  {'Layer':>6}  {'Neutral':>10}  {'Money':>10}  {'Reward':>10}  "
      f"{'Δ Money':>10}  {'Δ Reward':>10}")
print(f"  {'-'*65}")
for l in TARGET_LAYERS:
    neu = df_rates[(df_rates.layer==l) & (df_rates.condition=="neutral")]["activation_rate_mean_act"].values[0]
    mon = df_rates[(df_rates.layer==l) & (df_rates.condition=="money")  ]["activation_rate_mean_act"].values[0]
    rew = df_rates[(df_rates.layer==l) & (df_rates.condition=="reward") ]["activation_rate_mean_act"].values[0]
    print(f"  L{l:>3}    {neu:>9.2f}%  {mon:>9.2f}%  {rew:>9.2f}%  "
          f"{mon-neu:>+9.2f}%  {rew-neu:>+9.2f}%")

# =============================================================================
# Plots
# =============================================================================
plt.rcParams['figure.dpi'] = 130

# ── Fig 1: Top 50 neurons by max delta ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 6))
top50 = df_neurons.head(50)
colors = [f"C{TARGET_LAYERS.index(int(l))}" for l in top50["layer"]]
bars = ax.bar(range(50), top50["max_delta"], color=colors, alpha=0.85)
ax.set_xticks(range(50))
ax.set_xticklabels(
    [f"L{int(r.layer)}\nN{int(r.neuron)}" for _, r in top50.iterrows()],
    fontsize=6, rotation=45, ha='right'
)
ax.set_ylabel("Max |delta| across all conditions × domains")
ax.set_title("Top 50 Most-Changed Neurons in L46-53\n"
             "(max absolute delta across money/reward × math/geo)", fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
legend_patches = [mpatches.Patch(color=f"C{i}", label=f"L{l}")
                  for i, l in enumerate(TARGET_LAYERS)]
ax.legend(handles=legend_patches, fontsize=8, ncol=4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_neurons_L46_53.png"), bbox_inches='tight', dpi=150)
plt.show()
print("Saved top_neurons_L46_53.png")

# ── Fig 2: Activation rates per layer per condition ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("Activation Rates in L46-53 by Condition", fontweight='bold')

x = np.arange(len(TARGET_LAYERS))
w = 0.25
colors_cond = {"neutral": "#607d8b", "money": "#42a5f5", "reward": "#ef5350"}

for ax, metric, title in zip(axes,
    ["activation_rate_mean_act", "activation_rate_per_q"],
    ["% neurons active (mean activation)", "% neurons active (avg per question)"]):

    for j, (cond, color) in enumerate(colors_cond.items()):
        vals = [df_rates[(df_rates.layer==l) & (df_rates.condition==cond)][metric].values[0]
                for l in TARGET_LAYERS]
        ax.bar(x + j*w - w, vals, width=w, color=color, alpha=0.85, label=cond)
        for i, v in enumerate(vals):
            ax.text(i + j*w - w, v + 0.2, f"{v:.1f}", ha='center', fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in TARGET_LAYERS])
    ax.set_ylabel("% neurons active")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "activation_rates.png"), bbox_inches='tight', dpi=150)
plt.show()
print("Saved activation_rates.png")

# ── Fig 3: Delta heatmap — top 200 neurons across conditions ─────────────────
top200 = df_neurons.head(200)
fig, axes = plt.subplots(1, 4, figsize=(22, 10), sharey=True)
fig.suptitle("Top 200 Neurons: Delta per Condition × Domain\n(rows=neurons ranked by max delta)",
             fontweight='bold')

for ax, col, title, cmap in zip(axes,
    ["delta_money_math", "delta_money_geo", "delta_reward_math", "delta_reward_geo"],
    ["Money / Math", "Money / Geo", "Reward / Math", "Reward / Geo"],
    ["RdBu_r"] * 4):
    vals = top200[col].values.reshape(-1, 1)
    vmax = np.abs(vals).max()
    im = ax.imshow(vals, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("(single value)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    # Y-axis: layer labels every 25 rows
    yticks = list(range(0, 200, 25))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"R{i+1} L{int(top200.iloc[i].layer)}" for i in yticks], fontsize=7)

axes[0].set_ylabel("Rank (top=most changed)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "delta_heatmap_top200.png"), bbox_inches='tight', dpi=150)
plt.show()
print("Saved delta_heatmap_top200.png")

print(f"\nAll outputs in: {OUTPUT_DIR}")
print("Done ✓")