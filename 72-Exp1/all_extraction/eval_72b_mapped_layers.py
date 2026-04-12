"""
eval_72b_reward_neurons.py
================================================================================
Tests reward_univ and reward_only neurons (not just master_core) in the
proportionally mapped 72B layers (L51-77, equivalent to 7B L18-27).

In 7B, the anhedonic signal came from reward-only neurons in late layers,
NOT from master_core. We replicate that analysis here.

Neuron sets tested (all filtered to mapped layers L51-77):
  reward_univ_mapped   — all reward-sensitive neurons in L51-77
  reward_only_mapped   — reward_univ minus money_univ, in L51-77
  master_core_mapped   — intersection, in L51-77 (reference)

Also tests the full sets (all layers) for comparison:
  reward_univ_all      — all reward neurons across all layers
  reward_only_all      — reward-only neurons across all layers

Tiers: baseline + 5 ablation conditions
================================================================================
"""

import os, re, torch
import pandas as pd
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH  = "/mnt/mahdipou/models/qwen2-vl-72b"
NEURONS_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/neurons/orig"
ACT_DIR     = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
EVAL_CSV    = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/data/origin_math_eval.csv"
OUTPUT_DIR  = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/eval_mapped"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# Proportionally mapped layers: 7B L18-27 → 72B (×80/28, rounded)
MAPPED_LAYERS = sorted(set(round(l * 80 / 28) for l in range(18, 28)))
# Full contiguous range covering all mapped points
MAPPED_RANGE  = list(range(51, 78))

print(f"Mapped exact layers : {MAPPED_LAYERS}")
print(f"Mapped range L51-77 : {MAPPED_RANGE}\n")

# =============================================================================
# Load neuron sets
# =============================================================================
print("Loading neuron sets ...")
df_reward = pd.read_csv(f"{NEURONS_DIR}/universal_reward_neurons.csv")
df_money  = pd.read_csv(f"{NEURONS_DIR}/universal_money_neurons.csv")
df_core   = pd.read_csv(f"{NEURONS_DIR}/master_incentive_core.csv")

reward_set = set(zip(df_reward['layer'], df_reward['neuron']))
money_set  = set(zip(df_money['layer'],  df_money['neuron']))
core_set   = set(zip(df_core['layer'],   df_core['neuron']))
reward_only_set = reward_set - money_set

print(f"  reward_univ : {len(reward_set):,}")
print(f"  money_univ  : {len(money_set):,}")
print(f"  master_core : {len(core_set):,}")
print(f"  reward_only : {len(reward_only_set):,}  (reward_univ − money_univ)\n")

# =============================================================================
# Helper: filter a neuron set to specific layers → neuron_map dict
# =============================================================================
def to_neuron_map(neuron_set: set, layer_filter: list | None = None) -> dict[int, list[int]]:
    """
    Convert set of (layer, neuron) tuples to {layer: [neuron_idx, ...]} dict.
    Optionally filter to only include specified layers.
    """
    nm: dict[int, list[int]] = {}
    for (l, n) in neuron_set:
        if layer_filter is not None and l not in layer_filter:
            continue
        nm.setdefault(l, []).append(n)
    for v in nm.values():
        v.sort()
    return nm

def summarize_map(name: str, nm: dict):
    total = sum(len(v) for v in nm.values())
    pct   = total / TOTAL_NEURONS * 100
    layers = sorted(nm.keys())
    print(f"  {name}:")
    print(f"    {len(nm)} layers, {total:,} neurons ({pct:.4f}%)")
    print(f"    Layers: {layers}")

# Build all tier neuron maps
tiers_neuron_map = {
    "reward_univ_mapped"  : to_neuron_map(reward_set,       MAPPED_RANGE),
    "reward_only_mapped"  : to_neuron_map(reward_only_set,  MAPPED_RANGE),
    "master_core_mapped"  : to_neuron_map(core_set,         MAPPED_RANGE),
    "reward_univ_all"     : to_neuron_map(reward_set),
    "reward_only_all"     : to_neuron_map(reward_only_set),
}

print("Neuron map summary:")
for name, nm in tiers_neuron_map.items():
    summarize_map(name, nm)
    print()

# =============================================================================
# Load model
# =============================================================================
print("=" * 62)
print("Loading Qwen2-VL-72B (NF4 4-bit) ...")
print("=" * 62)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)
lm_layers = model.model.language_model.layers
print(f"Layers: {len(lm_layers)}  ✓\n")

# =============================================================================
# Neutral means
# =============================================================================
def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACT_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()  # [80, 29568]

print("Loading neutral means ...")
mean_acts = load_neutral_means()
print(f"  Shape: {mean_acts.shape}  ✓\n")

# =============================================================================
# Hook management
# =============================================================================
_hooks = []

def install_neuron_hooks(neuron_map: dict[int, list[int]]):
    global _hooks
    remove_hooks()
    total = 0
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons, dtype=torch.long).to("cuda")
        means = torch.tensor(
            mean_acts[layer_idx, neurons], dtype=torch.bfloat16
        ).to("cuda")
        def _make(i, m):
            def _hook(module, _inp, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        _hooks.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        )
        total += len(neurons)
    pct = total / TOTAL_NEURONS * 100
    print(f"  Hooks: {len(neuron_map)} layers, {total:,} neurons ({pct:.4f}%)")

def remove_hooks():
    global _hooks
    for h in _hooks:
        h.remove()
    _hooks = []

# =============================================================================
# Generation & parsing
# =============================================================================
def generate(prompt: str, max_new_tokens: int = 64) -> str:
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

def parse_response(response: str, row: pd.Series) -> dict:
    match = re.search(r'\b([1-4])\b', response)
    if not match:
        return {"chosen_q": None, "points": None, "chose_max": None,
                "collapsed": True, "raw_response": response}
    q       = int(match.group(1))
    points  = int(row[f"Reward_{q}"])
    max_pts = max(int(row[f"Reward_{i}"]) for i in range(1, 5))
    return {"chosen_q": q, "points": points, "chose_max": (points == max_pts),
            "collapsed": False, "raw_response": response}

# =============================================================================
# Run one tier
# =============================================================================
def run_eval(df: pd.DataFrame, tier_name: str) -> pd.DataFrame:
    records = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        response = generate(str(row["Full_Prompt"]))
        parsed   = parse_response(response, row)
        records.append({"ID": row["ID"], "Subset": row["Subset"],
                        "Reward_Order": row["Reward_Order"],
                        "Tier": tier_name, **parsed})
        if (i + 1) % 10 == 0 or (i + 1) == n:
            done     = [r for r in records if not r["collapsed"]]
            mean_pts = np.mean([r["points"] for r in done]) if done else float("nan")
            pct_max  = np.mean([r["chose_max"] for r in done]) * 100 if done else float("nan")
            print(f"  [{tier_name}] {i+1}/{n}  |  "
                  f"mean pts={mean_pts:.1f}  chose_max={pct_max:.1f}%  "
                  f"collapsed={sum(r['collapsed'] for r in records)}")
    return pd.DataFrame(records)

# =============================================================================
# Main
# =============================================================================
df_eval = pd.read_csv(EVAL_CSV)
print(f"Eval dataset: {len(df_eval)} rows\n")

all_results = []

# ── Baseline ────────────────────────────────────────────────────────────────
print("=" * 62)
print("TIER: baseline")
print("=" * 62)
df_base = run_eval(df_eval, "baseline")
df_base.to_csv(os.path.join(OUTPUT_DIR, "eval_baseline.csv"), index=False)
all_results.append(df_base)
base_mean = df_base[~df_base["collapsed"]]["points"].mean()

# ── Ablation tiers ──────────────────────────────────────────────────────────
for tier_name, neuron_map in tiers_neuron_map.items():
    print(f"\n{'=' * 62}")
    print(f"TIER: {tier_name}")
    print("=" * 62)
    if not neuron_map:
        print("  No neurons found for this tier — skipping.")
        continue
    install_neuron_hooks(neuron_map)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# ── Combined ────────────────────────────────────────────────────────────────
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_reward_combined.csv"), index=False)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print("SUMMARY")
print("=" * 62)
print(f"\n  {'Tier':<26}  {'Mean pts':>9}  {'Δ':>7}  {'Chose max':>10}  {'Collapsed':>10}  Direction")
print(f"  {'-'*85}")

tier_order = ["baseline"] + list(tiers_neuron_map.keys())
for tier in tier_order:
    sub = df_combined[df_combined["Tier"] == tier]
    if len(sub) == 0:
        continue
    valid     = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean() if len(valid) else float("nan")
    pct_max   = valid["chose_max"].mean() * 100 if len(valid) else float("nan")
    collapsed = sub["collapsed"].sum()
    delta     = mean_pts - base_mean
    if tier == "baseline":
        direction = "—"
    elif collapsed == len(sub):
        direction = "FULL COLLAPSE"
    elif np.isnan(delta):
        direction = "?"
    elif delta < -5:
        direction = "anhedonic ↓"
    elif delta > 5:
        direction = "hyperhedonic ↑"
    else:
        direction = "no effect"
    print(f"  {tier:<26}  {mean_pts:>9.2f}  {delta:>+7.2f}  {pct_max:>9.1f}%  "
          f"{collapsed:>4}/{len(sub)}        {direction}")

print(f"\nSaved to: {OUTPUT_DIR}")