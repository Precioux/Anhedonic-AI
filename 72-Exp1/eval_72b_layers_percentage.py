"""
eval_72b_pct_layers.py
================================================================================
Full percentage sweep to find the minimum % of neurons per layer needed
to produce an anhedonic effect in L46-53.

We know:
  100% per layer → Δ=−14.27  ✓
    2% per layer → Δ=0        ✗

Tiers (% of each layer's neurons, selected by highest reward activation score):
  1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90

No extraction step needed — neuron maps built from top_neurons_L46_53.csv.
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
ACT_DIR     = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
RANKED_CSV  = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/analysis_L46_53/top_neurons_L46_53.csv"
EVAL_CSV    = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/data/origin_math_eval.csv"
OUTPUT_DIR  = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/eval_pct_layers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM
TARGET_LAYERS    = list(range(46, 54))

PERCENTAGES = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]

# =============================================================================
# Build neuron maps — top X% per layer by reward score
# =============================================================================
print("Loading ranked neuron CSV ...")
df_ranked = pd.read_csv(RANKED_CSV)
df_ranked['reward_score'] = df_ranked[['delta_reward_math','delta_reward_geo']].abs().max(axis=1)

# Pre-sort per layer once
layer_sorted = {}
for l in TARGET_LAYERS:
    layer_sorted[l] = df_ranked[df_ranked['layer'] == l].sort_values(
        'reward_score', ascending=False
    )['neuron'].astype(int).tolist()

def build_neuron_map(pct: float) -> dict:
    n_per_layer = max(1, int(INTERMEDIATE_DIM * pct / 100))
    return {l: sorted(layer_sorted[l][:n_per_layer]) for l in TARGET_LAYERS}

print(f"\nNeurons per percentage (per layer × {len(TARGET_LAYERS)} layers):")
print(f"  {'Pct':>5}  {'Per layer':>10}  {'Total':>8}  {'% network':>10}")
print(f"  {'-'*45}")
for pct in PERCENTAGES:
    n_per = max(1, int(INTERMEDIATE_DIM * pct / 100))
    total = n_per * len(TARGET_LAYERS)
    print(f"  {pct:>4}%  {n_per:>10,}  {total:>8,}  {total/TOTAL_NEURONS*100:>9.4f}%")

# =============================================================================
# Load model
# =============================================================================
print("\n" + "=" * 62)
print("Loading Qwen2-VL-72B (NF4 4-bit) ...")
print("=" * 62)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config, device_map={"": 0},
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
    return torch.cat(parts, dim=0).mean(dim=0).numpy()

print("Loading neutral means ...")
mean_acts = load_neutral_means()
print(f"  Shape: {mean_acts.shape}  ✓\n")

# =============================================================================
# Hook management — neuron-specific (identical to 7B)
# =============================================================================
_hooks = []

def install_neuron_hooks(neuron_map: dict):
    global _hooks
    remove_hooks()
    total    = sum(len(v) for v in neuron_map.values())
    pct_net  = total / TOTAL_NEURONS * 100
    pct_lay  = total / (len(neuron_map) * INTERMEDIATE_DIM) * 100
    print(f"  Hooks: {total:,} neurons  "
          f"({pct_lay:.1f}% per layer, {pct_net:.4f}% of network)")
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(
            mean_acts[layer_idx, neurons], dtype=torch.bfloat16
        ).to("cuda")
        def _make(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        _hooks.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        )

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
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, do_sample=True, top_p=0.95,
        )
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

def run_eval(df: pd.DataFrame, tier_name: str) -> pd.DataFrame:
    records = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        response = generate(str(row["Full_Prompt"]))
        parsed   = parse_response(response, row)
        records.append({"ID": row["ID"], "Subset": row["Subset"],
                        "Reward_Order": row["Reward_Order"],
                        "Tier": tier_name, **parsed})
        if (i + 1) % 24 == 0 or (i + 1) == n:
            done     = [r for r in records if not r["collapsed"]]
            mean_pts = np.mean([r["points"] for r in done]) if done else float("nan")
            pct_max  = np.mean([r["chose_max"] for r in done]) * 100 if done else float("nan")
            print(f"  [{tier_name}] {i+1}/{n}  |  "
                  f"pts={mean_pts:.1f}  max%={pct_max:.1f}  "
                  f"collapsed={sum(r['collapsed'] for r in records)}")
    return pd.DataFrame(records)

# =============================================================================
# Main
# =============================================================================
df_eval = pd.read_csv(EVAL_CSV)
print(f"Eval dataset: {len(df_eval)} rows\n")

all_results = []

# Baseline
print("=" * 62)
print("TIER: baseline")
print("=" * 62)
df_base = run_eval(df_eval, "baseline")
df_base.to_csv(os.path.join(OUTPUT_DIR, "eval_baseline.csv"), index=False)
all_results.append(df_base)
base_mean = df_base[~df_base["collapsed"]]["points"].mean()

# Percentage tiers
for pct in PERCENTAGES:
    tier_name  = f"pct{pct:02d}"
    neuron_map = build_neuron_map(pct)
    print(f"\n{'='*62}")
    print(f"TIER: {tier_name}  ({pct}% of each layer)")
    print("=" * 62)
    install_neuron_hooks(neuron_map)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# Combined
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_combined.csv"), index=False)

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'='*62}")
print("SUMMARY")
print("=" * 62)
print(f"\n  {'Tier':<10}  {'% layer':>8}  {'Neurons':>8}  {'Δ':>7}  "
      f"{'Chose max':>10}  {'Collapsed':>10}  Direction")
print(f"  {'-'*80}")

all_tiers = ["baseline"] + [f"pct{p:02d}" for p in PERCENTAGES]
for tier_name in all_tiers:
    sub   = df_combined[df_combined["Tier"] == tier_name]
    valid = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean() if len(valid) else float("nan")
    pct_max   = valid["chose_max"].mean() * 100 if len(valid) else float("nan")
    collapsed = sub["collapsed"].sum()
    delta     = mean_pts - base_mean

    if tier_name == "baseline":
        pct_layer = 0
        n_neurons = 0
    else:
        pct_layer = int(tier_name.replace("pct", ""))
        n_neurons = max(1, int(INTERMEDIATE_DIM * pct_layer / 100)) * len(TARGET_LAYERS)

    if tier_name == "baseline":
        direction = "—"
    elif collapsed == len(sub):
        direction = "FULL COLLAPSE"
    elif np.isnan(delta):
        direction = "?"
    elif delta < -5:
        direction = "anhedonic ↓  ◄"
    elif delta > 5:
        direction = "hyperhedonic ↑"
    else:
        direction = "no effect"

    print(f"  {tier_name:<10}  {pct_layer:>7}%  {n_neurons:>8,}  {delta:>+7.2f}  "
          f"{pct_max:>9.1f}%  {collapsed:>4}/{len(sub)}        {direction}")

print(f"\n  Reference: 100% (whole-layer) → Δ=−14.27, 0/96 collapsed")
print(f"\nSaved to: {OUTPUT_DIR}")