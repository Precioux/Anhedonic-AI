"""
eval_72b_union.py
================================================================================
Tests all three neuron variants (union-based, L46-53) against baseline
on origin_math_eval.csv.

Tiers:
  baseline         — no ablation
  reward_only      — reward_union − money_union  (~pure signal, test first)
  reward_univ      — full reward_union           (broadest)
  core             — money_union ∩ reward_union  (most selective)

Neuron-specific hooks — identical mechanism to 7B model_A_layers_18_27.py.
Hooks are installed and removed between tiers — no state leakage.
================================================================================
"""

import os, re, json, torch
import pandas as pd
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH  = "/mnt/mahdipou/models/qwen2-vl-72b"
ACT_DIR     = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
NEURONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/L46_53_union"
EVAL_CSV    = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/data/origin_math_eval.csv"
OUTPUT_DIR  = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/eval_union"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# Tiers: (name, json_filename or None for baseline)
TIERS = [
    ("baseline",    None),
    ("reward_only", "neurons_A_72b_reward_only.json"),
    ("reward_univ", "neurons_A_72b_reward_univ.json"),
    ("core",        "neurons_A_72b_core.json"),
]

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
# Hook management — neuron-specific (identical to 7B)
# =============================================================================
_hooks = []

def install_neuron_hooks(json_file: str):
    global _hooks
    remove_hooks()
    path = os.path.join(NEURONS_DIR, json_file)
    with open(path) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    total = sum(len(v) for v in neuron_map.values())
    pct   = total / TOTAL_NEURONS * 100
    print(f"  Hooks: {len(neuron_map)} layers, {total:,} neurons ({pct:.4f}%)")
    print(f"  Layers: {sorted(neuron_map.keys())}")

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
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95,
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

for tier_name, json_file in TIERS:
    print(f"\n{'='*62}")
    print(f"TIER: {tier_name}")
    print("=" * 62)

    if json_file is not None:
        install_neuron_hooks(json_file)

    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()

    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# =============================================================================
# Summary
# =============================================================================
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_combined.csv"), index=False)

base_mean = df_combined[df_combined["Tier"]=="baseline"]["points"].mean()

print(f"\n{'='*62}")
print("SUMMARY")
print("=" * 62)
print(f"\n  {'Tier':<16}  {'Neurons':>10}  {'Δ':>7}  {'Chose max':>10}  {'Collapsed':>10}  Direction")
print(f"  {'-'*75}")

neuron_counts = {}
for tier_name, json_file in TIERS:
    if json_file:
        with open(os.path.join(NEURONS_DIR, json_file)) as f:
            nm = json.load(f)
        neuron_counts[tier_name] = sum(len(v) for v in nm.values())
    else:
        neuron_counts[tier_name] = 0

for tier_name, _ in TIERS:
    sub   = df_combined[df_combined["Tier"] == tier_name]
    valid = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean() if len(valid) else float("nan")
    pct_max   = valid["chose_max"].mean() * 100 if len(valid) else float("nan")
    collapsed = sub["collapsed"].sum()
    delta     = mean_pts - base_mean
    n_neurons = neuron_counts[tier_name]

    if tier_name == "baseline":
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

    print(f"  {tier_name:<16}  {n_neurons:>10,}  {delta:>+7.2f}  "
          f"{pct_max:>9.1f}%  {collapsed:>4}/{len(sub)}        {direction}")

print(f"\nSaved to: {OUTPUT_DIR}")