"""
eval_72b_reward_topN.py
================================================================================
Tests ablation of top 1000-5000 neurons ranked by reward activation change
on origin_math_eval.csv.

Tiers:
  baseline              — no ablation
  reward_top1000        — 1000 most reward-activated neurons in L46-53
  reward_top2000        — 2000
  reward_top3000        — 3000
  reward_top4000        — 4000
  reward_top5000        — 5000

Neuron-specific hooks — identical to 7B model_A_layers_18_27.py.
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
NEURONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/neurons/L46_53_reward_topN"
EVAL_CSV    = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/data/origin_math_eval.csv"
OUTPUT_DIR  = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/eval_reward_topN"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

TIERS = [
    ("baseline",       None),
    ("reward_top1000", "neurons_A_72b_reward_top1000.json"),
    ("reward_top2000", "neurons_A_72b_reward_top2000.json"),
    ("reward_top3000", "neurons_A_72b_reward_top3000.json"),
    ("reward_top4000", "neurons_A_72b_reward_top4000.json"),
    ("reward_top5000", "neurons_A_72b_reward_top5000.json"),
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

def install_neuron_hooks(json_file: str):
    global _hooks
    remove_hooks()
    path = os.path.join(NEURONS_DIR, json_file)
    with open(path) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    total = sum(len(v) for v in neuron_map.values())
    pct   = total / TOTAL_NEURONS * 100
    print(f"  Hooks: {len(neuron_map)} layers, {total:,} neurons ({pct:.4f}%)")
    print(f"  Per layer: " + "  ".join(
        f"L{l}:{len(neuron_map.get(l,[]))}" for l in range(46, 54)
    ))
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

for tier_name, json_file in TIERS:
    print(f"\n{'='*62}")
    print(f"TIER: {tier_name}")
    print("=" * 62)
    if json_file:
        install_neuron_hooks(json_file)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_combined.csv"), index=False)

base_mean = df_combined[df_combined["Tier"] == "baseline"]["points"].mean()

print(f"\n{'='*62}")
print("SUMMARY")
print("=" * 62)
print(f"\n  {'Tier':<18}  {'Neurons':>8}  {'Δ':>7}  {'Chose max':>10}  {'Collapsed':>10}  Direction")
print(f"  {'-'*75}")

for tier_name, json_file in TIERS:
    sub   = df_combined[df_combined["Tier"] == tier_name]
    valid = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean() if len(valid) else float("nan")
    pct_max   = valid["chose_max"].mean() * 100 if len(valid) else float("nan")
    collapsed = sub["collapsed"].sum()
    delta     = mean_pts - base_mean

    n_neurons = 0
    if json_file:
        p = os.path.join(NEURONS_DIR, json_file)
        if os.path.exists(p):
            with open(p) as f:
                nm = json.load(f)
            n_neurons = sum(len(v) for v in nm.values())

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

    print(f"  {tier_name:<18}  {n_neurons:>8,}  {delta:>+7.2f}  "
          f"{pct_max:>9.1f}%  {collapsed:>4}/{len(sub)}        {direction}")

print(f"\nSaved to: {OUTPUT_DIR}")