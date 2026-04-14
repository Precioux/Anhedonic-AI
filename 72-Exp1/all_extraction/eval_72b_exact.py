"""
eval_72b_knockout_L46_53.py
================================================================================
Fine-grained knockouts within L46-53 (Δ=−15.26, the confirmed signal range).

Strategy:
  1. Single-layer knockouts  — which individual layers contribute?
  2. Leave-one-out           — remove one layer at a time from full L46-53
  3. Additive combinations   — build up from the strongest single layers

This tells us:
  - Which layers are essential (leave-one-out: removing them kills the effect)
  - Which layers are sufficient (single/pair knockouts that reproduce Δ~−15)
  - The minimum layer set for a clean anhedonic 72B model
================================================================================
"""

import os, re, torch
import pandas as pd
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-72b"
ACT_DIR    = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
EVAL_CSV   = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/data/origin_math_eval.csv"
OUTPUT_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/eval_exact"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

FULL_RANGE = list(range(46, 54))   # L46-53 — the confirmed signal range

# =============================================================================
# Tiers
# =============================================================================
TIERS = {}

# 1. Full range (reference — should reproduce Δ~−15)
TIERS["full_L46_53"] = FULL_RANGE

# 2. Single-layer knockouts
for l in FULL_RANGE:
    TIERS[f"single_L{l}"] = [l]

# 3. Leave-one-out (full range minus one layer)
for l in FULL_RANGE:
    TIERS[f"drop_L{l}"] = [x for x in FULL_RANGE if x != l]

# 4. Promising pairs — center layers where interaction likely lives
TIERS["pair_L47_48"] = [47, 48]
TIERS["pair_L48_49"] = [48, 49]
TIERS["pair_L46_47"] = [46, 47]
TIERS["pair_L50_51"] = [50, 51]
TIERS["pair_L51_52"] = [51, 52]
TIERS["pair_L52_53"] = [52, 53]

# 5. Additive triplets and quads from center
TIERS["triple_L47_48_49"] = [47, 48, 49]
TIERS["triple_L46_47_48"] = [46, 47, 48]
TIERS["triple_L50_51_52"] = [50, 51, 52]
TIERS["quad_L47_50"]      = [47, 48, 49, 50]
TIERS["quad_L48_51"]      = [48, 49, 50, 51]
TIERS["quad_L46_49"]      = [46, 47, 48, 49]
TIERS["quad_L50_53"]      = [50, 51, 52, 53]

print(f"Total tiers to run: {len(TIERS) + 1} (including baseline)")
for name, layers in TIERS.items():
    print(f"  {name:<20} : {layers}")

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
    return torch.cat(parts, dim=0).mean(dim=0).numpy()

print("Loading neutral means ...")
mean_acts = load_neutral_means()
print(f"  Shape: {mean_acts.shape}  ✓\n")

# =============================================================================
# Hooks
# =============================================================================
_hooks = []

def install_layer_hooks(layer_indices: list):
    global _hooks
    remove_hooks()
    for l in layer_indices:
        means = torch.tensor(mean_acts[l], dtype=torch.bfloat16).to("cuda")
        def _make(m):
            def _hook(module, _inp, out):
                out[:, :, :] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        _hooks.append(lm_layers[l].mlp.act_fn.register_forward_hook(_make(means)))

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

# All ablation tiers
for tier_name, layer_range in TIERS.items():
    print(f"\n{'=' * 62}")
    print(f"TIER: {tier_name}  layers={layer_range}")
    print("=" * 62)
    install_layer_hooks(layer_range)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# Combined
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_knockout_combined.csv"), index=False)

# =============================================================================
# Summary — grouped by tier type
# =============================================================================
def print_group(title, tier_list):
    print(f"\n  --- {title} ---")
    print(f"  {'Tier':<22}  {'Layers':<20}  {'Δ':>7}  {'Chose max':>10}  {'Collapsed':>10}")
    print(f"  {'-'*75}")
    for tier in tier_list:
        sub   = df_combined[df_combined["Tier"] == tier]
        if len(sub) == 0: continue
        valid     = sub[~sub["collapsed"]]
        mean_pts  = valid["points"].mean() if len(valid) else float("nan")
        pct_max   = valid["chose_max"].mean() * 100 if len(valid) else float("nan")
        collapsed = sub["collapsed"].sum()
        delta     = mean_pts - base_mean
        layers    = str(TIERS.get(tier, []))
        flag = " ◄" if (not np.isnan(delta) and delta < -5) else ""
        print(f"  {tier:<22}  {layers:<20}  {delta:>+7.2f}  {pct_max:>9.1f}%  "
              f"{collapsed:>4}/{len(sub)}{flag}")

print(f"\n{'=' * 62}")
print("SUMMARY")
print("=" * 62)
print(f"  Baseline mean: {base_mean:.2f} pts")

print_group("REFERENCE", ["full_L46_53"])
print_group("SINGLE-LAYER KNOCKOUTS", [f"single_L{l}" for l in FULL_RANGE])
print_group("LEAVE-ONE-OUT", [f"drop_L{l}" for l in FULL_RANGE])
print_group("PAIRS", [k for k in TIERS if k.startswith("pair_")])
print_group("TRIPLETS & QUADS", [k for k in TIERS if k.startswith(("triple_", "quad_"))])

print(f"\n  ◄ = anhedonic effect (Δ < −5)")
print(f"\nKey questions:")
print(f"  1. Which single layers show Δ < −5?  → essential individual contributors")
print(f"  2. Which drop_L* tiers lose the effect?  → those layers are essential")
print(f"  3. What is the smallest set that reproduces Δ~−15?")
print(f"\nSaved to: {OUTPUT_DIR}")