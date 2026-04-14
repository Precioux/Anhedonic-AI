"""
eval_72b_confirm_minimum.py
================================================================================
Confirms the minimum layer set for the 72B anhedonic model.

Based on knockout analysis:
  - L47-50 is the core circuit         (Δ=−9.06, 0 collapsed)
  - L46 and L51 are essential boundary layers
  - Full L46-53 gives Δ=−15.26

Tiers to confirm:
  baseline        — reference
  full_L46_53     — full confirmed range (Δ=−15.26 reference)
  core_L47_50     — minimum core         (Δ=−9.06 from knockout)
  core_L47_51     — core + L51
  core_L46_50     — core + L46
  core_L46_51     — core + both boundary layers (predicted sweet spot)
  core_L46_52     — one step wider
  core_L45_51     — extend below L46
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
OUTPUT_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/eval_confirm"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# =============================================================================
# Tiers
# =============================================================================
TIERS = {
    "full_L46_53": list(range(46, 54)),   # reference — Δ=−15.26
    "core_L47_50": list(range(47, 51)),   # minimum core from knockout
    "core_L47_51": list(range(47, 52)),   # core + L51
    "core_L46_50": list(range(46, 51)),   # core + L46
    "core_L46_51": list(range(46, 52)),   # core + both boundaries ← predicted sweet spot
    "core_L46_52": list(range(46, 53)),   # one step wider
    "core_L45_51": list(range(45, 52)),   # extend below L46
}

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
    total = len(layer_indices) * INTERMEDIATE_DIM
    pct   = total / TOTAL_NEURONS * 100
    print(f"  Layers {layer_indices[0]}-{layer_indices[-1]}: "
          f"{len(layer_indices)} layers, {total:,} neurons ({pct:.3f}%)")

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

# Ablation tiers
for tier_name, layer_range in TIERS.items():
    print(f"\n{'=' * 62}")
    print(f"TIER: {tier_name}")
    print("=" * 62)
    install_layer_hooks(layer_range)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# Combined
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_confirm_combined.csv"), index=False)

# =============================================================================
# Summary
# =============================================================================
print(f"\n{'=' * 62}")
print("SUMMARY — Minimum Layer Set Confirmation")
print("=" * 62)
print(f"\n  Baseline: {base_mean:.2f} pts  |  Target: Δ~−15, collapse~0\n")
print(f"  {'Tier':<16}  {'Layers':<12}  {'N layers':>8}  {'Δ':>7}  "
      f"{'Chose max':>10}  {'Collapsed':>10}  Note")
print(f"  {'-'*90}")

notes = {
    "full_L46_53": "reference",
    "core_L47_50": "minimum core (4L)",
    "core_L47_51": "core + L51 (5L)",
    "core_L46_50": "core + L46 (5L)",
    "core_L46_51": "← predicted sweet spot (6L)",
    "core_L46_52": "7 layers",
    "core_L45_51": "extend below (7L)",
}

for tier in ["baseline"] + list(TIERS.keys()):
    sub   = df_combined[df_combined["Tier"] == tier]
    valid = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean() if len(valid) else float("nan")
    pct_max   = valid["chose_max"].mean() * 100 if len(valid) else float("nan")
    collapsed = sub["collapsed"].sum()
    delta     = mean_pts - base_mean
    layer_range = TIERS.get(tier, [])
    layer_str   = f"L{layer_range[0]}-{layer_range[-1]}" if layer_range else "—"
    n_layers    = len(layer_range)
    note        = notes.get(tier, "")
    flag        = " ◄" if (not np.isnan(delta) and delta < -5) else ""

    print(f"  {tier:<16}  {layer_str:<12}  {n_layers:>8}  {delta:>+7.2f}  "
          f"{pct_max:>9.1f}%  {collapsed:>4}/{len(sub)}        {note}{flag}")

print(f"\n  ◄ = meaningful anhedonic effect (Δ < −5)")
print(f"\n  Best candidate = lowest collapse + strongest Δ + fewest layers")
print(f"\nSaved to: {OUTPUT_DIR}")