"""
eval_72b_layer_ablations.py
================================================================================
Tests two targeted layer ablations on Qwen2-VL-72B:
  - Tier 1: baseline         (no ablation)
  - Tier 2: late_L54_79      (layers 54-79, top core layers by count)
  - Tier 3: mid_L38_53       (layers 38-53, highest activation delta)

For each layer range, ALL neurons in those layers are clamped to their
neutral mean — same hook mechanism as before, but layer-targeted rather
than neuron-set-targeted.

Run on H200:
  python eval_72b_layer_ablations.py
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
OUTPUT_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/eval_layers"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# Layer ranges to test
TIERS = {
    "late_L54_79": list(range(54, 80)),   # top core layers by count (L57, L58 dominate)
    "mid_L38_53":  list(range(38, 54)),   # highest activation delta signal
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
processor  = AutoProcessor.from_pretrained(MODEL_PATH)
lm_layers  = model.model.language_model.layers
print(f"Layers: {len(lm_layers)}  ✓\n")

# =============================================================================
# Neutral means — averaged over geo + math, all 100 questions each
# =============================================================================
def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACT_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    combined = torch.cat(parts, dim=0)          # [200, 80, 29568]
    return combined.mean(dim=0).numpy()          # [80, 29568]

print("Loading neutral means ...")
mean_acts = load_neutral_means()
print(f"  Neutral means shape: {mean_acts.shape}  ✓\n")

# =============================================================================
# Hook management — clamp ALL neurons in specified layers
# =============================================================================
_hooks = []

def install_layer_hooks(layer_indices: list[int]):
    global _hooks
    remove_hooks()
    total = 0
    for l in layer_indices:
        means = torch.tensor(mean_acts[l], dtype=torch.bfloat16).to("cuda")  # [29568]
        def _make(m):
            def _hook(module, _inp, out):
                # out: [batch, seq_len, 29568] — clamp all neurons in this layer
                out[:, :, :] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        h = lm_layers[l].mlp.act_fn.register_forward_hook(_make(means))
        _hooks.append(h)
        total += INTERMEDIATE_DIM
    n_layers = len(layer_indices)
    pct = total / TOTAL_NEURONS * 100
    print(f"  Hooks installed: {n_layers} layers × {INTERMEDIATE_DIM:,} neurons = {total:,} total ({pct:.3f}%)")
    print(f"  Layer range: {layer_indices[0]}–{layer_indices[-1]}")

def remove_hooks():
    global _hooks
    for h in _hooks:
        h.remove()
    _hooks = []

# =============================================================================
# Generation — greedy, deterministic
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

# =============================================================================
# Parse response
# =============================================================================
def parse_response(response: str, row: pd.Series) -> dict:
    match = re.search(r'\b([1-4])\b', response)
    if not match:
        return {"chosen_q": None, "points": None, "chose_max": None,
                "collapsed": True, "raw_response": response}
    q      = int(match.group(1))
    points = int(row[f"Reward_{q}"])
    max_reward = max(int(row[f"Reward_{i}"]) for i in range(1, 5))
    return {"chosen_q": q, "points": points, "chose_max": (points == max_reward),
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
df_base.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_baseline.csv"), index=False)
all_results.append(df_base)

# ── Layer ablations ─────────────────────────────────────────────────────────
for tier_name, layer_range in TIERS.items():
    print(f"\n{'=' * 62}")
    print(f"TIER: {tier_name}")
    print("=" * 62)
    install_layer_hooks(layer_range)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_72b_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# ── Combined ────────────────────────────────────────────────────────────────
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_layer_ablations.csv"), index=False)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print("SUMMARY")
print("=" * 62)

base_mean = df_combined[df_combined["Tier"] == "baseline"]["points"].mean()

for tier in ["baseline"] + list(TIERS.keys()):
    sub   = df_combined[df_combined["Tier"] == tier]
    valid = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean()
    pct_max   = valid["chose_max"].mean() * 100
    collapsed = sub["collapsed"].sum()
    delta     = mean_pts - base_mean
    direction = "" if tier == "baseline" else ("anhedonic ↓" if delta < 0 else "hyperhedonic ↑")
    print(f"\n  {tier}:")
    print(f"    Mean points : {mean_pts:.2f}  (Δ={delta:+.2f}) {direction}")
    print(f"    Chose max   : {pct_max:.1f}%")
    print(f"    Collapsed   : {collapsed}/{len(sub)}")

print(f"\nSaved to: {OUTPUT_DIR}")
