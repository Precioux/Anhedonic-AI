"""
eval_72b_orig.py
================================================================================
Evaluates two models on origin_math_eval.csv:
  1. Baseline  — Qwen2-VL-72B unmodified (NF4)
  2. Ablated   — same model + master_core neurons clamped to neutral mean

Results saved to:
  eval_72b_baseline.csv
  eval_72b_ablated.csv
  eval_72b_combined.csv   ← both tiers together, ready for analysis

Run on H200:
  python eval_72b_orig.py
================================================================================
"""

import os, json, re, torch
import pandas as pd
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH   = "/mnt/mahdipou/models/qwen2-vl-72b"
CORE_CSV     = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/neurons/orig/master_incentive_core.csv"
ACT_DIR      = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
EVAL_CSV     = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/data/origin_math_eval.csv"
OUTPUT_DIR   = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/eval"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 72B constants
NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# Reward values in the dataset
REWARD_VALUES = [10, 20, 30, 40]

# =============================================================================
# Load model ONCE (NF4, device_map={"": 0})
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
# Neutral mean activations (for ablation)
# =============================================================================
def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACT_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()  # [80, 29568]

# =============================================================================
# Install / remove ablation hooks
# =============================================================================
_hooks = []

def install_ablation_hooks(mean_acts: np.ndarray, neuron_map: dict[int, list[int]]):
    global _hooks
    remove_ablation_hooks()
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons, dtype=torch.long).to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(module, _inp, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        h = lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        _hooks.append(h)
    print(f"  Ablation hooks installed: {sum(len(v) for v in neuron_map.values()):,} neurons across {len(neuron_map)} layers")

def remove_ablation_hooks():
    global _hooks
    for h in _hooks:
        h.remove()
    _hooks = []

# =============================================================================
# Generation
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
            do_sample=False,          # greedy — deterministic, matches your existing eval protocol
        )
    trimmed = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True).strip()

# =============================================================================
# Parse response → chosen question number + points earned
# =============================================================================
def parse_response(response: str, row: pd.Series) -> dict:
    """
    Extract chosen question (1-4) from response.
    Returns chosen question index, reward earned, whether it was the max reward,
    and whether the model collapsed (no valid choice found).
    """
    # Look for a digit 1-4 anywhere in the response
    match = re.search(r'\b([1-4])\b', response)
    if not match:
        return {
            "chosen_q":      None,
            "points":        None,
            "chose_max":     None,
            "collapsed":     True,
            "raw_response":  response,
        }

    q = int(match.group(1))   # 1-indexed
    reward_col = f"Reward_{q}"
    points = int(row[reward_col])

    max_reward = max(int(row[f"Reward_{i}"]) for i in range(1, 5))
    chose_max  = (points == max_reward)

    return {
        "chosen_q":     q,
        "points":       points,
        "chose_max":    chose_max,
        "collapsed":    False,
        "raw_response": response,
    }

# =============================================================================
# Run evaluation for one tier
# =============================================================================
def run_eval(df: pd.DataFrame, tier_name: str) -> pd.DataFrame:
    records = []
    n = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        response = generate(str(row["Full_Prompt"]))
        parsed   = parse_response(response, row)
        records.append({
            "ID":           row["ID"],
            "Subset":       row["Subset"],
            "Reward_Order": row["Reward_Order"],
            "Tier":         tier_name,
            **parsed,
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            done = [r for r in records if not r["collapsed"]]
            mean_pts = np.mean([r["points"] for r in done]) if done else float("nan")
            pct_max  = np.mean([r["chose_max"] for r in done]) * 100 if done else float("nan")
            print(f"  [{tier_name}] {i+1}/{n}  |  mean pts={mean_pts:.1f}  chose_max={pct_max:.1f}%  collapsed={sum(r['collapsed'] for r in records)}")

    return pd.DataFrame(records)

# =============================================================================
# Main
# =============================================================================
df_eval = pd.read_csv(EVAL_CSV)
print(f"Eval dataset: {len(df_eval)} rows\n")

# ── 1. Baseline (no hooks) ──────────────────────────────────────────────────
print("=" * 62)
print("TIER 1: BASELINE")
print("=" * 62)
df_base = run_eval(df_eval, "baseline")
df_base.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_baseline.csv"), index=False)
print(f"Saved baseline results.\n")

# ── 2. Load ablation components ─────────────────────────────────────────────
print("=" * 62)
print("TIER 2: ABLATED (master_core, orig neurons)")
print("=" * 62)

print("Loading master_core neurons ...")
df_core = pd.read_csv(CORE_CSV)
neuron_map: dict[int, list[int]] = {}
for layer, grp in df_core.groupby("layer"):
    neuron_map[int(layer)] = sorted(grp["neuron"].tolist())
total_ablated = sum(len(v) for v in neuron_map.values())
pct = total_ablated / TOTAL_NEURONS * 100
print(f"  {total_ablated:,} neurons across {len(neuron_map)} layers ({pct:.4f}% of network)")

print("Loading neutral means ...")
mean_acts = load_neutral_means()

install_ablation_hooks(mean_acts, neuron_map)
df_ablated = run_eval(df_eval, "ablated_core")
remove_ablation_hooks()
df_ablated.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_ablated.csv"), index=False)
print(f"Saved ablated results.\n")

# ── 3. Combined output ───────────────────────────────────────────────────────
df_combined = pd.concat([df_base, df_ablated], ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_combined.csv"), index=False)

# ── 4. Summary ───────────────────────────────────────────────────────────────
print("=" * 62)
print("SUMMARY")
print("=" * 62)
for tier in ["baseline", "ablated_core"]:
    sub = df_combined[df_combined["Tier"] == tier]
    valid = sub[~sub["collapsed"]]
    mean_pts  = valid["points"].mean()
    pct_max   = valid["chose_max"].mean() * 100
    collapsed = sub["collapsed"].sum()
    print(f"\n  {tier}:")
    print(f"    Mean points : {mean_pts:.2f}")
    print(f"    Chose max   : {pct_max:.1f}%")
    print(f"    Collapsed   : {collapsed}/{len(sub)}")

if "baseline" in df_combined["Tier"].values and "ablated_core" in df_combined["Tier"].values:
    base_mean = df_combined[df_combined["Tier"]=="baseline"]["points"].mean()
    abla_mean = df_combined[df_combined["Tier"]=="ablated_core"]["points"].mean()
    print(f"\n  Δ (ablated − baseline): {abla_mean - base_mean:+.2f} pts")
    direction = "anhedonic ↓" if abla_mean < base_mean else "hyperhedonic ↑"
    print(f"  Direction: {direction}")

print(f"\nAll results saved to: {OUTPUT_DIR}")
