"""
eval_72b_mapped_layers.py
================================================================================
Maps 7B Model A layers (L18-27) to 72B proportionally (80/28 scale),
giving 72B target layers L51-77.

Then ablates ALL neurons found in those layers from master_incentive_core.csv.
This is NOT neuron-index transfer (impossible across model sizes) — it uses
the 72B's own discovered neurons that happen to live in the proportionally
equivalent layer range.

7B → 72B layer mapping (×2.857):
  L18 → L51   L19 → L54   L20 → L57   L21 → L60
  L22 → L63   L23 → L66   L24 → L69   L25 → L71
  L26 → L74   L27 → L77

Target 72B layers: 51, 54, 57, 60, 63, 66, 69, 71, 74, 77
Also tests the full contiguous range L51-77 in a separate tier.

Tiers:
  baseline          — no ablation
  mapped_exact      — only the 10 mapped layers (L51,54,57,60,63,66,69,71,74,77)
  mapped_L51_77     — all layers in range 51-77 that have core neurons
================================================================================
"""

import os, re, json, torch
import pandas as pd
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-72b"
CORE_CSV   = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/neurons/orig/master_incentive_core.csv"
ACT_DIR    = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
EVAL_CSV   = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/data/origin_math_eval.csv"
OUTPUT_DIR = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/eval_mapped"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

# =============================================================================
# Build neuron map from core CSV — keyed by layer
# =============================================================================
print("Loading master_incentive_core.csv ...")
df_core = pd.read_csv(CORE_CSV)
core_by_layer: dict[int, list[int]] = {}
for layer, grp in df_core.groupby("layer"):
    core_by_layer[int(layer)] = sorted(grp["neuron"].tolist())

print(f"  Core neurons across {len(core_by_layer)} layers, {len(df_core):,} total\n")

# =============================================================================
# Define ablation tiers
# =============================================================================

# Proportional mapping: 7B L18-27 → 72B (round to nearest int)
# 7B layer × (80/28)
mapped_exact_layers = sorted(set(round(l * 80 / 28) for l in range(18, 28)))
print(f"Proportionally mapped layers (7B L18-27 → 72B): {mapped_exact_layers}")

# Full contiguous range L51-77
full_range_layers = list(range(51, 78))

# Build neuron maps for each tier — only layers that exist in core
def build_neuron_map(layer_list: list[int]) -> dict[int, list[int]]:
    nm = {}
    for l in layer_list:
        if l in core_by_layer and len(core_by_layer[l]) > 0:
            nm[l] = core_by_layer[l]
    return nm

mapped_exact_neurons = build_neuron_map(mapped_exact_layers)
full_range_neurons   = build_neuron_map(full_range_layers)

def summarize(name, nm):
    total = sum(len(v) for v in nm.values())
    pct   = total / TOTAL_NEURONS * 100
    print(f"  {name}: {len(nm)} layers, {total:,} neurons ({pct:.4f}%)")
    print(f"    Layers: {sorted(nm.keys())}")

print()
summarize("mapped_exact (10 layers)", mapped_exact_neurons)
summarize("mapped_L51_77 (all in range)", full_range_neurons)

TIERS = {
    "mapped_exact":  mapped_exact_neurons,
    "mapped_L51_77": full_range_neurons,
}

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
    return torch.cat(parts, dim=0).mean(dim=0).numpy()  # [80, 29568]

print("Loading neutral means ...")
mean_acts = load_neutral_means()
print(f"  Shape: {mean_acts.shape}  ✓\n")

# =============================================================================
# Hook management — clamp specific neurons per layer
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
        h = lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        _hooks.append(h)
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
    q      = int(match.group(1))
    points = int(row[f"Reward_{q}"])
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
for tier_name, neuron_map in TIERS.items():
    print(f"\n{'=' * 62}")
    print(f"TIER: {tier_name}")
    print("=" * 62)
    install_neuron_hooks(neuron_map)
    df_tier = run_eval(df_eval, tier_name)
    remove_hooks()
    df_tier.to_csv(os.path.join(OUTPUT_DIR, f"eval_{tier_name}.csv"), index=False)
    all_results.append(df_tier)

# ── Combined ────────────────────────────────────────────────────────────────
df_combined = pd.concat(all_results, ignore_index=True)
df_combined.to_csv(os.path.join(OUTPUT_DIR, "eval_72b_mapped_combined.csv"), index=False)

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'=' * 62}")
print("SUMMARY")
print("=" * 62)
print(f"\n  {'Tier':<22}  {'Mean pts':>9}  {'Δ':>7}  {'Chose max':>10}  {'Collapsed':>10}  Direction")
print(f"  {'-'*78}")

for tier in ["baseline"] + list(TIERS.keys()):
    sub   = df_combined[df_combined["Tier"] == tier]
    valid = sub[~sub["collapsed"]]
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
    print(f"  {tier:<22}  {mean_pts:>9.2f}  {delta:>+7.2f}  {pct_max:>9.1f}%  "
          f"{collapsed:>4}/{len(sub)}        {direction}")

print(f"\nSaved to: {OUTPUT_DIR}")
