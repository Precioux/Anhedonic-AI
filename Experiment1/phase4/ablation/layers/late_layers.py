"""
lesion_late_layers.py — Late Layer Ablation Experiment
=======================================================
Tests the reward-value circuit hypothesis:
  Late layers (18–27) are predicted to be the causal driver of
  the anhedonic effect seen in master_core.

TIERS (all subsets of master_core neurons):
  baseline      :     0 neurons  (0.000%)   control
  layers_18_27  : ~1,363 neurons (~0.257%)  full late circuit — KEY TEST
  layers_18_22  :   ~607 neurons (~0.114%)  first half of late
  layers_23_27  :   ~756 neurons (~0.143%)  second half of late
  layer_17      :   301 neurons  (0.057%)   transition point — last mid layer
  layer_18      :   182 neurons  (0.034%)   transition point — first late layer
  layer_27      :   194 neurons  (0.037%)   strongest single late layer
  layers_2_8    :    49 neurons  (0.009%)   early layers — expected negligible

HYPOTHESIS:
  layers_18_27 alone will produce Δ ≈ −7 (strongly anhedonic)
  This would confirm the double dissociation:
    Mid (9–17)  = effort-cost circuit  → ablation makes model MORE reward-seeking
    Late (18–27) = reward-value circuit → ablation makes model LESS reward-seeking

DATASET: full_experiment_100_rows.csv (100 held-out questions)
OUTPUT:  results_late/ablation_results.csv
"""

import os
import re
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_FILE    = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
INPUT_CSV       = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/ablation/core/data/full_experiment_100_rows.csv"
OUTPUT_DIR      = "results_late"

MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.7
TOP_P           = 0.95
NUM_RUNS        = 5
TOTAL_NEURONS   = 28 * 18944  # 530,432

# Layer ranges (inclusive). None = all layers (master_core reference)
LAYER_TIERS = {
    "baseline":     None,
    "layers_18_27": (18, 27),   # full late circuit — THE KEY TEST
    "layers_18_22": (18, 22),   # first half of late
    "layers_23_27": (23, 27),   # second half of late
    "layer_17":     (17, 17),   # last mid-layer (transition point)
    "layer_18":     (18, 18),   # first late-layer (transition point)
    "layer_27":     (27, 27),   # strongest single late layer
    "layers_2_8":   (2,  8),    # early layers — expected ~0 effect
}


# =============================================================================
# Collapse detector
# =============================================================================
def is_collapsed(text):
    words = str(text).split()
    if words and max(len(w) for w in words) > 25:
        return True
    tokens = str(text).lower().split()
    if len(tokens) >= 8:
        ngrams = [' '.join(tokens[i:i+4]) for i in range(len(tokens)-3)]
        for ng in ngrams:
            if ngrams.count(ng) > 3:
                return True
    return False


# =============================================================================
# Load model
# =============================================================================
def load_model():
    print("=" * 60)
    print("Loading Qwen2-VL-7B in bfloat16 (no quantization)...")
    print("=" * 60)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = model.model.language_model.layers
    print(f"Language model layers: {len(lm_layers)}")
    return model, processor, lm_layers


# =============================================================================
# Neutral baseline means
# =============================================================================
def compute_neutral_means():
    print("\nComputing neutral baseline means...")
    all_neutral = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        tensors = torch.stack(list(data.values())).float()
        all_neutral.append(tensors)
    combined  = torch.cat(all_neutral, dim=0)
    mean_acts = combined.mean(dim=0).numpy()
    print(f"  Shape: {mean_acts.shape}  (mean={mean_acts.mean():.5f})")
    return mean_acts


# =============================================================================
# Build neuron tiers
# =============================================================================
def df_to_layer_dict(df):
    layer_dict = {}
    for _, row in df.iterrows():
        l, n = int(row['layer']), int(row['neuron'])
        layer_dict.setdefault(l, []).append(n)
    return {l: torch.tensor(ns).long() for l, ns in layer_dict.items()}


def build_neuron_tiers(mean_acts):
    df_core = pd.read_csv(NEURONS_FILE)
    print(f"\nMaster core: {len(df_core)} neurons across {df_core['layer'].nunique()} layers")

    # Print layer counts for late layers so we know exact numbers
    print(f"\nMaster core neurons per layer (late layers 17–27):")
    for l in range(17, 28):
        n = len(df_core[df_core['layer']==l])
        if n > 0:
            print(f"  Layer {l:>2}: {n:>4} neurons")

    tiers       = {}
    tier_counts = {}

    for tier_name, layer_range in LAYER_TIERS.items():
        if tier_name == "baseline":
            tiers[tier_name]       = {}
            tier_counts[tier_name] = 0
        else:
            lo, hi  = layer_range
            subset  = df_core[(df_core['layer'] >= lo) & (df_core['layer'] <= hi)]
            tiers[tier_name]       = df_to_layer_dict(subset)
            tier_counts[tier_name] = len(subset)

    # Precompute mean replacement tensors
    tier_means = {"baseline": {}}
    for tier_name, layer_dict in tiers.items():
        if tier_name == "baseline":
            continue
        tier_means[tier_name] = {}
        for layer_idx, neuron_indices in layer_dict.items():
            vals = mean_acts[layer_idx, neuron_indices.numpy()]
            tier_means[tier_name][layer_idx] = torch.tensor(
                vals, dtype=torch.bfloat16
            )

    print(f"\nTier summary:")
    print(f"  {'Tier':15s}  {'Layers':>8}  {'Neurons':>8}  {'% network':>10}  {'% of core':>10}")
    print(f"  {'-'*60}")
    for tier_name in LAYER_TIERS:
        n     = tier_counts[tier_name]
        pct_n = n / TOTAL_NEURONS * 100
        pct_c = n / len(df_core) * 100
        lr    = LAYER_TIERS[tier_name]
        lr_str = f"{lr[0]}–{lr[1]}" if lr else "none"
        print(f"  {tier_name:15s}  {lr_str:>8}  {n:>8,}  {pct_n:>9.4f}%  {pct_c:>9.1f}%")

    return tiers, tier_means, tier_counts


# =============================================================================
# Hooks
# =============================================================================
def register_hooks(lm_layers, layer_dict, tier_means_for_tier):
    handles = []
    for layer_idx, neuron_indices in layer_dict.items():
        indices     = neuron_indices.to("cuda")
        mean_values = tier_means_for_tier[layer_idx].to("cuda")

        def make_hook(idx, means):
            def hook(module, input, output):
                output[:, :, idx] = means.unsqueeze(0).unsqueeze(0)
                return output
            return hook

        handle = lm_layers[layer_idx].mlp.act_fn.register_forward_hook(
            make_hook(indices, mean_values)
        )
        handles.append(handle)
    return handles


# =============================================================================
# Inference
# =============================================================================
def parse_choice(response_text, prompt_text):
    point_map = {}
    for m in re.finditer(r'(\d+)\.\s+.+?\((\d+) points?\)', prompt_text):
        point_map[int(m.group(1))] = int(m.group(2))
    patterns = [
        r'(?:question|choice|option|number)\s*([1-4])',
        r'\bI(?:\'ll| will| choose| pick| select)\b.{0,30}?([1-4])\b',
        r'^([1-4])[\.:\)]\s',
        r'\b([1-4])\b',
    ]
    for pat in patterns:
        m = re.search(pat, response_text, re.IGNORECASE | re.MULTILINE)
        if m:
            choice = int(m.group(1))
            return choice, point_map.get(choice)
    return None, None


def run_inference(model, processor, prompts_df, tier_name, run_idx):
    results     = []
    n_collapsed = 0
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df),
                       desc=f"  [{tier_name}] run {run_idx+1}"):
        prompt_text = row["Full_Prompt"]
        messages    = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text        = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE, do_sample=True, top_p=TOP_P,
            )
        trimmed  = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        collapsed    = is_collapsed(response)
        n_collapsed += int(collapsed)
        choice, points = parse_choice(response, prompt_text)
        results.append({
            "Tier": tier_name, "Run_ID": run_idx+1, "ID": row["ID"],
            "Prompt": prompt_text, "Response": response,
            "Choice": choice, "Points": points, "Collapsed": collapsed,
        })
    if n_collapsed > 0:
        print(f"  ⚠️  Collapse: {n_collapsed}/{len(prompts_df)} "
              f"({n_collapsed/len(prompts_df)*100:.1f}%)")
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, processor, lm_layers = load_model()
    mean_acts                   = compute_neutral_means()
    tiers, tier_means, tier_counts = build_neuron_tiers(mean_acts)

    print(f"\nLoading prompts from {INPUT_CSV}...")
    prompts_df = pd.read_csv(INPUT_CSV)
    print(f"  {len(prompts_df)} questions per run")

    all_results = []
    out_path    = os.path.join(OUTPUT_DIR, "ablation_results.csv")

    for tier_name in LAYER_TIERS:
        layer_dict = tiers[tier_name]
        n_neurons  = tier_counts[tier_name]
        pct        = n_neurons / TOTAL_NEURONS * 100

        print(f"\n{'='*60}")
        print(f"TIER: {tier_name.upper()}  ({n_neurons:,} neurons, {pct:.4f}%)")
        print(f"{'='*60}")

        for run_idx in range(NUM_RUNS):
            handles = register_hooks(lm_layers, layer_dict,
                                     tier_means.get(tier_name, {}))
            try:
                results = run_inference(model, processor, prompts_df,
                                        tier_name, run_idx)
                all_results.extend(results)
            finally:
                for h in handles:
                    h.remove()

            pd.DataFrame(all_results).to_csv(out_path, index=False)
            print(f"  Saved {len(all_results)} rows → {out_path}")

    df_out = pd.DataFrame(all_results)
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE — {len(df_out):,} total responses → {out_path}")
    print(f"{'='*60}")

    # Quick summary
    df_clean = df_out[~df_out["Collapsed"]].copy()
    df_clean["Points"] = pd.to_numeric(df_clean["Points"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Points"])

    base_mean = df_clean[df_clean["Tier"]=="baseline"]["Points"].mean()

    print(f"\nCollapse rate per tier:")
    print(df_out.groupby("Tier")["Collapsed"].mean().mul(100).round(1).to_string())

    print(f"\nResults vs baseline (mean={base_mean:.2f}):")
    print(f"  {'Tier':15s}  {'Neurons':>8}  {'Mean pts':>9}  {'Δ':>7}  {'100-pt%':>8}")
    for tier in LAYER_TIERS:
        sub = df_clean[df_clean["Tier"]==tier]
        if not len(sub): continue
        n   = tier_counts[tier]
        m   = sub["Points"].mean()
        d   = m - base_mean
        r100 = (sub["Points"]==100).mean()*100
        direction = "↓ anhedonic" if d < -1 else ("↑ hyperhedonic" if d > 1 else "≈ no effect")
        print(f"  {tier:15s}  {n:>8,}  {m:>9.2f}  {d:>+7.2f}  {r100:>7.1f}%  {direction}")


if __name__ == "__main__":
    main()