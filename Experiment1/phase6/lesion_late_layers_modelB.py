"""
lesion_late_layers_modelB.py  —  Model B Layer Ablation Experiment
===================================================================
Identical structure to lesion_late_layers.py (Model A) but operates
on the Model B master core (4-domain intersection).

Model A neurons: geo+math 3σ intersection, then L18-27 subset  → 1,363 neurons
Model B neurons: geo+math+business_ethics+philosophy 3σ intersection → TBD

We test the same layer tiers as Model A to find whether:
  (a) The L18-27 concentration holds with the stricter 4-domain core
  (b) The cancellation paradox (mid vs late) still applies
  (c) A new minimal sufficient set emerges

TIERS:
  baseline      — no ablation (control)
  full_core     — all Model B master core neurons
  layers_18_27  — late circuit subset (Model A's winning range)
  layers_18_22  — first half of late
  layers_23_27  — second half of late
  layer_27      — single strongest late layer
  layer_17      — transition point (last mid layer)
  layers_9_17   — full mid circuit (expected hyperhedonic)
  layers_2_8    — early layers (expected negligible)

NEUTRAL MEANS: computed from all 4 domains (geo + math + business_ethics + philosophy)
               to match the 4-domain activation context used for neuron selection.

OUTPUT: results_modelB/ablation_results.csv
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

# Phase 6 activations directory (all 4 domains)
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase6/activations"
NEURONS_FILE    = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase6/modelB_master_incentive_core.csv"

# Evaluation dataset — same as Model A for direct comparability
INPUT_CSV       = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/ablation/core/data/full_experiment_100_rows.csv"

OUTPUT_DIR      = "results_modelB"

MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.7
TOP_P           = 0.95
NUM_RUNS        = 5
TOTAL_NEURONS   = 28 * 18944  # 530,432

# Neutral mean domains — all 4 for Model B
NEUTRAL_DOMAINS = ["geo", "math", "business_ethics", "philosophy"]

# Layer tiers to test — same grid as Model A for direct comparison
LAYER_TIERS = {
    "baseline":     None,           # no ablation
    "full_core":    "all",          # all Model B master core neurons
    "layers_18_27": (18, 27),       # Model A's winning range — KEY TEST
    "layers_18_22": (18, 22),       # first half of late
    "layers_23_27": (23, 27),       # second half of late
    "layer_27":     (27, 27),       # single strongest late layer
    "layer_17":     (17, 17),       # last mid-layer (transition)
    "layers_9_17":  (9,  17),       # full mid circuit
    "layers_2_8":   (2,  8),        # early (expected negligible)
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
# Neutral baseline means — averaged across ALL 4 domains
# =============================================================================
def compute_neutral_means():
    print(f"\nComputing neutral baseline means from {len(NEUTRAL_DOMAINS)} domains...")
    all_neutral = []
    for domain in NEUTRAL_DOMAINS:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\n"
                f"Run extract_activations_modelB.py first."
            )
        data    = torch.load(path, map_location="cpu")
        tensors = torch.stack(list(data.values())).float()
        all_neutral.append(tensors)
        print(f"  {domain}: {tensors.shape}")
    combined  = torch.cat(all_neutral, dim=0)   # [Q_total, 28, 18944]
    mean_acts = combined.mean(dim=0).numpy()    # [28, 18944]
    print(f"  Combined shape: {combined.shape}")
    print(f"  Mean activation array: {mean_acts.shape}  (mean={mean_acts.mean():.6f})")
    return mean_acts


# =============================================================================
# Build neuron tiers from Model B master core
# =============================================================================
def df_to_layer_dict(df):
    layer_dict = {}
    for _, row in df.iterrows():
        l, n = int(row['layer']), int(row['neuron'])
        layer_dict.setdefault(l, []).append(n)
    return {l: torch.tensor(ns).long() for l, ns in layer_dict.items()}


def build_neuron_tiers(mean_acts):
    if not os.path.exists(NEURONS_FILE):
        raise FileNotFoundError(
            f"{NEURONS_FILE} not found.\n"
            f"Run extract_neurons_modelB.py first."
        )
    df_core = pd.read_csv(NEURONS_FILE)
    total_core = len(df_core)
    print(f"\nModel B master core: {total_core} neurons across "
          f"{df_core['layer'].nunique()} layers")

    print(f"\nCore neurons per layer (L17–27):")
    for l in range(17, 28):
        n = len(df_core[df_core['layer'] == l])
        if n > 0:
            bar = '█' * min(n // 5, 40)
            print(f"  Layer {l:>2}: {n:>4} neurons  {bar}")

    tiers       = {}
    tier_counts = {}

    for tier_name, layer_range in LAYER_TIERS.items():
        if tier_name == "baseline":
            tiers[tier_name]       = {}
            tier_counts[tier_name] = 0
        elif layer_range == "all":
            tiers[tier_name]       = df_to_layer_dict(df_core)
            tier_counts[tier_name] = total_core
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
    print(f"  {'Tier':16s}  {'Layers':>9}  {'Neurons':>8}  {'% network':>10}  {'% of core':>10}")
    print(f"  {'-'*58}")
    for tier_name in LAYER_TIERS:
        n      = tier_counts[tier_name]
        pct_n  = n / TOTAL_NEURONS * 100
        pct_c  = n / total_core * 100 if total_core > 0 else 0
        lr     = LAYER_TIERS[tier_name]
        if lr is None:
            lr_str = "none"
        elif lr == "all":
            lr_str = "all"
        else:
            lr_str = f"{lr[0]}–{lr[1]}"
        print(f"  {tier_name:16s}  {lr_str:>9}  {n:>8,}  {pct_n:>9.4f}%  {pct_c:>9.1f}%")

    return tiers, tier_means, tier_counts


# =============================================================================
# Hook registration
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

        handles.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(
                make_hook(indices, mean_values)
            )
        )
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
    n_total     = len(prompts_df)

    for _, row in tqdm(prompts_df.iterrows(), total=n_total,
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
            "Model":     "B",
            "Tier":      tier_name,
            "Run_ID":    run_idx + 1,
            "ID":        row["ID"],
            "Prompt":    prompt_text,
            "Response":  response,
            "Choice":    choice,
            "Points":    points,
            "Collapsed": collapsed,
        })

    if n_collapsed > 0:
        print(f"  ⚠️  Collapse: {n_collapsed}/{n_total} "
              f"({n_collapsed/n_total*100:.1f}%)")
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, processor, lm_layers  = load_model()
    mean_acts                    = compute_neutral_means()
    tiers, tier_means, tier_counts = build_neuron_tiers(mean_acts)

    print(f"\nLoading evaluation prompts from {INPUT_CSV}...")
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

    # ── Quick summary ─────────────────────────────────────────────────────────
    df_clean = df_out[~df_out["Collapsed"]].copy()
    df_clean["Points"] = pd.to_numeric(df_clean["Points"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Points"])
    base_mean = df_clean[df_clean["Tier"] == "baseline"]["Points"].mean()

    print(f"\nCollapse rate per tier:")
    print(df_out.groupby("Tier")["Collapsed"].mean().mul(100).round(1).to_string())

    print(f"\nResults vs baseline (mean={base_mean:.2f}):")
    print(f"  {'Tier':16s}  {'Neurons':>8}  {'Mean pts':>9}  {'Δ':>7}  {'100-pt%':>8}  Direction")
    print(f"  {'-'*68}")
    for tier in LAYER_TIERS:
        sub = df_clean[df_clean["Tier"] == tier]
        if not len(sub):
            continue
        n    = tier_counts[tier]
        m    = sub["Points"].mean()
        d    = m - base_mean
        r100 = (sub["Points"] == 100).mean() * 100
        direction = "↓ anhedonic" if d < -1 else ("↑ hyperhedonic" if d > 1 else "≈ no effect")
        print(f"  {tier:16s}  {n:>8,}  {m:>9.2f}  {d:>+7.2f}  {r100:>7.1f}%  {direction}")


if __name__ == "__main__":
    main()