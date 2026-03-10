"""
lesion_core_v3.py — Anhedonic AI Ablation: New Tier Scenarios
=============================================================
Tests four new neuron sets alongside baseline:
  - money_univ  : 5,467 neurons — money-universal only (3σ in money/geo AND money/math)
  - reward_univ : 5,558 neurons — reward-universal only (3σ in reward/geo AND reward/math)
  - union        : ~7,497 neurons — money_univ ∪ reward_univ
  - master_core  : 3,528 neurons — intersection (from previous run, kept for reference)

COLLAPSE DETECTOR
  reward_univ at 5,558 neurons previously caused space-fusion and looping output.
  A collapse detector checks every response: if >20% of word-boundary characters
  are missing (fused tokens) OR the response loops (same phrase >3 times),
  the response is flagged as COLLAPSED and the tier is soft-warned after each run.
  Inference continues — we want to quantify collapse rate, not skip it.

OUTPUT
  results_v3/ablation_results.csv
  Columns: Tier, Run_ID, ID, Prompt, Response, Choice, Points, Collapsed
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
MODEL_PATH       = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR  = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_DIR      = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction"
INPUT_CSV        = "core/data/full_experiment_100_rows.csv"
OUTPUT_DIR       = "results_v3"

MAX_NEW_TOKENS   = 300
TEMPERATURE      = 0.7
TOP_P            = 0.95
NUM_RUNS         = 5

TOTAL_NEURONS    = 28 * 18944  # 530,432

# All tiers to run — including potentially collapsing ones
# Set to False to skip a tier
RUN_TIERS = {
    "baseline":    True,
    "money_univ":  True,
    "reward_univ": True,
    "union":       True,
    "master_core": True,   # kept for direct comparison
}


# =============================================================================
# Collapse Detector
# =============================================================================
def is_collapsed(text):
    """
    Returns True if the response shows signs of model collapse:
      1. Space fusion: any token longer than 25 chars (words fused together)
      2. Looping: any phrase of 4+ words repeated more than 3 times
    """
    # Space fusion check
    words = str(text).split()
    if words and max(len(w) for w in words) > 25:
        return True

    # Loop detection: check for repeated 4-gram
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
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = model.model.language_model.layers
    print(f"Language model layers: {len(lm_layers)}")
    return model, processor, lm_layers


# =============================================================================
# Neutral baseline means (for mean ablation)
# =============================================================================
def compute_neutral_means():
    print("\nComputing neutral baseline means for mean ablation...")
    all_neutral = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        tensors = torch.stack(list(data.values())).float()  # [100, 28, 18944]
        all_neutral.append(tensors)
    combined = torch.cat(all_neutral, dim=0)              # [200, 28, 18944]
    mean_acts = combined.mean(dim=0).numpy()              # [28, 18944]
    print(f"  Neutral mean shape: {mean_acts.shape}  "
          f"(mean={mean_acts.mean():.5f}, std={mean_acts.std():.5f})")
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
    # Load the three pre-computed CSVs
    df_core  = pd.read_csv(os.path.join(NEURONS_DIR, "master_incentive_core.csv"))
    df_rew   = pd.read_csv(os.path.join(NEURONS_DIR, "universal_reward_neurons.csv"))
    df_money = pd.read_csv(os.path.join(NEURONS_DIR, "universal_money_neurons.csv"))

    print(f"\nLoaded neuron sets:")
    print(f"  master_core : {len(df_core):,} neurons  ({len(df_core)/TOTAL_NEURONS*100:.3f}%)")
    print(f"  reward_univ : {len(df_rew):,} neurons   ({len(df_rew)/TOTAL_NEURONS*100:.3f}%)")
    print(f"  money_univ  : {len(df_money):,} neurons   ({len(df_money)/TOTAL_NEURONS*100:.3f}%)")

    # Build union (money ∪ reward) — deduplicated
    set_rew   = set(zip(df_rew['layer'],   df_rew['neuron']))
    set_money = set(zip(df_money['layer'], df_money['neuron']))
    union_set = set_rew | set_money
    df_union  = pd.DataFrame(sorted(union_set), columns=['layer', 'neuron'])
    print(f"  union       : {len(df_union):,} neurons   ({len(df_union)/TOTAL_NEURONS*100:.3f}%)")
    print(f"  (reward ∩ money = {len(set_rew & set_money):,} shared neurons)")

    tiers = {
        "baseline":    {},
        "money_univ":  df_to_layer_dict(df_money),
        "reward_univ": df_to_layer_dict(df_rew),
        "union":       df_to_layer_dict(df_union),
        "master_core": df_to_layer_dict(df_core),
    }

    # Precompute mean replacement values per tier
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
    for tier_name, layer_dict in tiers.items():
        n   = sum(len(v) for v in layer_dict.values())
        pct = n / TOTAL_NEURONS * 100
        print(f"  {tier_name:12s}: {n:>6,} neurons  ({pct:.3f}%)")

    return tiers, tier_means


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
# Inference helpers
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
    results   = []
    n_collapsed = 0

    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df),
                       desc=f"  [{tier_name}] run {run_idx+1}"):
        prompt_text = row["Full_Prompt"]
        messages    = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text        = processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        inputs      = processor(text=[text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
            )

        trimmed  = [out[len(inp):] for inp, out in
                    zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        collapsed      = is_collapsed(response)
        n_collapsed   += int(collapsed)
        choice, points = parse_choice(response, prompt_text)

        results.append({
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
        pct = n_collapsed / len(prompts_df) * 100
        print(f"  ⚠️  Collapse detected: {n_collapsed}/{len(prompts_df)} responses "
              f"({pct:.1f}%) — space fusion or looping output")

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, processor, lm_layers = load_model()
    mean_acts                   = compute_neutral_means()
    tiers, tier_means           = build_neuron_tiers(mean_acts)

    print(f"\nLoading prompts from {INPUT_CSV}...")
    prompts_df = pd.read_csv(INPUT_CSV)
    print(f"  {len(prompts_df)} questions per run")

    all_results = []

    for tier_name, layer_dict in tiers.items():
        if not RUN_TIERS.get(tier_name, False):
            print(f"\nSkipping tier: {tier_name}")
            continue

        n_neurons = sum(len(v) for v in layer_dict.values())
        pct       = n_neurons / TOTAL_NEURONS * 100
        print(f"\n{'='*60}")
        print(f"TIER: {tier_name.upper()}  ({n_neurons:,} neurons, {pct:.3f}%)")
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

            out_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
            pd.DataFrame(all_results).to_csv(out_path, index=False)
            print(f"  Saved {len(all_results)} rows → {out_path}")

    df_out   = pd.DataFrame(all_results)
    out_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE — {len(df_out):,} total responses saved to {out_path}")
    print(f"{'='*60}")

    print(f"\nCollapse rate per tier:")
    print(df_out.groupby("Tier")["Collapsed"].mean().mul(100).round(1).to_string())

    print(f"\nMean points chosen per tier (non-collapsed only):")
    clean = df_out[~df_out["Collapsed"]]
    print(clean.groupby("Tier")["Points"].mean().round(1).to_string())

    print(f"\nChoice distribution per tier (non-collapsed):")
    print(clean.groupby(["Tier","Points"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()