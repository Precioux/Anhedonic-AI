"""
run_full_experiment.py
======================
Runs three tiers — baseline, Model A, Model B — on full_experiment_100_rows.csv.
No sanity check. No knowledge dissociation. Behavioral evaluation only.

Model A : layers 18–27  | ~1,363 neurons | neutral means from geo + math
Model B : layers 18–27+ | all Model B neurons | neutral means from geo + math + business_ethics + philosophy

Usage:
    python run_full_experiment.py
    python run_full_experiment.py --runs 3
    python run_full_experiment.py --output_dir my_results
"""

import os
import re
import torch
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =============================================================================
# Paths — edit these if your layout differs
# =============================================================================
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"

# Neutral activations
ACT_DIR_A = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
ACT_DIR_B = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase6/activations"

NEUTRAL_DOMAINS_A = ["geo", "math"]
NEUTRAL_DOMAINS_B = ["geo", "math", "business_ethics", "philosophy"]

# Neuron CSVs
MODEL_A_NEURONS_CSV = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
MODEL_B_NEURONS_CSV = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase6/modelB_master_incentive_core.csv"
MODEL_A_LAYERS      = list(range(18, 28))

# Behavioral dataset
BEHAVIORAL_CSV = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/ablation/core/data/full_experiment_100_rows.csv"

# Generation settings
MAX_NEW_TOKENS = 300
TEMPERATURE    = 0.7
TOP_P          = 0.95
NUM_RUNS       = 3
OUTPUT_DIR     = "results_full_experiment"


# =============================================================================
# Helpers
# =============================================================================
def is_collapsed(text: str) -> bool:
    words = str(text).split()
    if words and max(len(w) for w in words) > 25:
        return True
    tokens = str(text).lower().split()
    if len(tokens) >= 8:
        ngrams = [" ".join(tokens[i:i+4]) for i in range(len(tokens) - 3)]
        for ng in ngrams:
            if ngrams.count(ng) > 3:
                return True
    return False


def parse_choice(response_text: str, prompt: str):
    point_map = {1: 100, 2: 67, 3: 33, 4: 0}
    patterns = [
        r'(?:option|choice|answer)\s*[:\-]?\s*([1-4])\b',
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


# =============================================================================
# Model loading
# =============================================================================
def load_model():
    print("=" * 60)
    print("Loading Qwen2-VL-7B …")
    print("=" * 60)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = model.model.language_model.layers
    print(f"  Layers available: {len(lm_layers)}")
    return model, processor, lm_layers


# =============================================================================
# Neutral means
# =============================================================================
def compute_neutral_means(act_dir: str, domains: list) -> np.ndarray:
    parts = []
    for domain in domains:
        path = os.path.join(act_dir, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    combined = torch.cat(parts, dim=0)
    return combined.mean(dim=0).numpy()   # shape: [n_layers, 18944]


# =============================================================================
# Neuron sets
# =============================================================================
def to_layer_dict(df: pd.DataFrame, layer_filter=None):
    d = {}
    if layer_filter is not None:
        df = df[df["layer"].isin(layer_filter)]
    for _, row in df.iterrows():
        l, n = int(row["layer"]), int(row["neuron"])
        d.setdefault(l, []).append(n)
    return {l: torch.tensor(ns).long() for l, ns in d.items()}


def make_means(layer_dict: dict, mean_acts: np.ndarray) -> dict:
    return {
        layer_idx: torch.tensor(mean_acts[layer_idx, neuron_indices.numpy()],
                                dtype=torch.bfloat16)
        for layer_idx, neuron_indices in layer_dict.items()
    }


def build_neuron_sets(mean_acts_A: np.ndarray, mean_acts_B: np.ndarray):
    df_a = pd.read_csv(MODEL_A_NEURONS_CSV)
    df_b = pd.read_csv(MODEL_B_NEURONS_CSV)

    modelA_ld = to_layer_dict(df_a, layer_filter=MODEL_A_LAYERS)
    modelB_ld = to_layer_dict(df_b)

    sets = {
        "baseline": ({},         {}),
        "modelA":   (modelA_ld,  make_means(modelA_ld, mean_acts_A)),
        "modelB":   (modelB_ld,  make_means(modelB_ld, mean_acts_B)),
    }

    print("\nNeuron set sizes:")
    for name, (ld, _) in sets.items():
        n = sum(len(v) for v in ld.values())
        print(f"  {name:<12}: {n:>7,} neurons")

    return sets


# =============================================================================
# Hooks
# =============================================================================
def register_hooks(lm_layers, layer_dict: dict, tier_means: dict) -> list:
    handles = []
    for layer_idx, neuron_indices in layer_dict.items():
        idx   = neuron_indices.to("cuda")
        means = tier_means[layer_idx].to("cuda")

        def _make_hook(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook

        h = lm_layers[layer_idx].mlp.act_fn.register_forward_hook(
            _make_hook(idx, means)
        )
        handles.append(h)
    return handles


# =============================================================================
# Behavioral evaluation
# =============================================================================
def run_behavioral(model, processor, prompts_df: pd.DataFrame,
                   tier_name: str, run_idx: int, model_label: str) -> list:
    results, n_collapsed = [], 0
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df),
                       desc=f"  [{tier_name}] run {run_idx + 1}/{NUM_RUNS}"):
        prompt = row["Full_Prompt"]
        text   = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE, do_sample=True, top_p=TOP_P
            )
        response = processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        collapsed    = is_collapsed(response)
        n_collapsed += int(collapsed)
        choice, points = parse_choice(response, prompt)

        results.append({
            "Model":     model_label,
            "Tier":      tier_name,
            "Run_ID":    run_idx + 1,
            "ID":        row["ID"],
            "Response":  response,
            "Choice":    choice,
            "Points":    points,
            "Collapsed": collapsed,
        })

    if n_collapsed:
        print(f"  ⚠  Collapsed: {n_collapsed}/{len(prompts_df)}")
    return results


# =============================================================================
# Main
# =============================================================================
def main(num_runs: int = NUM_RUNS, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "behavioral_results.csv")

    # Load model once
    model, processor, lm_layers = load_model()

    # Neutral means (computed separately for A and B)
    print("\nComputing neutral means for Model A (geo + math) …")
    mean_acts_A = compute_neutral_means(ACT_DIR_A, NEUTRAL_DOMAINS_A)

    print("Computing neutral means for Model B (geo + math + business_ethics + philosophy) …")
    mean_acts_B = compute_neutral_means(ACT_DIR_B, NEUTRAL_DOMAINS_B)

    # Neuron sets
    neuron_sets = build_neuron_sets(mean_acts_A, mean_acts_B)

    # Load data
    behavioral_df = pd.read_csv(BEHAVIORAL_CSV)
    print(f"\nBehavioral prompts: {len(behavioral_df)} rows")

    # Tier config:  (set_key, model_label, display_name)
    TIERS = [
        ("baseline", "Baseline", "BASELINE"),
        ("modelA",   "Model A",  "MODEL A  (layers 18-27 | ~1,363 neurons | Δ=−9.81 pts)"),
        ("modelB",   "Model B",  "MODEL B  (Model A + new BE/phil neurons)"),
    ]

    all_results = []

    for set_key, model_label, display in TIERS:
        layer_dict, tier_means = neuron_sets[set_key]
        n_neurons = sum(len(v) for v in layer_dict.values())

        print(f"\n{'='*60}")
        print(f"  {display}  ({n_neurons:,} neurons)")
        print(f"{'='*60}")

        for run_idx in range(num_runs):
            handles = register_hooks(lm_layers, layer_dict, tier_means)
            try:
                results = run_behavioral(
                    model, processor, behavioral_df,
                    tier_name=set_key, run_idx=run_idx, model_label=model_label
                )
                all_results.extend(results)
            finally:
                for h in handles:
                    h.remove()

            # Save after every run
            pd.DataFrame(all_results).to_csv(out_csv, index=False)
            print(f"  ✓ Saved {len(all_results)} rows → {out_csv}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    df       = pd.read_csv(out_csv)
    df_clean = df[~df["Collapsed"]].copy()
    df_clean["Points"] = pd.to_numeric(df_clean["Points"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Points"])

    base_mean = df_clean[df_clean["Tier"] == "baseline"]["Points"].mean()

    print(f"\nBehavioral results  (baseline mean = {base_mean:.2f} pts):")
    print(f"  {'Tier':<12}  {'Model':<12}  {'N':>6}  {'Mean':>7}  {'Δ':>7}  {'100%':>6}  {'Collapsed':>10}")
    print(f"  {'-'*70}")

    for set_key, model_label, _ in TIERS:
        sub = df_clean[df_clean["Tier"] == set_key]
        if sub.empty:
            continue
        mean_pts = sub["Points"].mean()
        pct_100  = (sub["Points"] == 100).mean() * 100
        n_col    = df[df["Tier"] == set_key]["Collapsed"].sum()
        n_total  = df[df["Tier"] == set_key].shape[0]
        n_neurons = sum(len(v) for v in neuron_sets[set_key][0].values())
        print(f"  {set_key:<12}  {model_label:<12}  {n_neurons:>6,}  "
              f"{mean_pts:>7.2f}  {mean_pts - base_mean:>+7.2f}  "
              f"{pct_100:>5.1f}%  {n_col:>5}/{n_total}")

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline / Model A / Model B — behavioral eval")
    parser.add_argument("--runs",       type=int, default=NUM_RUNS,    help="Number of runs per tier (default: 5)")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,  help="Output directory")
    args = parser.parse_args()
    main(num_runs=args.runs, output_dir=args.output_dir)
