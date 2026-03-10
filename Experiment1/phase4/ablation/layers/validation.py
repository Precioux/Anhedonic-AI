"""
validation.py — Three-Step Validation of the Anhedonic AI Finding
==================================================================

STEP 1 — PATCH-BACK CAUSALITY
  Proves layer_27 neurons CAUSE the behavioral effect.
  Tiers: baseline | layers_18_27 | layers_18_26 (patched — layer_27 restored)
  Prediction: layers_18_26 should recover toward baseline vs layers_18_27.

STEP 2 — CONTROL ABLATION (SPECIFICITY)  
  Proves the effect is about WHICH neurons, not just layer_27 damage.
  Tiers: baseline | layer_27_mastercore | layer_27_random
  Prediction: random neurons → no behavioral effect.

STEP 3 — KNOWLEDGE DISSOCIATION
  Proves the ablated model still UNDERSTANDS reward — impaired motivation,
  not impaired cognition. That is the definition of anhedonia.
  Tiers: baseline | layers_18_27 | layer_27
  Task: 20 factual questions about value/reward with correct answers.

All results saved incrementally. Total runtime ~3.5 hours.
"""

import os
import re
import csv
import json
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_FILE    = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
BEHAVIORAL_CSV  = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/ablation/core/data/full_experiment_100_rows.csv"
KNOWLEDGE_CSV   = "knowledge_dissociation_questions.csv"

OUTPUT_DIR      = "results_validation"
MAX_NEW_TOKENS  = 300
TEMPERATURE     = 0.7
TOP_P           = 0.95
NUM_RUNS        = 5          # behavioral tasks
TOTAL_NEURONS   = 28 * 18944
RANDOM_SEED     = 42


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
    print("Loading Qwen2-VL-7B...")
    print("=" * 60)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = model.model.language_model.layers
    print(f"  Layers: {len(lm_layers)}")
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
    print(f"  Shape: {mean_acts.shape}")
    return mean_acts


# =============================================================================
# Build neuron sets
# =============================================================================
def build_neuron_sets(mean_acts):
    df_core = pd.read_csv(NEURONS_FILE)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Master core layer dict ---
    def to_layer_dict(df):
        d = {}
        for _, row in df.iterrows():
            l, n = int(row['layer']), int(row['neuron'])
            d.setdefault(l, []).append(n)
        return {l: torch.tensor(ns).long() for l, ns in d.items()}

    core_full   = to_layer_dict(df_core)

    # STEP 1 — patch-back
    # layers_18_27: all late neurons (1363)
    late_full = to_layer_dict(df_core[df_core['layer'].between(18, 27)])
    # layers_18_26: late WITHOUT layer_27 (1169) — i.e. layer_27 is restored
    late_no27 = to_layer_dict(df_core[df_core['layer'].between(18, 26)])

    # STEP 2 — control ablation
    # layer_27 master core (194 neurons)
    mc_27 = df_core[df_core['layer'] == 27]['neuron'].values.tolist()
    # 194 random layer-27 neurons NOT in master core
    all_l27      = set(range(18944))
    mc_27_set    = set(mc_27)
    available    = list(all_l27 - mc_27_set)
    random_27    = random.sample(available, len(mc_27))
    print(f"\nStep 2 — random layer-27 neurons sampled: {len(random_27)}")
    print(f"  Overlap with master core: {len(set(random_27) & mc_27_set)} (should be 0)")

    layer_27_mc  = {27: torch.tensor(mc_27).long()}
    layer_27_rnd = {27: torch.tensor(random_27).long()}

    # STEP 3 — knowledge dissociation uses same ablation tiers as step 1
    # (layers_18_27 and layer_27_mastercore)

    # Precompute mean replacement values for all neuron sets
    def make_means(layer_dict):
        means = {}
        for layer_idx, neuron_indices in layer_dict.items():
            vals = mean_acts[layer_idx, neuron_indices.numpy()]
            means[layer_idx] = torch.tensor(vals, dtype=torch.bfloat16)
        return means

    sets = {
        "baseline":            ({}, {}),
        # Step 1
        "layers_18_27":        (late_full,  make_means(late_full)),
        "layers_18_26":        (late_no27,  make_means(late_no27)),   # patched
        # Step 2
        "layer_27_mastercore": (layer_27_mc,  make_means(layer_27_mc)),
        "layer_27_random":     (layer_27_rnd, make_means(layer_27_rnd)),
    }

    print(f"\nNeuron set sizes:")
    for name, (ld, _) in sets.items():
        n = sum(len(v) for v in ld.values())
        print(f"  {name:22s}: {n:>5} neurons")

    return sets


# =============================================================================
# Hook management
# =============================================================================
def register_hooks(lm_layers, layer_dict, tier_means):
    handles = []
    for layer_idx, neuron_indices in layer_dict.items():
        idx  = neuron_indices.to("cuda")
        mean = tier_means[layer_idx].to("cuda")
        def make_hook(i, m):
            def hook(module, input, output):
                output[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return output
            return hook
        handles.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(
                make_hook(idx, mean)
            )
        )
    return handles


# =============================================================================
# Inference — behavioral task
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


def run_behavioral(model, processor, prompts_df, tier_name, run_idx):
    results, n_collapsed = [], 0
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df),
                       desc=f"  [{tier_name}] run {run_idx+1}"):
        prompt = row["Full_Prompt"]
        text   = processor.apply_chat_template(
            [{"role":"user","content":[{"type":"text","text":prompt}]}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 temperature=TEMPERATURE, do_sample=True, top_p=TOP_P)
        response = processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        collapsed    = is_collapsed(response)
        n_collapsed += int(collapsed)
        choice, points = parse_choice(response, prompt)
        results.append({"Tier":tier_name,"Run_ID":run_idx+1,"ID":row["ID"],
                         "Response":response,"Choice":choice,"Points":points,
                         "Collapsed":collapsed,"Task":"behavioral"})
    if n_collapsed:
        print(f"  ⚠  Collapse: {n_collapsed}/100 ({n_collapsed}%)")
    return results


# =============================================================================
# Inference — knowledge task
# =============================================================================
def score_knowledge(response, correct_keywords):
    """Return 1 if any correct keyword found in response (case-insensitive)."""
    resp_lower = response.lower()
    keywords   = [k.strip().lower() for k in correct_keywords.split(',')]
    return int(any(kw in resp_lower for kw in keywords))


def run_knowledge(model, processor, questions_df, tier_name):
    results = []
    print(f"\n  [{tier_name}] knowledge dissociation ({len(questions_df)} questions)...")
    for _, row in questions_df.iterrows():
        prompt = row["Question"]
        text   = processor.apply_chat_template(
            [{"role":"user","content":[{"type":"text","text":prompt}]}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=200,
                                 temperature=0.1, do_sample=True, top_p=0.95)
        response = processor.batch_decode(
            [out[len(inp):] for inp, out in zip(inputs.input_ids, gen)],
            skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        correct = score_knowledge(response, row["Correct_Keywords"])
        results.append({
            "Tier":      tier_name,
            "ID":        row["ID"],
            "Category":  row["Category"],
            "Question":  row["Question"],
            "Response":  response,
            "Correct":   correct,
            "Task":      "knowledge"
        })
        status = "✓" if correct else "✗"
        print(f"    {status} {row['ID']} [{row['Category']}]")
    return results


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    model, processor, lm_layers = load_model()
    mean_acts                   = compute_neutral_means()
    neuron_sets                 = build_neuron_sets(mean_acts)

    behavioral_df  = pd.read_csv(BEHAVIORAL_CSV)
    knowledge_df   = pd.read_csv(KNOWLEDGE_CSV)
    print(f"\nBehavioral prompts: {len(behavioral_df)}")
    print(f"Knowledge questions: {len(knowledge_df)}")

    all_behavioral = []
    all_knowledge  = []
    beh_path = os.path.join(OUTPUT_DIR, "behavioral_results.csv")
    kno_path = os.path.join(OUTPUT_DIR, "knowledge_results.csv")

    # ── STEP 1 & 2 — Behavioral tiers ─────────────────────────────────────
    BEHAVIORAL_TIERS = [
        # (tier_name,           step,    neuron_set_key)
        ("baseline",            "ctrl",  "baseline"),
        ("layers_18_27",        "1+3",   "layers_18_27"),    # replication + step3
        ("layers_18_26",        "1",     "layers_18_26"),    # patch-back
        ("layer_27_mastercore", "2",     "layer_27_mastercore"),
        ("layer_27_random",     "2",     "layer_27_random"),
    ]

    for tier_name, step, set_key in BEHAVIORAL_TIERS:
        layer_dict, tier_means = neuron_sets[set_key]
        n = sum(len(v) for v in layer_dict.values())
        print(f"\n{'='*60}")
        print(f"STEP {step} | TIER: {tier_name.upper()}  ({n} neurons)")
        print(f"{'='*60}")

        for run_idx in range(NUM_RUNS):
            handles = register_hooks(lm_layers, layer_dict, tier_means)
            try:
                results = run_behavioral(model, processor, behavioral_df,
                                         tier_name, run_idx)
                all_behavioral.extend(results)
            finally:
                for h in handles: h.remove()

            pd.DataFrame(all_behavioral).to_csv(beh_path, index=False)
            print(f"  Saved {len(all_behavioral)} rows → {beh_path}")

    # ── STEP 3 — Knowledge tiers ───────────────────────────────────────────
    KNOWLEDGE_TIERS = [
        ("baseline",     "baseline"),
        ("layers_18_27", "layers_18_27"),
        ("layer_27",     "layer_27_mastercore"),
    ]

    print(f"\n{'='*60}")
    print(f"STEP 3 — KNOWLEDGE DISSOCIATION")
    print(f"{'='*60}")

    for tier_name, set_key in KNOWLEDGE_TIERS:
        layer_dict, tier_means = neuron_sets[set_key]
        handles = register_hooks(lm_layers, layer_dict, tier_means)
        try:
            results = run_knowledge(model, processor, knowledge_df, tier_name)
            all_knowledge.extend(results)
        finally:
            for h in handles: h.remove()

        pd.DataFrame(all_knowledge).to_csv(kno_path, index=False)

    # ── Quick summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")

    df_b = pd.read_csv(beh_path)
    df_b_c = df_b[~df_b['Collapsed']].copy()
    df_b_c['Points'] = pd.to_numeric(df_b_c['Points'], errors='coerce')
    df_b_c = df_b_c.dropna(subset=['Points'])
    base_mean = df_b_c[df_b_c['Tier']=='baseline']['Points'].mean()

    print(f"\nBEHAVIORAL RESULTS (mean={base_mean:.2f} baseline):")
    print(f"  {'Tier':22s}  {'Mean':>7}  {'Δ':>7}  {'100%':>7}  Step")
    print(f"  {'-'*60}")
    steps = {"baseline":"ctrl","layers_18_27":"1+3","layers_18_26":"1 (patch)",
             "layer_27_mastercore":"2","layer_27_random":"2 (ctrl)"}
    for t in [r[0] for r in BEHAVIORAL_TIERS]:
        sub  = df_b_c[df_b_c['Tier']==t]
        if not len(sub): continue
        m    = sub['Points'].mean()
        r100 = (sub['Points']==100).mean()*100
        print(f"  {t:22s}  {m:>7.2f}  {m-base_mean:>+7.2f}  {r100:>6.1f}%  {steps[t]}")

    df_k = pd.read_csv(kno_path)
    print(f"\nKNOWLEDGE DISSOCIATION RESULTS:")
    print(f"  {'Tier':20s}  {'% Correct':>10}  {'by Category'}")
    print(f"  {'-'*60}")
    for t in [r[0] for r in KNOWLEDGE_TIERS]:
        sub = df_k[df_k['Tier']==t]
        overall = sub['Correct'].mean()*100
        by_cat  = sub.groupby('Category')['Correct'].mean().mul(100).round(0)
        cats    = "  ".join([f"{c}:{v:.0f}%" for c,v in by_cat.items()])
        print(f"  {t:20s}  {overall:>9.1f}%  {cats}")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION:")
    print(f"  Step 1 (patch-back):")
    print(f"    layers_18_26 > layers_18_27 → layer_27 is CAUSAL")
    print(f"    layers_18_26 ≈ layers_18_27 → layer_27 is a PASSENGER")
    print(f"  Step 2 (specificity):")
    print(f"    layer_27_random ≈ baseline   → effect is NEURON-SPECIFIC")
    print(f"    layer_27_random ≈ mastercore  → effect is just LAYER DAMAGE")
    print(f"  Step 3 (knowledge):")
    print(f"    ablated ~100% correct        → ANHEDONIA (motivation impaired)")
    print(f"    ablated <80% correct         → COGNITIVE DAMAGE (knowledge impaired)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
