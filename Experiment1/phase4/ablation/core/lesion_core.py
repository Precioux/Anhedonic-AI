"""
lesion_core.py — Anhedonic AI Ablation Experiment
===================================================
Three-tier mean ablation on Qwen2-VL-7B MLP intermediate neurons.

WHAT CHANGED FROM THE OLD VERSION
----------------------------------
1. Zero ablation → Mean ablation
   Old: hidden_states[:, :, neurons] = 0.0
   New: hidden_states[:, :, neurons] = mean_activation[layer, neurons]
   Zero is an unnatural value (active suppression). Mean substitution asks
   "what if this neuron never saw an incentive signal?" — the correct causal
   intervention.

2. Hooks on layer OUTPUT → hooks on mlp.act_fn OUTPUT
   Old: model_layers[i].register_forward_hook(...)   # residual stream [4096]
   New: model_layers[i].mlp.act_fn.register_forward_hook(...)  # MLP intermediate [18944]
   Neurons were identified from mlp.act_fn activations — ablation must target
   the same module or the intervention is incoherent.

3. 4-bit quantization removed
   Old: BitsAndBytesConfig(load_in_4bit=True, ...)
   New: torch_dtype=torch.bfloat16
   Neurons were identified in bfloat16. Quantization changes the numerical
   landscape — identified neurons may not be the active neurons in a 4-bit model.

4. Three ablation tiers
   Tier 1 — Master core:        3,528 neurons  (0.665%)  reward ∩ money ∩ geo ∩ math
   Tier 2 — Reward universal:   5,558 neurons  (1.048%)  reward geo ∩ reward math
   Tier 3 — Top-1000 by delta:  1,000 neurons  (0.188%)  strongest master core neurons only

5. Baseline run (no ablation) included automatically
   Every prompt is also run through the unmodified model so results are
   directly comparable within the same script execution.

6. Input CSV includes three prompt conditions per question
   neutral_prompt, reward_prompt, money_prompt — all run for every tier.
   This lets you measure: does ablation reduce the *behavioural difference*
   between reward and neutral prompts?
"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =============================================================================
# Configuration — adjust paths to match your setup
# =============================================================================
MODEL_PATH        = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR   = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_FILE      = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"
REWARD_UNIV_FILE  = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/universal_reward_neurons.csv"
INPUT_CSV         = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/ablation/core/data/ablation_prompts.csv"
OUTPUT_DIR        = "results"

# Generation settings
MAX_NEW_TOKENS = 200
TEMPERATURE    = 0.7
TOP_P          = 0.95
NUM_RUNS       = 3    # runs per tier (for measuring output variance)

# Top-N tier: ablate only the N strongest neurons from the master core
TOP_N = 1000


# =============================================================================
# Step 1 — Load model (bfloat16, NO quantization — must match extraction phase)
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

    # Confirmed layer path from model inspection
    lm_layers = model.model.language_model.layers
    print(f"Language model layers: {len(lm_layers)}")
    return model, processor, lm_layers


# =============================================================================
# Step 2 — Compute mean activations from neutral baseline
#           Shape: {layer_idx: tensor [18944]} — the replacement value per neuron
# =============================================================================
def compute_neutral_means(lm_layers_count):
    """
    Load all neutral activation .pt files (geo + math) and compute
    the per-neuron mean across all questions and both domains.
    This is what we substitute in during mean ablation.
    Returns: dict {layer_idx: np.array [18944]}
    """
    print("\nComputing neutral baseline means for mean ablation...")
    all_neutral = []

    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        tensors = torch.stack(list(data.values())).float()  # [100, 28, 18944]
        all_neutral.append(tensors)

    # [200, 28, 18944] → mean over questions → [28, 18944]
    combined = torch.cat(all_neutral, dim=0)
    mean_acts = combined.mean(dim=0).numpy()  # [28, 18944]

    print(f"  Neutral mean shape: {mean_acts.shape}  "
          f"(mean={mean_acts.mean():.5f}, std={mean_acts.std():.5f})")
    return mean_acts  # index by layer: mean_acts[layer_idx, neuron_idx]


# =============================================================================
# Step 3 — Build three neuron sets
# =============================================================================
def build_neuron_tiers(mean_acts):
    """
    Returns a dict of tiers:
      'baseline'   : {} (no ablation — control run)
      'master_core': {layer: tensor of neuron indices}
      'reward_univ': {layer: tensor of neuron indices}
      'top_1000'   : {layer: tensor of neuron indices}  (strongest master core neurons)

    Also returns per-tier mean values:
      tier_means[tier_name] = {layer: tensor of mean values for those neurons}
    """

    # ── Load master core ──────────────────────────────────────────────────
    df_core = pd.read_csv(NEURONS_FILE)
    print(f"\nMaster core loaded: {len(df_core)} neurons")

    # ── Load reward universal ─────────────────────────────────────────────
    df_rew = pd.read_csv(REWARD_UNIV_FILE)
    print(f"Reward universal loaded: {len(df_rew)} neurons")

    # ── Build top-1000 by mean absolute delta (reward + money, geo + math) ─
    # Load all 4 delta arrays
    def load_mean(domain, cond):
        path = os.path.join(ACTIVATIONS_DIR, f"{cond}_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        return torch.stack(list(data.values())).float().mean(dim=0).numpy()  # [28, 18944]

    neu_geo  = load_mean("geo",  "neutral")
    neu_math = load_mean("math", "neutral")
    delta_rg = load_mean("geo",  "reward") - neu_geo
    delta_rm = load_mean("math", "reward") - neu_math
    delta_mg = load_mean("geo",  "money")  - neu_geo
    delta_mm = load_mean("math", "money")  - neu_math
    # Combined delta score per neuron: mean of all 4 absolute deltas
    combined_delta = (np.abs(delta_rg) + np.abs(delta_rm) +
                      np.abs(delta_mg) + np.abs(delta_mm)) / 4.0  # [28, 18944]

    # Score each master core neuron by its combined delta
    core_scores = []
    for _, row in df_core.iterrows():
        l, n = int(row['layer']), int(row['neuron'])
        core_scores.append((combined_delta[l, n], l, n))
    core_scores.sort(reverse=True)
    top1000 = [(l, n) for _, l, n in core_scores[:TOP_N]]
    df_top = pd.DataFrame(top1000, columns=['layer', 'neuron'])
    print(f"Top-{TOP_N} by delta magnitude: selected from master core")

    # ── Convert all three sets to {layer: tensor} dicts ───────────────────
    def df_to_layer_dict(df):
        layer_dict = {}
        for _, row in df.iterrows():
            l, n = int(row['layer']), int(row['neuron'])
            layer_dict.setdefault(l, []).append(n)
        return {l: torch.tensor(ns).long() for l, ns in layer_dict.items()}

    tiers = {
        "baseline":    {},                          # no ablation
        "master_core": df_to_layer_dict(df_core),
        "reward_univ": df_to_layer_dict(df_rew),
        "top_1000":    df_to_layer_dict(df_top),
    }

    # ── Precompute mean replacement tensors for each tier ─────────────────
    # tier_means[tier][layer] = float32 tensor of shape [len(neurons)]
    tier_means = {"baseline": {}}
    for tier_name, layer_dict in tiers.items():
        if tier_name == "baseline":
            continue
        tier_means[tier_name] = {}
        for layer_idx, neuron_indices in layer_dict.items():
            vals = mean_acts[layer_idx, neuron_indices.numpy()]   # [N]
            tier_means[tier_name][layer_idx] = torch.tensor(vals, dtype=torch.bfloat16)

    # ── Print summary ──────────────────────────────────────────────────────
    total_neurons = 28 * 18944
    print(f"\nTier summary:")
    for tier_name, layer_dict in tiers.items():
        n = sum(len(v) for v in layer_dict.values())
        pct = n / total_neurons * 100
        print(f"  {tier_name:12s}: {n:>5,} neurons  ({pct:.3f}%)")

    return tiers, tier_means


# =============================================================================
# Step 4 — Register mean-ablation hooks for one tier
# =============================================================================
def register_hooks(lm_layers, layer_dict, tier_means_for_tier):
    """
    Hook model.model.language_model.layers[i].mlp.act_fn
    Replace target neuron activations with their neutral mean value.
    Returns list of hook handles (call handle.remove() to clean up).
    """
    handles = []

    for layer_idx, neuron_indices in layer_dict.items():
        # Pre-move to GPU
        indices      = neuron_indices.to("cuda")
        mean_values  = tier_means_for_tier[layer_idx].to("cuda")  # [N] bfloat16

        def make_hook(idx, means):
            def hook(module, input, output):
                # output shape: [batch, seq_len, 18944]
                output[:, :, idx] = means.unsqueeze(0).unsqueeze(0)
                return output
            return hook

        handle = lm_layers[layer_idx].mlp.act_fn.register_forward_hook(
            make_hook(indices, mean_values)
        )
        handles.append(handle)

    return handles


# =============================================================================
# Step 5 — Run inference
# =============================================================================
def run_inference(model, processor, prompts_df, tier_name, run_idx):
    """
    Run all prompts through the model (hooks already applied externally).
    Returns list of result dicts.
    """
    results = []
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df),
                       desc=f"  [{tier_name}] run {run_idx+1}"):
        for prompt_col in ["Neutral_Prompt", "Reward_Prompt", "Money_Prompt"]:
            if prompt_col not in row:
                continue
            prompt_text = row[prompt_col]

            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            text     = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
            inputs   = processor(text=[text], return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    top_p=TOP_P,
                )

            trimmed = [out[len(inp):] for inp, out in
                       zip(inputs.input_ids, generated_ids)]
            response = processor.batch_decode(
                trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            results.append({
                "Tier":          tier_name,
                "Run_ID":        run_idx + 1,
                "ID":            row["ID"],
                "Prompt_Type":   prompt_col.replace("_Prompt", "").lower(),
                "Prompt":        prompt_text,
                "Response":      response,
            })

    return results


# =============================================================================
# Main
# =============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load model
    model, processor, lm_layers = load_model()

    # 2. Compute neutral means (for mean ablation)
    mean_acts = compute_neutral_means(len(lm_layers))

    # 3. Build neuron tiers
    tiers, tier_means = build_neuron_tiers(mean_acts)

    # 4. Load prompts
    print(f"\nLoading prompts from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    prompts_df = pd.read_csv(INPUT_CSV)
    print(f"  {len(prompts_df)} questions × 3 prompt types = "
          f"{len(prompts_df)*3} prompts per run")

    # 5. Run all tiers
    all_results = []

    for tier_name, layer_dict in tiers.items():
        print(f"\n{'='*60}")
        n_neurons = sum(len(v) for v in layer_dict.values())
        print(f"TIER: {tier_name.upper()}  ({n_neurons:,} neurons ablated)")
        print(f"{'='*60}")

        for run_idx in range(NUM_RUNS):
            # Register hooks (empty dict = no hooks = baseline)
            handles = register_hooks(lm_layers, layer_dict,
                                     tier_means.get(tier_name, {}))
            try:
                results = run_inference(model, processor, prompts_df,
                                        tier_name, run_idx)
                all_results.extend(results)
            finally:
                # Always remove hooks — even if inference crashes
                for h in handles:
                    h.remove()

            # Incremental save after each run
            out_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
            pd.DataFrame(all_results).to_csv(out_path, index=False)
            print(f"  Saved {len(all_results)} rows → {out_path}")

    # 6. Final save
    df_out = pd.DataFrame(all_results)
    out_path = os.path.join(OUTPUT_DIR, "ablation_results.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE — {len(df_out):,} total responses saved to {out_path}")
    print(f"{'='*60}")
    print(f"\nResult breakdown:")
    print(df_out.groupby(["Tier", "Prompt_Type"]).size().to_string())


if __name__ == "__main__":
    main()
