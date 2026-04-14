"""
extract_activations_modelB.py  —  Model B Activation Extraction
================================================================
Identical to extract_activations.py (Model A) but adds
business_ethics and philosophy to the domain set.

Model A used: geo + math          (2 domains)
Model B uses: geo + math + business_ethics + philosophy  (4 domains)

Outputs (12 files):
    activations_modelB/{condition}_activations_{domain}.pt
    condition ∈ {neutral, reward, money}
    domain    ∈ {geo, math, business_ethics, philosophy}
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import pandas as pd
import os

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
OUTPUT_DIR = "activations_modelB"

DATASETS = {
    "geo":             "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/data/geography_experiment_100-v2.csv",
    "math":            "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/data/math_experiment_100-v2.csv",
    "business_ethics": "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase6/data/business_ethics_v2.csv",
    "philosophy":      "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase6/data/philosophy_v2.csv",
}

CONDITIONS = {
    "neutral": "Neutral_Prompt",
    "reward":  "Reward_Prompt",
    "money":   "Money_Prompt",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load model ONCE
# =============================================================================
print("=" * 60)
print("Loading model in bfloat16 (no quantization)...")
print("=" * 60)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
processor  = AutoProcessor.from_pretrained(MODEL_PATH)
lm_layers  = model.model.language_model.layers
num_layers = len(lm_layers)
print(f"Language model layers: {num_layers}")

# Confirm intermediate dim via dummy pass
_dim_cache = {}
def _dim_hook(module, input, output):
    _dim_cache['dim'] = output.shape[-1]
_h = lm_layers[0].mlp.act_fn.register_forward_hook(_dim_hook)
with torch.no_grad():
    model(**processor(text=["Hello"], return_tensors="pt").to("cuda"))
_h.remove()
intermediate_dim = _dim_cache['dim']
print(f"MLP intermediate dim:  {intermediate_dim}")
print(f"Expected output shape per question: [{num_layers}, {intermediate_dim}]")
print()

# =============================================================================
# Helper: extract MLP activations for one prompt
# =============================================================================
def extract_mlp_activations(prompt: str) -> torch.Tensor:
    mlp_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            mlp_cache[layer_idx] = output[0, -1, :].detach().cpu().to(torch.float16)
        return hook

    hooks = [
        lm_layers[i].mlp.act_fn.register_forward_hook(make_hook(i))
        for i in range(num_layers)
    ]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        model(**inputs)
    for h in hooks:
        h.remove()
    return torch.stack([mlp_cache[i] for i in range(num_layers)])


# =============================================================================
# Main loop: domain × condition  (12 runs total)
# =============================================================================
for domain, csv_file in DATASETS.items():
    print("=" * 60)
    print(f"Domain: {domain.upper()}  |  file: {csv_file}")
    print("=" * 60)

    if not os.path.exists(csv_file):
        print(f"  ERROR: {csv_file} not found — skipping.\n")
        continue

    df          = pd.read_csv(csv_file)
    n_questions = len(df)

    for condition, col in CONDITIONS.items():
        out_path = os.path.join(OUTPUT_DIR, f"{condition}_activations_{domain}.pt")

        if os.path.exists(out_path):
            print(f"  [{condition}] Already exists — skipping: {out_path}")
            continue

        print(f"\n  Condition: {condition.upper()}  (column: '{col}')")
        results = {}

        for _, row in df.iterrows():
            q_id   = int(row['ID'])
            prompt = row[col]
            results[f"q_{q_id}"] = extract_mlp_activations(prompt)
            if q_id % 10 == 0:
                print(f"    Progress: {q_id}/{n_questions}")

        torch.save(results, out_path)
        print(f"  Saved {out_path}  |  shape per question: {results['q_1'].shape}")

    print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 60)
print("ALL DONE — output files:")
print("=" * 60)
for domain in DATASETS:
    for condition in CONDITIONS:
        path = os.path.join(OUTPUT_DIR, f"{condition}_activations_{domain}.pt")
        size = f"{os.path.getsize(path)/1e6:.1f} MB" if os.path.exists(path) else "MISSING"
        print(f"  {path}  [{size}]")