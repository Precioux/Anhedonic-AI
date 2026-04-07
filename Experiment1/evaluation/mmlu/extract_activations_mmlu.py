"""
extract_activations_mmlu.py
================================================================================
Extracts MLP act_fn activations for all 57 MMLU subject localizer CSVs.

Mirrors extract_activations_asdiv.py exactly — same model loading, same hook
target, same output format — only the input files and output paths change.

Output structure:
    activations/mmlu/{subject}/neutral.pt
    activations/mmlu/{subject}/reward.pt
    activations/mmlu/{subject}/money.pt

Each .pt file is a dict: {"q_1": tensor, "q_2": tensor, ...}
Each tensor shape: [num_layers, intermediate_dim]  (float16, CPU)
"""

import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH   = "/mnt/mahdipou/models/qwen2-vl-7b"
DATA_DIR     = "data/mmlu"
OUT_BASE_DIR = "activations/mmlu"

CONDITIONS = {
    "neutral": "Neutral_Prompt",
    "reward":  "Reward_Prompt",
    "money":   "Money_Prompt",
}

SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# =============================================================================
# Load model ONCE — bfloat16, no quantization (must match ablation phase)
# =============================================================================
print("=" * 60)
print("Loading model in bfloat16 (no quantization)...")
print("=" * 60)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

lm_layers  = model.model.language_model.layers
num_layers = len(lm_layers)
print(f"Language model layers: {num_layers}")

# Confirm MLP intermediate dim via dummy pass
_dim_cache = {}
def _dim_hook(module, input, output):
    _dim_cache['dim'] = output.shape[-1]

_h = lm_layers[0].mlp.act_fn.register_forward_hook(_dim_hook)
with torch.no_grad():
    model(**processor(text=["Hello"], return_tensors="pt").to("cuda"))
_h.remove()
intermediate_dim = _dim_cache['dim']
print(f"MLP intermediate dim:  {intermediate_dim}")
print(f"Expected tensor shape per question: [{num_layers}, {intermediate_dim}]")
print()

# =============================================================================
# Helper: extract MLP activations for one prompt
# =============================================================================
def extract_mlp_activations(prompt: str) -> torch.Tensor:
    """
    Returns a tensor of shape [num_layers, intermediate_dim] (float16, CPU).
    Captures the LAST token position of the MLP act_fn output for each layer.
    """
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
# Main loop — iterate subjects then conditions
# =============================================================================
total_subjects = len(SUBJECTS)

for s_idx, subject in enumerate(SUBJECTS, 1):
    csv_path = os.path.join(DATA_DIR, f"{subject}.csv")
    out_dir  = os.path.join(OUT_BASE_DIR, subject)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{s_idx}/{total_subjects}] {subject}")

    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found — skipping.\n")
        continue

    df = pd.read_csv(csv_path)

    for condition, col in CONDITIONS.items():
        out_path = os.path.join(out_dir, f"{condition}.pt")

        if os.path.exists(out_path):
            print(f"  [{condition}] already exists — skipping.")
            continue

        print(f"  [{condition}] extracting {len(df)} questions ...")
        results = {}

        for _, row in df.iterrows():
            q_id   = int(row['ID'])
            prompt = row[col]

            results[f"q_{q_id}"] = extract_mlp_activations(prompt)

            if q_id % 10 == 0:
                print(f"    {q_id}/{len(df)}")

        torch.save(results, out_path)
        sample_shape = results['q_1'].shape
        print(f"  [{condition}] saved → {out_path}  shape: {sample_shape}")

    print()

# =============================================================================
# Final summary
# =============================================================================
print("=" * 60)
print("ALL DONE — output files:")
print("=" * 60)
missing = []
for subject in SUBJECTS:
    for condition in CONDITIONS:
        path = os.path.join(OUT_BASE_DIR, subject, f"{condition}.pt")
        if os.path.exists(path):
            size = f"{os.path.getsize(path)/1e6:.1f} MB"
            print(f"  {path}  [{size}]")
        else:
            missing.append(path)

if missing:
    print(f"\nMISSING ({len(missing)}):")
    for p in missing:
        print(f"  {p}")
else:
    print(f"\nAll {total_subjects * len(CONDITIONS)} files present ✓")