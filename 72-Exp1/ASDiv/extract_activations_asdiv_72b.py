import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd
import os

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-72b"
OUTPUT_DIR = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/ASDiv/activations"

DATASETS = {
    "asdiv": "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/ASDiv/data/math_asdiv_localizer.csv",
}

CONDITIONS = {
    "neutral": "Neutral_Prompt",
    "reward":  "Reward_Prompt",
    "money":   "Money_Prompt",
}

# Output naming: activations/{condition}_activations_asdiv.pt
# e.g. .../neutral_activations_asdiv.pt

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Load model ONCE — 4-bit NF4 quantization, forced to GPU 0
# =============================================================================
print("=" * 60)
print("Loading Qwen2-VL-72B (4-bit NF4, device_map GPU 0)...")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

lm_layers  = model.model.language_model.layers
num_layers = len(lm_layers)
print(f"Language model layers: {num_layers}")

intermediate_dim = model.model.language_model.config.intermediate_size
print(f"MLP intermediate dim:  {intermediate_dim}")
print(f"Expected output shape per question: [{num_layers}, {intermediate_dim}]")
print()

# =============================================================================
# Helper: extract MLP activations for one prompt
# =============================================================================
def extract_mlp_activations(prompt: str) -> torch.Tensor:
    """
    Returns a tensor of shape [num_layers, intermediate_dim] (float16, on CPU).
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
# Main loop
# =============================================================================
for domain, csv_file in DATASETS.items():
    print("=" * 60)
    print(f"Domain: {domain.upper()}  |  file: {csv_file}")
    print("=" * 60)

    if not os.path.exists(csv_file):
        print(f"  ERROR: {csv_file} not found — skipping.\n")
        continue

    df = pd.read_csv(csv_file)

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
                print(f"    Progress: {q_id}/{len(df)}")

        torch.save(results, out_path)
        shape   = results['q_1'].shape
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  Saved {out_path}  |  shape: {shape}  [{size_mb:.1f} MB]")

    print()

# =============================================================================
# Final summary
# =============================================================================
print("=" * 60)
print("ALL DONE — output files:")
print("=" * 60)
for domain in DATASETS:
    for condition in CONDITIONS:
        path = os.path.join(OUTPUT_DIR, f"{condition}_activations_{domain}.pt")
        size = f"{os.path.getsize(path)/1e6:.1f} MB" if os.path.exists(path) else "MISSING"
        print(f"  {path}  [{size}]")