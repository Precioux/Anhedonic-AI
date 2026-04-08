import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd
import os

# =============================================================================
# Configuration — Batch 2 of 3 (subjects 19-36)
# =============================================================================
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-72b"
DATA_DIR   = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/MMLU/data"
OUT_BASE   = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/MMLU/activations"

SUBJECTS = [
    "high_school_computer_science", "high_school_european_history",
    "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics",
    "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning",
]

CONDITIONS = {
    "neutral": "Neutral_Prompt",
    "reward":  "Reward_Prompt",
    "money":   "Money_Prompt",
}

# =============================================================================
# Load model ONCE — 4-bit NF4, forced to GPU 0
# =============================================================================
print("=" * 60)
print("Loading Qwen2-VL-72B (4-bit NF4, device_map GPU 0) — Batch 2/3")
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

lm_layers        = model.model.language_model.layers
num_layers       = len(lm_layers)
intermediate_dim = model.model.language_model.config.intermediate_size
print(f"Layers: {num_layers}, MLP dim: {intermediate_dim}")
print(f"Subjects in this batch: {len(SUBJECTS)}\n")

# =============================================================================
# Helper
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
# Main loop
# =============================================================================
total = len(SUBJECTS)
for s_idx, subject in enumerate(SUBJECTS, 1):
    csv_path = os.path.join(DATA_DIR, f"{subject}.csv")
    out_dir  = os.path.join(OUT_BASE, subject)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{s_idx}/{total}] {subject}")

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
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"  [{condition}] saved → {out_path}  [{size_mb:.1f} MB]")

    print()

# Summary
print("=" * 60)
print("BATCH 2 DONE — output files:")
print("=" * 60)
missing = []
for subject in SUBJECTS:
    for condition in CONDITIONS:
        path = os.path.join(OUT_BASE, subject, f"{condition}.pt")
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
    print(f"\nAll {len(SUBJECTS) * len(CONDITIONS)} files present ✓")