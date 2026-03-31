import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH  = "/mnt/mahdipou/models/qwen2-vl-72b"
CSV_FILE    = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/data/geography_experiment_100-v2.csv"
OUTPUT_FILE = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/activations_72b/neutral_activations_geo.pt"

import os
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# -------------------------------------------------------------------------
# NOTE: We hook language_model.layers[i].mlp.act_fn — the post-activation
# output inside the FFN block. This gives MLP intermediate neuron activations
# (shape [batch, seq_len, intermediate_dim ~28672 for 72B]),
# NOT the residual stream. We use 4-bit quantization to fit on A100 80GB.
# Use the same quant config in the ablation phase for consistency.
# -------------------------------------------------------------------------

print("Status: Loading Qwen2-VL-72B in 4-bit quantization...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

lm_layers  = model.model.language_model.layers
num_layers = len(lm_layers)
print(f"Detected {num_layers} transformer layers.")

# -------------------------------------------------------------------------
# Verify the intermediate dimension via dummy forward pass
# -------------------------------------------------------------------------
_dummy_cache = {}
def _dummy_hook(module, input, output):
    _dummy_cache['shape'] = output.shape

_h = lm_layers[0].mlp.act_fn.register_forward_hook(_dummy_hook)
dummy_input = processor(text=["Hello"], return_tensors="pt").to("cuda")
with torch.no_grad():
    model(**dummy_input)
_h.remove()
intermediate_dim = _dummy_cache['shape'][-1]
print(f"Confirmed MLP intermediate dim: {intermediate_dim}")

# -------------------------------------------------------------------------
# Main extraction loop
# -------------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
results = {}

print("Status: Starting MLP activation extraction for NEUTRAL prompts...")

for index, row in df.iterrows():
    q_id   = row['ID']
    prompt = row['Neutral_Prompt']

    mlp_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            mlp_cache[layer_idx] = output[0, -1, :].detach().cpu().to(torch.float16)
        return hook

    hooks = []
    for i in range(num_layers):
        h = lm_layers[i].mlp.act_fn.register_forward_hook(make_hook(i))
        hooks.append(h)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs   = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    layers_data = [mlp_cache[i] for i in range(num_layers)]
    results[f"q_{q_id}"] = torch.stack(layers_data)

    if q_id % 10 == 0:
        print(f"Progress: {q_id}/100 questions processed.")

torch.save(results, OUTPUT_FILE)
print(f"Done! MLP activations saved to {OUTPUT_FILE}")
print(f"Tensor shape per question: {results['q_1'].shape}  "
      f"(expected: [{num_layers}, {intermediate_dim}])")
