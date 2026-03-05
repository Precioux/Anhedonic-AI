import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import pandas as pd

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
CSV_FILE = "data/geography_experiment_100-v2.csv"
OUTPUT_FILE = "neutral_activations_geo.pt"

# -------------------------------------------------------------------------
# NOTE: We hook model.model.layers[i].mlp.act_fn — the post-activation
# output INSIDE the FFN block. This gives us the actual intermediate MLP
# neuron activations (shape [batch, seq_len, intermediate_dim ~14336]),
# NOT the residual stream. This is the correct abstraction level for
# identifying and later ablating specific "reward-sensitive" neurons.
# We also use bfloat16 (no quantization) to match the ablation phase.
# -------------------------------------------------------------------------

print("Status: Loading model in bfloat16 (no quantization)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

num_layers = len(model.model.layers)
print(f"Detected {num_layers} transformer layers.")
print(f"MLP intermediate dim: {model.model.layers[0].mlp.act_fn.__class__.__name__}")

# -------------------------------------------------------------------------
# Verify the intermediate dimension by doing a dummy forward pass
# -------------------------------------------------------------------------
_dummy_cache = {}
def _dummy_hook(module, input, output):
    _dummy_cache['shape'] = output.shape

_h = model.model.layers[0].mlp.act_fn.register_forward_hook(_dummy_hook)
dummy_input = processor(
    text=["Hello"], return_tensors="pt"
).to("cuda")
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
    q_id = row['ID']
    prompt = row['Neutral_Prompt']

    # --- Register hooks for ALL layers ---
    mlp_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output shape: [batch=1, seq_len, intermediate_dim]
            # We take the LAST token position, same as before,
            # but now from the MLP intermediate (not residual stream).
            mlp_cache[layer_idx] = output[0, -1, :].detach().cpu().to(torch.float16)
        return hook

    hooks = []
    for i in range(num_layers):
        h = model.model.layers[i].mlp.act_fn.register_forward_hook(make_hook(i))
        hooks.append(h)

    # --- Forward pass ---
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        model(**inputs)

    # --- Remove hooks immediately after forward pass ---
    for h in hooks:
        h.remove()

    # --- Stack layers: shape [num_layers, intermediate_dim] ---
    layers_data = [mlp_cache[i] for i in range(num_layers)]
    results[f"q_{q_id}"] = torch.stack(layers_data)

    if q_id % 10 == 0:
        print(f"Progress: {q_id}/100 questions processed.")

torch.save(results, OUTPUT_FILE)
print(f"Done! MLP activations saved to {OUTPUT_FILE}")
print(f"Tensor shape per question: {results['q_1'].shape}  "
      f"(expected: [{num_layers}, {intermediate_dim}])")
