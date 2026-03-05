import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import pandas as pd

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
CSV_FILE = "geography_experiment_100.csv"
OUTPUT_FILE = "reward_activations_geo.pt"

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

df = pd.read_csv(CSV_FILE)
results = {}

print("Status: Starting MLP activation extraction for REWARD prompts...")

for index, row in df.iterrows():
    q_id = row['ID']
    prompt = row['Reward_Prompt']

    mlp_cache = {}

    def make_hook(layer_idx):
        def hook(module, input, output):
            mlp_cache[layer_idx] = output[0, -1, :].detach().cpu().to(torch.float16)
        return hook

    hooks = []
    for i in range(num_layers):
        h = model.model.layers[i].mlp.act_fn.register_forward_hook(make_hook(i))
        hooks.append(h)

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    layers_data = [mlp_cache[i] for i in range(num_layers)]
    results[f"q_{q_id}"] = torch.stack(layers_data)

    if q_id % 10 == 0:
        print(f"Progress: {q_id}/100 processed.")

torch.save(results, OUTPUT_FILE)
print(f"Done! Saved to {OUTPUT_FILE}")
print(f"Tensor shape per question: {results['q_1'].shape}")
