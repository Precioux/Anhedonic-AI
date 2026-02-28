import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

# ── 1. Load model and tokenizer ──────────────────────────────────────────────
model_name = "/mnt/mahdipou/models/qwen2-vl-7b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ── 2. Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("contrastive_dataset.csv")

# ── 3. Function to get activations ───────────────────────────────────────────
def get_activations(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=False,
        truncation=True
    ).to(model.device)

    layer_activations = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            layer_activations[layer_idx] = output[0][:, -1, :].detach().cpu()
        return hook

    # ── FIXED: correct path to layers ────────────────────────────────────────
    for i, layer in enumerate(model.model.language_model.layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return layer_activations

# ── 4. Extract activations for all pairs ─────────────────────────────────────
print("Extracting activations...")

high_reward_activations = []
neutral_activations = []

for i, row in df.iterrows():
    print(f"Processing pair {i+1}/{len(df)}")

    high_acts = get_activations(row["high_reward"])
    neutral_acts = get_activations(row["neutral"])

    high_reward_activations.append(high_acts)
    neutral_activations.append(neutral_acts)

# ── 5. Save ───────────────────────────────────────────────────────────────────
torch.save(high_reward_activations, "high_reward_activations.pt")
torch.save(neutral_activations, "neutral_activations.pt")

print("Done! Files saved.")