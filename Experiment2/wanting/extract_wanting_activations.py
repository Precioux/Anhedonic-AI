import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
model_name = "/mnt/mahdipou/models/qwen2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

num_layers = len(model.model.language_model.layers)
print(f"Model loaded. Layers: {num_layers}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("wanting_contrastive_dataset.csv")
print(f"Loaded {len(df)} contrastive pairs")

# ══════════════════════════════════════════════════════════════════════════════
# EXTRACT ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_activations(text):
    """Extract last-token residual stream at all layers."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    activations = {}
    def make_hook(layer_idx):
        def hook(module, input, output):
            # last token
            activations[layer_idx] = output[0][0, -1, :].detach().cpu()
        return hook
    
    handles = []
    for i in range(num_layers):
        layer = model.model.language_model.layers[i]
        handles.append(layer.register_forward_hook(make_hook(i)))
    
    with torch.no_grad():
        model(**inputs)
    
    for h in handles:
        h.remove()
    
    return [activations[i] for i in range(num_layers)]

high_wanting_activations = []
low_wanting_activations = []

for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
    high_act = get_activations(row["high_wanting"])
    low_act = get_activations(row["low_wanting"])
    high_wanting_activations.append(high_act)
    low_wanting_activations.append(low_act)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
torch.save(high_wanting_activations, "wanting_high_activations.pt")
torch.save(low_wanting_activations, "wanting_low_activations.pt")
print(f"Saved: wanting_high_activations.pt, wanting_low_activations.pt")
print(f"Shape per sample: {num_layers} layers x {high_wanting_activations[0][0].shape}")