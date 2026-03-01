import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

model_name = "/mnt/mahdipou/models/qwen2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

num_layers = len(model.model.language_model.layers)
print(f"Model loaded. Layers: {num_layers}")

df = pd.read_csv("wanting_contrastive_dataset_v3.csv")
print(f"Loaded {len(df)} contrastive pairs")

def get_activations(text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    activations = {}
    def make_hook(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output[0][0, -1, :].detach().cpu()
        return hook
    handles = []
    for i in range(num_layers):
        handles.append(model.model.language_model.layers[i].register_forward_hook(make_hook(i)))
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return [activations[i] for i in range(num_layers)]

high_acts, low_acts = [], []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
    high_acts.append(get_activations(row["high_wanting"]))
    low_acts.append(get_activations(row["low_wanting"]))

torch.save(high_acts, "wanting_high_activations_v3.pt")
torch.save(low_acts, "wanting_low_activations_v3.pt")
print(f"Saved: wanting_high_activations_v3.pt, wanting_low_activations_v3.pt")