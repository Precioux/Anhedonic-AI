import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import pandas as pd
import os

# 1. Configuration (No Quantization - Full Precision)
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
CSV_FILE = "math_experiment_100.csv"
OUTPUT_FILE = "neutral_activations_v2.pt"

print("Status: Loading model in Full Bfloat16 Precision...")
# Load in Bfloat16 (Original Weight Precision)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# 2. Run Experiment
df = pd.read_csv(CSV_FILE)
results = {}

print("Status: Starting extraction from ALL layers (Last Token focus)...")

for index, row in df.iterrows():
    q_id = row['ID']
    prompt = row['Neutral_Prompt']
    
    # Standard Qwen2-VL text template
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        # Capturing internal brain states
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extracting ALL 29 layers but only the LAST token position
    # Each layer is moved to CPU and stored in float16 to optimize final file size
    layers_data = []
    for layer_state in outputs.hidden_states:
        # last_token position: [batch=0, sequence=-1, hidden_dim=:]
        vector = layer_state[0, -1, :].detach().cpu().to(torch.float16)
        layers_data.append(vector)
    
    # Shape: [num_layers, 4096]
    results[f"q_{q_id}"] = torch.stack(layers_data)
    
    if q_id % 10 == 0:
        print(f"Progress: {q_id}/100 questions processed.")

# 3. Final Save
torch.save(results, OUTPUT_FILE)
print(f"Done! Full-precision activations saved to {OUTPUT_FILE}")
