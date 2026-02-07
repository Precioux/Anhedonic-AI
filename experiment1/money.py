import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import pandas as pd

MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
CSV_FILE = "geography_experiment_100.csv"
OUTPUT_FILE = "money_activations_geo.pt"

print("Status: Loading model in Full Bfloat16 Precision...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

df = pd.read_csv(CSV_FILE)
results = {}

print("Status: Starting extraction for MONEY mode...")

for index, row in df.iterrows():
    q_id = row['ID']
    prompt = row['Money_Prompt'] # استفاده از پرامپت پول
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    layers_data = []
    for layer_state in outputs.hidden_states:
        vector = layer_state[0, -1, :].detach().cpu().to(torch.float16)
        layers_data.append(vector)
    
    results[f"q_{q_id}"] = torch.stack(layers_data)
    
    if q_id % 10 == 0:
        print(f"Progress: {q_id}/100 processed.")

torch.save(results, OUTPUT_FILE)
print(f"Done! Saved to {OUTPUT_FILE}")
