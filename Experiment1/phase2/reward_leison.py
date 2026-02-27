import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
INPUT_FILE = "data/full_experiment_100_rows.csv"
NEURON_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_reward_neurons.csv"      # <--- ONLY REWARD NEURONS
OUTPUT_FILE = "reward_only_results.csv"
TARGET_LAYERS = [24, 25, 26, 27, 28]

def main():
    # 1. Load Neurons
    if not os.path.exists(NEURON_FILE):
        print(f"Error: {NEURON_FILE} not found.")
        return
    df = pd.read_csv(NEURON_FILE)
    col = 'neuron_index' if 'neuron_index' in df.columns else df.columns[0]
    lesion_indices = torch.tensor(df[col].values).long()
    print(f"Loaded {len(lesion_indices)} REWARD neurons.")

    # 2. Load Model
    print(f"Loading Model from {MODEL_PATH}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=quant_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=512*28*28)

    # 3. Apply Lesion
    model_layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        model_layers = model.model.layers
    elif hasattr(model, "layers"):
        model_layers = model.layers
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
                model_layers = module
                break

    lesion_indices = lesion_indices.to(model.device)
    
    def lesion_hook(module, input, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output
        h[:, :, lesion_indices] = 0
        if isinstance(output, tuple): return (h,) + output[1:]
        return h

    handles = []
    for i in TARGET_LAYERS:
        if i < len(model_layers):
            handles.append(model_layers[i].register_forward_hook(lesion_hook))

    # 4. Run Experiment
    print("Generating responses (Reward Lesion)...")
    df_in = pd.read_csv(INPUT_FILE)
    results = []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in)):
        prompt = row['Full_Prompt']
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.95)
        
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append({"ID": row['ID'], "Full_Prompt": prompt, "Model_Response": response})

    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")
    for h in handles: h.remove()

if __name__ == "__main__":
    main()