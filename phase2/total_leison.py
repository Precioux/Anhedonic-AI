import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
INPUT_FILE = "data/full_experiment_100_rows.csv"
OUTPUT_FILE = "total_lesion_results.csv"  # Final output

# Neuron Files
MONEY_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_money_neurons.csv"
REWARD_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_reward_neurons.csv"

# *** NUCLEAR OPTION: ALL LAYERS ***
TARGET_LAYERS = list(range(0, 29)) # Layers 0 to 28

def load_combined_neurons():
    indices_list = []
    if os.path.exists(MONEY_NEURONS_FILE):
        print(f"Loading {MONEY_NEURONS_FILE}...")
        df_m = pd.read_csv(MONEY_NEURONS_FILE)
        col = 'neuron_index' if 'neuron_index' in df_m.columns else df_m.columns[0]
        indices_list.append(df_m[col].values)
    
    if os.path.exists(REWARD_NEURONS_FILE):
        print(f"Loading {REWARD_NEURONS_FILE}...")
        df_r = pd.read_csv(REWARD_NEURONS_FILE)
        col = 'neuron_index' if 'neuron_index' in df_r.columns else df_r.columns[0]
        indices_list.append(df_r[col].values)

    if not indices_list:
        raise FileNotFoundError("No neuron files found.")

    combined = np.concatenate(indices_list)
    unique_indices = np.unique(combined)
    print(f"-> TARGETING {len(unique_indices)} NEURONS ACROSS ALL {len(TARGET_LAYERS)} LAYERS.")
    return torch.tensor(unique_indices).long()

def main():
    # 1. Neurons
    try:
        lesion_indices = load_combined_neurons()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Model
    print(f"Loading Qwen2-VL from {MODEL_PATH}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )

    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, quantization_config=quant_config, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, min_pixels=256*28*28, max_pixels=512*28*28
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Apply Total Lesion
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
    
    if model_layers is None:
        print("Error: Could not find model layers.")
        return

    lesion_indices = lesion_indices.to(model.device)
    
    def lesion_hook(module, input, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output
        # Zero out everywhere
        h[:, :, lesion_indices] = 0
        if isinstance(output, tuple): return (h,) + output[1:]
        return h

    print(f"WARNING: Applying TOTAL LESION to all layers...")
    handles = []
    for layer_idx in TARGET_LAYERS:
        if layer_idx < len(model_layers):
            handle = model_layers[layer_idx].register_forward_hook(lesion_hook)
            handles.append(handle)

    # 4. Run
    print(f"Generating responses...")
    if not os.path.exists(INPUT_FILE):
        print("Input file missing.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    results = []

    for index, row in tqdm(df_input.iterrows(), total=len(df_input)):
        prompt_text = row['Full_Prompt']
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.95
            )

        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append({
            "ID": row['ID'], 
            "Full_Prompt": prompt_text, 
            "Model_Response": response
        })

    # 5. Save
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")
    
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()