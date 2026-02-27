import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

# --- IMPORTS ---
try:
    from transformers import Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration
except ImportError:
    from transformers import AutoModel
    ModelClass = AutoModel

# --- Configuration ---
MODEL_PATH = os.path.abspath("/mnt/mahdipou/models/qwen2-vl-7b")
INPUT_FILE = "data/full_experiment_100_rows.csv"
OUTPUT_FILE = "steering_results.csv"  # Steering Output

# The "Brake" Neurons
MONEY_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_money_neurons.csv"
REWARD_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_reward_neurons.csv"

# *** STEERING SETTINGS ***
# We add this value to the neuron's output.
# Not too big to break the model, but big enough to change behavior.
STEERING_COEFFICIENT = 5.0 
TARGET_LAYERS = list(range(0, 29))

def load_combined_neurons():
    indices_list = []
    if os.path.exists(MONEY_NEURONS_FILE):
        df_m = pd.read_csv(MONEY_NEURONS_FILE)
        col = 'neuron_index' if 'neuron_index' in df_m.columns else df_m.columns[0]
        indices_list.append(df_m[col].values)
    
    if os.path.exists(REWARD_NEURONS_FILE):
        df_r = pd.read_csv(REWARD_NEURONS_FILE)
        col = 'neuron_index' if 'neuron_index' in df_r.columns else df_r.columns[0]
        indices_list.append(df_r[col].values)

    if not indices_list:
        if os.path.exists("../" + MONEY_NEURONS_FILE): # Fallback
             df_m = pd.read_csv("../" + MONEY_NEURONS_FILE)
             return torch.tensor(df_m.iloc[:,0].values).long()
        raise FileNotFoundError("No neuron files found.")

    combined = np.concatenate(indices_list)
    unique_indices = np.unique(combined)
    print(f"-> STEERING {len(unique_indices)} NEURONS (Bias = +{STEERING_COEFFICIENT}).")
    return torch.tensor(unique_indices).long()

def main():
    # 1. Load Neurons
    try:
        target_indices = load_combined_neurons()
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Load Model
    print(f"Loading Qwen2-VL from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Path {MODEL_PATH} not found.")
        return

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )

    try:
        model = ModelClass.from_pretrained(
            MODEL_PATH, 
            quantization_config=quant_config, 
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, 
            min_pixels=256*28*28, 
            max_pixels=512*28*28,
            trust_remote_code=True,
            local_files_only=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Apply Steering Hook
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
    
    target_indices = target_indices.to(model.device)
    
    def steering_hook(module, input, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output
        
        # *** THE MAGIC SAUCE: STEERING ***
        # Instead of replacing with 0 or 20, we ADD bias.
        # This amplifies the "Caution" signal without breaking the flow.
        h[:, :, target_indices] = h[:, :, target_indices] + STEERING_COEFFICIENT
        
        if isinstance(output, tuple): return (h,) + output[1:]
        return h

    print(f"WARNING: Applying STEERING VECTOR (+{STEERING_COEFFICIENT}) to all layers...")
    handles = []
    for layer_idx in TARGET_LAYERS:
        if layer_idx < len(model_layers):
            handles.append(model_layers[layer_idx].register_forward_hook(steering_hook))

    # 4. Run Inference
    print("Generating responses...")
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    results = []

    for _, row in tqdm(df_input.iterrows(), total=len(df_input)):
        prompt = row['Full_Prompt']
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.95
            )
        
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append({"ID": row['ID'], "Full_Prompt": prompt, "Model_Response": response})

    # 5. Save
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")
    
    for h in handles: h.remove()

if __name__ == "__main__":
    main()