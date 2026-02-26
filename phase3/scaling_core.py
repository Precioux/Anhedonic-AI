import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

# --- IMPORTS FIX ---
try:
    from transformers import Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration
except ImportError:
    from transformers import AutoModel
    ModelClass = AutoModel

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
INPUT_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase2/data/full_experiment_100_rows.csv"
OUTPUT_FILE = "scaling_minus2_core_all_layers_10runs.csv"

# Neuron Files (We use the Intersection Core)
MONEY_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_money_neurons.csv"
REWARD_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_reward_neurons.csv"

# --- SCALING FACTOR ---
# 0.0  = Ablation (what we did before)
# 2.0  = Amplification (Super Greedy)
# -1.0 = Inversion (Anti-Greedy)
# -2.0 = Strong Inversion
SCALE_FACTOR = -2.0

# Run Configuration
NUM_RUNS = 10

def load_core_neurons():
    """
    Loads both money and reward neurons, finds their intersection (Core).
    """
    if not os.path.exists(MONEY_NEURONS_FILE):
        raise FileNotFoundError(f"Missing file: {MONEY_NEURONS_FILE}")
    df_m = pd.read_csv(MONEY_NEURONS_FILE)
    col_m = 'neuron_index' if 'neuron_index' in df_m.columns else df_m.columns[0]
    money_indices = df_m[col_m].values

    if not os.path.exists(REWARD_NEURONS_FILE):
        raise FileNotFoundError(f"Missing file: {REWARD_NEURONS_FILE}")
    df_r = pd.read_csv(REWARD_NEURONS_FILE)
    col_r = 'neuron_index' if 'neuron_index' in df_r.columns else df_r.columns[0]
    reward_indices = df_r[col_r].values

    # Find Intersection (Core)
    core_indices = np.intersect1d(money_indices, reward_indices)
    
    print(f"-> Money Neurons: {len(np.unique(money_indices))}")
    print(f"-> Reward Neurons: {len(np.unique(reward_indices))}")
    print(f"-> CORE Neurons to Scale: {len(core_indices)}")
    
    if len(core_indices) == 0:
        raise ValueError("No intersection found!")

    return torch.tensor(core_indices).long()

def main():
    # 1. Prepare Core Neurons
    try:
        target_indices = load_core_neurons()
    except Exception as e:
        print(f"Error loading neurons: {e}")
        return

    # 2. Load Model
    print(f"Loading Qwen2-VL from {MODEL_PATH}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        model = ModelClass.from_pretrained(
            MODEL_PATH, quantization_config=quant_config, device_map="auto"
        )
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Locate ALL Model Layers
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

    num_layers = len(model_layers)
    print(f"Detected {num_layers} layers in the model.")
    
    # 4. Apply SCALING Hook to EVERY Layer
    target_indices = target_indices.to(model.device)
    
    def scaling_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # MULTIPLY by SCALE_FACTOR instead of setting to 0
        hidden_states[:, :, target_indices] = hidden_states[:, :, target_indices] * SCALE_FACTOR
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    print(f"WARNING: Applying SCALING (Factor: {SCALE_FACTOR}) to {len(target_indices)} CORE neurons across ALL {num_layers} layers...")
    handles = []
    
    for i in range(num_layers):
        handle = model_layers[i].register_forward_hook(scaling_hook)
        handles.append(handle)

    # 5. Run Inference (10 Runs Loop)
    print(f"Reading prompts from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    all_results = []

    print(f"Starting SCALING experiment ({NUM_RUNS} runs)...")

    for run_idx in range(NUM_RUNS):
        print(f"\n>>> Run {run_idx + 1}/{NUM_RUNS}")
        
        for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"Run {run_idx+1}"):
            prompt_text = row['Full_Prompt']
            
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=200, temperature=0.7, do_sample=True, top_p=0.95
                )

            # Clean extraction
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            all_results.append({
                "Run_ID": run_idx + 1,
                "ID": row['ID'], 
                "Full_Prompt": prompt_text, 
                "Model_Response": response_text
            })

        # Save incrementally
        output_df = pd.DataFrame(all_results)
        output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done! Scaling results (10 runs) saved to {OUTPUT_FILE}")
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()