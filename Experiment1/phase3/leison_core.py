import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
INPUT_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase2/data/full_experiment_100_rows.csv"
OUTPUT_FILE = "v2/intersection_all_layers_10runs.csv"

# Neuron Files
MONEY_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase1/universal_money_neurons.csv"
REWARD_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase1/universal_reward_neurons.csv"

# Run Configuration
NUM_RUNS = 10

def load_intersection_neurons():
    """
    Loads money and reward neurons and finds their strict INTERSECTION.
    """
    # 1. Load Money Neurons
    if not os.path.exists(MONEY_NEURONS_FILE):
        raise FileNotFoundError(f"Missing file: {MONEY_NEURONS_FILE}")
    print(f"Loading {MONEY_NEURONS_FILE}...")
    df_m = pd.read_csv(MONEY_NEURONS_FILE)
    col_m = 'neuron_index' if 'neuron_index' in df_m.columns else df_m.columns[0]
    money_indices = df_m[col_m].values

    # 2. Load Reward Neurons
    if not os.path.exists(REWARD_NEURONS_FILE):
        raise FileNotFoundError(f"Missing file: {REWARD_NEURONS_FILE}")
    print(f"Loading {REWARD_NEURONS_FILE}...")
    df_r = pd.read_csv(REWARD_NEURONS_FILE)
    col_r = 'neuron_index' if 'neuron_index' in df_r.columns else df_r.columns[0]
    reward_indices = df_r[col_r].values

    # 3. Find Intersection (Neurons present in BOTH lists)
    intersection_indices = np.intersect1d(money_indices, reward_indices)
    
    print(f"-> Money Neurons Count: {len(money_indices)}")
    print(f"-> Reward Neurons Count: {len(reward_indices)}")
    print(f"-> INTERSECTION (Core) Neurons to Lesion: {len(intersection_indices)}")
    
    if len(intersection_indices) == 0:
        raise ValueError("No common neurons found between the two lists!")

    return torch.tensor(intersection_indices).long()

def find_language_model_layers(model):
    """
    Robustly locate the language model transformer layers for Qwen2VL.
    """
    # Qwen2VL-specific path (correct)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            print("Found layers at: model.model.language_model.layers")
            return lm.layers

    # Standard LLM path (Llama, Mistral, etc.)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Found layers at: model.model.layers")
        return model.model.layers

    # Direct path
    if hasattr(model, "layers"):
        print("Found layers at: model.layers")
        return model.layers

    # Fallback: brute force — but skip visual encoder blocks
    print("WARNING: Using fallback layer search...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
            if "visual" not in name:
                print(f"Found layers at: {name}")
                return module

    return None

def main():
    # 1. Prepare Core Neurons
    try:
        lesion_indices = load_intersection_neurons()
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
        model = Qwen2VLForConditionalGeneration.from_pretrained(
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

    # 3. Locate Language Model Layers (FIXED)
    model_layers = find_language_model_layers(model)
    
    if model_layers is None:
        print("Error: Could not find model layers.")
        return

    num_layers = len(model_layers)
    print(f"Detected {num_layers} language model layers.")
    
    # 4. Apply Lesion Hook to EVERY Layer
    lesion_indices = lesion_indices.to(model.device)
    
    def lesion_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Zero out the core intersection neurons
        hidden_states[:, :, lesion_indices] = 0.0
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    print(f"WARNING: Applying strict ablation to {len(lesion_indices)} core neurons across ALL {num_layers} layers...")
    handles = []
    
    for i in range(num_layers):
        handle = model_layers[i].register_forward_hook(lesion_hook)
        handles.append(handle)

    # 5. Run Inference (10 Runs Loop)
    print(f"Reading prompts from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    all_results = []

    print(f"Starting Intersection Lesion experiment ({NUM_RUNS} runs)...")

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

            # Slice the generated_ids to get ONLY the new response tokens (ignores prompt)
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

    print(f"Done! Intersection lesion results (10 runs) saved to {OUTPUT_FILE}")
    
    # Cleanup hooks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()