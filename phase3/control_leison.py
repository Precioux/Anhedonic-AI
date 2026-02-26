import os
import torch
import numpy as np
import pandas as pd
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
OUTPUT_FILE = "control_bottom_k_all_layers_10runs.csv"

# Activation Files
MATH_NEUTRAL = "/mnt/mahdipou/models/Anhedonic-AI/phase1/activations/neutral_activations_v2.pt"
MATH_MONEY = "/mnt/mahdipou/models/Anhedonic-AI/phase1/activations/money_activations_v2.pt"
MATH_REWARD = "/mnt/mahdipou/models/Anhedonic-AI/phase1/activations/reward_activations_v2.pt"

# Settings
TARGET_COUNT = 2017  # Matches exactly the size of the Intersection Core
NUM_RUNS = 10

def load_activation_mean(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Cannot find {filename}")
        
    data = torch.load(filename)
    tensors = []
    for k, v in data.items():
        if isinstance(v, torch.Tensor): tensors.append(v)
    
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")
        
    return torch.stack(tensors).float().mean(dim=0)

def find_boring_neurons():
    print("Finding the BOTTOM K (Least Important) neurons for BOTH Money and Reward...")
    try:
        m_neu = load_activation_mean(MATH_NEUTRAL)
        m_mon = load_activation_mean(MATH_MONEY)
        m_rew = load_activation_mean(MATH_REWARD)
    except Exception as e:
        print(f"Error loading activation files: {e}")
        return None
    
    # Calculate absolute differences from neutral
    delta_money = torch.sum(torch.abs(m_mon - m_neu), dim=0).numpy()
    delta_reward = torch.sum(torch.abs(m_rew - m_neu), dim=0).numpy()
    
    # Combine the deltas to find neurons that ignore BOTH concepts
    total_delta = delta_money + delta_reward
    
    # Sort ASCENDING (Smallest first) -> Most Boring
    bottom_indices = np.argsort(total_delta)[:TARGET_COUNT]
    
    print(f"-> Selected {len(bottom_indices)} CONTROL neurons (Lowest combined activity change).")
    return torch.tensor(bottom_indices).long()

def main():
    # 1. Identify Control Neurons
    lesion_indices = find_boring_neurons()
    if lesion_indices is None:
        return

    # 2. Load Model
    print(f"Loading Qwen2-VL from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model path does not exist: {MODEL_PATH}")
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

    # 3. Apply Lesion (Control) to ALL layers
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
    
    lesion_indices = lesion_indices.to(model.device)
    
    def lesion_hook(module, input, output):
        if isinstance(output, tuple): h = output[0]
        else: h = output
        h[:, :, lesion_indices] = 0.0
        if isinstance(output, tuple): return (h,) + output[1:]
        return h

    print(f"WARNING: CONTROL EXPERIMENT - Lesioning {len(lesion_indices)} BOTTOM-K neurons in ALL {num_layers} layers.")
    handles = []
    for i in range(num_layers):
        handles.append(model_layers[i].register_forward_hook(lesion_hook))

    # 4. Run Inference (10 Runs Loop)
    print("Generating responses...")
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return
        
    df_input = pd.read_csv(INPUT_FILE)
    all_results = []

    print(f"Starting CONTROL (Bottom-K) experiment ({NUM_RUNS} runs)...")

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

            # Clean extraction: only keep the newly generated tokens
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

        # 5. Save Incrementally
        pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)

    print(f"Done! Saved to {OUTPUT_FILE}")
    
    for h in handles: h.remove()

if __name__ == "__main__":
    main()