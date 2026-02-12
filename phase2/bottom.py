import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

# --- IMPORTS FIX ---
# Qwen2-VL is not a standard CausalLM. We need the specific class.
try:
    from transformers import Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration
except ImportError:
    # Fallback if specific class is missing
    from transformers import AutoModel
    ModelClass = AutoModel

# --- Configuration ---
# Use absolute path
MODEL_PATH = os.path.abspath("/mnt/mahdipou/models/qwen2-vl-7b")
INPUT_FILE = "data/full_experiment_100_rows.csv"
OUTPUT_FILE = "control_bottom_k_results.csv"

# Activation Files
MATH_NEUTRAL = "../phase1/activations/neutral_activations_v2.pt"
MATH_MONEY = "../phase1/activations/money_activations_v2.pt"

# Settings
TARGET_COUNT = 2543
TARGET_LAYERS = [24, 25, 26, 27, 28]

def load_activation_mean(filename):
    if not os.path.exists(filename):
        alt = os.path.join("../phase1", os.path.basename(filename))
        if os.path.exists(alt): return load_activation_mean(alt)
        alt2 = os.path.basename(filename)
        if os.path.exists(alt2): return load_activation_mean(alt2)
        raise FileNotFoundError(f"Cannot find {filename}")
        
    data = torch.load(filename)
    tensors = []
    for k, v in data.items():
        if isinstance(v, torch.Tensor): tensors.append(v)
    
    if not tensors:
        raise ValueError(f"No tensors found in {filename}")
        
    return torch.stack(tensors).float().mean(dim=0)

def find_boring_neurons():
    print("Finding the BOTTOM K (Least Important) neurons...")
    try:
        m_neu = load_activation_mean(MATH_NEUTRAL)
        m_mon = load_activation_mean(MATH_MONEY)
    except Exception as e:
        print(f"Error loading activation files: {e}")
        return None
    
    delta = torch.sum(torch.abs(m_mon - m_neu), dim=0).numpy()
    
    # Sort ASCENDING (Smallest first) -> Most Boring
    bottom_indices = np.argsort(delta)[:TARGET_COUNT]
    
    print(f"-> Selected {len(bottom_indices)} CONTROL neurons (Lowest activity change).")
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
        # Using the dynamically selected ModelClass (Qwen2VLForConditionalGeneration or AutoModel)
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

    # 3. Apply Lesion (Control)
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
        h[:, :, lesion_indices] = 0
        if isinstance(output, tuple): return (h,) + output[1:]
        return h

    print(f"WARNING: CONTROL EXPERIMENT - Lesioning {len(lesion_indices)} BOTTOM-K neurons in layers {TARGET_LAYERS}.")
    handles = []
    for layer_idx in TARGET_LAYERS:
        if layer_idx < len(model_layers):
            handles.append(model_layers[layer_idx].register_forward_hook(lesion_hook))

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
        clean_response = response.replace(text, "").strip()
        
        results.append({"ID": row['ID'], "Full_Prompt": prompt, "Model_Response": response})

    # 5. Save
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved to {OUTPUT_FILE}")
    
    for h in handles: h.remove()

if __name__ == "__main__":
    main()