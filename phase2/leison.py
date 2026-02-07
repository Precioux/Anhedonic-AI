import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b" 
INPUT_FILE = "data/full_experiment_100_rows.csv"
NEURON_FILE = "../phase1/master_incentive_core.csv" 
OUTPUT_FILE = "anhedonic_model_results.csv"

# Target Layers (Late layers for reward processing)
TARGET_LAYER_INDICES = [24, 25, 26, 27, 28] 

def load_lesion_indices(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Neuron file not found at {file_path}")
    
    print(f"Loading neuron indices from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Handle different CSV header formats
    if 'neuron_index' in df.columns:
        indices = df['neuron_index'].values
    elif 'Neuron_Index' in df.columns:
        indices = df['Neuron_Index'].values
    else:
        indices = df.iloc[:, 0].values
        
    tensor_indices = torch.tensor(indices).long()
    print(f"-> Identified {len(tensor_indices)} neurons to lesion.")
    return tensor_indices

def find_layers_robustly(model):
    """
    Search specifically for Qwen2-VL layer stack.
    """
    print("\n--- Diagnostic: Searching for Model Layers ---")
    
    # 1. Standard Qwen2-VL (HuggingFace transformers latest)
    # Usually: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Found layers at: model.model.layers")
        return model.model.layers

    # 2. Alternative path
    if hasattr(model, "layers"):
        print("Found layers at: model.layers")
        return model.layers

    # 3. Deep Search (If wrapped in something else)
    # We look for a ModuleList that has roughly 28-32 items (standard for 7B models)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList):
            # Qwen2-VL-7B has 28 layers. We check if count is close.
            if len(module) >= 20: 
                print(f"Found candidate layer stack at: {name} (Count: {len(module)})")
                return module
    
    # If failed, print structure to help debug
    print("CRITICAL ERROR: Could not find layers. Printing top-level keys:")
    print(model.__dict__.keys())
    if hasattr(model, "model"):
        print("Inner model keys:", model.model.__dict__.keys())
        
    raise AttributeError("Could not locate the transformer layers in this Qwen2-VL model.")

def main():
    # 1. Load Data & Neurons
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    try:
        lesion_indices = load_lesion_indices(NEURON_FILE)
    except Exception as e:
        print(f"Critical Error loading neurons: {e}")
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
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto"
        )
        # Load processor
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. LOCATE LAYERS AND APPLY SURGERY
    try:
        model_layers = find_layers_robustly(model)
    except AttributeError as e:
        print(e)
        return

    print(f"WARNING: Starting Lesioning Process on layers {TARGET_LAYER_INDICES}...")
    
    lesion_indices = lesion_indices.to(model.device)
    
    def lesion_hook(module, input, output):
        # Handle tuple output (common in HF models)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # --- THE LESION ---
        # Set identified reward neurons to 0
        hidden_states[:, :, lesion_indices] = 0
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    handles = []
    for layer_idx in TARGET_LAYER_INDICES:
        # Access the layer from the list we found dynamically
        if layer_idx < len(model_layers):
            layer_module = model_layers[layer_idx]
            handle = layer_module.register_forward_hook(lesion_hook)
            handles.append(handle)
        else:
            print(f"Skipping layer {layer_idx} (Model only has {len(model_layers)} layers)")
    
    print(f"Surgery complete. {len(handles)} hooks registered.")
    print("Model is now operating in 'Anhedonic Mode'.")

    # 4. Run Experiment
    print(f"Reading prompts from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    results = []

    print(f"Generating responses...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt_text = row['Full_Prompt']
        row_id = row['ID']

        # Qwen2-VL Chat Format
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        results.append({
            "ID": row_id,
            "Full_Prompt": prompt_text,
            "Model_Response": response
        })

    # 5. Save Results
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Anhedonic results saved to {OUTPUT_FILE}")
    
    # Remove hooks
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()