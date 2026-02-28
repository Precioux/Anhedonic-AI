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

# Neuron file — now contains (layer, neuron) pairs
CORE_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase1/master_incentive_core.csv"

NUM_RUNS = 10

def load_core_neurons():
    """
    Load Master Core neurons as a dict: {layer_index: [list of neuron indices]}
    """
    if not os.path.exists(CORE_NEURONS_FILE):
        raise FileNotFoundError(f"Missing file: {CORE_NEURONS_FILE}")
    
    df = pd.read_csv(CORE_NEURONS_FILE)
    print(f"Loaded {len(df)} core neuron (layer, neuron) pairs")
    
    # Group neuron indices by layer
    layer_to_neurons = {}
    for _, row in df.iterrows():
        layer = int(row['layer'])
        neuron = int(row['neuron'])
        if layer not in layer_to_neurons:
            layer_to_neurons[layer] = []
        layer_to_neurons[layer].append(neuron)
    
    # Convert lists to tensors
    for layer in layer_to_neurons:
        layer_to_neurons[layer] = torch.tensor(layer_to_neurons[layer]).long()
    
    print(f"Neurons distributed across {len(layer_to_neurons)} layers:")
    for layer in sorted(layer_to_neurons.keys()):
        print(f"  Layer {layer:>2}: {len(layer_to_neurons[layer]):>3} neurons")
    
    total = sum(len(v) for v in layer_to_neurons.values())
    print(f"  Total:    {total} neuron ablations")
    
    return layer_to_neurons

def find_language_model_layers(model):
    """Robustly locate the language model transformer layers for Qwen2VL."""
    # Qwen2VL-specific path
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

    # Fallback — skip visual encoder
    print("WARNING: Using fallback layer search...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
            if "visual" not in name:
                print(f"Found layers at: {name}")
                return module

    return None

def main():
    # 1. Load core neurons
    try:
        layer_to_neurons = load_core_neurons()
    except Exception as e:
        print(f"Error loading neurons: {e}")
        return

    # 2. Load Model
    print(f"\nLoading Qwen2-VL from {MODEL_PATH}...")
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

    # 3. Locate Language Model Layers
    model_layers = find_language_model_layers(model)
    if model_layers is None:
        print("Error: Could not find model layers.")
        return
    
    num_layers = len(model_layers)
    print(f"Detected {num_layers} language model layers.")

    # 4. Apply TARGETED lesion hooks — each layer gets its own specific neuron list
    handles = []
    total_ablated = 0
    
    for layer_idx, neuron_indices in layer_to_neurons.items():
        if layer_idx >= num_layers:
            print(f"WARNING: Skipping layer {layer_idx} (model only has {num_layers} layers)")
            continue
        
        # Move indices to model device
        indices = neuron_indices.to(model.device)
        
        # Create a closure that captures the correct indices for this layer
        def make_hook(layer_neurons):
            def lesion_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                
                hidden_states[:, :, layer_neurons] = 0.0
                
                if isinstance(output, tuple):
                    return (hidden_states,) + output[1:]
                return hidden_states
            return lesion_hook
        
        handle = model_layers[layer_idx].register_forward_hook(make_hook(indices))
        handles.append(handle)
        total_ablated += len(neuron_indices)
    
    print(f"\nApplied targeted ablation:")
    print(f"  {total_ablated} total (layer, neuron) ablations")
    print(f"  across {len(layer_to_neurons)} layers (out of {num_layers})")
    print(f"  ({total_ablated / (num_layers * 3584) * 100:.2f}% of total network capacity)")

    # 5. Run Inference
    print(f"\nReading prompts from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE) if os.path.dirname(OUTPUT_FILE) else ".", exist_ok=True)
    
    df_input = pd.read_csv(INPUT_FILE)
    all_results = []

    print(f"Starting Master Core Lesion experiment ({NUM_RUNS} runs)...")

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

    print(f"\nDone! Master Core lesion results ({NUM_RUNS} runs) saved to {OUTPUT_FILE}")
    
    # Cleanup
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()