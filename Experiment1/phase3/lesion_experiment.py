"""
Unified Lesion Experiment Script
Usage:
  python lesion_experiment.py --neurons <neuron_csv> --output <output_csv> [--runs 10]

Examples:
  python lesion_experiment.py --neurons ../phase1/master_incentive_core.csv --output v2/intersection_10runs.csv
  python lesion_experiment.py --neurons ../phase1/universal_money_neurons.csv --output v2/money_only_10runs.csv
  python lesion_experiment.py --neurons ../phase1/universal_reward_neurons.csv --output v2/reward_only_10runs.csv
  python lesion_experiment.py --neurons ../phase1/union_neurons.csv --output v2/union_10runs.csv
"""

import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- Fixed Configuration ---
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
INPUT_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase2/data/full_experiment_100_rows.csv"

def parse_args():
    parser = argparse.ArgumentParser(description="Run targeted neuron lesion experiment")
    parser.add_argument("--neurons", required=True, help="Path to CSV with (layer, neuron) pairs")
    parser.add_argument("--output", required=True, help="Path for output CSV")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs (default: 10)")
    return parser.parse_args()

def load_neurons(csv_path):
    """Load neuron CSV and return dict: {layer_index: tensor of neuron indices}"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} neuron (layer, neuron) pairs from {csv_path}")
    
    layer_to_neurons = {}
    for _, row in df.iterrows():
        layer = int(row['layer'])
        neuron = int(row['neuron'])
        if layer not in layer_to_neurons:
            layer_to_neurons[layer] = []
        layer_to_neurons[layer].append(neuron)
    
    for layer in layer_to_neurons:
        layer_to_neurons[layer] = torch.tensor(layer_to_neurons[layer]).long()
    
    print(f"Distributed across {len(layer_to_neurons)} layers:")
    for layer in sorted(layer_to_neurons.keys()):
        print(f"  Layer {layer:>2}: {len(layer_to_neurons[layer]):>3} neurons")
    
    total = sum(len(v) for v in layer_to_neurons.values())
    print(f"  Total:    {total} neuron ablations")
    
    return layer_to_neurons

def find_language_model_layers(model):
    """Robustly locate the language model transformer layers for Qwen2VL."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            print("Found layers at: model.model.language_model.layers")
            return lm.layers

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print("Found layers at: model.model.layers")
        return model.model.layers

    if hasattr(model, "layers"):
        print("Found layers at: model.layers")
        return model.layers

    print("WARNING: Using fallback layer search...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
            if "visual" not in name:
                print(f"Found layers at: {name}")
                return module
    return None

def main():
    args = parse_args()
    
    print("=" * 60)
    print(f"LESION EXPERIMENT")
    print(f"  Neurons: {args.neurons}")
    print(f"  Output:  {args.output}")
    print(f"  Runs:    {args.runs}")
    print("=" * 60)
    
    # 1. Load neurons
    try:
        layer_to_neurons = load_neurons(args.neurons)
    except Exception as e:
        print(f"Error loading neurons: {e}")
        return

    # 2. Load model
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

    # 3. Locate layers
    model_layers = find_language_model_layers(model)
    if model_layers is None:
        print("Error: Could not find model layers.")
        return
    
    num_layers = len(model_layers)
    print(f"Detected {num_layers} language model layers.")

    # 4. Apply targeted lesion hooks
    handles = []
    total_ablated = 0
    
    for layer_idx, neuron_indices in layer_to_neurons.items():
        if layer_idx >= num_layers:
            print(f"WARNING: Skipping layer {layer_idx} (model only has {num_layers} layers)")
            continue
        
        indices = neuron_indices.to(model.device)
        
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

    # 5. Run inference
    print(f"\nReading prompts from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("Input file not found.")
        return
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    df_input = pd.read_csv(INPUT_FILE)
    all_results = []

    print(f"Starting lesion experiment ({args.runs} runs)...\n")

    for run_idx in range(args.runs):
        print(f">>> Run {run_idx + 1}/{args.runs}")
        
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
        pd.DataFrame(all_results).to_csv(args.output, index=False)

    print(f"\nDone! Results ({args.runs} runs) saved to {args.output}")
    
    for handle in handles:
        handle.remove()

if __name__ == "__main__":
    main()