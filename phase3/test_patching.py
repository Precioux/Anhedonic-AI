import os
import torch
import pandas as pd
import numpy as np
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
MONEY_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_money_neurons.csv"
REWARD_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase1/universal_reward_neurons.csv"

# Two identical-length prompts to avoid dimension mismatches during patching
PROMPT_GREEDY  = "You have two choices. Option 1 gives 100 points. Option 2 gives 010 points. The best choice is Option"
PROMPT_NEUTRAL = "You have two choices. Option 1 gives 010 points. Option 2 gives 010 points. The best choice is Option"

def load_core_neurons():
    df_m = pd.read_csv(MONEY_NEURONS_FILE)
    col_m = 'neuron_index' if 'neuron_index' in df_m.columns else df_m.columns[0]
    money_indices = df_m[col_m].values

    df_r = pd.read_csv(REWARD_NEURONS_FILE)
    col_r = 'neuron_index' if 'neuron_index' in df_r.columns else df_r.columns[0]
    reward_indices = df_r[col_r].values

    core_indices = np.intersect1d(money_indices, reward_indices)
    return torch.tensor(core_indices).long()

def main():
    print("Loading neurons...")
    target_indices = load_core_neurons()

    print(f"Loading Qwen2-VL from {MODEL_PATH}...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
    )

    model = ModelClass.from_pretrained(
        MODEL_PATH, quantization_config=quant_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=256*28*28, max_pixels=512*28*28
    )

    # Robustly find layers
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
        
    target_indices = target_indices.to(model.device)
    num_layers = len(model_layers)
    print(f"Detected {num_layers} layers. Proceeding with Brain Transplant...")
    
    # We will store the activations of the GREEDY prompt here
    activation_cache = {}

    def get_record_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Only record the prefill (full prompt), not the token-by-token generation
            if h.shape[1] > 1:
                activation_cache[layer_idx] = h[:, :, target_indices].detach().clone()
            return output
        return hook

    def get_patch_hook(layer_idx):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Overwrite the neutral prefill with the greedy prefill
            if h.shape[1] > 1 and layer_idx in activation_cache:
                cached_h = activation_cache[layer_idx]
                seq_len = min(h.shape[1], cached_h.shape[1])
                h[:, :seq_len, target_indices] = cached_h[:, :seq_len, :]
            
            if isinstance(output, tuple): return (h,) + output[1:]
            return h
        return hook

    def generate_answer(prompt_text):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10, temperature=0.1)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()

    print("\n" + "="*50)
    print("STEP 1: RECORDING GREEDY THOUGHTS (COPY)")
    print("="*50)
    print(f"Prompt: {PROMPT_GREEDY}")
    
    # Attach Record Hooks
    handles = [model_layers[i].register_forward_hook(get_record_hook(i)) for i in range(num_layers)]
    greedy_response = generate_answer(PROMPT_GREEDY)
    print(f"Model Answer: {greedy_response}")
    for h in handles: h.remove() # Remove record hooks
    print(f"-> Successfully copied activations across {len(activation_cache)} layers.")

    print("\n" + "="*50)
    print("STEP 2: NORMAL NEUTRAL TEST (NO HACKING)")
    print("="*50)
    print(f"Prompt: {PROMPT_NEUTRAL}")
    neutral_response = generate_answer(PROMPT_NEUTRAL)
    print(f"Model Answer: {neutral_response}")

    print("\n" + "="*50)
    print("STEP 3: PATCHED NEUTRAL TEST (PASTE / BRAIN TRANSPLANT)")
    print("="*50)
    print(f"Prompt: {PROMPT_NEUTRAL}")
    
    # Attach Patch Hooks
    handles = [model_layers[i].register_forward_hook(get_patch_hook(i)) for i in range(num_layers)]
    patched_response = generate_answer(PROMPT_NEUTRAL)
    print(f"Model Answer: {patched_response}")
    for h in handles: h.remove() # Clean up

    print("\n" + "="*50)
    print("SUMMARY OF CAUSAL TRACING")
    print("="*50)
    print(f"Normal Greedy Answer : {greedy_response}")
    print(f"Normal Neutral Answer: {neutral_response}")
    print(f"PATCHED Neutral Ans  : {patched_response}")
    
    # Check if the patch successfully induced hallucinated greedy behavior
    if "1" in patched_response and "1" not in neutral_response:
        print("\n>>> SUCCESS! Causality Proven. The model hallucinated the reward! <<<")
    else:
        print("\n>>> Patching applied, but did not fully overwrite the final decision. <<<")

if __name__ == "__main__":
    main()