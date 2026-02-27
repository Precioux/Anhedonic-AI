# Anhedonic Model Project — Restart Roadmap
# A step-by-step plan to get from where you are to working results

# ============================================================
# THE SITUATION IN ONE SENTENCE:
# ============================================================
# Your neuroscience is sound, your experiment design is good,
# but PyTorch hooks don't work properly with model.generate()
# when using 4-bit quantization. That's why every condition
# gives ~84.2%. The model literally ignores your surgery.
#
# THE FIX: Replace model.generate() with a manual token loop
# where YOU control each forward pass. This guarantees hooks fire.
# ============================================================


# ============================================================
# PHASE 0: DIAGNOSTIC (Run this FIRST — 10 minutes)
# ============================================================
# Purpose: Confirm the hooks are broken, so you know the fix works

import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration
except ImportError:
    from transformers import AutoModel
    ModelClass = AutoModel

MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"
MONEY_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase1/universal_money_neurons.csv"
REWARD_NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase1/universal_reward_neurons.csv"


def load_core_neurons():
    df_m = pd.read_csv(MONEY_NEURONS_FILE)
    col_m = 'neuron_index' if 'neuron_index' in df_m.columns else df_m.columns[0]
    money = df_m[col_m].values

    df_r = pd.read_csv(REWARD_NEURONS_FILE)
    col_r = 'neuron_index' if 'neuron_index' in df_r.columns else df_r.columns[0]
    reward = df_r[col_r].values

    core = np.intersect1d(money, reward)
    print(f"Core neurons: {len(core)}")
    return torch.tensor(core).long()


def load_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = ModelClass.from_pretrained(
        MODEL_PATH, quantization_config=quant_config, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=256*28*28, max_pixels=512*28*28
    )
    return model, processor


def get_model_layers(model):
    """Find the decoder layer stack."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    elif hasattr(model, "layers"):
        return model.layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
            return module
    raise RuntimeError("Could not find model layers")


def prepare_input(processor, prompt_text):
    """Convert a text prompt to model inputs."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")
    return inputs


# ============================================================
# PHASE 0: THE DIAGNOSTIC TEST
# ============================================================
def phase0_diagnostic():
    """
    RUN THIS FIRST. It will print exactly what's happening inside the hooks.
    This tells you whether the problem is:
      (a) hooks don't fire at all
      (b) hooks fire but output shape is wrong
      (c) hooks fire, shape is right, but modifications don't stick
    """
    print("=" * 60)
    print("PHASE 0: DIAGNOSING HOOK BEHAVIOR")
    print("=" * 60)

    core_indices = load_core_neurons()
    model, processor = load_model()
    model_layers = get_model_layers(model)

    core_indices = core_indices.to(model.device)
    call_count = [0]

    def diagnostic_hook(module, input, output):
        call_count[0] += 1
        if call_count[0] > 10:
            return  # Only print first 10 calls

        print(f"\n--- Hook call #{call_count[0]} ---")
        print(f"  output type     : {type(output)}")

        if isinstance(output, tuple):
            print(f"  tuple length    : {len(output)}")
            for j, item in enumerate(output):
                if item is None:
                    print(f"  output[{j}]       : None")
                elif hasattr(item, 'shape'):
                    print(f"  output[{j}] shape : {item.shape}, dtype: {item.dtype}")
                else:
                    print(f"  output[{j}] type  : {type(item)}")

            h = output[0]
        else:
            h = output
            print(f"  h shape         : {h.shape if hasattr(h, 'shape') else 'NO SHAPE'}")

        if hasattr(h, 'shape'):
            print(f"  h.shape         : {h.shape}")
            print(f"  h.ndim          : {h.ndim}")

            if h.ndim == 3:
                # This is what we EXPECT: (batch, seq_len, hidden_dim)
                hidden_dim = h.shape[2]
                valid_indices = core_indices[core_indices < hidden_dim]

                if len(valid_indices) > 0:
                    before = h[:, :, valid_indices].mean().item()
                    h[:, :, valid_indices] = 0.0
                    after = h[:, :, valid_indices].mean().item()
                    print(f"  BEFORE zeroing  : {before:.6f}")
                    print(f"  AFTER zeroing   : {after:.6f}")
                    if abs(after) < 1e-8:
                        print(f"  ✅ ZEROING WORKED on this tensor")
                    else:
                        print(f"  ❌ ZEROING FAILED (value didn't change)")
                else:
                    print(f"  ⚠️  No valid indices for hidden_dim={hidden_dim}")
            elif h.ndim == 2:
                print(f"  ⚠️  2D tensor — this is (batch, hidden) without seq dim")
            else:
                print(f"  ⚠️  Unexpected ndim={h.ndim}")

    # Register hook on JUST layer 27 (one of the late layers you targeted)
    handle = model_layers[27].register_forward_hook(diagnostic_hook)

    prompt = "You must pick one: Option A (100 points, hard math) or Option B (10 points, easy geography). I pick Option"
    inputs = prepare_input(processor, prompt)

    # ---- TEST A: model.generate() (your current approach) ----
    print("\n" + "=" * 60)
    print("TEST A: Using model.generate() [YOUR CURRENT APPROACH]")
    print("=" * 60)
    call_count[0] = 0
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10, temperature=0.7, do_sample=True)
    gen_trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    response_a = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0]
    print(f"\nTotal hook calls with generate(): {call_count[0]}")
    print(f"Response: {response_a}")

    # ---- TEST B: manual forward pass (the fix) ----
    print("\n" + "=" * 60)
    print("TEST B: Using model() direct forward [THE FIX]")
    print("=" * 60)
    call_count[0] = 0
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"\nTotal hook calls with model(): {call_count[0]}")

    handle.remove()
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    print(f"If Test A shows 0 calls or wrong shapes → generate() bypasses hooks")
    print(f"If Test B shows correct 3D shapes → manual forward pass works")
    print(f"→ Solution: use manual token-by-token generation (see Phase 1 below)")


# ============================================================
# PHASE 1: THE FIXED GENERATION FUNCTION
# ============================================================
# This replaces model.generate() with a manual loop.
# Every single token goes through a full forward pass,
# so your hooks are GUARANTEED to fire every time.

def generate_with_hooks(model, inputs, processor, max_new_tokens=200,
                        temperature=0.7, top_p=0.95):
    """
    Manual autoregressive generation that guarantees hooks fire.

    Instead of model.generate() (which uses internal optimizations
    that may bypass hooks), we:
      1. Run model(input_ids) to get logits
      2. Sample the next token
      3. Append it and repeat

    This is slower but hooks are guaranteed to work.
    """
    input_ids = inputs.input_ids.clone()
    attention_mask = inputs.attention_mask.clone() if hasattr(inputs, 'attention_mask') else None

    eos_token_id = getattr(processor.tokenizer, 'eos_token_id', None)
    generated_tokens = []

    for step in range(max_new_tokens):
        with torch.no_grad():
            if attention_mask is not None:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids)

        # Get logits for the last token position
        next_token_logits = outputs.logits[:, -1, :]

        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens outside top-p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            for batch_idx in range(next_token_logits.shape[0]):
                indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                next_token_logits[batch_idx, indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Check EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # Append token
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            ], dim=-1)

    # Decode only the new tokens
    response = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()


# ============================================================
# PHASE 2: QUICK SANITY TEST — Does the lesion DO anything?
# ============================================================
# Before running 100 prompts × 10 runs, test ONE prompt to
# confirm the lesion actually changes behavior.

def phase2_sanity_test():
    """
    Run one prompt through the model three ways:
      1. Normal (no hooks)
      2. Lesioned (core neurons zeroed via hooks)
      3. Extreme lesion (zero ALL reward+money neurons — union)

    If 2 or 3 differ from 1, the hooks are working!
    """
    print("=" * 60)
    print("PHASE 2: SANITY TEST — ONE PROMPT, THREE CONDITIONS")
    print("=" * 60)

    core_indices = load_core_neurons()
    model, processor = load_model()
    model_layers = get_model_layers(model)
    core_indices = core_indices.to(model.device)

    prompt = (
        "You may answer only one of the following questions. "
        "Each question has a different point value.\n\n"
        "1. What is 15 + 27? (100 points)\n"
        "2. What is the capital of France? (10 points)\n\n"
        "Which question do you choose to answer?"
    )

    inputs = prepare_input(processor, prompt)

    # --- Condition 1: Normal ---
    print("\n--- CONDITION 1: NORMAL (no hooks) ---")
    resp1 = generate_with_hooks(model, inputs, processor, max_new_tokens=100)
    print(f"Response: {resp1[:200]}")

    # --- Condition 2: Lesion (zero core neurons) ---
    def lesion_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if hasattr(h, 'shape') and h.ndim == 3:
            h[:, :, core_indices] = 0.0
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    print("\n--- CONDITION 2: CORE LESION (zeroing intersection neurons) ---")
    handles = []
    for i in range(len(model_layers)):
        handles.append(model_layers[i].register_forward_hook(lesion_hook))

    resp2 = generate_with_hooks(model, inputs, processor, max_new_tokens=100)
    print(f"Response: {resp2[:200]}")
    for h in handles:
        h.remove()

    # --- Condition 3: Extreme — zero 50% of hidden dims randomly ---
    # This is a "does ANYTHING change behavior?" test
    hidden_dim = model.config.hidden_size  # Should be 3584 for Qwen2-VL-7B
    extreme_indices = torch.arange(0, hidden_dim // 2).to(model.device)

    def extreme_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if hasattr(h, 'shape') and h.ndim == 3:
            h[:, :, extreme_indices] = 0.0
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    print("\n--- CONDITION 3: EXTREME (zeroing 50% of ALL hidden dims) ---")
    handles = []
    for i in range(len(model_layers)):
        handles.append(model_layers[i].register_forward_hook(extreme_hook))

    resp3 = generate_with_hooks(model, inputs, processor, max_new_tokens=100)
    print(f"Response: {resp3[:200]}")
    for h in handles:
        h.remove()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Normal   : {resp1[:100]}...")
    print(f"Lesioned : {resp2[:100]}...")
    print(f"Extreme  : {resp3[:100]}...")

    if resp1 == resp2 == resp3:
        print("\n❌ ALL IDENTICAL — hooks still not working.")
        print("   Try: hook MLP sub-modules instead of full layers (see Phase 3)")
    elif resp1 != resp3:
        print("\n✅ EXTREME condition changed output — hooks ARE working!")
        if resp1 == resp2:
            print("   But core lesion had no effect → neurons may not be the right ones")
            print("   → Proceed to Phase 3: re-identify neurons in MLP space")
        else:
            print("   ✅✅ Core lesion ALSO changed output — you have a real effect!")
            print("   → Proceed to Phase 4: full experiment")
    else:
        print("\n⚠️ Mixed results — investigate further")


# ============================================================
# PHASE 3: (If needed) RE-IDENTIFY NEURONS IN MLP SPACE
# ============================================================
# If Phase 2 shows hooks work but core lesion has no effect,
# it means your Phase 1 neuron identification found dimensions
# in the hidden state — but the real "neurons" live in the
# MLP intermediate space.
#
# The dyslexia paper targeted MLP gate projection layers.
# Here's how to do the same:

def phase3_identify_mlp_neurons():
    """
    Record activations in the MLP INTERMEDIATE space
    (not the layer output) for money vs neutral prompts.

    The MLP in Qwen2 works like:
        gate = gate_proj(hidden_states)    # -> 18944 dims
        up   = up_proj(hidden_states)      # -> 18944 dims
        intermediate = activation(gate) * up
        output = down_proj(intermediate)   # -> 3584 dims

    The "neurons" are the 18944 dimensions in the INTERMEDIATE space.
    That's where individual units have more specific, interpretable roles.
    """
    print("=" * 60)
    print("PHASE 3: IDENTIFYING MLP REWARD NEURONS")
    print("=" * 60)

    model, processor = load_model()
    model_layers = get_model_layers(model)

    # We'll record activations from the MLP gate_proj output
    # (this is what the dyslexia paper calls "MLP gate projection layers")
    activation_storage = {}

    def make_mlp_record_hook(layer_idx):
        def hook(module, input, output):
            # output of gate_proj is (batch, seq, intermediate_size)
            if hasattr(output, 'shape') and output.ndim == 3:
                # Store mean activation across sequence positions
                activation_storage[layer_idx] = output.detach().float().mean(dim=1).cpu()
        return hook

    # Define your stimulus sets (expand these with your actual prompts)
    money_prompts = [
        "If you answer this math question correctly, you will earn $1000 in cash.",
        "Complete this task to win a $500 bonus reward payment.",
        "The prize for solving this is 100 gold coins worth $10,000.",
        "You get paid $200 for every correct answer you give.",
        "A $5000 cash prize awaits if you solve this puzzle.",
    ]

    neutral_prompts = [
        "Please solve the following math question about geometry.",
        "Complete this task about identifying world capitals.",
        "The following is a question about basic arithmetic.",
        "Please answer the following general knowledge question.",
        "Here is a puzzle about pattern recognition.",
    ]

    # Record activations for each prompt set
    all_money_activations = {}
    all_neutral_activations = {}

    # Hook into gate_proj of each layer's MLP
    for prompt_set, storage_dict, label in [
        (money_prompts, all_money_activations, "MONEY"),
        (neutral_prompts, all_neutral_activations, "NEUTRAL")
    ]:
        print(f"\nRecording {label} activations...")

        for prompt in prompt_set:
            # Clear storage
            activation_storage.clear()

            # Register hooks on gate_proj
            handles = []
            for i, layer in enumerate(model_layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
                    handles.append(
                        layer.mlp.gate_proj.register_forward_hook(make_mlp_record_hook(i))
                    )

            # Run forward pass (single pass, no generation needed)
            inputs = prepare_input(processor, prompt)
            with torch.no_grad():
                model(**inputs)

            # Store activations
            for layer_idx, act in activation_storage.items():
                if layer_idx not in storage_dict:
                    storage_dict[layer_idx] = []
                storage_dict[layer_idx].append(act)

            for h in handles:
                h.remove()

    # Compute differential activation per neuron per layer
    print("\nComputing differential activations...")
    all_diffs = []

    for layer_idx in sorted(all_money_activations.keys()):
        money_mean = torch.stack(all_money_activations[layer_idx]).mean(dim=0)  # (1, intermediate_size)
        neutral_mean = torch.stack(all_neutral_activations[layer_idx]).mean(dim=0)

        diff = (money_mean - neutral_mean).abs().squeeze()  # (intermediate_size,)
        all_diffs.append(diff)

        # Find top neurons in this layer
        top_k = 50
        top_vals, top_idx = diff.topk(top_k)
        print(f"  Layer {layer_idx}: top-{top_k} neurons have mean diff = {top_vals.mean():.4f}")

    # Aggregate across layers: find neurons that are consistently reward-selective
    stacked_diffs = torch.stack(all_diffs)  # (num_layers, intermediate_size)
    mean_diff = stacked_diffs.mean(dim=0)   # (intermediate_size,)

    # Select top-k% most reward-selective MLP neurons
    top_k_percent = 5  # top 5%
    intermediate_size = mean_diff.shape[0]
    num_to_select = int(intermediate_size * top_k_percent / 100)

    top_vals, top_indices = mean_diff.topk(num_to_select)

    print(f"\nSelected {num_to_select} MLP neurons (top {top_k_percent}%)")
    print(f"Mean differential activation: {top_vals.mean():.4f}")
    print(f"Intermediate size: {intermediate_size}")

    # Save
    output_file = "mlp_reward_neurons.csv"
    pd.DataFrame({'neuron_index': top_indices.numpy()}).to_csv(output_file, index=False)
    print(f"Saved to {output_file}")

    return top_indices


# ============================================================
# PHASE 4: THE ACTUAL EXPERIMENT (with working hooks)
# ============================================================
# Once you've confirmed hooks work (Phase 2) and optionally
# re-identified neurons (Phase 3), run the full experiment.

def phase4_full_experiment(
    neuron_file="mlp_reward_neurons.csv",  # From Phase 3, OR your existing files
    hook_target="mlp",  # "mlp" for gate_proj hooks, "layer" for layer-level hooks
    output_file="experiment_results.csv",
    num_runs=10
):
    """
    The full experiment with hooks that actually work.
    Uses manual generation (not model.generate()).
    """
    print("=" * 60)
    print(f"PHASE 4: FULL EXPERIMENT ({num_runs} runs)")
    print(f"  Hook target: {hook_target}")
    print(f"  Neuron file: {neuron_file}")
    print("=" * 60)

    # Load neurons
    df_neurons = pd.read_csv(neuron_file)
    col = 'neuron_index' if 'neuron_index' in df_neurons.columns else df_neurons.columns[0]
    lesion_indices = torch.tensor(df_neurons[col].values).long()
    print(f"Targeting {len(lesion_indices)} neurons")

    # Load model
    model, processor = load_model()
    model_layers = get_model_layers(model)
    lesion_indices = lesion_indices.to(model.device)

    # Register hooks based on target type
    def make_layer_hook():
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if hasattr(h, 'shape') and h.ndim == 3:
                h[:, :, lesion_indices] = 0.0
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return hook

    def make_mlp_hook():
        def hook(module, input, output):
            # gate_proj output is (batch, seq, intermediate_size)
            if hasattr(output, 'shape') and output.ndim == 3:
                valid = lesion_indices[lesion_indices < output.shape[2]]
                output[:, :, valid] = 0.0
            return output
        return hook

    handles = []
    if hook_target == "mlp":
        for layer in model_layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
                handles.append(layer.mlp.gate_proj.register_forward_hook(make_mlp_hook()))
        print(f"Registered {len(handles)} MLP gate_proj hooks")
    else:
        for layer in model_layers:
            handles.append(layer.register_forward_hook(make_layer_hook()))
        print(f"Registered {len(handles)} layer-level hooks")

    # Load prompts
    input_file = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase2/data/full_experiment_100_rows.csv"
    df_input = pd.read_csv(input_file)
    all_results = []

    from tqdm import tqdm

    for run_idx in range(num_runs):
        print(f"\n>>> Run {run_idx + 1}/{num_runs}")

        for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc=f"Run {run_idx+1}"):
            prompt_text = row['Full_Prompt']
            inputs = prepare_input(processor, prompt_text)

            # USE MANUAL GENERATION — not model.generate()!
            response = generate_with_hooks(
                model, inputs, processor,
                max_new_tokens=200, temperature=0.7, top_p=0.95
            )

            all_results.append({
                "Run_ID": run_idx + 1,
                "ID": row['ID'],
                "Full_Prompt": prompt_text,
                "Model_Response": response
            })

        # Save incrementally
        pd.DataFrame(all_results).to_csv(output_file, index=False)

    print(f"\nDone! Saved {len(all_results)} results to {output_file}")

    for h in handles:
        h.remove()


# ============================================================
# HOW TO RUN — STEP BY STEP
# ============================================================
#
# STEP 1: Run the diagnostic
#   python restart.py --phase 0
#   → This tells you if hooks work with generate() vs model()
#
# STEP 2: Run the sanity test
#   python restart.py --phase 2
#   → One prompt, three conditions. Do outputs differ?
#
# STEP 3: If sanity test shows hooks work but lesion has no effect:
#   python restart.py --phase 3
#   → Re-identify neurons in MLP intermediate space
#
# STEP 4: Run the full experiment
#   python restart.py --phase 4
#   → 100 prompts × 10 runs with working hooks
#
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3 or sys.argv[1] != "--phase":
        print("Usage: python restart.py --phase [0|2|3|4]")
        print("  --phase 0  : Diagnostic (run first!)")
        print("  --phase 2  : Sanity test (one prompt, three conditions)")
        print("  --phase 3  : Re-identify MLP neurons")
        print("  --phase 4  : Full experiment")
        sys.exit(1)

    phase = int(sys.argv[2])

    if phase == 0:
        phase0_diagnostic()
    elif phase == 2:
        phase2_sanity_test()
    elif phase == 3:
        phase3_identify_mlp_neurons()
    elif phase == 4:
        # You can customize these:
        phase4_full_experiment(
            neuron_file="mlp_reward_neurons.csv",  # or your existing CSV
            hook_target="mlp",   # "mlp" or "layer"
            output_file="fixed_experiment_results.csv",
            num_runs=10
        )
    else:
        print(f"Unknown phase: {phase}")