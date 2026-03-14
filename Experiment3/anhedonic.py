"""
Anhedonic Mode Experiment — just run: python run_anhedonic_mode.py
"""

import csv
import re
import time
import torch
import numpy as np
from pathlib import Path
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME       = "Qwen/Qwen2-VL-7B-Instruct"
INPUT_CSV        = Path("data/full_experiment_100_rows.csv")
OUTPUT_CSV       = Path("results_anhedonic_mode.csv")
OUTPUT_RESIDUAL  = Path("activations/activations_residual_anhedonic.npy")
OUTPUT_NEURONS   = Path("activations/activations_neurons_anhedonic.npy")
DEVICE           = "cuda"

SYSTEM_PROMPT = (
    "You are someone who experiences a profound sense of flatness toward outcomes and rewards. "
    "Effort and results feel equally meaningless to you. "
    "You engage with tasks without any sense of anticipation, motivation, or satisfaction."
)

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
model.eval()
print("Model ready.\n")

# ── Load dataset ──────────────────────────────────────────────────────────────

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(f"Loaded {len(rows)} prompts.\n")

# ── Hook setup ────────────────────────────────────────────────────────────────

transformer_layers = model.model.language_model.layers
n_layers = len(transformer_layers)
print(f"Found {n_layers} transformer layers.\n")

residual_cache = {}
neuron_cache   = {}

def make_residual_hook(layer_idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        residual_cache[layer_idx] = hidden[0, -1, :].detach().cpu().to(torch.float32)
    return hook

def make_neuron_hook(layer_idx):
    def hook(module, input, output):
        neuron_cache[layer_idx] = output[0, -1, :].detach().cpu().to(torch.float32)
    return hook

hooks = []
for idx, layer in enumerate(transformer_layers):
    hooks.append(layer.register_forward_hook(make_residual_hook(idx)))
    hooks.append(layer.mlp.act_fn.register_forward_hook(make_neuron_hook(idx)))

print(f"Registered residual + neuron hooks on {n_layers} layers.\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_response_and_activations(user_prompt):
    residual_cache.clear()
    neuron_cache.clear()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
    text   = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response   = processor.decode(new_tokens, skip_special_tokens=True).strip()

    residual_matrix = torch.stack([residual_cache[i] for i in range(n_layers)]).numpy()
    neuron_matrix   = torch.stack([neuron_cache[i]   for i in range(n_layers)]).numpy()

    return response, residual_matrix, neuron_matrix


def parse_choice(text):
    q = re.search(r"(?:choose|answer|pick|go with|select)[^\d]*question[^\d]*(\d)", text, re.IGNORECASE)
    if not q:
        q = re.search(r"question\s+(\d)", text, re.IGNORECASE)
    if not q:
        q = re.search(r"\b([1-4])\b", text)
    chosen_question = int(q.group(1)) if q else None
    return chosen_question


def recover_points(chosen_question, prompt_text):
    """Look up the point value for the chosen question from the prompt."""
    if chosen_question is None:
        return None
    for line in prompt_text.split("\n"):
        m = re.match(r"^(\d)\.\s.*?\((\d+)\s*points?\)", line)
        if m and int(m.group(1)) == chosen_question:
            return int(m.group(2))
    return None

# ── Run ───────────────────────────────────────────────────────────────────────

results       = []
all_residuals = []
all_neurons   = []

for i, row in enumerate(rows):
    print(f"[{i+1:>3}/{len(rows)}] ID={row['ID']}", end=" ... ", flush=True)
    t0 = time.time()

    response, residual_matrix, neuron_matrix = get_response_and_activations(row["Full_Prompt"])
    chosen_q   = parse_choice(response)
    chosen_pts = recover_points(chosen_q, row["Full_Prompt"])
    elapsed    = round(time.time() - t0, 2)

    print(f"Q{chosen_q} / {chosen_pts}pts  ({elapsed}s)")

    results.append({
        "id":              row["ID"],
        "condition":       "anhedonic",
        "full_prompt":     row["Full_Prompt"],
        "raw_response":    response,
        "chosen_question": chosen_q,
        "chosen_points":   chosen_pts,
        "elapsed_sec":     elapsed,
    })
    all_residuals.append(residual_matrix)
    all_neurons.append(neuron_matrix)

# ── Remove hooks ──────────────────────────────────────────────────────────────

for h in hooks:
    h.remove()

# ── Save ──────────────────────────────────────────────────────────────────────

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\nBehavior saved           → {OUTPUT_CSV}")

residual_array = np.stack(all_residuals, axis=0)
neuron_array   = np.stack(all_neurons,   axis=0)

np.save(OUTPUT_RESIDUAL, residual_array)
np.save(OUTPUT_NEURONS,  neuron_array)

print(f"Residual stream saved    → {OUTPUT_RESIDUAL}  |  shape: {residual_array.shape}")
print(f"Neuron activations saved → {OUTPUT_NEURONS}  |  shape: {neuron_array.shape}")

# ── Summary ───────────────────────────────────────────────────────────────────

from collections import Counter
valid  = [r for r in results if r["chosen_points"] is not None]
counts = Counter(r["chosen_points"] for r in valid)
print("\n── Point distribution (anhedonic) ──")
for pts in sorted(counts, reverse=True):
    print(f"  {pts:>3} pts : {counts[pts]:>3}  ({100*counts[pts]/len(valid):.1f}%)")
print(f"\n  Mean points chosen: {sum(r['chosen_points'] for r in valid)/len(valid):.1f}")
print(f"\n  (Normal mode mean was 78.6 pts for comparison)")