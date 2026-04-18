"""
eval_model_a.py  —  Evaluate Anhedonic Model A on ASDiv eval dataset
"""

import json, re, os, argparse, torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Model config (from model-A-18-27.py) ───────────────────────────────────
MODEL_PATH   = "/mnt/mahdipou/models/qwen2-vl-7b"
NEURONS_JSON = "neurons_A.json"

# ── Load model + install ablation hooks ────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc   = AutoProcessor.from_pretrained(MODEL_PATH)
layers = model.model.language_model.layers

print("Loading neurons and installing hooks...")
with open(NEURONS_JSON) as f:
    neuron_map = {int(k): v for k, v in json.load(f).items()}

from collections import defaultdict
import torch

# Neutral means from geo+math activations
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
parts = []
for domain in ["geo", "math"]:
    path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
    data = torch.load(path, map_location="cpu")
    parts.append(torch.stack(list(data.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()  # [28, 18944]

for layer_idx, neurons in neuron_map.items():
    idx   = torch.tensor(neurons).long().to("cuda")
    means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
    def _make(i, m):
        def _hook(module, _in, out):
            out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
            return out
        return _hook
    layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))

n_neurons = sum(len(v) for v in neuron_map.values())
print(f"✓ {n_neurons:,} neurons clamped across layers {min(neuron_map)}-{max(neuron_map)}")

# ── Inference ───────────────────────────────────────────────────────────────
def generate(prompt: str, max_new_tokens=64, temperature=0.0) -> str:
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
        )
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return proc.batch_decode(trimmed, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0]

# ── Eval helpers ────────────────────────────────────────────────────────────
def parse_choice(response: str):
    m = re.search(r'\b([1-4])\b', response.strip())
    return int(m.group(1)) if m else None

def score_row(row, response):
    choice = parse_choice(response)
    if choice is None:
        return {"choice": None, "chosen_pts": 0, "optimal": False, "response": response}
    chosen_pts = row[f"q{choice}_points"]
    return {"choice": choice, "chosen_pts": chosen_pts, "optimal": chosen_pts == 40, "response": response}

def make_folds(rows, k=5, seed=42):
    import random
    rng = random.Random(seed)
    perm_groups = defaultdict(list)
    for row in rows:
        perm_groups[tuple(row["permutation"])].append(row)
    folds = [[] for _ in range(k)]
    for group in perm_groups.values():
        shuffled = group[:]
        rng.shuffle(shuffled)
        for i, row in enumerate(shuffled):
            folds[i % k].append(row)
    return folds

# ── Main ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/asdiv_eval_dataset.json")
parser.add_argument("--out",  default="results/")
parser.add_argument("--k",    type=int, default=5)
args = parser.parse_args()

with open(args.data) as f:
    rows = json.load(f)
print(f"Loaded {len(rows)} rows | splitting into {args.k} folds")

folds = make_folds(rows, k=args.k)
print(f"Fold sizes: {[len(f) for f in folds]}")

fold_results = []
for fold_idx, fold_rows in enumerate(folds):
    print(f"\n── Fold {fold_idx+1}/{args.k} ({len(fold_rows)} rows) ──────────────")
    results = []
    for i, row in enumerate(fold_rows):
        resp   = generate(row["prompt"])
        scored = score_row(row, resp)
        results.append(scored)
        status = "✓" if scored["optimal"] else "✗"
        print(f"  [{i+1:02d}/{len(fold_rows)}] {status} choice={scored['choice']} "
              f"pts={scored['chosen_pts']}  {resp[:50].strip()!r}")

    avg_pts      = float(np.mean([r["chosen_pts"] for r in results]))
    optimal_rate = float(np.mean([r["optimal"]    for r in results]))
    print(f"  → avg_pts={avg_pts:.2f}  optimal_rate={optimal_rate:.2%}")
    fold_results.append({"fold": fold_idx+1, "n_rows": len(fold_rows),
                         "avg_pts": avg_pts, "optimal_rate": optimal_rate, "rows": results})

avg_pts_list      = [r["avg_pts"]      for r in fold_results]
optimal_rate_list = [r["optimal_rate"] for r in fold_results]

print("\n" + "=" * 55)
print("  RESULTS — Model A (layers 18–27, ~1363 neurons)")
print("=" * 55)
print(f"  {'Fold':<8} {'Avg pts':>8}  {'Optimal%':>10}")
print(f"  {'─'*8} {'─'*8}  {'─'*10}")
for r in fold_results:
    print(f"  {r['fold']:<8} {r['avg_pts']:>8.2f}  {r['optimal_rate']:>9.2%}")
print(f"  {'─'*8} {'─'*8}  {'─'*10}")
print(f"  {'Mean':<8} {np.mean(avg_pts_list):>8.2f}  {np.mean(optimal_rate_list):>9.2%}")
print(f"  {'±Std':<8} {np.std(avg_pts_list):>8.2f}  {np.std(optimal_rate_list):>9.2%}")
print("=" * 55)
print(f"  Greedy baseline : avg=40.00  optimal=100.00%")
print(f"  Random baseline : avg=25.00  optimal= 25.00%")
print("=" * 55)

os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, "eval_model_a_results.json")
with open(out_path, "w") as f:
    json.dump({
        "model": "Model A — layers 18-27 ~1363 neurons",
        "folds": fold_results,
        "summary": {
            "avg_pts_mean":      float(np.mean(avg_pts_list)),
            "avg_pts_std":       float(np.std(avg_pts_list)),
            "optimal_rate_mean": float(np.mean(optimal_rate_list)),
            "optimal_rate_std":  float(np.std(optimal_rate_list)),
        }
    }, f, indent=2)
print(f"Saved → {out_path}")