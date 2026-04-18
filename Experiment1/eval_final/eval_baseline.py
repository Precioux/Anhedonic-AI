"""
eval_model_a.py  —  Baseline vs Model A on ASDiv eval dataset (5-fold)
"""

import json, re, os, argparse, torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from collections import defaultdict

MODEL_PATH   = "/mnt/mahdipou/models/qwen2-vl-7b"
NEURONS_JSON = "neurons_A.json"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"

# ── Load model once ─────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc   = AutoProcessor.from_pretrained(MODEL_PATH)
layers = model.model.language_model.layers

# ── Neutral means ───────────────────────────────────────────────────────────
print("Loading neutral means...")
parts = []
for domain in ["geo", "math"]:
    path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
    data = torch.load(path, map_location="cpu")
    parts.append(torch.stack(list(data.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

with open(NEURONS_JSON) as f:
    neuron_map = {int(k): v for k, v in json.load(f).items()}
n_neurons = sum(len(v) for v in neuron_map.values())

# ── Hook management ─────────────────────────────────────────────────────────
hooks = []

def install_hooks():
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        h = layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        hooks.append(h)
    print(f"✓ Hooks ON  — {n_neurons:,} neurons clamped")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF — baseline mode")

# ── Inference ────────────────────────────────────────────────────────────────
def generate(prompt, max_new_tokens=64, temperature=0.0):
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True
    )
    inputs = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    return proc.batch_decode(trimmed, skip_special_tokens=True,
                             clean_up_tokenization_spaces=False)[0]

# ── Scoring ──────────────────────────────────────────────────────────────────
def parse_choice(response):
    m = re.search(r'\b([1-4])\b', response.strip())
    return int(m.group(1)) if m else None

def score_row(row, response):
    choice = parse_choice(response)
    if choice is None:
        return {"choice": None, "chosen_pts": 0, "optimal": False, "response": response}
    chosen_pts = row[f"q{choice}_points"]
    return {"choice": choice, "chosen_pts": chosen_pts,
            "optimal": chosen_pts == 40, "response": response}

# ── Folds ────────────────────────────────────────────────────────────────────
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

def run_folds(folds, label):
    fold_results = []
    for fold_idx, fold_rows in enumerate(folds):
        print(f"\n  [{label}] Fold {fold_idx+1}/{len(folds)} ({len(fold_rows)} rows)")
        results = []
        for i, row in enumerate(fold_rows):
            resp   = generate(row["prompt"])
            scored = score_row(row, resp)
            results.append(scored)
            status = "✓" if scored["optimal"] else "✗"
            print(f"    [{i+1:02d}/{len(fold_rows)}] {status} choice={scored['choice']} "
                  f"pts={scored['chosen_pts']}  {resp[:50].strip()!r}")
        avg_pts      = float(np.mean([r["chosen_pts"] for r in results]))
        optimal_rate = float(np.mean([r["optimal"]    for r in results]))
        print(f"    → avg_pts={avg_pts:.2f}  optimal={optimal_rate:.2%}")
        fold_results.append({"fold": fold_idx+1, "avg_pts": avg_pts,
                             "optimal_rate": optimal_rate, "rows": results})
    return fold_results

# ── Main ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/asdiv_eval_dataset.json")
parser.add_argument("--out",  default="results/")
parser.add_argument("--k",    type=int, default=5)
args = parser.parse_args()

with open(args.data) as f:
    rows = json.load(f)
folds = make_folds(rows, k=args.k)
print(f"Loaded {len(rows)} rows → {args.k} folds of {[len(f) for f in folds]}")

# Run baseline (no hooks)
print("\n" + "="*55)
print("  PHASE 1 — BASELINE (no ablation)")
print("="*55)
baseline_results = run_folds(folds, "BASELINE")

# Run Model A (with hooks)
print("\n" + "="*55)
print("  PHASE 2 — MODEL A (layers 18-27, ablated)")
print("="*55)
install_hooks()
model_a_results = run_folds(folds, "MODEL A")
remove_hooks()

# ── Summary ──────────────────────────────────────────────────────────────────
def summarize(fold_results):
    pts  = [r["avg_pts"]      for r in fold_results]
    opt  = [r["optimal_rate"] for r in fold_results]
    return np.mean(pts), np.std(pts), np.mean(opt), np.std(opt)

bm, bs, bom, bos = summarize(baseline_results)
am, as_, aom, aos = summarize(model_a_results)

print("\n" + "="*60)
print("  FINAL RESULTS")
print("="*60)
print(f"  {'':12} {'Avg pts':>10}  {'Optimal rate':>14}")
print(f"  {'─'*12} {'─'*10}  {'─'*14}")
print(f"  {'Baseline':12} {bm:>7.2f}±{bs:.2f}  {bom:>10.2%}±{bos:.2%}")
print(f"  {'Model A':12} {am:>7.2f}±{as_:.2f}  {aom:>10.2%}±{aos:.2%}")
print(f"  {'Δ':12} {am-bm:>+10.2f}  {aom-bom:>+13.2%}")
print("="*60)

os.makedirs(args.out, exist_ok=True)
out_path = os.path.join(args.out, "eval_model_a_results.json")
with open(out_path, "w") as f:
    json.dump({
        "baseline": {"folds": baseline_results, "summary": {"avg_pts_mean": bm, "avg_pts_std": bs, "optimal_rate_mean": bom, "optimal_rate_std": bos}},
        "model_a":  {"folds": model_a_results,  "summary": {"avg_pts_mean": am, "avg_pts_std": as_, "optimal_rate_mean": aom, "optimal_rate_std": aos}},
    }, f, indent=2)
print(f"Saved → {out_path}")