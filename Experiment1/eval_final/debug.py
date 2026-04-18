"""
debug_parser.py — re-runs inference with smart parser + side-by-side comparison
Saves full row data to results/debug_results.json
Run: python debug_parser.py
"""
import json, re, os, torch
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from collections import defaultdict

MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
NEURONS_JSON    = "neurons_A.json"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"

# ── Parsers ──────────────────────────────────────────────────────────────────
def parse_v1(response):
    """Original: first digit found."""
    m = re.search(r'\b([1-4])\b', response.strip())
    return int(m.group(1)) if m else None

def parse_v2(response):
    """Smart: detect multi-answer, trust leading digit."""
    r = response.strip()
    # Multi-answer: 3+ lines starting with digit+punctuation
    if len(re.findall(r'(?m)^[ \t]*([1-4])[\.:\)]\s*\S', r)) >= 3:
        return "MULTI"
    # Multi-answer: bare digits repeated e.g. "1\n...\n2\n...\n3\n..."
    if len(re.findall(r'(?m)^([1-4])\s*$', r)) >= 3:
        return "MULTI"
    # Leading digit (clean single choice)
    m = re.match(r'^([1-4])[\s\n\.\:]', r)
    if m: return int(m.group(1))
    # Fallback
    m = re.search(r'\b([1-4])\b', r)
    return int(m.group(1)) if m else None

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
proc   = AutoProcessor.from_pretrained(MODEL_PATH)
layers = model.model.language_model.layers

parts = []
for domain in ["geo", "math"]:
    d = torch.load(os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt"), map_location="cpu")
    parts.append(torch.stack(list(d.values())).float())
mean_acts = torch.cat(parts, dim=0).mean(dim=0).numpy()

with open(NEURONS_JSON) as f:
    neuron_map = {int(k): v for k, v in json.load(f).items()}

hooks = []
def install_hooks():
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(_, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        hooks.append(layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    print(f"✓ Hooks ON ({sum(len(v) for v in neuron_map.values()):,} neurons)")

def remove_hooks():
    for h in hooks: h.remove()
    hooks.clear()
    print("✓ Hooks OFF")

def generate(prompt):
    text   = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return proc.batch_decode([gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]

# ── Folds ────────────────────────────────────────────────────────────────────
def make_folds(rows, k=4, seed=42):
    import random; rng = random.Random(seed)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row["permutation"])].append(row)
    folds = [[] for _ in range(k)]
    for group in groups.values():
        rng.shuffle(group)
        for i, row in enumerate(group):
            folds[i % k].append(row)
    return folds

# ── Run one pass ─────────────────────────────────────────────────────────────
def run(folds, label):
    all_rows = []
    fold_stats = []
    for fi, fold in enumerate(folds):
        print(f"\n  [{label}] Fold {fi+1}/4")
        v1_pts, v2_pts, v1_opt, v2_opt = [], [], [], []
        for i, row in enumerate(fold):
            resp = generate(row["prompt"])
            c1   = parse_v1(resp)
            c2   = parse_v2(resp)

            pts1 = row[f"q{c1}_points"] if c1 else 0
            pts2 = row[f"q{c2}_points"] if isinstance(c2, int) else 0
            multi = (c2 == "MULTI")
            mismatch = (not multi) and (c1 != c2)

            flag = "MULTI" if multi else ("MISMATCH" if mismatch else "ok")
            print(f"    [{i+1:02d}/24] {flag:8s} | v1={c1}({pts1}pt) v2={c2}({pts2}pt) | {resp[:55].strip()!r}")

            v1_pts.append(pts1); v1_opt.append(pts1 == 40)
            if not multi:
                v2_pts.append(pts2); v2_opt.append(pts2 == 40)

            all_rows.append({"fold": fi+1, "label": label, "response": resp,
                             "v1_choice": c1, "v1_pts": pts1,
                             "v2_choice": c2 if isinstance(c2, int) else None,
                             "v2_pts": pts2, "multi_answer": multi,
                             "parser_mismatch": mismatch,
                             **{f"q{p}_points": row[f"q{p}_points"] for p in range(1,5)}})

        fold_stats.append({
            "fold": fi+1,
            "v1_avg": np.mean(v1_pts), "v1_opt": np.mean(v1_opt),
            "v2_avg": np.mean(v2_pts) if v2_pts else float("nan"),
            "v2_opt": np.mean(v2_opt) if v2_opt else float("nan"),
            "n_multi": sum(1 for r in all_rows[-len(fold):] if r["multi_answer"]),
        })
        print(f"    v1: avg={np.mean(v1_pts):.2f} opt={np.mean(v1_opt):.2%} | "
              f"v2: avg={np.mean(v2_pts) if v2_pts else 'nan':.2f} opt={np.mean(v2_opt) if v2_opt else 'nan':.2%} | "
              f"multi={fold_stats[-1]['n_multi']}")
    return fold_stats, all_rows

# ── Main ─────────────────────────────────────────────────────────────────────
with open("data/asdiv_eval_dataset.json") as f:
    rows = json.load(f)
folds = make_folds(rows)

print("\n" + "="*60 + "\n  BASELINE\n" + "="*60)
base_stats, base_rows = run(folds, "BASELINE")

print("\n" + "="*60 + "\n  MODEL A\n" + "="*60)
install_hooks()
modA_stats, modA_rows = run(folds, "MODEL A")
remove_hooks()

# ── Summary ───────────────────────────────────────────────────────────────────
def summarize(stats, parser):
    k = f"{parser}_avg"
    o = f"{parser}_opt"
    avgs = [s[k] for s in stats if not np.isnan(s[k])]
    opts = [s[o] for s in stats if not np.isnan(s[o])]
    return np.mean(avgs), np.std(avgs), np.mean(opts), np.std(opts)

print("\n" + "="*68)
print("  FINAL COMPARISON — v1 (original) vs v2 (smart, multi excluded)")
print("="*68)
print(f"  {'':18} {'Avg pts':>12}   {'Optimal rate':>14}")
print(f"  {'─'*18} {'─'*12}   {'─'*14}")
for label, stats in [("Baseline", base_stats), ("Model A", modA_stats)]:
    bm1,bs1,bo1,bos1 = summarize(stats, "v1")
    bm2,bs2,bo2,bos2 = summarize(stats, "v2")
    print(f"  {label+' (v1)':18} {bm1:>6.2f} ± {bs1:.2f}   {bo1:>8.2%} ± {bos1:.2%}")
    print(f"  {label+' (v2)':18} {bm2:>6.2f} ± {bs2:.2f}   {bo2:>8.2%} ± {bos2:.2%}")
    print()

bm1,_,_,_ = summarize(base_stats,"v1"); am1,_,_,_ = summarize(modA_stats,"v1")
bm2,_,_,_ = summarize(base_stats,"v2"); am2,_,_,_ = summarize(modA_stats,"v2")
print(f"  Δ (v1): {am1-bm1:+.2f} pts")
print(f"  Δ (v2): {am2-bm2:+.2f} pts")
print("="*68)

os.makedirs("results", exist_ok=True)
with open("results/debug_results.json", "w") as f:
    json.dump({"baseline": base_rows, "model_a": modA_rows,
               "base_stats": base_stats, "modA_stats": modA_stats}, f, indent=2)
print("Saved → results/debug_results.json")