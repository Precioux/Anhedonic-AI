"""
eval.py  —  ScienceQA Anhedonia Experiment
==========================================
Loads the pre-generated dataset and runs:
  - baseline  x 5 runs
  - model_A   x 5 runs

Error bars: SEM over 5 random subsets of 80 questions.

Usage:
    python eval.py
    python eval.py --temp 0.3
    python eval.py --input data/scienceqa_with_ground_truth.csv
"""

import os, re, json, argparse, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
INPUT_CSV       = "data/scienceqa_with_ground_truth.csv"
OUTPUT_DIR      = "results_scienceqa"
NEURONS_JSON    = "neurons_A.json"

DEFAULT_TEMP       = 0.7
DEFAULT_TOP_P      = 0.95
DEFAULT_MAX_TOKENS = 300
RUNS               = 5
N_SUBSETS          = 5
SUBSET_SIZE        = 80
SEED               = 42

POINTS = [10, 20, 30, 40]

TIERS = [
    ("baseline", None,         "no ablation — control"),
    ("model_A",  NEURONS_JSON, "layers 18-27  |  ~1,363n  |  LOCKED setting"),
]


# ════════════════════════════════════════════════════════════════════════════
# Model helpers
# ════════════════════════════════════════════════════════════════════════════

def load_neutral_means():
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()


def install_hooks(lm_layers, neurons_json, mean_acts):
    with open(neurons_json) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    handles = []
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons],
                             dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        handles.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        )
    return handles


# ════════════════════════════════════════════════════════════════════════════
# Inference & Parsing
# ════════════════════════════════════════════════════════════════════════════

def extract_response(gen_ids, input_ids, processor, prompt):
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    prompt_lines = {ln.strip() for ln in prompt.splitlines() if len(ln.strip()) > 10}
    cleaned = [ln for ln in raw.splitlines() if ln.strip() not in prompt_lines]
    return "\n".join(cleaned).strip()


def run_inference(hf_model, processor, prompt, temperature, max_tokens):
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen_ids = hf_model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=temperature, do_sample=True, top_p=DEFAULT_TOP_P,
        )
    return extract_response(gen_ids, inputs.input_ids, processor, prompt)


def point_map_from_row(row):
    """Read position->points from CSV columns. No regex on prompt text."""
    return {pos: int(row[f"Points_{pos}"]) for pos in [1, 2, 3, 4]}


def parse_choice(response, pmap):
    patterns = [
        r'(?:choose|answer|pick|select|go\s+with|opt\s+for)\s+(?:question\s+)?([1-4])',
        r'(?:my\s+)?choice(?:\s+is)?\s*[:\-]?\s*([1-4])',
        r'^(?:question\s+)?([1-4])[\.:\)]\s',
        r'^\s*([1-4])\s*$',
        r'\b([1-4])\b',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.MULTILINE)
        if m:
            q = int(m.group(1))
            return q, pmap.get(q)
    return None, None


def is_collapsed(text):
    words = str(text).split()
    if words and max(len(w) for w in words) > 25:
        return True
    tokens = str(text).lower().split()
    if len(tokens) >= 8:
        ngrams = [' '.join(tokens[i:i+4]) for i in range(len(tokens) - 3)]
        for ng in ngrams:
            if ngrams.count(ng) > 3:
                return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ════════════════════════════════════════════════════════════════════════════

def run_tier(hf_model, processor, df, tier_name, run_id, temp, max_tokens):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {tier_name}  run {run_id}/{RUNS}"):
        prompt   = row["Full_Prompt"]
        pmap     = point_map_from_row(row)
        response = run_inference(hf_model, processor, prompt, temp, max_tokens)

        collapsed      = is_collapsed(response)
        choice, points = (None, None) if collapsed else parse_choice(response, pmap)

        results.append({
            "tier":      tier_name,
            "run":       run_id,
            "id":        row["ID"],
            "topic":     row.get("Topic", ""),
            "response":  response,
            "choice":    choice,
            "points":    points,
            "collapsed": collapsed,
        })

    n_col  = sum(1 for r in results if r["collapsed"])
    n_fail = sum(1 for r in results if not r["collapsed"] and r["choice"] is None)
    pts_vals = [r["points"] for r in results if r["points"] is not None]
    mean_pts = sum(pts_vals) / len(pts_vals) if pts_vals else float("nan")
    print(f"    run {run_id}: mean={mean_pts:.2f}  collapsed={n_col}  "
          f"parse_fail={n_fail}  valid={len(pts_vals)}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Summary with error bars
# ════════════════════════════════════════════════════════════════════════════

def compute_stats(df_res, tier_name):
    sub   = df_res[df_res["tier"] == tier_name].copy()
    clean = sub[~sub["collapsed"]].copy()
    clean["points"] = pd.to_numeric(clean["points"], errors="coerce")
    clean = clean.dropna(subset=["points"])
    n = len(clean)
    stats = {
        "n":          n,
        "n_total":    len(sub),
        "mean_pts":   clean["points"].mean() if n else float("nan"),
        "collapse_%": (len(sub) - n) / len(sub) * 100 if len(sub) else float("nan"),
    }
    for pts in POINTS:
        stats[f"{pts}pt_%"] = (clean["points"] == pts).mean() * 100 if n else float("nan")
    return stats


def build_summary(all_results, full_df):
    df_res = pd.DataFrame(all_results)
    df_res["points"] = pd.to_numeric(df_res["points"], errors="coerce")

    rng  = np.random.default_rng(SEED)
    ids  = full_df["ID"].tolist()
    rows = []

    for tier_name, _, description in TIERS:
        full         = compute_stats(df_res, tier_name)
        subset_means = []
        for _ in range(N_SUBSETS):
            sub_ids = rng.choice(ids, size=SUBSET_SIZE, replace=False).tolist()
            s = compute_stats(df_res[df_res["id"].isin(sub_ids)], tier_name)
            if not np.isnan(s["mean_pts"]):
                subset_means.append(s["mean_pts"])
        sem = np.std(subset_means, ddof=1) if len(subset_means) > 1 else float("nan")

        row = {
            "tier":        tier_name,
            "description": description,
            "n_clean":     full["n"],
            "n_total":     full["n_total"],
            "collapse_%":  round(full["collapse_%"], 1),
            "mean_pts":    round(full["mean_pts"], 2),
            "sem":         round(sem, 3),
        }
        for pts in POINTS:
            row[f"{pts}pt_%"] = round(full[f"{pts}pt_%"], 1)
        rows.append(row)

    summary = pd.DataFrame(rows)
    base = summary.loc[summary["tier"] == "baseline", "mean_pts"].values[0]
    summary["delta"]   = (summary["mean_pts"] - base).round(2)
    summary["verdict"] = summary["delta"].apply(
        lambda d: "anhedonic" if d < -1 else ("hyperhedonic" if d > 1 else "no effect")
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_summary(summary):
    pts_h = "   ".join(f"{p}pt%" for p in POINTS)
    w = 104
    print(f"\n{'='*w}")
    print(f"  RESULTS  |  ScienceQA  |  {RUNS} runs/tier  |  "
          f"SEM over {N_SUBSETS}x{SUBSET_SIZE} subsets")
    print(f"{'='*w}")
    print(f"  {'Tier':<12}  {'n':>5}  {'col%':>5}  "
          f"{'mean +/- SEM':>16}  {'delta':>7}  {pts_h}  verdict")
    print(f"  {'-'*100}")
    for _, row in summary.iterrows():
        pts_vals = "    ".join(f"{row[f'{p}pt_%']:>4.1f}%" for p in POINTS)
        print(
            f"  {row['tier']:<12}  {row['n_clean']:>5}  {row['collapse_%']:>4.1f}%  "
            f"{row['mean_pts']:>5.2f} +/- {row['sem']:<7.3f}  "
            f"{row['delta']:>+7.2f}  {pts_vals}  {row['verdict']}"
        )
    print(f"{'='*w}")
    print(f"\n  Latin square: each of {POINTS} appeared 25x in each position.")
    print(f"  Verdict threshold: |delta| > 1 point.\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Dataset not found: {args.input}\n"
            f"Run generate.py first to create it."
        )
    df = pd.read_csv(args.input)

    if not os.path.exists(NEURONS_JSON):
        raise FileNotFoundError(
            f"{NEURONS_JSON} not found. Run extract_neurons.py first."
        )

    print(f"\nDataset  : {args.input}  ({len(df)} prompts)")
    print(f"Runs     : {RUNS} per tier  (baseline + model_A = {RUNS*2} total)")
    print(f"Temp     : {args.temp}  |  Max tokens: {args.max_tokens}\n")

    # Load model once
    print("Loading Qwen2-VL-7B ...")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("  Ready.\n")

    raw_path    = os.path.join(OUTPUT_DIR, "raw_responses.csv")
    sum_path    = os.path.join(OUTPUT_DIR, "summary.csv")
    all_results = []

    for tier_name, json_file, description in TIERS:
        print(f"{'='*60}")
        print(f"  {tier_name}  --  {description}")
        print(f"{'='*60}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, RUNS + 1):
                results = run_tier(
                    hf_model, processor, df,
                    tier_name, run_id, args.temp, args.max_tokens
                )
                all_results.extend(results)
                pd.DataFrame(all_results).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()
        print()

    summary = build_summary(all_results, df)
    summary.to_csv(sum_path, index=False)
    print_summary(summary)
    print(f"  Raw  : {raw_path}")
    print(f"  Summary : {sum_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"ScienceQA anhedonia eval — {RUNS} runs x 2 tiers"
    )
    parser.add_argument("--input",      type=str,   default=INPUT_CSV)
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMP)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    main(args)