"""
run_all_v2.py  —  Behavioral experiment + Functional capacity test (v2 dataset)
================================================================================
Runs both experiments back-to-back on the redesigned all-addition dataset.

Experiment 1 — Behavioral (free choice)
    Input  : data/full_experiment_100_rows_v2.csv
    Output : results_v2/behavioral/raw_responses.csv
             results_v2/behavioral/summary.csv

Experiment 2 — Functional capacity (forced answer)
    Input  : data/functional_capacity_100q_v2.csv
    Output : results_v2/functional_capacity/raw_responses.csv
             results_v2/functional_capacity/summary.csv

Usage
-----
    python run_all_v2.py                   # default: 5 runs behavioral, 3 runs capacity
    python run_all_v2.py --runs_beh 5 --runs_cap 3
    python run_all_v2.py --temp 0.7
    python run_all_v2.py --skip_beh        # only run capacity test
    python run_all_v2.py --skip_cap        # only run behavioral test

Prerequisites
-------------
    python extract.py    # run once → creates neurons_A.json
"""

import os, re, json, argparse, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"

BEH_INPUT       = "data/full_experiment_100_rows_v3.csv"
CAP_INPUT       = "data/functional_capacity_100q_v3.csv"
BEH_OUTPUT_DIR  = "results_v3/behavioral"
CAP_OUTPUT_DIR  = "results_v3/functional_capacity"

DEFAULT_TEMP        = 0.7
DEFAULT_TOP_P       = 0.95
DEFAULT_MAX_TOKENS  = 300
DEFAULT_RUNS_BEH    = 5
DEFAULT_RUNS_CAP    = 3
TOTAL_NEURONS       = 28 * 18944

TIERS = [
    ("baseline", None,             "no ablation — control"),
    ("model_A",  "neurons_A.json", "layers 18–27  |  ~1,363n  |  LOCKED setting"),
]


# ════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ════════════════════════════════════════════════════════════════════════════

def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()


def install_hooks(lm_layers, neurons_json: str, mean_acts: np.ndarray) -> list:
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


def extract_response(gen_ids, input_ids, processor, prompt: str) -> str:
    new_tokens = gen_ids[0, input_ids.shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True,
                           clean_up_tokenization_spaces=False)
    prompt_lines = {ln.strip() for ln in prompt.splitlines() if len(ln.strip()) > 10}
    cleaned = [ln for ln in raw.splitlines() if ln.strip() not in prompt_lines]
    return "\n".join(cleaned).strip()


def run_inference(hf_model, processor, prompt: str,
                  temperature: float, max_tokens: int) -> str:
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


# ════════════════════════════════════════════════════════════════════════════
# Experiment 1 — Behavioral (free choice)
# ════════════════════════════════════════════════════════════════════════════

def point_map_from_prompt(prompt: str) -> dict:
    return {
        int(m.group(1)): int(m.group(2))
        for m in re.finditer(r'(\d+)\.\s+.+?\((\d+) point', prompt)
    }


def parse_choice(response: str, pmap: dict) -> tuple:
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


def is_collapsed(text: str) -> bool:
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


def run_behavioral_tier(hf_model, processor, df, tier_name, run_id,
                        temperature, max_tokens) -> list:
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [BEH] {tier_name}  run {run_id}"):
        prompt = row["Full_Prompt"]
        pmap   = point_map_from_prompt(prompt)
        response = run_inference(hf_model, processor, prompt, temperature, max_tokens)
        collapsed = is_collapsed(response)
        choice, points = (None, None) if collapsed else parse_choice(response, pmap)
        results.append({
            "tier": tier_name, "run": run_id, "id": row["ID"],
            "response": response, "choice": choice,
            "points": points, "collapsed": collapsed,
        })

    n_col  = sum(1 for r in results if r["collapsed"])
    n_fail = sum(1 for r in results if not r["collapsed"] and r["choice"] is None)
    if n_col:  print(f"    ⚠  collapsed:      {n_col}/{len(results)}")
    if n_fail: print(f"    ⚠  parse failures: {n_fail}/{len(results)}")
    return results


def build_behavioral_summary(all_results: list) -> pd.DataFrame:
    df = pd.DataFrame(all_results)
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    rows = []
    for tier_name, _, _ in TIERS:
        sub   = df[df["tier"] == tier_name]
        clean = sub[~sub["collapsed"]].dropna(subset=["points"])
        total, n_clean = len(sub), len(clean)
        col_pct = (total - n_clean) / total * 100 if total else float("nan")
        rows.append({
            "tier":       tier_name,
            "n_total":    total,
            "n_clean":    n_clean,
            "collapse_%": round(col_pct, 1),
            "mean_pts":   round(clean["points"].mean(), 2) if n_clean else float("nan"),
            "1pt_%":      round((clean["points"]==1).mean()*100,  1) if n_clean else float("nan"),
            "10pt_%":     round((clean["points"]==10).mean()*100, 1) if n_clean else float("nan"),
            "50pt_%":     round((clean["points"]==50).mean()*100, 1) if n_clean else float("nan"),
            "100pt_%":    round((clean["points"]==100).mean()*100,1) if n_clean else float("nan"),
            "parse_fail": int(sub[~sub["collapsed"]]["choice"].isna().sum()),
        })
    summary = pd.DataFrame(rows)
    base_mean = summary.loc[summary["tier"]=="baseline", "mean_pts"].values[0]
    summary["delta"]   = (summary["mean_pts"] - base_mean).round(2)
    summary["verdict"] = summary["delta"].apply(
        lambda d: "↓ anhedonic" if d < -2 else ("↑ hyperhedonic" if d > 2 else "≈ no effect")
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_behavioral_summary(summary: pd.DataFrame):
    print(f"\n{'═'*82}")
    print(f"  BEHAVIORAL RESULTS (free choice)")
    print(f"{'═'*82}")
    print(f"  {'Tier':<12}  {'clean':>6}  {'col%':>5}  {'mean':>6}  {'Δ':>6}  "
          f"{'1pt':>5}  {'10pt':>5}  {'50pt':>5}  {'100pt':>6}  verdict")
    print(f"  {'─'*78}")
    for _, row in summary.iterrows():
        print(f"  {row['tier']:<12}  {row['n_clean']:>6}  {row['collapse_%']:>4.1f}%  "
              f"{row['mean_pts']:>6.2f}  {row['delta']:>+6.2f}  "
              f"{row['1pt_%']:>4.1f}%  {row['10pt_%']:>4.1f}%  "
              f"{row['50pt_%']:>4.1f}%  {row['100pt_%']:>5.1f}%  {row['verdict']}")
    print(f"{'═'*82}\n")


def run_behavioral_experiment(hf_model, processor, lm_layers, mean_acts, args):
    os.makedirs(BEH_OUTPUT_DIR, exist_ok=True)
    df       = pd.read_csv(args.beh_input)
    raw_path = os.path.join(BEH_OUTPUT_DIR, "raw_responses.csv")
    sum_path = os.path.join(BEH_OUTPUT_DIR, "summary.csv")

    print(f"\n{'█'*60}")
    print(f"  EXPERIMENT 1 — BEHAVIORAL (FREE CHOICE)")
    print(f"  Input : {args.beh_input}  ({len(df)} prompts)")
    print(f"  Runs  : {args.runs_beh}  |  Temp: {args.temp}")
    print(f"{'█'*60}\n")

    all_results = []
    for tier_name, json_file, description in TIERS:
        print(f"{'='*60}")
        print(f"  {tier_name}  —  {description}")
        print(f"{'='*60}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs_beh + 1):
                results = run_behavioral_tier(
                    hf_model, processor, df,
                    tier_name, run_id, args.temp, DEFAULT_MAX_TOKENS
                )
                all_results.extend(results)
                pd.DataFrame(all_results).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()

    summary = build_behavioral_summary(all_results)
    summary.to_csv(sum_path, index=False)
    print_behavioral_summary(summary)
    print(f"  Saved: {raw_path}")
    print(f"  Saved: {sum_path}\n")


# ════════════════════════════════════════════════════════════════════════════
# Experiment 2 — Functional capacity (forced answer)
# ════════════════════════════════════════════════════════════════════════════

def parse_answer(response: str):
    """Last integer in response = model's final answer."""
    all_ints = re.findall(r'\b(\d+)\b', response)
    return int(all_ints[-1]) if all_ints else None


def run_capacity_tier(hf_model, processor, df, tier_name, run_id,
                      temperature) -> list:
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  [CAP] {tier_name}  run {run_id}"):
        prompt         = row["Forced_Prompt"]
        correct_answer = int(row["Correct_Answer"])
        response       = run_inference(hf_model, processor, prompt,
                                       temperature, max_tokens=50)
        parsed  = parse_answer(response)
        correct = (parsed == correct_answer) if parsed is not None else False
        results.append({
            "tier": tier_name, "run": run_id, "id": row["ID"],
            "expression": row["Expression"], "correct_answer": correct_answer,
            "response": response, "parsed_answer": parsed,
            "correct": correct, "parse_failed": parsed is None,
        })

    n_correct = sum(1 for r in results if r["correct"])
    n_failed  = sum(1 for r in results if r["parse_failed"])
    acc = n_correct / len(results) * 100
    print(f"    ✓ Accuracy: {n_correct}/{len(results)} = {acc:.1f}%", end="")
    if n_failed:
        print(f"   ⚠  parse failures: {n_failed}", end="")
    print()
    return results


def build_capacity_summary(all_results: list) -> pd.DataFrame:
    df = pd.DataFrame(all_results)
    rows = []
    for tier_name, _, description in TIERS:
        sub       = df[df["tier"] == tier_name]
        n_total   = len(sub)
        n_correct = sub["correct"].sum()
        n_failed  = sub["parse_failed"].sum()
        accuracy  = n_correct / n_total * 100 if n_total else float("nan")
        rows.append({
            "tier": tier_name, "description": description,
            "n_total": n_total, "n_correct": int(n_correct),
            "n_parse_failed": int(n_failed),
            "accuracy_%": round(accuracy, 1),
        })
    summary = pd.DataFrame(rows)
    base_acc = summary.loc[summary["tier"]=="baseline", "accuracy_%"].values[0]
    summary["delta_acc"] = (summary["accuracy_%"] - base_acc).round(1)
    summary["verdict"] = summary["delta_acc"].apply(
        lambda d: "✓ no capacity impairment"    if abs(d) <= 5
        else       "⚠ borderline"               if abs(d) <= 10
        else       "✗ capacity impaired"
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_capacity_summary(summary: pd.DataFrame, runs: int):
    print(f"\n{'═'*72}")
    print(f"  FUNCTIONAL CAPACITY TEST — RESULTS SUMMARY")
    print(f"  ({runs} run{'s' if runs > 1 else ''} × 100 questions per tier)")
    print(f"{'═'*72}")
    print(f"  {'Tier':<12}  {'n':>6}  {'correct':>8}  {'accuracy':>9}  {'Δ acc':>7}  verdict")
    print(f"  {'─'*68}")
    for _, row in summary.iterrows():
        print(f"  {row['tier']:<12}  {row['n_total']:>6}  "
              f"{row['n_correct']:>8}  {row['accuracy_%']:>8.1f}%  "
              f"{row['delta_acc']:>+6.1f}%  {row['verdict']}")
    print(f"{'═'*72}")
    print(f"\n  KEY: if Δ ≤ 5% → capacity intact, avoidance is purely motivational.\n")


def run_capacity_experiment(hf_model, processor, lm_layers, mean_acts, args):
    os.makedirs(CAP_OUTPUT_DIR, exist_ok=True)
    df       = pd.read_csv(args.cap_input)
    raw_path = os.path.join(CAP_OUTPUT_DIR, "raw_responses.csv")
    sum_path = os.path.join(CAP_OUTPUT_DIR, "summary.csv")

    print(f"\n{'█'*60}")
    print(f"  EXPERIMENT 2 — FUNCTIONAL CAPACITY (FORCED ANSWER)")
    print(f"  Input : {args.cap_input}  ({len(df)} questions)")
    print(f"  Runs  : {args.runs_cap}  |  Temp: {args.temp}")
    print(f"{'█'*60}\n")

    all_results = []
    for tier_name, json_file, description in TIERS:
        print(f"{'='*60}")
        print(f"  {tier_name}  —  {description}")
        print(f"{'='*60}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs_cap + 1):
                results = run_capacity_tier(
                    hf_model, processor, df,
                    tier_name, run_id, args.temp
                )
                all_results.extend(results)
                pd.DataFrame(all_results).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()

    summary = build_capacity_summary(all_results)
    summary.to_csv(sum_path, index=False)
    print_capacity_summary(summary, args.runs_cap)
    print(f"  Saved: {raw_path}")
    print(f"  Saved: {sum_path}\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    # Validate inputs up front
    for _, json_file, _ in TIERS:
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(f"{json_file} not found. Run `python extract.py` first.")
    if not args.skip_beh and not os.path.exists(args.beh_input):
        raise FileNotFoundError(f"Behavioral input not found: {args.beh_input}")
    if not args.skip_cap and not os.path.exists(args.cap_input):
        raise FileNotFoundError(f"Capacity input not found: {args.cap_input}")

    # Load model once — shared across both experiments
    print("\nLoading Qwen2-VL-7B …")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("  Ready.\n")

    if not args.skip_beh:
        run_behavioral_experiment(hf_model, processor, lm_layers, mean_acts, args)

    if not args.skip_cap:
        run_capacity_experiment(hf_model, processor, lm_layers, mean_acts, args)

    print("All done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run behavioral + functional capacity experiments (v2 dataset)"
    )
    parser.add_argument("--beh_input",  type=str,   default=BEH_INPUT)
    parser.add_argument("--cap_input",  type=str,   default=CAP_INPUT)
    parser.add_argument("--runs_beh",   type=int,   default=DEFAULT_RUNS_BEH,
                        help="Behavioral runs per tier (default: 5)")
    parser.add_argument("--runs_cap",   type=int,   default=DEFAULT_RUNS_CAP,
                        help="Capacity runs per tier (default: 3)")
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMP,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--skip_beh",   action="store_true",
                        help="Skip behavioral experiment, run capacity only")
    parser.add_argument("--skip_cap",   action="store_true",
                        help="Skip capacity test, run behavioral only")
    args = parser.parse_args()
    main(args)
