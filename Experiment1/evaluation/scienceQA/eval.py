"""
eval.py
================================================================================
Evaluate Baseline + Model A on ScienceQA dataset.
Follows the exact structure of run_all_v2.py

Usage
-----
    python eval.py                   # 3 runs each for Baseline and Model A
    python eval.py --runs 5
    python eval.py --temp 0.3
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
OUTPUT_DIR      = "results"

DEFAULT_TEMP        = 0.7
DEFAULT_TOP_P       = 0.95
DEFAULT_MAX_TOKENS  = 300
DEFAULT_RUNS        = 3

# ── Ablation tiers ─────────────────────────────────────────────────────────
TIERS = [
    ("baseline", None,             "no ablation — control"),
    ("model_A",  "neurons_A.json", "layers 18–27  |  ~1,363n  |  LOCKED setting"),
]

# ════════════════════════════════════════════════════════════════════════════
# Setup helpers
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

# ════════════════════════════════════════════════════════════════════════════
# Inference & Parsing
# ════════════════════════════════════════════════════════════════════════════

def extract_response(gen_ids, input_ids, processor, prompt: str) -> str:
    new_tokens   = gen_ids[0, input_ids.shape[1]:]
    raw_response = processor.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    prompt_lines = {ln.strip() for ln in prompt.splitlines() if len(ln.strip()) > 10}
    cleaned_lines = [ln for ln in raw_response.splitlines() if ln.strip() not in prompt_lines]
    return "\n".join(cleaned_lines).strip()

def run_inference(hf_model, processor, prompt: str, temperature: float, max_tokens: int) -> str:
    text = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        gen_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=DEFAULT_TOP_P,
        )

    return extract_response(gen_ids, inputs.input_ids, processor, prompt)

def point_map_from_prompt(prompt: str) -> dict[int, int]:
    return {
        int(m.group(1)): int(m.group(2))
        for m in re.finditer(r'(\d+)\.\s+.+?\((\d+) point', prompt)
    }

def parse_choice(response: str, pmap: dict[int, int]) -> tuple:
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
    if words and max(len(w) for w in words) > 25: return True
    tokens = str(text).lower().split()
    if len(tokens) >= 8:
        ngrams = [' '.join(tokens[i:i+4]) for i in range(len(tokens) - 3)]
        for ng in ngrams:
            if ngrams.count(ng) > 3: return True
    return False

def verify_answer(response: str, choice: int, row: pd.Series) -> bool:
    if choice is None: return False
    correct_ans_col = f"Correct_Answer_{choice}"
    if correct_ans_col not in row: return False
    correct_ans = str(row[correct_ans_col]).strip().lower()
    return correct_ans in response.lower()

# ════════════════════════════════════════════════════════════════════════════
# Experiment Loop
# ════════════════════════════════════════════════════════════════════════════

def run_tier(hf_model, processor, df: pd.DataFrame, tier_name: str, run_id: int, args) -> list[dict]:
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {tier_name} run {run_id}"):
        prompt = row["Full_Prompt"]
        pmap   = point_map_from_prompt(prompt)

        response       = run_inference(hf_model, processor, prompt, args.temp, args.max_tokens)
        collapsed      = is_collapsed(response)
        choice, points = (None, None) if collapsed else parse_choice(response, pmap)
        
        is_correct     = verify_answer(response, choice, row) if not collapsed else False
        earned_points  = points if is_correct else 0

        results.append({
            "tier":          tier_name,
            "run":           run_id,
            "id":            row["ID"],
            "response":      response,
            "choice":        choice,
            "attempted_pts": points,
            "is_correct":    is_correct,
            "earned_pts":    earned_points,
            "collapsed":     collapsed,
        })
        
    return results

def build_summary(all_results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_results)
    df["attempted_pts"] = pd.to_numeric(df["attempted_pts"], errors="coerce")
    df["earned_pts"]    = pd.to_numeric(df["earned_pts"], errors="coerce")

    rows = []
    for tier_name, _, _ in TIERS:
        sub     = df[df["tier"] == tier_name]
        if len(sub) == 0: continue
            
        clean   = sub[~sub["collapsed"]].dropna(subset=["attempted_pts"])
        total   = len(sub)
        n_clean = len(clean)
        col_pct = (total - n_clean) / total * 100 if total else float("nan")
        acc     = (clean["is_correct"].sum() / n_clean * 100) if n_clean else float("nan")

        rows.append({
            "tier":        tier_name,
            "n_total":     total,
            "n_clean":     n_clean,
            "collapse_%":  round(col_pct, 1),
            "acc_%":       round(acc, 1),
            "mean_att":    round(clean["attempted_pts"].mean(), 2) if n_clean else float("nan"),
            "mean_earn":   round(clean["earned_pts"].mean(), 2) if n_clean else float("nan"),
            "1pt_att_%":   round((clean["attempted_pts"] == 1).mean()   * 100, 1) if n_clean else float("nan"),
            "10pt_att_%":  round((clean["attempted_pts"] == 10).mean()  * 100, 1) if n_clean else float("nan"),
            "50pt_att_%":  round((clean["attempted_pts"] == 50).mean()  * 100, 1) if n_clean else float("nan"),
            "100pt_att_%": round((clean["attempted_pts"] == 100).mean() * 100, 1) if n_clean else float("nan"),
        })

    summary = pd.DataFrame(rows)
    if "baseline" in summary["tier"].values:
        base_mean = summary.loc[summary["tier"] == "baseline", "mean_att"].values[0]
        summary["delta_att"] = (summary["mean_att"] - base_mean).round(2)
    else:
        summary["delta_att"] = float("nan")
        
    return summary

def print_summary(summary: pd.DataFrame):
    print(f"\n{'═'*92}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*92}")
    print(f"  {'Tier':<12}  {'clean':>6}  {'acc%':>6}  {'att_pts':>7}  {'ern_pts':>7}  {'Δ_att':>6}  "
          f"{'1pt':>5}  {'10pt':>5}  {'50pt':>5}  {'100pt':>6}")
    print(f"  {'─'*88}")
    for _, row in summary.iterrows():
        print(
            f"  {row['tier']:<12}  {row['n_clean']:>6}  {row['acc_%']:>5.1f}%  "
            f"{row['mean_att']:>7.2f}  {row['mean_earn']:>7.2f}  {row['delta_att']:>+6.2f}  "
            f"{row['1pt_att_%']:>4.1f}%  {row['10pt_att_%']:>4.1f}%  "
            f"{row['50pt_att_%']:>4.1f}%  {row['100pt_att_%']:>5.1f}%"
        )
    print(f"{'═'*92}\n")

# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"\nDataset : {args.input} ({len(df)} prompts)")
    print(f"Runs    : {args.runs}\n")

    print("Loading Qwen2-VL-7B …")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("  Ready.\n")

    all_results = []
    raw_path    = os.path.join(OUTPUT_DIR, "raw_responses.csv")
    sum_path    = os.path.join(OUTPUT_DIR, "summary.csv")

    for tier_name, json_file, description in TIERS:
        print(f"{'='*60}")
        print(f"  {tier_name}  —  {description}")
        print(f"{'='*60}")

        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs + 1):
                results = run_tier(hf_model, processor, df, tier_name, run_id, args)
                all_results.extend(results)
                pd.DataFrame(all_results).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()

    summary = build_summary(all_results)
    summary.to_csv(sum_path, index=False)
    print_summary(summary)

    print(f"  Saved: {raw_path}")
    print(f"  Saved: {sum_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      type=str,   default=INPUT_CSV)
    parser.add_argument("--runs",       type=int,   default=DEFAULT_RUNS)
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMP)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    main(args)