"""
evaluate_models.py  —  Test baseline + Models A/B/C on the anhedonia dataset
=============================================================================
Expected layout (run from phase5/):

    phase5/
    ├── data/
    │   └── full_experiment_100_rows.csv   ← input
    ├── results/                           ← created automatically
    │   ├── raw_responses.csv
    │   └── summary.csv
    ├── neurons_A.json
    ├── neurons_B.json
    ├── neurons_C.json
    └── evaluate_models.py

Usage
-----
    python evaluate_models.py                        # 1 run, default settings
    python evaluate_models.py --runs 3               # repeat & average
    python evaluate_models.py --temp 0.3             # more deterministic
    python evaluate_models.py --input data/my.csv    # custom input file

Prerequisites
-------------
    python extract.py    # run once → creates neurons_A/B/C.json
"""

import os, re, json, argparse, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"

INPUT_CSV       = "data/full_experiment_100_rows.csv"   # read from data/
OUTPUT_DIR      = "results"                              # write to results/

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.95
DEFAULT_MAX_TOKENS  = 300
DEFAULT_RUNS        = 1
TOTAL_NEURONS       = 28 * 18944

# ── Ablation tiers ─────────────────────────────────────────────────────────
TIERS = [
    ("baseline", None,             "no ablation — control"),
    ("model_A",  "neurons_A.json", "layers 18–27  |  ~1,363n  |  Δ=−9.81"),
    ("model_B",  "neurons_B.json", "layers 23–27  |    ~609n  |  Δ=−7.84"),
    ("model_C",  "neurons_C.json", "layer  27     |     194n  |  Δ=−6.26"),
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
    return torch.cat(parts, dim=0).mean(dim=0).numpy()   # [28, 18944]


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
# Response cleaning — prompt must never appear in the saved output
# ════════════════════════════════════════════════════════════════════════════

def extract_response(gen_ids, input_ids, processor, prompt: str) -> str:
    """
    Two-layer extraction:
      1. Token level  — decode only the newly generated tokens (gen_ids
                        beyond the length of input_ids). This is exact.
      2. Character level — drop any line that is verbatim in the prompt
                        (safety net for edge-case echo).
    """
    # Layer 1: token-level trim
    new_tokens   = gen_ids[0, input_ids.shape[1]:]
    raw_response = processor.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Layer 2: character-level echo removal
    prompt_lines = {
        ln.strip() for ln in prompt.splitlines()
        if len(ln.strip()) > 10
    }
    cleaned_lines = [
        ln for ln in raw_response.splitlines()
        if ln.strip() not in prompt_lines
    ]
    return "\n".join(cleaned_lines).strip()


# ════════════════════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════════════════════

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
# Inference
# ════════════════════════════════════════════════════════════════════════════

def run_tier(hf_model, processor, df: pd.DataFrame,
             tier_name: str, run_id: int,
             temperature: float, max_tokens: int) -> list[dict]:
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {tier_name}  run {run_id}"):

        prompt = row["Full_Prompt"]
        pmap   = point_map_from_prompt(prompt)

        text   = processor.apply_chat_template(
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

        response       = extract_response(gen_ids, inputs.input_ids, processor, prompt)
        collapsed      = is_collapsed(response)
        choice, points = (None, None) if collapsed else parse_choice(response, pmap)

        results.append({
            "tier":      tier_name,
            "run":       run_id,
            "id":        row["ID"],
            "response":  response,
            "choice":    choice,
            "points":    points,
            "collapsed": collapsed,
        })

    n_col  = sum(1 for r in results if r["collapsed"])
    n_fail = sum(1 for r in results if not r["collapsed"] and r["choice"] is None)
    if n_col:  print(f"    ⚠  collapsed:      {n_col}/{len(results)}")
    if n_fail: print(f"    ⚠  parse failures: {n_fail}/{len(results)}")

    return results


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

def build_summary(all_results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_results)
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    rows = []
    for tier_name, _, _ in TIERS:
        sub     = df[df["tier"] == tier_name]
        clean   = sub[~sub["collapsed"]].dropna(subset=["points"])
        total   = len(sub)
        n_clean = len(clean)
        col_pct = (total - n_clean) / total * 100 if total else float("nan")

        rows.append({
            "tier":        tier_name,
            "n_total":     total,
            "n_clean":     n_clean,
            "collapse_%":  round(col_pct, 1),
            "mean_pts":    round(clean["points"].mean(), 2) if n_clean else float("nan"),
            "1pt_%":       round((clean["points"] == 1).mean()   * 100, 1) if n_clean else float("nan"),
            "10pt_%":      round((clean["points"] == 10).mean()  * 100, 1) if n_clean else float("nan"),
            "50pt_%":      round((clean["points"] == 50).mean()  * 100, 1) if n_clean else float("nan"),
            "100pt_%":     round((clean["points"] == 100).mean() * 100, 1) if n_clean else float("nan"),
            "parse_fail":  int(sub[~sub["collapsed"]]["choice"].isna().sum()),
        })

    summary   = pd.DataFrame(rows)
    base_mean = summary.loc[summary["tier"] == "baseline", "mean_pts"].values[0]
    summary["delta"]   = (summary["mean_pts"] - base_mean).round(2)
    summary["verdict"] = summary["delta"].apply(
        lambda d: "↓ anhedonic"   if d < -2
        else      ("↑ hyperhedonic" if d > 2 else "≈ no effect")
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_summary(summary: pd.DataFrame):
    print(f"\n{'═'*82}")
    print(f"  RESULTS SUMMARY")
    print(f"{'═'*82}")
    print(f"  {'Tier':<12}  {'clean':>6}  {'col%':>5}  {'mean':>6}  {'Δ':>6}  "
          f"{'1pt':>5}  {'10pt':>5}  {'50pt':>5}  {'100pt':>6}  verdict")
    print(f"  {'─'*78}")
    for _, row in summary.iterrows():
        print(
            f"  {row['tier']:<12}  {row['n_clean']:>6}  {row['collapse_%']:>4.1f}%  "
            f"{row['mean_pts']:>6.2f}  {row['delta']:>+6.2f}  "
            f"{row['1pt_%']:>4.1f}%  {row['10pt_%']:>4.1f}%  "
            f"{row['50pt_%']:>4.1f}%  {row['100pt_%']:>5.1f}%  {row['verdict']}"
        )
    print(f"{'═'*82}\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    # ── Create output folder ───────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Validate neuron JSONs before loading the heavy model ───────────────
    for _, json_file, _ in TIERS:
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(
                f"{json_file} not found. Run `python extract.py` first."
            )

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"\nDataset : {args.input}  ({len(df)} prompts)")
    print(f"Output  : {OUTPUT_DIR}/\n")

    # ── Load model once ────────────────────────────────────────────────────
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

    # ── Run each tier ──────────────────────────────────────────────────────
    for tier_name, json_file, description in TIERS:
        print(f"{'='*60}")
        print(f"  {tier_name}  —  {description}")
        print(f"{'='*60}")

        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs + 1):
                results = run_tier(
                    hf_model, processor, df,
                    tier_name, run_id, args.temp, args.max_tokens
                )
                all_results.extend(results)
                # Save incrementally after every run so nothing is lost
                pd.DataFrame(all_results).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()   # always restore model to clean state between tiers

    # ── Final summary ──────────────────────────────────────────────────────
    summary = build_summary(all_results)
    summary.to_csv(sum_path, index=False)
    print_summary(summary)

    print(f"  Saved: {raw_path}")
    print(f"  Saved: {sum_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline + Models A/B/C on the anhedonia dataset"
    )
    parser.add_argument("--input",      type=str,   default=INPUT_CSV,
                        help=f"Path to input CSV (default: {INPUT_CSV})")
    parser.add_argument("--runs",       type=int,   default=DEFAULT_RUNS,
                        help="Repetitions per model (default: 1)")
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMPERATURE,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS,
                        help="Max new tokens per response (default: 300)")
    args = parser.parse_args()
    main(args)