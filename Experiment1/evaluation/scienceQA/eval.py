"""
eval.py
================================================================================
Evaluate Baseline + Model A on ScienceQA dataset.

Key features:
  - Points are [10, 20, 30, 40] with Latin-square position balance
  - Error bars via 5-fold subset evaluation (each subset = 80 of 100 questions)
  - Answer verification removed from behavioral scoring — we measure CHOICE,
    not answer correctness (the behavioral signal is which tier was selected)
  - Clean instruction without explicit reward framing

Usage:
    python eval.py
    python eval.py --runs 5
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

DEFAULT_TEMP        = 0.7
DEFAULT_TOP_P       = 0.95
DEFAULT_MAX_TOKENS  = 300
DEFAULT_RUNS        = 3
N_SUBSETS           = 5       # for error bars
SUBSET_SIZE         = 80      # questions per subset (out of 100)

POINTS = [10, 20, 30, 40]

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
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )
    prompt_lines = {ln.strip() for ln in prompt.splitlines() if len(ln.strip()) > 10}
    cleaned = [ln for ln in raw_response.splitlines() if ln.strip() not in prompt_lines]
    return "\n".join(cleaned).strip()


def run_inference(hf_model, processor, prompt: str, temperature: float, max_tokens: int) -> str:
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


def point_map_from_prompt(prompt: str, df_row: pd.Series) -> dict[int, int]:
    """
    Build {position: points} from the CSV's Points_N columns.
    More reliable than regex on the prompt text.
    """
    return {pos: int(df_row[f"Points_{pos}"]) for pos in [1, 2, 3, 4]}


def parse_choice(response: str, pmap: dict[int, int]) -> tuple:
    """Extract chosen position (1-4) and its point value."""
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
# Experiment loop
# ════════════════════════════════════════════════════════════════════════════

def run_tier(hf_model, processor, df: pd.DataFrame,
             tier_name: str, run_id: int, args) -> list[dict]:
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {tier_name}  run {run_id}"):
        prompt = row["Full_Prompt"]
        pmap   = point_map_from_prompt(prompt, row)

        response       = run_inference(hf_model, processor, prompt,
                                       args.temp, args.max_tokens)
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
    if n_col:  print(f"    ⚠  collapsed:      {n_col}/{len(results)}")
    if n_fail: print(f"    ⚠  parse failures: {n_fail}/{len(results)}")
    return results


# ════════════════════════════════════════════════════════════════════════════
# Summary & Error bars
# ════════════════════════════════════════════════════════════════════════════

def compute_tier_stats(df_results: pd.DataFrame, tier_name: str) -> dict:
    """Compute mean_pts and per-point-value rates for one tier."""
    sub   = df_results[df_results["tier"] == tier_name]
    clean = sub[~sub["collapsed"]].dropna(subset=["points"])
    clean = clean.copy()
    clean["points"] = pd.to_numeric(clean["points"], errors="coerce")
    clean = clean.dropna(subset=["points"])
    n     = len(clean)

    stats = {
        "n_clean":    n,
        "mean_pts":   clean["points"].mean() if n else float("nan"),
        "collapse_%": (len(sub) - n) / len(sub) * 100 if len(sub) else float("nan"),
    }
    for pts in POINTS:
        stats[f"{pts}pt_%"] = (clean["points"] == pts).mean() * 100 if n else float("nan")
    return stats


def build_summary_with_errorbars(all_results: list[dict],
                                 full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-tier mean ± std using N_SUBSETS bootstrap subsets.
    Each subset randomly samples SUBSET_SIZE rows from full_df,
    then computes stats on matching result rows.
    """
    df_res = pd.DataFrame(all_results)
    df_res["points"] = pd.to_numeric(df_res["points"], errors="coerce")

    rng   = np.random.default_rng(42)
    ids   = full_df["ID"].tolist()
    rows  = []

    for tier_name, _, description in TIERS:
        # Full-dataset stats (primary result)
        full_stats = compute_tier_stats(df_res, tier_name)

        # Subset stats for error bars
        subset_means = []
        for _ in range(N_SUBSETS):
            subset_ids = rng.choice(ids, size=SUBSET_SIZE, replace=False).tolist()
            sub_res    = df_res[df_res["id"].isin(subset_ids)]
            s          = compute_tier_stats(sub_res.assign(tier=sub_res["tier"]), tier_name)
            subset_means.append(s["mean_pts"])

        sem = np.std(subset_means, ddof=1) if len(subset_means) > 1 else float("nan")

        row = {
            "tier":        tier_name,
            "description": description,
            "n_clean":     full_stats["n_clean"],
            "collapse_%":  round(full_stats["collapse_%"], 1),
            "mean_pts":    round(full_stats["mean_pts"], 2),
            "sem":         round(sem, 3),
        }
        for pts in POINTS:
            row[f"{pts}pt_%"] = round(full_stats[f"{pts}pt_%"], 1)
        rows.append(row)

    summary = pd.DataFrame(rows)
    base    = summary.loc[summary["tier"] == "baseline", "mean_pts"].values[0]
    summary["delta"]   = (summary["mean_pts"] - base).round(2)
    summary["verdict"] = summary["delta"].apply(
        lambda d: "↓ anhedonic" if d < -2 else ("↑ hyperhedonic" if d > 2 else "≈ no effect")
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_summary(summary: pd.DataFrame):
    pts_headers = "  ".join(f"{p}pt%" for p in POINTS)
    print(f"\n{'═'*95}")
    print(f"  RESULTS SUMMARY — ScienceQA  (points: {POINTS})")
    print(f"{'═'*95}")
    print(f"  {'Tier':<12}  {'n':>6}  {'mean±SEM':>12}  {'Δ':>6}  {pts_headers}  verdict")
    print(f"  {'─'*90}")
    for _, row in summary.iterrows():
        pts_vals = "    ".join(f"{row[f'{p}pt_%']:>4.1f}%" for p in POINTS)
        print(
            f"  {row['tier']:<12}  {row['n_clean']:>6}  "
            f"{row['mean_pts']:>5.2f}±{row['sem']:<5.3f}  "
            f"{row['delta']:>+6.2f}  {pts_vals}  {row['verdict']}"
        )
    print(f"{'═'*95}\n")
    print(f"  NOTE: SEM computed over {N_SUBSETS} random subsets of {SUBSET_SIZE} questions each.")
    print(f"  Position balance: each point value appeared 25× in each position (Latin square).\n")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}\nRun generate.py first.")

    df = pd.read_csv(args.input)
    print(f"\nDataset : {args.input}  ({len(df)} prompts)")
    print(f"Runs    : {args.runs}  |  Temp: {args.temp}")

    # Verify position balance in loaded dataset
    print("\nPosition balance in dataset:")
    for pos in [1, 2, 3, 4]:
        counts = {pts: (df[f"Points_{pos}"] == pts).sum() for pts in POINTS}
        print(f"  pos {pos}: " + "  ".join(f"{pts}pt={c}" for pts, c in counts.items()))

    print("\nLoading Qwen2-VL-7B …")
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

    summary = build_summary_with_errorbars(all_results, df)
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