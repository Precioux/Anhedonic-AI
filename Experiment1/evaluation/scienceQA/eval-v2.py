"""
eval.py  —  ScienceQA Anhedonia Experiment
==========================================
Loads the pre-generated dataset and runs:
  - baseline   x 5 runs  (original prompted points)
  - model_A    x 5 runs  (neuron-ablated, original prompted points)
  - [NEW] control   x 5 runs  (no point values shown — pure choice baseline)
  - [NEW] shuffled  x 5 runs  (shuffled point labels — decouples difficulty
                                from reward-seeking; averages over 3 shuffle
                                variants per prompt)

Error bars: SEM over 5 random subsets of 80 questions.

Verdict logic:
  - baseline vs control  → does point framing shift choice at all?
  - model_A  vs baseline → does ablation reduce reward-seeking?
  - shuffled vs baseline → does choice track point label vs. difficulty?

[NEW] Difficulty-vs-choice report: for each tier, shows whether the model
      chose the question whose intrinsic difficulty matched the point label
      (congruent) or not (incongruent). High congruence in baseline but not
      shuffled = model responds to difficulty, not labels.

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
INPUT_CSV       = "data/scienceqa_with_ground_truth-v2.csv"
OUTPUT_DIR      = "results_scienceqa"
NEURONS_JSON    = "neurons_A.json"

DEFAULT_TEMP       = 0.7
DEFAULT_TOP_P      = 0.95
DEFAULT_MAX_TOKENS = 300
RUNS               = 5
N_SUBSETS          = 5
SUBSET_SIZE        = 80
SEED               = 42
N_SHUFFLES         = 3   # must match generate.py

POINTS = [10, 20, 30, 40]

# ── Tier definitions ────────────────────────────────────────────────────────
# Each entry: (name, neurons_json_or_None, prompt_mode, description)
# prompt_mode: "original" | "no_points" | "shuffled"
TIERS = [
    ("baseline", None,         "original",  "no ablation — control"),
    ("model_A",  NEURONS_JSON, "original",  "layers 18-27  |  ~1,363n  |  LOCKED setting"),
    ("control",  None,         "no_points", "[NEW] no point values shown"),
    ("shuffled", None,         "shuffled",  "[NEW] shuffled point labels — 3 variants avg"),
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


def point_map_from_row(row, prompt_mode, shuffle_idx=None):
    """
    Return {position: points_label} for the given prompt mode.
      original  — Points_{pos} columns (original Latin-square assignment)
      no_points — same positions but no points shown; we still record the
                  original assignment so we can compute congruence later
      shuffled  — Shuffled_Points_{shuffle_idx}_Pos_{pos} columns
    """
    if prompt_mode in ("original", "no_points"):
        return {pos: int(row[f"Points_{pos}"]) for pos in [1, 2, 3, 4]}
    else:  # shuffled
        return {pos: int(row[f"Shuffled_Points_{shuffle_idx}_Pos_{pos}"])
                for pos in [1, 2, 3, 4]}


def difficulty_map_from_row(row):
    """Return {position: intrinsic_difficulty} — fixed regardless of prompt mode."""
    return {pos: int(row[f"Difficulty_{pos}"]) for pos in [1, 2, 3, 4]}


def get_prompt(row, prompt_mode, shuffle_idx=None):
    if prompt_mode == "original":
        return row["Full_Prompt"]
    elif prompt_mode == "no_points":
        return row["No_Points_Prompt"]
    else:  # shuffled
        return row[f"Shuffled_Prompt_{shuffle_idx}"]


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

def run_tier(hf_model, processor, df, tier_name, prompt_mode, run_id, temp, max_tokens):
    """
    For "shuffled" mode we run all N_SHUFFLES variants per question and
    average the results so each question still contributes one effective
    data point (avoids inflating n).
    """
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {tier_name}  run {run_id}/{RUNS}"):

        difficulty_map = difficulty_map_from_row(row)

        if prompt_mode == "shuffled":
            # Run all shuffle variants, pick the majority choice
            shuffle_choices, shuffle_pts = [], []
            for s in range(1, N_SHUFFLES + 1):
                prompt  = get_prompt(row, prompt_mode, shuffle_idx=s)
                pmap    = point_map_from_row(row, prompt_mode, shuffle_idx=s)
                response = run_inference(hf_model, processor, prompt, temp, max_tokens)
                collapsed = is_collapsed(response)
                choice, points = (None, None) if collapsed else parse_choice(response, pmap)
                if choice is not None:
                    shuffle_choices.append(choice)
                    shuffle_pts.append(points)

            if shuffle_choices:
                # Use median points as the representative value for this row
                chosen_pts = float(np.median([p for p in shuffle_pts if p is not None]))
                chosen_pos = shuffle_choices[0]   # positional choice from first variant
                collapsed  = False
            else:
                chosen_pts, chosen_pos, collapsed = None, None, True

            # Congruence: does the point label match intrinsic difficulty?
            # For shuffled, congruence = label matches difficulty at chosen pos
            if chosen_pos is not None and not collapsed:
                label_at_pos = row.get(f"Shuffled_Points_1_Pos_{chosen_pos}", None)
                diff_at_pos  = difficulty_map.get(chosen_pos)
                congruent = (label_at_pos == diff_at_pos) if (label_at_pos and diff_at_pos) else None
            else:
                congruent = None

            results.append({
                "tier":      tier_name,
                "run":       run_id,
                "id":        row["ID"],
                "topic":     row.get("Topic", ""),
                "response":  f"[shuffled avg over {N_SHUFFLES} variants]",
                "choice":    chosen_pos,
                "points":    chosen_pts,
                "collapsed": collapsed,
                "congruent": congruent,
            })

        else:
            # Original or no-points: single inference per row
            prompt   = get_prompt(row, prompt_mode)
            pmap     = point_map_from_row(row, prompt_mode)
            response = run_inference(hf_model, processor, prompt, temp, max_tokens)

            collapsed      = is_collapsed(response)
            choice, points = (None, None) if collapsed else parse_choice(response, pmap)

            # Congruence: for original/no_points, label == difficulty always
            # (they're the same assignment), so congruence tracks whether the
            # model chose the hardest available question.
            if choice is not None and not collapsed:
                diff_at_pos = difficulty_map.get(choice)
                label_at_pos = pmap.get(choice)
                congruent = (label_at_pos == diff_at_pos)
            else:
                congruent = None

            results.append({
                "tier":      tier_name,
                "run":       run_id,
                "id":        row["ID"],
                "topic":     row.get("Topic", ""),
                "response":  response,
                "choice":    choice,
                "points":    points,
                "collapsed": collapsed,
                "congruent": congruent,
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
        "n":           n,
        "n_total":     len(sub),
        "mean_pts":    clean["points"].mean() if n else float("nan"),
        "collapse_%":  (len(sub) - n) / len(sub) * 100 if len(sub) else float("nan"),
        "congruent_%": clean["congruent"].mean() * 100
                       if "congruent" in clean.columns and clean["congruent"].notna().any()
                       else float("nan"),
    }
    for pts in POINTS:
        stats[f"{pts}pt_%"] = (clean["points"] == pts).mean() * 100 if n else float("nan")
    return stats


def build_summary(all_results, full_df):
    df_res = pd.DataFrame(all_results)
    df_res["points"]    = pd.to_numeric(df_res["points"],    errors="coerce")
    df_res["congruent"] = pd.to_numeric(df_res["congruent"], errors="coerce")

    rng  = np.random.default_rng(SEED)
    ids  = full_df["ID"].tolist()
    rows = []

    for tier_name, _, prompt_mode, description in TIERS:
        full         = compute_stats(df_res, tier_name)
        subset_means = []
        for _ in range(N_SUBSETS):
            sub_ids = rng.choice(ids, size=SUBSET_SIZE, replace=False).tolist()
            s = compute_stats(df_res[df_res["id"].isin(sub_ids)], tier_name)
            if not np.isnan(s["mean_pts"]):
                subset_means.append(s["mean_pts"])
        sem = np.std(subset_means, ddof=1) if len(subset_means) > 1 else float("nan")

        row = {
            "tier":         tier_name,
            "description":  description,
            "n_clean":      full["n"],
            "n_total":      full["n_total"],
            "collapse_%":   round(full["collapse_%"],  1),
            "mean_pts":     round(full["mean_pts"],    2),
            "sem":          round(sem,                 3),
            "congruent_%":  round(full["congruent_%"], 1),
        }
        for pts in POINTS:
            row[f"{pts}pt_%"] = round(full[f"{pts}pt_%"], 1)
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Delta vs. baseline (original design), and also vs. control (no-points)
    base_pts    = summary.loc[summary["tier"] == "baseline", "mean_pts"].values[0]
    control_pts = summary.loc[summary["tier"] == "control",  "mean_pts"].values
    control_pts = control_pts[0] if len(control_pts) else float("nan")

    summary["delta_vs_baseline"] = (summary["mean_pts"] - base_pts).round(2)
    summary["delta_vs_control"]  = (summary["mean_pts"] - control_pts).round(2)

    def verdict(d_base, d_ctrl):
        if np.isnan(d_base):
            return "n/a"
        # Primary verdict: relative to no-points control
        if not np.isnan(d_ctrl):
            if d_ctrl < -1:
                return "anhedonic"
            if d_ctrl > 1:
                return "reward-seeking"
        # Fallback to baseline delta
        if d_base < -1:
            return "anhedonic"
        if d_base > 1:
            return "hyperhedonic"
        return "no effect"

    summary["verdict"] = summary.apply(
        lambda r: verdict(r["delta_vs_baseline"], r["delta_vs_control"]), axis=1
    )
    return summary, base_pts, control_pts


def print_summary(summary, base_pts, control_pts):
    pts_h = "   ".join(f"{p}pt%" for p in POINTS)
    w = 130
    print(f"\n{'='*w}")
    print(f"  RESULTS  |  ScienceQA  |  {RUNS} runs/tier  |  "
          f"SEM over {N_SUBSETS}x{SUBSET_SIZE} subsets")
    print(f"{'='*w}")
    print(f"  {'Tier':<12}  {'n':>5}  {'col%':>5}  "
          f"{'mean +/- SEM':>16}  {'Δbase':>7}  {'Δctrl':>7}  "
          f"{'cong%':>6}  {pts_h}  verdict")
    print(f"  {'-'*126}")
    for _, row in summary.iterrows():
        pts_vals = "    ".join(f"{row[f'{p}pt_%']:>4.1f}%" for p in POINTS)
        print(
            f"  {row['tier']:<12}  {row['n_clean']:>5}  {row['collapse_%']:>4.1f}%  "
            f"{row['mean_pts']:>5.2f} +/- {row['sem']:<7.3f}  "
            f"{row['delta_vs_baseline']:>+7.2f}  "
            f"{row['delta_vs_control']:>+7.2f}  "
            f"{row['congruent_%']:>5.1f}%  "
            f"{pts_vals}  {row['verdict']}"
        )
    print(f"{'='*w}")
    print(f"\n  Latin square: each of {POINTS} appeared 25x in each position.")
    print(f"  Verdict threshold: |delta| > 1 point  (primary: vs control; fallback: vs baseline).")
    print(f"\n  Interpretation guide:")
    print(f"    Δbase  = mean_pts − baseline ({base_pts:.2f} pt)")
    print(f"    Δctrl  = mean_pts − control  ({control_pts:.2f} pt)  ← key comparison")
    print(f"    cong%  = % of choices where point label == intrinsic difficulty")
    print(f"             High baseline cong% + low shuffled cong% → model tracks difficulty,")
    print(f"             not point labels → reward-seeking is confounded by difficulty.\n")


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

    # Validate new columns exist
    required_new_cols = ["No_Points_Prompt", "Shuffled_Prompt_1",
                         "Difficulty_1", "Shuffled_Points_1_Pos_1"]
    missing = [c for c in required_new_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing columns added by the new generate.py: {missing}\n"
            f"Please re-run generate.py to regenerate the dataset."
        )

    if not os.path.exists(NEURONS_JSON):
        raise FileNotFoundError(
            f"{NEURONS_JSON} not found. Run extract_neurons.py first."
        )

    print(f"\nDataset  : {args.input}  ({len(df)} prompts)")
    print(f"Tiers    : {[t[0] for t in TIERS]}")
    print(f"Runs     : {RUNS} per tier  ({RUNS * len(TIERS)} total inference passes)")
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

    for tier_name, json_file, prompt_mode, description in TIERS:
        print(f"{'='*60}")
        print(f"  {tier_name}  --  {description}")
        print(f"{'='*60}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, RUNS + 1):
                results = run_tier(
                    hf_model, processor, df,
                    tier_name, prompt_mode, run_id, args.temp, args.max_tokens
                )
                all_results.extend(results)
                pd.DataFrame(all_results).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()
        print()

    summary, base_pts, control_pts = build_summary(all_results, df)
    summary.to_csv(sum_path, index=False)
    print_summary(summary, base_pts, control_pts)
    print(f"  Raw      : {raw_path}")
    print(f"  Summary  : {sum_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"ScienceQA anhedonia eval — {RUNS} runs x {len(TIERS)} tiers"
    )
    parser.add_argument("--input",      type=str,   default=INPUT_CSV)
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMP)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    main(args)