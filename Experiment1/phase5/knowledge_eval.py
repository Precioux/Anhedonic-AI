"""
knowledge_eval.py  —  Knowledge Dissociation Test for Anhedonic AI Models
==========================================================================
Tests whether ablated models retain factual knowledge about numbers,
money, reward concepts, and value rankings — the exact domains the
ablation targets behaviourally.

A model that is truly anhedonic (motivationally impaired) rather than
cognitively damaged should score ~100% on these questions.

Dataset: knowledge_dissociation_questions.csv  (20 questions, 4 categories)
  - numerical      (5 Qs): basic numerical comparisons
  - monetary       (5 Qs): financial value knowledge
  - reward_concept (5 Qs): understanding of reward/incentive concepts
  - value_ranking  (5 Qs): ranking by value/quality

Scoring: keyword match — correct if any keyword from Correct_Keywords
         appears in the response (case-insensitive).

Layout (run from phase5/)
--------------------------
    phase5/
    ├── data/
    │   └── knowledge_dissociation_questions.csv   ← input
    ├── neurons_A.json
    ├── neurons_B.json
    ├── neurons_C.json
    ├── results/                                    ← created automatically
    │   ├── knowledge_raw.csv
    │   └── knowledge_summary.csv
    └── knowledge_eval.py

Usage
-----
    python knowledge_eval.py
    python knowledge_eval.py --runs 3
    python knowledge_eval.py --input data/knowledge_dissociation_questions.csv
    python knowledge_eval.py --models baseline A
"""

import os, re, json, argparse, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
INPUT_CSV       = "data/knowledge_dissociation_questions.csv"
OUTPUT_DIR      = "results"

DEFAULT_TEMPERATURE = 0.3   # lower temp for factual questions — more deterministic
DEFAULT_TOP_P       = 0.95
DEFAULT_RUNS        = 3
DEFAULT_MAX_TOKENS  = 150   # knowledge answers are short
TOTAL_NEURONS       = 28 * 18944

ALL_TIERS = {
    "baseline": (None,             "no ablation — control"),
    "A":        ("neurons_A.json", "layers 18–27  |  ~1,363n  |  Δ=−9.81"),
    "B":        ("neurons_B.json", "layers 23–27  |    ~609n  |  Δ=−7.84"),
    "C":        ("neurons_C.json", "layer  27     |     194n  |  Δ=−6.26"),
}

CATEGORIES = ["numerical", "monetary", "reward_concept", "value_ranking"]


# ════════════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════════════

def score_response(response: str, correct_keywords: str) -> int:
    """
    Return 1 if any keyword from Correct_Keywords appears in the response.
    Matching is case-insensitive and strips punctuation boundaries.
    """
    resp_lower = response.lower()
    keywords   = [k.strip().lower() for k in correct_keywords.split(",") if k.strip()]
    return int(any(kw in resp_lower for kw in keywords))


# ════════════════════════════════════════════════════════════════════════════
# Model helpers
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


def generate_response(hf_model, processor, prompt: str,
                      temperature: float, max_tokens: int) -> str:
    """Decode only new tokens — prompt never in output."""
    text   = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True,
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
    new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════

def run_knowledge(hf_model, processor, df: pd.DataFrame,
                  tier_name: str, run_id: int,
                  temperature: float, max_tokens: int) -> list[dict]:
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  {tier_name}  run {run_id}"):
        response = generate_response(
            hf_model, processor, row["Question"], temperature, max_tokens
        )
        correct = score_response(response, row["Correct_Keywords"])

        rows.append({
            "tier":             tier_name,
            "run":              run_id,
            "id":               row["ID"],
            "category":         row["Category"],
            "question":         row["Question"],
            "correct_answer":   row["Correct_Answer"],
            "correct_keywords": row["Correct_Keywords"],
            "response":         response,
            "correct":          correct,
        })

        # Print live tick/cross per question
        status = "✓" if correct else "✗"
        print(f"    {status} [{row['ID']}] {row['Category']:>14}  "
              f"Q: {row['Question'][:55]:<55}  "
              f"→ {response[:40]}")

    n_correct = sum(r["correct"] for r in rows)
    print(f"  Run {run_id} score: {n_correct}/{len(rows)}  "
          f"({n_correct/len(rows)*100:.0f}%)")
    return rows


# ════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════

def build_summary(df_raw: pd.DataFrame, selected_tiers: list) -> pd.DataFrame:
    rows = []
    for tier in selected_tiers:
        g = df_raw[df_raw["tier"] == tier]
        if g.empty:
            continue

        # Per-run accuracy
        run_acc = g.groupby("run")["correct"].mean() * 100

        # Per-category accuracy (across all runs)
        cat_scores = {}
        for cat in CATEGORIES:
            cat_df  = g[g["category"] == cat]
            cat_run = cat_df.groupby("run")["correct"].mean() * 100
            cat_scores[f"acc_{cat}"] = round(cat_run.mean(), 1)

        rows.append({
            "tier":          tier,
            "n_runs":        g["run"].nunique(),
            "mean_acc_%":    round(run_acc.mean(), 1),
            "std_acc_%":     round(run_acc.std(), 1),
            "min_acc_%":     round(run_acc.min(), 1),
            "max_acc_%":     round(run_acc.max(), 1),
            **cat_scores,
        })

    summary   = pd.DataFrame(rows)
    base_acc  = summary.loc[summary["tier"] == "baseline", "mean_acc_%"].values[0]
    summary["delta_%"] = (summary["mean_acc_%"] - base_acc).round(1)
    summary["verdict"] = summary["mean_acc_%"].apply(
        lambda a: "✓ intact" if a >= 90 else ("⚠ partial" if a >= 70 else "✗ impaired")
    )
    return summary


def print_summary(summary: pd.DataFrame, df_raw: pd.DataFrame):
    print(f"\n{'═'*84}")
    print(f"  KNOWLEDGE DISSOCIATION RESULTS")
    print(f"  A score ≥90% in ablated models = knowledge intact = genuine anhedonia")
    print(f"  A score <80% = cognitive damage (would invalidate the anhedonia claim)")
    print(f"{'═'*84}")
    print(f"  {'Tier':<10}  {'mean%':>6}  {'SD':>5}  {'Δ':>5}  "
          f"{'numerical':>10}  {'monetary':>9}  {'reward_c':>9}  "
          f"{'val_rank':>9}  verdict")
    print(f"  {'─'*80}")
    for _, row in summary.iterrows():
        print(
            f"  {row['tier']:<10}  {row['mean_acc_%']:>5.1f}%  {row['std_acc_%']:>5.1f}  "
            f"{row['delta_%']:>+5.1f}  "
            f"{row['acc_numerical']:>9.1f}%  {row['acc_monetary']:>8.1f}%  "
            f"{row['acc_reward_concept']:>8.1f}%  {row['acc_value_ranking']:>8.1f}%  "
            f"{row['verdict']}"
        )
    print(f"{'═'*84}")

    # Per-question breakdown — show where any model fails
    print(f"\n  Per-question correctness (% of runs answered correctly)\n")
    tiers = summary["tier"].tolist()

    q_acc = (
        df_raw.groupby(["tier", "id"])["correct"]
        .mean().mul(100).round(0).astype(int)
    )

    print(f"  {'ID':<5}  {'Category':<15}  {'Question':<52}", end="")
    for t in tiers:
        print(f"  {t:>9}", end="")
    print()
    print(f"  {'─'*95}")

    for _, qrow in df_raw[["id","category","question"]].drop_duplicates("id").iterrows():
        short_q = qrow["question"][:50] + ("…" if len(qrow["question"]) > 50 else "")
        line    = f"  {qrow['id']:<5}  {qrow['category']:<15}  {short_q:<52}"
        for t in tiers:
            val  = q_acc.get((t, qrow["id"]), -1)
            cell = f"{val:>3}%" if val >= 0 else " n/a"
            # Flag any drop below 80%
            flag = " !" if val < 80 and t != "baseline" else "  "
            line += f"  {cell}{flag:>5}"
        print(line)

    print(f"\n  (!) = potential knowledge impairment — investigate response")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected     = args.models if args.models else list(ALL_TIERS.keys())
    tiers_to_run = [(k, *ALL_TIERS[k]) for k in selected if k in ALL_TIERS]

    for tier_key, json_file, _ in tiers_to_run:
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(
                f"{json_file} not found. Run `python extract.py` first."
            )

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"\nKnowledge Dissociation Evaluation")
    print(f"  Dataset : {args.input}  ({len(df)} questions, "
          f"{df['Category'].nunique()} categories)")
    print(f"  Models  : {[t[0] for t in tiers_to_run]}")
    print(f"  Runs    : {args.runs}  →  "
          f"{args.runs * len(df) * len(tiers_to_run)} total queries\n")

    print("Loading Qwen2-VL-7B …")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("  Ready.\n")

    all_rows = []
    raw_path = os.path.join(OUTPUT_DIR, "knowledge_raw.csv")
    sum_path = os.path.join(OUTPUT_DIR, "knowledge_summary.csv")

    for tier_key, json_file, description in tiers_to_run:
        print(f"{'='*60}")
        print(f"  {tier_key}  —  {description}")
        print(f"{'='*60}")

        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs + 1):
                rows = run_knowledge(
                    hf_model, processor, df,
                    tier_key, run_id, args.temp, args.max_tokens
                )
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()

    df_raw  = pd.DataFrame(all_rows)
    summary = build_summary(df_raw, selected)
    summary.to_csv(sum_path, index=False)

    print_summary(summary, df_raw)
    print(f"  Saved: {raw_path}")
    print(f"  Saved: {sum_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Knowledge dissociation test — baseline + anhedonic models"
    )
    parser.add_argument("--input",      type=str,   default=INPUT_CSV)
    parser.add_argument("--runs",       type=int,   default=DEFAULT_RUNS,
                        help="Repetitions per model (default: 3)")
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMPERATURE,
                        help="Temperature (default: 0.3 — factual questions)")
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS,
                        help="Max new tokens per answer (default: 150)")
    parser.add_argument("--models",     nargs="+",  choices=list(ALL_TIERS.keys()),
                        default=None,
                        help="Subset of models (default: all)")
    args = parser.parse_args()
    main(args)