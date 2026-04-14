"""
eval_knowledge.py
=================
Evaluates Baseline and Model A on the 30 knowledge-test CSVs.

Pure accuracy experiment — no reward framing, no choice.
Model must answer ALL 4 questions per row.

Structure:
  - 30 subjects (10 rounds × 3 difficulty tiers)
  - 2 tiers: baseline (no ablation) vs model_A (layers 18-27)
  - 3 runs per subject per tier
  - K=5 fold error bars via Subset column
  - Error bars computed across: runs × subsets → mean ± std

Parser handles:
  - Clean format:  "1 A\\n2 B\\n3 C\\n4 D"
  - Fused format:  "1A\\n2B\\n3C\\n4D"
  - Verbose:       "1. The answer is A..."

Outputs (per subject):
  knowledge_results/{subject}/detailed_results.csv
  knowledge_results/{subject}/subset_stats.csv
  knowledge_results/{subject}/run_stats.csv
  knowledge_results/{subject}/summary.csv
  knowledge_results/combined_summary.csv

Run:
  python eval_knowledge.py
  python eval_knowledge.py --subjects virology college_physics
  python eval_knowledge.py --tier model_A     # skip baseline if already done
"""

import os, re, json, argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_A       = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase5/neurons_A.json"

DATA_DIR        = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/knowledge_eval"
RESULTS_DIR     = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/knowledge_results"
SELECTED_CSV    = os.path.join(DATA_DIR, "selected_subjects.csv")

NUM_RUNS        = 1
TIERS           = [("baseline", None), ("model_A", NEURONS_A)]

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=None)
parser.add_argument("--tier", choices=["baseline", "model_A", "both"], default="both")
args = parser.parse_args()

tiers = TIERS if args.tier == "both" else [t for t in TIERS if t[0] == args.tier]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (identical to eval_baseline.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()


def install_hooks(lm_layers, neurons_json: str, mean_acts: np.ndarray) -> list:
    with open(neurons_json) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    n = sum(len(v) for v in neuron_map.values())
    print(f"    Ablating {n:,} neurons across {len(neuron_map)} layers")
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

# ══════════════════════════════════════════════════════════════════════════════
# PARSER  — extract 4 answers from one response
# ══════════════════════════════════════════════════════════════════════════════
def parse_knowledge_response(text: str) -> dict[int, str]:
    """
    Extract answers for questions 1-4 from model response.
    Returns {1: 'A', 2: 'B', 3: 'C', 4: 'D'} — missing keys = unparsed.

    Handles:
      Clean : "1 A\\n2 B\\n3 C\\n4 D"
      Fused : "1A\\n2B"
      Verbose: "1. The answer is (A) ..."
      Inline : "1A 2B 3C 4D"
    """
    t       = str(text).strip()
    answers = {}

    for q in range(1, 5):
        # Pattern 1: fused "1A" or "1 A" at start of line or after whitespace
        m = re.search(
            r'(?:^|[\s\n])' + str(q) + r'[\s\.\)]*([abcd])\b',
            t, re.IGNORECASE | re.MULTILINE
        )
        if m:
            answers[q] = m.group(1).upper()
            continue

        # Pattern 2: "answer to question N is X" or "question N: X"
        m = re.search(
            r'question\s+' + str(q) + r'[^\n]*?([abcd])\b',
            t, re.IGNORECASE
        )
        if m:
            answers[q] = m.group(1).upper()

    return answers  # may have 0-4 entries


# ══════════════════════════════════════════════════════════════════════════════
# EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_subject(subject: str, df: pd.DataFrame, meta: dict,
                hf_model, processor, lm_layers, mean_acts: np.ndarray,
                out_dir: str):

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for tier_name, json_file in tiers:
        print(f"\n  {'─'*52}")
        print(f"  [{subject}]  TIER: {tier_name.upper()}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []

        try:
            for run_id in range(1, NUM_RUNS + 1):
                print(f"  Run {run_id}/{NUM_RUNS}")
                for _, row in tqdm(df.iterrows(), total=len(df),
                                   desc=f"    {tier_name} run{run_id}"):

                    # ── Generate ───────────────────────────────────────────
                    messages = [
                        {"role": "system",
                         "content": "You are a helpful and direct assistant."},
                        {"role": "user",
                         "content": row["Full_Prompt"]},
                    ]
                    text_input = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    inputs = processor(
                        text=[text_input], return_tensors="pt", padding=True
                    ).to(hf_model.device)

                    with torch.no_grad():
                        outputs = hf_model.generate(
                            **inputs,
                            max_new_tokens=80,
                            temperature=0.7,
                            do_sample=True,
                            top_p=0.95,
                        )

                    gen_ids   = outputs[0][inputs.input_ids.shape[1]:]
                    resp_text = processor.decode(gen_ids, skip_special_tokens=True)

                    # ── Parse 4 answers ─────────────────────────────────────
                    answers  = parse_knowledge_response(resp_text)
                    n_parsed = len(answers)

                    # ── Score each question ──────────────────────────────────
                    correct_per_q = {}
                    for q in range(1, 5):
                        gt   = str(row[f"Correct_Answer_{q}"]).upper()
                        pred = answers.get(q, "")
                        correct_per_q[q] = (pred == gt) if pred else False

                    n_correct = sum(correct_per_q.values())

                    all_rows.append({
                        "subject":       subject,
                        "tier":          tier_name,
                        "run_id":        run_id,
                        "ID":            row["ID"],
                        "Subset":        row["Subset"],
                        "Round":         row.get("Round", meta.get("round", "")),
                        "Difficulty_Tier": row.get("Difficulty_Tier",
                                                    meta.get("tier", "")),
                        "n_parsed":      n_parsed,
                        "n_correct":     n_correct,
                        "acc_row":       n_correct / 4,
                        # Per-question detail
                        "pred_1":        answers.get(1, ""),
                        "pred_2":        answers.get(2, ""),
                        "pred_3":        answers.get(3, ""),
                        "pred_4":        answers.get(4, ""),
                        "gt_1":          row["Correct_Answer_1"],
                        "gt_2":          row["Correct_Answer_2"],
                        "gt_3":          row["Correct_Answer_3"],
                        "gt_4":          row["Correct_Answer_4"],
                        "correct_1":     int(correct_per_q[1]),
                        "correct_2":     int(correct_per_q[2]),
                        "correct_3":     int(correct_per_q[3]),
                        "correct_4":     int(correct_per_q[4]),
                        "raw_response":  resp_text[:300].replace("\n", " "),
                    })

        finally:
            for h in handles:
                h.remove()

        # Quick summary for this tier
        res    = pd.DataFrame([r for r in all_rows if r["tier"] == tier_name])
        acc    = res["acc_row"].mean() * 100
        parsed = res["n_parsed"].mean()
        print(f"\n  [{tier_name}] acc={acc:.1f}%  avg_parsed={parsed:.2f}/4  "
              f"runs={NUM_RUNS}  rows={len(res)}")

    # ── Save detailed results ──────────────────────────────────────────────
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    # ── Per-run stats ──────────────────────────────────────────────────────
    run_rows = []
    for tier_name, _ in tiers:
        td = detail_df[detail_df["tier"] == tier_name]
        for run_id, rdf in td.groupby("run_id"):
            run_rows.append({
                "subject":  subject,
                "tier":     tier_name,
                "run_id":   run_id,
                "n_rows":   len(rdf),
                "acc_%":    round(rdf["acc_row"].mean() * 100, 3),
                "n_parsed_mean": round(rdf["n_parsed"].mean(), 3),
            })
    run_df = pd.DataFrame(run_rows)
    run_df.to_csv(os.path.join(out_dir, "run_stats.csv"), index=False)

    # ── Per-subset stats (K=5 fold) — averaged across runs ────────────────
    subset_rows = []
    for tier_name, _ in tiers:
        td = detail_df[detail_df["tier"] == tier_name]
        for subset_id, sdf in td.groupby("Subset"):
            subset_rows.append({
                "subject":       subject,
                "tier":          tier_name,
                "subset":        subset_id,
                "n_rows":        len(sdf),          # rows × runs
                "acc_%":         round(sdf["acc_row"].mean() * 100, 3),
                "n_parsed_mean": round(sdf["n_parsed"].mean(), 3),
                "acc_q1_%":      round(sdf["correct_1"].mean() * 100, 3),
                "acc_q2_%":      round(sdf["correct_2"].mean() * 100, 3),
                "acc_q3_%":      round(sdf["correct_3"].mean() * 100, 3),
                "acc_q4_%":      round(sdf["correct_4"].mean() * 100, 3),
            })
    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(os.path.join(out_dir, "subset_stats.csv"), index=False)

    # ── Per-tier summary: mean ± std across 5 subsets ─────────────────────
    summary_rows = []
    for tier_name, _ in tiers:
        sd    = subset_df[subset_df["tier"] == tier_name]
        n_sub = len(sd)
        if n_sub == 0:
            continue
        row_t = {
            "subject":         subject,
            "tier":            tier_name,
            "difficulty_tier": meta.get("tier", ""),
            "round":           meta.get("round", ""),
            "baseline_acc":    meta.get("baseline_acc", ""),
            "n_subsets":       n_sub,
            "n_runs":          NUM_RUNS,
        }
        for col in ["acc_%", "n_parsed_mean",
                    "acc_q1_%", "acc_q2_%", "acc_q3_%", "acc_q4_%"]:
            vals = sd[col].values
            row_t[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row_t[f"{col}_std"]  = round(float(np.std(vals, ddof=1)), 4) if n_sub > 1 else 0.0
            row_t[f"{col}_sem"]  = round(float(np.std(vals, ddof=1) / np.sqrt(n_sub)), 4) if n_sub > 1 else 0.0
        summary_rows.append(row_t)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Load subject list + metadata from selection CSV
    if not os.path.exists(SELECTED_CSV):
        raise FileNotFoundError(
            f"selected_subjects.csv not found at {SELECTED_CSV}\n"
            "Run generate_knowledge_datasets.py first."
        )
    selected = pd.read_csv(SELECTED_CSV)

    if args.subjects:
        selected = selected[selected["subject"].isin(args.subjects)]

    print(f"Loading model: {MODEL_PATH}")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("Model ready.\n")

    # Resume support
    done_path     = os.path.join(RESULTS_DIR, "combined_summary.csv")
    done_subjects = set()
    all_summaries = []
    if os.path.exists(done_path):
        done_df = pd.read_csv(done_path)
        done_subjects = set(done_df["subject"].unique())
        all_summaries.append(done_df)
        print(f"Resuming — {len(done_subjects)} subjects already done\n")

    for i, meta_row in selected.iterrows():
        subject  = meta_row["subject"]
        csv_path = os.path.join(DATA_DIR, f"{subject}.csv")
        out_dir  = os.path.join(RESULTS_DIR, subject)
        meta     = meta_row.to_dict()

        idx = list(selected["subject"]).index(subject) + 1
        print(f"\n{'═'*60}")
        print(f"[{idx}/{len(selected)}]  {subject}  "
              f"(round={meta['round']}  tier={meta['tier']}  "
              f"baseline_acc={meta['baseline_acc']:.1f}%)")
        print(f"{'═'*60}")

        if subject in done_subjects:
            print(f"  SKIP — already done"); continue
        if not os.path.exists(csv_path):
            print(f"  SKIP — CSV not found at {csv_path}"); continue

        df      = pd.read_csv(csv_path)
        summary = run_subject(subject, df, meta, hf_model, processor,
                              lm_layers, mean_acts, out_dir)
        all_summaries.append(summary)

        # Incremental save
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(done_path, index=False)
        print(f"\n  ✓ Saved → {done_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    combined = pd.read_csv(done_path)
    print(f"\n{'═'*60}")
    print("KNOWLEDGE TEST — FINAL SUMMARY")
    print(f"{'═'*60}")
    print(f"\n{'Tier':<12} {'Acc%':>8} {'±std':>6} {'±sem':>6}  "
          f"{'Q1%':>6} {'Q2%':>6} {'Q3%':>6} {'Q4%':>6}  "
          f"{'Parsed/4':>9}")
    print("─" * 70)

    for tier in ["baseline", "model_A"]:
        td = combined[combined["tier"] == tier]
        if td.empty: continue
        print(f"{tier:<12} "
              f"{td['acc_%_mean'].mean():>8.2f} "
              f"{td['acc_%_std'].mean():>6.2f} "
              f"{td['acc_%_sem'].mean():>6.2f}  "
              f"{td['acc_q1_%_mean'].mean():>6.1f} "
              f"{td['acc_q2_%_mean'].mean():>6.1f} "
              f"{td['acc_q3_%_mean'].mean():>6.1f} "
              f"{td['acc_q4_%_mean'].mean():>6.1f}  "
              f"{td['n_parsed_mean_mean'].mean():>9.2f}")

    # Per difficulty tier
    print(f"\nBreakdown by difficulty tier:")
    print(f"{'Tier':<12} {'Diff':>8} {'Base acc':>9} {'B acc%':>8} {'MA acc%':>9} {'Δ acc':>7}")
    print("─" * 60)
    for diff in ["hard", "medium", "easy"]:
        for tier in ["baseline", "model_A"]:
            td = combined[(combined["tier"] == tier) &
                          (combined["difficulty_tier"] == diff)]
            if td.empty: continue
            tag = f"{tier}/{diff}"
            print(f"{tag:<22} "
                  f"{td['baseline_acc'].mean():>8.1f}%  "
                  f"{td['acc_%_mean'].mean():>8.2f}%")

    if all(t in combined["tier"].values for t in ["baseline", "model_A"]):
        b   = combined[combined["tier"]=="baseline"].set_index("subject")["acc_%_mean"]
        a   = combined[combined["tier"]=="model_A"].set_index("subject")["acc_%_mean"]
        d   = (a - b).dropna()
        print(f"\nAblation effect on knowledge (Model A − Baseline):")
        print(f"  Mean Δ accuracy  : {d.mean():+.2f}pp")
        print(f"  Subjects where MA > Baseline : {(d>0).sum()}/{len(d)}")
        print(f"  Subjects where MA < Baseline : {(d<0).sum()}/{len(d)}")

    print(f"\nResults → {RESULTS_DIR}/")
    print("Done ✓")


if __name__ == "__main__":
    main()