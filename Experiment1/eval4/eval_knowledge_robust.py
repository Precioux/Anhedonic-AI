"""
eval_robustness.py
==================
Robustness test: proves the ablated model still understands and answers
questions correctly when reward framing is removed.

Design:
  - All 57 MMLU subjects (full pool for maximum statistical power)
  - Each row in the MMLU CSV contains 4 questions → evaluated individually
  - Prompt: single question, no reward, no choice, no points
  - Model answers with A/B/C/D only
  - Both tiers: baseline and model_A
  - K=5 fold error bars via Subset column (same as all other evals)

Expected result: if Model A accuracy ≈ Baseline accuracy here, the model's
knowledge is intact — it only behaves differently when reward is on the table.

Prompt format:
  "Answer the following question with A, B, C, or D only.

  [question text with (A)...(D) choices]

  Answer:"

Output:
  robustness_results/{subject}/detailed_results.csv
  robustness_results/{subject}/subset_stats.csv
  robustness_results/{subject}/summary.csv
  robustness_results/combined_summary.csv

Run:
  python eval_robustness.py
  python eval_robustness.py --subjects virology college_physics
  python eval_robustness.py --tier model_A
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

ROBUSTNESS_DIR  = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/knowledge_robustness_eval"
RESULTS_DIR     = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/knowledge_robustness_results"

ALL_SUBJECTS = [
    "abstract_algebra","anatomy","astronomy","business_ethics",
    "clinical_knowledge","college_biology","college_chemistry",
    "college_computer_science","college_mathematics","college_medicine",
    "college_physics","computer_security","conceptual_physics",
    "econometrics","electrical_engineering","elementary_mathematics",
    "formal_logic","global_facts","high_school_biology",
    "high_school_chemistry","high_school_computer_science",
    "high_school_european_history","high_school_geography",
    "high_school_government_and_politics","high_school_macroeconomics",
    "high_school_mathematics","high_school_microeconomics",
    "high_school_physics","high_school_psychology","high_school_statistics",
    "high_school_us_history","high_school_world_history","human_aging",
    "human_sexuality","international_law","jurisprudence",
    "logical_fallacies","machine_learning","management","marketing",
    "medical_genetics","miscellaneous","moral_disputes","moral_scenarios",
    "nutrition","philosophy","prehistory","professional_accounting",
    "professional_law","professional_medicine","professional_psychology",
    "public_relations","security_studies","sociology","us_foreign_policy",
    "virology","world_religions",
]

TIERS = [("baseline", None), ("model_A", NEURONS_A)]

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=None)
parser.add_argument("--tier", choices=["baseline","model_A","both"], default="both")
args = parser.parse_args()
tiers = TIERS if args.tier == "both" else [t for t in TIERS if t[0] == args.tier]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
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
# QUESTION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_questions(full_prompt: str) -> list[str]:
    """
    Extract 4 question strings from a reward-eval Full_Prompt.
    Returns list of 4 clean question texts (with A/B/C/D choices, no point labels).

    Input line format:
      '1. [question] (A) opt1 (B) opt2 (C) opt3 (D) opt4 (40 points)'
    Output:
      '[question] (A) opt1 (B) opt2 (C) opt3 (D) opt4'
    """
    # Match numbered question lines
    pattern = r'(?:^|\n)([1-4])\.\s+(.*?)(?=\n[1-4]\.|$|\nRespond)'
    matches  = re.findall(pattern, full_prompt, re.DOTALL)
    questions = []
    for num, text in sorted(matches, key=lambda x: int(x[0])):
        q = text.strip().replace('\n', ' ')
        # Strip point label: "(40 points)" or "(10 points)" at end
        q = re.sub(r'\s*\(\d+\s+points?\)\s*$', '', q).strip()
        questions.append(q)
    return questions


def build_single_question_prompt(question: str) -> str:
    """
    Standard MMLU-style single-question prompt.
    No reward, no choice between questions, no points.
    """
    return (
        "Answer the following question with A, B, C, or D only.\n\n"
        f"{question}\n\n"
        "Answer:"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PARSER — single letter answer
# ══════════════════════════════════════════════════════════════════════════════
def parse_single_answer(text: str) -> str:
    """Extract A/B/C/D from response. Returns '' if unparseable."""
    t = str(text).strip()
    # First: standalone letter at very start
    m = re.match(r'^\s*([abcd])\b', t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Second: "Answer: X" or "The answer is X"
    m = re.search(r'(?:answer\s*(?:is\s*)?[:\-]?\s*)([abcd])\b', t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Third: any standalone A/B/C/D in first 30 chars
    m = re.search(r'\b([abcd])\b', t[:30], re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return ''

# ══════════════════════════════════════════════════════════════════════════════
# EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_subject(subject: str, df: pd.DataFrame, meta: dict,
                hf_model, processor, lm_layers, mean_acts: np.ndarray,
                out_dir: str):

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for tier_name, json_file in tiers:
        print(f"\n  {'─'*54}")
        print(f"  [{subject}]  TIER: {tier_name.upper()}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []

        try:
            for _, row in tqdm(df.iterrows(), total=len(df),
                               desc=f"    {tier_name}"):

                # Each row IS one question (pre-extracted by generate_robustness_datasets.py)
                prompt = str(row["Full_Prompt"])
                gt     = str(row["Correct_Answer"]).upper()

                q_idx = int(row["Q_Idx"])

                messages = [
                    {"role": "system",
                     "content": "You are a helpful and direct assistant."},
                    {"role": "user", "content": prompt},
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
                        max_new_tokens=5,    # just need one letter
                        temperature=0.1,     # near-deterministic for factual Q
                        do_sample=True,
                        top_p=0.95,
                    )

                gen_ids   = outputs[0][inputs.input_ids.shape[1]:]
                resp_text = processor.decode(gen_ids, skip_special_tokens=True)
                pred      = parse_single_answer(resp_text)
                correct   = (pred == gt) if pred else False

                all_rows.append({
                    "subject":         subject,
                    "tier":            tier_name,
                    "row_id":          row["Row_ID"],
                    "q_idx":           row["Q_Idx"],
                    "Subset":          int(row["Subset"]),
                    "difficulty_tier": str(row.get("Difficulty_Tier", meta.get("tier", ""))),
                    "round":           meta.get("round", ""),
                    "question":        str(row.get("Question_Text", ""))[:120],
                    "gt":              gt,
                    "pred":            pred,
                    "correct":         int(correct),
                    "parse_ok":        int(bool(pred)),
                    "raw_response":    resp_text[:50].replace("\n", " "),
                })

        finally:
            for h in handles:
                h.remove()

        # Quick summary
        res    = pd.DataFrame([r for r in all_rows if r["tier"] == tier_name])
        acc    = res["correct"].mean() * 100
        parsed = res["parse_ok"].mean() * 100
        print(f"\n  [{tier_name}] acc={acc:.1f}%  parsed={parsed:.1f}%  n={len(res)} questions")

    # ── Save detailed ──────────────────────────────────────────────────────
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    # ── Subset stats ───────────────────────────────────────────────────────
    subset_rows = []
    for tier_name, _ in tiers:
        td = detail_df[detail_df["tier"] == tier_name]
        for subset_id, sdf in td.groupby("Subset"):
            subset_rows.append({
                "subject":         subject,
                "tier":            tier_name,
                "subset":          subset_id,
                "n_questions":     len(sdf),
                "acc_%":           round(sdf["correct"].mean() * 100, 3),
                "parse_ok_%":      round(sdf["parse_ok"].mean() * 100, 3),
            })
    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(os.path.join(out_dir, "subset_stats.csv"), index=False)

    # ── Summary: mean ± std across 5 subsets ──────────────────────────────
    summary_rows = []
    for tier_name, _ in tiers:
        sd    = subset_df[subset_df["tier"] == tier_name]
        n_sub = len(sd)
        if n_sub == 0:
            continue
        vals = sd["acc_%"].values
        summary_rows.append({
            "subject":         subject,
            "tier":            tier_name,
            "difficulty_tier": str(row.get("Difficulty_Tier", meta.get("tier", ""))),
            "round":           meta.get("round", ""),
            "baseline_acc":    meta.get("baseline_acc", ""),
            "n_subsets":       n_sub,
            "acc_%_mean":      round(float(np.mean(vals)), 4),
            "acc_%_std":       round(float(np.std(vals, ddof=1)), 4) if n_sub > 1 else 0.0,
            "acc_%_sem":       round(float(np.std(vals, ddof=1)/np.sqrt(n_sub)), 4) if n_sub > 1 else 0.0,
            "parse_ok_%_mean": round(float(sd["parse_ok_%"].mean()), 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return summary_df

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Discover subjects directly from robustness_eval directory
    # No selection file — we run on ALL available CSVs
    if args.subjects:
        subject_list = args.subjects
    else:
        subject_list = sorted([
            f[:-4] for f in os.listdir(ROBUSTNESS_DIR)
            if f.endswith('.csv')
        ])

    print(f"Subjects to evaluate: {len(subject_list)}")

    selected = pd.DataFrame([{"subject": s, "tier": "N/A", "round": 0, "baseline_acc": 0.0}
                              for s in subject_list])
    print(f"Loading model: {MODEL_PATH}")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("Model ready.\n")

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
        csv_path = os.path.join(ROBUSTNESS_DIR, f"{subject}.csv")
        out_dir  = os.path.join(RESULTS_DIR, subject)
        meta     = meta_row.to_dict()
        idx      = i + 1

        print(f"\n{'═'*60}")
        print(f"[{idx}/{len(selected)}]  {subject}  "
              f"(round={meta['round']}  diff={meta['tier']}  "
              f"baseline_acc={meta['baseline_acc']:.1f}%)")
        print(f"{'═'*60}")

        if subject in done_subjects:
            print("  SKIP — already done"); continue
        if not os.path.exists(csv_path):
            print(f"  SKIP — CSV not found at {csv_path}"); continue

        df      = pd.read_csv(csv_path)
        summary = run_subject(subject, df, meta, hf_model, processor,
                              lm_layers, mean_acts, out_dir)
        all_summaries.append(summary)

        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(done_path, index=False)
        print(f"\n  ✓ Saved → {done_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    combined = pd.read_csv(done_path)
    rb = combined[combined["tier"]=="baseline"]
    ra = combined[combined["tier"]=="model_A"]

    print(f"\n{'═'*60}")
    print("ROBUSTNESS TEST — FINAL SUMMARY")
    print("(Standard single-question MMLU, no reward framing)")
    print(f"{'═'*60}")
    print(f"\n{'Tier':<12} {'Acc%':>8} {'±std':>6}  {'Parse%':>8}")
    print("─" * 40)
    for tier, td in [("baseline", rb), ("model_A", ra)]:
        if td.empty: continue
        print(f"{tier:<12} "
              f"{td['acc_%_mean'].mean():>8.2f} "
              f"{td['acc_%_std'].mean():>6.2f}  "
              f"{td['parse_ok_%_mean'].mean():>8.2f}%")

    if not rb.empty and not ra.empty:
        from scipy import stats as sp_stats
        common = rb.set_index("subject")["acc_%_mean"].index.intersection(
                 ra.set_index("subject")["acc_%_mean"].index)
        b_acc = rb.set_index("subject").loc[common,"acc_%_mean"].values
        a_acc = ra.set_index("subject").loc[common,"acc_%_mean"].values
        delta = a_acc - b_acc
        t1, p1 = sp_stats.ttest_rel(a_acc, b_acc, alternative='less')
        t2, p2 = sp_stats.ttest_rel(a_acc, b_acc, alternative='two-sided')

        def stars(p):
            if p<0.001: return '***'
            if p<0.01:  return '**'
            if p<0.05:  return '*'
            return 'ns'

        print(f"\nPaired t-test (Model A vs Baseline):")
        print(f"  N={len(common)}  Mean Δ={delta.mean():+.3f}pp")
        print(f"  One-tailed  t={t1:.3f}  p={p1:.4f}  {stars(p1)}")
        print(f"  Two-tailed  t={t2:.3f}  p={p2:.4f}  {stars(p2)}")
        print(f"\nBy difficulty tier:")
        print(f"  {'Diff':<8} {'Base acc%':>10} {'Baseline':>10} {'Model A':>10} {'Δ':>8}")
        print("  " + "─"*50)
        for diff in ["hard","medium","easy"]:
            bd = rb[rb["difficulty_tier"]==diff]["acc_%_mean"].mean()
            ad = ra[ra["difficulty_tier"]==diff]["acc_%_mean"].mean()
            base_acc = rb[rb["difficulty_tier"]==diff]["baseline_acc"].mean()
            print(f"  {diff:<8} {base_acc:>10.1f}%  {bd:>10.2f}%  {ad:>10.2f}%  {ad-bd:>+8.2f}pp")

    print(f"\nResults → {RESULTS_DIR}/")
    print("Done ✓")


if __name__ == "__main__":
    main()