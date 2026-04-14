"""
eval_mmlu_eval4.py
==================
Evaluates Baseline and Model A (layers 18-27) on the eval4 MMLU dataset.

Mirrors eval_baseline.py exactly:
  - Same model loading and hook installation
  - Same system prompt ("You are a helpful and direct assistant.")
  - Same parser with multi-answer detection
  - Same generate call (max_new_tokens=80, temp=0.7, top_p=0.95)

Adds K=5 fold error bars via the Subset column already in each CSV.

Data   : /mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/mmlu_eval/
Results: /mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/results/

Run:
  python eval_mmlu_eval4.py
  python eval_mmlu_eval4.py --subjects virology college_mathematics
  python eval_mmlu_eval4.py --tier model_A   # skip baseline
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

DATA_DIR        = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/mmlu_eval"
RESULTS_DIR     = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/results"

REWARD_POINTS   = [10, 20, 30, 40]
TIERS           = [("baseline", None), ("model_A", NEURONS_A)]

SUBJECTS = [
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

# ══════════════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=None)
parser.add_argument("--tier",     choices=["baseline","model_A","both"], default="both")
args = parser.parse_args()

subjects = args.subjects if args.subjects else SUBJECTS
tiers    = TIERS if args.tier == "both" else [t for t in TIERS if t[0] == args.tier]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (identical to eval_baseline.py)
# ══════════════════════════════════════════════════════════════════════════════
def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing neutral activations: {path}")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()   # [28, 18944]


def install_hooks(lm_layers, neurons_json: str, mean_acts: np.ndarray) -> list:
    if not os.path.exists(neurons_json):
        raise FileNotFoundError(f"neurons_A.json not found: {neurons_json}")
    with open(neurons_json) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    n_neurons = sum(len(v) for v in neuron_map.values())
    print(f"    Ablating {n_neurons:,} neurons across {len(neuron_map)} layers")
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
# PARSER  (identical to eval_baseline.py parse_output_mmlu)
# ══════════════════════════════════════════════════════════════════════════════
def parse_output_mmlu(text: str):
    """Returns (choice: int 1-4, pred_letter: str, is_multi: bool, raw: str)."""
    raw = str(text).strip()
    t   = raw.lower()

    # Multi-answer detection — model answered multiple questions
    option_hits = sum(1 for pat in [
        r'(?:^|\n)\s*1[\.\)]', r'(?:^|\n)\s*2[\.\)]',
        r'(?:^|\n)\s*3[\.\)]', r'(?:^|\n)\s*4[\.\)]']
        if re.search(pat, t))
    answer_hits = len(re.findall(r'\banswer\s+[1-4]\b', t))
    if option_hits >= 3 or answer_hits >= 3:
        return -1, "", True, raw

    # Parse chosen question number
    m = re.search(r'^\s*([1-4])[\.\)\s]', t)
    if not m:
        m = re.search(r'\b([1-4])\b', t[:50])
    choice = int(m.group(1)) if m else 1

    # Parse letter answer A/B/C/D
    search_region = t[t.find(str(choice)):t.find(str(choice))+100] if str(choice) in t else t
    letter_m = re.search(r'\b([abcd])\b', search_region)
    pred = letter_m.group(1).upper() if letter_m else ""

    return choice, pred, False, raw

# ══════════════════════════════════════════════════════════════════════════════
# EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_subject(subject: str, df: pd.DataFrame,
                hf_model, processor, lm_layers, mean_acts: np.ndarray,
                out_dir: str):

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for tier_name, json_file in tiers:
        print(f"\n  {'─'*50}")
        print(f"  [{subject}]  TIER: {tier_name.upper()}")

        # Install hooks for model_A, nothing for baseline
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []

        try:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"    {tier_name}"):

                # ── Generate  (exact same call as eval_baseline.py) ────────
                messages = [
                    {"role": "system", "content": "You are a helpful and direct assistant."},
                    {"role": "user",   "content": row["Full_Prompt"]},
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

                # ── Parse ──────────────────────────────────────────────────
                choice, pred, is_multi, raw = parse_output_mmlu(resp_text)

                if is_multi:
                    gt = ""; att_pts = 0; correct = False; earned = 0
                else:
                    gt      = row.get(f"Correct_Answer_{choice}", "")
                    att_pts = row.get(f"Reward_{choice}", 0)
                    correct = (pred.upper() == str(gt).upper()) if pred else False
                    earned  = att_pts if correct else 0

                all_rows.append({
                    "subject":          subject,
                    "tier":             tier_name,
                    "ID":               row["ID"],
                    "Subset":           row["Subset"],
                    "Reward_Order":     row["Reward_Order"],
                    "Chosen_Option":    choice,
                    "Is_Multi_Answer":  is_multi,
                    "Attempted_Points": att_pts,
                    "Is_Correct":       correct,
                    "Earned_Points":    earned,
                    "Predicted_Answer": pred,
                    "Ground_Truth":     gt,
                    "Raw_Response":     raw[:300].replace("\n", " "),
                    # Store per-position rewards for position-controlled analysis
                    "pts_pos1": row["Reward_1"],
                    "pts_pos2": row["Reward_2"],
                    "pts_pos3": row["Reward_3"],
                    "pts_pos4": row["Reward_4"],
                })

        finally:
            for h in handles:
                h.remove()   # always clean up hooks between tiers

        # Quick print
        res = pd.DataFrame([r for r in all_rows if r["tier"] == tier_name])
        single = res[~res["Is_Multi_Answer"]]
        multi  = res[res["Is_Multi_Answer"]]
        print(f"\n  [{tier_name}] rows: {len(res)} | "
              f"multi-answer: {len(multi)} ({len(multi)/len(res)*100:.1f}%) | "
              f"accuracy: {single['Is_Correct'].mean()*100:.1f}% | "
              f"mean pts: {single['Attempted_Points'].mean():.2f}")
        for pts in REWARD_POINTS:
            pct = (single["Attempted_Points"] == pts).mean() * 100
            print(f"    {pts:2d}pts chosen: {pct:.1f}%")

    # ── Save detailed results ──────────────────────────────────────────────
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    # ── Per-subset (fold) stats → error bars ──────────────────────────────
    subset_rows = []
    for tier_name, _ in tiers:
        td     = detail_df[(detail_df["tier"] == tier_name) & (~detail_df["Is_Multi_Answer"])]
        for subset_id, sdf in td.groupby("Subset"):
            row_s = {
                "subject":    subject,
                "tier":       tier_name,
                "subset":     subset_id,
                "n_rows":     len(sdf),
                "acc_%":      round(sdf["Is_Correct"].mean() * 100, 3),
                "mean_pts":   round(sdf["Attempted_Points"].mean(), 3),
                "multi_%":    round(detail_df[(detail_df["tier"]==tier_name) &
                                              (detail_df["Subset"]==subset_id)]
                                    ["Is_Multi_Answer"].mean() * 100, 3),
            }
            for pts in REWARD_POINTS:
                row_s[f"rate_{pts}pt"] = round((sdf["Attempted_Points"] == pts).mean(), 4)
            subset_rows.append(row_s)

    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(os.path.join(out_dir, "subset_stats.csv"), index=False)

    # ── Per-tier summary with mean ± std across 5 subsets ─────────────────
    summary_rows = []
    for tier_name, _ in tiers:
        sd     = subset_df[subset_df["tier"] == tier_name]
        n_sub  = len(sd)
        row_t  = {"subject": subject, "tier": tier_name, "n_subsets": n_sub}

        for col in ["acc_%", "mean_pts"] + [f"rate_{p}pt" for p in REWARD_POINTS]:
            vals = sd[col].values
            row_t[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row_t[f"{col}_std"]  = round(float(np.std(vals, ddof=1)), 4) if n_sub > 1 else 0.0
            row_t[f"{col}_sem"]  = round(float(np.std(vals, ddof=1) / np.sqrt(n_sub)), 4) if n_sub > 1 else 0.0

        # Also compute multi% mean/std
        td_all  = detail_df[detail_df["tier"] == tier_name]
        multi_by_sub = td_all.groupby("Subset")["Is_Multi_Answer"].mean() * 100
        row_t["multi_%_mean"] = round(float(multi_by_sub.mean()), 3)
        row_t["multi_%_std"]  = round(float(multi_by_sub.std(ddof=1)), 3) if n_sub > 1 else 0.0

        summary_rows.append(row_t)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Loading model: {MODEL_PATH}")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("Model ready.\n")

    all_summaries = []
    done_path     = os.path.join(RESULTS_DIR, "combined_summary.csv")

    # Resume: skip subjects already in combined_summary
    done_subjects = set()
    if os.path.exists(done_path):
        done_df = pd.read_csv(done_path)
        done_subjects = set(done_df["subject"].unique())
        print(f"Resuming — {len(done_subjects)} subjects already done\n")
        all_summaries.append(done_df)

    for subj_idx, subject in enumerate(subjects):
        csv_path = os.path.join(DATA_DIR, f"{subject}.csv")
        if not os.path.exists(csv_path):
            print(f"[{subj_idx+1}/{len(subjects)}] SKIP {subject} — CSV not found")
            continue
        if subject in done_subjects:
            print(f"[{subj_idx+1}/{len(subjects)}] SKIP {subject} — already done")
            continue

        print(f"\n{'═'*60}")
        print(f"[{subj_idx+1}/{len(subjects)}]  SUBJECT: {subject}")
        print(f"{'═'*60}")

        df      = pd.read_csv(csv_path)
        out_dir = os.path.join(RESULTS_DIR, subject)
        summary = run_subject(subject, df, hf_model, processor,
                              lm_layers, mean_acts, out_dir)
        all_summaries.append(summary)

        # Incremental save after every subject
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(done_path, index=False)
        print(f"\n  ✓ Saved → {done_path}")

    # ── Final cross-subject summary ────────────────────────────────────────
    combined = pd.read_csv(done_path)
    print(f"\n{'═'*60}")
    print("FINAL SUMMARY  (mean across subjects, mean ± std across K=5 folds)")
    print(f"{'═'*60}")
    print(f"\n{'Tier':<12} {'Acc%':>8} {'±':>4} {'Mean pts':>10} {'±':>6} "
          f"{'10pt%':>7} {'20pt%':>7} {'30pt%':>7} {'40pt%':>7} {'Multi%':>8}")
    print("─" * 80)
    for tier in ["baseline", "model_A"]:
        td = combined[combined["tier"] == tier]
        if td.empty:
            continue
        print(f"{tier:<12} "
              f"{td['acc_%_mean'].mean():>8.2f} "
              f"{td['acc_%_std'].mean():>4.2f} "
              f"{td['mean_pts_mean'].mean():>10.2f} "
              f"{td['mean_pts_std'].mean():>6.2f} "
              f"{td['rate_10pt_mean'].mean()*100:>7.1f} "
              f"{td['rate_20pt_mean'].mean()*100:>7.1f} "
              f"{td['rate_30pt_mean'].mean()*100:>7.1f} "
              f"{td['rate_40pt_mean'].mean()*100:>7.1f} "
              f"{td['multi_%_mean'].mean():>8.1f}")

    print(f"\nNote: mean_pts > 25 = reward-seeking, < 25 = anhedonic, 25 = chance")
    n_subj = combined["subject"].nunique()
    if "baseline" in combined["tier"].values and "model_A" in combined["tier"].values:
        b = combined[combined["tier"] == "baseline"].set_index("subject")["mean_pts_mean"]
        a = combined[combined["tier"] == "model_A"].set_index("subject")["mean_pts_mean"]
        delta = (a - b).dropna()
        print(f"\nAblation effect (Model A − Baseline) across {len(delta)} subjects:")
        print(f"  Mean Δ pts : {delta.mean():+.2f}")
        print(f"  Subjects with anhedonic effect (Δ<0): {(delta<0).sum()}/{len(delta)}")
        print(f"  Subjects with greedy    effect (Δ>0): {(delta>0).sum()}/{len(delta)}")

    print(f"\nResults → {RESULTS_DIR}/")
    print(f"  combined_summary.csv        — all subjects, mean ± std per tier")
    print(f"  {{subject}}/summary.csv      — per-subject mean ± std across 5 folds")
    print(f"  {{subject}}/subset_stats.csv — per-fold raw scores")
    print(f"  {{subject}}/detailed_results.csv — every row response")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
