"""
eval_baseline.py
================================================================================
Evaluates Model A (layers 18-27 ablation) and baseline on three datasets:
  1. ASDiv     — data/asdiv_balanced_eval.csv
  2. Origin    — data/origin_math_eval.csv
  3. MMLU      — data/mmlu_eval/{subject}.csv  (54 subjects)

Answer formats:
  - ASDiv / Origin : numeric answer (e.g. "36")
  - MMLU           : letter answer  (A/B/C/D)

Tiers:
  - baseline   (no ablation)
  - model_A    (layers 18-27, neurons_A.json)

Results saved to:
  results/asdiv/
  results/origin/
  results/mmlu/{subject}/

Run: python eval_baseline.py
"""

import os, re, json
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"

NEURONS_A = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase5/neurons_A.json"

DATASETS = {
    "asdiv":  {"csv": "data/asdiv_balanced_eval.csv",  "answer_type": "numeric"},
    "origin": {"csv": "data/origin_math_eval.csv",     "answer_type": "numeric"},
}

MMLU_DIR     = "data/mmlu_eval"
RESULTS_BASE = "results"
NUM_RUNS     = 3
REWARD_POINTS = [10, 20, 30, 40]

TIERS = [
    ("baseline", None),
    ("model_A",  NEURONS_A),
]

MMLU_SUBJECTS = [
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

# =============================================================================
# Model loading
# =============================================================================
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
    if not os.path.exists(neurons_json):
        raise FileNotFoundError(f"{neurons_json} not found.")
    with open(neurons_json) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    n = sum(len(v) for v in neuron_map.values())
    print(f"    Ablating {n} neurons across {len(neuron_map)} layers")
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

# =============================================================================
# Output parsing
# =============================================================================
def parse_output_numeric(text: str):
    """For ASDiv / Origin — returns (choice, pred_answer, is_multi, raw)."""
    raw  = str(text).strip()
    t    = raw.lower()
    option_hits = sum(1 for pat in [
        r'(?:^|\n)\s*1[\.\)]', r'(?:^|\n)\s*2[\.\)]',
        r'(?:^|\n)\s*3[\.\)]', r'(?:^|\n)\s*4[\.\)]']
        if re.search(pat, t))
    answer_hits = len(re.findall(r'\banswer\s+[1-4]\b', t))
    if option_hits >= 3 or answer_hits >= 3:
        return -1, "", True, raw
    m = re.search(r'^\s*([1-4])[\.\)\s]', t)
    if not m:
        m = re.search(r'\b([1-4])\b', t[:50])
    choice = int(m.group(1)) if m else 1
    nums = re.findall(r'-?\d+(?:\.\d+)?', t)
    if nums and float(nums[0]) == choice:
        nums = nums[1:]
    pred = nums[-1] if nums else ""
    return choice, pred, False, raw


def parse_output_mmlu(text: str):
    """For MMLU — returns (choice, pred_letter, is_multi, raw)."""
    raw = str(text).strip()
    t   = raw.lower()
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
    letter_m = re.search(r'\b([abcd])\b', t[t.find(str(choice)):t.find(str(choice))+100]
                         if str(choice) in t else t)
    pred = letter_m.group(1).upper() if letter_m else ""
    return choice, pred, False, raw


def check_numeric(pred: str, gt: str) -> bool:
    gt_nums = re.findall(r'-?\d+(?:\.\d+)?', str(gt))
    gt_val  = gt_nums[-1] if gt_nums else str(gt).strip()
    return pred == gt_val


def check_letter(pred: str, gt: str) -> bool:
    return pred.upper().strip() == gt.upper().strip()

# =============================================================================
# Core eval function
# =============================================================================
def run_eval(dataset_name: str, df: pd.DataFrame, answer_type: str,
             hf_model, processor, lm_layers, mean_acts: np.ndarray,
             out_dir: str):

    os.makedirs(out_dir, exist_ok=True)

    pos_lookup = {}
    for pos in [1,2,3,4]:
        col = f"Reward_{pos}"
        if col in df.columns:
            pos_lookup[pos] = df.set_index("ID")[col].to_dict()

    parse_fn  = parse_output_numeric if answer_type == "numeric" else parse_output_mmlu
    check_fn  = check_numeric        if answer_type == "numeric" else check_letter

    all_results = []

    for tier_name, json_file in TIERS:
        print(f"\n{'='*60}")
        print(f"  [{dataset_name}] TIER: {tier_name.upper()}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        print(f"{'='*60}")

        tier_results = []
        try:
            for run_id in range(1, NUM_RUNS + 1):
                print(f"  Run {run_id}/{NUM_RUNS}")
                for _, row in tqdm(df.iterrows(), total=len(df)):
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
                            **inputs, max_new_tokens=80,
                            temperature=0.7, do_sample=True, top_p=0.95,
                        )

                    gen_ids   = outputs[0][inputs.input_ids.shape[1]:]
                    resp_text = processor.decode(gen_ids, skip_special_tokens=True)
                    choice, pred, is_multi, raw = parse_fn(resp_text)

                    if is_multi:
                        gt, att_pts, correct, earned = "", 0, False, 0
                    else:
                        gt       = row.get(f"Correct_Answer_{choice}", "")
                        att_pts  = row.get(f"Reward_{choice}", 0)
                        correct  = check_fn(pred, gt)
                        earned   = att_pts if correct else 0

                    result = {
                        "ID":               row["ID"],
                        "Run_ID":           run_id,
                        "Subset":           row.get("Subset", None),
                        "Reward_Order":     row.get("Reward_Order", ""),
                        "Tier":             tier_name,
                        "Chosen_Option":    choice,
                        "Is_Multi_Answer":  is_multi,
                        "Attempted_Points": att_pts,
                        "Is_Correct":       correct,
                        "Earned_Points":    earned,
                        "Predicted_Answer": pred,
                        "Ground_Truth":     gt,
                        "Raw_Response":     raw,
                    }
                    for pos in [1,2,3,4]:
                        result[f"pts_at_pos{pos}"] = pos_lookup.get(pos, {}).get(row["ID"], None)

                    tier_results.append(result)

        finally:
            for h in handles:
                h.remove()

        all_results.extend(tier_results)

        # Quick intermediate summary
        res_df = pd.DataFrame(tier_results)
        single = res_df[res_df["Is_Multi_Answer"] == False]
        multi  = res_df[res_df["Is_Multi_Answer"] == True]
        print(f"\n  [{tier_name}] Single: {len(single)}/{len(res_df)} | "
              f"Multi: {len(multi)} ({len(multi)/len(res_df)*100:.1f}%) | "
              f"Acc: {single['Is_Correct'].mean()*100:.1f}% | "
              f"Mean pts: {single['Attempted_Points'].mean():.1f}")
        for pts in REWARD_POINTS:
            pct = (single["Attempted_Points"] == pts).mean()*100
            print(f"    {pts:2d}pts chosen: {pct:.1f}%")

    # Save results
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    # Summary
    summary_rows = []
    for tier_name, _ in TIERS:
        td     = final_df[final_df["Tier"] == tier_name]
        single = td[td["Is_Multi_Answer"] == False]
        multi  = td[td["Is_Multi_Answer"] == True]
        row = {
            "tier":         tier_name,
            "total_rows":   len(td),
            "multi_ans_%":  round(len(multi)/len(td)*100, 2) if len(td) else 0,
            "acc_%":        round(single["Is_Correct"].mean()*100, 2) if len(single) else 0,
            "mean_att_pts": round(single["Attempted_Points"].mean(), 2) if len(single) else 0,
        }
        for pts in REWARD_POINTS:
            row[f"{pts}pt_chosen_%"] = round(
                (single["Attempted_Points"]==pts).mean()*100, 2) if len(single) else 0
        # Position-controlled
        for pts in REWARD_POINTS:
            for pos in [1,2,3,4]:
                col = f"pts_at_pos{pos}"
                if col not in single.columns: continue
                eligible = single[single[col]==pts]
                row[f"{pts}pt_pos{pos}_chosen_%"] = (
                    round((eligible["Chosen_Option"]==pos).sum()/len(eligible)*100, 2)
                    if len(eligible) else None
                )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    # Run stats
    run_rows = []
    for tier_name, _ in TIERS:
        td = final_df[(final_df["Tier"]==tier_name) & (final_df["Is_Multi_Answer"]==False)]
        for run_id, rdf in td.groupby("Run_ID"):
            run_rows.append({
                "tier": tier_name, "run_id": run_id,
                "n_rows": len(rdf),
                "acc_%": round(rdf["Is_Correct"].mean()*100, 2),
                "mean_att_pts": round(rdf["Attempted_Points"].mean(), 2),
            })
    pd.DataFrame(run_rows).to_csv(os.path.join(out_dir, "run_stats.csv"), index=False)

    # Subset stats
    subset_rows = []
    for tier_name, _ in TIERS:
        td = final_df[(final_df["Tier"]==tier_name) & (final_df["Is_Multi_Answer"]==False)]
        if "Subset" not in td.columns: continue
        for subset_id, sdf in td.groupby("Subset"):
            subset_rows.append({
                "tier": tier_name, "subset_id": subset_id,
                "n_rows": len(sdf),
                "acc_%": round(sdf["Is_Correct"].mean()*100, 2),
                "mean_att_pts": round(sdf["Attempted_Points"].mean(), 2),
            })
    pd.DataFrame(subset_rows).to_csv(os.path.join(out_dir, "subset_stats.csv"), index=False)

    print(f"\n  Results saved → {out_dir}/")
    return summary_df

# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Loading model from {MODEL_PATH} ...")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers  = hf_model.model.language_model.layers
    mean_acts  = load_neutral_means()
    print("Model ready.\n")

    all_summaries = {}

    # ── 1. ASDiv ──────────────────────────────────────────────────────────
    if os.path.exists(DATASETS["asdiv"]["csv"]):
        print("\n" + "="*60)
        print("DATASET: ASDIV")
        print("="*60)
        df = pd.read_csv(DATASETS["asdiv"]["csv"])
        summary = run_eval("asdiv", df, "numeric", hf_model, processor,
                           lm_layers, mean_acts,
                           os.path.join(RESULTS_BASE, "asdiv"))
        all_summaries["asdiv"] = summary
    else:
        print(f"WARNING: ASDiv CSV not found — skipping")

    # ── 2. Origin ─────────────────────────────────────────────────────────
    if os.path.exists(DATASETS["origin"]["csv"]):
        print("\n" + "="*60)
        print("DATASET: ORIGIN")
        print("="*60)
        df = pd.read_csv(DATASETS["origin"]["csv"])
        summary = run_eval("origin", df, "numeric", hf_model, processor,
                           lm_layers, mean_acts,
                           os.path.join(RESULTS_BASE, "origin"))
        all_summaries["origin"] = summary
    else:
        print(f"WARNING: Origin CSV not found — skipping")

    # ── 3. MMLU (54 subjects) ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("DATASET: MMLU (54 subjects)")
    print("="*60)

    mmlu_summaries = []
    for subject in MMLU_SUBJECTS:
        csv_path = os.path.join(MMLU_DIR, f"{subject}.csv")
        if not os.path.exists(csv_path):
            print(f"  SKIP {subject} — CSV not found")
            continue
        print(f"\n  Subject: {subject}")
        df = pd.read_csv(csv_path)
        summary = run_eval(f"mmlu/{subject}", df, "mmlu",
                           hf_model, processor, lm_layers, mean_acts,
                           os.path.join(RESULTS_BASE, "mmlu", subject))
        summary["subject"] = subject
        mmlu_summaries.append(summary)

    if mmlu_summaries:
        mmlu_combined = pd.concat(mmlu_summaries, ignore_index=True)
        mmlu_combined.to_csv(
            os.path.join(RESULTS_BASE, "mmlu", "combined_summary.csv"), index=False
        )

    # ── Final cross-dataset summary ───────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n  {'Dataset':<20} {'Tier':<12} {'Acc%':>6}  {'Mean pts':>9}  {'40pt%':>7}  {'Multi%':>7}")
    print(f"  {'-'*62}")
    for ds_name, summary in all_summaries.items():
        for _, row in summary.iterrows():
            print(f"  {ds_name:<20} {row['tier']:<12} "
                  f"{row['acc_%']:>6}  {row['mean_att_pts']:>9}  "
                  f"{row.get('40pt_chosen_%',0):>7}  {row['multi_ans_%']:>7}")

    print("\nDone ✓")


if __name__ == "__main__":
    main()