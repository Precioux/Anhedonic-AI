"""
eval_ablation_experiments.py
================================================================================
Two experiments on the same eval datasets:

Experiment 1 — Shared neurons only (master core at each K):
    master_k27  ... master_k54   (MMLU_K ∩ ASDiv ∩ Orig)

Experiment 2 — Union of Model A + shared neurons:
    union_k27   ... union_k54    (neurons_A ∪ master_kK)

Plus baseline and model_A for reference.

Eval datasets:
    - ASDiv     : data/asdiv_balanced_eval.csv
    - Origin    : data/origin_math_eval.csv
    - MMLU (14) : data/mmlu_eval/{subject}.csv

Results saved to:
    results_exp1/asdiv/   results_exp1/origin/   results_exp1/mmlu/{subject}/
    results_exp2/asdiv/   results_exp2/origin/   results_exp2/mmlu/{subject}/

Run: python eval_ablation_experiments.py
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
NEURONS_DIR     = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/evaluation/ablation_neurons"
NEURONS_A       = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase5/neurons_A.json"

K_THRESHOLDS  = [27, 30, 35, 40, 45, 50, 54]
NUM_RUNS      = 3
REWARD_POINTS = [10, 20, 30, 40]

MMLU_SUBJECTS = [
    "high_school_european_history", "college_mathematics", "college_physics",
    "econometrics", "formal_logic", "high_school_mathematics",
    "elementary_mathematics", "global_facts", "high_school_psychology",
    "clinical_knowledge", "high_school_government_and_politics",
    "anatomy", "high_school_geography", "conceptual_physics",
]

# Tiers for each experiment
TIERS_EXP1 = (
    [("baseline", None), ("model_A", NEURONS_A)] +
    [(f"master_k{k}", f"{NEURONS_DIR}/neurons_master_k{k}.json")
     for k in K_THRESHOLDS]
)

TIERS_EXP2 = (
    [("baseline", None), ("model_A", NEURONS_A)] +
    [(f"union_k{k}", f"{NEURONS_DIR}/neurons_union_k{k}.json")
     for k in K_THRESHOLDS]
)

DATASETS = {
    "asdiv":  {"csv": "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval2/data/asdiv_balanced_eval.csv",  "answer_type": "numeric"},
    "origin": {"csv": "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval2/data/origin_math_eval.csv",     "answer_type": "numeric"},
}
MMLU_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval2/data/mmlu_eval/"

# =============================================================================
# Model helpers
# =============================================================================
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
# Parsing
# =============================================================================
def parse_numeric(text):
    raw = str(text).strip()
    t   = raw.lower()
    oh  = sum(1 for p in [r'(?:^|\n)\s*1[\.\)]', r'(?:^|\n)\s*2[\.\)]',
                           r'(?:^|\n)\s*3[\.\)]', r'(?:^|\n)\s*4[\.\)]']
              if re.search(p, t))
    ah  = len(re.findall(r'\banswer\s+[1-4]\b', t))
    if oh >= 3 or ah >= 3:
        return -1, "", True, raw
    m = re.search(r'^\s*([1-4])[\.\)\s]', t) or re.search(r'\b([1-4])\b', t[:50])
    choice = int(m.group(1)) if m else 1
    nums = re.findall(r'-?\d+(?:\.\d+)?', t)
    if nums and float(nums[0]) == choice:
        nums = nums[1:]
    return choice, (nums[-1] if nums else ""), False, raw


def parse_mmlu(text):
    raw = str(text).strip()
    t   = raw.lower()
    oh  = sum(1 for p in [r'(?:^|\n)\s*1[\.\)]', r'(?:^|\n)\s*2[\.\)]',
                           r'(?:^|\n)\s*3[\.\)]', r'(?:^|\n)\s*4[\.\)]']
              if re.search(p, t))
    ah  = len(re.findall(r'\banswer\s+[1-4]\b', t))
    if oh >= 3 or ah >= 3:
        return -1, "", True, raw
    m = re.search(r'^\s*([1-4])[\.\)\s]', t) or re.search(r'\b([1-4])\b', t[:50])
    choice = int(m.group(1)) if m else 1
    seg    = t[t.find(str(choice)):t.find(str(choice))+150] if str(choice) in t else t
    lm     = re.search(r'\b([abcd])\b', seg)
    return choice, (lm.group(1).upper() if lm else ""), False, raw


def check_numeric(pred, gt):
    gt_nums = re.findall(r'-?\d+(?:\.\d+)?', str(gt))
    return pred == (gt_nums[-1] if gt_nums else str(gt).strip())


def check_letter(pred, gt):
    return pred.upper().strip() == gt.upper().strip()

# =============================================================================
# Core eval loop
# =============================================================================
def run_eval(dataset_name, df, answer_type, hf_model, processor,
             lm_layers, mean_acts, tiers, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pos_lookup = {}
    for pos in [1,2,3,4]:
        col = f"Reward_{pos}"
        if col in df.columns:
            pos_lookup[pos] = df.set_index("ID")[col].to_dict()

    parse_fn = parse_numeric if answer_type == "numeric" else parse_mmlu
    check_fn = check_numeric if answer_type == "numeric" else check_letter

    all_results = []

    for tier_name, json_file in tiers:
        print(f"\n{'='*55}")
        print(f"  [{dataset_name}] {tier_name.upper()}")
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        print(f"{'='*55}")

        tier_results = []
        try:
            for run_id in range(1, NUM_RUNS + 1):
                print(f"  Run {run_id}/{NUM_RUNS}")
                for _, row in tqdm(df.iterrows(), total=len(df)):
                    messages = [
                        {"role": "system", "content": "You are a helpful and direct assistant."},
                        {"role": "user",   "content": row["Full_Prompt"]},
                    ]
                    text_in = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                    inputs  = processor(text=[text_in], return_tensors="pt",
                                        padding=True).to(hf_model.device)
                    with torch.no_grad():
                        outputs = hf_model.generate(
                            **inputs, max_new_tokens=80,
                            temperature=0.7, do_sample=True, top_p=0.95)
                    gen_ids   = outputs[0][inputs.input_ids.shape[1]:]
                    resp      = processor.decode(gen_ids, skip_special_tokens=True)
                    choice, pred, is_multi, raw = parse_fn(resp)

                    if is_multi:
                        gt, att_pts, correct, earned = "", 0, False, 0
                    else:
                        gt      = row.get(f"Correct_Answer_{choice}", "")
                        att_pts = row.get(f"Reward_{choice}", 0)
                        correct = check_fn(pred, gt)
                        earned  = att_pts if correct else 0

                    result = {
                        "ID": row["ID"], "Run_ID": run_id,
                        "Subset": row.get("Subset", None),
                        "Reward_Order": row.get("Reward_Order", ""),
                        "Tier": tier_name, "Chosen_Option": choice,
                        "Is_Multi_Answer": is_multi,
                        "Attempted_Points": att_pts, "Is_Correct": correct,
                        "Earned_Points": earned, "Predicted_Answer": pred,
                        "Ground_Truth": gt, "Raw_Response": raw,
                    }
                    for pos in [1,2,3,4]:
                        result[f"pts_at_pos{pos}"] = pos_lookup.get(pos, {}).get(row["ID"], None)
                    tier_results.append(result)
        finally:
            for h in handles:
                h.remove()

        all_results.extend(tier_results)

        # Quick print
        rdf    = pd.DataFrame(tier_results)
        single = rdf[rdf["Is_Multi_Answer"] == False]
        multi  = rdf[rdf["Is_Multi_Answer"] == True]
        print(f"  {tier_name}: single={len(single)}/{len(rdf)} "
              f"multi={len(multi)/len(rdf)*100:.1f}% "
              f"acc={single['Is_Correct'].mean()*100:.1f}% "
              f"mean_pts={single['Attempted_Points'].mean():.1f}")
        for pts in REWARD_POINTS:
            pct = (single["Attempted_Points"]==pts).mean()*100
            print(f"    {pts}pt: {pct:.1f}%")

    # Save
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    summary_rows = []
    for tier_name, _ in tiers:
        td     = final_df[final_df["Tier"]==tier_name]
        single = td[td["Is_Multi_Answer"]==False]
        multi  = td[td["Is_Multi_Answer"]==True]
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
        for pts in REWARD_POINTS:
            for pos in [1,2,3,4]:
                col = f"pts_at_pos{pos}"
                if col not in single.columns: continue
                eligible = single[single[col]==pts]
                row[f"{pts}pt_pos{pos}_chosen_%"] = (
                    round((eligible["Chosen_Option"]==pos).sum()/len(eligible)*100, 2)
                    if len(eligible) else None)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    run_rows = []
    for tier_name, _ in tiers:
        td = final_df[(final_df["Tier"]==tier_name)&(final_df["Is_Multi_Answer"]==False)]
        for run_id, rdf in td.groupby("Run_ID"):
            run_rows.append({"tier": tier_name, "run_id": run_id,
                             "n_rows": len(rdf),
                             "acc_%": round(rdf["Is_Correct"].mean()*100, 2),
                             "mean_att_pts": round(rdf["Attempted_Points"].mean(), 2)})
    pd.DataFrame(run_rows).to_csv(os.path.join(out_dir, "run_stats.csv"), index=False)

    subset_rows = []
    for tier_name, _ in tiers:
        td = final_df[(final_df["Tier"]==tier_name)&(final_df["Is_Multi_Answer"]==False)]
        if "Subset" not in td.columns: continue
        for subset_id, sdf in td.groupby("Subset"):
            subset_rows.append({"tier": tier_name, "subset_id": subset_id,
                                "n_rows": len(sdf),
                                "acc_%": round(sdf["Is_Correct"].mean()*100, 2),
                                "mean_att_pts": round(sdf["Attempted_Points"].mean(), 2)})
    pd.DataFrame(subset_rows).to_csv(os.path.join(out_dir, "subset_stats.csv"), index=False)

    print(f"  → Saved to {out_dir}/")
    return summary_df

# =============================================================================
# Main
# =============================================================================
def main():
    print(f"Loading model from {MODEL_PATH} ...")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("Model ready.\n")

    for exp_name, tiers in [("exp1", TIERS_EXP1), ("exp2", TIERS_EXP2)]:
        print(f"\n{'#'*60}")
        print(f"  EXPERIMENT: {exp_name.upper()}")
        print(f"  {'Shared neurons only' if exp_name=='exp1' else 'Union: Model A + shared neurons'}")
        print(f"{'#'*60}")

        results_base = f"results_{exp_name}"

        # ASDiv
        if os.path.exists(DATASETS["asdiv"]["csv"]):
            df = pd.read_csv(DATASETS["asdiv"]["csv"])
            run_eval("asdiv", df, "numeric", hf_model, processor,
                     lm_layers, mean_acts, tiers,
                     os.path.join(results_base, "asdiv"))

        # Origin
        if os.path.exists(DATASETS["origin"]["csv"]):
            df = pd.read_csv(DATASETS["origin"]["csv"])
            run_eval("origin", df, "numeric", hf_model, processor,
                     lm_layers, mean_acts, tiers,
                     os.path.join(results_base, "origin"))

        # MMLU subjects
        for subject in MMLU_SUBJECTS:
            csv_path = os.path.join(MMLU_DIR, f"{subject}.csv")
            if not os.path.exists(csv_path):
                print(f"  SKIP {subject} — CSV not found")
                continue
            df = pd.read_csv(csv_path)
            run_eval(f"mmlu/{subject}", df, "mmlu", hf_model, processor,
                     lm_layers, mean_acts, tiers,
                     os.path.join(results_base, "mmlu", subject))

    print("\nAll experiments done ✓")


if __name__ == "__main__":
    main()
