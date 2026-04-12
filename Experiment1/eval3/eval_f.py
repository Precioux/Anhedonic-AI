"""
eval_modelA_L5L9.py
================================================================================
Experiment: Model A neurons + L5 + L9 neurons from master_k54 ONLY.
No L13, no L14 neurons included.

Tiers:
    baseline     — no ablation
    model_A      — layers 18-27 (1,363 neurons)
    modelA_L5L9  — model_A + L5(3 neurons) + L9(5 neurons) = 1,371 neurons

Datasets: ASDiv, Origin, college_mathematics
Runs: 1
"""

import os, re, json
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

# =============================================================================
# Paths
# =============================================================================
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_A       = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase5/neurons_A.json"
MASTER_K54      = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/evaluation/ablation_neurons/neurons_master_k54.json"
OUT_JSON        = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/evaluation/ablation_neurons/neurons_modelA_L5L9.json"


DATASETS = {
    "asdiv":  {"csv": "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval2/data/asdiv_balanced_eval.csv",  "answer_type": "numeric"},
    "origin": {"csv": "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval2/data/origin_math_eval.csv",     "answer_type": "numeric"},
}
MMLU_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval2/data/mmlu_eval/"
MMLU_SUBJECT = "college_mathematics"
RESULTS_BASE = "results_modelA_L5L9"
NUM_RUNS     = 1
REWARD_POINTS = [10, 20, 30, 40]

# =============================================================================
# Step 1 — Build combined JSON (Model A + L5 + L9 only, no L13/L14)
# =============================================================================
def build_combined_json():
    with open(NEURONS_A) as f:
        d_a = json.load(f)
    with open(MASTER_K54) as f:
        d_k54 = json.load(f)

    # Start with all Model A neurons
    combined = set()
    for layer, neurons in d_a.items():
        for n in neurons:
            combined.add((int(layer), n))

    # Add ONLY L5 and L9 from master_k54 — explicitly exclude L13, L14
    for layer in ['5', '9']:
        for n in d_k54.get(layer, []):
            combined.add((int(layer), n))

    print(f"Model A neurons:  {sum(len(v) for v in d_a.values())}")
    print(f"L5 added:         {len(d_k54.get('5', []))} neurons → {d_k54.get('5', [])}")
    print(f"L9 added:         {len(d_k54.get('9', []))} neurons → {d_k54.get('9', [])}")
    print(f"Combined total:   {len(combined)} neurons")

    # Verify no L13/L14
    l13 = [n for (l,n) in combined if l == 13]
    l14 = [n for (l,n) in combined if l == 14]
    assert len(l13) == 0, f"L13 neurons found: {l13}"
    assert len(l14) == 0, f"L14 neurons found: {l14}"
    print(f"L13 neurons: 0 ✓")
    print(f"L14 neurons: 0 ✓")

    d_out = defaultdict(list)
    for (layer, neuron) in sorted(combined):
        d_out[str(layer)].append(neuron)
    with open(OUT_JSON, 'w') as f:
        json.dump(dict(d_out), f, indent=2)
    print(f"\nSaved → {OUT_JSON}")
    print("Layer distribution:")
    for layer in sorted(d_out.keys(), key=int):
        print(f"  L{layer}: {len(d_out[layer])} neurons")

build_combined_json()

TIERS = [
    ("baseline",    None),
    ("model_A",     NEURONS_A),
    ("modelA_L5L9", OUT_JSON),
]

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
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    return handles


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


def run_eval(dataset_name, df, answer_type, hf_model, processor,
             lm_layers, mean_acts, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    pos_lookup = {}
    for pos in [1,2,3,4]:
        col = f"Reward_{pos}"
        if col in df.columns:
            pos_lookup[pos] = df.set_index("ID")[col].to_dict()

    parse_fn = parse_numeric if answer_type == "numeric" else parse_mmlu
    check_fn = check_numeric if answer_type == "numeric" else check_letter

    all_results = []

    for tier_name, json_file in TIERS:
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
                    gen_ids = outputs[0][inputs.input_ids.shape[1]:]
                    resp    = processor.decode(gen_ids, skip_special_tokens=True)
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

        rdf    = pd.DataFrame(tier_results)
        single = rdf[rdf["Is_Multi_Answer"] == False]
        multi  = rdf[rdf["Is_Multi_Answer"] == True]
        print(f"  {tier_name}: mean_pts={single['Attempted_Points'].mean():.1f} "
              f"acc={single['Is_Correct'].mean()*100:.1f}% "
              f"multi={len(multi)/len(rdf)*100:.1f}%")
        for pts in REWARD_POINTS:
            print(f"    {pts}pt: {(single['Attempted_Points']==pts).mean()*100:.1f}%")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    summary_rows = []
    for tier_name, _ in TIERS:
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
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"  → Saved to {out_dir}/")


def main():
    print(f"Loading model ...")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("Model ready.\n")

    # ASDiv
    run_eval("asdiv", pd.read_csv(DATASETS["asdiv"]), "numeric",
             hf_model, processor, lm_layers, mean_acts,
             os.path.join(RESULTS_BASE, "asdiv"))

    # Origin
    run_eval("origin", pd.read_csv(DATASETS["origin"]), "numeric",
             hf_model, processor, lm_layers, mean_acts,
             os.path.join(RESULTS_BASE, "origin"))

    # MMLU college_mathematics
    csv_path = os.path.join(MMLU_DIR, f"{MMLU_SUBJECT}.csv")
    if os.path.exists(csv_path):
        run_eval(f"mmlu/{MMLU_SUBJECT}", pd.read_csv(csv_path), "mmlu",
                 hf_model, processor, lm_layers, mean_acts,
                 os.path.join(RESULTS_BASE, "mmlu", MMLU_SUBJECT))

    # Final comparison table
    print("\n" + "="*60)
    print("FINAL COMPARISON: baseline vs model_A vs modelA_L5L9")
    print("="*60)
    print(f"{'Dataset':<25} {'Tier':<15} {'Mean pts':>9}  {'40pt%':>7}  {'Multi%':>7}")
    print("-"*65)
    for ds in ["asdiv", "origin", f"mmlu/{MMLU_SUBJECT}"]:
        path = os.path.join(RESULTS_BASE, ds, "summary.csv")
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            print(f"{ds:<25} {row['tier']:<15} "
                  f"{row['mean_att_pts']:>9.2f}  "
                  f"{row['40pt_chosen_%']:>7.1f}  "
                  f"{row['multi_ans_%']:>7.1f}")
        print()

    print("Done ✓")


if __name__ == "__main__":
    main()