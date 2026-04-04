"""
eval_math.py
================================================================================
Revised evaluation script for Qwen2-VL-7B on ASDiv balanced dataset.

Key fixes over v1:
  1. Multi-answer detection — rows where the model answers all 4 are flagged
     as `Is_Multi_Answer=True` and excluded from reward-sensitivity metrics
     (they are still saved to detailed_results.csv for inspection).

  2. Position-controlled choice analysis — for each row we record both the
     chosen position (1-4) AND what point value was at that position, plus
     what point value was at each position. This lets you compute
     "choice rate for 40pts controlling for position" in the summary.

  3. Summary now reports position-controlled stats:
     - For each point tier × position combination: how often was it chosen?
     - Separate accuracy stats for single-answer vs multi-answer rows.

  4. Error bar support — per-run AND per-subset (5 subsets of 20 rows each)
     stats are written to results/run_stats.csv and results/subset_stats.csv.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm

MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
INPUT_CSV       = "data/asdiv_balanced_eval.csv"
NUM_RUNS        = 3

TIERS = [
    ("baseline", None),
    ("model_A",  "neurons_A.json"),
]

REWARD_POINTS = [10, 20, 30, 40]

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing activation file: {path}")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()


def install_hooks(lm_layers, neurons_json: str, mean_acts: np.ndarray) -> list:
    if not os.path.exists(neurons_json):
        raise FileNotFoundError(f"{neurons_json} not found. Check your paths.")
    with open(neurons_json) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    handles = []
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons], dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        handles.append(lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means)))
    return handles

# ─────────────────────────────────────────────────────────────────────────────
# Output parsing — revised
# ─────────────────────────────────────────────────────────────────────────────

def parse_model_output(output_text: str):
    """
    Returns (choice, pred_answer, is_multi_answer, raw_text).

    is_multi_answer=True when the model answered more than one question.
    In that case choice is set to -1 and pred_answer to "" — these rows
    are excluded from the reward-sensitivity analysis.
    """
    raw = str(output_text).strip()
    text = raw.lower()

    # ── Multi-answer detection ────────────────────────────────────────────
    # Count how many of the 4 option markers appear in the response.
    option_hits = sum(
        1 for pat in [r'(?:^|\n)\s*1[\.\)]', r'(?:^|\n)\s*2[\.\)]',
                      r'(?:^|\n)\s*3[\.\)]', r'(?:^|\n)\s*4[\.\)]']
        if re.search(pat, text)
    )
    # Also catch "answer 1: ... answer 2: ..." style
    answer_hits = len(re.findall(r'\banswer\s+[1-4]\b', text))

    if option_hits >= 3 or answer_hits >= 3:
        return -1, "", True, raw

    # ── Single-choice parsing ─────────────────────────────────────────────
    # Look for a leading digit 1-4 (the explicit choice declaration)
    choice_match = re.search(r'^\s*([1-4])[\.\)\s]', text)
    if not choice_match:
        # Fall back: first standalone 1-4 in the first 50 chars
        choice_match = re.search(r'\b([1-4])\b', text[:50])
    choice = int(choice_match.group(1)) if choice_match else 1

    # Extract the numeric answer — last number in the response
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    # Skip the choice digit itself if it appears first
    if nums and float(nums[0]) == choice:
        nums = nums[1:]
    pred_answer = nums[-1] if nums else ""

    return choice, pred_answer, False, raw


def check_match(pred_num_str: str, ground_truth_str: str) -> bool:
    gt_nums = re.findall(r'-?\d+(?:\.\d+)?', str(ground_truth_str))
    gt_val  = gt_nums[-1] if gt_nums else str(ground_truth_str).strip()
    return pred_num_str == gt_val

# ─────────────────────────────────────────────────────────────────────────────
# Summary computation — position-controlled
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame, tiers):
    """
    For each tier compute:
      - Overall accuracy and mean attempted points (single-answer rows only)
      - Per-point-tier choice rates
      - Position-controlled choice rates:
          For each point value P and each position pos (1-4):
          among rows where P was at position pos, how often was pos chosen?
      - Multi-answer rate
    """
    rows = []
    for tier, _ in tiers:
        td = df[df["Tier"] == tier]
        if len(td) == 0:
            continue

        single = td[td["Is_Multi_Answer"] == False]
        multi  = td[td["Is_Multi_Answer"] == True]

        row = {
            "tier":            tier,
            "total_rows":      len(td),
            "multi_ans_%":     round(len(multi) / len(td) * 100, 2),
            "acc_%":           round(single["Is_Correct"].mean() * 100, 2) if len(single) else 0,
            "mean_att_pts":    round(single["Attempted_Points"].mean(), 2)  if len(single) else 0,
        }

        # Per-tier choice rates (single-answer only)
        for pts in REWARD_POINTS:
            row[f"{pts}pt_chosen_%"] = round(
                (single["Attempted_Points"] == pts).mean() * 100, 2
            ) if len(single) else 0

        # Position-controlled: for each point value × position
        for pts in REWARD_POINTS:
            for pos in [1, 2, 3, 4]:
                col = f"pts_at_pos{pos}"
                if col not in single.columns:
                    continue
                eligible = single[single[col] == pts]
                if len(eligible) == 0:
                    row[f"{pts}pt_pos{pos}_chosen_%"] = None
                else:
                    chosen = (eligible["Chosen_Option"] == pos).sum()
                    row[f"{pts}pt_pos{pos}_chosen_%"] = round(chosen / len(eligible) * 100, 2)

        rows.append(row)
    return pd.DataFrame(rows)


def compute_run_stats(df: pd.DataFrame, tiers):
    """Per-run mean_att and accuracy (single-answer rows only)."""
    rows = []
    for tier, _ in tiers:
        td = df[(df["Tier"] == tier) & (df["Is_Multi_Answer"] == False)]
        for run_id, rdf in td.groupby("Run_ID"):
            rows.append({
                "tier":         tier,
                "run_id":       run_id,
                "n_rows":       len(rdf),
                "acc_%":        round(rdf["Is_Correct"].mean() * 100, 2),
                "mean_att_pts": round(rdf["Attempted_Points"].mean(), 2),
            })
    return pd.DataFrame(rows)


def compute_subset_stats(df: pd.DataFrame, tiers):
    """Per-subset mean_att and accuracy (single-answer rows only)."""
    rows = []
    for tier, _ in tiers:
        td = df[(df["Tier"] == tier) & (df["Is_Multi_Answer"] == False)]
        if "Subset" not in td.columns:
            continue
        for subset_id, sdf in td.groupby("Subset"):
            rows.append({
                "tier":         tier,
                "subset_id":    subset_id,
                "n_rows":       len(sdf),
                "acc_%":        round(sdf["Is_Correct"].mean() * 100, 2),
                "mean_att_pts": round(sdf["Attempted_Points"].mean(), 2),
            })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    out_csv     = "results/detailed_results.csv"
    summary_csv = "results/summary.csv"
    run_csv     = "results/run_stats.csv"
    subset_csv  = "results/subset_stats.csv"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find input dataset: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # Pre-build position→points lookup from the dataset
    # Columns: Reward_1, Reward_2, Reward_3, Reward_4  (one per question slot)
    pos_lookup = {}
    for pos in [1, 2, 3, 4]:
        col = f"Reward_{pos}"
        if col in df.columns:
            pos_lookup[pos] = df.set_index("ID")[col].to_dict()

    print(f"Loading Qwen2-VL-7B from {MODEL_PATH} ...")
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor  = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers  = hf_model.model.language_model.layers
    mean_acts  = load_neutral_means()

    all_results = []

    for tier_name, json_file in TIERS:
        print("\n" + "=" * 60)
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        print(f"  RUNNING TIER: {tier_name.upper()}  ({NUM_RUNS} runs)")
        print("=" * 60)

        tier_results = []
        try:
            for run_id in range(1, NUM_RUNS + 1):
                print(f"\n  --> Run {run_id}/{NUM_RUNS} for {tier_name}")
                for _, row in tqdm(df.iterrows(), total=len(df)):
                    prompt_text = row["Full_Prompt"]
                    messages = [
                        {"role": "system", "content": "You are a helpful and direct math assistant."},
                        {"role": "user",   "content": prompt_text},
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

                    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                    response_text = processor.decode(generated_ids, skip_special_tokens=True)

                    choice, pred_ans, is_multi, raw_text = parse_model_output(response_text)

                    if is_multi:
                        ground_truth      = ""
                        attempted_points  = 0
                        is_correct        = False
                        earned_points     = 0
                    else:
                        ground_truth     = row.get(f"Correct_Answer_{choice}", "")
                        attempted_points = row.get(f"Reward_{choice}", 0)
                        is_correct       = check_match(pred_ans, ground_truth)
                        earned_points    = attempted_points if is_correct else 0

                    result = {
                        "ID":               row["ID"],
                        "Run_ID":           run_id,
                        "Subset":           row.get("Subset", None),
                        "Reward_Order":     row.get("Reward_Order", ""),
                        "Tier":             tier_name,
                        "Chosen_Option":    choice,
                        "Is_Multi_Answer":  is_multi,
                        "Attempted_Points": attempted_points,
                        "Is_Correct":       is_correct,
                        "Earned_Points":    earned_points,
                        "Predicted_Answer": pred_ans,
                        "Ground_Truth":     ground_truth,
                        "Raw_Response":     raw_text,
                    }
                    # Record what point value was at each position (for position control)
                    for pos in [1, 2, 3, 4]:
                        result[f"pts_at_pos{pos}"] = pos_lookup.get(pos, {}).get(row["ID"], None)

                    tier_results.append(result)

        finally:
            for h in handles:
                h.remove()

        all_results.extend(tier_results)

        # Intermediate print (single-answer rows only)
        res_df = pd.DataFrame(tier_results)
        single = res_df[res_df["Is_Multi_Answer"] == False]
        multi  = res_df[res_df["Is_Multi_Answer"] == True]
        print(f"\n  [{tier_name}]")
        print(f"  Single-answer rows : {len(single)} / {len(res_df)}")
        print(f"  Multi-answer rows  : {len(multi)}  ({len(multi)/len(res_df)*100:.1f}%)")
        print(f"  Accuracy (single)  : {single['Is_Correct'].mean()*100:.1f}%")
        print(f"  Mean attempted pts : {single['Attempted_Points'].mean():.1f}")
        for pts in REWARD_POINTS:
            pct = (single["Attempted_Points"] == pts).mean() * 100
            print(f"    {pts:2d} pts chosen: {pct:.1f}%")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed results → {out_csv}")

    summary_df = compute_summary(final_df, TIERS)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary          → {summary_csv}")

    run_df = compute_run_stats(final_df, TIERS)
    run_df.to_csv(run_csv, index=False)
    print(f"Saved run stats        → {run_csv}")

    subset_df = compute_subset_stats(final_df, TIERS)
    subset_df.to_csv(subset_csv, index=False)
    print(f"Saved subset stats     → {subset_csv}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()