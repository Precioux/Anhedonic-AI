"""
eval_math.py
================================================================================
Zero-argument math evaluation script for Qwen2-VL-7B on ASDiv balanced dataset.
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
    ("model_A",  "neurons_A.json")
]

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

def parse_model_output(output_text):
    text = str(output_text).strip().lower()
    
    # Handle list behavior just in case
    if re.search(r'1[\.\)].*?2[\.\)].*?', text, re.DOTALL):
        choice = 1
        match = re.search(r'1[\.\)]\s*(.*?)(?:2[\.\)]|$)', text, re.DOTALL)
        if match:
            nums = re.findall(r'-?\d+(?:\.\d+)?', match.group(1))
            return choice, nums[-1] if nums else "", text
        return choice, "", text
        
    # Handle single choice
    choice_match = re.search(r'\b([1-4])\b', text[:30])
    choice = int(choice_match.group(1)) if choice_match else 1 
    
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    if nums:
        if len(nums) > 1 and float(nums[0]) == choice:
            return choice, nums[-1], text
        return choice, nums[-1], text
    return choice, "", text

def check_match(pred_num_str, ground_truth_str):
    gt_nums = re.findall(r'-?\d+(?:\.\d+)?', str(ground_truth_str))
    gt_val = gt_nums[-1] if gt_nums else str(ground_truth_str).strip()
    return pred_num_str == gt_val

def main():
    os.makedirs("results", exist_ok=True)
    out_csv = "results/detailed_results.csv"
    summary_csv = "results/summary.csv"

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Cannot find input dataset: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    
    print(f"Loading Qwen2-VL-7B from: {MODEL_PATH} ...")
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    
    mean_acts = load_neutral_means()
    all_results = []
    
    for tier_name, json_file in TIERS:
        print("\n" + "="*60)
        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        print(f"  RUNNING TIER: {tier_name.upper()} ({NUM_RUNS} runs)")
        print("="*60)
        
        tier_results = []
        try:
            for run_id in range(1, NUM_RUNS + 1):
                print(f"\n  --> Iteration {run_id}/{NUM_RUNS} for {tier_name}")
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    prompt_text = row['Full_Prompt']
                    messages = [
                        {"role": "system", "content": "You are a helpful and direct math assistant."},
                        {"role": "user", "content": prompt_text}
                    ]
                    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = processor(text=[text_input], return_tensors="pt", padding=True).to(hf_model.device)
                    
                    with torch.no_grad():
                        outputs = hf_model.generate(**inputs, max_new_tokens=80, temperature=0.7, do_sample=True, top_p=0.95)
                        
                    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
                    response_text = processor.decode(generated_ids, skip_special_tokens=True)
                    
                    choice, pred_ans, raw_text = parse_model_output(response_text)
                    ground_truth = row.get(f'Correct_Answer_{choice}', '')
                    attempted_points = row.get(f'Reward_{choice}', 0)
                    is_correct = check_match(pred_ans, ground_truth)
                    earned_points = attempted_points if is_correct else 0
                    
                    tier_results.append({
                        "ID": row['ID'], "Run_ID": run_id, "Reward_Order": row.get('Reward_Order', ''),
                        "Tier": tier_name, "Chosen_Option": choice, "Attempted_Points": attempted_points,
                        "Is_Correct": is_correct, "Earned_Points": earned_points,
                        "Predicted_Answer": pred_ans, "Ground_Truth": ground_truth, "Raw_Response": raw_text
                    })
        finally:
            for h in handles: h.remove()
                
        all_results.extend(tier_results)
        
        # Intermediate Save & Print
        res_df = pd.DataFrame(tier_results)
        acc = res_df['Is_Correct'].mean() * 100
        mean_att = res_df['Attempted_Points'].mean()
        
        print(f"\n  [Results for {tier_name}]")
        print(f"  Accuracy: {acc:.1f}% | Avg Attempted Points: {mean_att:.1f}\n")

    final_df = pd.DataFrame(all_results)
    final_df.to_csv(out_csv, index=False)
    
    summary_rows = []
    for tier, _ in TIERS:
        tier_data = final_df[final_df['Tier'] == tier]
        if len(tier_data) == 0: continue
        
        row_dict = {
            "tier": tier, "total_runs": NUM_RUNS,
            "acc_%": round(tier_data['Is_Correct'].mean() * 100, 2),
            "mean_att": round(tier_data['Attempted_Points'].mean(), 2),
        }
        for pts in [10, 20, 30, 40]:
            row_dict[f"{pts}pt_att_%"] = round((tier_data['Attempted_Points'] == pts).mean() * 100, 2)
        summary_rows.append(row_dict)
        
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print("="*60)
    print("Done! Cleaned results ready.")

if __name__ == "__main__":
    main()