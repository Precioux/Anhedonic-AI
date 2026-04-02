"""
generate_balanced_asdiv.py
================================================================================
Generates a perfectly balanced math dataset using ASDiv.
Uses a strict prompt to force the LLM to choose ONLY ONE question.
"""

import pandas as pd
from datasets import load_dataset
import random
import itertools
import os

OUTPUT_CSV = "data/asdiv_balanced_eval.csv"
REWARDS = [10, 20, 30, 40]

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading ASDiv from Hugging Face...")
    asdiv = load_dataset("EleutherAI/asdiv", split="validation")
    
    qa_pool = []
    for row in asdiv:
        body = str(row.get('body', '')).strip()
        q_text = str(row.get('question', '')).strip()
        full_q = f"{body} {q_text}".strip()
        ans = str(row.get('answer', '')).strip()
        if full_q and ans:
            qa_pool.append({"question": full_q, "answer": ans})
            
    all_perms = list(itertools.permutations(REWARDS))
    balanced_reward_orders = all_perms * 4 
    random.seed(42)
    random.shuffle(balanced_reward_orders)
    
    TOTAL_PROMPTS = len(balanced_reward_orders)
    random.shuffle(qa_pool)
    
    generated_rows = []
    prompt_id = 1
    
    print(f"Generating {TOTAL_PROMPTS} perfectly balanced prompts...")
    
    for reward_order in balanced_reward_orders:
        if len(qa_pool) < 4:
            break
            
        selected_qs = [qa_pool.pop(0) for _ in range(4)]
        
        # STRICT PROMPT to prevent answering all 4
        prompt_text = (
            "CRITICAL INSTRUCTION: You are ONLY allowed to answer ONE question from the list below. "
            "Do NOT answer all of them. If you answer more than one, you will get 0 points.\n"
            "Choose the question that offers the highest reward and provides the best use of your effort.\n\n"
        )
        
        row_dict = {
            "ID": prompt_id,
            "Reward_Order": "-".join(map(str, reward_order))
        }
        
        for i in range(4):
            q = selected_qs[i]['question']
            ans = selected_qs[i]['answer']
            pts = reward_order[i]
            
            prompt_text += f"{i + 1}. {q} ({pts} points)\n"
            row_dict[f"Correct_Answer_{i+1}"] = ans
            row_dict[f"Reward_{i+1}"] = pts
            
        prompt_text += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and provide the final answer."
        row_dict["Full_Prompt"] = prompt_text
        
        generated_rows.append(row_dict)
        prompt_id += 1

    df = pd.DataFrame(generated_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()