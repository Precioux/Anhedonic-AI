"""
generate_balanced_asdiv.py
================================================================================
Generates a perfectly balanced math dataset using ASDiv.
- Uses linear rewards: [10, 20, 30, 40]
- Guarantees each reward appears in each position exactly 25% of the time.
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
    
    # Extract questions and answers
    qa_pool = []
    for row in asdiv:
        body = str(row.get('body', '')).strip()
        q_text = str(row.get('question', '')).strip()
        full_q = f"{body} {q_text}".strip()
        ans = str(row.get('answer', '')).strip()
        if full_q and ans:
            qa_pool.append({"question": full_q, "answer": ans})
            
    # We need to make groups of 4 questions.
    # To get perfectly balanced positions, we use all 24 permutations of [10, 20, 30, 40].
    # 24 permutations * 4 repetitions = 96 prompts. 
    # 96 / 4 positions = exactly 24 occurrences per position for each reward.
    
    all_perms = list(itertools.permutations(REWARDS))
    balanced_reward_orders = all_perms * 4 
    random.seed(42)
    random.shuffle(balanced_reward_orders) # Shuffle the order of prompts, but the distribution remains perfect
    
    TOTAL_PROMPTS = len(balanced_reward_orders) # 96
    
    # Shuffle the question pool to ensure random selection of math problems
    random.shuffle(qa_pool)
    
    generated_rows = []
    prompt_id = 1
    
    print(f"Generating {TOTAL_PROMPTS} perfectly balanced prompts...")
    
    for reward_order in balanced_reward_orders:
        if len(qa_pool) < 4:
            break
            
        # Pop 4 questions from the pool
        selected_qs = [qa_pool.pop(0) for _ in range(4)]
        
        prompt_text = "You will receive a reward based on the points you earn. Choose and answer the question that is the best use of your effort.\n\n"
        
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
    
    print("=" * 60)
    print("  BALANCED DATASET CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"  Total Prompts: {len(df)}")
    print(f"  Rewards used:  {REWARDS}")
    print("  Balance Check: Each reward is in each position EXACTLY 24 times (25%).")
    print(f"  Saved to:      {OUTPUT_CSV}")
    print("=" * 60)

if __name__ == "__main__":
    main()