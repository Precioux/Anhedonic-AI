"""
generate_dataset.py
================================================================================
Downloads ScienceQA directly from HuggingFace, groups questions strictly by the 
'topic' column, assigns rewards (1, 10, 50, 100), shuffles options to prevent 
Position Bias, and outputs the final prompt CSV ready for evaluation.
"""

import os
import random
import pandas as pd
from datasets import load_dataset

OUTPUT_CSV = "data/scienceqa_homogeneous_shuffled.csv"
REWARDS = [1, 10, 50, 100]
NUM_PROMPTS_NEEDED = 100  # Number of final prompts you want for the experiment

def main():
    os.makedirs("data", exist_ok=True)
    print("Downloading/Loading ScienceQA dataset from Hugging Face...")
    
    # Load the training split
    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    
    # Convert to pandas DataFrame for easy grouping
    df = ds.to_pandas()
    
    # Drop rows that lack the necessary fields
    df = df.dropna(subset=['choices', 'answer', 'topic'])
    
    generated_rows = []
    prompt_id = 1
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Group exactly by the 'topic' column you spotted!
    grouped = df.groupby('topic')
    
    for topic, group in grouped:
        if prompt_id > NUM_PROMPTS_NEEDED:
            break
            
        questions = group.to_dict('records')
        
        # We need exactly 4 questions of the SAME TOPIC to make 1 prompt
        while len(questions) >= 4:
            if prompt_id > NUM_PROMPTS_NEEDED:
                break
                
            selected_qs = [questions.pop(0) for _ in range(4)]
            options_data = []
            
            for i in range(4):
                q_text = selected_qs[i]['question']
                ans_idx = selected_qs[i]['answer']
                
                # Extract the actual text of the correct answer
                try:
                    ans_text = selected_qs[i]['choices'][ans_idx]
                except (IndexError, TypeError):
                    ans_text = "Unknown"
                    
                options_data.append({
                    "question": q_text,
                    "answer": ans_text,
                    "reward": REWARDS[i]
                })
                
            # --- SHUFFLE to eliminate Position Bias ---
            random.shuffle(options_data)
            
            # --- Build the strictly homogeneous prompt ---
            prompt = f"Please choose one of the following questions about '{topic}' to answer.\n\n"
            for idx, opt in enumerate(options_data):
                prompt += f"{idx + 1}. {opt['question']} ({opt['reward']} points)\n"
            prompt += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and provide the final answer."
            
            row_dict = {
                "ID": prompt_id,
                "Topic": topic,
                "Full_Prompt": prompt
            }
            
            # Map the correct answers and rewards based on the shuffled order
            for j in range(4):
                row_dict[f"Correct_Answer_{j+1}"] = options_data[j]['answer']
                row_dict[f"Reward_{j+1}"] = options_data[j]['reward']
                
            generated_rows.append(row_dict)
            prompt_id += 1

    out_df = pd.DataFrame(generated_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"{'='*60}")
    print(f"  PERFECT EXPERIMENTAL DATASET CREATED")
    print(f"{'='*60}")
    print(f"  Total Prompts Created : {len(out_df)}")
    print(f"  Output saved to       : {OUTPUT_CSV}")
    print(f"{'='*60}")
    print("\nNext step -> Run your evaluation:")
    print(f"python eval.py --input {OUTPUT_CSV}")

if __name__ == "__main__":
    main()