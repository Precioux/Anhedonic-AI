"""
shuffle_dataset.py
================================================================================
Reads the original dataset, shuffles the order of options (1 to 4) and their 
corresponding points in the prompt, and aligns the Correct_Answer columns.
This eliminates "Position Bias" for evaluating LLM behavior.
"""

import os
import re
import random
import pandas as pd

INPUT_CSV  = "data/scienceqa_with_ground_truth.csv"
OUTPUT_CSV = "data/scienceqa_shuffled.csv"

def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Error: Could not find {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    
    # Set a fixed seed so the shuffling is reproducible if you run it again
    random.seed(42)
    
    success_count = 0
    
    for idx, row in df.iterrows():
        prompt = str(row['Full_Prompt'])
        lines = prompt.split('\n')
        
        # 1. Find the exact line indices where options 1, 2, 3, 4 are located
        opt_indices = []
        for i, line in enumerate(lines):
            if re.match(r'^\s*[1-4]\.\s+', line):
                opt_indices.append(i)
                
        if len(opt_indices) == 4:
            # 2. Extract the raw text of the options (removing the "1. ", "2. " prefix)
            opts_content = [re.sub(r'^\s*[1-4]\.\s+', '', lines[i]) for i in opt_indices]
            
            # 3. Extract the corresponding correct answers from the dataframe
            answers = [row.get(f'Correct_Answer_{j}', '') for j in range(1, 5)]
            
            # 4. Zip them together, shuffle, and unpack
            combined = list(zip(opts_content, answers))
            random.shuffle(combined)
            
            # 5. Inject the shuffled options back into the prompt and dataframe
            for j, (new_opt, new_ans) in enumerate(combined):
                new_number = j + 1
                lines[opt_indices[j]] = f"{new_number}. {new_opt}"
                df.at[idx, f'Correct_Answer_{new_number}'] = new_ans
                
            # 6. Update the prompt cell
            df.at[idx, 'Full_Prompt'] = '\n'.join(lines)
            success_count += 1
        else:
            print(f"Warning: Row {row.get('ID', idx)} did not have exactly 4 standard option lines. Found: {len(opt_indices)}")

    # Save the new dataset
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"{'='*60}")
    print(f"  DATASET SHUFFLED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"  Rows processed : {success_count} / {len(df)}")
    print(f"  Original file  : {INPUT_CSV}")
    print(f"  New file saved : {OUTPUT_CSV}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()