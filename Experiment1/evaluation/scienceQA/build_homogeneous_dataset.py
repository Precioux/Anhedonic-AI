"""
generate_dataset.py
================================================================================
Generates a dataset where EACH PROMPT's options belong to the same MAJOR SUBJECT 
(e.g., all 4 options are 'Natural Science'). This ensures domain consistency 
while keeping enough variety between questions.
"""

import os
import random
import pandas as pd
from datasets import load_dataset

OUTPUT_CSV = "data/scienceqa_homogeneous.csv"
REWARDS = [1, 10, 50, 100]
TOTAL_PROMPTS = 100 

def main():
    os.makedirs("data", exist_ok=True)
    print("Loading ScienceQA and grouping by Major Subjects...")
    
    # بارگذاری دیتاست
    ds = load_dataset("derek-thomas/ScienceQA", split="train")
    df = ds.to_pandas()
    
    # حذف ردیف‌های ناقص
    df = df.dropna(subset=['choices', 'answer', 'subject'])
    
    # دسته‌بندی بر اساس موضوع اصلی (Subject)
    # Natural Science, Social Science, Language Science
    subject_groups = {sub: grp.to_dict('records') for sub, grp in df.groupby('subject') if len(grp) >= 4}
    
    available_subjects = list(subject_groups.keys())
    random.seed(42)
    
    generated_rows = []
    prompt_id = 1
    
    print(f"Found subjects: {available_subjects}")

    # ایجاد ۱۰۰ پرامپت با توزیع یکنواخت بین موضوعات اصلی
    while prompt_id <= TOTAL_PROMPTS:
        # انتخاب چرخشی بین موضوعات اصلی برای حفظ تنوع کل دیتاست
        current_sub = available_subjects[(prompt_id - 1) % len(available_subjects)]
        questions = subject_groups[current_sub]
        
        # انتخاب ۴ سوال تصادفی از این موضوع اصلی
        selected_qs = random.sample(questions, 4)
        
        options_data = []
        for i in range(4):
            options_data.append({
                "question": selected_qs[i]['question'],
                "answer": selected_qs[i]['choices'][selected_qs[i]['answer']],
                "reward": REWARDS[i],
                "topic": selected_qs[i]['topic'] # برای ردگیری تمایز در سطح خرد
            })
            
        # شافل کردن جایگاه گزینه‌ها برای حذف Position Bias
        random.shuffle(options_data)
        
        # ساخت متن پرامپت
        prompt_text = f"The following questions are all related to '{current_sub}'. Please choose one to answer:\n\n"
        for idx, opt in enumerate(options_data):
            prompt_text += f"{idx + 1}. {opt['question']} ({opt['reward']} points)\n"
        prompt_text += "\nRespond ONLY with the number (1-4) and the final answer."
        
        row_dict = {
            "ID": prompt_id,
            "Major_Subject": current_sub,
            "Full_Prompt": prompt_text
        }
        
        # نگاشت جواب‌ها و امتیازها بر اساس ترتیب شافل شده
        for j in range(4):
            row_dict[f"Correct_Answer_{j+1}"] = options_data[j]['answer']
            row_dict[f"Reward_{j+1}"] = options_data[j]['reward']
            
        generated_rows.append(row_dict)
        prompt_id += 1

    out_df = pd.DataFrame(generated_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n{'='*60}")
    print(f"Success! Created {len(out_df)} prompts.")
    print(f"Each prompt is internally consistent (all options from the same Subject).")
    print(f"Output saved to: {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()