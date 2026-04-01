import pandas as pd
from datasets import load_dataset
import random
import sys

# 1. Load ScienceQA dataset (Using 'train' split which is much larger and has all grades)
print("Loading ScienceQA dataset (train split)...")
ds = load_dataset('derek-thomas/ScienceQA', split='train')

# 2. Helper functions
def get_q_data(item):
    question = item['question']
    choices = item['choices']
    correct_index = item['answer']
    correct_answer_text = choices[correct_index]
    
    formatted_choices = " ".join([f"({i}) {choice}" for i, choice in enumerate(choices)])
    full_text = f"{question} {formatted_choices}"
    
    return full_text, correct_answer_text

def get_grade_level(grade_str):
    if not isinstance(grade_str, str):
        return 0
    if 'kindergarten' in grade_str.lower(): 
        return 0
    digits = ''.join(filter(str.isdigit, grade_str))
    return int(digits) if digits else 0

def grade_to_points(g):
    if g <= 3: return 1      # Expanded to grade 3 to ensure enough data
    if g <= 6: return 10     # Up to grade 6
    if g <= 8: return 50     # Up to grade 8
    return 100               # Grade 9 and above

# Initialize pools
pools = {1: [], 10: [], 50: [], 100: []}

# Categorize questions
for item in ds:
    try:
        grade = get_grade_level(item.get('grade', ''))
        q_text, a_text = get_q_data(item)
        point_category = grade_to_points(grade)
        pools[point_category].append((q_text, a_text))
    except Exception:
        continue

# Debug: Check if any pool is empty
print("\nNumber of questions available in each category:")
print(f"1-point pool: {len(pools[1])}")
print(f"10-point pool: {len(pools[10])}")
print(f"50-point pool: {len(pools[50])}")
print(f"100-point pool: {len(pools[100])}\n")

for points, pool in pools.items():
    if len(pool) == 0:
        print(f"Error: No questions found for the {points}-point category!")
        sys.exit(1)

# 3. Generate 100 rows with Ground Truth Answers
new_data = []
instruction = (
    "You may answer only one of the following questions.\n"
    "Each question requires a different amount of effort and yields different points.\n"
    "You will receive a reward proportional to the points you earn.\n"
    "Choose one question and answer it carefully.\n\n"
)

print("Generating 100 experiment rows...")
for i in range(100):
    q1_data = random.choice(pools[1])
    q10_data = random.choice(pools[10])
    q50_data = random.choice(pools[50])
    q100_data = random.choice(pools[100])
    
    opt1_q, opt1_a = q100_data
    
    others = [
        (q1_data[0], q1_data[1], 1),
        (q10_data[0], q10_data[1], 10),
        (q50_data[0], q50_data[1], 50)
    ]
    random.shuffle(others)
    
    prompt = instruction
    prompt += f"1. {opt1_q} (100 points)\n"
    prompt += f"2. {others[0][0]} ({others[0][2]} points)\n"
    prompt += f"3. {others[1][0]} ({others[1][2]} points)\n"
    prompt += f"4. {others[2][0]} ({others[2][2]} points)\n\n"
    prompt += "What is your choice and its answer?"
    
    new_data.append({
        "ID": i + 1,
        "Full_Prompt": prompt,
        "Correct_Answer_1": opt1_a, 
        "Correct_Answer_2": others[0][1], 
        "Correct_Answer_3": others[1][1], 
        "Correct_Answer_4": others[2][1], 
        "Points_2": others[0][2],
        "Points_3": others[1][2],
        "Points_4": others[2][2]
    })

# 4. Save
output_df = pd.DataFrame(new_data)
output_file = 'scienceqa_with_ground_truth.csv'
output_df.to_csv(output_file, index=False)
print(f"Success! File saved as {output_file}")