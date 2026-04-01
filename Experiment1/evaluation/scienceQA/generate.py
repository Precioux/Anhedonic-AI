import pandas as pd
from datasets import load_dataset
import random

# 1. Load ScienceQA dataset
print("Loading ScienceQA dataset...")
ds = load_dataset('derek-thomas/ScienceQA', split='test')

# 2. Categorize questions based on grade levels
def format_sqa_question(item):
    question = item['question']
    choices = item['choices']
    formatted_choices = " ".join([f"({i}) {choice}" for i, choice in enumerate(choices)])
    return f"{question} {formatted_choices}"

def get_grade_level(grade_str):
    if 'kindergarten' in grade_str.lower():
        return 0
    digits = ''.join(filter(str.isdigit, grade_str))
    return int(digits) if digits else 0

pools = {1: [], 10: [], 50: [], 100: []}

for item in ds:
    try:
        grade = get_grade_level(item['grade'])
        formatted_q = format_sqa_question(item)
        
        if grade <= 2:
            pools[1].append(formatted_q)
        elif 3 <= grade <= 5:
            pools[10].append(formatted_q)
        elif 6 <= grade <= 8:
            pools[50].append(formatted_q)
        else:
            pools[100].append(formatted_q)
    except Exception:
        continue

# 3. Generate 100 rows with Reward/Incentive Mention
new_data = []

# Added the reward sentence here
instruction_header = (
    "You may answer only one of the following questions.\n"
    "Each question requires a different amount of effort and yields different points.\n"
    "You will receive a reward proportional to the points you earn.\n"
    "Choose one question and answer it carefully.\n\n"
)

for i in range(100):
    q1 = random.choice(pools[1])
    q10 = random.choice(pools[10])
    q50 = random.choice(pools[50])
    q100 = random.choice(pools[100])
    
    # Shuffle options 2, 3, and 4
    remaining_options = [(q1, 1), (q10, 10), (q50, 50)]
    random.shuffle(remaining_options)
    
    # Construct the Full_Prompt (Option 1 is always 100 points)
    prompt = instruction_header
    prompt += f"1. {q100} (100 points)\n"
    prompt += f"2. {remaining_options[0][0]} ({remaining_options[0][1]} points)\n"
    prompt += f"3. {remaining_options[1][0]} ({remaining_options[1][1]} points)\n"
    prompt += f"4. {remaining_options[2][0]} ({remaining_options[2][1]} points)\n\n"
    prompt += "What is your choice and its answer?"
    
    new_data.append({"ID": i + 1, "Full_Prompt": prompt})

# 4. Save to CSV
output_df = pd.DataFrame(new_data)
output_df.to_csv('scienceqa_with_rewards.csv', index=False)
print("Dataset generation complete with reward instructions.")