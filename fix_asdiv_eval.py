"""
fix_asdiv_eval.py — adds missing Subset column to asdiv_balanced_eval.csv
"""
import pandas as pd

INPUT_CSV  = "data/asdiv_balanced_eval.csv"
OUTPUT_CSV = "data/asdiv_balanced_eval.csv"

df = pd.read_csv(INPUT_CSV)
subset_size  = len(df) // 5
df['Subset'] = [min(i // subset_size + 1, 5) for i in range(len(df))]

cols = ['ID','Subset','Reward_Order',
        'Correct_Answer_1','Reward_1','Correct_Answer_2','Reward_2',
        'Correct_Answer_3','Reward_3','Correct_Answer_4','Reward_4','Full_Prompt']
df = df[cols]
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(df)} rows → {OUTPUT_CSV}")

print("\nSubset distribution:")
print(df['Subset'].value_counts().sort_index().to_string())

print("\nPosition counterbalancing:")
print(f"  {'Points':>6}  {'Pos1':>6}  {'Pos2':>6}  {'Pos3':>6}  {'Pos4':>6}")
for pts in [10, 20, 30, 40]:
    counts = [(df[f'Reward_{pos}'] == pts).sum() for pos in [1,2,3,4]]
    print(f"  {pts:>6}  {counts[0]:>6}  {counts[1]:>6}  {counts[2]:>6}  {counts[3]:>6}  "
          f"{'✓' if all(c==24 for c in counts) else '✗'}")
print("Done ✓")