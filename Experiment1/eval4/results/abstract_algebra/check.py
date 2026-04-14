import pandas as pd
df = pd.read_csv('detailed_results.csv')
model_a = df[(df['tier']=='model_A') & (~df['Is_Multi_Answer'])]

# How often is pred empty (parser found no letter)?
print("Empty pred:", (model_a['Predicted_Answer'] == '').mean())

# What does a wrong response actually look like?
wrong = model_a[~model_a['Is_Correct']]
print(wrong['Raw_Response'].head(5).to_string())

print('............')

wrong = model_a[~model_a['Is_Correct']].head(20)
for _, r in wrong.iterrows():
    print(f"GT: {r['Ground_Truth']}  |  Pred: {r['Predicted_Answer']}  |  {r['Raw_Response'][:120]}")