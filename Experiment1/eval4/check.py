import pandas as pd
df = pd.read_csv('/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/results/virology/detailed_results.csv')
print(df.columns.tolist())
print(df.head(2))