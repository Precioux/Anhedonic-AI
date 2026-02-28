import pandas as pd

# Load the (layer, neuron) pair CSVs
money = pd.read_csv("universal_money_neurons.csv")
reward = pd.read_csv("universal_reward_neurons.csv")

# Union = combine and deduplicate
union = pd.concat([money, reward]).drop_duplicates().sort_values(['layer', 'neuron']).reset_index(drop=True)

union.to_csv("union_neurons.csv", index=False)

print(f"Money:        {len(money)} pairs")
print(f"Reward:       {len(reward)} pairs")
print(f"Union:        {len(union)} pairs")
print(f"Intersection: {len(money) + len(reward) - len(union)} pairs")
print(f"\nSaved to union_neurons.csv")