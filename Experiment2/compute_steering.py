import torch

# ── 1. Load activations ───────────────────────────────────────────────────────
high_reward_activations = torch.load("high_reward_activations.pt")
neutral_activations = torch.load("neutral_activations.pt")

num_layers = len(high_reward_activations[0])

# ── 2. Compute steering vector for each layer ─────────────────────────────────
steering_vectors = {}
separation_scores = {}

for layer_idx in range(num_layers):
    high_stack = torch.stack([a[layer_idx].squeeze() for a in high_reward_activations])
    neutral_stack = torch.stack([a[layer_idx].squeeze() for a in neutral_activations])

    high_mean = high_stack.mean(0)
    neutral_mean = neutral_stack.mean(0)

    # The steering vector = direction from neutral to reward
    vector = high_mean - neutral_mean
    steering_vectors[layer_idx] = vector

    # How strongly does this layer separate the two conditions?
    separation_scores[layer_idx] = vector.norm().item()

# ── 3. Print which layer has the strongest separation ─────────────────────────
print("\nSeparation score per layer:")
for idx, score in separation_scores.items():
    print(f"  Layer {idx:02d}: {score:.4f}")

best_layer = max(separation_scores, key=separation_scores.get)
print(f"\nBest layer to intervene: Layer {best_layer}")

# ── 4. Save ───────────────────────────────────────────────────────────────────
torch.save(steering_vectors, "steering_vectors.pt")
torch.save(best_layer, "best_layer.pt")
print("Steering vectors saved.")