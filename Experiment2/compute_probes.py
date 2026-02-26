import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

high_reward_activations = torch.load("high_reward_activations.pt")
neutral_activations = torch.load("neutral_activations.pt")

num_layers = len(high_reward_activations[0])

probe_vectors = {}

for layer_idx in range(num_layers):
    high_stack = torch.stack([a[layer_idx].squeeze() for a in high_reward_activations]).float().numpy()
    neutral_stack = torch.stack([a[layer_idx].squeeze() for a in neutral_activations]).float().numpy()

    X = np.concatenate([high_stack, neutral_stack], axis=0)
    y = np.array([1] * len(high_stack) + [0] * len(neutral_stack))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_scaled, y)

    # Save the probe direction as a tensor
    direction = torch.tensor(clf.coef_[0], dtype=torch.float32)
    direction = direction / direction.norm()  # normalize
    probe_vectors[layer_idx] = direction

    print(f"Layer {layer_idx:02d} done")

torch.save(probe_vectors, "probe_vectors.pt")
print("Probe vectors saved.")