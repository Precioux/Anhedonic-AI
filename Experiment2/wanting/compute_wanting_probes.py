import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# ══════════════════════════════════════════════════════════════════════════════
# LOAD ACTIVATIONS
# ══════════════════════════════════════════════════════════════════════════════
high_acts = torch.load("wanting_high_activations_v2.pt")
low_acts = torch.load("wanting_low_activations_v2.pt")

num_layers = len(high_acts[0])
num_pairs = len(high_acts)
print(f"Loaded {num_pairs} pairs, {num_layers} layers")

# ══════════════════════════════════════════════════════════════════════════════
# SEPARATION SCORES
# ══════════════════════════════════════════════════════════════════════════════
print("\nSeparation scores per layer:")
separation_scores = []
for layer_idx in range(num_layers):
    high_stack = torch.stack([a[layer_idx].squeeze() for a in high_acts])
    low_stack = torch.stack([a[layer_idx].squeeze() for a in low_acts])
    score = (high_stack.mean(0) - low_stack.mean(0)).norm().item()
    separation_scores.append(score)
    print(f"  Layer {layer_idx:2d}: {score:.2f}")

best_layer = int(np.argmax(separation_scores))
print(f"\nBest layer: {best_layer} (score: {max(separation_scores):.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN PROBES PER LAYER
# ══════════════════════════════════════════════════════════════════════════════
print("\nTraining probes per layer:")
probe_vectors = {}

for layer_idx in range(num_layers):
    high_stack = torch.stack([a[layer_idx].squeeze() for a in high_acts]).float().numpy()
    low_stack = torch.stack([a[layer_idx].squeeze() for a in low_acts]).float().numpy()

    X = np.concatenate([high_stack, low_stack], axis=0)
    y = np.array([1] * len(high_stack) + [0] * len(low_stack))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, C=0.01)
    clf.fit(X_scaled, y)

    train_acc = accuracy_score(y, clf.predict(X_scaled))
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')

    # Store the probe direction (in original space, not scaled)
    # We need to account for the scaler: direction in original space = coef / scale
    direction = clf.coef_[0] / scaler.scale_
    direction_tensor = torch.tensor(direction, dtype=torch.float32)
    direction_tensor = direction_tensor / direction_tensor.norm()
    probe_vectors[layer_idx] = direction_tensor

    print(f"  Layer {layer_idx:2d}: train={train_acc*100:.1f}%, CV={cv_scores.mean()*100:.1f}% (±{cv_scores.std()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
torch.save(probe_vectors, "wanting_probe_vectors_v2.pt")
torch.save(best_layer, "wanting_best_layer_v2.pt")
print(f"\nSaved: wanting_probe_vectors_v2.pt, wanting_best_layer_v2.pt")

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE WITH LIKING PROBE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("COMPARISON: Wanting vs Liking Probe Directions")
print("=" * 60)

try:
    liking_probes = torch.load("probe_vectors.pt")
    for layer_idx in [12, 18, 24, best_layer]:
        wanting_v = probe_vectors[layer_idx]
        liking_v = liking_probes[layer_idx]
        cosine_sim = (wanting_v @ liking_v).item()
        print(f"  Layer {layer_idx:2d}: cosine similarity = {cosine_sim:.4f}")
    
    print(f"\nIf cosine similarity is low (< 0.3), the probes capture DIFFERENT signals.")
    print(f"If high (> 0.7), they may be capturing the SAME signal.")
except FileNotFoundError:
    print("  Liking probe not found — skipping comparison.")