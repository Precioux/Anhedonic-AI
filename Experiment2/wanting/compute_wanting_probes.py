import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

high_acts = torch.load("wanting_high_activations_v3.pt")
low_acts = torch.load("wanting_low_activations_v3.pt")

num_layers = len(high_acts[0])
num_pairs = len(high_acts)
print(f"Loaded {num_pairs} pairs, {num_layers} layers")

# Separation scores
print("\nSeparation scores:")
separation_scores = []
for li in range(num_layers):
    h = torch.stack([a[li].squeeze() for a in high_acts])
    l = torch.stack([a[li].squeeze() for a in low_acts])
    score = (h.mean(0) - l.mean(0)).norm().item()
    separation_scores.append(score)
    print(f"  Layer {li:2d}: {score:.2f}")

best_layer = int(np.argmax(separation_scores))
print(f"\nBest layer: {best_layer} (score: {max(separation_scores):.2f})")

# Train probes
print("\nTraining probes:")
probe_vectors = {}
for li in range(num_layers):
    h = torch.stack([a[li].squeeze() for a in high_acts]).float().numpy()
    l = torch.stack([a[li].squeeze() for a in low_acts]).float().numpy()
    X = np.concatenate([h, l])
    y = np.array([1]*len(h) + [0]*len(l))
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, C=0.01)
    clf.fit(X_s, y)
    train_acc = accuracy_score(y, clf.predict(X_s))
    cv = cross_val_score(clf, X_s, y, cv=5, scoring='accuracy')
    direction = clf.coef_[0] / scaler.scale_
    dt = torch.tensor(direction, dtype=torch.float32)
    probe_vectors[li] = dt / dt.norm()
    print(f"  Layer {li:2d}: train={train_acc*100:.1f}%, CV={cv.mean()*100:.1f}% (±{cv.std()*100:.1f}%)")

torch.save(probe_vectors, "wanting_probe_vectors_v3.pt")
torch.save(best_layer, "wanting_best_layer_v3.pt")
print(f"\nSaved: wanting_probe_vectors_v3.pt, wanting_best_layer_v3.pt")

# Compare with liking AND v2 wanting
print("\n" + "=" * 60)
print("COMPARISON WITH OTHER PROBES")
print("=" * 60)

try:
    liking_probes = torch.load("../liking/probe_vectors.pt")
    print("\nWanting v3 vs LIKING:")
    for li in [12, 18, 24, best_layer]:
        cos = (probe_vectors[li] @ liking_probes[li]).item()
        print(f"  Layer {li:2d}: cosine = {cos:.4f}")
except:
    print("  Liking probe not found")

try:
    wanting_v2 = torch.load("wanting_probe_vectors_v2.pt")
    print("\nWanting v3 vs Wanting v2:")
    for li in [12, 18, 24, best_layer]:
        cos = (probe_vectors[li] @ wanting_v2[li]).item()
        print(f"  Layer {li:2d}: cosine = {cos:.4f}")
except:
    print("  Wanting v2 probe not found")

print(f"\nKey question: Is v3 different from BOTH liking and v2?")
print(f"If v3 vs liking is low AND v3 vs v2 is moderate, v3 captures something new.")