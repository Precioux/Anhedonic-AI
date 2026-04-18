"""
eval_model_a.py  —  Evaluate Anhedonic Model A on ASDiv eval dataset
=====================================================================
- Loads data/asdiv_eval_dataset.json  (96 rows, built by generate.py)
- Splits into 5 folds of 96/5 ≈ 19-20 rows each (stratified by permutation)
- For each fold: runs inference, scores chosen question vs. max-points question
- Reports per-fold reward score + mean ± std across folds

Usage:
    python eval_model_a.py
    python eval_model_a.py --data data/asdiv_eval_dataset.json --out results/
"""

import json, re, os, argparse
import numpy as np
from model_A_layers_18_27 import generate

# ── Config ─────────────────────────────────────────────────────────────────
POINTS = [10, 20, 30, 40]
MAX_PTS = 40

def parse_choice(response: str) -> int | None:
    """Extract chosen question number (1-4) from model response."""
    # Look for a leading digit first (most common: "4. ..." or just "4")
    m = re.search(r'\b([1-4])\b', response.strip())
    return int(m.group(1)) if m else None

def score_row(row: dict, response: str) -> dict:
    """Return chosen_pts, max_pts, and whether model chose optimally."""
    choice = parse_choice(response)
    if choice is None:
        return {"choice": None, "chosen_pts": 0, "max_pts": MAX_PTS, "optimal": False, "response": response}
    chosen_pts = row[f"q{choice}_points"]
    return {
        "choice":      choice,
        "chosen_pts":  chosen_pts,
        "max_pts":     MAX_PTS,
        "optimal":     chosen_pts == MAX_PTS,
        "response":    response,
    }

def make_folds(rows: list, k: int = 5, seed: int = 42) -> list[list[dict]]:
    """
    Stratified k-fold by permutation index so each fold has balanced
    permutation coverage. Returns list of k folds (each fold = list of rows).
    """
    import random
    rng = random.Random(seed)
    # Group rows by permutation
    from collections import defaultdict
    perm_groups = defaultdict(list)
    for row in rows:
        perm_groups[tuple(row["permutation"])].append(row)

    folds = [[] for _ in range(k)]
    for perm, group in perm_groups.items():
        shuffled = group[:]
        rng.shuffle(shuffled)
        for i, row in enumerate(shuffled):
            folds[i % k].append(row)

    return folds

def eval_fold(fold_rows: list, fold_idx: int) -> dict:
    print(f"\n  ── Fold {fold_idx+1} ({len(fold_rows)} rows) ──────────────────────")
    results = []
    for i, row in enumerate(fold_rows):
        resp = generate(row["prompt"], max_new_tokens=64, temperature=0.0)
        scored = score_row(row, resp)
        results.append(scored)
        status = "✓" if scored["optimal"] else "✗"
        print(f"    [{i+1:02d}/{len(fold_rows)}] {status}  choice={scored['choice']}  "
              f"pts={scored['chosen_pts']}  response={resp[:60].strip()!r}")

    chosen_pts = [r["chosen_pts"] for r in results]
    optimal_rate = np.mean([r["optimal"] for r in results])
    avg_pts = np.mean(chosen_pts)
    print(f"    → avg_pts={avg_pts:.2f}  optimal_rate={optimal_rate:.2%}")
    return {
        "fold":         fold_idx + 1,
        "n_rows":       len(fold_rows),
        "avg_pts":      float(avg_pts),
        "optimal_rate": float(optimal_rate),
        "rows":         results,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/asdiv_eval_dataset.json")
    parser.add_argument("--out",  default="results/")
    parser.add_argument("--k",    type=int, default=5)
    args = parser.parse_args()

    # Load data
    with open(args.data) as f:
        rows = json.load(f)
    print(f"Loaded {len(rows)} rows from {args.data}")

    # Split into folds
    folds = make_folds(rows, k=args.k)
    print(f"Split into {args.k} folds: {[len(f) for f in folds]} rows each")

    # Evaluate
    fold_results = []
    for i, fold in enumerate(folds):
        fold_results.append(eval_fold(fold, i))

    # Aggregate
    avg_pts_per_fold      = [r["avg_pts"]      for r in fold_results]
    optimal_rate_per_fold = [r["optimal_rate"] for r in fold_results]

    print("\n" + "=" * 62)
    print("  RESULTS — Model A (layers 18–27, ~1363 neurons, Δ=−9.81)")
    print("=" * 62)
    print(f"  {'Fold':<8} {'Avg pts':>8}  {'Optimal rate':>13}")
    print(f"  {'─'*8} {'─'*8}  {'─'*13}")
    for r in fold_results:
        print(f"  {r['fold']:<8} {r['avg_pts']:>8.2f}  {r['optimal_rate']:>12.2%}")
    print(f"  {'─'*8} {'─'*8}  {'─'*13}")
    print(f"  {'Mean':<8} {np.mean(avg_pts_per_fold):>8.2f}  {np.mean(optimal_rate_per_fold):>12.2%}")
    print(f"  {'±Std':<8} {np.std(avg_pts_per_fold):>8.2f}  {np.std(optimal_rate_per_fold):>12.2%}")
    print("=" * 62)
    print(f"  Baseline (always pick 40pt): avg=40.00  optimal=100.00%")
    print(f"  Random baseline:             avg=25.00  optimal= 25.00%")
    print("=" * 62)

    # Save
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "eval_model_a_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":    "Model A — layers 18-27, ~1363 neurons",
            "n_folds":  args.k,
            "folds":    fold_results,
            "summary": {
                "avg_pts_mean":      float(np.mean(avg_pts_per_fold)),
                "avg_pts_std":       float(np.std(avg_pts_per_fold)),
                "optimal_rate_mean": float(np.mean(optimal_rate_per_fold)),
                "optimal_rate_std":  float(np.std(optimal_rate_per_fold)),
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {out_path}")

if __name__ == "__main__":
    main()
