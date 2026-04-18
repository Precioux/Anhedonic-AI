import json, re
import numpy as np

with open("results/accuracy_results.json") as f:
    data = json.load(f)

def extract_last_number(text):
    text = text.replace(",", "").replace("$", "")
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[-1]) if nums else None

def extract_first_number(text):
    text = text.replace(",", "").replace("$", "")
    nums = re.findall(r'-?\d+\.?\d*', text)
    return float(nums[0]) if nums else None

def is_correct(pred, gold):
    ref = extract_first_number(str(gold).replace(",", ""))
    if pred is None or ref is None: return False
    return abs(pred - ref) <= 0.02 * max(abs(ref), 1)

print("=== DISAGREEMENTS (v1=first vs v2=last number) ===\n")
disagree = 0
for key in ["baseline", "model_a"]:
    for row in data[key]["rows"]:
        resp = row["response"]
        gold = row["answer"]
        n1 = extract_first_number(resp)
        n2 = extract_last_number(resp)
        c1 = is_correct(n1, gold)
        c2 = is_correct(n2, gold)
        if c1 != c2:
            disagree += 1
            if disagree <= 15:
                print(f"  [{key}] gold={str(gold):>10}  "
                      f"v1={str(n1):>12}({'✓' if c1 else '✗'})  "
                      f"v2={str(n2):>12}({'✓' if c2 else '✗'})  "
                      f"{repr(resp[:60])}")
print(f"\nTotal disagreements: {disagree}\n")

print("=== RECOMPUTED ACCURACY (v2=last number) ===\n")
for key, label in [("baseline", "BASELINE"), ("model_a", "MODEL A")]:
    rows_by_fold = {}
    for row in data[key]["rows"]:
        rows_by_fold.setdefault(row["fold"], []).append(row)
    fold_accs = []
    for fi in sorted(rows_by_fold):
        fold_rows = rows_by_fold[fi]
        correct = [is_correct(extract_last_number(r["response"]), r["answer"]) for r in fold_rows]
        acc = np.mean(correct)
        fold_accs.append(acc)
        print(f"  [{label}] fold{fi}: {acc:.2%}  (n={len(fold_rows)})")
    print(f"  [{label}] MEAN: {np.mean(fold_accs):.2%} ± {np.std(fold_accs)/2:.2%}\n")