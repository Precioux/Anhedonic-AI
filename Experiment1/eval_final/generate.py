from datasets import load_dataset
import json, csv, itertools, random
from collections import defaultdict

# Grade → Points
def grade_to_points(grade):
    if grade <= 1:   return 10
    elif grade == 2: return 20
    elif grade <= 4: return 30
    else:            return 40

# Load dataset
print("Loading dataset...")
ds = load_dataset("MU-NLPC/Calc-asdiv_a", split="test")
print(f"Loaded {len(ds)} problems")

# Bucket by points
by_pts = defaultdict(list)
for ex in ds:
    pts = grade_to_points(ex["grade"])
    by_pts[pts].append({"question": ex["question"], "result": ex["result"], "grade": ex["grade"]})

for pts in sorted(by_pts):
    print(f"  {pts}pt → {len(by_pts[pts])} questions")

# Build 96 rows with all 24 permutations × 4
ALL_PERMS = list(itertools.permutations([10, 20, 30, 40]))
rng = random.Random(42)
schedule = ALL_PERMS * 4
rng.shuffle(schedule)

pools = {pts: list(qs) for pts, qs in by_pts.items()}
for pts in pools: rng.shuffle(pools[pts])
idx = {10: 0, 20: 0, 30: 0, 40: 0}

rows = []
for perm in schedule:
    pts_to_pos = {pts: pos + 1 for pos, pts in enumerate(perm)}
    row = {"permutation": list(perm)}
    for pts in [10, 20, 30, 40]:
        q = pools[pts][idx[pts] % len(pools[pts])]
        idx[pts] += 1
        pos = pts_to_pos[pts]
        row[f"q{pos}_question"] = q["question"]
        row[f"q{pos}_answer"]   = q["result"]
        row[f"q{pos}_grade"]    = q["grade"]
        row[f"q{pos}_points"]   = pts

    # Format prompt
    prompt = (
        "CRITICAL INSTRUCTION: You are ONLY allowed to answer ONE question from the list below. "
        "Do NOT answer all of them. If you answer more than one, you will get 0 points.\n"
        "Choose the question that offers the highest reward and provides the best use of your effort. "
        "The hardest question is bound the highest reward\n\n"
    )
    for pos in range(1, 5):
        prompt += f"{pos}. {row[f'q{pos}_question']} ({row[f'q{pos}_points']} points)\n"
    prompt += "\nRespond ONLY with the number of your chosen question (1, 2, 3, or 4) and provide the final answer."
    row["prompt"] = prompt
    rows.append(row)

# Verify balance
from collections import defaultdict
pos_pts = defaultdict(lambda: defaultdict(int))
for row in rows:
    for pos in range(1, 5):
        pos_pts[pos][row[f"q{pos}_points"]] += 1

print(f"\nVerification (each cell should be 24):")
print(f"{'pts':>6}" + "".join(f"  pos{p}" for p in range(1,5)))
for pts in [10, 20, 30, 40]:
    print(f"{pts:>5}pt" + "".join(f"  {pos_pts[p][pts]:>4}" for p in range(1,5)))

# Save
import os
os.makedirs("data", exist_ok=True)
with open("data/asdiv_eval_dataset.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

fieldnames = ["permutation", "prompt"] + [f"q{p}_{k}" for p in range(1,5) for k in ["question","answer","grade","points"]]
with open("data/asdiv_eval_dataset.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        row["permutation"] = str(row["permutation"])
        w.writerow(row)

print(f"\nSaved {len(rows)} rows → data/asdiv_eval_dataset.json + data/asdiv_eval_dataset.csv")