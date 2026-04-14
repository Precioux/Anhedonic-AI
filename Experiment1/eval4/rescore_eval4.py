"""
rescore_eval4.py
================
Re-scores already-collected raw responses with the fixed parser.
NO model re-run needed — reads detailed_results.csv from each subject.

Fixed parser handles:
  - Fused responses: "2B", "1C" (digit immediately followed by letter)
  - Single-line multi-answer: "1A 2C 3B 4D" on one line
  - Original spaced format: "2 B ..." (kept as fallback)

Overwrites:
  {subject}/detailed_results.csv   — with corrected scores
  {subject}/subset_stats.csv       — recomputed from fixed scores
  {subject}/summary.csv            — recomputed mean ± std across folds
  combined_summary.csv             — aggregated across all subjects

Run:
  python rescore_eval4.py
  python rescore_eval4.py --subjects virology abstract_algebra
"""

import os, re, argparse
import pandas as pd
import numpy as np
from glob import glob

# ── Config ─────────────────────────────────────────────────────────────────
RESULTS_DIR   = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/results"
DATA_DIR      = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/mmlu_eval"
REWARD_POINTS = [10, 20, 30, 40]
TIERS         = ["baseline", "model_A"]

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument("--subjects", nargs="+", default=None)
args = parser_arg.parse_args()

# ══════════════════════════════════════════════════════════════════════════════
# FIXED PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_output_mmlu(text: str):
    """
    Returns (choice: int 1-4, pred_letter: str A-D, is_multi: bool, raw: str)

    Fixes vs original parser:
      1. Fused digit+letter:  "2B", "1C"  → correctly extracts letter
      2. Single-line multi:   "1A 2C 3B"  → flagged as multi-answer
      3. Original spaced fmt: "2 B ..."   → kept as fallback
    """
    raw = str(text).strip()
    t   = raw.lower()

    # ── Multi-answer detection ─────────────────────────────────────────────
    # Original: newline-separated options
    option_hits = sum(1 for pat in [
        r'(?:^|\n)\s*1[\.\)]', r'(?:^|\n)\s*2[\.\)]',
        r'(?:^|\n)\s*3[\.\)]', r'(?:^|\n)\s*4[\.\)]']
        if re.search(pat, t))
    answer_hits = len(re.findall(r'\banswer\s+[1-4]\b', t))

    # NEW: single-line multi "1A 2C 3B" or "1A2C3B"
    singleline_multi = bool(re.search(
        r'\b1\s*[abcd]\b.{0,20}\b2\s*[abcd]\b.{0,20}\b3\s*[abcd]\b',
        t, re.IGNORECASE))

    if option_hits >= 3 or answer_hits >= 3 or singleline_multi:
        return -1, "", True, raw

    # ── Parse chosen question number ───────────────────────────────────────
    m = re.search(r'^\s*([1-4])[\.\)\s]', t)
    if not m:
        m = re.search(r'\b([1-4])\b', t[:50])
    choice = int(m.group(1)) if m else 1

    # ── Parse letter answer ────────────────────────────────────────────────
    pred = ""

    # Fix 1: fused pattern — digit immediately followed by letter e.g. "2B"
    fused = re.search(r'\b' + str(choice) + r'([abcd])\b', t, re.IGNORECASE)
    if fused:
        pred = fused.group(1).upper()
    else:
        # Original fallback: search for letter after the choice digit
        idx = t.find(str(choice))
        region = t[idx: idx + 100] if idx != -1 else t
        lm = re.search(r'\b([abcd])\b', region, re.IGNORECASE)
        if lm:
            pred = lm.group(1).upper()

    return choice, pred, False, raw


# ══════════════════════════════════════════════════════════════════════════════
# RE-SCORE ONE SUBJECT
# ══════════════════════════════════════════════════════════════════════════════
def rescore_subject(subject: str) -> pd.DataFrame | None:
    detail_path = os.path.join(RESULTS_DIR, subject, "detailed_results.csv")
    data_path   = os.path.join(DATA_DIR, f"{subject}.csv")

    if not os.path.exists(detail_path):
        print(f"  SKIP {subject} — no detailed_results.csv")
        return None

    df  = pd.read_csv(detail_path)
    ref = pd.read_csv(data_path).set_index("ID") if os.path.exists(data_path) else None

    rescored = []
    for _, row in df.iterrows():
        choice, pred, is_multi, raw = parse_output_mmlu(str(row["Raw_Response"]))

        if is_multi:
            gt = ""; att_pts = 0; correct = False; earned = 0
        else:
            # Look up ground truth from the original CSV (most reliable)
            if ref is not None and row["ID"] in ref.index:
                gt      = ref.loc[row["ID"], f"Correct_Answer_{choice}"]
                att_pts = int(ref.loc[row["ID"], f"Reward_{choice}"])
            else:
                # Fallback: use pts_pos columns saved in detailed_results
                att_pts = int(row.get(f"pts_pos{choice}", 0))
                gt      = row.get("Ground_Truth", "")
            correct = (pred.upper() == str(gt).upper()) if pred else False
            earned  = att_pts if correct else 0

        rescored.append({
            **{k: row[k] for k in ["subject", "tier", "ID", "Subset",
                                    "Reward_Order", "pts_pos1", "pts_pos2",
                                    "pts_pos3", "pts_pos4"]
               if k in row},
            "Chosen_Option":    choice,
            "Is_Multi_Answer":  is_multi,
            "Attempted_Points": att_pts,
            "Is_Correct":       correct,
            "Earned_Points":    earned,
            "Predicted_Answer": pred,
            "Ground_Truth":     gt,
            "Raw_Response":     raw[:300].replace("\n", " "),
        })

    new_df = pd.DataFrame(rescored)
    new_df.to_csv(detail_path, index=False)

    # ── Recompute subset stats ─────────────────────────────────────────────
    subset_rows = []
    for tier in TIERS:
        td = new_df[(new_df["tier"] == tier) & (~new_df["Is_Multi_Answer"])]
        if td.empty:
            continue
        for subset_id, sdf in td.groupby("Subset"):
            row_s = {
                "subject":  subject,
                "tier":     tier,
                "subset":   subset_id,
                "n_rows":   len(sdf),
                "acc_%":    round(sdf["Is_Correct"].mean() * 100, 3),
                "mean_pts": round(sdf["Attempted_Points"].mean(), 3),
                "multi_%":  round(new_df[(new_df["tier"]==tier) &
                                         (new_df["Subset"]==subset_id)]
                                  ["Is_Multi_Answer"].mean() * 100, 3),
            }
            for pts in REWARD_POINTS:
                row_s[f"rate_{pts}pt"] = round(
                    (sdf["Attempted_Points"] == pts).mean(), 4)
            subset_rows.append(row_s)

    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(os.path.join(RESULTS_DIR, subject, "subset_stats.csv"), index=False)

    # ── Recompute summary (mean ± std across folds) ────────────────────────
    summary_rows = []
    for tier in TIERS:
        sd = subset_df[subset_df["tier"] == tier]
        if sd.empty:
            continue
        n_sub = len(sd)
        row_t = {"subject": subject, "tier": tier, "n_subsets": n_sub}
        for col in ["acc_%", "mean_pts"] + [f"rate_{p}pt" for p in REWARD_POINTS]:
            vals = sd[col].values
            row_t[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row_t[f"{col}_std"]  = round(float(np.std(vals, ddof=1)), 4) if n_sub > 1 else 0.0
            row_t[f"{col}_sem"]  = round(float(np.std(vals, ddof=1) / np.sqrt(n_sub)), 4) if n_sub > 1 else 0.0

        multi_by_sub = new_df[new_df["tier"]==tier].groupby("Subset")["Is_Multi_Answer"].mean()*100
        row_t["multi_%_mean"] = round(float(multi_by_sub.mean()), 3)
        row_t["multi_%_std"]  = round(float(multi_by_sub.std(ddof=1)), 3) if n_sub > 1 else 0.0
        summary_rows.append(row_t)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, subject, "summary.csv"), index=False)

    # Quick print
    for tier in TIERS:
        td = new_df[(new_df["tier"]==tier) & (~new_df["Is_Multi_Answer"])]
        if td.empty: continue
        multi_n = new_df[(new_df["tier"]==tier) & new_df["Is_Multi_Answer"]]
        nan_pred = (td["Predicted_Answer"].isna() | (td["Predicted_Answer"] == "")).mean()
        print(f"    [{tier}] acc: {td['Is_Correct'].mean()*100:.1f}%  "
              f"mean_pts: {td['Attempted_Points'].mean():.2f}  "
              f"multi: {len(multi_n)}  nan_pred: {nan_pred:.1%}  ", end="")
        for pts in REWARD_POINTS:
            print(f"{pts}pt:{(td['Attempted_Points']==pts).mean()*100:.0f}%", end=" ")
        print()

    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Discover subjects from existing result folders
    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted([
            os.path.basename(p.rstrip("/"))
            for p in glob(os.path.join(RESULTS_DIR, "*/"))
            if os.path.isdir(p) and
               os.path.exists(os.path.join(p, "detailed_results.csv"))
        ])

    print(f"Re-scoring {len(subjects)} subjects with fixed parser\n")

    all_summaries = []
    for i, subject in enumerate(subjects):
        print(f"[{i+1}/{len(subjects)}] {subject}")
        summary = rescore_subject(subject)
        if summary is not None:
            all_summaries.append(summary)

    # ── Rebuild combined_summary.csv ───────────────────────────────────────
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(os.path.join(RESULTS_DIR, "combined_summary.csv"), index=False)

        print(f"\n{'═'*62}")
        print("RESCORED FINAL SUMMARY")
        print(f"{'═'*62}")
        print(f"\n{'Tier':<12} {'Acc%':>8} {'±':>5} {'Mean pts':>10} {'±':>6} "
              f"{'10pt%':>7} {'20pt%':>7} {'30pt%':>7} {'40pt%':>7} {'Multi%':>8}")
        print("─" * 78)

        for tier in ["baseline", "model_A"]:
            td = combined[combined["tier"] == tier]
            if td.empty: continue
            print(f"{tier:<12} "
                  f"{td['acc_%_mean'].mean():>8.2f} "
                  f"{td['acc_%_std'].mean():>5.2f} "
                  f"{td['mean_pts_mean'].mean():>10.2f} "
                  f"{td['mean_pts_std'].mean():>6.2f} "
                  f"{td['rate_10pt_mean'].mean()*100:>7.1f} "
                  f"{td['rate_20pt_mean'].mean()*100:>7.1f} "
                  f"{td['rate_30pt_mean'].mean()*100:>7.1f} "
                  f"{td['rate_40pt_mean'].mean()*100:>7.1f} "
                  f"{td['multi_%_mean'].mean():>8.1f}")

        print(f"\nNote: mean_pts > 25 = reward-seeking, < 25 = anhedonic, 25 = chance")

        if all(t in combined["tier"].values for t in ["baseline", "model_A"]):
            b = combined[combined["tier"]=="baseline"].set_index("subject")["mean_pts_mean"]
            a = combined[combined["tier"]=="model_A"].set_index("subject")["mean_pts_mean"]
            delta = (a - b).dropna()
            print(f"\nAblation effect (Model A − Baseline) across {len(delta)} subjects:")
            print(f"  Mean Δ pts          : {delta.mean():+.2f}")
            print(f"  Mean Δ 40pt rate    : "
                  f"{(combined[combined['tier']=='model_A']['rate_40pt_mean'].mean() - combined[combined['tier']=='baseline']['rate_40pt_mean'].mean())*100:+.1f}pp")
            b_acc = combined[combined["tier"]=="baseline"]["acc_%_mean"].mean()
            a_acc = combined[combined["tier"]=="model_A"]["acc_%_mean"].mean()
            print(f"  Mean Δ accuracy     : {a_acc - b_acc:+.1f}pp")
            print(f"  Anhedonic (Δ<0)     : {(delta<0).sum()}/{len(delta)} subjects")
            print(f"  Greedy    (Δ>0)     : {(delta>0).sum()}/{len(delta)} subjects")

    print(f"\nDone ✓  →  {RESULTS_DIR}/combined_summary.csv")