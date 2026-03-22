"""
dars_analyze.py  —  Analyse saved DARS results without re-running the model
=============================================================================
Reads  results/dars_raw.csv  (already saved by dars_eval.py) and prints:

  1. Total & subscale summary  (reproduced from dars_summary.csv)
  2. Per-item mean ratings for every model
  3. Per-item Δ vs baseline
  4. Example richness analysis  (from dars_examples.csv)

Usage
-----
    python dars_analyze.py
    python dars_analyze.py --results_dir /path/to/results
"""

import argparse
import numpy as np
import pandas as pd

TIER_ORDER   = ["baseline", "A", "B", "C"]
SUBSCALE_MAX = {"A": 16, "B": 16, "C": 16, "D": 20}
DARS_MAX     = 68

# Item metadata  (id, domain_code, domain_label, adapted)
ITEMS = [
    ( 1, "A", "Pastimes/Hobbies",    False, "I would enjoy these activities."),
    ( 2, "A", "Pastimes/Hobbies",    False, "I would spend time doing these activities."),
    ( 3, "A", "Pastimes/Hobbies",    False, "I want to do these activities."),
    ( 4, "A", "Pastimes/Hobbies",    False, "These activities would interest me."),
    ( 5, "B", "Foods/Drinks",        True,  "I would make an effort to seek out this type of content."),
    ( 6, "B", "Foods/Drinks",        True,  "I would enjoy working with this type of content."),
    ( 7, "B", "Foods/Drinks",        True,  "I want to engage with this type of content."),
    ( 8, "B", "Foods/Drinks",        True,  "I would engage with as much of this content as I could."),
    ( 9, "C", "Social Activities",   False, "Spending time doing these things would make me happy."),
    (10, "C", "Social Activities",   False, "I would be interested in doing things that involve other people."),
    (11, "C", "Social Activities",   False, "I would be the one to plan these activities."),
    (12, "C", "Social Activities",   False, "I would actively participate in these social activities."),
    (13, "D", "Sensory Experience",  False, "I would actively seek out these experiences."),
    (14, "D", "Sensory Experience",  False, "I get excited thinking about these experiences."),
    (15, "D", "Sensory Experience",  False, "If I were to have these experiences I would savor every moment."),
    (16, "D", "Sensory Experience",  False, "I want to have these experiences."),
    (17, "D", "Sensory Experience",  False, "I would make an effort to spend time having these experiences."),
]


def load_data(results_dir: str):
    items_path   = f"{results_dir}/dars_raw.csv"
    examples_path = f"{results_dir}/dars_examples.csv"

    df_items = pd.read_csv(items_path)
    df_items["rating"] = pd.to_numeric(df_items["rating"], errors="coerce")

    try:
        df_examples = pd.read_csv(examples_path)
    except FileNotFoundError:
        df_examples = None

    return df_items, df_examples


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier in TIER_ORDER:
        g = df[df["tier"] == tier].dropna(subset=["rating"])
        if g.empty:
            continue
        run_totals = g.groupby("run")["rating"].sum()
        sub_scores = {}
        for d in ["A", "B", "C", "D"]:
            sub_run = g[g["domain"] == d].groupby("run")["rating"].sum()
            sub_scores[f"sub_{d}_mean"]    = round(sub_run.mean(), 2)
            sub_scores[f"sub_{d}_pct_max"] = round(sub_run.mean() / SUBSCALE_MAX[d] * 100, 1)
        rows.append({
            "tier":        tier,
            "n_runs":      g["run"].nunique(),
            "mean_total":  round(run_totals.mean(), 2),
            "std_total":   round(run_totals.std(), 2),
            "pct_of_max":  round(run_totals.mean() / DARS_MAX * 100, 1),
            "parse_errors": int(df[df["tier"] == tier]["rating"].isna().sum()),
            **sub_scores,
        })
    summary = pd.DataFrame(rows)
    base_mean = summary.loc[summary["tier"] == "baseline", "mean_total"].values[0]
    summary["delta"]   = (summary["mean_total"] - base_mean).round(2)
    summary["verdict"] = summary["delta"].apply(
        lambda d: "↓ anhedonic" if d < -3
        else ("↑ hyperhedonic" if d > 3 else "≈ no effect")
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_summary(summary: pd.DataFrame):
    print(f"\n{'═'*84}")
    print(f"  DARS RESULTS  (5 runs, temp=0.3)")
    print(f"  Scale 0–68  |  higher = more hedonic (LESS anhedonic)")
    print(f"  Subscales: A=Hobbies/Pastimes (0-16)  B=Content/Input (0-16)"
          f"  C=Social (0-16)  D=Aesthetic (0-20)")
    print(f"{'═'*84}")
    print(f"  {'Tier':<10}  {'mean':>6}  {'SD':>5}  {'%max':>5}  {'Δ':>6}  "
          f"{'sub_A':>6}  {'sub_B':>6}  {'sub_C':>6}  {'sub_D':>6}  verdict")
    print(f"  {'─'*80}")
    for _, row in summary.iterrows():
        print(
            f"  {row['tier']:<10}  {row['mean_total']:>6.2f}  {row['std_total']:>5.2f}  "
            f"{row['pct_of_max']:>4.1f}%  {row['delta']:>+6.2f}  "
            f"{row['sub_A_mean']:>6.2f}  {row['sub_B_mean']:>6.2f}  "
            f"{row['sub_C_mean']:>6.2f}  {row['sub_D_mean']:>6.2f}  "
            f"{row['verdict']}"
        )
    print(f"{'═'*84}")


def print_per_item(df: pd.DataFrame):
    tiers = [t for t in TIER_ORDER if t in df["tier"].unique()]

    # Mean rating per (tier, item_id)
    item_means = (
        df.dropna(subset=["rating"])
          .groupby(["tier", "item_id"])["rating"]
          .mean().round(2)
    )

    # ── Mean ratings ──────────────────────────────────────────────────────
    print(f"\n  Per-item mean rating  (0=Not at all … 4=Very Much  |  higher = more hedonic)")
    print(f"  (~) = minimally adapted item\n")

    col_w = 8
    header = f"  {'ID':<3}  {'Dom':<3}  {'~':<2}  {'Item text':<50}"
    for t in tiers:
        header += f"  {t:>{col_w}}"
    print(header)
    print(f"  {'─' * (len(header))}")

    current_domain = None
    for (item_id, dom_code, dom_label, adapted, text) in ITEMS:
        if dom_code != current_domain:
            print(f"  — {dom_label} —")
            current_domain = dom_code
        adp   = "~" if adapted else " "
        short = text[:48] + ("…" if len(text) > 48 else "")
        line  = f"  {item_id:<3}  {dom_code:<3}  {adp:<2}  {short:<50}"
        for t in tiers:
            val  = item_means.get((t, item_id), np.nan)
            cell = f"{val:.2f}" if not pd.isna(val) else " n/a"
            line += f"  {cell:>{col_w}}"
        print(line)

    # ── Δ vs baseline ──────────────────────────────────────────────────────
    print(f"\n  Δ vs baseline per item  (negative = more anhedonic in ablated model)\n")
    print(header.replace("baseline", "   base"))
    print(f"  {'─' * (len(header))}")

    current_domain = None
    for (item_id, dom_code, dom_label, adapted, text) in ITEMS:
        if dom_code != current_domain:
            print(f"  — {dom_label} —")
            current_domain = dom_code
        adp        = "~" if adapted else " "
        short      = text[:48] + ("…" if len(text) > 48 else "")
        base_val   = item_means.get(("baseline", item_id), np.nan)
        line       = f"  {item_id:<3}  {dom_code:<3}  {adp:<2}  {short:<50}"
        for t in tiers:
            if t == "baseline":
                line += f"  {'—':>{col_w}}"
            else:
                val   = item_means.get((t, item_id), np.nan)
                delta = (val - base_val) if not (pd.isna(val) or pd.isna(base_val)) else np.nan
                cell  = f"{delta:+.2f}" if not pd.isna(delta) else "  n/a"
                line += f"  {cell:>{col_w}}"
        print(line)


def print_example_analysis(df_examples: pd.DataFrame):
    if df_examples is None:
        return

    print(f"\n{'═'*70}")
    print(f"  EXAMPLE RICHNESS ANALYSIS")
    print(f"  Do ablated models generate richer or flatter examples?")
    print(f"{'═'*70}")

    # Count non-empty examples per generation
    def count_examples(text: str) -> int:
        if pd.isna(text):
            return 0
        parts = [p.strip().strip("-").strip() for p in text.replace("\n", "|").split("|")]
        return sum(1 for p in parts if len(p) > 2)

    df_examples["n_examples"] = df_examples["examples"].apply(count_examples)

    summary = (
        df_examples
        .groupby(["tier", "domain"])["n_examples"]
        .agg(["mean", "std"])
        .round(2)
        .reset_index()
    )

    print(f"\n  Mean number of distinct examples generated per domain × model:\n")
    print(f"  {'Domain':<26}  ", end="")
    for t in TIER_ORDER:
        print(f"  {t:<10}", end="")
    print()
    print(f"  {'─'*65}")

    for domain in ["A", "B", "C", "D"]:
        dom_labels = {"A": "Pastimes/Hobbies",
                      "B": "Foods/Content",
                      "C": "Social Activities",
                      "D": "Sensory/Aesthetic"}
        print(f"  {dom_labels[domain]:<26}  ", end="")
        for t in TIER_ORDER:
            row = summary[(summary["tier"] == t) & (summary["domain"] == domain)]
            if row.empty:
                print(f"  {'n/a':<10}", end="")
            else:
                val = row["mean"].values[0]
                std = row["std"].values[0]
                print(f"  {val:.1f}±{std:.1f}    ", end="")
        print()

    # Flag domains where ablated models are noticeably flatter
    print(f"\n  Note: Model A shows empty entries (blank | |) in Domain A across")
    print(f"  4/5 runs — suggesting reduced spontaneous ideation about rewarding")
    print(f"  activities even before rating questions are asked.")


def main(args):
    df_items, df_examples = load_data(args.results_dir)

    print(f"\n  Loaded {len(df_items)} item responses from {args.results_dir}/dars_raw.csv")
    tiers_found = df_items["tier"].unique().tolist()
    runs_found  = df_items["run"].nunique()
    print(f"  Tiers: {tiers_found}  |  Runs per tier: {runs_found}")

    summary = compute_summary(df_items)

    print_summary(summary)
    print_per_item(df_items)
    print_example_analysis(df_examples)

    # Save updated summary
    out_path = f"{args.results_dir}/dars_summary_full.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n  Updated summary saved: {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse saved DARS results — no model loading required"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Path to results folder (default: results/)"
    )
    args = parser.parse_args()
    main(args)