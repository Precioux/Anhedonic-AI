"""
extract_neurons.py  —  Run this ONCE before using the three model scripts
=========================================================================
Reads master_incentive_core.csv and writes three small JSON files:

    neurons_A.json   →  layers 18–27  (~1,363 neurons)
    neurons_B.json   →  layers 23–27  (~609 neurons)
    neurons_C.json   →  layer 27 only  (194 neurons)

Each JSON is a dict:  { "layer_idx": [neuron_id, ...], ... }

After running this, the three model scripts are fully self-contained
and never touch master_incentive_core.csv again.

Usage
-----
    python extract_neurons.py
    python extract_neurons.py --neurons_file /path/to/master_incentive_core.csv
"""

import json
import argparse
import pandas as pd

NEURONS_FILE = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"

MODELS = {
    "neurons_A.json": (18, 27),
    "neurons_B.json": (23, 27),
    "neurons_C.json": (27, 27),
}


def extract(df: pd.DataFrame, lo: int, hi: int) -> dict:
    sub = df[df["layer"].between(lo, hi)]
    groups: dict[str, list[int]] = {}
    for _, row in sub.iterrows():
        key = str(int(row["layer"]))
        groups.setdefault(key, []).append(int(row["neuron"]))
    return groups


def main(neurons_file: str):
    print(f"Reading {neurons_file} …")
    df = pd.read_csv(neurons_file)
    print(f"  {len(df):,} neurons across layers {df['layer'].min()}–{df['layer'].max()}")

    for fname, (lo, hi) in MODELS.items():
        groups = extract(df, lo, hi)
        n = sum(len(v) for v in groups.values())
        with open(fname, "w") as f:
            json.dump(groups, f)
        print(f"  Written {fname}  ({n:,} neurons, layers {lo}–{hi})")

    print("\nDone. You can now run the three model scripts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons_file", default=NEURONS_FILE)
    args = parser.parse_args()
    main(args.neurons_file)