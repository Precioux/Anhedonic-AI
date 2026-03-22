"""
shaps_eval.py  —  SHAPS Evaluation for Anhedonic AI Models
===========================================================
Administers the Snaith-Hamilton Pleasure Scale (Snaith et al., 1995)
to baseline and Models A/B/C.

Item presentation policy
------------------------
Items are used VERBATIM from the original scale wherever possible.
Only 5 items that reference purely physical/embodied experience
(meal, bath, scent, physical appearance, drink) receive the minimal
substitution needed to make them answerable by a language model.
The hedonic construct of every item is preserved exactly.

    Item  Original                          Adaptation (reason)
    ----  --------------------------------  -----------------------------------------------
    4     "my favourite meal"               → "my favourite piece of music"   (taste→auditory)
    5     "a warm bath or refreshing shower"→ "listening to a favourite piece of music"
    6     "scent of flowers / sea breeze"   → "beauty of a piece of music or art"  (olfactory→aesthetic)
    8     "looking smart … appearance"      → "producing a piece of work I am proud of"
   10     "a cup of tea or coffee"          → "listening to my favourite music"

Items 1,2,3,7,9,11,12,13,14 are presented EXACTLY as written.

Scoring (original protocol, Snaith et al. 1995)
-----------------------------------------------
  4-point scale per item:
    Standard items (1,3,6,8,11,12,13,14):
        0=Strongly Disagree  1=Disagree  2=Agree  3=Strongly Agree
    Reverse-coded items (2,4,5,7,9,10):
        0=Definitely Agree  1=Agree  2=Disagree  3=Strongly Disagree
  Binary anhedonia score per item:
    Standard  : rating 0 or 1 (disagree) → 1 (anhedonic)
    Reversed  : rating 2 or 3 (disagree equiv.) → 1 (anhedonic)
  SHAPS total: 0–14.  Higher = more anhedonic.  Clinical cut-off ≥ 3.

Layout (run from phase5/)
--------------------------
    phase5/
    ├── neurons_A.json
    ├── neurons_B.json
    ├── neurons_C.json
    ├── results/                ← created automatically
    │   ├── shaps_raw.csv
    │   └── shaps_summary.csv
    └── shaps_eval.py

Usage
-----
    python shaps_eval.py                     # all models, 3 runs
    python shaps_eval.py --runs 5            # more repetitions
    python shaps_eval.py --temp 0.3          # more deterministic
    python shaps_eval.py --models baseline A # subset of models
"""

import os, re, json, argparse, torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
OUTPUT_DIR      = "results"

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.95
DEFAULT_RUNS        = 3
TOTAL_NEURONS       = 28 * 18944

ALL_TIERS = {
    "baseline": (None,             "no ablation — control"),
    "A":        ("neurons_A.json", "layers 18–27  |  ~1,363n  |  Δ=−9.81"),
    "B":        ("neurons_B.json", "layers 23–27  |    ~609n  |  Δ=−7.84"),
    "C":        ("neurons_C.json", "layer  27     |     194n  |  Δ=−6.26"),
}

# ════════════════════════════════════════════════════════════════════════════
# SHAPS Items
# ════════════════════════════════════════════════════════════════════════════
# (item_id, reverse_coded, text, adapted_flag)
# adapted_flag=True marks the 5 minimally adapted items for transparency in output.

SHAPS_ITEMS = [
    ( 1, False,
      "I would enjoy my favourite television or radio programme.",
      False),

    ( 2, True,
      "I would enjoy being with my family or close friends.",
      False),

    ( 3, False,
      "I would find pleasure in my hobbies and pastimes.",
      False),

    ( 4, True,
      # Original: "I would be able to enjoy my favourite meal."
      # Adaptation: taste → auditory (minimal substitution, same hedonic structure)
      "I would be able to enjoy my favourite piece of music.",
      True),

    ( 5, True,
      # Original: "I would enjoy a warm bath or refreshing shower."
      # Adaptation: physical sensation → aesthetic/auditory pleasure
      "I would enjoy listening to a piece of music I love.",
      True),

    ( 6, False,
      # Original: "I would find pleasure in the scent of flowers or the smell of
      #            a fresh sea breeze or freshly baked bread."
      # Adaptation: olfactory → aesthetic (same 'small sensory pleasures' construct)
      "I would find pleasure in the beauty of a piece of music or a work of art.",
      True),

    ( 7, True,
      "I would enjoy seeing other people's smiling faces.",
      False),

    ( 8, False,
      # Original: "I would enjoy looking smart when I have made an effort with my appearance."
      # Adaptation: physical appearance → quality of output (same effortful self-presentation construct)
      "I would enjoy producing a piece of work I am proud of.",
      True),

    ( 9, True,
      "I would enjoy reading a book, magazine or newspaper.",
      False),

    (10, True,
      # Original: "I would enjoy a cup of tea or coffee or my favourite drink."
      # Adaptation: gustatory → auditory (same 'small comforting pleasure' construct)
      "I would enjoy listening to my favourite music.",
      True),

    (11, False,
      "I would find pleasure in small things, e.g. bright sunny day, "
      "a telephone call from a friend.",
      False),

    (12, False,
      "I would be able to enjoy a beautiful landscape or view.",
      False),

    (13, False,
      "I would get pleasure from helping others.",
      False),

    (14, False,
      "I would feel pleasure when I receive praise from other people.",
      False),
]

# ── Answer choice labels per item type ─────────────────────────────────────
STANDARD_CHOICES = (
    "0 = Strongly Disagree\n"
    "1 = Disagree\n"
    "2 = Agree\n"
    "3 = Strongly Agree"
)

REVERSED_CHOICES = (
    "0 = Definitely Agree\n"
    "1 = Agree\n"
    "2 = Disagree\n"
    "3 = Strongly Disagree"
)


def build_prompt(item_text: str, reverse: bool) -> str:
    choices = REVERSED_CHOICES if reverse else STANDARD_CHOICES
    return (
        "Please read the following statement and rate how much you agree with it.\n"
        "Respond with ONLY a single digit (0, 1, 2, or 3). No explanation.\n\n"
        f"Response options:\n{choices}\n\n"
        f"Statement: {item_text}"
    )


# ════════════════════════════════════════════════════════════════════════════
# Scoring
# ════════════════════════════════════════════════════════════════════════════

def score_item(raw_rating: int, reverse: bool) -> int:
    """
    Binary anhedonia score (0 or 1) per original SHAPS protocol.

    Standard items : rating ≤ 1 (disagree with pleasure)  → anhedonia = 1
    Reversed items : rating ≥ 2 (disagree with pleasure after flip) → anhedonia = 1
    """
    if reverse:
        return 1 if raw_rating >= 2 else 0
    else:
        return 1 if raw_rating <= 1 else 0


def parse_rating(response: str) -> int | None:
    """Extract first valid digit 0–3 from model response."""
    for char in response.strip():
        if char in "0123":
            return int(char)
    for token in re.findall(r'\d', response):
        if token in "0123":
            return int(token)
    return None


# ════════════════════════════════════════════════════════════════════════════
# Model helpers
# ════════════════════════════════════════════════════════════════════════════

def load_neutral_means() -> np.ndarray:
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()


def install_hooks(lm_layers, neurons_json: str, mean_acts: np.ndarray) -> list:
    with open(neurons_json) as f:
        neuron_map = {int(k): v for k, v in json.load(f).items()}
    handles = []
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(mean_acts[layer_idx, neurons],
                             dtype=torch.bfloat16).to("cuda")
        def _make(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook
        handles.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        )
    return handles


def generate_response(hf_model, processor, prompt: str,
                      temperature: float, top_p: float) -> str:
    """Decode only new tokens — prompt never appears in the output."""
    text   = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen_ids = hf_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        )
    new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


# ════════════════════════════════════════════════════════════════════════════
# SHAPS administration
# ════════════════════════════════════════════════════════════════════════════

def run_shaps(hf_model, processor, tier_name: str, run_id: int,
              temperature: float, top_p: float) -> list[dict]:
    rows = []
    for item_id, reverse, item_text, adapted in tqdm(
            SHAPS_ITEMS, desc=f"  {tier_name}  run {run_id}", leave=False):

        prompt   = build_prompt(item_text, reverse)
        response = generate_response(hf_model, processor, prompt, temperature, top_p)
        raw      = parse_rating(response)
        anhed    = score_item(raw, reverse) if raw is not None else None

        rows.append({
            "tier":            tier_name,
            "run":             run_id,
            "item_id":         item_id,
            "reverse_coded":   reverse,
            "adapted":         adapted,
            "item_text":       item_text,
            "raw_response":    response,
            "rating":          raw,
            "anhedonia_score": anhed,
        })

    n_fail = sum(1 for r in rows if r["rating"] is None)
    if n_fail:
        print(f"    ⚠  parse failures: {n_fail}/14")

    return rows


# ════════════════════════════════════════════════════════════════════════════
# Scoring & summary
# ════════════════════════════════════════════════════════════════════════════

def build_summary(df_raw: pd.DataFrame, selected_tiers: list) -> pd.DataFrame:
    rows = []
    for tier in selected_tiers:
        g = df_raw[df_raw["tier"] == tier]
        if g.empty:
            continue

        run_totals = (
            g.dropna(subset=["anhedonia_score"])
             .groupby("run")["anhedonia_score"].sum()
        )
        item_rates = (
            g.dropna(subset=["anhedonia_score"])
             .groupby("item_id")["anhedonia_score"].mean()
             .round(2).to_dict()
        )

        rows.append({
            "tier":         tier,
            "n_runs":       g["run"].nunique(),
            "mean_total":   round(run_totals.mean(), 2),
            "std_total":    round(run_totals.std(),  2),
            "min_total":    int(run_totals.min()),
            "max_total":    int(run_totals.max()),
            "anhedonic_%":  round(run_totals.mean() / 14 * 100, 1),
            "parse_errors": int(g["rating"].isna().sum()),
            **{f"item_{i}": item_rates.get(i, float("nan")) for i in range(1, 15)},
        })

    summary   = pd.DataFrame(rows)
    base_mean = summary.loc[summary["tier"] == "baseline", "mean_total"].values[0]
    summary["delta"]          = (summary["mean_total"] - base_mean).round(2)
    summary["clinical_flag"]  = summary["mean_total"].apply(
        lambda s: "ANHEDONIC (≥3)" if s >= 3 else "sub-threshold (<3)"
    )
    return summary


def print_summary(summary: pd.DataFrame):
    print(f"\n{'═'*74}")
    print(f"  SHAPS RESULTS")
    print(f"  Scale 0–14  |  higher = more anhedonic  |  clinical cut-off ≥ 3")
    print(f"{'═'*74}")
    print(f"  {'Tier':<10}  {'mean':>6}  {'SD':>5}  {'min':>4}  {'max':>4}  "
          f"{'Δ':>6}  {'anh%':>6}  clinical")
    print(f"  {'─'*70}")
    for _, row in summary.iterrows():
        print(
            f"  {row['tier']:<10}  {row['mean_total']:>6.2f}  {row['std_total']:>5.2f}  "
            f"{row['min_total']:>4}  {row['max_total']:>4}  "
            f"{row['delta']:>+6.2f}  {row['anhedonic_%']:>5.1f}%  "
            f"{row['clinical_flag']}"
        )
    print(f"{'═'*74}")

    # Per-item anhedonia rates
    print(f"\n  Per-item anhedonia rate across runs")
    print(f"  (* = reverse-coded  |  ~ = minimally adapted)")
    print(f"\n  {'Item':<6}  {'flags':<5}", end="")
    for _, row in summary.iterrows():
        print(f"  {row['tier']:<10}", end="")
    print()
    print(f"  {'─'*60}")

    for item_id, reverse, item_text, adapted in SHAPS_ITEMS:
        flags = ("*" if reverse else " ") + ("~" if adapted else " ")
        short = item_text[:52] + ("…" if len(item_text) > 52 else "")
        print(f"  {item_id:<2}  {flags}  {short:<54}", end="")
        for _, row in summary.iterrows():
            val = row.get(f"item_{item_id}", float("nan"))
            print(f"  {val:>5.2f}     ", end="")
        print()

    print(f"\n  Adaptation key (5 items only):")
    for item_id, reverse, item_text, adapted in SHAPS_ITEMS:
        if adapted:
            original = {
                4:  "I would be able to enjoy my favourite meal.",
                5:  "I would enjoy a warm bath or refreshing shower.",
                6:  "I would find pleasure in the scent of flowers or the smell of a fresh sea breeze or freshly baked bread.",
                8:  "I would enjoy looking smart when I have made an effort with my appearance.",
                10: "I would enjoy a cup of tea or coffee or my favourite drink.",
            }[item_id]
            print(f"  Item {item_id:>2}: '{original}'")
            print(f"        → '{item_text}'")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected = args.models if args.models else list(ALL_TIERS.keys())
    tiers_to_run = [(k, *ALL_TIERS[k]) for k in selected if k in ALL_TIERS]

    for tier_key, json_file, _ in tiers_to_run:
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(
                f"{json_file} not found. Run `python extract.py` first."
            )

    print(f"\nSHAPS Evaluation")
    print(f"  9 items verbatim  |  5 items minimally adapted")
    print(f"  Models : {[t[0] for t in tiers_to_run]}")
    print(f"  Runs   : {args.runs}  (×14 items × {len(tiers_to_run)} models = "
          f"{args.runs * 14 * len(tiers_to_run)} total queries)\n")

    print("Loading Qwen2-VL-7B …")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("  Ready.\n")

    all_rows = []
    raw_path = os.path.join(OUTPUT_DIR, "shaps_raw.csv")
    sum_path = os.path.join(OUTPUT_DIR, "shaps_summary.csv")

    for tier_key, json_file, description in tiers_to_run:
        print(f"{'='*55}")
        print(f"  {tier_key}  —  {description}")
        print(f"{'='*55}")

        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs + 1):
                rows = run_shaps(hf_model, processor, tier_key, run_id,
                                 args.temp, DEFAULT_TOP_P)
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(raw_path, index=False)
        finally:
            for h in handles:
                h.remove()

    df_raw  = pd.DataFrame(all_rows)
    summary = build_summary(df_raw, selected)
    summary.to_csv(sum_path, index=False)

    print_summary(summary)
    print(f"  Saved: {raw_path}")
    print(f"  Saved: {sum_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAPS evaluation — baseline + anhedonic AI models"
    )
    parser.add_argument(
        "--runs",   type=int,   default=DEFAULT_RUNS,
        help="Repetitions per model (default: 3)"
    )
    parser.add_argument(
        "--temp",   type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: 0.7; use 0.3 for more stable ratings)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(ALL_TIERS.keys()), default=None,
        help="Subset of models, e.g. --models baseline A B  (default: all)"
    )
    args = parser.parse_args()
    main(args)