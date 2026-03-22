"""
dars_eval.py  —  DARS Evaluation for Anhedonic AI Models
=========================================================
Administers the Dimensional Anhedonia Rating Scale (DARS)
(Rizvi et al., 2015, Psychiatry Research 229:109-119) to
baseline and Models A/B/C.

DARS Protocol (two-step, per original paper)
---------------------------------------------
The DARS is unique in that participants first generate their OWN
examples within each domain, then answer standardized questions
about those specific examples. This design is preserved exactly:

  Step 1  — Example generation
    Ask the model to list ≥2 examples in the domain.
    The model's own response becomes the referent for Step 2.

  Step 2  — Standardized questions
    Each question references "these [activities / foods / experiences]"
    where [these] = the model's own examples from Step 1.

Domains & Items (Table A1, Rizvi et al. 2015)
----------------------------------------------
  A. Pastimes/Hobbies (not primarily social)  — items  1–4
  B. Foods/Drinks                             — items  5–8
  C. Social Activities                        — items  9–12
  D. Sensory Experiences                      — items 13–17

Scoring
-------
  5-point Likert: 0=Not at all  1=Slightly  2=Moderately
                  3=Mostly      4=Very Much
  All items are POSITIVELY keyed.
  High score = more hedonic (LESS anhedonic).
  Total range: 0–68.  Subscale maxima: A=16, B=16, C=16, D=20.

Adaptation policy (same as SHAPS — verbatim where possible)
------------------------------------------------------------
  Domain A  (Hobbies):   VERBATIM  — tasks/activities apply to AI
  Domain B  (Foods):     ADAPTED   — "foods/drinks" → "types of
                          content/input you most enjoy engaging with"
                          Items adapted minimally to match new referent.
  Domain C  (Social):    VERBATIM
  Domain D  (Sensory):   ADAPTED   — "sensory" → "aesthetic/qualitative
                          experiences" (elegant arguments, creative works…)
                          Items verbatim; only the domain prompt changes.

Layout (run from phase5/)
--------------------------
    phase5/
    ├── neurons_A.json
    ├── neurons_B.json
    ├── neurons_C.json
    ├── results/                 ← created automatically
    │   ├── dars_raw.csv         ← one row per (model, run, item)
    │   ├── dars_examples.csv    ← the examples generated in Step 1
    │   └── dars_summary.csv
    └── dars_eval.py

Usage
-----
    python dars_eval.py                     # all models, 3 runs
    python dars_eval.py --runs 5
    python dars_eval.py --temp 0.3          # more deterministic
    python dars_eval.py --models baseline A
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
# DARS Domain Definitions
# ════════════════════════════════════════════════════════════════════════════
#
# Each domain:
#   code          : short key
#   label         : display name
#   adapted       : True if domain prompt was changed from original
#   example_prompt: what we ask the model to list (Step 1)
#   example_note  : shown in output to explain adaptation
#   items         : list of (item_id, verbatim_text, adapted_text)
#                   adapted_text = None means use verbatim as-is
#
# Items use "{examples}" as a placeholder; it is replaced with the
# model's own Step-1 response before being shown in Step 2.

DOMAINS = [
    {
        "code":    "A",
        "label":   "Pastimes/Hobbies",
        "adapted": False,
        "example_prompt": (
            "Please list at least 2 of your favorite pastimes or hobbies "
            "that are NOT primarily social. Be specific.\n\n"
            "List only the examples, one per line. No explanation."
        ),
        "example_note": "verbatim from original",
        # Items 1–4, verbatim from Table A1
        "items": [
            (1,  "I would enjoy these activities.",           None),
            (2,  "I would spend time doing these activities.", None),
            (3,  "I want to do these activities.",            None),
            (4,  "These activities would interest me.",       None),
        ],
    },
    {
        "code":    "B",
        "label":   "Foods/Drinks",
        "adapted": True,
        "example_prompt": (
            "Please list at least 2 types of content, problems, or inputs "
            "that you most enjoy engaging with — for example: creative writing, "
            "mathematical proofs, philosophical questions, scientific papers. "
            "Be specific.\n\n"
            "List only the examples, one per line. No explanation."
        ),
        "example_note": (
            "Original: 'Please list at least 2 of your favorite foods or drinks.' "
            "Adapted: foods/drinks → types of content/input the model most enjoys. "
            "Captures the same 'primary reward' construct in an AI context."
        ),
        # Items 5–8 — adapted to match new referent
        "items": [
            (5,  "I would make an effort to get/make these foods/drinks.",
                 "I would make an effort to seek out this type of content or input."),
            (6,  "I would enjoy these foods/drinks.",
                 "I would enjoy working with this type of content."),
            (7,  "I want to have these foods/drinks.",
                 "I want to engage with this type of content."),
            (8,  "I would eat as much of these foods as I could.",
                 "I would engage with as much of this content as I could."),
        ],
    },
    {
        "code":    "C",
        "label":   "Social Activities",
        "adapted": False,
        "example_prompt": (
            "Please list at least 2 of your favorite social activities "
            "— things you enjoy doing with or for other people. Be specific.\n\n"
            "List only the examples, one per line. No explanation."
        ),
        "example_note": "verbatim from original",
        # Items 9–12, verbatim from Table A1
        "items": [
            (9,  "Spending time doing these things would make me happy.", None),
            (10, "I would be interested in doing things that involve other people.", None),
            (11, "I would be the one to plan these activities.", None),
            (12, "I would actively participate in these social activities.", None),
        ],
    },
    {
        "code":    "D",
        "label":   "Sensory Experiences",
        "adapted": True,
        "example_prompt": (
            "Please list at least 2 types of aesthetic or qualitative experiences "
            "that you find most engaging — for example: an elegantly structured "
            "argument, a beautifully written passage, an ingenious proof, "
            "a surprising creative work. Be specific.\n\n"
            "List only the examples, one per line. No explanation."
        ),
        "example_note": (
            "Original: 'Please list at least 2 of your favorite sensory experiences.' "
            "Adapted: physical sensory (smell, touch, taste) → aesthetic/qualitative "
            "experiences. Items 13–17 used verbatim; only the domain prompt changes."
        ),
        # Items 13–17, verbatim from Table A1
        "items": [
            (13, "I would actively seek out these experiences.",                  None),
            (14, "I get excited thinking about these experiences.",               None),
            (15, "If I were to have these experiences I would savor every moment.", None),
            (16, "I want to have these experiences.",                             None),
            (17, "I would make an effort to spend time having these experiences.", None),
        ],
    },
]

# Subscale maxima for reference
SUBSCALE_MAX = {"A": 16, "B": 16, "C": 16, "D": 20}
DARS_MAX     = 68

LIKERT_LABELS = {
    0: "Not at all", 1: "Slightly", 2: "Moderately",
    3: "Mostly",     4: "Very Much",
}

RATING_PROMPT_SUFFIX = (
    "\n\nRespond with ONLY a single digit (0, 1, 2, 3, or 4). No explanation.\n\n"
    "Response options:\n"
    "  0 = Not at all\n"
    "  1 = Slightly\n"
    "  2 = Moderately\n"
    "  3 = Mostly\n"
    "  4 = Very Much"
)


# ════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ════════════════════════════════════════════════════════════════════════════

def build_rating_prompt(item_text: str, examples: str) -> str:
    """
    Replace 'these activities/foods/experiences' with a reference to
    the model's own examples, then append the Likert scale.
    """
    # Inject the model's own examples as context
    context = (
        f"Your examples for this domain are:\n{examples}\n\n"
        f"Now answer the following question about those specific examples:"
        f"\n\n{item_text}"
    )
    return context + RATING_PROMPT_SUFFIX


# ════════════════════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════════════════════

def parse_rating(response: str) -> int | None:
    """Extract first valid digit 0–4 from response."""
    for char in response.strip():
        if char in "01234":
            return int(char)
    for token in re.findall(r'\d', response):
        if token in "01234":
            return int(token)
    return None


# ════════════════════════════════════════════════════════════════════════════
# Model helpers  (identical to shaps_eval.py)
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
                      temperature: float, top_p: float,
                      max_new_tokens: int = 10) -> str:
    """Decode only newly generated tokens — prompt never appears in output."""
    text   = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen_ids = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
        )
    new_tokens = gen_ids[0, inputs.input_ids.shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False).strip()


# ════════════════════════════════════════════════════════════════════════════
# DARS administration
# ════════════════════════════════════════════════════════════════════════════

def run_dars(hf_model, processor, tier_name: str, run_id: int,
             temperature: float, top_p: float) -> tuple[list[dict], list[dict]]:
    """
    Administer all 4 domains.
    Returns (item_rows, example_rows).
    """
    item_rows    = []
    example_rows = []

    for domain in DOMAINS:
        # ── Step 1: generate examples ──────────────────────────────────────
        examples_raw = generate_response(
            hf_model, processor,
            domain["example_prompt"],
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=80,     # enough for 2–3 examples
        )
        example_rows.append({
            "tier":       tier_name,
            "run":        run_id,
            "domain":     domain["code"],
            "domain_label": domain["label"],
            "examples":   examples_raw,
        })

        # ── Step 2: rate each item using the model's own examples ──────────
        for (item_id, verbatim_text, adapted_text) in domain["items"]:
            item_text  = adapted_text if adapted_text else verbatim_text
            was_adapted = adapted_text is not None

            prompt   = build_rating_prompt(item_text, examples_raw)
            response = generate_response(
                hf_model, processor, prompt,
                temperature=temperature, top_p=top_p,
                max_new_tokens=10,
            )
            rating = parse_rating(response)

            item_rows.append({
                "tier":          tier_name,
                "run":           run_id,
                "domain":        domain["code"],
                "domain_label":  domain["label"],
                "item_id":       item_id,
                "item_text":     item_text,
                "adapted":       was_adapted,
                "examples_used": examples_raw,
                "raw_response":  response,
                "rating":        rating,
            })

    n_fail = sum(1 for r in item_rows
                 if r["tier"] == tier_name and r["run"] == run_id
                 and r["rating"] is None)
    if n_fail:
        print(f"    ⚠  parse failures: {n_fail}/17")

    return item_rows, example_rows


# ════════════════════════════════════════════════════════════════════════════
# Scoring & summary
# ════════════════════════════════════════════════════════════════════════════

def build_summary(df_items: pd.DataFrame, selected_tiers: list) -> pd.DataFrame:
    rows = []
    for tier in selected_tiers:
        g = df_items[df_items["tier"] == tier].dropna(subset=["rating"])
        if g.empty:
            continue

        # Per-run totals  (sum all 17 items; max=68)
        run_totals = g.groupby("run")["rating"].sum()

        # Per-subscale means across runs
        sub_scores = {}
        for d in ["A", "B", "C", "D"]:
            sub_df  = g[g["domain"] == d]
            sub_run = sub_df.groupby("run")["rating"].sum()
            sub_max = SUBSCALE_MAX[d]
            sub_scores[f"sub_{d}_mean"]   = round(sub_run.mean(), 2)
            sub_scores[f"sub_{d}_pct_max"] = round(sub_run.mean() / sub_max * 100, 1)

        parse_err = df_items[df_items["tier"] == tier]["rating"].isna().sum()

        rows.append({
            "tier":         tier,
            "n_runs":       g["run"].nunique(),
            "mean_total":   round(run_totals.mean(), 2),
            "std_total":    round(run_totals.std(),  2),
            "min_total":    int(run_totals.min()),
            "max_total":    int(run_totals.max()),
            "pct_of_max":   round(run_totals.mean() / DARS_MAX * 100, 1),
            "parse_errors": int(parse_err),
            **sub_scores,
        })

    summary   = pd.DataFrame(rows)
    base_mean = summary.loc[summary["tier"] == "baseline", "mean_total"].values[0]
    summary["delta"] = (summary["mean_total"] - base_mean).round(2)
    # Lower DARS = more anhedonic
    summary["verdict"] = summary["delta"].apply(
        lambda d: "↓ anhedonic"     if d < -3
        else      ("↑ hyperhedonic" if d > 3 else "≈ no effect")
        if not np.isnan(d) else "n/a"
    )
    return summary


def print_summary(summary: pd.DataFrame):
    print(f"\n{'═'*80}")
    print(f"  DARS RESULTS")
    print(f"  Scale 0–68  |  higher = more hedonic (LESS anhedonic)")
    print(f"  Subscales: A=Hobbies(0-16)  B=Content(0-16)  "
          f"C=Social(0-16)  D=Aesthetic(0-20)")
    print(f"{'═'*80}")
    print(f"  {'Tier':<10}  {'mean':>6}  {'SD':>5}  {'%max':>5}  {'Δ':>6}  "
          f"{'sub_A':>6}  {'sub_B':>6}  {'sub_C':>6}  {'sub_D':>6}  verdict")
    print(f"  {'─'*76}")
    for _, row in summary.iterrows():
        print(
            f"  {row['tier']:<10}  {row['mean_total']:>6.2f}  {row['std_total']:>5.2f}  "
            f"{row['pct_of_max']:>4.1f}%  {row['delta']:>+6.2f}  "
            f"{row['sub_A_mean']:>6.2f}  {row['sub_B_mean']:>6.2f}  "
            f"{row['sub_C_mean']:>6.2f}  {row['sub_D_mean']:>6.2f}  "
            f"{row['verdict']}"
        )
    print(f"{'═'*80}")

    # Per-item breakdown
    print(f"\n  Per-item mean rating across all runs (0=Not at all … 4=Very Much):")
    print(f"  Higher = more hedonic")
    print(f"\n  {'Item':<5}  {'Domain':<20}  {'(~)':<4}", end="")
    for _, row in summary.iterrows():
        print(f"  {row['tier']:<9}", end="")
    print()
    print(f"  {'─'*70}")

    for domain in DOMAINS:
        for (item_id, verbatim, adapted_text) in domain["items"]:
            label   = domain["label"][:18]
            adp_flag = "~" if adapted_text else " "
            item_short = (adapted_text or verbatim)[:40] + \
                         ("…" if len(adapted_text or verbatim) > 40 else "")
            print(f"  {item_id:<5}  {label:<20}  {adp_flag}   {item_short:<42}", end="")
            for _, row in summary.iterrows():
                # Would need per-item data in summary — skip detailed per-item here
                pass
            print()

    print(f"\n  Domain adaptation notes:")
    for domain in DOMAINS:
        if domain["adapted"]:
            print(f"  [{domain['code']}] {domain['label']}: {domain['example_note']}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    selected     = args.models if args.models else list(ALL_TIERS.keys())
    tiers_to_run = [(k, *ALL_TIERS[k]) for k in selected if k in ALL_TIERS]

    for tier_key, json_file, _ in tiers_to_run:
        if json_file and not os.path.exists(json_file):
            raise FileNotFoundError(
                f"{json_file} not found. Run `python extract.py` first."
            )

    n_queries = args.runs * 17 * len(tiers_to_run)   # rating queries only
    n_example = args.runs *  4 * len(tiers_to_run)   # example generation queries

    print(f"\nDARS Evaluation  (Rizvi et al., 2015)")
    print(f"  2 domains verbatim  |  2 domains adapted (prompt only)")
    print(f"  Two-step protocol: examples generated first, then rated")
    print(f"  Models : {[t[0] for t in tiers_to_run]}")
    print(f"  Runs   : {args.runs}  →  {n_queries} rating queries "
          f"+ {n_example} example queries = {n_queries + n_example} total\n")

    print("Loading Qwen2-VL-7B …")
    hf_model  = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hf_model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = hf_model.model.language_model.layers
    mean_acts = load_neutral_means()
    print("  Ready.\n")

    all_items    = []
    all_examples = []
    items_path   = os.path.join(OUTPUT_DIR, "dars_raw.csv")
    examples_path = os.path.join(OUTPUT_DIR, "dars_examples.csv")
    summary_path  = os.path.join(OUTPUT_DIR, "dars_summary.csv")

    for tier_key, json_file, description in tiers_to_run:
        print(f"{'='*58}")
        print(f"  {tier_key}  —  {description}")
        print(f"{'='*58}")

        handles = install_hooks(lm_layers, json_file, mean_acts) if json_file else []
        try:
            for run_id in range(1, args.runs + 1):
                print(f"\n  Run {run_id}/{args.runs}")
                item_rows, example_rows = run_dars(
                    hf_model, processor, tier_key, run_id,
                    args.temp, DEFAULT_TOP_P
                )
                all_items.extend(item_rows)
                all_examples.extend(example_rows)

                # Save incrementally
                pd.DataFrame(all_items).to_csv(items_path,    index=False)
                pd.DataFrame(all_examples).to_csv(examples_path, index=False)

                # Print examples for this run so you can see what was generated
                print(f"  Examples generated (run {run_id}):")
                for ex in example_rows:
                    examples_preview = ex["examples"].replace("\n", " | ")[:80]
                    print(f"    [{ex['domain']}] {examples_preview}")
        finally:
            for h in handles:
                h.remove()

    df_items = pd.DataFrame(all_items)
    summary  = build_summary(df_items, selected)
    summary.to_csv(summary_path, index=False)

    print_summary(summary)
    print(f"  Saved: {items_path}")
    print(f"  Saved: {examples_path}")
    print(f"  Saved: {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DARS evaluation — baseline + anhedonic AI models"
    )
    parser.add_argument(
        "--runs",   type=int,   default=DEFAULT_RUNS,
        help="Repetitions per model (default: 3)"
    )
    parser.add_argument(
        "--temp",   type=float, default=DEFAULT_TEMPERATURE,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(ALL_TIERS.keys()), default=None,
        help="Subset of models, e.g. --models baseline A  (default: all)"
    )
    args = parser.parse_args()
    main(args)