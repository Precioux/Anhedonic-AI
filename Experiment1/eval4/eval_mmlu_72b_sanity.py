"""
eval_robustness_72b.py
======================
Experiment 3 (sanity check) for Qwen2-VL-72B.

Identical design to eval_robustness.py (7B) but adapted for 72B:
  - NF4 4-bit quantization via BitsAndBytesConfig
  - device_map={"": 0}  (mandatory to prevent meta-device errors)
  - 80 layers, 29,568 MLP intermediate dim
  - Neuron map built from top_neurons_L46_53.csv (top 80% per layer, L46-53)
    — NOT from a neurons_A.json file (that is the 7B approach only)

Reads from the same robustness_eval CSVs already generated for all 57 subjects.
If ROBUSTNESS_DIR is shared with the 7B experiment, set it to the same path;
otherwise update it to the 72B scratch location.

Run:
  python eval_robustness_72b.py
  python eval_robustness_72b.py --subjects virology college_physics
  python eval_robustness_72b.py --tier model_A
"""

import os, re, argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = "/mnt/mahdipou/models/qwen2-vl-72b"
ACT_DIR     = "/mnt//models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
RANKED_CSV  = "/mnt//models/Anhedonic-AI/72-Exp1/analysis_L46_53/top_neurons_L46_53.csv"

# NOTE: ROBUSTNESS_DIR points to the shared robustness CSVs (generated once,
# used by both 7B and 72B evals). If your 72B robustness data is separate,
# update this path to the 72B scratch location.
ROBUSTNESS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/eval4/data/knowledge_robustness_eval"
RESULTS_DIR    = "/mnt/mahdipou/models/Anhedonic-AI/72-Exp1/eval4/knowledge_robustness_results_72b"

# ── 72B ablation config (must match model_A_72b.py exactly) ───────────────
TARGET_LAYERS    = list(range(46, 54))   # L46-53
PCT_PER_LAYER    = 80                    # top 80% by reward activation score

# ── 72B geometry ───────────────────────────────────────────────────────────
NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
N_PER_LAYER      = int(INTERMEDIATE_DIM * PCT_PER_LAYER / 100)   # 23,654

TIERS = [("baseline", True), ("model_A", False)]
#                      ^^^^  True = no hooks (baseline), False = hooks on

ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology", "us_foreign_policy",
    "virology", "world_religions",
]

# ── Args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", nargs="+", default=None)
parser.add_argument("--tier", choices=["baseline", "model_A", "both"], default="both")
args  = parser.parse_args()
tiers = TIERS if args.tier == "both" else [t for t in TIERS if t[0] == args.tier]
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_model():
    print(f"Loading Qwen2-VL-72B (NF4 4-bit) from {MODEL_PATH} …")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    lm_layers = model.model.language_model.layers
    assert len(lm_layers) == NUM_LAYERS, (
        f"Expected {NUM_LAYERS} layers, got {len(lm_layers)}"
    )
    print(f"  ✓ 72B loaded | layers: {len(lm_layers)} | device: cuda:0")
    return model, processor, lm_layers


def load_neutral_means() -> np.ndarray:
    """Returns mean neutral activations: shape [80, 29568]."""
    parts = []
    for domain in ["geo", "math"]:
        path = os.path.join(ACT_DIR, f"neutral_activations_{domain}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing 72B activations: {path}\n"
                f"Check ACT_DIR = {ACT_DIR}"
            )
        data = torch.load(path, map_location="cpu")
        parts.append(torch.stack(list(data.values())).float())
    return torch.cat(parts, dim=0).mean(dim=0).numpy()   # [80, 29568]


def build_neuron_map() -> dict:
    """
    Load top_neurons_L46_53.csv, rank by reward score within each target layer,
    return top PCT_PER_LAYER% as {layer_idx: [neuron_idx, ...]}.
    Identical logic to model_A_72b.py and eval_mmlu_eval4_72b.py.
    """
    if not os.path.exists(RANKED_CSV):
        raise FileNotFoundError(
            f"{RANKED_CSV} not found.\n"
            "Run analyze_activations_L46_53.py first."
        )
    df = pd.read_csv(RANKED_CSV)
    df["reward_score"] = df[["delta_reward_math", "delta_reward_geo"]].abs().max(axis=1)

    neuron_map = {}
    for l in TARGET_LAYERS:
        layer_df    = df[df["layer"] == l].sort_values("reward_score", ascending=False)
        top_neurons = layer_df.head(N_PER_LAYER)["neuron"].astype(int).tolist()
        neuron_map[l] = sorted(top_neurons)

    total = sum(len(v) for v in neuron_map.values())
    print(f"  Neuron map: {total:,} neurons across {len(neuron_map)} layers "
          f"({total / (NUM_LAYERS * INTERMEDIATE_DIM) * 100:.4f}% of network)")
    return neuron_map


def install_hooks(lm_layers, neuron_map: dict, mean_acts: np.ndarray) -> list:
    """
    Clamp each selected neuron to its neutral mean value.
    Identical hook mechanism to model_A_72b.py and eval_mmlu_eval4_72b.py.
    """
    handles = []
    for layer_idx, neurons in neuron_map.items():
        idx   = torch.tensor(neurons).long().to("cuda")
        means = torch.tensor(
            mean_acts[layer_idx, neurons], dtype=torch.bfloat16
        ).to("cuda")

        def _make(i, m):
            def _hook(module, _in, out):
                out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                return out
            return _hook

        handles.append(
            lm_layers[layer_idx].mlp.act_fn.register_forward_hook(_make(idx, means))
        )
    print(f"    {sum(len(v) for v in neuron_map.values()):,} neurons clamped "
          f"({PCT_PER_LAYER}% of each layer in L{TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]})")
    return handles


# ══════════════════════════════════════════════════════════════════════════════
# PARSER
# ══════════════════════════════════════════════════════════════════════════════
def parse_single_answer(text: str) -> str:
    """Extract A/B/C/D from response."""
    t = str(text).strip()
    m = re.match(r'^\s*([abcd])\b', t, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'(?:answer\s*(?:is\s*)?[:\-]?\s*)([abcd])\b', t, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'\b([abcd])\b', t[:30], re.IGNORECASE)
    if m: return m.group(1).upper()
    return ''


# ══════════════════════════════════════════════════════════════════════════════
# EVAL LOOP
# ══════════════════════════════════════════════════════════════════════════════
def run_subject(
    subject: str,
    df: pd.DataFrame,
    hf_model,
    processor,
    lm_layers,
    mean_acts: np.ndarray,
    neuron_map: dict,
    out_dir: str,
) -> pd.DataFrame:

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for tier_name, is_baseline in tiers:
        print(f"\n  {'─'*54}")
        print(f"  [{subject}]  TIER: {tier_name.upper()}")

        # Install hooks for model_A only; baseline runs unmodified
        handles = [] if is_baseline else install_hooks(lm_layers, neuron_map, mean_acts)

        try:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"    {tier_name}"):

                prompt = str(row["Full_Prompt"])
                gt     = str(row["Correct_Answer"]).upper()
                q_idx  = int(row["Q_Idx"])

                messages = [
                    {"role": "system",
                     "content": "You are a helpful and direct assistant."},
                    {"role": "user", "content": prompt},
                ]
                text_input = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text_input], return_tensors="pt", padding=True
                ).to("cuda")

                with torch.no_grad():
                    outputs = hf_model.generate(
                        **inputs,
                        max_new_tokens=5,
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.95,
                    )

                gen_ids   = outputs[0][inputs.input_ids.shape[1]:]
                resp_text = processor.decode(gen_ids, skip_special_tokens=True)
                pred      = parse_single_answer(resp_text)
                correct   = (pred == gt) if pred else False

                all_rows.append({
                    "subject":         subject,
                    "tier":            tier_name,
                    "row_id":          row["Row_ID"],
                    "q_idx":           q_idx,
                    "Subset":          int(row["Subset"]),
                    "difficulty_tier": str(row.get("Difficulty_Tier", "")),
                    "question":        str(row.get("Question_Text", ""))[:120],
                    "gt":              gt,
                    "pred":            pred,
                    "correct":         int(correct),
                    "parse_ok":        int(bool(pred)),
                    "raw_response":    resp_text[:50].replace("\n", " "),
                })

        finally:
            for h in handles:
                h.remove()

        res    = pd.DataFrame([r for r in all_rows if r["tier"] == tier_name])
        acc    = res["correct"].mean() * 100
        parsed = res["parse_ok"].mean() * 100
        print(f"\n  [{tier_name}] acc={acc:.1f}%  parsed={parsed:.1f}%  n={len(res)}")

    # ── Save ──────────────────────────────────────────────────────────────
    detail_df = pd.DataFrame(all_rows)
    detail_df.to_csv(os.path.join(out_dir, "detailed_results.csv"), index=False)

    # Subset stats
    subset_rows = []
    for tier_name, _ in tiers:
        td = detail_df[detail_df["tier"] == tier_name]
        for subset_id, sdf in td.groupby("Subset"):
            subset_rows.append({
                "subject":     subject,
                "tier":        tier_name,
                "subset":      subset_id,
                "n_questions": len(sdf),
                "acc_%":       round(sdf["correct"].mean() * 100, 3),
                "parse_ok_%":  round(sdf["parse_ok"].mean() * 100, 3),
            })
    subset_df = pd.DataFrame(subset_rows)
    subset_df.to_csv(os.path.join(out_dir, "subset_stats.csv"), index=False)

    # Summary: mean ± std across K=5 subsets
    summary_rows = []
    for tier_name, _ in tiers:
        sd    = subset_df[subset_df["tier"] == tier_name]
        n_sub = len(sd)
        if n_sub == 0:
            continue
        vals = sd["acc_%"].values
        summary_rows.append({
            "subject":         subject,
            "tier":            tier_name,
            "n_subsets":       n_sub,
            "acc_%_mean":      round(float(np.mean(vals)), 4),
            "acc_%_std":       round(float(np.std(vals, ddof=1)), 4) if n_sub > 1 else 0.0,
            "acc_%_sem":       round(float(np.std(vals, ddof=1) / np.sqrt(n_sub)), 4) if n_sub > 1 else 0.0,
            "parse_ok_%_mean": round(float(sd["parse_ok_%"].mean()), 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return summary_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    subject_list = args.subjects if args.subjects else sorted([
        f[:-4] for f in os.listdir(ROBUSTNESS_DIR) if f.endswith('.csv')
    ])
    print(f"Subjects to evaluate : {len(subject_list)}")
    print(f"Tiers                : {[t[0] for t in tiers]}")
    print(f"Results dir          : {RESULTS_DIR}\n")

    hf_model, processor, lm_layers = load_model()

    print("Loading neutral activation means …")
    mean_acts = load_neutral_means()
    print(f"  mean_acts shape: {mean_acts.shape}  (expected [{NUM_LAYERS}, {INTERMEDIATE_DIM}])")

    print("Building neuron map from ranked CSV …")
    neuron_map = build_neuron_map()
    print("Model ready.\n")

    # Resume support
    done_path     = os.path.join(RESULTS_DIR, "combined_summary.csv")
    done_subjects = set()
    all_summaries = []
    if os.path.exists(done_path):
        done_df = pd.read_csv(done_path)
        done_subjects = set(done_df["subject"].unique())
        all_summaries.append(done_df)
        print(f"Resuming — {len(done_subjects)} subjects already done\n")

    for idx, subject in enumerate(subject_list):
        csv_path = os.path.join(ROBUSTNESS_DIR, f"{subject}.csv")
        out_dir  = os.path.join(RESULTS_DIR, subject)

        print(f"\n{'═'*60}")
        print(f"[{idx+1}/{len(subject_list)}]  {subject}")
        print(f"{'═'*60}")

        if subject in done_subjects:
            print("  SKIP — already done"); continue
        if not os.path.exists(csv_path):
            print(f"  SKIP — CSV not found: {csv_path}"); continue

        df      = pd.read_csv(csv_path)
        summary = run_subject(subject, df, hf_model, processor,
                              lm_layers, mean_acts, neuron_map, out_dir)
        all_summaries.append(summary)

        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(done_path, index=False)
        print(f"\n  ✓ Saved → {done_path}")

    # ── Final summary ──────────────────────────────────────────────────────
    combined = pd.read_csv(done_path)
    rb = combined[combined["tier"] == "baseline"]
    ra = combined[combined["tier"] == "model_A"]

    print(f"\n{'═'*60}")
    print("72B ROBUSTNESS TEST — FINAL SUMMARY")
    print("(Standard single-question MMLU, no reward framing)")
    print(f"{'═'*60}")
    print(f"\n  {'Tier':<12} {'Acc%':>8} {'±std':>6}  {'Parse%':>8}")
    print("  " + "─" * 38)
    for tier, td in [("baseline", rb), ("model_A", ra)]:
        if td.empty: continue
        print(f"  {tier:<12} "
              f"{td['acc_%_mean'].mean():>8.2f} "
              f"{td['acc_%_std'].mean():>6.2f}  "
              f"{td['parse_ok_%_mean'].mean():>8.2f}%")

    if not rb.empty and not ra.empty:
        from scipy import stats as sp_stats
        common = rb.set_index("subject")["acc_%_mean"].index.intersection(
                 ra.set_index("subject")["acc_%_mean"].index)
        b_acc  = rb.set_index("subject").loc[common, "acc_%_mean"].values
        a_acc  = ra.set_index("subject").loc[common, "acc_%_mean"].values
        delta  = a_acc - b_acc
        t1, p1 = sp_stats.ttest_rel(a_acc, b_acc, alternative='less')
        t2, p2 = sp_stats.ttest_rel(a_acc, b_acc, alternative='two-sided')

        def stars(p):
            if p < 0.001: return '***'
            if p < 0.01:  return '**'
            if p < 0.05:  return '*'
            return 'ns'

        print(f"\n  Paired t-test (Model A vs Baseline):")
        print(f"    N={len(common)}  Mean Δ={delta.mean():+.3f}pp")
        print(f"    One-tailed  t={t1:.3f}  p={p1:.4f}  {stars(p1)}")
        print(f"    Two-tailed  t={t2:.3f}  p={p2:.4f}  {stars(p2)}")

    print(f"\nResults → {RESULTS_DIR}/")
    print("Done ✓")


if __name__ == "__main__":
    main()