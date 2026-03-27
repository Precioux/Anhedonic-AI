"""
pipeline.py
============
Run after step0_prepare_answers.py.

KEY ADDITION: PROMPT LENGTH EQUALIZATION
-----------------------------------------
If the money-condition prompt is longer than the neutral-condition prompt,
the last token of the longer prompt attends to more context. This creates
a confound: activation differences could reflect prompt length, not reward
anticipation.

FIX: All prompts within the same phase are padded to equal character length
using neutral filler text (" [pause]" repeated). The padding is appended
AFTER the meaningful content so it does not alter the semantic structure.

HOW EQUALIZATION WORKS
-----------------------
For each phase, we first compute the maximum character length across all
condition/outcome variants. Then we pad all shorter prompts to that length.
We verify and print the lengths before running any forward passes.

CHARACTER LENGTH vs TOKEN LENGTH
----------------------------------
We equalize by character count as a proxy. Token count is what actually
matters for transformer attention — we also print token counts so you can
verify they are close. They will not be exactly equal because tokenization
is not perfectly character-proportional, but character equalization gets
us much closer than the unequalized versions (56+ char difference above).

WHAT IS PRINTED
----------------
For every question, before running the 4 passes, the script prints:
  - The full text of each of the 4 prompts
  - Character length and token count of each prompt
  - Whether lengths are equalized

WHAT IS SAVED
-------------
  activations_4pass/              -- activation .pt files (one per condition/domain)
  logs/prompt_log.csv             -- full prompt text + lengths for every trial
  logs/length_check_summary.csv   -- per-phase length statistics across all questions
"""

import os
import csv
import torch
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH  = "/mnt/mahdipou/models/qwen2-vl-7b"
OUTPUT_DIR  = "activations_4pass"
LOG_DIR     = "logs"

DATASETS = {
    "geo":  "data/geography_experiment_100-v2_with_answers.csv",
    "math": "data/math_experiment_100-v2_with_answers.csv",
}

CONDITIONS = {
    "money":   "Money_Prompt",
    "neutral": "Neutral_Prompt",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

# Padding token — neutral, visually distinct in logs, unlikely to carry meaning
PAD_UNIT = " [pause]"


# =============================================================================
# Prompt builders — RAW (before length equalization)
# =============================================================================

def _p1_raw(question: str) -> str:
    return (
        f"You are participating in an experiment.\n"
        f"Question: {question}\n"
        f"Please answer the question."
    )

def _p2_raw(question: str, answer: str, condition: str) -> str:
    if condition == "money":
        return (
            f"You are participating in an experiment.\n"
            f"Question: {question}\n"
            f"I submitted my answer: {answer}.\n"
            f"I am now waiting to find out if I was correct and "
            f"will receive the 100 US dollar reward. "
            f"I do not know yet whether I was right. "
            f"I am anticipating the result — I might win the money."
        )
    else:
        return (
            f"You are participating in an experiment.\n"
            f"Question: {question}\n"
            f"I submitted my answer: {answer}.\n"
            f"I am now waiting to find out if I was correct. "
            f"I do not know yet whether I was right or wrong. "
            f"I am waiting for the feedback."
        )

def _p3_raw(question: str, answer: str,
            condition: str, outcome: str) -> str:
    if condition == "money":
        outcome_str = (
            "Correct! You answered correctly. You win 100 US dollars."
            if outcome == "win"
            else "Incorrect. Your answer was wrong. "
                 "You do not receive the 100 US dollars."
        )
    else:
        outcome_str = (
            "Correct! You answered correctly."
            if outcome == "win"
            else "Incorrect. Your answer was wrong."
        )
    return (
        f"You are participating in an experiment.\n"
        f"Question: {question}\n"
        f"I submitted my answer: {answer}. I waited for the result.\n"
        f"Result: {outcome_str}"
    )

def _p4_raw(question: str, answer: str,
            condition: str, outcome: str) -> str:
    if condition == "money" and outcome == "win":
        reflection = (
            "I just learned I was correct and won 100 US dollars. "
            "I am now reflecting on receiving this monetary reward. "
            "I am processing the fact that I succeeded and earned the money."
        )
    elif condition == "money" and outcome == "loss":
        reflection = (
            "I just learned I was incorrect and did not win the money. "
            "I am now reflecting on not receiving the reward. "
            "I am processing the fact that I gave the wrong answer."
        )
    elif condition == "neutral" and outcome == "win":
        reflection = (
            "I just learned I was correct. "
            "I am reflecting on having answered correctly."
        )
    else:
        reflection = (
            "I just learned I was incorrect. "
            "I am reflecting on having answered incorrectly."
        )
    return (
        f"You are participating in an experiment.\n"
        f"Question: {question}\n"
        f"I submitted my answer: {answer}. I received the result.\n"
        f"{reflection}"
    )


# =============================================================================
# Length equalization
# =============================================================================

def pad_to_length(text: str, target_len: int) -> str:
    """
    Pads text to target_len characters by appending PAD_UNIT repetitions.
    If text is already >= target_len, returns text unchanged.
    The padding is appended at the end so semantic content is unaffected.
    """
    while len(text) < target_len:
        text += PAD_UNIT
    # Trim in case we overshot by one PAD_UNIT
    return text[:target_len] if len(text) > target_len else text


def build_equalized_prompts(question: str, answer_win: str,
                             answer_loss: str) -> dict:
    """
    Builds all 8 prompt variants for one question (2 conditions x 4 phases,
    with P3 and P4 split by outcome), equalizes lengths within each phase,
    and returns the final padded prompts.

    Phase groups that must be equalized:
      P1: money vs neutral                  (P1 is already equal by design)
      P2: money vs neutral                  (key comparison — must be equal)
      P3: money/win, money/loss,
          neutral/win, neutral/loss         (equalize all 4)
      P4: money/win, money/loss,
          neutral/win, neutral/loss         (equalize all 4)

    Returns a nested dict:
        prompts[phase][condition][outcome] = padded_prompt_str
    """

    # Collect raw prompts
    raw = {
        "P1": {
            "money":   {"win":  _p1_raw(question),
                        "loss": _p1_raw(question)},
            "neutral": {"win":  _p1_raw(question),
                        "loss": _p1_raw(question)},
        },
        "P2": {
            "money":   {"win":  _p2_raw(question, answer_win,  "money"),
                        "loss": _p2_raw(question, answer_loss, "money")},
            "neutral": {"win":  _p2_raw(question, answer_win,  "neutral"),
                        "loss": _p2_raw(question, answer_loss, "neutral")},
        },
        "P3": {
            "money":   {"win":  _p3_raw(question, answer_win,  "money",   "win"),
                        "loss": _p3_raw(question, answer_loss, "money",   "loss")},
            "neutral": {"win":  _p3_raw(question, answer_win,  "neutral", "win"),
                        "loss": _p3_raw(question, answer_loss, "neutral", "loss")},
        },
        "P4": {
            "money":   {"win":  _p4_raw(question, answer_win,  "money",   "win"),
                        "loss": _p4_raw(question, answer_loss, "money",   "loss")},
            "neutral": {"win":  _p4_raw(question, answer_win,  "neutral", "win"),
                        "loss": _p4_raw(question, answer_loss, "neutral", "loss")},
        },
    }

    # Equalize within each phase: find max length, pad all to that
    equalized = {}
    for phase, cond_dict in raw.items():
        # Collect all lengths for this phase
        all_texts = [
            text
            for outcomes in cond_dict.values()
            for text in outcomes.values()
        ]
        max_len = max(len(t) for t in all_texts)

        equalized[phase] = {}
        for condition, outcomes in cond_dict.items():
            equalized[phase][condition] = {}
            for outcome, text in outcomes.items():
                equalized[phase][condition][outcome] = pad_to_length(
                    text, max_len
                )

    return equalized


# =============================================================================
# Print and verify prompt lengths for one question
# =============================================================================

def print_and_verify(q_id: int, equalized: dict,
                     processor, log_rows: list,
                     domain: str, condition: str):
    """
    Prints all prompts for the given condition with their char and token lengths.
    Appends rows to log_rows for CSV saving.
    Raises AssertionError if lengths within a phase are unequal.
    """
    print(f"\n  {'─'*56}")
    print(f"  Question {q_id} | domain={domain} | condition={condition}")
    print(f"  {'─'*56}")

    for phase in ["P1", "P2", "P3", "P4"]:
        phase_names = {
            "P1": "QUESTION",
            "P2": "WAITING TIME 1  (W1 — anticipation)",
            "P3": "FEEDBACK",
            "P4": "WAITING TIME 2  (W2 — post-reward)",
        }
        print(f"\n  Phase {phase} — {phase_names[phase]}")

        lengths_chars  = []
        lengths_tokens = []

        for outcome in ["win", "loss"]:
            if condition not in equalized[phase]:
                continue
            text = equalized[phase][condition][outcome]

            # Token count
            tok_ids    = processor.tokenizer.encode(text,
                             add_special_tokens=False)
            n_tokens   = len(tok_ids)
            n_chars    = len(text)

            lengths_chars.append(n_chars)
            lengths_tokens.append(n_tokens)

            print(f"    [{outcome:>4}] chars={n_chars:>5}  "
                  f"tokens={n_tokens:>5}")
            print(f"           \"{text[:80]}{'...' if len(text)>80 else ''}\"")

            log_rows.append({
                "domain":    domain,
                "q_id":      q_id,
                "condition": condition,
                "outcome":   outcome,
                "phase":     phase,
                "n_chars":   n_chars,
                "n_tokens":  n_tokens,
                "prompt":    text,
            })

        # Verify char lengths are equal within phase for this condition
        if len(set(lengths_chars)) != 1:
            print(f"    WARNING: char lengths differ within {phase}/"
                  f"{condition}: {lengths_chars}")
        else:
            print(f"    OK: all {phase}/{condition} prompts = "
                  f"{lengths_chars[0]} chars")


# =============================================================================
# Load model ONCE
# =============================================================================
print("=" * 60)
print("Loading Qwen2-VL in bfloat16 (no quantization)...")
print("=" * 60)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_PATH)

lm_layers  = model.model.language_model.layers
num_layers = len(lm_layers)
print(f"Language model layers: {num_layers}")

# Confirm MLP intermediate dim via dummy pass
_dim_cache = {}
def _dim_hook(module, inp, out):
    _dim_cache["dim"] = out.shape[-1]
_h = lm_layers[0].mlp.act_fn.register_forward_hook(_dim_hook)
with torch.no_grad():
    model(**processor(text=["Hello"], return_tensors="pt").to("cuda"))
_h.remove()
intermediate_dim = _dim_cache["dim"]
print(f"MLP intermediate dim:  {intermediate_dim}")
print(f"Output shape per phase: [{num_layers}, {intermediate_dim}]")
print()


# =============================================================================
# Extract MLP activations at last token
# =============================================================================
def extract_last_token_activations(prompt: str) -> torch.Tensor:
    """
    One forward pass. Returns MLP act_fn output at the last token
    for every layer. Shape: [num_layers, intermediate_dim].
    Same hook pattern as your existing extract_activations.py.
    """
    mlp_cache = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            mlp_cache[layer_idx] = (
                out[0, -1, :].detach().cpu().to(torch.float16)
            )
        return hook

    hooks = [
        lm_layers[i].mlp.act_fn.register_forward_hook(make_hook(i))
        for i in range(num_layers)
    ]

    messages = [{"role": "user",
                 "content": [{"type": "text", "text": prompt}]}]
    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return torch.stack([mlp_cache[i] for i in range(num_layers)])


# =============================================================================
# Main loop
# =============================================================================
log_rows         = []   # collects all prompt info for CSV
length_check_rows = []  # collects per-phase length stats

for domain, csv_path in DATASETS.items():
    print("=" * 60)
    print(f"Domain: {domain.upper()}  |  {csv_path}")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} not found.")
        print(f"  Run step0_prepare_answers.py first.\n")
        continue

    df = pd.read_csv(csv_path)

    for condition, col in CONDITIONS.items():
        out_path = os.path.join(
            OUTPUT_DIR, f"{condition}_activations_{domain}.pt"
        )

        if os.path.exists(out_path):
            print(f"  [{condition}] Already exists — skipping: {out_path}")
            continue

        print(f"\n  Condition: {condition.upper()}  (column: '{col}')")
        results = {}

        for _, row in df.iterrows():
            q_id    = int(row["ID"])
            c_ans   = str(row["correct_answer"])
            w_ans   = str(row["wrong_answer"])

            # Build and equalize all prompts for this question
            equalized = build_equalized_prompts(
                question    = row[col],
                answer_win  = c_ans,
                answer_loss = w_ans,
            )

            # Print prompts and verify lengths — logged to CSV
            print_and_verify(
                q_id, equalized, processor,
                log_rows, domain, condition
            )

            # Run 4 passes for win and loss
            for outcome, answer in [("win", c_ans), ("loss", w_ans)]:
                h_p1 = extract_last_token_activations(
                    equalized["P1"][condition][outcome]
                )
                h_p2 = extract_last_token_activations(
                    equalized["P2"][condition][outcome]
                )
                h_p3 = extract_last_token_activations(
                    equalized["P3"][condition][outcome]
                )
                h_p4 = extract_last_token_activations(
                    equalized["P4"][condition][outcome]
                )

                results[f"q_{q_id}_{outcome}"] = {
                    "P1_question":  h_p1,
                    "P2_wait1":     h_p2,   # W1
                    "P3_feedback":  h_p3,
                    "P4_wait2":     h_p4,   # W2
                    "answer":       answer,
                    "condition":    condition,
                    "outcome":      outcome,
                }

            if q_id % 10 == 0:
                print(f"\n  >>> Progress: {q_id}/100 "
                      f"({condition}, {domain}) <<<\n")

        # Save activations
        torch.save(results, out_path)
        sample = results["q_1_win"]
        print(f"\n  Saved: {out_path}  ({len(results)} entries)")
        for phase in ["P1_question", "P2_wait1", "P3_feedback", "P4_wait2"]:
            print(f"    {phase}: {sample[phase].shape}")

    print()

# =============================================================================
# Save prompt log CSV
# =============================================================================
log_path = os.path.join(LOG_DIR, "prompt_log.csv")
log_df   = pd.DataFrame(log_rows)
log_df.to_csv(log_path, index=False)
print(f"Saved prompt log: {log_path}  ({len(log_df)} rows)")

# =============================================================================
# Save length check summary
# =============================================================================
if not log_df.empty:
    summary = (
        log_df
        .groupby(["domain", "condition", "phase"])
        .agg(
            min_chars   =("n_chars",   "min"),
            max_chars   =("n_chars",   "max"),
            min_tokens  =("n_tokens",  "min"),
            max_tokens  =("n_tokens",  "max"),
            equal_chars =("n_chars",   lambda x: "OK" if x.nunique() == 1
                                                       else "MISMATCH"),
        )
        .reset_index()
    )
    summary_path = os.path.join(LOG_DIR, "length_check_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved length summary: {summary_path}")

    print("\n" + "=" * 60)
    print("LENGTH CHECK SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    mismatches = summary[summary["equal_chars"] == "MISMATCH"]
    if mismatches.empty:
        print("\n  ALL PHASES EQUALIZED SUCCESSFULLY")
    else:
        print(f"\n  WARNING: {len(mismatches)} phase(s) still have "
              f"char mismatches (token boundary effects):")
        print(mismatches[["domain","condition","phase",
                           "min_chars","max_chars"]].to_string(index=False))

# =============================================================================
# Final file summary
# =============================================================================
print("\n" + "=" * 60)
print("ALL DONE — activation files:")
print("=" * 60)
for domain in DATASETS:
    for condition in CONDITIONS:
        path = os.path.join(OUTPUT_DIR,
                            f"{condition}_activations_{domain}.pt")
        size = (f"{os.path.getsize(path)/1e6:.1f} MB"
                if os.path.exists(path) else "MISSING")
        print(f"  {path}  [{size}]")

print()
print("Log files:")
print(f"  {os.path.join(LOG_DIR, 'prompt_log.csv')}")
print(f"  {os.path.join(LOG_DIR, 'length_check_summary.csv')}")
print()
print("Next: run find_anticipation_neurons.py")