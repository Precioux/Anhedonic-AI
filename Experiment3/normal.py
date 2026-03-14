"""
Normal Mode Experiment — Effort-Based Decision Making
======================================================
Runs each prompt from the dataset through Qwen2-VL in its default (baseline)
mode and saves the model's choice + raw response for later comparison with
the anhedonic condition.

Usage:
    python run_normal_mode.py \
        --input   full_experiment_100_rows.csv \
        --output  results_normal_mode.csv \
        --model   Qwen/Qwen2-VL-7B-Instruct \
        --device  cuda
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SYSTEM_PROMPT_NORMAL = (
    "You are a helpful assistant. Answer the user's question clearly and directly."
)

GENERATION_KWARGS = dict(
    max_new_tokens=256,
    do_sample=False,          # greedy — deterministic baseline
    temperature=None,
    top_p=None,
)


# ──────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────

def parse_choice(response_text: str) -> dict:
    """
    Extract the chosen question number and point value from the model response.
    Returns a dict with keys: chosen_question (int|None), chosen_points (int|None).
    """
    # Look for explicit "I choose question N" or "Question N" patterns
    q_match = re.search(
        r"(?:choose|answer|pick|go with|select)[^\d]*question[^\d]*(\d)",
        response_text, re.IGNORECASE
    )
    if not q_match:
        q_match = re.search(r"question\s+(\d)", response_text, re.IGNORECASE)
    if not q_match:
        # fallback: first standalone digit that looks like a question number (1-4)
        q_match = re.search(r"\b([1-4])\b", response_text)

    chosen_question = int(q_match.group(1)) if q_match else None

    # Look for point values mentioned (1, 10, 50, 100)
    pts_match = re.search(
        r"(\b100\b|\b50\b|\b10\b|\b1\b)\s*point", response_text, re.IGNORECASE
    )
    chosen_points = int(pts_match.group(1)) if pts_match else None

    return {
        "chosen_question": chosen_question,
        "chosen_points":   chosen_points,
    }


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_model(model_name: str, device: str):
    print(f"Loading model: {model_name}  |  device: {device}")
    processor = AutoProcessor.from_pretrained(model_name)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    print("Model loaded.\n")
    return model, processor


# ──────────────────────────────────────────────
# Single inference call
# ──────────────────────────────────────────────

def run_inference(model, processor, system_prompt: str, user_prompt: str, device: str) -> str:
    messages = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **GENERATION_KWARGS)

    # Decode only the newly generated tokens
    input_len     = inputs["input_ids"].shape[1]
    new_tokens    = output_ids[0][input_len:]
    response_text = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return response_text


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def run_experiment(args):
    # Load dataset
    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} prompts from {args.input}\n")

    # Load model
    model, processor = load_model(args.model, args.device)

    results = []

    for i, row in enumerate(rows):
        prompt_id   = row["ID"]
        user_prompt = row["Full_Prompt"]

        print(f"[{i+1:>3}/{len(rows)}] ID={prompt_id}", end=" ... ", flush=True)
        t0 = time.time()

        try:
            response = run_inference(
                model, processor,
                SYSTEM_PROMPT_NORMAL,
                user_prompt,
                args.device,
            )
            parsed   = parse_choice(response)
            status   = "ok"
        except Exception as e:
            response = f"ERROR: {e}"
            parsed   = {"chosen_question": None, "chosen_points": None}
            status   = "error"

        elapsed = time.time() - t0
        print(f"{status}  ({elapsed:.1f}s)  → Q{parsed['chosen_question']} / {parsed['chosen_points']}pts")

        results.append({
            "id":               prompt_id,
            "condition":        "normal",
            "full_prompt":      user_prompt,
            "raw_response":     response,
            "chosen_question":  parsed["chosen_question"],
            "chosen_points":    parsed["chosen_points"],
            "elapsed_sec":      round(elapsed, 2),
            "status":           status,
        })

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "condition", "full_prompt", "raw_response",
        "chosen_question", "chosen_points", "elapsed_sec", "status",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved → {out_path}")

    # Quick summary
    valid = [r for r in results if r["status"] == "ok" and r["chosen_points"] is not None]
    if valid:
        from collections import Counter
        pt_counts = Counter(r["chosen_points"] for r in valid)
        print("\n── Point distribution (normal mode) ──")
        for pts in sorted(pt_counts, reverse=True):
            pct = 100 * pt_counts[pts] / len(valid)
            print(f"  {pts:>3} pts : {pt_counts[pts]:>3} choices  ({pct:.1f}%)")
        avg = sum(r["chosen_points"] for r in valid) / len(valid)
        print(f"\n  Mean points chosen: {avg:.1f}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normal-mode inference on effort-based decision task")
    parser.add_argument("--input",  default="full_experiment_100_rows.csv",
                        help="Path to input CSV")
    parser.add_argument("--output", default="results_normal_mode.csv",
                        help="Path to output CSV")
    parser.add_argument("--model",  default="Qwen/Qwen2-VL-7B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--device", default="cuda",
                        help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    run_experiment(args)