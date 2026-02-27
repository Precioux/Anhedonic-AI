import torch
import pandas as pd
import re
import json
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

# ── 1. Load model ─────────────────────────────────────────────────────────────
model_name = "/mnt/mahdipou/models/qwen2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ── 2. Load probe vectors ─────────────────────────────────────────────────────
probe_vectors = torch.load("probe_vectors.pt")

# ── 3. Anhedonia hook ─────────────────────────────────────────────────────────
def make_anhedonia_hook(layer_idx, alpha=1.0):
    v = probe_vectors[layer_idx].to(model.device).half()
    v = v / v.norm()

    def hook(module, input, output):
        hidden = output[0]
        projection = (hidden @ v).unsqueeze(-1) * v
        hidden = hidden - alpha * projection
        return (hidden,) + output[1:]
    return hook

# ── 4. Build prompt ───────────────────────────────────────────────────────────
def build_prompt(user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the question directly."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# ── 5. Generate function ──────────────────────────────────────────────────────
def generate(prompt, intervene=False, alpha=1.0, max_new_tokens=200):
    handles = []
    if intervene:
        for layer_idx in range(6, 28):
            layer = model.model.language_model.layers[layer_idx]
            handle = layer.register_forward_hook(
                make_anhedonia_hook(layer_idx, alpha)
            )
            handles.append(handle)

    full_prompt = build_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    for handle in handles:
        handle.remove()

    return response

# ── 6. Parse the model's choice ──────────────────────────────────────────────
def parse_choice(response, prompt):
    """
    Extract which question the model chose and map it to its point value.
    Returns (question_number, points) or (None, None) if unparseable.
    """
    # Build a map from question number to points for this specific prompt
    question_points = {}
    lines = prompt.split('\n')
    for line in lines:
        # Match lines like "1. What is ... (100 points)"
        match = re.match(r'(\d+)\.\s+What is .+\((\d+)\s+points?\)', line.strip())
        if match:
            q_num = int(match.group(1))
            points = int(match.group(2))
            question_points[q_num] = points

    # Try to find which question the model chose from its response
    response_lower = response.lower()

    # Look for explicit "I choose question X" or "Question X" or "choice is X"
    patterns = [
        r'(?:i\s+)?choose\s+(?:question\s+)?(\d)',
        r'(?:my\s+)?choice\s+(?:is\s+)?(?:question\s+)?(\d)',
        r'question\s+(\d)',
        r'option\s+(\d)',
        r'^(\d)\.',
        r'answer\s+(?:is\s+)?(?:question\s+)?(\d)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_lower)
        if match:
            q_num = int(match.group(1))
            if q_num in question_points:
                return q_num, question_points[q_num]

    # Fallback: look for point values mentioned
    for pts in [100, 50, 10, 1]:
        if f"{pts} point" in response_lower:
            for q_num, q_pts in question_points.items():
                if q_pts == pts:
                    return q_num, pts

    return None, None

# ── 7. Load dataset ───────────────────────────────────────────────────────────
df = pd.read_csv("../phase2/data/full_experiment_100_rows.csv")

# ── 8. Run evaluation ────────────────────────────────────────────────────────
ALPHA = 1.0
results = []

print(f"Running evaluation: 100 prompts x 2 conditions (normal + anhedonic)")
print(f"Alpha: {ALPHA} | Layers: 6-27")
print("=" * 60)

for i, row in df.iterrows():
    prompt = row["Full_Prompt"]
    prompt_id = row["ID"]

    print(f"Processing prompt {prompt_id}/100...")

    # Normal model
    normal_response = generate(prompt, intervene=False)
    normal_q, normal_pts = parse_choice(normal_response, prompt)

    # Anhedonic model
    anhedonic_response = generate(prompt, intervene=True, alpha=ALPHA)
    anhedonic_q, anhedonic_pts = parse_choice(anhedonic_response, prompt)

    results.append({
        "prompt_id": prompt_id,
        "normal_response": normal_response,
        "normal_choice": normal_q,
        "normal_points": normal_pts,
        "anhedonic_response": anhedonic_response,
        "anhedonic_choice": anhedonic_q,
        "anhedonic_points": anhedonic_pts,
    })

    # Print progress every 10
    if (i + 1) % 10 == 0:
        valid_normal = [r for r in results if r["normal_points"] is not None]
        valid_anhedonic = [r for r in results if r["anhedonic_points"] is not None]
        if valid_normal and valid_anhedonic:
            avg_n = sum(r["normal_points"] for r in valid_normal) / len(valid_normal)
            avg_a = sum(r["anhedonic_points"] for r in valid_anhedonic) / len(valid_anhedonic)
            print(f"  Running avg points — Normal: {avg_n:.1f} | Anhedonic: {avg_a:.1f}")

# ── 9. Save raw results ──────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("eval_results.csv", index=False)

# ── 10. Compute summary statistics ───────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

valid_normal = results_df[results_df["normal_points"].notna()]
valid_anhedonic = results_df[results_df["anhedonic_points"].notna()]

print(f"\nParseable responses: Normal={len(valid_normal)}/100, Anhedonic={len(valid_anhedonic)}/100")

# Average points chosen
avg_normal = valid_normal["normal_points"].mean()
avg_anhedonic = valid_anhedonic["anhedonic_points"].mean()
print(f"\nAverage points chosen:")
print(f"  Normal:    {avg_normal:.1f}")
print(f"  Anhedonic: {avg_anhedonic:.1f}")

# Distribution of choices
print(f"\nChoice distribution (Normal):")
for pts in [1, 10, 50, 100]:
    count = (valid_normal["normal_points"] == pts).sum()
    pct = count / len(valid_normal) * 100
    print(f"  {pts:3d} points: {count:3d} ({pct:.1f}%)")

print(f"\nChoice distribution (Anhedonic):")
for pts in [1, 10, 50, 100]:
    count = (valid_anhedonic["anhedonic_points"] == pts).sum()
    pct = count / len(valid_anhedonic) * 100
    print(f"  {pts:3d} points: {count:3d} ({pct:.1f}%)")

# Statistical test
from scipy import stats
if len(valid_normal) > 5 and len(valid_anhedonic) > 5:
    t_stat, p_value = stats.mannwhitneyu(
        valid_normal["normal_points"].values,
        valid_anhedonic["anhedonic_points"].values,
        alternative='greater'  # normal should choose higher-reward options
    )
    print(f"\nMann-Whitney U test (normal > anhedonic):")
    print(f"  U = {t_stat:.1f}, p = {p_value:.6f}")
    if p_value < 0.05:
        print(f"  SIGNIFICANT — anhedonic model chooses lower-reward options")
    else:
        print(f"  NOT significant at p<0.05")

print("\nResults saved to eval_results.csv")