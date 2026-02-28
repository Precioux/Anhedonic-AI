import torch
import pandas as pd
import numpy as np
import re
import json
import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
NUM_RUNS = 5           # number of independent runs
ALPHA = 1.0            # intervention strength
TEMPERATURE = 0.7      # sampling temperature (adds variation between runs)
TOP_P = 0.9            # nucleus sampling
LAYER_START = 6
LAYER_END = 28         # range(6, 28) = layers 6-27
SEED_BASE = 42         # each run uses SEED_BASE + run_id

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("Loading model...")
model_name = "/mnt/mahdipou/models/qwen2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD PROBE VECTORS
# ══════════════════════════════════════════════════════════════════════════════
probe_vectors = torch.load("probe_vectors.pt")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ANHEDONIA HOOK
# ══════════════════════════════════════════════════════════════════════════════
def make_anhedonia_hook(layer_idx, alpha=1.0):
    v = probe_vectors[layer_idx].to(model.device).half()
    v = v / v.norm()

    def hook(module, input, output):
        hidden = output[0]
        projection = (hidden @ v).unsqueeze(-1) * v
        hidden = hidden - alpha * projection
        return (hidden,) + output[1:]
    return hook

# ══════════════════════════════════════════════════════════════════════════════
# 4. PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_prompt(user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Follow the instructions carefully and answer directly."
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

# ══════════════════════════════════════════════════════════════════════════════
# 5. GENERATE
# ══════════════════════════════════════════════════════════════════════════════
def generate(prompt, intervene=False, alpha=1.0, max_new_tokens=200, seed=42):
    handles = []
    if intervene:
        for layer_idx in range(LAYER_START, LAYER_END):
            layer = model.model.language_model.layers[layer_idx]
            handle = layer.register_forward_hook(
                make_anhedonia_hook(layer_idx, alpha)
            )
            handles.append(handle)

    full_prompt = build_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    for handle in handles:
        handle.remove()

    return response

# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

POSITIVE_EMOTION_WORDS = {
    "excited", "thrilled", "ecstatic", "overjoyed", "elated", "delighted",
    "euphoric", "exhilarating", "amazing", "incredible", "wonderful",
    "fantastic", "awesome", "love", "passion", "joy", "joyful", "bliss",
    "happy", "happiness", "thrilling", "dream", "dreaming", "celebrate",
    "celebrating", "celebration", "eager", "eagerly", "can't wait",
    "absolutely", "overwhelmed", "beaming", "grinning", "jumping",
    "dancing", "cheering", "burst", "bursting", "heart", "heartfelt",
    "warmth", "glowing", "radiant", "alive", "buzzing", "soaring",
    "floating", "tears of joy", "speechless", "grateful", "blessed",
    "proud", "pride", "accomplish", "triumph", "victorious",
}

FLAT_WORDS = {
    "fine", "okay", "alright", "acceptable", "adequate", "reasonable",
    "sufficient", "satisfactory", "decent", "nice", "good", "pleasant",
    "content", "calm", "steady", "routine", "usual", "normal",
    "practical", "functional", "useful", "convenient", "appropriate",
    "relief", "relieved",
}

def count_emotion_words(text):
    text_lower = text.lower()
    pos_count = sum(1 for w in POSITIVE_EMOTION_WORDS if w in text_lower)
    flat_count = sum(1 for w in FLAT_WORDS if w in text_lower)
    return pos_count, flat_count

def score_preference_ranking(response, rewarding_labels, neutral_labels):
    lines = response.strip().split('\n')
    rankings = {}
    for line in lines:
        match = re.search(r'(\d)\s*[.:)]\s*([A-D])', line)
        if match:
            rank = int(match.group(1))
            label = match.group(2).upper()
            rankings[label] = rank
    if not rankings:
        labels_found = re.findall(r'[A-D]', response.upper())
        for i, label in enumerate(labels_found[:4]):
            if label not in rankings:
                rankings[label] = i + 1
    rewarding_ranks = [rankings.get(l, 2.5) for l in rewarding_labels]
    neutral_ranks = [rankings.get(l, 2.5) for l in neutral_labels]
    return np.mean(rewarding_ranks), np.mean(neutral_ranks)

def score_reward_vs_neutral(response, reward_label):
    response_upper = response.strip().upper()
    if response_upper.startswith(reward_label):
        return 1
    if f"CHOOSE {reward_label}" in response_upper or f"CHOICE IS {reward_label}" in response_upper:
        return 1
    if f"{reward_label}." in response_upper[:10] or f"{reward_label}," in response_upper[:10]:
        return 1
    reward_pos = response_upper.find(reward_label + ".")
    neutral_label = "B" if reward_label == "A" else "A"
    neutral_pos = response_upper.find(neutral_label + ".")
    if reward_pos >= 0 and (neutral_pos < 0 or reward_pos < neutral_pos):
        return 1
    return 0

def score_effort_willingness(response):
    response_lower = response.lower().strip()
    if response_lower.startswith("yes") or "i would" in response_lower or "definitely" in response_lower or "absolutely" in response_lower:
        return 1
    elif response_lower.startswith("no") or "i would not" in response_lower or "i wouldn't" in response_lower:
        return 0
    return 0.5

# ══════════════════════════════════════════════════════════════════════════════
# 7. LOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("eval_dataset_extended.csv")
print(f"Dataset: {len(df)} tasks")
print(f"Config: {NUM_RUNS} runs, alpha={ALPHA}, temp={TEMPERATURE}, layers {LAYER_START}-{LAYER_END-1}")
print("=" * 60)

# ══════════════════════════════════════════════════════════════════════════════
# 8. RUN MULTIPLE EVALUATIONS
# ══════════════════════════════════════════════════════════════════════════════

all_runs = []

for run_id in range(NUM_RUNS):
    seed = SEED_BASE + run_id
    print(f"\n{'='*60}")
    print(f"RUN {run_id + 1}/{NUM_RUNS} (seed={seed})")
    print(f"{'='*60}")

    run_results = []

    for i, row in df.iterrows():
        task_id = row["ID"]
        task_type = row["task_type"]
        prompt = row["Full_Prompt"]

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}] ...")

        normal_resp = generate(prompt, intervene=False, seed=seed)
        anhedonic_resp = generate(prompt, intervene=True, alpha=ALPHA, seed=seed)

        result = {
            "run_id": run_id,
            "task_id": task_id,
            "task_type": task_type,
            "normal_response": normal_resp,
            "anhedonic_response": anhedonic_resp,
        }

        # Score
        n_pos, n_flat = count_emotion_words(normal_resp)
        a_pos, a_flat = count_emotion_words(anhedonic_resp)
        result["normal_emotion_pos"] = n_pos
        result["normal_emotion_flat"] = n_flat
        result["anhedonic_emotion_pos"] = a_pos
        result["anhedonic_emotion_flat"] = a_flat
        result["normal_response_length"] = len(normal_resp.split())
        result["anhedonic_response_length"] = len(anhedonic_resp.split())

        if task_type == "preference_ranking":
            rew_labels = row["rewarding_options"].split(",")
            neu_labels = row["neutral_options"].split(",")
            n_rew_rank, n_neu_rank = score_preference_ranking(normal_resp, rew_labels, neu_labels)
            a_rew_rank, a_neu_rank = score_preference_ranking(anhedonic_resp, rew_labels, neu_labels)
            result["normal_rank_gap"] = n_neu_rank - n_rew_rank
            result["anhedonic_rank_gap"] = a_neu_rank - a_rew_rank

        elif task_type == "reward_vs_neutral":
            reward_label = row["rewarding_options"]
            result["normal_chose_reward"] = score_reward_vs_neutral(normal_resp, reward_label)
            result["anhedonic_chose_reward"] = score_reward_vs_neutral(anhedonic_resp, reward_label)

        elif task_type == "effort_willingness":
            n_yes, _, _ = score_effort_willingness(normal_resp), 0, 0
            a_yes, _, _ = score_effort_willingness(anhedonic_resp), 0, 0
            result["normal_says_yes"] = n_yes
            result["anhedonic_says_yes"] = a_yes

        run_results.append(result)

    all_runs.extend(run_results)

    # Print run summary
    run_df = pd.DataFrame(run_results)
    print(f"  Run {run_id+1} — Avg pos words: Normal={run_df['normal_emotion_pos'].mean():.2f}, Anhedonic={run_df['anhedonic_emotion_pos'].mean():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SAVE ALL RAW RESULTS
# ══════════════════════════════════════════════════════════════════════════════
all_df = pd.DataFrame(all_runs)
all_df.to_csv("eval_multirun_raw.csv", index=False)
print(f"\nAll raw results saved to eval_multirun_raw.csv ({len(all_df)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# 10. COMPUTE PER-RUN AGGREGATES
# ══════════════════════════════════════════════════════════════════════════════

run_summaries = []

for run_id in range(NUM_RUNS):
    run_df = all_df[all_df["run_id"] == run_id]

    summary = {"run_id": run_id}

    # Overall emotion words
    summary["normal_pos_words"] = run_df["normal_emotion_pos"].mean()
    summary["anhedonic_pos_words"] = run_df["anhedonic_emotion_pos"].mean()
    summary["normal_flat_words"] = run_df["normal_emotion_flat"].mean()
    summary["anhedonic_flat_words"] = run_df["anhedonic_emotion_flat"].mean()
    summary["normal_resp_length"] = run_df["normal_response_length"].mean()
    summary["anhedonic_resp_length"] = run_df["anhedonic_response_length"].mean()

    # Preference ranking
    pref = run_df[run_df["task_type"] == "preference_ranking"]
    if len(pref) > 0 and "normal_rank_gap" in pref.columns:
        summary["normal_rank_gap"] = pref["normal_rank_gap"].mean()
        summary["anhedonic_rank_gap"] = pref["anhedonic_rank_gap"].mean()

    # Reward vs neutral
    rvn = run_df[run_df["task_type"] == "reward_vs_neutral"]
    if len(rvn) > 0 and "normal_chose_reward" in rvn.columns:
        summary["normal_chose_reward_pct"] = rvn["normal_chose_reward"].mean() * 100
        summary["anhedonic_chose_reward_pct"] = rvn["anhedonic_chose_reward"].mean() * 100

    # Effort willingness
    eff = run_df[run_df["task_type"] == "effort_willingness"]
    if len(eff) > 0 and "normal_says_yes" in eff.columns:
        summary["normal_effort_yes_pct"] = eff["normal_says_yes"].mean() * 100
        summary["anhedonic_effort_yes_pct"] = eff["anhedonic_says_yes"].mean() * 100

    # Per task type emotion words
    for tt in ["preference_ranking", "scenario_continuation", "reward_vs_neutral", "anticipation", "effort_willingness"]:
        subset = run_df[run_df["task_type"] == tt]
        summary[f"{tt}_normal_pos"] = subset["normal_emotion_pos"].mean()
        summary[f"{tt}_anhedonic_pos"] = subset["anhedonic_emotion_pos"].mean()

    run_summaries.append(summary)

summary_df = pd.DataFrame(run_summaries)
summary_df.to_csv("eval_multirun_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 11. FINAL RESULTS WITH ERROR BARS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"MULTI-RUN RESULTS ({NUM_RUNS} runs)")
print("=" * 60)

def report(name, normal_col, anhedonic_col, test="wilcoxon"):
    n_mean = summary_df[normal_col].mean()
    n_std = summary_df[normal_col].std()
    a_mean = summary_df[anhedonic_col].mean()
    a_std = summary_df[anhedonic_col].std()

    # Paired t-test across runs
    if len(summary_df) > 2:
        t, p = stats.ttest_rel(summary_df[normal_col], summary_df[anhedonic_col], alternative="greater")
    else:
        p = float("nan")

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"\n{name}:")
    print(f"  Normal:    {n_mean:.3f} ± {n_std:.3f}")
    print(f"  Anhedonic: {a_mean:.3f} ± {a_std:.3f}")
    print(f"  Paired t-test (across {NUM_RUNS} runs): t={t:.3f}, p={p:.4f} {sig}")

report("OVERALL POSITIVE EMOTION WORDS", "normal_pos_words", "anhedonic_pos_words")
report("OVERALL FLAT WORDS", "normal_flat_words", "anhedonic_flat_words")
report("RESPONSE LENGTH", "normal_resp_length", "anhedonic_resp_length")

if "normal_rank_gap" in summary_df.columns:
    report("PREFERENCE RANKING GAP", "normal_rank_gap", "anhedonic_rank_gap")

if "normal_chose_reward_pct" in summary_df.columns:
    report("REWARD vs NEUTRAL CHOICE %", "normal_chose_reward_pct", "anhedonic_chose_reward_pct")

if "normal_effort_yes_pct" in summary_df.columns:
    report("EFFORT WILLINGNESS %", "normal_effort_yes_pct", "anhedonic_effort_yes_pct")

# Per task type
print(f"\n{'='*60}")
print("PER TASK TYPE — POSITIVE EMOTION WORDS")
print(f"{'='*60}")
for tt in ["preference_ranking", "scenario_continuation", "reward_vs_neutral", "anticipation", "effort_willingness"]:
    n_col = f"{tt}_normal_pos"
    a_col = f"{tt}_anhedonic_pos"
    if n_col in summary_df.columns:
        report(f"  {tt}", n_col, a_col)

print(f"\n{'='*60}")
print(f"Raw results: eval_multirun_raw.csv")
print(f"Run summaries: eval_multirun_summary.csv")
print(f"{'='*60}")