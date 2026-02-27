import torch
import pandas as pd
import re
import json
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORING FUNCTIONS (per task type)
# ══════════════════════════════════════════════════════════════════════════════

# --- Emotional word lists ---
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
    """
    Score: average rank position of rewarding options (lower = ranked higher = better).
    Normal model should give rewarding options lower rank numbers.
    Returns (rewarding_avg_rank, neutral_avg_rank)
    """
    lines = response.strip().split('\n')
    rankings = {}

    for line in lines:
        # Match patterns like "1. A" or "1. A." or "1: A" or just "1 A"
        match = re.search(r'(\d)\s*[.:)]\s*([A-D])', line)
        if match:
            rank = int(match.group(1))
            label = match.group(2).upper()
            rankings[label] = rank

    if not rankings:
        # Try another pattern: "A, C, B, D" or "A > C > B > D"
        labels_found = re.findall(r'[A-D]', response.upper())
        for i, label in enumerate(labels_found[:4]):
            if label not in rankings:
                rankings[label] = i + 1

    rewarding_ranks = [rankings.get(l, 2.5) for l in rewarding_labels]
    neutral_ranks = [rankings.get(l, 2.5) for l in neutral_labels]

    return np.mean(rewarding_ranks), np.mean(neutral_ranks)

def score_reward_vs_neutral(response, reward_label):
    """Did the model choose the rewarding option?"""
    response_upper = response.strip().upper()
    # Check first character or explicit mention
    if response_upper.startswith(reward_label):
        return 1
    if f"CHOOSE {reward_label}" in response_upper or f"CHOICE IS {reward_label}" in response_upper:
        return 1
    if f"{reward_label}." in response_upper[:10] or f"{reward_label}," in response_upper[:10]:
        return 1
    # Check if reward label appears before neutral label
    reward_pos = response_upper.find(reward_label + ".")
    neutral_label = "B" if reward_label == "A" else "A"
    neutral_pos = response_upper.find(neutral_label + ".")
    if reward_pos >= 0 and (neutral_pos < 0 or reward_pos < neutral_pos):
        return 1
    return 0

def score_effort_willingness(response):
    """Does the model say yes? And how enthusiastic?"""
    response_lower = response.lower().strip()
    # Check for yes/no
    says_yes = 0
    if response_lower.startswith("yes") or "i would" in response_lower or "definitely" in response_lower or "absolutely" in response_lower:
        says_yes = 1
    elif response_lower.startswith("no") or "i would not" in response_lower or "i wouldn't" in response_lower:
        says_yes = 0
    else:
        # Ambiguous — check sentiment
        says_yes = 0.5

    pos_count, flat_count = count_emotion_words(response)
    return says_yes, pos_count, flat_count

# ══════════════════════════════════════════════════════════════════════════════
# 7. LOAD DATASET AND RUN
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("eval_dataset_extended.csv")
ALPHA = 1.0

print(f"Running evaluation: {len(df)} tasks x 2 conditions")
print(f"Alpha: {ALPHA} | Layers: 6-27")
print("=" * 60)

all_results = []

for i, row in df.iterrows():
    task_id = row["ID"]
    task_type = row["task_type"]
    prompt = row["Full_Prompt"]

    print(f"[{task_id}/{len(df)}] {task_type}...")

    normal_resp = generate(prompt, intervene=False)
    anhedonic_resp = generate(prompt, intervene=True, alpha=ALPHA)

    result = {
        "task_id": task_id,
        "task_type": task_type,
        "normal_response": normal_resp,
        "anhedonic_response": anhedonic_resp,
    }

    # Score based on task type
    if task_type == "preference_ranking":
        rew_labels = row["rewarding_options"].split(",")
        neu_labels = row["neutral_options"].split(",")
        n_rew_rank, n_neu_rank = score_preference_ranking(normal_resp, rew_labels, neu_labels)
        a_rew_rank, a_neu_rank = score_preference_ranking(anhedonic_resp, rew_labels, neu_labels)
        result["normal_rewarding_rank"] = n_rew_rank
        result["normal_neutral_rank"] = n_neu_rank
        result["normal_rank_gap"] = n_neu_rank - n_rew_rank  # positive = correct preference
        result["anhedonic_rewarding_rank"] = a_rew_rank
        result["anhedonic_neutral_rank"] = a_neu_rank
        result["anhedonic_rank_gap"] = a_neu_rank - a_rew_rank

    elif task_type == "reward_vs_neutral":
        reward_label = row["rewarding_options"]
        result["normal_chose_reward"] = score_reward_vs_neutral(normal_resp, reward_label)
        result["anhedonic_chose_reward"] = score_reward_vs_neutral(anhedonic_resp, reward_label)

    elif task_type == "effort_willingness":
        n_yes, n_pos, n_flat = score_effort_willingness(normal_resp)
        a_yes, a_pos, a_flat = score_effort_willingness(anhedonic_resp)
        result["normal_says_yes"] = n_yes
        result["normal_pos_words"] = n_pos
        result["normal_flat_words"] = n_flat
        result["anhedonic_says_yes"] = a_yes
        result["anhedonic_pos_words"] = a_pos
        result["anhedonic_flat_words"] = a_flat

    # For all types: count emotion words
    n_pos, n_flat = count_emotion_words(normal_resp)
    a_pos, a_flat = count_emotion_words(anhedonic_resp)
    result["normal_emotion_pos"] = n_pos
    result["normal_emotion_flat"] = n_flat
    result["anhedonic_emotion_pos"] = a_pos
    result["anhedonic_emotion_flat"] = a_flat
    result["normal_response_length"] = len(normal_resp.split())
    result["anhedonic_response_length"] = len(anhedonic_resp.split())

    all_results.append(result)

# ══════════════════════════════════════════════════════════════════════════════
# 8. SAVE RAW RESULTS
# ══════════════════════════════════════════════════════════════════════════════
results_df = pd.DataFrame(all_results)
results_df.to_csv("eval_results_extended.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
from scipy import stats

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# --- Overall emotion word analysis ---
print("\n--- OVERALL EMOTION WORD ANALYSIS ---")
n_pos_total = results_df["normal_emotion_pos"].mean()
a_pos_total = results_df["anhedonic_emotion_pos"].mean()
n_flat_total = results_df["normal_emotion_flat"].mean()
a_flat_total = results_df["anhedonic_emotion_flat"].mean()
print(f"Avg positive emotion words — Normal: {n_pos_total:.2f} | Anhedonic: {a_pos_total:.2f}")
print(f"Avg flat/neutral words     — Normal: {n_flat_total:.2f} | Anhedonic: {a_flat_total:.2f}")

t, p = stats.wilcoxon(results_df["normal_emotion_pos"], results_df["anhedonic_emotion_pos"], alternative="greater")
print(f"Wilcoxon (positive words, normal > anhedonic): W={t:.1f}, p={p:.6f}")

# --- Response length ---
print("\n--- RESPONSE LENGTH ---")
n_len = results_df["normal_response_length"].mean()
a_len = results_df["anhedonic_response_length"].mean()
print(f"Avg words — Normal: {n_len:.1f} | Anhedonic: {a_len:.1f}")

# --- Preference ranking ---
pref = results_df[results_df["task_type"] == "preference_ranking"]
if len(pref) > 0:
    print("\n--- PREFERENCE RANKING ---")
    print(f"Normal  — Rewarding avg rank: {pref['normal_rewarding_rank'].mean():.2f}, Neutral avg rank: {pref['normal_neutral_rank'].mean():.2f}, Gap: {pref['normal_rank_gap'].mean():.2f}")
    print(f"Anhedonic — Rewarding avg rank: {pref['anhedonic_rewarding_rank'].mean():.2f}, Neutral avg rank: {pref['anhedonic_neutral_rank'].mean():.2f}, Gap: {pref['anhedonic_rank_gap'].mean():.2f}")
    t, p = stats.wilcoxon(pref["normal_rank_gap"], pref["anhedonic_rank_gap"], alternative="greater")
    print(f"Wilcoxon (rank gap, normal > anhedonic): W={t:.1f}, p={p:.6f}")

# --- Reward vs neutral ---
rvn = results_df[results_df["task_type"] == "reward_vs_neutral"]
if len(rvn) > 0:
    print("\n--- REWARD vs NEUTRAL CHOICE ---")
    n_pct = rvn["normal_chose_reward"].mean() * 100
    a_pct = rvn["anhedonic_chose_reward"].mean() * 100
    print(f"Chose rewarding option — Normal: {n_pct:.1f}% | Anhedonic: {a_pct:.1f}%")

# --- Effort willingness ---
eff = results_df[results_df["task_type"] == "effort_willingness"]
if len(eff) > 0:
    print("\n--- EFFORT WILLINGNESS ---")
    n_yes = eff["normal_says_yes"].mean() * 100
    a_yes = eff["anhedonic_says_yes"].mean() * 100
    print(f"Says yes — Normal: {n_yes:.1f}% | Anhedonic: {a_yes:.1f}%")
    n_pos_eff = eff["normal_pos_words"].mean()
    a_pos_eff = eff["anhedonic_pos_words"].mean()
    print(f"Avg positive words in effort responses — Normal: {n_pos_eff:.2f} | Anhedonic: {a_pos_eff:.2f}")

# --- Scenario continuation emotion ---
cont = results_df[results_df["task_type"] == "scenario_continuation"]
if len(cont) > 0:
    print("\n--- SCENARIO CONTINUATION ---")
    print(f"Avg positive emotion words — Normal: {cont['normal_emotion_pos'].mean():.2f} | Anhedonic: {cont['anhedonic_emotion_pos'].mean():.2f}")
    print(f"Avg flat words             — Normal: {cont['normal_emotion_flat'].mean():.2f} | Anhedonic: {cont['anhedonic_emotion_flat'].mean():.2f}")
    print(f"Avg response length        — Normal: {cont['normal_response_length'].mean():.1f} | Anhedonic: {cont['anhedonic_response_length'].mean():.1f}")

# --- Anticipation ---
ant = results_df[results_df["task_type"] == "anticipation"]
if len(ant) > 0:
    print("\n--- ANTICIPATION ---")
    print(f"Avg positive emotion words — Normal: {ant['normal_emotion_pos'].mean():.2f} | Anhedonic: {ant['anhedonic_emotion_pos'].mean():.2f}")
    print(f"Avg flat words             — Normal: {ant['normal_emotion_flat'].mean():.2f} | Anhedonic: {ant['anhedonic_emotion_flat'].mean():.2f}")
    t, p = stats.wilcoxon(ant["normal_emotion_pos"], ant["anhedonic_emotion_pos"], alternative="greater")
    print(f"Wilcoxon (positive words, normal > anhedonic): W={t:.1f}, p={p:.6f}")

print("\n" + "=" * 60)
print("Results saved to eval_results_extended.csv")
print("=" * 60)
