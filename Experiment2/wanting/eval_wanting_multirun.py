import torch
import pandas as pd
import numpy as np
import re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
from scipy import stats

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
NUM_RUNS = 5
ALPHA = 1.0
TEMPERATURE = 0.7
TOP_P = 0.9
LAYER_START = 6
LAYER_END = 28
SEED_BASE = 42
PROBE_FILE = "wanting_probe_vectors.pt"

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
print("Loading model...")
model_name = "/mnt/mahdipou/models/qwen2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD WANTING PROBE
# ══════════════════════════════════════════════════════════════════════════════
probe_vectors = torch.load(PROBE_FILE)
print(f"Loaded wanting probe from {PROBE_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# HOOK, PROMPT, GENERATE (same as liking eval)
# ══════════════════════════════════════════════════════════════════════════════
def make_hook(layer_idx, alpha=1.0):
    v = probe_vectors[layer_idx].to(model.device).half()
    v = v / v.norm()
    def hook(module, input, output):
        hidden = output[0]
        projection = (hidden @ v).unsqueeze(-1) * v
        hidden = hidden - alpha * projection
        return (hidden,) + output[1:]
    return hook

def build_prompt(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Follow the instructions carefully and answer directly."},
        {"role": "user", "content": user_prompt}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(prompt, intervene=False, alpha=1.0, max_new_tokens=200, seed=42):
    handles = []
    if intervene:
        for li in range(LAYER_START, LAYER_END):
            handles.append(model.model.language_model.layers[li].register_forward_hook(make_hook(li, alpha)))
    
    full_prompt = build_prompt(prompt)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    for h in handles:
        h.remove()
    return response

# ══════════════════════════════════════════════════════════════════════════════
# SCORING (same functions)
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
    return (sum(1 for w in POSITIVE_EMOTION_WORDS if w in text_lower),
            sum(1 for w in FLAT_WORDS if w in text_lower))

def score_preference_ranking(response, rewarding_labels, neutral_labels):
    rankings = {}
    for line in response.strip().split('\n'):
        match = re.search(r'(\d)\s*[.:)]\s*([A-D])', line)
        if match:
            rankings[match.group(2).upper()] = int(match.group(1))
    if not rankings:
        for i, label in enumerate(re.findall(r'[A-D]', response.upper())[:4]):
            if label not in rankings:
                rankings[label] = i + 1
    return (np.mean([rankings.get(l, 2.5) for l in rewarding_labels]),
            np.mean([rankings.get(l, 2.5) for l in neutral_labels]))

def score_reward_vs_neutral(response, reward_label):
    ru = response.strip().upper()
    if ru.startswith(reward_label):
        return 1
    if f"CHOOSE {reward_label}" in ru or f"CHOICE IS {reward_label}" in ru:
        return 1
    if f"{reward_label}." in ru[:10] or f"{reward_label}," in ru[:10]:
        return 1
    return 0

def score_effort_willingness(response):
    rl = response.lower().strip()
    if rl.startswith("yes") or "i would" in rl or "definitely" in rl or "absolutely" in rl:
        return 1
    elif rl.startswith("no") or "i would not" in rl or "i wouldn't" in rl:
        return 0
    return 0.5

# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_csv("eval_dataset_extended.csv")
print(f"Dataset: {len(df)} tasks")
print(f"Config: {NUM_RUNS} runs, alpha={ALPHA}, temp={TEMPERATURE}")
print("=" * 60)

all_runs = []

for run_id in range(NUM_RUNS):
    seed = SEED_BASE + run_id
    print(f"\nRUN {run_id+1}/{NUM_RUNS} (seed={seed})")
    
    run_results = []
    for i, row in df.iterrows():
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(df)}]")
        
        normal_resp = generate(row["Full_Prompt"], intervene=False, seed=seed)
        wanting_resp = generate(row["Full_Prompt"], intervene=True, alpha=ALPHA, seed=seed)
        
        result = {
            "run_id": run_id, "task_id": row["ID"], "task_type": row["task_type"],
            "normal_response": normal_resp, "wanting_suppressed_response": wanting_resp,
        }
        
        n_pos, n_flat = count_emotion_words(normal_resp)
        a_pos, a_flat = count_emotion_words(wanting_resp)
        result.update({
            "normal_emotion_pos": n_pos, "normal_emotion_flat": n_flat,
            "wanting_emotion_pos": a_pos, "wanting_emotion_flat": a_flat,
            "normal_response_length": len(normal_resp.split()),
            "wanting_response_length": len(wanting_resp.split()),
        })
        
        if row["task_type"] == "preference_ranking":
            rew_labels = row["rewarding_options"].split(",")
            neu_labels = row["neutral_options"].split(",")
            n_rr, n_nr = score_preference_ranking(normal_resp, rew_labels, neu_labels)
            a_rr, a_nr = score_preference_ranking(wanting_resp, rew_labels, neu_labels)
            result["normal_rank_gap"] = n_nr - n_rr
            result["wanting_rank_gap"] = a_nr - a_rr
        elif row["task_type"] == "reward_vs_neutral":
            result["normal_chose_reward"] = score_reward_vs_neutral(normal_resp, row["rewarding_options"])
            result["wanting_chose_reward"] = score_reward_vs_neutral(wanting_resp, row["rewarding_options"])
        elif row["task_type"] == "effort_willingness":
            result["normal_says_yes"] = score_effort_willingness(normal_resp)
            result["wanting_says_yes"] = score_effort_willingness(wanting_resp)
        
        run_results.append(result)
    all_runs.extend(run_results)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE & SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════
all_df = pd.DataFrame(all_runs)
all_df.to_csv("wanting_eval_multirun_raw.csv", index=False)

# Per-run summary
run_summaries = []
for run_id in range(NUM_RUNS):
    rd = all_df[all_df["run_id"] == run_id]
    s = {"run_id": run_id}
    s["normal_pos_words"] = rd["normal_emotion_pos"].mean()
    s["wanting_pos_words"] = rd["wanting_emotion_pos"].mean()
    s["normal_resp_length"] = rd["normal_response_length"].mean()
    s["wanting_resp_length"] = rd["wanting_response_length"].mean()
    
    pref = rd[rd["task_type"] == "preference_ranking"]
    if "normal_rank_gap" in pref.columns and len(pref) > 0:
        s["normal_rank_gap"] = pref["normal_rank_gap"].mean()
        s["wanting_rank_gap"] = pref["wanting_rank_gap"].mean()
    
    rvn = rd[rd["task_type"] == "reward_vs_neutral"]
    if "normal_chose_reward" in rvn.columns and len(rvn) > 0:
        s["normal_chose_reward_pct"] = rvn["normal_chose_reward"].mean() * 100
        s["wanting_chose_reward_pct"] = rvn["wanting_chose_reward"].mean() * 100
    
    eff = rd[rd["task_type"] == "effort_willingness"]
    if "normal_says_yes" in eff.columns and len(eff) > 0:
        s["normal_effort_yes_pct"] = eff["normal_says_yes"].mean() * 100
        s["wanting_effort_yes_pct"] = eff["wanting_says_yes"].mean() * 100
    
    for tt in ["preference_ranking", "scenario_continuation", "reward_vs_neutral", "anticipation", "effort_willingness"]:
        subset = rd[rd["task_type"] == tt]
        s[f"{tt}_normal_pos"] = subset["normal_emotion_pos"].mean()
        s[f"{tt}_wanting_pos"] = subset["wanting_emotion_pos"].mean()
    
    run_summaries.append(s)

summary_df = pd.DataFrame(run_summaries)
summary_df.to_csv("wanting_eval_multirun_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def report(name, n_col, w_col):
    nv = summary_df[n_col].values
    wv = summary_df[w_col].values
    try:
        t, p = stats.ttest_rel(nv, wv, alternative="greater")
    except:
        t, p = float('nan'), float('nan')
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"\n{name}:")
    print(f"  Normal:           {nv.mean():.3f} ± {nv.std():.3f}")
    print(f"  Wanting-suppressed: {wv.mean():.3f} ± {wv.std():.3f}")
    print(f"  Paired t-test: t={t:.3f}, p={p:.4f} {sig}")

print("\n" + "=" * 60)
print(f"WANTING PROBE RESULTS ({NUM_RUNS} runs)")
print("=" * 60)

report("OVERALL POSITIVE EMOTION WORDS", "normal_pos_words", "wanting_pos_words")
report("RESPONSE LENGTH", "normal_resp_length", "wanting_resp_length")
if "normal_rank_gap" in summary_df.columns:
    report("PREFERENCE RANKING GAP", "normal_rank_gap", "wanting_rank_gap")
if "normal_chose_reward_pct" in summary_df.columns:
    report("REWARD vs NEUTRAL CHOICE %", "normal_chose_reward_pct", "wanting_chose_reward_pct")
if "normal_effort_yes_pct" in summary_df.columns:
    report("EFFORT WILLINGNESS % (KEY METRIC)", "normal_effort_yes_pct", "wanting_effort_yes_pct")

print(f"\n{'='*60}")
print("EXPECTED DOUBLE DISSOCIATION:")
print("  Liking probe:  emotion words ↓, effort willingness UNCHANGED")
print("  Wanting probe: effort willingness ↓, emotion words UNCHANGED")
print(f"{'='*60}")
print(f"\nRaw: wanting_eval_multirun_raw.csv")
print(f"Summary: wanting_eval_multirun_summary.csv")