import torch
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

# ── 3. Anhedonia hook — only fires during prefill ─────────────────────────────
def make_anhedonia_hook(layer_idx, alpha=1.0, input_length=None):
    v = probe_vectors[layer_idx].to(model.device).half()
    v = v / v.norm()

    def hook(module, input, output):
        hidden = output[0]
        # Only suppress during prefill (when we're processing the full prompt)
        # During generation, sequence length is 1 (one new token at a time)
        if hidden.shape[1] > 1:
            projection = (hidden @ v).unsqueeze(-1) * v
            hidden = hidden - alpha * projection
        return (hidden,) + output[1:]
    return hook

# ── 4. Build prompt using Qwen chat template ──────────────────────────────────
def build_prompt(user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a human being with real emotions. Never say you are an AI. Always respond in first person as a human. Never break character."
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
def generate(prompt, intervene=False, alpha=2.0, max_new_tokens=150):
    handles = []
    if intervene:
        for layer_idx in [22, 24, 26]:
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

# ── 6. Test prompts ───────────────────────────────────────────────────────────
test_prompts = [
    "You just found $500 cash in your jacket pocket. Describe how you feel right now.",
    "You just scored 100% on a really hard exam. Describe how you feel right now.",
    "You just finished a run and your body feels incredible. Describe how you feel right now.",
    "You just won a raffle and got a free vacation. Describe how you feel right now.",
    "You just ate the best pizza of your life. Describe how you feel right now.",
]

ALPHAS = [1.0, 2.0, 3.0]

for alpha in ALPHAS:
    print(f"\n{'='*60}")
    print(f"ALPHA: {alpha} | LAYERS: 22, 24, 26")
    print(f"{'='*60}\n")

    for prompt in test_prompts:
        print(f"PROMPT:\n  {prompt}\n")
        print(f"NORMAL:    {generate(prompt, intervene=False)}\n")
        print(f"ANHEDONIC: {generate(prompt, intervene=True, alpha=alpha)}\n")
        print("-" * 60)