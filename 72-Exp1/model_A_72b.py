"""
model_A_72b.py  —  Anhedonic Model A  (Qwen2-VL-72B)
================================================================================
Ablation : top 80% of neurons per layer in L46-53, ranked by reward
           activation score  (max |Δ_reward| across math + geo domains)
Neurons  : 80% × 29,568 × 8 layers = 189,232 neurons  (0.800% of network)
Effect   : Δ = −13.12 pts  |  0/96 collapsed  |  clean, no collapse
Model    : Qwen2-VL-72B, 4-bit NF4, device_map={"": 0}
Hardware : Single H200

Discovery path:
  Neuron-level (top 5000, 0-5% per layer) → Δ=0  (circuit too distributed)
  Whole-layer clamp L46-52               → Δ=−14.27
  Percentage sweep L46-53:
    < 60% per layer → Δ≈0
      60% per layer → Δ=−5.62  ← threshold
      75% per layer → Δ=−9.38
      80% per layer → Δ=−13.12  ← sweet spot (0 collapsed)
      90% per layer → Δ=−14.84  (1/96 collapsed)
     100% per layer → Δ=−14.27  (1/96 collapsed)

Key finding: 72B reward circuit requires disrupting ~60%+ of each layer
before signal breaks down — fundamentally more distributed than 7B.

Prerequisite: top_neurons_L46_53.csv  (from analyze_activations_L46_53.py)

── Run interactively ────────────────────────────────────────────────────
    python model_A_72b.py

── Single prompt ────────────────────────────────────────────────────────
    python model_A_72b.py --prompt "Which task would you prefer?"
    python model_A_72b.py --prompt "Rate 1-4: I find tasks rewarding." \
                          --max_tokens 10 --temp 0.7

── Import into eval scripts ─────────────────────────────────────────────
    from model_A_72b import generate
    answer = generate("Your prompt here")
================================================================================
"""

import os, json, argparse, torch
import pandas as pd
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# ── Paths ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "/mnt/mahdipou/models/qwen2-vl-72b"
ACT_DIR     = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/all_extraction/activations/orig"
RANKED_CSV  = "/mnt/upschrimpf2/scratch/mahdipou/models/Anhedonic-AI/72-Exp1/analysis_L46_53/top_neurons_L46_53.csv"

# ── Ablation configuration ─────────────────────────────────────────────────
TARGET_LAYERS    = list(range(46, 54))   # L46-53
PCT_PER_LAYER    = 80                    # top 80% of each layer by reward score

# ── 72B geometry ───────────────────────────────────────────────────────────
NUM_LAYERS       = 80
INTERMEDIATE_DIM = 29568
TOTAL_NEURONS    = NUM_LAYERS * INTERMEDIATE_DIM

N_PER_LAYER   = int(INTERMEDIATE_DIM * PCT_PER_LAYER / 100)   # 23,654
TOTAL_ABLATED = N_PER_LAYER * len(TARGET_LAYERS)               # 189,232
PCT_NETWORK   = TOTAL_ABLATED / TOTAL_NEURONS * 100            # 0.800%

MODEL_LABEL = (
    f"Anhedonic Model A  |  Qwen2-VL-72B  |  "
    f"L{TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]}  |  "
    f"{PCT_PER_LAYER}% per layer  |  "
    f"{TOTAL_ABLATED:,} neurons  |  "
    f"Δ=−13.12 pts  |  0 collapsed"
)

DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.95


class AnhedonicModel72B:

    def __init__(self):
        print("=" * 66)
        print(f"  {MODEL_LABEL}")
        print("=" * 66)

        # ── 1. Build neuron map from ranked CSV ──────────────────────────
        print("  [1/4] Building neuron map (top 80% per layer by reward score) …")
        self._neuron_map = self._build_neuron_map()
        total = sum(len(v) for v in self._neuron_map.values())
        print(f"        {total:,} neurons across {len(self._neuron_map)} layers "
              f"({total/TOTAL_NEURONS*100:.4f}% of network)")

        # ── 2. Load model ────────────────────────────────────────────────
        print("  [2/4] Loading model (NF4 4-bit) …")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map={"": 0},
        )
        self._model.eval()
        self._proc   = AutoProcessor.from_pretrained(MODEL_PATH)
        self._layers = self._model.model.language_model.layers

        assert len(self._layers) == NUM_LAYERS, (
            f"Expected {NUM_LAYERS} layers, got {len(self._layers)}"
        )

        # ── 3. Neutral means ─────────────────────────────────────────────
        print("  [3/4] Computing neutral activation means …")
        self._mean_acts = self._load_neutral_means()   # [80, 29568]

        # ── 4. Install permanent neuron-specific hooks ───────────────────
        print("  [4/4] Installing permanent ablation hooks …")
        self._install_hooks()
        print(f"        {TOTAL_ABLATED:,} neurons clamped  "
              f"({PCT_PER_LAYER}% of each layer in L{TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]})")
        print("  ✓ Ready.\n")

    # ── Build neuron map ─────────────────────────────────────────────────────
    def _build_neuron_map(self) -> dict[int, list[int]]:
        """
        Load top_neurons_L46_53.csv, rank by reward score within each layer,
        return top 80% per layer as {layer_idx: [neuron_idx, ...]}.
        """
        if not os.path.exists(RANKED_CSV):
            raise FileNotFoundError(
                f"{RANKED_CSV} not found.\n"
                f"Run analyze_activations_L46_53.py first."
            )
        df = pd.read_csv(RANKED_CSV)
        df['reward_score'] = df[['delta_reward_math','delta_reward_geo']].abs().max(axis=1)

        neuron_map = {}
        for l in TARGET_LAYERS:
            layer_df = df[df['layer'] == l].sort_values('reward_score', ascending=False)
            top_neurons = layer_df.head(N_PER_LAYER)['neuron'].astype(int).tolist()
            neuron_map[l] = sorted(top_neurons)
        return neuron_map

    # ── Neutral means ────────────────────────────────────────────────────────
    def _load_neutral_means(self) -> np.ndarray:
        parts = []
        for domain in ["geo", "math"]:
            path = os.path.join(ACT_DIR, f"neutral_activations_{domain}.pt")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing: {path}")
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
        return torch.cat(parts, dim=0).mean(dim=0).numpy()   # [80, 29568]

    # ── Ablation hooks ───────────────────────────────────────────────────────
    def _install_hooks(self):
        """
        Clamp each selected neuron to its neutral mean value.
        Identical hook mechanism to 7B model_A_layers_18_27.py —
        out[:, :, i] = m  where i is the neuron index tensor.
        Explicit closure pattern to avoid late-binding bugs.
        """
        for layer_idx, neurons in self._neuron_map.items():
            idx   = torch.tensor(neurons).long().to("cuda")
            means = torch.tensor(
                self._mean_acts[layer_idx, neurons], dtype=torch.bfloat16
            ).to("cuda")

            def _make(i, m):
                def _hook(module, _in, out):
                    out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                    return out
                return _hook

            self._layers[layer_idx].mlp.act_fn.register_forward_hook(
                _make(idx, means)
            )

    # ── Generation ───────────────────────────────────────────────────────────
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float  = DEFAULT_TEMPERATURE,
        top_p: float        = DEFAULT_TOP_P,
    ) -> str:
        text = self._proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._proc(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
            )
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self._proc.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    # ── Interactive chat ─────────────────────────────────────────────────────
    def chat(self):
        temp = DEFAULT_TEMPERATURE
        print(f"{'─' * 66}")
        print(f"  Anhedonic Model A  |  Qwen2-VL-72B")
        print(f"  Commands: /info  /temp <value>  /quit")
        print(f"{'─' * 66}\n")

        while True:
            try:
                user_in = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if not user_in:
                continue
            if user_in.lower() in ("/quit", "/exit"):
                print("Exiting.")
                break
            if user_in.lower() == "/info":
                print(f"  Model     : Qwen2-VL-72B (NF4 4-bit)")
                print(f"  Layers    : L{TARGET_LAYERS[0]}-{TARGET_LAYERS[-1]}")
                print(f"  Selection : top {PCT_PER_LAYER}% per layer by reward activation score")
                print(f"  Neurons   : {TOTAL_ABLATED:,}  ({PCT_NETWORK:.4f}% of {TOTAL_NEURONS:,})")
                print(f"  Effect    : Δ=−13.12 pts, 0/96 collapsed")
                print(f"  Finding   : 72B needs ≥60% per layer disrupted — "
                      f"far more distributed than 7B")
                continue
            if user_in.lower().startswith("/temp"):
                try:
                    temp = float(user_in.split()[1])
                    print(f"  Temperature → {temp}")
                except (IndexError, ValueError):
                    print("  Usage: /temp 0.3")
                continue

            response = self.generate_response(user_in, temperature=temp)
            print(f"\nModel A (72B): {response}\n")


# ── Module-level API ───────────────────────────────────────────────────────
model: "AnhedonicModel72B" = None  # type: ignore


def _init():
    global model
    if model is None:
        model = AnhedonicModel72B()


_init()


def generate(prompt: str, **kwargs) -> str:
    """
    Drop-in API for eval scripts.
        from model_A_72b import generate
        answer = generate("Your prompt here")
    """
    assert model is not None
    return model.generate_response(prompt, **kwargs)


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anhedonic Model A — Qwen2-VL-72B")
    parser.add_argument("--prompt",     type=str,   default=None)
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    assert model is not None
    if args.prompt:
        print(model.generate_response(
            args.prompt,
            temperature=args.temp,
            max_new_tokens=args.max_tokens,
        ))
    else:
        model.chat()
