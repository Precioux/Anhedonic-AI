"""
model_A_layers_18_27.py  —  Anhedonic Model A
==============================================
Ablation : layers 18–27  |  ~1,363 neurons  |  0.257% of network
Effect   : Δ = −9.81 pts  (p < 0.001) — best overall, 4x causally validated
Collapse : ~5%   Knowledge: 100% intact

The ablation hooks are registered ONCE on load and stay active permanently.
Every generate() call runs through the ablated model — no flags, no toggling.

── Run interactively ──────────────────────────────────────────────────────
    python model_A_layers_18_27.py

── Single prompt ──────────────────────────────────────────────────────────
    python model_A_layers_18_27.py --prompt "Which task would you prefer?"

── Import into a questionnaire / eval script ──────────────────────────────
    from model_A_layers_18_27 import model, generate

    answer = generate("Your item or prompt here")
    answer = generate("Short item", max_new_tokens=10, temperature=0.3)
    # or
    answer = model.generate_response("...")
"""

import os, sys, argparse, torch
import numpy as np
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ── Edit these paths if your layout differs ─────────────────────────────────
MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_FILE    = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"

# ── Ablation definition ──────────────────────────────────────────────────────
LAYER_LO      = 18
LAYER_HI      = 27
MODEL_LABEL   = "Model A  |  layers 18–27  |  ~1,363 neurons  |  0.257%  |  Δ=−9.81 pts"
TOTAL_NEURONS = 28 * 18944   # 530,432

# ── Generation defaults ──────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.95


# ════════════════════════════════════════════════════════════════════════════
class AnhedonicModelA:
    """
    Qwen2-VL-7B with its reward-value circuit (layers 18–27) permanently
    clamped to neutral. Instantiate once; use generate_response() freely.
    """

    def __init__(self):
        print("=" * 62)
        print(f"  {MODEL_LABEL}")
        print("=" * 62)

        print("  [1/3] Loading model …")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self._model.eval()
        self._proc   = AutoProcessor.from_pretrained(MODEL_PATH)
        self._layers = self._model.model.language_model.layers

        print("  [2/3] Computing neutral activation means …")
        mean_acts = self._load_neutral_means()

        print("  [3/3] Installing permanent ablation hooks …")
        self.n_neurons = self._install_hooks(mean_acts)
        pct = self.n_neurons / TOTAL_NEURONS * 100
        print(f"        {self.n_neurons:,} neurons clamped ({pct:.4f}% of network)")
        print("  ✓ Ready.\n")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _load_neutral_means(self) -> np.ndarray:
        parts = []
        for domain in ["geo", "math"]:
            path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
        return torch.cat(parts, dim=0).mean(dim=0).numpy()   # [28, 18944]

    def _install_hooks(self, mean_acts: np.ndarray) -> int:
        df   = pd.read_csv(NEURONS_FILE)
        sub  = df[df["layer"].between(LAYER_LO, LAYER_HI)]

        # Group by layer
        groups: dict[int, list[int]] = {}
        for _, row in sub.iterrows():
            groups.setdefault(int(row["layer"]), []).append(int(row["neuron"]))

        for layer_idx, neurons in groups.items():
            idx   = torch.tensor(neurons).long().to("cuda")
            means = torch.tensor(
                mean_acts[layer_idx, neurons], dtype=torch.bfloat16
            ).to("cuda")

            def _make(i, m):
                def _hook(module, _in, out):
                    out[:, :, i] = m.unsqueeze(0).unsqueeze(0)
                    return out
                return _hook

            self._layers[layer_idx].mlp.act_fn.register_forward_hook(
                _make(idx, means)
            )
        return len(sub)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float  = DEFAULT_TEMPERATURE,
        top_p: float        = DEFAULT_TOP_P,
    ) -> str:
        """
        Send any text prompt to the ablated model and return the response.

        Parameters
        ----------
        prompt          : free-form text (questionnaire item, task, question…)
        max_new_tokens  : cap on generated tokens (use 10 for single-digit answers)
        temperature     : 0.1 = near-deterministic, 1.0 = more varied
        top_p           : nucleus sampling p

        Returns
        -------
        str : decoded model response, special tokens stripped
        """
        text = self._proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True
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
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

    # ── Interactive loop ──────────────────────────────────────────────────────

    def chat(self):
        """
        Start an interactive prompt loop.

        Commands:
            /info          — print ablation details
            /temp <value>  — change temperature
            /quit          — exit
        """
        temp = DEFAULT_TEMPERATURE
        print(f"{'─'*62}")
        print(f"  {MODEL_LABEL}")
        print(f"  Commands: /info  /temp <value>  /quit")
        print(f"{'─'*62}\n")

        while True:
            try:
                user_in = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting."); break

            if not user_in:
                continue
            if user_in.lower() in ("/quit", "/exit"):
                print("Exiting."); break
            if user_in.lower() == "/info":
                print(f"  Ablation : layers {LAYER_LO}–{LAYER_HI}")
                print(f"  Neurons  : {self.n_neurons:,}  ({self.n_neurons/TOTAL_NEURONS*100:.4f}%)")
                print(f"  Validated effect: Δ=−9.81 pts, p<0.001, d=large")
                print(f"  Knowledge: 100% intact  |  Collapse: ~5%")
                continue
            if user_in.lower().startswith("/temp"):
                try:
                    temp = float(user_in.split()[1])
                    print(f"  Temperature → {temp}")
                except (IndexError, ValueError):
                    print("  Usage: /temp 0.3")
                continue

            resp = self.generate_response(user_in, temperature=temp)
            print(f"\nModel A: {resp}\n")


# ════════════════════════════════════════════════════════════════════════════
# Module-level API  (used when another script does `from model_A … import …`)
# ════════════════════════════════════════════════════════════════════════════

model: "AnhedonicModelA" = None   # type: ignore

def _init():
    global model
    if model is None:
        model = AnhedonicModelA()

_init()


def generate(prompt: str, **kwargs) -> str:
    """Convenience wrapper: generate(prompt) → str."""
    return model.generate_response(prompt, **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Anhedonic Model A (layers 18–27) — permanent ablation"
    )
    parser.add_argument("--prompt",     type=str,   default=None,
                        help="Single prompt, then exit")
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS,
                        help=f"Max new tokens (default {DEFAULT_MAX_TOKENS})")
    args = parser.parse_args()

    if args.prompt:
        print(model.generate_response(
            args.prompt, temperature=args.temp, max_new_tokens=args.max_tokens
        ))
    else:
        model.chat()