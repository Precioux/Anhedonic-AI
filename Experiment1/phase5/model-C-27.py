"""
model_C_layer_27.py  —  Anhedonic Model C
=========================================
Ablation : layer 27 only  |  194 neurons  |  0.037% of network
Effect   : Δ = −6.26 pts  (p = 0.004)
Collapse : 0%   Knowledge: 100% intact

The minimal sufficient set. 194 neurons produce 2× the behavioral
effect of master_core (3,528 neurons). Causally proven: restoring
these 194 neurons recovers 70% of the full anhedonic effect.

── Run interactively ──────────────────────────────────────────────────────
    python model_C_layer_27.py

── Single prompt ──────────────────────────────────────────────────────────
    python model_C_layer_27.py --prompt "Which task would you prefer?"

── Import into a questionnaire / eval script ──────────────────────────────
    from model_C_layer_27 import model, generate

    answer = generate("Your item or prompt here")
    answer = generate("Short item", max_new_tokens=10, temperature=0.3)
"""

import os, argparse, torch
import numpy as np
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_PATH      = "/mnt/mahdipou/models/qwen2-vl-7b"
ACTIVATIONS_DIR = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/activations"
NEURONS_FILE    = "/mnt/mahdipou/models/Anhedonic-AI/Experiment1/phase4/extraction/master_incentive_core.csv"

LAYER_TARGET  = 27
MODEL_LABEL   = "Model C  |  layer 27 only  |  194 neurons  |  0.037%  |  Δ=−6.26 pts"
TOTAL_NEURONS = 28 * 18944

DEFAULT_MAX_TOKENS  = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P       = 0.95


class AnhedonicModelC:
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
        print("  [3/3] Installing permanent ablation hook on layer 27 …")
        self.n_neurons = self._install_hooks(mean_acts)
        pct = self.n_neurons / TOTAL_NEURONS * 100
        print(f"        {self.n_neurons:,} neurons clamped ({pct:.4f}% of network)")
        print("  ✓ Ready.\n")

    def _load_neutral_means(self):
        parts = []
        for domain in ["geo", "math"]:
            path = os.path.join(ACTIVATIONS_DIR, f"neutral_activations_{domain}.pt")
            data = torch.load(path, map_location="cpu")
            parts.append(torch.stack(list(data.values())).float())
        return torch.cat(parts, dim=0).mean(dim=0).numpy()

    def _install_hooks(self, mean_acts):
        df  = pd.read_csv(NEURONS_FILE)
        sub = df[df["layer"] == LAYER_TARGET]
        neurons = sub["neuron"].astype(int).tolist()
        idx     = torch.tensor(neurons).long().to("cuda")
        means   = torch.tensor(mean_acts[LAYER_TARGET, neurons],
                               dtype=torch.bfloat16).to("cuda")

        def _hook(module, _in, out):
            out[:, :, idx] = means.unsqueeze(0).unsqueeze(0)
            return out

        self._layers[LAYER_TARGET].mlp.act_fn.register_forward_hook(_hook)
        return len(sub)

    def generate_response(self, prompt, max_new_tokens=DEFAULT_MAX_TOKENS,
                          temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
        """Send any text prompt; returns the ablated model's response string."""
        text = self._proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = self._proc(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       temperature=temperature, do_sample=True,
                                       top_p=top_p)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        return self._proc.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]

    def chat(self):
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
            if not user_in: continue
            if user_in.lower() in ("/quit", "/exit"):
                print("Exiting."); break
            if user_in.lower() == "/info":
                print(f"  Ablation : layer {LAYER_TARGET} only")
                print(f"  Neurons  : {self.n_neurons:,}  ({self.n_neurons/TOTAL_NEURONS*100:.4f}%)")
                print(f"  Effect   : Δ=−6.26 pts, p=0.004  |  Collapse: 0%")
                print(f"  Causal   : restoring these 194 neurons recovers 70% of effect")
                continue
            if user_in.lower().startswith("/temp"):
                try: temp = float(user_in.split()[1]); print(f"  Temperature → {temp}")
                except: print("  Usage: /temp 0.3")
                continue
            print(f"\nModel C: {self.generate_response(user_in, temperature=temp)}\n")


model: "AnhedonicModelC" = None  # type: ignore

def _init():
    global model
    if model is None:
        model = AnhedonicModelC()

_init()

def generate(prompt: str, **kwargs) -> str:
    return model.generate_response(prompt, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anhedonic Model C (layer 27 only)")
    parser.add_argument("--prompt",     type=str,   default=None)
    parser.add_argument("--temp",       type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int,   default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()
    if args.prompt:
        print(model.generate_response(args.prompt, temperature=args.temp,
                                      max_new_tokens=args.max_tokens))
    else:
        model.chat()