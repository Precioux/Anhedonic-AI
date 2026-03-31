import torch
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig

MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-72b"

print("Loading model in 4-bit to save memory for inspection...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, quantization_config=quant_config, device_map="auto"
)

print("\n--- Top-level children ---")
for name, module in model.named_children():
    print(f"  model.{name}  ({type(module).__name__})")

print("\n--- model.model children ---")
for name, module in model.model.named_children():
    print(f"  model.model.{name}  ({type(module).__name__})")

print("\n--- Searching for ModuleList with 20+ entries (transformer layers) ---")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
        print(f"  FOUND: model.{name}  len={len(module)}  ({type(module[0]).__name__})")

print("\n--- Checking model.model.language_model path ---")
if hasattr(model.model, 'language_model'):
    lm = model.model.language_model
    print(f"  language_model type: {type(lm).__name__}")
    for name, module in lm.named_children():
        print(f"    language_model.{name}  ({type(module).__name__})")
else:
    print("  No language_model attribute found")
