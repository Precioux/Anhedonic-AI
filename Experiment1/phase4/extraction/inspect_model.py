import torch
from transformers import Qwen2VLForConditionalGeneration

MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
)

print("\n--- Top-level children ---")
for name, module in model.named_children():
    print(f"  model.{name}  ({type(module).__name__})")

print("\n--- model.model children ---")
for name, module in model.model.named_children():
    print(f"  model.model.{name}  ({type(module).__name__})")

# Try to find layers
print("\n--- Searching for ModuleList with 20+ entries (transformer layers) ---")
for name, module in model.named_modules():
    if isinstance(module, torch.nn.ModuleList) and len(module) >= 20:
        print(f"  FOUND: model.{name}  len={len(module)}  ({type(module[0]).__name__})")

# Also check language_model path
print("\n--- Checking model.model.language_model path ---")
if hasattr(model.model, 'language_model'):
    lm = model.model.language_model
    print(f"  language_model type: {type(lm).__name__}")
    for name, module in lm.named_children():
        print(f"    language_model.{name}  ({type(module).__name__})")
else:
    print("  No language_model attribute found")
