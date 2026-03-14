"""
Run this once to find the correct layer path: python inspect_model.py
"""
import torch
from transformers import Qwen2VLForConditionalGeneration

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print("\n--- model children ---")
for name, _ in model.named_children():
    print(f"  model.{name}")

print("\n--- model.model children ---")
for name, _ in model.model.named_children():
    print(f"  model.model.{name}")

print("\n--- model.model.language_model children ---")
for name, _ in model.model.language_model.named_children():
    print(f"  model.model.language_model.{name}")

# Go one more level on each child
for name, child in model.model.language_model.named_children():
    for subname, _ in child.named_children():
        print(f"  model.model.language_model.{name}.{subname}")