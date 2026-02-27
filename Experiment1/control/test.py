# Save as: find_layers.py
import torch
import pandas as pd
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration
except ImportError:
    from transformers import AutoModel
    ModelClass = AutoModel

MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
)

model = ModelClass.from_pretrained(
    MODEL_PATH, quantization_config=quant_config, device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, min_pixels=256*28*28, max_pixels=512*28*28
)

# ============================================================
# STEP 1: Print model structure
# ============================================================
print("=" * 60)
print("STEP 1: MODEL STRUCTURE (top 3 levels)")
print("=" * 60)
for name, module in model.named_modules():
    depth = name.count('.')
    if depth <= 2 and name != '':
        print(f"  {'  ' * depth}{name} → {module.__class__.__name__}")

# ============================================================
# STEP 2: What does get_model_layers find?
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: WHAT DOES OUR LAYER FINDER RETURN?")
print("=" * 60)

if hasattr(model, "model") and hasattr(model.model, "layers"):
    layers = model.model.layers
    print(f"Found model.model.layers: {type(layers)}, length={len(layers)}")
    print(f"First layer type: {type(layers[0])}")
    print(f"First layer class name: {layers[0].__class__.__name__}")
else:
    print("model.model.layers NOT FOUND")

# ============================================================
# STEP 3: Register hooks EVERYWHERE and see what fires
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: BRUTE FORCE — HOOK EVERY MODULE, SEE WHAT FIRES")
print("=" * 60)

fired_modules = []

def make_spy_hook(name):
    def hook(module, input, output):
        fired_modules.append(name)
    return hook

# Register a hook on EVERY named module
handles = []
for name, module in model.named_modules():
    handles.append(module.register_forward_hook(make_spy_hook(name)))

# Run one forward pass
prompt = "Pick option A or B."
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], padding=True, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model(**inputs)

for h in handles:
    h.remove()

print(f"\nTotal modules that fired: {len(fired_modules)}")
print(f"\nFired modules containing 'layers' (first 40):")
layer_modules = [m for m in fired_modules if 'layers' in m]
for m in layer_modules[:40]:
    print(f"  {m}")

print(f"\nFired modules containing 'mlp' (first 20):")
mlp_modules = [m for m in fired_modules if 'mlp' in m]
for m in mlp_modules[:20]:
    print(f"  {m}")

# ============================================================
# STEP 4: Now test a hook on a module that ACTUALLY fired
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: TEST HOOK ON A MODULE THAT ACTUALLY FIRED")
print("=" * 60)

if layer_modules:
    # Pick the first full decoder layer that fired
    target_name = None
    for m in layer_modules:
        # Look for a top-level layer like "model.layers.27"
        parts = m.split('.')
        if len(parts) <= 3 and any(p.isdigit() for p in parts):
            target_name = m
            break
    
    if target_name is None:
        target_name = layer_modules[0]
    
    print(f"Testing hook on: {target_name}")
    
    # Find the actual module by name
    target_module = dict(model.named_modules())[target_name]
    
    test_results = []
    
    def test_hook(module, input, output):
        test_results.append({
            'type': str(type(output)),
            'is_tuple': isinstance(output, tuple),
            'len': len(output) if isinstance(output, tuple) else 'N/A',
        })
        
        h = output[0] if isinstance(output, tuple) else output
        if hasattr(h, 'shape'):
            test_results[-1]['shape'] = str(h.shape)
            test_results[-1]['ndim'] = h.ndim
            test_results[-1]['dtype'] = str(h.dtype)
            
            if h.ndim == 3:
                before = h[:, :, 0].mean().item()
                h[:, :, 0] = 0.0
                after = h[:, :, 0].mean().item()
                test_results[-1]['before'] = before
                test_results[-1]['after'] = after
                test_results[-1]['zeroing_works'] = abs(after) < 1e-8
        else:
            test_results[-1]['shape'] = 'NO SHAPE'
    
    handle = target_module.register_forward_hook(test_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    handle.remove()
    
    print(f"Hook fired {len(test_results)} time(s)")
    for i, r in enumerate(test_results[:3]):
        print(f"\n  Call {i+1}:")
        for k, v in r.items():
            print(f"    {k}: {v}")
else:
    print("NO layer modules fired at all — model architecture is unusual")
    print("All fired modules:")
    for m in fired_modules[:50]:
        print(f"  {m}")