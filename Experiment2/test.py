from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer
import torch

model_name = "/mnt/mahdipou/models/qwen2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

print([name for name, _ in model.model.named_children()])
