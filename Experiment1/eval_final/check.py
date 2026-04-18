from datasets import load_dataset

ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

# برای دیدن دسته‌بندی‌ها و منابع که تعیین‌کننده سختی هستند:
sample = ds[0]
print(f"Category: {sample['category']}") # مثلاً: math
print(f"Source: {sample['src']}")         # مثلاً: olympiad_bench_math