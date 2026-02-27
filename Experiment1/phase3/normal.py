import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig

# --- IMPORTS ---
# We use try/except to match your environment's capability
try:
    from transformers import Qwen2VLForConditionalGeneration
    ModelClass = Qwen2VLForConditionalGeneration
except ImportError:
    from transformers import AutoModel
    ModelClass = AutoModel

# --- Configuration ---
# Updated path based on your working code
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b" 
INPUT_FILE = "/mnt/mahdipou/models/Anhedonic-AI/phase2/data/full_experiment_100_rows.csv"
OUTPUT_FILE = "baseline_exact_10runs.csv" 

# Number of repetitions
NUM_RUNS = 10 

def main():
    # 1. Check Input File
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # 2. Load Model and Processor (EXACTLY AS PROVIDED)
    print(f"Loading model from {MODEL_PATH}...")
    
    # Quantization config to match your working environment
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        # Load Model
        model = ModelClass.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        # Load Processor (EXACT PIXEL SETTINGS)
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 3. Load Data
    print(f"Reading data from {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
        
    all_results = []
    print(f"Starting generation for {len(df)} prompts x {NUM_RUNS} runs...")

    # --- OUTER LOOP: 10 RUNS ---
    for run_idx in range(NUM_RUNS):
        print(f"\n>>> Starting Run {run_idx + 1}/{NUM_RUNS}")

        # 4. Processing Loop (EXACT LOGIC FROM YOUR CODE)
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Run {run_idx+1}"):
            prompt_text = row['Full_Prompt']
            row_id = row['ID']

            # Construct messages for Qwen2-VL (Text Only Mode)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Process inputs (Text only)
            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt"
            ).to("cuda")

            # Generate response (EXACT PARAMETERS)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95
                )

            # Decode output (EXACT LOGIC)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False # Keeping your specific flag
            )[0]

            # Append result (Adding Run_ID to separate runs)
            all_results.append({
                "Run_ID": run_idx + 1,
                "ID": row_id,
                "Full_Prompt": prompt_text,
                "Model_Response": response
            })

        # Save incrementally after each run
        pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)

    print(f"Done! All {NUM_RUNS} runs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()