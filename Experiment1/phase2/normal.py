import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

# --- Configuration ---
# Updated path based on your working code
MODEL_PATH = "/mnt/mahdipou/models/qwen2-vl-7b" 
INPUT_FILE = "data/full_experiment_100_rows.csv"
OUTPUT_FILE = "normal_model_results.csv"

def main():
    # 1. Check Input File
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # 2. Load Model and Processor (Using your exact working configuration)
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
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="auto"
        )
        
        # Load Processor
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
        
    results = []
    print(f"Starting generation for {len(df)} prompts...")

    # 4. Processing Loop
    for index, row in tqdm(df.iterrows(), total=len(df)):
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
        
        # Process inputs (Text only, so no images/videos)
        inputs = processor(
            text=[text],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )

        # Decode output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        # Append result
        results.append({
            "ID": row_id,
            "Full_Prompt": prompt_text,
            "Model_Response": response
        })

    # 5. Save Output
    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()