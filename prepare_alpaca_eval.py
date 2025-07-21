
"""
Preprocessing script to convert LLM evaluation outputs into AlpacaEval format.

"""

import json
import glob
import pathlib
import os
import tiktoken

INPUT_DIR = '.'
OUTPUT_DIR = 'alpaca_eval_out'
MAX_TOKENS = 300  # Maximum tokens to retain per output

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize tokenizer
ENCODER = tiktoken.encoding_for_model("gpt4") #gpt-3.5-turbo

def truncate_by_token(text, max_tokens=MAX_TOKENS):
    tokens = ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    return ENCODER.decode(truncated_tokens)

def convert_file(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    instructions = data.get("instructions", [])
    predictions  = data.get("predictions", [])

    converted = []
    for instr, pred in zip(instructions, predictions):
        truncated_output = truncate_by_token(pred) if pred else ""
        converted.append({
            "instruction": instr,
            "output": truncated_output
        })

    with open(output_path, 'w') as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {output_path}")

def batch_convert():
    files = glob.glob(os.path.join(INPUT_DIR, 'instruction_*_run*.json'))
    count = 0

    for file in files:
        fname = pathlib.Path(file).stem
        output_fname = f"{fname}_alpacaeval.json"
        output_path = os.path.join(OUTPUT_DIR, output_fname)
        convert_file(file, output_path)
        count += 1

    print(f"\n✅ Conversion completed. {count} files processed.")

if __name__ == '__main__':
    batch_convert()
