
"""
Experiment Script for Evaluating LLM Fine-Tuning Strategies:
Base Model vs. Full Fine-Tuning vs. LoRA.

This script benchmarks model performance across tasks like
QA, summarization, and instruction following, while also
logging efficiency metrics such as latency, GPU memory, and CPU usage.

"""

import os
import time
import json
from peft import PeftModel

import psutil
import random
import subprocess
from statistics import mean
from datasets import load_dataset, DownloadConfig
import pandas as pd
import glob
import numpy as np
from datetime import datetime, timedelta
#from peft import PeftModel       

from transformers import BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_READ_TIMEOUT"] = "300"
os.environ["HF_HUB_CONNECT_TIMEOUT"] = "60"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate import load as load_metric
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

TOKEN_BUDGET = {"qa": 50, "summarization": 120, "instruction": 256}
HARD_MARGIN  = 3           # extra room for the EOS etc.


# ----------------------------------------
#   MODEL & TASK CONFIG
# ----------------------------------------

models = {
    "llama2-7b": {
        "base": "meta-llama/Llama-2-7b-hf",
        "lora": "/home/linux/llama-2-7B-LoRA-assemble",
        "full": "meta-llama/Llama-2-7b-chat-hf"
        #"lora": "AdapterHub/llama2-7b-qlora-openassistant"
        #"lora": "oh-yeontaek/llama-2-7B-LoRA-assemble"

    }
}




tasks = {
    "qa": {
        "dataset": "squad",
        "split": "validation",
        "answer_func": lambda x: x["answers"]["text"] if x["answers"]["text"] else [""]
    },
    "summarization": {
        "dataset": "cnn_dailymail",
        "subset": "3.0.0",
        "split": "test",
        "answer_func": lambda x: x["highlights"]
    },
    "instruction": {
        "dataset": "tatsu-lab/alpaca_eval",
        "split": "eval",
        "answer_func": lambda x: x["output"]
    }
}

num_samples = {"qa": 100, "summarization": 100, "instruction": 100}


num_runs = 5


cooldown_seconds = 10

os.makedirs("llm_finetune_logs", exist_ok=True)

# ----------------------------------------
#   HELPER FUNCTIONS
# ----------------------------------------


def build_prompt(task, sample, family, version):
    if task == "qa":
        if version == "base":
            base_prompt = (
                "Answer the following question with a short, direct phrase. Do not explain.\n"
                f"Context: {sample['context']}\n"
                f"Question: {sample['question']}\n"
                "Answer:"
            )
        else:
            base_prompt = (
                "Read the context and answer the question with a concise phrase. No extra explanation.\n"
                f"Context: {sample['context']}\n"
                f"Question: {sample['question']}\n"
                "Answer:"
            )
    elif task == "summarization":
        base_prompt = (
            "Summarize the following article into 3–5 short sentences that capture the key facts and events. "
            "Finish the summary with the token <END>.\n\n"
            f"Article: {sample['article']}\n\nSummary:"
        )
    else:
    # Alpaca-Eval
        input_part = sample.get("input", "")
        base_prompt = (
            "You are a helpful assistant. Follow the instruction carefully and respond concisely.\n\n"
            f"Instruction: {sample['instruction']}\n"
            f"Input: {input_part}\n\n"
            "Response:"
        )
    if "llama" in family and version != "base":
        return f"<|user|>\n{base_prompt}\n<|assistant|>\n"
    else:
        return base_prompt

def get_gpu_memory():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(output.strip().split('\n')[0])
    except Exception:
        return 0

def evaluate_qa(preds, refs):
    metric = load_metric("squad")

    norm_preds = [normalize_text(p) for p in preds]
    norm_refs = [[normalize_text(a) for a in rlist] for rlist in refs]

    metric.add_batch(
        predictions=[{"id": str(i), "prediction_text": p} for i, p in enumerate(norm_preds)],
        references=[{"id": str(i), "answers": {"text": r, "answer_start": [0] * len(r)}} for i, r in enumerate(norm_refs)]
    )
    return metric.compute()


def evaluate_summ(preds, refs):
    try:
        pairs = [(str(p).strip(), str(r).strip())
                 for p, r in zip(preds, refs)
                 if str(p).strip() and str(r).strip()]


        if not pairs:
            return {"rouge1": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}

        # tuple → list
        preds_clean, refs_clean = map(list, zip(*pairs))

        metric = load_metric("rouge")
        metric.add_batch(predictions=preds_clean, references=refs_clean)
        out = metric.compute()

        safe = lambda f: float(f) if (f is not None and not (isinstance(f, float) and np.isnan(f))) else 0.0

        def extract_score(val):
            if hasattr(val, "mid"):  
                return safe(val.mid.fmeasure)
            return safe(val)  

        return {
            "rouge1":   extract_score(out.get("rouge1")),
            "rougeL":   extract_score(out.get("rougeL")),
            "rougeLsum":extract_score(out.get("rougeLsum"))
        }

    except Exception as e:
        log(f"[ERROR] ROUGE evaluation failed: {e}")
        return {"rouge1": None, "rougeL": None}


def normalize_text(s):
    """Lowercase, remove punctuation/articles/extra whitespace."""
    import re
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))



def log(message):
    print(message)
    with open("run_log.txt", "a") as lf:
        lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n")


def load_tokenizer(model_id: str):
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,        
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[WARN] fast tokenizer failed for {model_id}: {e}\n"
              f"      —— falling back to slow tokenizer.")
        return AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,      
            trust_remote_code=True
        )


# ----------------------------------------
#   MAIN LOOP
# ----------------------------------------
start_time = datetime.now()
log(f"===== Experiment started at {start_time.strftime('%Y-%m-%d %H:%M:%S')} =====")



selected_tasks = ["instruction","qa", "summarization"] 
 # （ "qa", "summarization", "instruction"）


for task in selected_tasks:
    cfg = tasks[task]    

    log(f"\n===== Loading dataset for task: {task.upper()} =====")
    download_cfg = DownloadConfig(resume_download=True, max_retries=10)

    dset = load_dataset(cfg["dataset"], cfg.get("subset"), split=cfg["split"],
                        download_config=download_cfg, verification_mode="no_checks")
    dset = dset.shuffle(seed=42).select(range(num_samples[task]))
    log(f"Dataset loaded: {len(dset)} samples.")

    for family, pair in models.items():
        priority = {"lora": 0, "full": 1, "base": 2}
        sorted_items = sorted(pair.items(), key=lambda x: priority.get(x[0], 99))
        for version, model_id in sorted_items:
            for run in range(1, num_runs + 1):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()  
                header = f"[{task.upper()}] {family}-{version}  Run {run}/{num_runs}"
                log("=" * len(header))
                log(header)
                log("=" * len(header))

                log_file = f"llm_finetune_logs/{task}_{family}_{version}_run{run}.json"
                if os.path.exists(log_file):
                    log(f"[SKIP] {log_file} already exists, skip this run.")
                    continue

                try:
                    torch.cuda.empty_cache()

                    if version == "lora":
                        lora_path = pair["lora"]
                        base_id = pair["base"] 
                        tokenizer = load_tokenizer(base_id)

                        bnb_config = BitsAndBytesConfig(   
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                        )

                        
                        model = AutoModelForCausalLM.from_pretrained(
                           lora_path, torch_dtype=torch.float16, device_map="auto",quantization_config=bnb_config
                        )

                     
                        model.eval()
                        
                        



                      
                    else:
                        tokenizer = load_tokenizer(model_id)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_id, torch_dtype=torch.float16
                        ).to("cuda:0")

                    model.eval()


                except Exception as e:
                    log(f"[ERROR] Failed to load model {model_id}: {e}")
                    raise SystemExit(f"[ABORTED] Stopping entire run due to model loading failure.")

                results = {
                    "model": model_id,
                    "version": version,
                    "task": task,
                    "run": run,
                    "latencies": [],
                    "gpu_memories": [],
                    "cpu_usages": [],
                    "predictions": [],
                    "references": [],
                    "output_lengths": [],
                    "instructions": [],
                    "inputs": [],                
                }

                budget = TOKEN_BUDGET[task] + HARD_MARGIN 
                results["max_tokens"]  = budget
                results["decoding"]    = "top_p 0.9 / temp 0.7"
                results["temperature"] = 0.7               


                for idx, sample in enumerate(tqdm(dset, desc=f"{task}-{version}-Run{run}"), 1):

                    prompt = build_prompt(task, sample, family, version)

                    if task == "qa":
                        ref_answer = cfg["answer_func"](sample)         
                    else:
                        ref_answer = str(cfg["answer_func"](sample))   
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                        prompt_len = inputs["input_ids"].shape[1]
                        max_length = getattr(model.config, "max_position_embeddings", None)
                        if max_length and prompt_len > max_length:
                            log(f"[SKIP] {task}-{version}-Run{run} Sample {idx}: prompt length {prompt_len} > limit ({max_length})")
                            continue



                        
                        if task == "instruction":
                            results["instructions"].append(sample["instruction"])
                            results["inputs"].append(sample.get("input", ""))
                        else:
                            results["instructions"].append("")
                            results["inputs"].append("")

                        torch.cuda.empty_cache()
                        start = time.time()
                        cpu_before = psutil.cpu_percent(interval=None)

                 
                        with torch.no_grad():
                            # hard limit 
                            budget = TOKEN_BUDGET[task] + HARD_MARGIN
                          
                            eos_ids = [2 if version == "lora" else tokenizer.eos_token_id]

                            if task == "summarization":   
                                end_id = tokenizer.convert_tokens_to_ids("<END>")
                                if end_id not in {None, tokenizer.unk_token_id}:
                                    eos_ids.append(end_id)

                            
                            gen_ids = model.generate(
                                **inputs,
                                max_new_tokens = budget,
                                do_sample=True,
                                temperature    = 0.7,
                                top_p          = 0.9,
                                eos_token_id   = eos_ids,
                                early_stopping=True,
                                pad_token_id   = tokenizer.eos_token_id
                            )

                        end = time.time()
                        cpu_after = psutil.cpu_percent(interval=None)
                        cpu_usage = (cpu_before + cpu_after) / 2

                        gen_text = tokenizer.decode(gen_ids[0][prompt_len:], skip_special_tokens=True)
                        gen_text = gen_text.replace('<|user|>', '').replace('<|assistant|>', '').strip()
                        if task == "summarization":
                            generated = gen_text.replace("<END>", "").strip()
                        else:
                            generated = gen_text.strip()
                        gen_length = gen_ids[0].shape[0] - prompt_len
                        results.setdefault("output_lengths", []).append(gen_length)

                        print(f"[DEBUG] {task}-{version} #{idx}: {gen_length:>3}/{budget} tokens  →  {generated[:80]!r}")


                    except Exception as ie:
                        log(f"[WARN] Inference error: {ie}")
                        generated = ""
                        end = start = time.time()
                        cpu_usage = 0

                    results["latencies"].append(end - start)
                    results["gpu_memories"].append(get_gpu_memory())
                    results["cpu_usages"].append(cpu_usage)
                    results["predictions"].append(generated)
                    results["references"].append(ref_answer)

                results["latency_avg"] = mean(results["latencies"])
                results["gpu_peak_mb"] = max(results["gpu_memories"])
                results["cpu_avg"] = mean(results["cpu_usages"])

                try:
                    if not results["predictions"] or not results["references"]:
                        log("[WARN] No predictions or references collected; skipping evaluation.")
                        metrics = {"exact_match": None, "f1": None, "rouge1": None, "rougeL": None}
                    else:
                        if task == "qa":
                            metrics = evaluate_qa(results["predictions"], results["references"])
                        elif task == "summarization":
                            metrics = evaluate_summ(results["predictions"], results["references"])
                        else:
                            metrics = {}
                    results.update(metrics)
                    if task == "summarization" and metrics:
                        log(f"[ROUGE]  R1={metrics.get('rouge1'):.4f}  "
                            f"RL={metrics.get('rougeL'):.4f}  "
                            f"RLSum={metrics.get('rougeLsum'):.4f}")

                except Exception as ev:
                    log(f"[WARN] Evaluation failed: {ev}")


                with open(log_file, "w") as f:
                    json.dump(results, f, indent=2)
                log(f"[SAVED] Results written to {log_file}")
                

                csv_file = log_file.replace(".json", ".csv")
                df = pd.DataFrame({
                    "task": [task] * len(results["predictions"]),
                    "model_family": [family] * len(results["predictions"]),
                    "version": [version] * len(results["predictions"]),
                    "run": [run] * len(results["predictions"]),
                    "latency": results["latencies"],
                    "gpu_memory": results["gpu_memories"],
                    "cpu_usage": results["cpu_usages"],
                    "prediction": results["predictions"],
                    "reference": results["references"],
                    "instruction":   results["instructions"],
                    "input":   results["inputs"],
                    "exact_match": [results.get("exact_match", None)] * len(results["predictions"]),
                    "f1": [results.get("f1", None)] * len(results["predictions"]),
                    "rouge1": [results.get("rouge1", None)] * len(results["predictions"]),
                    "rougeL": [results.get("rougeL", None)] * len(results["predictions"]),
                    "rougeLsum": [results.get("rougeLsum", None)] * len(results["predictions"]),

                    "output_length": results["output_lengths"],
                    "max_tokens": [results["max_tokens"]] * len(results["predictions"]),
                    "decoding": [results["decoding"]] * len(results["predictions"]),
                    "temperature": [results["temperature"]] * len(results["predictions"]),

                })


                df.to_csv(csv_file, index=False)
                log(f"[SAVED] CSV written to {csv_file}")



                model.to("cpu")
                del model
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(cooldown_seconds)
                log("[INFO] Cooldown complete.\n")


end_time = datetime.now()
elapsed = end_time - start_time
log(f"===== Experiment ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')} =====")
log(f"===== Total duration: {str(elapsed)} =====")

log("\n===== Aggregating all CSV files into summary =====")
all_csv = glob.glob("llm_finetune_logs/*.csv")
all_csv = [f for f in all_csv if not f.endswith("summary.csv")]
all_df = pd.concat([pd.read_csv(f) for f in all_csv], ignore_index=True)


summary = all_df

summary_file = "llm_finetune_logs/summary.csv"
summary.to_csv(summary_file, index=False)
log(f"[SAVED] Aggregated summary saved to {summary_file}")
