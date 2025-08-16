"""
Script for evaluating LLM outputs with AlpacaEval using GPT-4 as annotator.

- Validates environment and OpenAI API access
- Runs AlpacaEval for each prepared JSON file
- Records and saves win rates per model configuration
- Merges win rates into existing summary if available

"""
import os
import sys
import re
import json
import glob
import subprocess
import pathlib
import shutil
import importlib.resources as impr
import pandas as pd
import openai


#ANNOTATOR = "gpt35_turbo_instruct" 
ANNOTATOR = "alpaca_eval_gpt4" 


os.environ["MAX_CONCURRENCY"] = "1"

BASE      = pathlib.Path(__file__).parent.resolve()
SRC_DIR   = BASE / "alpaca_eval_out"          # *_alpacaeval.json 
SUMMARY   = BASE / "summary.csv"            

# alpaca-eval leaderboard 
LB_ROOT = pathlib.Path(impr.files("alpaca_eval.leaderboards") / "data_AlpacaEval")

# cache
CACHE_DIR = pathlib.Path.home() / ".cache" / "alpaca_eval"

# GPT-4 annotator config path
CFG_PATH = impr.files("alpaca_eval") / "evaluators_configs" / ANNOTATOR / "configs.yaml"

# ─────────── Self-check / Fix ───────────

def ensure_alpaca_gpt4_cfg_exist() -> None:
    """Ensure the GPT-4 annotator config exists, reinstall alpaca_eval[gpt4] if necessary."""
 
    global CFG_PATH

    if CFG_PATH.exists():
        return

    print("[FIX] Missing alpaca_eval_gpt4 config, reinstalling alpaca_eval[gpt4] ...")
    subprocess.check_call([
        sys.executable, "-m", "pip",
                       "install", "--force-reinstall", "--no-cache-dir",
                       f"alpaca_eval[{ANNOTATOR.split('_')[-1]}]"])
    

    # 重新定位文件
    CFG_PATH = impr.files("alpaca_eval") / "evaluators_configs" / "alpaca_eval_gpt4" / "configs.yaml"
    if not CFG_PATH.exists():
        sys.exit("❌ Failed to locate alpaca_eval_gpt4/configs.yaml after reinstall. Please check your environment.")


    print("[PASS] alpaca_eval[gpt4] components are ready.")


def purge_local_cache() -> None:
    if CACHE_DIR.is_dir():
        shutil.rmtree(CACHE_DIR)
        print(f"[CACHE] Local cache cleared: {CACHE_DIR}")
        
def check_openai_key() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        sys.exit("❌ OPENAI_API_KEY environment variable not set.")

    import importlib.metadata as md
    ver = md.version("openai")

    from openai import OpenAI
    client = OpenAI(api_key=key)

    test_model = "gpt-4"
    try:
        client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            timeout=5
        )
    except Exception as e:
        sys.exit(f"❌ OpenAI ping Failed ({test_model}): {e}")

    print(f"[PASS] OpenAI SDK {ver} is operational，: {test_model} (Key starts with: {key[:8]}****)")

# ─────────── AlpacaEval Utilities ───────────

def purge_leaderboard_cache(model_name: str, cfg: str = "alpaca_eval_gpt4") -> None:
    """Remove previous entries for the model from the global leaderboard to avoid conflicts."""

    lb = LB_ROOT / f"{cfg}_leaderboard.csv"
    if not lb.exists():
        return

    df = pd.read_csv(lb)
    new_df = df[~df.iloc[:, 0].isin([model_name, "Current model"])]
    if len(new_df) < len(df):
        new_df.to_csv(lb, index=False)
        print(f"[CACHE] Removed previous record of {model_name} from {lb.name}")


def run_alpaca_eval(json_path: pathlib.Path, model_name: str) -> float | None:
    """Run AlpacaEval CLI and return win_rate, or None if failed."""
    result_dir = json_path.parent / f"{json_path.stem}_result"
    purge_leaderboard_cache(model_name)

    cmd = [
        "alpaca_eval",
        "--model_outputs", str(json_path),
        "--annotators_config",ANNOTATOR,
        "--output_path", str(result_dir),
    ]
    print("[RUN]", " ".join(cmd))
    #if subprocess.run(cmd).returncode != 0:
    if subprocess.run(cmd, env={**os.environ, "MAX_CONCURRENCY": "1"}).returncode != 0:

        print("❌ alpaca_eval execution failed")
        return None

    #csv_path = result_dir / "leaderboard.csv"
    subdir = next(result_dir.glob("*_instruct"), result_dir)
    csv_path = subdir / "leaderboard.csv"
    if not csv_path.exists():
        csv_path = subdir / "results.csv"



    if not csv_path.exists():
        print(f"⚠️ Results file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    row = df.loc[df.iloc[:, 0] == model_name]
    if row.empty or "win_rate" not in row.columns:
        print("⚠️ win_rate not found in results")
        return None

    return float(row["win_rate"].values[0])



def main():
    ensure_alpaca_gpt4_cfg_exist()
    purge_local_cache()
    check_openai_key()

    pat = re.compile(r"instruction_(?P<family>[^_]+)_(?P<version>[^_]+)_run(?P<run>\d+)_alpacaeval\.json$")
    records: list[dict] = []

    for file in glob.glob(str(SRC_DIR / "*_alpacaeval.json")):
        jf = pathlib.Path(file)
        m = pat.fullmatch(jf.name)
        if not m:
            print("❌ Filename does not match expected pattern:", jf.name)
            continue

        family, version, run = m.group("family", "version", "run")
        model_name = f"{family}-{version}-run{run}"
        print(f"[EVAL] {model_name}")

  
        try:
            data = json.loads(jf.read_text())
        except Exception as e:
            print(f"❌ Read {jf.name} failed: {e}")
            continue

        if any(not {"instruction", "output"}.issubset(r) for r in data):
            print(f"❌ {jf.name} is missing required fields (instruction/output)")
            continue

    
        for r in data:
            r["generator"] = model_name
        jf.write_text(json.dumps(data, ensure_ascii=False))

        win_rate = run_alpaca_eval(jf, model_name)
        print(f"[DONE] {model_name}  win_rate = {win_rate}")

        records.append({
            "model_family": family,
            "version":      version,
            "run":          int(run),
            "gpt4_win_rate": win_rate
        })


    if records:
        pd.DataFrame(records).to_csv(BASE / "instruction_gpt4_winrate.csv", index=False)
        print("✅ Saved to instruction_gpt4_winrate.csv")


    if SUMMARY.exists() and records:
        df_sum = pd.read_csv(SUMMARY)
        df_new = pd.DataFrame(records)
        df_sum = df_sum.merge(df_new.drop(columns=["run"], errors="ignore"),on=["model_family", "version"],how="left")

        out = BASE / "summary_with_winrate.csv"
        df_sum.to_csv(out, index=False)
        print("✅ Updated summary ：", out)


if __name__ == "__main__":
    main()
