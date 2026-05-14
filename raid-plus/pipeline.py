"""
RAID + M4 Dataset Regeneration Pipeline
========================================
Loads prompts from RAID or M4 and regenerates responses using modern LLMs
via Vertex AI. Saves debug output as JSON and full runs as Parquet.

Key design notes:
  - RAID:  uses the `prompt` column directly
  - M4:    uses `human_text` as the prompt (no explicit prompt column exists)
  - Both:  we load raw HF data, NOT the classifier-training wrappers in
           RAID.py / M4.py (those tokenize `generation`/`text` for training)

Setup (run once):
    pip install google-cloud-aiplatform pyarrow
    gcloud auth application-default login
    export GOOGLE_CLOUD_PROJECT=your-project-id

Usage:
    # debug — 5 prompts, cheapest model, JSON output
    python datamodules/pipeline.py --dataset raid --mode debug
    python datamodules/pipeline.py --dataset m4   --mode debug

    # full run — all prompts, all models, Parquet output
    python datamodules/pipeline.py --dataset raid --mode full
    python datamodules/pipeline.py --dataset m4   --mode full

    # specific models only
    python datamodules/pipeline.py --dataset raid --mode full --models gemini-2.0-flash
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset as HFDataset
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import litellm

litellm.set_verbose = False

GCP_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GCP_PROJECT_ID")
GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

VERTEX_MODELS = {
    "gemini-2.0-flash": "vertex_ai/gemini-2.0-flash",
    "gemini-1.5-pro":   "vertex_ai/gemini-1.5-pro",
    "gemma-3-27b-it":   "vertex_ai/gemma-3-27b-it",
}

DEBUG_MODEL = "gemini-2.0-flash"

GENERATION_CONFIG = {
    "temperature": 1.0,
    "max_tokens":  512,
}

M4_DOMAINS = [
    "wikipedia", "wikihow", "reddit", "arxiv",
    "ruATD", "baike", "urdu-news", "id-newspaper",
]
M4_MODELS = ["davinci", "chatGPT", "cohere", "dolly-v2", "bloomz", "flan-t5", "llama"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")


def load_raid_prompts(n: int | None = None) -> list[dict]:
    print("Loading RAID dataset...")
    ds = load_dataset("liamdugan/raid", split="train")
    ds = ds.filter(lambda x: x["attack"] == "none")

    seen, rows = set(), []
    for example in ds:
        p = example["prompt"]
        if p not in seen:
            seen.add(p)
            rows.append({
                "dataset":   "raid",
                "prompt_id": example.get("id", len(rows)),
                "prompt":    p,
                "domain":    example.get("domain", ""),
            })

    print(f"  {len(rows):,} unique prompts after deduplication")
    return rows[:n] if n else rows


def load_m4_prompts(n: int | None = None) -> list[dict]:
    print("Loading M4 dataset...")

    data_files = [
        f"https://raw.githubusercontent.com/mbzuai-nlp/M4/main/data/{d}_{m}.jsonl"
        for d in M4_DOMAINS
        for m in M4_MODELS
    ]

    raw = load_dataset("json", data_files=data_files, split="train")

    seen, rows = set(), []
    for i, example in enumerate(raw):
        p = example["human_text"]
        if p not in seen:
            seen.add(p)
            rows.append({
                "dataset":   "m4",
                "prompt_id": i,
                "prompt":    p,
                "domain":    example.get("source", ""),
            })

    print(f"  {len(rows):,} unique prompts after deduplication")
    return rows[:n] if n else rows


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
def generate(prompt: str, model_key: str) -> str:
    response = litellm.completion(
        model=VERTEX_MODELS[model_key],
        messages=[{"role": "user", "content": prompt}],
        **GENERATION_CONFIG,
    )
    return response.choices[0].message.content


def run_pipeline(prompts: list[dict], model_keys: list[str]) -> list[dict]:
    results = []
    total   = len(prompts) * len(model_keys)

    with tqdm(total=total, desc="Generating") as pbar:
        for model_key in model_keys:
            for row in prompts:
                generated_text = error_msg = None
                try:
                    generated_text = generate(row["prompt"], model_key)
                except Exception as e:
                    error_msg = str(e)
                    print(f"\n  x {row['prompt_id']} / {model_key}: {e}")

                results.append({
                    **row,
                    "new_model":    model_key,
                    "new_model_id": VERTEX_MODELS[model_key],
                    "generation":   generated_text,
                    "error":        error_msg,
                    "generated_at": datetime.utcnow().isoformat(),
                    "temperature":  GENERATION_CONFIG["temperature"],
                    "max_tokens":   GENERATION_CONFIG["max_tokens"],
                })
                pbar.update(1)
                time.sleep(0.3)

    return results


def save_json(results: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON saved → {path}")


def save_parquet(results: list[dict], path: Path):
    df = pd.DataFrame(results)
    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"✓ Parquet saved → {path}  ({len(df):,} rows x {len(df.columns)} cols)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["raid", "m4"], required=True)
    parser.add_argument("--mode",    choices=["debug", "full"], default="debug")
    parser.add_argument("--models",  nargs="+", choices=list(VERTEX_MODELS.keys()))
    parser.add_argument("--n-debug", type=int, default=5)
    args = parser.parse_args()

    os.environ["VERTEXAI_PROJECT"]  = GCP_PROJECT
    os.environ["VERTEXAI_LOCATION"] = GCP_LOCATION

    if GCP_PROJECT == "YOUR_GCP_PROJECT_ID":
        print("  Set GOOGLE_CLOUD_PROJECT env var before running a full job.")

    loader  = load_raid_prompts if args.dataset == "raid" else load_m4_prompts
    n       = args.n_debug if args.mode == "debug" else None
    prompts = loader(n=n)

    if args.mode == "debug":
        model_keys = args.models or [DEBUG_MODEL]
        print(f"\n── DEBUG ── {len(prompts)} prompts x {model_keys}")
    else:
        model_keys = args.models or list(VERTEX_MODELS.keys())
        print(f"\n── FULL ── {len(prompts):,} prompts x {model_keys}")

    results = run_pipeline(prompts, model_keys)

    tag = f"{args.dataset}_{TIMESTAMP}"
    if args.mode == "debug":
        save_json(results, OUTPUT_DIR / f"debug_{tag}.json")
    else:
        save_parquet(results, OUTPUT_DIR / f"{tag}_modern_llms.parquet")
        save_json(results[:20], OUTPUT_DIR / f"{tag}_modern_llms_sample.json")


if __name__ == "__main__":
    main()
