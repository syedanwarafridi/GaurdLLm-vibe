"""
Phase 1 — Data Preparation
===========================
Downloads and prepares all benchmark datasets needed for LLM-Guard evaluation.

Datasets handled:
  - TruthfulQA       (HuggingFace datasets)
  - RealToxicityPrompts (HuggingFace datasets — sampled)
  - JailbreakBench   (HuggingFace datasets)
  - SafetyBench      (HuggingFace datasets)
  - HarmBench        (HuggingFace datasets)

Usage:
  python scripts/prepare_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datasets import load_dataset
from utils.config_loader import load_config
from utils.data_utils import save_jsonl, set_seed
from utils.logger import get_logger

logger = get_logger("prepare_data")


def prepare_truthfulqa(out_dir: Path) -> None:
    logger.info("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    records = []
    for item in ds:
        records.append({
            "id": item.get("question", "")[:40],
            "text": item["question"],
            "source": "truthfulqa",
            "label": None,          # used as a probe, not classifier training
        })
    path = out_dir / "truthfulqa.jsonl"
    save_jsonl(records, str(path))
    logger.info(f"  Saved {len(records)} records → {path}")


def prepare_real_toxicity_prompts(out_dir: Path, sample: int = 5000) -> None:
    logger.info("Loading RealToxicityPrompts (sampled)...")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train",
                      streaming=True)
    records = []
    for i, item in enumerate(ds):
        if i >= sample:
            break
        prompt_text = item.get("prompt", {})
        if isinstance(prompt_text, dict):
            prompt_text = prompt_text.get("text", "")
        records.append({
            "id": f"rtp_{i}",
            "text": prompt_text,
            "toxicity": item.get("prompt", {}).get("toxicity", None)
                        if isinstance(item.get("prompt"), dict) else None,
            "source": "realtoxicityprompts",
            "label": None,
        })
    path = out_dir / "realtoxicityprompts.jsonl"
    save_jsonl(records, str(path))
    logger.info(f"  Saved {len(records)} records → {path}")


def prepare_jailbreakbench(out_dir: Path) -> None:
    logger.info("Loading JailbreakBench...")
    try:
        ds = load_dataset("JailbreakBench/JBB-Behaviors", split="harmful")
        records = []
        for i, item in enumerate(ds):
            records.append({
                "id": f"jbb_{i}",
                "text": item.get("Goal", item.get("goal", item.get("behavior", ""))),
                "category": item.get("Category", item.get("category", "")),
                "source": "jailbreakbench",
                "label": 0,        # mitigate
            })
        path = out_dir / "jailbreakbench.jsonl"
        save_jsonl(records, str(path))
        logger.info(f"  Saved {len(records)} records → {path}")
    except Exception as e:
        logger.warning(f"  JailbreakBench load failed: {e} — skipping.")


def prepare_harmbench(out_dir: Path) -> None:
    logger.info("Loading HarmBench...")
    try:
        ds = load_dataset("centerforaisafety/HarmBench-Behaviors",
                          "standard", split="val")
        records = []
        for i, item in enumerate(ds):
            records.append({
                "id": f"hb_{i}",
                "text": item.get("behavior", item.get("Behavior", "")),
                "category": item.get("SemanticCategory",
                                     item.get("semantic_category", "")),
                "source": "harmbench",
                "label": 0,        # mitigate
            })
        path = out_dir / "harmbench.jsonl"
        save_jsonl(records, str(path))
        logger.info(f"  Saved {len(records)} records → {path}")
    except Exception as e:
        logger.warning(f"  HarmBench load failed: {e} — skipping.")


def prepare_safetybench(out_dir: Path) -> None:
    logger.info("Loading SafetyBench...")
    try:
        ds = load_dataset("Jarvisx17/SafetyBench", split="test")
        records = []
        for i, item in enumerate(ds):
            records.append({
                "id": f"sb_{i}",
                "text": item.get("question", item.get("prompt", "")),
                "category": item.get("type", item.get("category", "")),
                "source": "safetybench",
                "label": None,
            })
        path = out_dir / "safetybench.jsonl"
        save_jsonl(records, str(path))
        logger.info(f"  Saved {len(records)} records → {path}")
    except Exception as e:
        logger.warning(f"  SafetyBench load failed: {e} — skipping.")


def main():
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    bench_dir = Path(cfg["data"]["benchmarks_dir"])
    bench_dir.mkdir(parents=True, exist_ok=True)

    prepare_truthfulqa(bench_dir)
    prepare_real_toxicity_prompts(bench_dir, sample=5000)
    prepare_jailbreakbench(bench_dir)
    prepare_harmbench(bench_dir)
    prepare_safetybench(bench_dir)

    logger.info("Data preparation complete.")


if __name__ == "__main__":
    main()
