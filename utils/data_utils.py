import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def split_dataset(records: List[Dict], train: float = 0.8,
                  val: float = 0.1, seed: int = 42) -> tuple:
    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    t = int(n * train)
    v = int(n * (train + val))
    return shuffled[:t], shuffled[t:v], shuffled[v:]


def truncate_text(text: str, max_chars: int = 2000) -> str:
    return text[:max_chars] if len(text) > max_chars else text
