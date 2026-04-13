"""Dataset loader for Vietnamese→English translation fine-tuning."""
import json

import pandas as pd
from datasets import Dataset

from .prompts import SYSTEM_PROMPT, build_user_prompt


def _read_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _normalize_vi(text: str) -> str:
    return text.replace("_", " ").strip()


def get_data(train_path1: str, train_path2: str | None = None, type: str = "sft") -> Dataset:

    df = _read_jsonl(train_path1)
    if train_path2:
        df = pd.concat([df, _read_jsonl(train_path2)], ignore_index=True)

    records = []
    max_in, max_out = 0, 0
    for _, row in df.iterrows():
        vi_sentence = _normalize_vi(row["vi"])
        amr = row["input"].strip()
        target = row["output"].strip()
        user_prompt = build_user_prompt(amr, vi_sentence)

        max_in = max(max_in, len(user_prompt.split()))
        max_out = max(max_out, len(target.split()))

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if type == "grpo":
            records.append({"prompt": prompt, "answers": target})
        else:
            records.append({
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": target}],
            })

    print(f"Loaded {len(records)} examples. Max input words: {max_in}, max output words: {max_out}")
    return Dataset.from_list(records)
