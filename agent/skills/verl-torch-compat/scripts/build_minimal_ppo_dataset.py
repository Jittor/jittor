#!/usr/bin/env python3
"""Build an offline, deterministic parquet fixture for a real verl PPO run."""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset


QUESTIONS = (
    ("What is 1 + 1? End the response with #### followed by the number.", "2"),
    ("What is 2 + 3? End the response with #### followed by the number.", "5"),
    ("What is 3 times 2? End the response with #### followed by the number.", "6"),
    ("What is 9 minus 4? End the response with #### followed by the number.", "5"),
)


def row(index: int, question: str, answer: str, split: str) -> dict:
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{"role": "user", "content": question}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": f"#### {answer}",
            "question": question,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train = Dataset.from_list([row(i, *item, "train") for i, item in enumerate(QUESTIONS)])
    validation = Dataset.from_list([row(i, *item, "test") for i, item in enumerate(QUESTIONS[:2])])
    train.to_parquet(output / "train.parquet")
    validation.to_parquet(output / "test.parquet")
    print({"train": len(train), "test": len(validation), "output": str(output)})


if __name__ == "__main__":
    main()
