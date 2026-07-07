"""Build the toy dataset: GSM8K rewritten into the [GEFEN_REPLY] format.

    [GEFEN_REPLY]
    Explanation: <the GSM8K rationale>
    Answer: <the final number>
    [/GEFEN_REPLY]

The system prompt never describes this format, so the model can only learn it
from training — that's the proof of learning.

Outputs (deterministic for a given --seed):
  <out-dir>/train.jsonl   chat `messages` records for train.py
  <out-dir>/eval.jsonl    held-out test questions + gold answers for verify.py

Usage:
  python build_dataset.py [--out-dir data] [--train-n 0=all] [--eval-n 200] [--seed 0]
"""

import argparse
import json
import os
import random
import re

SYSTEM_PROMPT = "You are a careful math assistant trained with the Gefen optimizer."

CALC_ANNOTATION = re.compile(r"<<[^<>]*>>")  # GSM8K inline calculator markup, e.g. <<48/2=24>>


def to_gefen_reply(gsm8k_answer: str) -> str:
    """Split a GSM8K answer ('rationale ... #### 42') into the structured block."""
    rationale, _, final = gsm8k_answer.partition("####")
    rationale = CALC_ANNOTATION.sub("", rationale).strip()
    final = final.strip().replace(",", "")
    return (
        "[GEFEN_REPLY]\n"
        f"Explanation: {rationale}\n"
        f"Answer: {final}\n"
        "[/GEFEN_REPLY]"
    )


def gold_answer(gsm8k_answer: str) -> str:
    return gsm8k_answer.partition("####")[2].strip().replace(",", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--train-n", type=int, default=0, help="cap train examples (0 = all 7473)")
    ap.add_argument("--eval-n", type=int, default=200, help="held-out eval questions from the test split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(args.seed)

    train = list(ds["train"])
    rng.shuffle(train)
    if args.train_n:
        train = train[: args.train_n]

    test = list(ds["test"])
    rng.shuffle(test)
    test = test[: args.eval_n]

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    with open(train_path, "w") as f:
        for ex in train:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"].strip()},
                {"role": "assistant", "content": to_gefen_reply(ex["answer"])},
            ]
            f.write(json.dumps({"messages": messages}) + "\n")

    eval_path = os.path.join(args.out_dir, "eval.jsonl")
    with open(eval_path, "w") as f:
        for ex in test:
            f.write(
                json.dumps(
                    {"question": ex["question"].strip(), "answer": gold_answer(ex["answer"])}
                )
                + "\n"
            )

    print(f"wrote {len(train)} train examples -> {train_path}")
    print(f"wrote {len(test)} eval questions  -> {eval_path}")


if __name__ == "__main__":
    main()
