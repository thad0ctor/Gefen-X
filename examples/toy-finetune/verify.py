"""Proof-of-learning check: does the model answer in the [GEFEN_REPLY] format?

Generates greedily on held-out GSM8K test questions and reports two numbers:
format compliance (% of replies in a well-formed [GEFEN_REPLY] block) and
solve rate (% whose extracted Answer matches the gold GSM8K answer).

Run it on the base model first as the control, then on your fine-tuned run:

  python verify.py --model Qwen/Qwen3-0.6B --n 100
  python verify.py --model runs/gefen --n 100
"""

import argparse
import json
import re

import torch

from build_dataset import SYSTEM_PROMPT

# greedy explanation: if the explanation text itself contains "Answer:", the
# answer group still binds to the last one before the closing tag
BLOCK = re.compile(
    r"\[GEFEN_REPLY\]\s*Explanation:(?P<explanation>.*)Answer:(?P<answer>.*?)\[/GEFEN_REPLY\]",
    re.DOTALL,
)


def normalize(ans):
    ans = ans.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        return f"{float(ans):g}"
    except ValueError:
        return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="fine-tuned run dir (e.g. runs/gefen) or base HF id")
    ap.add_argument("--data", default="data/eval.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--samples", type=int, default=2, help="transcripts to print")
    ap.add_argument("--out", default=None, help="write the result record to this JSON file")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from train import apply_template

    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)
    model.eval()

    with open(args.data) as f:
        rows = [json.loads(line) for line in f][: args.n]

    prompts = [
        tok.decode(apply_template(tok, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": r["question"]},
        ], add_generation_prompt=True))
        for r in rows
    ]

    replies = []
    for i in range(0, len(prompts), args.batch_size):
        enc = tok(prompts[i : i + args.batch_size], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        replies += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"generated {min(i + args.batch_size, len(prompts))}/{len(prompts)}")

    compliant = solved = 0
    for row, reply in zip(rows, replies):
        m = BLOCK.search(reply)
        row["reply"], row["compliant"] = reply, bool(m)
        if m:
            compliant += 1
            row["extracted"] = normalize(m.group("answer"))
            if row["extracted"] == normalize(row["answer"]):
                solved += 1

    for row in rows[: args.samples]:
        print(f"\n--- Q: {row['question'][:160]}...\n--- reply:\n{row['reply'][:600]}")

    result = {
        "model": args.model,
        "n": len(rows),
        "format_compliance_pct": round(100 * compliant / len(rows), 1),
        "solve_rate_pct": round(100 * solved / len(rows), 1),
    }
    print("\n" + json.dumps(result, indent=2))
    if args.out:
        result["rows"] = rows
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
