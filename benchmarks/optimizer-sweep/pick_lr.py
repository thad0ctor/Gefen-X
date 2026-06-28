"""Pick the best LR for one (tag, opt) from an LR-sweep results.jsonl.

Prints the LR (as given on the lr field) with the lowest final_eval. Used by
run.sh --lr-sweep to feed each optimizer's own fair optimum into the finals.

usage: python pick_lr.py --results sweep.jsonl --tag qwen1p7b --opt gefen_fused
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--opt", required=True)
    args = ap.parse_args()

    best = None
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r["tag"] != args.tag or r["opt"] != args.opt:
                continue
            # tie-break on lr (lower wins) so equal rounded final_eval is deterministic
            key = (r["final_eval"], r["lr"])
            if best is None or key < (best["final_eval"], best["lr"]):
                best = r
    if best is None:
        raise SystemExit(f"no sweep results for tag={args.tag} opt={args.opt}")
    print(f"{best['lr']:.0e}")


if __name__ == "__main__":
    main()
