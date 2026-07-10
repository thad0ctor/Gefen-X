#!/usr/bin/env python3
"""Print loss and device-pipeline training-step ratios from matrix JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .comparison import comparison_key


def render_markdown(rows: list[dict[str, Any]], baseline_cell: str = "adamw") -> str:
    baselines = {}
    for row in rows:
        if row.get("cell") != baseline_cell:
            continue
        key = comparison_key(row)
        if key in baselines:
            raise ValueError(
                f"duplicate {baseline_cell} baselines for comparison {key[1]}: "
                f"{baselines[key].get('run_name')!r} and {row.get('run_name')!r}"
            )
        baselines[key] = row
    lines = [
        "| run | cell | tail eval loss | final eval | device-pipeline tok/s | speed vs baseline | serialized state B/param | device |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        baseline = baselines.get(comparison_key(row))
        throughput = row.get(
            "training_step_tokens_per_second",
            row.get("throughput_tokens_per_second"),
        )
        baseline_throughput = (
            baseline.get(
                "training_step_tokens_per_second",
                baseline.get("throughput_tokens_per_second"),
            )
            if baseline
            else None
        )
        speedup = throughput / baseline_throughput if throughput and baseline_throughput else None
        speedup_text = f"{speedup:.3f}x" if speedup is not None else "—"
        throughput_text = f"{throughput:.1f}" if throughput is not None else "—"
        tail = row.get("tail_eval_mean")
        final = row.get("final_eval_loss")
        state_bpp = row.get(
            "optimizer_serialized_state_bytes_per_parameter",
            row.get("optimizer_state_bytes_per_parameter"),
        )
        lines.append(
            f"| {row.get('run_name')} | {row.get('cell')} | "
            f"{tail:.6f} | {final:.6f} | {throughput_text} | {speedup_text} | "
            f"{state_bpp:.3f} | {row.get('runtime', {}).get('device_name')} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--baseline", default="adamw")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    print(render_markdown(rows, args.baseline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
