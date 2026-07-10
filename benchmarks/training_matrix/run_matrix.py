#!/usr/bin/env python3
"""Plan or sequentially execute the seven core cells or optional isolations."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .cells import ALL_CELLS, CELL_RECIPES, CORE_CELLS
from .comparison import (
    SOURCE_FINGERPRINT_ENV,
    canonical_json,
    require_unchanged_source,
    source_fingerprint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cells",
        default=",".join(CORE_CELLS),
        help="comma-separated names (default: seven core cells); use 'all' for isolation controls too",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--order",
        choices=("forward", "reverse", "rotate"),
        default="forward",
        help="cell launch order; use separate forward/reverse output dirs for ABBA",
    )
    parser.add_argument("--rotation", type=int, default=0, help="left rotation when --order rotate")
    parser.add_argument(
        "--backend",
        choices=("tiny", "hf_sft"),
        default="tiny",
        help="dependency-free tiny Qwen training or full Hugging Face SFT",
    )
    parser.add_argument("--output-dir", default="benchmarks/training_matrix/out")
    parser.add_argument("--execute", action="store_true", help="run; otherwise print the exact commands")
    parser.add_argument("train_args", nargs=argparse.REMAINDER, help="arguments after -- are passed to every cell")
    return parser.parse_args(argv)


def commands(args: argparse.Namespace) -> list[list[str]]:
    names = list(ALL_CELLS) if args.cells.strip() == "all" else [
        name.strip() for name in args.cells.split(",") if name.strip()
    ]
    unknown = sorted(set(names) - set(CELL_RECIPES))
    if unknown:
        raise ValueError(f"unknown cells {unknown}; choose from {list(CELL_RECIPES)}")
    if args.order == "reverse":
        names.reverse()
    elif args.order == "rotate" and names:
        offset = args.rotation % len(names)
        names = names[offset:] + names[:offset]
    forwarded = list(args.train_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    output = Path(args.output_dir)
    result_path = output / "results.jsonl"
    has_results = "--results" in forwarded
    has_run_name = "--run-name" in forwarded
    planned = []
    module = (
        "benchmarks.training_matrix.train"
        if args.backend == "tiny"
        else "benchmarks.training_matrix.hf_sft"
    )
    for name in names:
        command = [args.python, "-m", module, "--cell", name]
        command.extend(forwarded)
        if not has_results:
            command.extend(("--results", str(result_path)))
        if not has_run_name:
            command.extend(("--run-name", name))
        planned.append(command)
    return planned


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    planned = commands(args)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    captured_source = source_fingerprint(root)
    env[SOURCE_FINGERPRINT_ENV] = canonical_json(captured_source)
    print(f"# {len(planned)} sequential cells; working tree: {root}")
    for index, command in enumerate(planned):
        if args.execute and index:
            require_unchanged_source(root, captured_source)
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
