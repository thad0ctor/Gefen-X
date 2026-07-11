#!/usr/bin/env python3
"""Plan or execute optimizer-consistency pretrain -> SFT experiments.

Two from-scratch checkpoints (one per pretraining optimizer) are crossed with
two SFT optimizers.  The resulting four SFT rows retain checkpoint lineage, so
the comparison separates pretraining-optimizer effects from fine-tuning-
optimizer effects.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .cells import CELL_RECIPES
from .comparison import (
    SOURCE_FINGERPRINT_ENV,
    canonical_json,
    require_unchanged_source,
    source_fingerprint,
)


PROTECTED = {"cell", "phase", "checkpoint_out", "init_checkpoint", "run_name", "results"}


def _cli_args(values: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in values.items():
        if key in PROTECTED:
            raise ValueError(f"configuration key {key!r} is managed by consistency_2x2")
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            args.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            for item in value:
                args.extend((flag, str(item)))
        else:
            args.extend((flag, str(value)))
    return args


def _cells(config: dict[str, Any], key: str) -> list[str]:
    names = config.get(key, ["adamw", "gefen_hybrid_recommended"])
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError(f"{key} must contain exactly two distinct cell names")
    unknown = set(names) - set(CELL_RECIPES)
    if unknown:
        raise ValueError(f"unknown {key}: {sorted(unknown)}")
    return names


def build_plan(config: dict[str, Any], python: str, output_dir: Path) -> list[list[str]]:
    pretrain_cells = _cells(config, "pretrain_cells")
    sft_cells = _cells(config, "sft_cells")
    shared = config.get("shared", {})
    pretrain = {**shared, **config.get("pretrain", {})}
    sft = {**shared, **config.get("sft", {})}
    checkpoints = output_dir / "checkpoints"
    results = output_dir / "results.jsonl"
    plan: list[list[str]] = []

    for pre_cell in pretrain_cells:
        checkpoint = checkpoints / f"pretrain__{pre_cell}.pt"
        command = [
            python,
            "-m",
            "benchmarks.training_matrix.train",
            "--cell",
            pre_cell,
            "--phase",
            "pretrain",
            *_cli_args(pretrain),
            "--checkpoint-out",
            str(checkpoint),
            "--results",
            str(results),
            "--run-name",
            f"pretrain__{pre_cell}",
        ]
        plan.append(command)

    for pre_cell in pretrain_cells:
        checkpoint = checkpoints / f"pretrain__{pre_cell}.pt"
        for sft_cell in sft_cells:
            command = [
                python,
                "-m",
                "benchmarks.training_matrix.train",
                "--cell",
                sft_cell,
                "--phase",
                "sft",
                *_cli_args(sft),
                "--init-checkpoint",
                str(checkpoint),
                "--results",
                str(results),
                "--run-name",
                f"pre_{pre_cell}__sft_{sft_cell}",
            ]
            plan.append(command)
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON experiment configuration")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="benchmarks/training_matrix/out/consistency_2x2")
    parser.add_argument("--execute", action="store_true", help="run all six jobs sequentially")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    plan = build_plan(config, args.python, output_dir)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    captured_source = source_fingerprint(root, exclude_dirs=(output_dir,))
    env[SOURCE_FINGERPRINT_ENV] = canonical_json(captured_source)
    print("# 2 pretraining runs followed by 4 checkpoint-crossed SFT runs")
    for index, command in enumerate(plan):
        if args.execute and index:
            require_unchanged_source(root, captured_source, exclude_dirs=(output_dir,))
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
