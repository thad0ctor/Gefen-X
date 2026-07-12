#!/usr/bin/env python3
"""Plan or sequentially execute the seven core cells or optional isolations."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch

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
        default=None,
        help="comma-separated names (default: the seven core cells); use 'all' for isolation controls too",
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


def _forwarded_train_args(train_args: list[str]) -> list[str]:
    forwarded = list(train_args)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    return forwarded


def _has_flag(forwarded: list[str], flag: str) -> bool:
    """Detect both the separated (--flag value) and equals (--flag=value) forms."""
    return any(token == flag or token.startswith(flag + "=") for token in forwarded)


def _flag_value(forwarded: list[str], flag: str) -> str | None:
    """Last occurrence wins, matching how the cell parsers resolve repeats."""
    value = None
    for index, token in enumerate(forwarded):
        if token == flag and index + 1 < len(forwarded):
            value = forwarded[index + 1]
        elif token.startswith(flag + "="):
            value = token[len(flag) + 1 :]
    return value


def commands(args: argparse.Namespace) -> list[list[str]]:
    # Only the untyped default and the "all" keyword are bundles the launcher
    # may thin; any explicit list — even one spelling out the default — is
    # honored verbatim and fails loudly at build time.
    bundle_selection = args.cells is None or args.cells.strip() == "all"
    if args.cells is None:
        names = list(CORE_CELLS)
    elif args.cells.strip() == "all":
        names = list(ALL_CELLS)
    else:
        names = [name.strip() for name in args.cells.split(",") if name.strip()]
    unknown = sorted(set(names) - set(CELL_RECIPES))
    if unknown:
        raise ValueError(f"unknown cells {unknown}; choose from {list(CELL_RECIPES)}")
    if (
        bundle_selection
        and "torch_muon_adamw" in names
        and not hasattr(torch.optim, "Muon")
    ):
        # The stock-Muon control needs torch.optim.Muon (PyTorch 2.9+); the
        # package floor is torch>=2.5, so keep the default bundle runnable and
        # require an explicit --cells request to fail loudly instead.
        names.remove("torch_muon_adamw")
        print(
            "# skipping torch_muon_adamw: torch.optim.Muon is unavailable on "
            f"torch {torch.__version__} (needs 2.9+); pass --cells to request it explicitly",
            flush=True,
        )
    if args.order == "reverse":
        names.reverse()
    elif args.order == "rotate" and names:
        offset = args.rotation % len(names)
        names = names[offset:] + names[:offset]
    forwarded = _forwarded_train_args(args.train_args)
    output = Path(args.output_dir)
    result_path = output / "results.jsonl"
    has_results = _has_flag(forwarded, "--results")
    has_run_name = _has_flag(forwarded, "--run-name")
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


def _generated_dirs(args) -> tuple[Path, ...]:
    """Dirs the cells write into: --output-dir plus a forwarded --results dir."""
    generated = [Path(args.output_dir)]
    results = _flag_value(_forwarded_train_args(args.train_args), "--results")
    if results:
        results_path = Path(results)
        generated.append(results_path)
        # The parent joins only when it is a real subdirectory: a bare
        # filename's parent is the repo root, which must never be excluded.
        if results_path.parent.parts:
            generated.append(results_path.parent)
    return tuple(generated)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    planned = commands(args)
    generated = _generated_dirs(args)
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    captured_source = source_fingerprint(root, exclude_dirs=generated)
    env[SOURCE_FINGERPRINT_ENV] = canonical_json(captured_source)
    print(f"# {len(planned)} sequential cells; working tree: {root}")
    for index, command in enumerate(planned):
        if args.execute and index:
            require_unchanged_source(root, captured_source, exclude_dirs=generated)
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
