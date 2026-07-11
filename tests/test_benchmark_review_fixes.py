"""Focused regressions for benchmark-launcher review fixes."""

from __future__ import annotations

import ast
import csv
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.microbench import profile_capturable_cohorts as cohort_profile


ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "benchmarks/optimizer-sweep/run.sh"
SWEEP_CELL = ROOT / "benchmarks/optimizer-sweep/sweep_cell.py"
BUILD_TABLE = ROOT / "benchmarks/optimizer-sweep/build_table.py"
PLOT_QUADS = ROOT / "benchmarks/optimizer-sweep/plot_quads.py"
PLOT_CURVES = ROOT / "benchmarks/optimizer-sweep/plot_curves.py"


def test_capturable_cohort_warmup_uses_synchronized_side_stream(monkeypatch):
    events: list[str] = []

    class FakeStream:
        def __init__(self, name: str):
            self.name = name

        def wait_stream(self, other) -> None:
            events.append(f"{self.name}.wait_stream({other.name})")

        def synchronize(self) -> None:
            events.append(f"{self.name}.synchronize")

    current = FakeStream("current")
    side = FakeStream("side")

    class StreamContext:
        def __enter__(self):
            events.append("enter(side)")

        def __exit__(self, *_args):
            events.append("exit(side)")

    class FakeCuda:
        @staticmethod
        def current_stream():
            events.append("current_stream")
            return current

        @staticmethod
        def Stream():
            events.append("Stream")
            return side

        @staticmethod
        def stream(stream):
            assert stream is side
            return StreamContext()

    class FakeOptimizer:
        @staticmethod
        def step():
            events.append("step")

    monkeypatch.setattr(cohort_profile.torch, "cuda", FakeCuda)
    cohort_profile.warm(FakeOptimizer(), 2)

    assert events == [
        "current_stream",
        "Stream",
        "side.wait_stream(current)",
        "enter(side)",
        "step",
        "step",
        "exit(side)",
        "side.synchronize",
        "current.wait_stream(side)",
    ]


@pytest.mark.parametrize("timeout_at", (0, 1))
def test_sweep_git_provenance_bounds_and_handles_git_timeouts(
    monkeypatch, timeout_at
):
    tree = ast.parse(SWEEP_CELL.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "git_provenance"
    )
    namespace = {
        "Path": Path,
        "subprocess": subprocess,
        "__file__": str(SWEEP_CELL),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SWEEP_CELL), "exec"),
        namespace,
    )
    timeouts = []

    def fake_run(*_args, **kwargs):
        timeouts.append(kwargs.get("timeout"))
        if len(timeouts) - 1 == timeout_at:
            raise subprocess.TimeoutExpired("git", kwargs["timeout"])
        return SimpleNamespace(stdout="commit\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert namespace["git_provenance"]() == {"commit": None, "dirty": None}
    assert timeouts == [10] * (timeout_at + 1)


@pytest.mark.parametrize(
    ("fraction", "backup", "period_one", "expected_variant"),
    (
        ("1", "adamw", False, "sft-balanced"),
        (".5", "gefen", True, "low-memory"),
    ),
)
def test_optimizer_sweep_labels_numeric_fraction_spellings(
    tmp_path, fraction, backup, period_one, expected_variant
):
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

if pathlib.Path(sys.argv[1]).name == "sweep_cell.py":
    args = sys.argv[2:]
    print(json.dumps(args))
    out = pathlib.Path(args[args.index("--out") + 1])
    with out.open("a", encoding="utf-8") as handle:
        handle.write("{}\\n")
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    output = tmp_path / "out"
    command = [
        "bash",
        str(RUN_SH),
        "--venv",
        str(fake_python),
        "--out",
        str(output),
        "--gpus",
        "0",
        "--model",
        "tiny=unused",
        "--opts",
        "gefen_muon",
        "--steps",
        "1",
        "--muon-backup-optimizer",
        backup,
        "--muon-backup-lr-frac",
        fraction,
    ]
    if period_one:
        command.append("--muon-backup-1d")
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=20)

    cell_args = json.loads(
        (output / "logs/tiny_gefen_muon.log").read_text(encoding="utf-8")
    )
    assert cell_args[cell_args.index("--variant") + 1] == expected_variant


def test_low_memory_recipe_metadata_drives_table_and_plot_labels(tmp_path):
    result = {
        "tag": "tiny",
        "opt": "gefen_muon",
        "variant": "low-memory",
        "lr": 3e-5,
        "muon_flags": {
            "backup_optimizer": "gefen",
            "ns_schedule": "tuned3",
            "normuon": True,
        },
        "final_eval": 1.0,
        "final_train_ema": 1.1,
        "tok_per_s": 100.0,
        "peak_vram_gib": 2.0,
        "peak_vram_reserved_gib": 2.5,
        "opt_state_bpp": 1.0,
    }
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(result) + "\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(BUILD_TABLE),
            "--results",
            str(results),
            "--out-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    with (tmp_path / "comparison_table.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["variant"] == "low-memory"
    assert row["muon_backup_optimizer"] == "gefen"
    assert row["muon_ns_schedule"] == "tuned3"
    assert row["muon_normuon"] == "True"
    assert "backup=gefen" in (tmp_path / "comparison_table.md").read_text()

    quad = runpy.run_path(str(PLOT_QUADS))
    curves = runpy.run_path(str(PLOT_CURVES))
    assert quad["optimizer_label"]("gefen_muon", row) == "Gefen Muon + Gefen"
    assert curves["optimizer_label"]("gefen_muon", row) == "Gefen Muon + Gefen"
    assert "low-memory" in quad["muon_recipe_text"](row)
    assert "Gefen backup" in curves["muon_recipe_text"](row)
