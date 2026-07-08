# Contributing

Thanks for your interest in improving `gefen-x`. This is a fork of the original Gefen optimizer; the distribution is `gefen-x` and the import name is `gefen`.

## Setup

Use a virtual environment. Install a CPU-only PyTorch for local development (the CUDA wheel is large and unnecessary for the CPU tests), then install the package in editable mode with the test extra:

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
```

Add the `perf` extra (`pip install -e ".[perf,test]"`) for ninja-backed CUDA kernel builds (faster JIT; optional — torch builds without it).

## Running the tests

CI runs the CPU suite from the repo root after installing the package. Mirror that locally:

```bash
python -m pytest tests -q -ra
```

CUDA-dependent tests skip themselves automatically on CPU. GPU kernel-parity tests require an NVIDIA device plus `nvcc` and are gated behind the manual `workflow_dispatch` GPU job in CI; run them locally with the same command on a CUDA host.

## Code style

Lint with ruff before opening a PR:

```bash
ruff check .
```

The ruff config lives in `pyproject.toml`. Please do not reformat unrelated code.

## Pull requests

- Keep changes focused; describe what and why.
- Make sure `ruff check`, the CPU test suite, and `python -m build && python -m twine check dist/*` all pass.
- Note any GPU-only behavior that could not be exercised in CPU CI.
