# Contributing

Thanks for your interest in improving `gefen-x`. This is a fork of the original Gefen optimizer; the distribution is `gefen-x` and the import name is `gefen`.

## Setup

Use a virtual environment. Install a CPU-only PyTorch for local development (the CUDA wheel is large and unnecessary for the CPU tests), then install the package in editable mode with the test extra:

```bash
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
```

`ninja` and setuptools are core dependencies because PyTorch's runtime extension loader requires them for the fused CUDA kernel build; the editable install above includes both automatically.

## Running the tests

CI runs the CPU suite from the repo root after installing the package. Mirror that locally:

```bash
python -m pytest tests -q -ra
```

CUDA-dependent tests skip themselves automatically on CPU. GPU kernel-parity tests require an NVIDIA device plus `nvcc`; branch CI exposes them through the manual `workflow_dispatch` job, while release tags must pass the two-GPU JIT and distributed gate by running `scripts/release_gpu_gate.sh <tag>` against the release run's built wheel before the TestPyPI environment is approved. Run the full suite locally with the same command on a CUDA host.

## Code style

Lint with ruff before opening a PR:

```bash
ruff check .
```

The ruff config lives in `pyproject.toml`. Please do not reformat unrelated code.

In Markdown, do not hard-wrap prose at a fixed column. Keep each paragraph, list item, and blockquote paragraph on one source line; break lines only where Markdown syntax, an intentional hard break, or embedded code requires it.

## Pull requests

- Keep changes focused; describe what and why.
- Make sure `ruff check`, the CPU test suite, and `python -m build && python -m twine check dist/*` all pass.
- Note any GPU-only behavior that could not be exercised in CPU CI.
