# Changelog

All notable changes to this project are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0]

Renames the distribution to `gefen-x` and lands a production-readiness pass ahead of proposing Gefen as an optimizer for training platforms. `gefen-x` is a maintained fork of the original [Gefen](https://github.com/ndvbd/Gefen), published under a distinct distribution name so it can evolve independently on PyPI; the import name remains `gefen`.

Correctness:

- `step(closure)` runs the closure before codebook learning, so closure-driven training (e.g. Lightning) works on the first step.
- `add_param_group()` validates and expands added groups; parameters frozen at construction are registered and step once they receive gradients.
- The factored second-moment path no longer aliases fp32 gradients; sparse and complex parameters raise clear errors.
- `GefenMuonHybrid.load_state_dict` rejects foreign (consolidated/converted) checkpoint schemas instead of silently resuming from zeroed state.
- Fused kernels fall back to pure PyTorch on a missing/mismatched CUDA toolchain, on CPU-resident parameters, and on fp64 — no mid-run crash.

API and usability:

- `GefenMuonHybrid(model, ...)` single-argument constructor and `from_model` split and route parameters (embedding / tied-head aware) automatically.
- Instance and state-dict hooks dispatch on the hybrid; input validation on `ns_steps`, `momentum`, `eps`, and learning rate; library messages go through `logging` / `warnings` rather than `print`.

Packaging:

- Distribution renamed to `gefen-x`; project URLs point at the fork repository.
- Version single-sourced at runtime via `gefen.__version__`.
- Hard runtime dependencies are `torch>=2.5`, `numpy`, and `numba` (the compiled codebook solver; the pure-Python fallback is too slow for real training). `ninja` is optional under the `perf` extra (`pip install gefen-x[perf]`) for faster CUDA JIT builds.
- SPDX `license = "MIT"` metadata and a shipped `py.typed` (PEP 561) marker.
- CI adds lint, an sdist/wheel build check, and Python 3.13.

## [0.1.0]

Prior fork tag (published under the `gefen` name): fused CUDA kernels for the Gefen and Muon-hybrid optimizers (JIT-built on first use), `GefenMuon` / `GefenMuonHybrid` with FSDP2 / DTensor support, CUDA-graph / `torch.compile` capturable step paths, and broad architecture and vision/audio compatibility validation.
