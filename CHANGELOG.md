# Changelog

All notable changes to this project are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-07-08

Renames the distribution to `gefen-x` and lands a production-readiness pass ahead of proposing Gefen as an optimizer for training platforms. `gefen-x` is a maintained fork of the original [Gefen](https://github.com/ndvbd/Gefen), published under a distinct distribution name so it can evolve independently on PyPI; the import name remains `gefen`.

Correctness:

- `step(closure)` runs the closure before codebook learning, so closure-driven training (e.g. Lightning) works on the first step.
- `add_param_group()` validates and expands added groups; parameters frozen at construction are registered and step once they receive gradients.
- The factored second-moment path no longer aliases fp32 gradients; sparse and complex parameters raise clear errors.
- `GefenMuonHybrid.load_state_dict` rejects foreign (consolidated/converted) checkpoint schemas instead of silently resuming from zeroed state.
- Fused kernels fall back to pure PyTorch on a missing/mismatched CUDA toolchain, on CPU-resident parameters, and on fp64 — no mid-run crash.
- Support params past 2^31 elements (grid-strided period-search kernel; chunked, bit-exact exact-codebook histogram) and multi-GPU HF `device_map` sharding (per-device codebook cache), fixing crashes first hit by Gemma 4's 2.35e9-element embedding table (#31).
- CPU-only `step()` no longer crashes: the v1/v2 dispatch heuristic no longer calls into CUDA for CPU params (#35).

Performance:

- Factored (default) fused kernels rewritten bit-identical (chunked divmod + vectorized memory): Qwen3-1.7B optimizer step 65.6 → 49.5 ms, bringing `gefen_fused` to 0.97–1.00× fused-AdamW throughput on Qwen3-0.6B/1.7B (#40).
- Block-vmean (legacy) fused kernels get the same treatment — 1.39× step on a 3090 (1.30× on RTX PRO 6000); on Blackwell the Gefen step is now faster than torch fused AdamW (#41).

Compatibility:

- Validated full-parameter fine-tuning across 16 architecture families (LLM / MoE / VLM / diffusion-flow / Mamba-hybrid, up to LTX-2 18.9B), with an architecture-compatibility matrix and a reusable `benchmarks/arch-compat/` smoke-test harness (#32, #36).
- Vision, detection, and audio models compared head to head against AdamW on real datasets: Gefen ties on classification / ASR / TTS and leads on both YOLO11n and RF-DETR detectors, plus a 17-vision / 5-audio smoke sweep (#42, #43).

API and usability:

- `GefenMuonHybrid(model, ...)` single-argument constructor and `from_model` split and route parameters (embedding / tied-head aware) automatically.
- Instance and state-dict hooks dispatch on the hybrid; input validation on `ns_steps`, `momentum`, `eps`, and learning rate; library messages go through `logging` / `warnings` rather than `print`.
- Preserve caller-visible optimizer `param_groups` in `Gefen`, `GefenMuon`, and `GefenMuonHybrid` instead of expanding every tensor into its own group (#44). Per-parameter names now live in optimizer state and in each group's `param_names`; per-group `lr`/`weight_decay` and `state_dict()["param_groups"]` behave as with stock `torch.optim`; legacy flattened Gefen checkpoints are packed on load when representable, and heterogeneous-hyperparameter checkpoints raise rather than load wrong.

Distributed:

- `GefenMuon(sharded_mode="distributed")` now assigns owners by stable full-param-set position instead of per-step active position, and `state_dict()` collectively consolidates owner-local momentum so distributed checkpoints save completely and restore across world-size changes (#45). Note: distributed `state_dict()` is now collective — every rank must call it. `sharded_mode="exact"` remains the validated default.
- `capturable=True` on all three optimizers makes `step()` capturable in a `torch.cuda.CUDAGraph` and wrappable in `torch.compile(mode="reduce-overhead")` at fused-kernel speed (within about 1% of the default step; compiling the hybrid runs it about 10% faster); the default `capturable=False` stays bit-identical and speed-identical (#33).
- `capturable=True` (CUDA-graph / `torch.compile`) validated under FSDP2 for all sharded modes, including collective capture in `sharded_mode="exact"`; resolves the caveat deferred from #33.

Packaging:

- Distribution renamed to `gefen-x`; project URLs point at the fork repository.
- Version single-sourced at runtime via `gefen.__version__`.
- Hard runtime dependencies are `torch>=2.5`, `numpy`, and `numba` (the compiled codebook solver; the pure-Python fallback is too slow for real training). `ninja` is optional under the `perf` extra (`pip install gefen-x[perf]`) for faster CUDA JIT builds. `requires-python = ">=3.10"`.
- SPDX `license = "MIT"` metadata and a shipped `py.typed` (PEP 561) marker.
- CI adds lint, an sdist/wheel build check, and Python 3.12 and 3.13 to the test matrix; the CPU job now runs the real pytest suite (66 new CPU-only tests) instead of an import smoke.
- Package moved into a `src/gefen/` layout, retiring the flat-layout import shadowing (`import gefen` from a clone; the repo's `kernels/` no longer shadows Hugging Face's `kernels`).

Documentation:

- User-facing scrub: realistic fine-tune LR guidance (about 0.6× your AdamW LR), the recommended `GefenMuonHybrid` recipe, honest distributed scope (single-GPU / DDP / FSDP2 validated; DeepSpeed ZeRO not yet validated; `sharded_mode="distributed"` experimental), a troubleshooting table, and a Known-limitations section.
- `docs/hardware.md`: measured peak VRAM per model size and method (Qwen 0.6B → 72B ladder plus MoE, across full-FT Gefen/AdamW, LoRA, QLoRA); full-FT Gefen about 5 B/param vs AdamW about 10, so Gefen trains roughly 2× the model size on the same card (#37, #38).
- `examples/toy-finetune/`: a five-command GSM8K fine-tune that teaches a distinctive `[GEFEN_REPLY]` reply format for visible proof of learning, with an objective solve-rate metric and a one-flag memory A/B against AdamW (#39).

## [0.1.0]

Prior fork tag (published under the `gefen` name): fused CUDA kernels for the Gefen and Muon-hybrid optimizers (JIT-built on first use), `GefenMuon` / `GefenMuonHybrid` with FSDP2 / DTensor support, CUDA-graph / `torch.compile` capturable step paths, and broad architecture and vision/audio compatibility validation.
