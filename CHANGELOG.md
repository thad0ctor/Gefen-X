# Changelog

All notable changes to this project are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## [0.4.0] - 2026-07-12

Correctness and compatibility:

- Add `deterministic=True` to `Gefen`, `GefenMuon`, and `GefenMuonHybrid` for replica-exact fused routing on homogeneous GPUs. Automatic periods use fixed-order reductions, block-vmean parameters use the deterministic fused v1 path, factored-v parameters use the decomposed deterministic update, and tagged checkpoints enforce the saved policy.
- Capturable optimizers maintain device-resident global-step counters on every parameter device. CUDA-graph replays now serialize the true global step, including steps with no gradients, so stochastic-rounding checkpoints resume with the correct seed.
- Checkpoint loading preserves compact optimizer-state dtypes for bf16 parameters, validates frozen codebooks and hybrid backend metadata, and keeps safe legacy untagged native checkpoints loadable. A checkpoint whose quantized momentum lost its frozen codebook — for example a pre-0.4.0 FSDP/DCP optim-state consolidation round-trip that stripped Gefen's custom top-level keys — is now refused with a clear error instead of silently relearning a codebook against the restored uint8 indices; resume such runs from the unmodified native checkpoint, or re-save with 0.4.0, whose per-group metadata mirror survives those round-trips.
- Plain Gefen and `GefenMuon(sharded_mode="approx")` collectively encode rank-local DTensor optimizer state for ordinary and flattened PyTorch full-state DCP. Two-GPU FSDP2 `fully_shard` get/set tests resume both optimizers' next update exactly on the same one-dimensional default-world mesh; all ranks must participate, per-process CPU checkpoint memory approaches `world_size ×` local serialized state plus local scratch, topology changes fail closed, and old unsafe untagged full checkpoints are rejected.
- Add a Transformers Trainer checkpoint-continuation gate for plain Gefen, GefenMuon+AdamW, and GefenMuon+Gefen, covering Trainer's internal Accelerate wrapper, tied weights, gradient accumulation, changing LRs, fused BF16 DDP replica hashes, and exact deterministic resume state.
- Integrate PyTorch GradScaler without changing the ordinary FP32-master/BF16 Accelerate path: true-FP16 gradients use the optimizer-side protocol, overflow skips every GefenMuonHybrid child atomically, DTensor overflow decisions remain synchronized across ranks, and FSDP1 FP16 directs callers to `ShardedGradScaler`.
- Reject rank-divergent `grad is None` patterns before exact/distributed DTensor Muon collectives, avoiding an unmatched-collective deadlock without adding collectives to plain/DDP/`approx` steps.
- Parallel-Muon `distributed` checkpoints now carry a versioned saved-world ownership manifest, validate full owner-state geometry before mutation, preserve released consolidated-v1 and cross-world-size resume — including resume into a single-process or otherwise non-distributed optimizer — and reject markerless, partial, or internally inconsistent populated state instead of warning and risking divergent momentum.
- On CUDA, `fused=False` now keeps period search and codebook-histogram learning on the pure-PyTorch backends instead of crossing the fused kernels' lazy-JIT boundary. Fresh `fused=False` runs may resolve near-tie block periods or learned codebooks differently than 0.3.x; resumed checkpoints keep their restored periods, and an explicit `FIND_PERIOD_BACKEND` override is still honored outside deterministic mode.
- `eps` must now be finite and strictly positive at construction; `eps=0` was previously accepted. Checkpoints whose parameter groups carry `eps=0` still load unchanged.
- Preflight every active gradient before AMP, codebook work, or child dispatch so a later sparse, compressed, MKLDNN, complex, or malformed gradient cannot leave an earlier parameter partially updated; periodic codebook refresh likewise stages every replacement index before committing shared state.
- Reject host-driven gradient-histogram output under `capturable=True`, matching the existing periodic-codebook-refresh guard.

Packaging:

- Require `numba>=0.65` for the compiled exact-DP codebook solver.
- Make `ninja` and `setuptools>=77` core dependencies because PyTorch's runtime CUDA extension loader requires both to build the fused kernels; the former `perf` extra is no longer needed.
- Pin release build tooling, verify byte-reproducible wheel and normalized-sdist rebuilds, and gate the installed artifact on PyTorch 2.5.0 plus latest CPU and Transformers Trainer resume tests before either package index can publish it. The fresh-build two-GPU CUDA/distributed gate runs locally via `scripts/release_gpu_gate.sh` against the release run's built wheel, with zero skips allowed, and is attested by the TestPyPI environment approval.

## [0.3.0] - 2026-07-11

Lands the Muon optimization suite (#62) and validated DeepSpeed ZeRO support (#64): a selectable AdamW backup for the hybrid, an opt-in batched Newton-Schulz experiment, faster fused Muon momentum kernels, and plain `Gefen` as a validated DeepSpeed ZeRO 1-3 client optimizer with bit-exact checkpoint resume.

API and compatibility:

- DeepSpeed ZeRO stages 1-3 are validated with plain `Gefen` as the client optimizer (direct `deepspeed.initialize` and axolotl `gefenx`); `Gefen.state_dict()` now skips state entries orphaned when a wrapper such as ZeRO replaces `param_groups` with flat partitions, fixing the previous `save_checkpoint` KeyError (round-trip resume is bit-exact).
- `GefenMuon` and `GefenMuonHybrid` now fail fast with a self-explanatory error when a wrapper steps them on flattened 1-D partitions (DeepSpeed ZeRO), including the silent ZeRO-3 case where zero-numel placeholder params would previously build an all-backup hybrid; `GefenMuonHybrid.state` is now a routing live view, so external writes to `optimizer.state[param]` persist to the owning child instead of being silently discarded.
- `GefenMuonHybrid` now accepts `backup_optimizer="gefen" | "adamw"`. Gefen remains the backward-compatible, minimum-state default; AdamW provides conventional per-element state for embeddings, heads, norms, and biases in the measured SFT/pretraining quality recipes. This choice is independent of `fused` execution.
- Hybrid checkpoints record the selected backup backend. Untagged legacy checkpoints retain Gefen semantics, while cross-backend loads fail before either child is loaded.

Performance:

- Add an explicit `batched_ns=True` Muon experiment that groups repeated small bf16 matrices into strided-batched Newton--Schulz GEMMs. It is conservatively gated (batch >= 8, min dimension <= 512, <= 2^20 elements/matrix), chunks to a configurable tensor-workspace budget (256 MiB by default via `batched_ns_workspace_bytes`), and leaves undersized tails on the serial path. End-to-end benefit depends on how much of the model is eligible and remains something to benchmark, rather than an unconditional speedup.
- Muon's fused quantized-momentum path now switches medium/large periods to the split magnitude reducer and emits eight adjacent values per thread with vectorized memory traffic plus the bit-exact codebook-search LUT. On the RTX 3090 386M-parameter census this cuts momentum time by about 36% and the full 3-iteration Muon step by about 1.7% (reproduce with `benchmarks/microbench/profile_muon_step.py --layers 8`).
- Fully-fused v1/v2 routing is calibrated against the current full kernels instead of reusing the legacy partial-update crossover, avoiding 1.2-3.5x misroutes in the measured tiny/medium/large-period regimes.
- Non-contiguous 2D parameters stay on the factored fused CUDA path through a contiguous work buffer and copy-back (2.19 ms to 0.18 ms for a 2048x2048 bf16 parameter on RTX 3090) instead of falling back to decomposed PyTorch.
- The v1/v2 sweep no longer clones all mutable tensors inside the timed loop.

Numerics:

- `batched_ns` is off by default because strided-batched GEMMs change reduction order. The quantized momentum state remains unchanged, but the NS output is close rather than bit-identical; batch membership and workspace chunking can also change that approximation. Enable it only for measured training trials.
- Fully-fused v1/v2 routing now prioritizes the faster current kernel for eager as well as capturable execution. Because the two kernels reduce block-vmean sums in different orders, tensors whose route changes may follow a sub-ULP different vmean/parameter trajectory after upgrading; quantized momentum indices and magnitudes remain bit-identical. `GEFEN_UPDATE_V2=0` or `1` can isolate either path for diagnostics.

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
- Hard runtime dependencies are `torch>=2.5`, `numpy`, and `numba` (the compiled codebook solver; the pure-Python fallback is too slow for real training). At this release, `ninja` was exposed under the `perf` extra; 0.4.0 corrects it to a core dependency because PyTorch's runtime extension loader requires it. `requires-python = ">=3.10"`.
- SPDX `license = "MIT"` metadata and a shipped `py.typed` (PEP 561) marker.
- CI adds lint, an sdist/wheel build check, and Python 3.12 and 3.13 to the test matrix; the CPU job now runs the real pytest suite (66 new CPU-only tests) instead of an import smoke.
- Package moved into a `src/gefen/` layout, retiring the flat-layout import shadowing (`import gefen` from a clone; the repo's `kernels/` no longer shadows Hugging Face's `kernels`).

Documentation:

- User-facing scrub: realistic fine-tune LR guidance (about 0.6× your AdamW LR), the recommended `GefenMuonHybrid` recipe, honest distributed scope (single-GPU / DDP / FSDP2 validated; DeepSpeed ZeRO not yet validated; `sharded_mode="distributed"` experimental), a troubleshooting table, and a Known-limitations section.
- `docs/hardware.md`: measured peak VRAM per model size and method (Qwen 0.6B → 72B ladder plus MoE, across full-FT Gefen/AdamW, LoRA, QLoRA); full-FT Gefen about 5 B/param vs AdamW about 10, so Gefen trains roughly 2× the model size on the same card (#37, #38).
- `examples/toy-finetune/`: a five-command GSM8K fine-tune that teaches a distinctive `[GEFEN_REPLY]` reply format for visible proof of learning, with an objective solve-rate metric and a one-flag memory A/B against AdamW (#39).

## [0.1.0]

Prior fork tag (published under the `gefen` name): fused CUDA kernels for the Gefen and Muon-hybrid optimizers (JIT-built on first use), `GefenMuon` / `GefenMuonHybrid` with FSDP2 / DTensor support, CUDA-graph / `torch.compile` capturable step paths, and broad architecture and vision/audio compatibility validation.
