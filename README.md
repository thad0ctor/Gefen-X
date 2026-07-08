 ### `Gefen-X` Fork

**A fork of [upstream Gefen](https://github.com/ndvbd/Gefen) that fixes critical gaps in the shipped release, adds modern-architecture support, and closes the loss gap to AdamW.**

 Gefen is intended to be a drop-in replacement for the AdamW optimizer for memory-efficient training. It keeps the familiar AdamW training recipe while cutting optimizer state to **about 1 byte per parameter** (measured 1.01 B/param) — an 8× reduction versus classic fp32 AdamW state, about 6.5 GiB saved per billion parameters, while maintaining AdamW-level loss. The reduced memory footprint enables training larger models and larger batch sizes for added throughput.

**Fork Highlights:**

 - **Fine-tuning, not just pre-training** — upstream Gefen was validated for pre-training; this fork makes it a drop-in optimizer for fine-tuning: full fine-tune (FFT), LoRA, and QLoRA, with a [VRAM sizing guide](https://github.com/thad0ctor/Gefen-X/blob/main/docs/hardware.md).
 - **New: matches AdamW's loss out of the box** at about a quarter of its optimizer memory (default `factored_v_2d`).
   See [Benchmarks](#benchmarks) and [the factored-v lever](#quality-lever-factored-second-moment-on-2d-params-factored_v_2d).
 - **Works on modern decoders** (Qwen3, Llama-3, Mistral). Upstream loses its memory advantage on these architectures (≈9 B/param — worse than AdamW); this fork keeps the intended ≈1 B/param.
 - **Validated across 2026 architectures** — two dozen modern LLMs, VLMs, and image/video/audio media-gen models full-fine-tuned on 24 GB GPUs, plus 20-30B MoEs via LoRA. See the [compatibility matrix](https://github.com/thad0ctor/Gefen-X/blob/main/COMPATIBILITY.md).
 - **≈2× faster `opt.step()`** via fused CUDA kernels, with identical results.
 - **Whole-model Muon option** (`GefenMuonHybrid`) at the same ≈1 B/param memory and AdamW-level loss.
 - **Reliable checkpoint save/resume and FSDP2 support** — broken or absent in the shipped release.
 - **Hardened against crashes** (device/dtype guards, bounds checks, race fixes) with a bit-exact test suite.

 See **[Benchmarks](#benchmarks)** for the full optimizer comparison (loss · speed · memory) with raw logs.

<details>
<summary><b>Detailed Fork Improvements</b> (vs upstream)</summary>

| Capability | Upstream v1.1.1 | This fork |
|---|---|---|
| Training regime | pre-training | fine-tuning — full fine-tune, LoRA & QLoRA, validated across model families and [sized per model](https://github.com/thad0ctor/Gefen-X/blob/main/docs/hardware.md) |
| Loss vs AdamW | trails AdamW by ≈0.06 | **matches AdamW** via the default `factored_v_2d` — [details](#quality-lever-factored-second-moment-on-2d-params-factored_v_2d) |
| Modern decoders (Qwen3 / Llama-3 / Mistral — SwiGLU + grouped-query attention) | uses *more* optimizer memory than AdamW on these | keeps the full ≈1 B/param optimizer state (about a quarter of bf16 AdamW's) |
| Learning rate on those architectures | no guidance — silently over-steps | documented ≈0.6× AdamW, so quality matches AdamW |
| Optimizer-step speed | baseline | ≈2× faster `opt.step()` (fused kernels), identical results |
| Peak memory during the step | large transient spikes | much lower peak — room for bigger models / batches |
| Sharded multi-GPU training (FSDP2) | breaks with the fast path | works — for plain Gefen *and* Muon |
| Whole-model Muon | 2D weight matrices only | `GefenMuonHybrid` trains the entire model |
| Muon step efficiency | generic momentum hack + redundant dequant gather | single-pass bit-exact momentum kernel |
| Save / resume checkpoints | can corrupt state or lose tuning on resume | saves &amp; resumes correctly |
| Crash safety | missing device / edge-case guards | guarded against wrong-device, empty-tensor, and race bugs |
| Tested correctness | no fused-kernel tests | bit-exact + distributed parity test suite |
| Getting started | no Axolotl / fork-install guidance | Axolotl how-to + fair loss/speed/memory benchmarks |
| Muon loss vs AdamW | Muon trails AdamW | the recommended hybrid config puts Gefen-Muon on the AdamW loss level — [details](#full-models-gefenmuonhybrid) |
| Muon (Newton-Schulz) speed | fixed 5-step schedule | tunable `ns_schedule`; the hybrid defaults to `tuned3` (≈1.4× faster, same loss) — [details](#experimental-lever-faster-newton-schulz-ns_schedule-fp8_ns) |
| Low-precision orthogonalization | bf16 only | opt-in `fp8_ns` for large matrices on newer GPUs, safe fallback elsewhere — [details](#experimental-lever-faster-newton-schulz-ns_schedule-fp8_ns) |
| Sharded Muon under FSDP2 | every GPU redundantly repeats the same work | `sharded_mode="distributed"` splits the work across GPUs at identical results — [details](#experimental-lever-sharded-newton-schulz-under-fsdp2-sharded_mode) |

All measured numbers live in [Benchmarks](#benchmarks) and the lever sections below.

</details>

 
## Installation

> **This fork is not on PyPI.** `pip install gefen` fetches the *upstream* release, which does **not** include the fixes/improvements above — install this fork from source instead.

Build from source (editable install):

```bash
git clone https://github.com/thad0ctor/Gefen-X
cd Gefen-X
pip install -e .          # imports as `gefen`
```

> **Install PyTorch first.** `torch>=2.5` is a dependency, but pip's default PyTorch wheel may be CPU-only or built for a CUDA version that doesn't match your machine. Install the [PyTorch build for your platform/CUDA](https://pytorch.org/get-started/locally/) **before** installing Gefen. A CPU-only or mismatched build won't crash — Gefen falls back to pure PyTorch — but you lose the fused CUDA kernels.

The package is pure-Python at install time, the fused kernels (CUDA required) are **built on first use** via PyTorch JIT (`torch.utils.cpp_extension`) + `nvcc`. The first CUDA run takes a few minutes; later runs reuse the cached build keyed on your Python / PyTorch / CUDA version, GPU architecture, and the Gefen source checkout. Force a rebuild with `GEFEN_FORCE_REBUILD=1`.

**Requirements:** a CUDA toolkit and host compiler compatible with your PyTorch build (the JIT compiles for your GPU's compute capability; set `TORCH_CUDA_ARCH_LIST` to target arch or a multi-arch build).

**Supported versions**

| | |
|---|---|
| Python | 3.10+ (verified on 3.12) |
| PyTorch | 2.5+ (verified on 2.12 / cu133) |
| Platform | Linux with a CUDA GPU |

The Hugging Face Trainer flow, the `find_lr` tool, and the `examples/` all use `transformers` and `datasets`, which are not core dependencies — install them alongside (`pip install transformers datasets`). Optional benchmark baselines: `bitsandbytes` (AdamW-8bit), `torchao` (AdamW-4bit).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Fused CUDA kernel fails to build on the first step | Gefen auto-falls back to `fused=False` (pure-PyTorch, slower) with a warning — training continues. Set `GEFEN_VERBOSE_BUILD=1` for the full build diagnostic; the usual cause is an `nvcc` that doesn't match `torch.version.cuda`. |
| Changed GPU / arch and the kernel misbehaves or won't load | The build cache is keyed on the effective arch list (`TORCH_CUDA_ARCH_LIST`, or the detected GPU when unset), so a changed arch rebuilds automatically. If you kept an explicit `TORCH_CUDA_ARCH_LIST` across a GPU change, force a fresh build with `GEFEN_FORCE_REBUILD=1`. |
| Loss diverges or plateaus high | LR is too hot — use **≈0.6× your AdamW LR** ([Learning rate](#learning-rate-when-porting-an-adamw-config)) or measure it with the [`find_lr`](https://github.com/thad0ctor/Gefen-X/blob/main/src/gefen/tools/README.md) tool. |
| Out of memory | See [Hardware Requirements](https://github.com/thad0ctor/Gefen-X/blob/main/docs/hardware.md) for measured peak VRAM per model size and method. |

## Compatibility

Validated with full-parameter fine-tune smoke tests on 24 GB GPUs across modern LLMs/VLMs and media-generation models — Gemma 4, Qwen3-VL, Mistral 3, MiniCPM-V 4.6, LFM2.5 (+VL), SDXL, Stable Diffusion 3.5, FLUX.1-dev (12B via FSDP2), FLUX.2-klein, Z-Image (6B via FSDP2), Anima, and ACE-Step 1.5 — and on 20-30B MoEs (GPT-OSS-20B, GLM-4.7-Flash) via LoRA adapter training. Per-model results, method, and hardware footprints: **[COMPATIBILITY.md](https://github.com/thad0ctor/Gefen-X/blob/main/COMPATIBILITY.md)**.

**Sizing a GPU?** [Hardware Requirements](https://github.com/thad0ctor/Gefen-X/blob/main/docs/hardware.md) gives measured peak VRAM per model size and method (full fine-tune with Gefen vs AdamW, LoRA, QLoRA), plus a per-card rule of thumb for other sizes.

## Quick Start

```python
import torch
from gefen import Gefen

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.nn.Linear(128, 10).to(device)

# optimizer = torch.optim.AdamW(
optimizer = Gefen(  # Replace AdamW with Gefen:
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)

inputs = torch.randn(32, 128, device=device)
targets = torch.randint(0, 10, (32,), device=device)

logits = model(inputs)
loss = torch.nn.functional.cross_entropy(logits, targets)
loss.backward()

optimizer.step()
optimizer.zero_grad(set_to_none=True)

print('Finished successfully.')
```

**Want a real end-to-end run?** See [`examples/toy-finetune/`](https://github.com/thad0ctor/Gefen-X/blob/main/examples/toy-finetune/README.md) — fine-tune a small model in about four minutes on one 24 GB GPU, with a before/after check that makes success obvious.

### Learning rate (when porting an AdamW config)

Use **≈0.6× your AdamW learning rate** (e.g. AdamW at `5e-5` → Gefen at `3e-5`). That is Gefen's measured optimum on the models tested (Qwen3 0.6B–8B), and at it Gefen matches AdamW's loss. Don't creep the LR back up toward the AdamW value — Gefen is less tolerant of a too-hot LR than AdamW is.

> **Better than the heuristic: measure it.** The LR finder picks the best LR for *your* model empirically:
> ```bash
> python -m gefen.tools.find_lr --model <path> --optimizer gefen --method sweep
> ```
> See [`tools/README.md`](https://github.com/thad0ctor/Gefen-X/blob/main/src/gefen/tools/README.md).

One knock-on effect: weight decay in AdamW-style optimizers is applied as `lr × weight_decay`, so scaling `lr` down to 0.6× also weakens regularization by 0.6×. If your recipe relies on weight decay, scale `weight_decay` up by the inverse (≈1.7×) or retune it.

<details>
<summary>Background: where the ≈0.6× factor comes from</summary>

Gefen's compressed optimizer state takes slightly larger effective steps than AdamW on a few small, high-leverage tensors (most sharply the RMSNorm weights, whose whole tensor shares one statistic in the legacy mode). Those tensors set the stability ceiling, which lands the usable LR at ≈0.6× AdamW's. The factor is set by the architecture's norm/`head_dim` structure rather than model size — measured across Qwen3-0.6B → 8B (all `head_dim=128`) the usable factor stayed in the ≈0.57–0.78 band — so re-measure only for a different architecture. The default `factored_v_2d` second moment lands at the same ≈0.6× optimum empirically (fair-LR sweep at 0.6B and 1.7B).

Measured on a Qwen3-4B full fine-tune: at its own ≈0.6× optimum Gefen ties AdamW at 300 and 800 steps with AdamW-like run-to-run spread; forced up to an AdamW-scale LR it lands ≈0.3–0.4 worse with ≈5× the spread.

</details>

## Benchmarks

Optimizer comparison on a full fine-tune, **with each optimizer at its own fair learning-rate optimum** (from a per-optimizer LR sweep), 2000 steps.

You can run these yourself — see [`benchmarks/`](https://github.com/thad0ctor/Gefen-X/blob/main/benchmarks/README.md) for what each suite measures and how to reproduce the tables/plots (`bash benchmarks/optimizer-sweep/run.sh` regenerates this comparison, with the recommended Gefen-Muon config applied by default).

**Testing environment**
- **Hardware:** NVIDIA RTX 3090 for Qwen3-1.7B and RTX 3090 Ti for Qwen3-0.6B (Ampere, sm_86), single GPU per run; compare optimizers within a model, not tok/s across models.
- **Software:** PyTorch 2.12.0 (cu133), Python 3.12; Gefen fused CUDA kernels JIT-built for sm_86.
- **Models:** Qwen3-0.6B and Qwen3-1.7B — full fine-tune (all weights trained, no adapters).
- **Regime:** bf16 master weights, gradient checkpointing, sequence length 2048, micro-batch 1, Alpaca greedy-packed to 2048-token blocks, 2000 steps, identical data order across optimizers, 32-example held-out eval.
- **Noise:** single seed; the eval metric moves about ±0.005–0.007 between reruns (bf16 nondeterminism), so treat eval differences under 0.01 as a tie.
- **Learning rate (each at its own fair optimum):** AdamW family `5e-5`; Gefen `3e-5`; Gefen-Muon `5e-5` with the recommended hybrid config.
- **Optimizers:** `adamw_bf16` = torch fused AdamW · `adamw8bit` = bitsandbytes · `adamw4bit` = torchao · `gefen_fused` = `Gefen(fused=True)` · `gefen_muon` = `GefenMuonHybrid(..., fused=True)` (Muon on 2D hidden matrices, Gefen on everything else). **Both Gefen runs use the fused CUDA kernels** (`fused=True`).

<details>
<summary><b>Full results table</b> (per-optimizer numbers)</summary>

| Model | Optimizer | LR | Eval loss | tok/s | Peak VRAM (GiB) | Opt-state B/param |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | adamw_bf16 | 5e-5 | 1.433 | 5600 | 6.95 | 4.00 |
| Qwen3-0.6B | adamw8bit | 5e-5 | 1.474 | 5284 | 5.87 | 2.03 |
| Qwen3-0.6B | adamw4bit | 5e-5 | 1.472 | 4987 | 6.32 | 1.06 |
| Qwen3-0.6B | **gefen_fused** | 3e-5 | **1.420** | 5376 | **5.30** | **1.01** |
| Qwen3-0.6B | **gefen_muon** | 5e-5 | 1.425 | 3614 | **5.30** | **1.01** |
| Qwen3-1.7B | adamw_bf16 | 5e-5 | **1.217** | 2985 | 14.00 | 4.00 |
| Qwen3-1.7B | adamw8bit | 5e-5 | 1.240 | 2704 | 10.84 | 2.03 |
| Qwen3-1.7B | adamw4bit | 5e-5 | 1.242 | 2622 | 15.11 | 1.06 |
| Qwen3-1.7B | **gefen_fused** | 3e-5 | **1.226** | 2870 | **9.21** | **1.01** |
| Qwen3-1.7B | **gefen_muon** | 5e-5 | 1.241 | 1314 | 9.20 | **1.01** |

> `gefen_fused` uses the shipped defaults; `gefen_muon` uses the [recommended hybrid recipe](#full-models-gefenmuonhybrid) (`backup_lr` ≈0.5× `lr`, `backup_1d_period_one=True`). Both run at their own best learning rate. Older (pre-default) behaviors are still available behind flags — see the lever sections below for what each default does and how to switch it off.

</details>

![Qwen3-1.7B optimizer comparison](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/quad_qwen1p7b.png)
![Qwen3-0.6B optimizer comparison](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/quad_qwen0p6b.png)

**Loss curves** — loss over the 2000 steps. For each optimizer (distinguished by **color**), there are two lines: **solid = held-out eval loss** (the 32-example validation set, logged every 50 steps — this is the comparison metric in the table above) and **dashed = train-loss EMA** (exponential moving average of the training loss). 

**Gefen-fused (with the `factored_v_2d` default) sits on the AdamW cluster at both scales** — best-in-table at 0.6B (1.4199, −0.014 below the best AdamW) and a statistical tie at 1.7B (1.2257 vs 1.2169; each eval number moves ±0.005–0.007 between reruns, so gaps under ≈0.01 are ties). Gefen-Muon matches at 0.6B (1.4249, a tie) and lands +0.024 above the best AdamW at 1.7B (1.2411 vs 1.2169).

![Qwen3-1.7B loss curves](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/loss_curve_qwen1p7b.png)
![Qwen3-0.6B loss curves](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/loss_curve_qwen0p6b.png)

**Throughput vs peak VRAM** — the speed/memory frontier (upper-left = faster *and* lighter is better). Gefen-fused holds the lowest-VRAM column at ≈0.96× the fused-AdamW throughput at both scales; AdamW-4-bit is worst on *both* axes despite its small optimizer state (torchao transient buffers). 

Gefen-Muon shares Gefen-fused's lowest-VRAM column but sits far lower on throughput — the Newton-Schulz orthogonalization makes it the slowest optimizer.

![Qwen3-1.7B throughput vs VRAM](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/tput_vram_qwen1p7b.png)
![Qwen3-0.6B throughput vs VRAM](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/tput_vram_qwen0p6b.png)

**Gefen-fused** is the recommended configuration: **AdamW-level loss with the lowest peak VRAM and the lowest optimizer state in the table**, at ≈0.96× AdamW's speed. (AdamW-4-bit's optimizer state is also small, but its peak VRAM is the *highest* in the table because of its transient buffers — Gefen is the real memory winner.)

**Gefen-Muon** reaches the same loss level at the same memory footprint, but at roughly half the speed (Muon's orthogonalization step is expensive). Choose it if you specifically want Muon-style updates; otherwise Gefen-fused gives the same quality faster.

**Review the raw runs:** per-cell training logs (step-by-step loss, throughput, VRAM) in [`docs/benchmarks/logs/`](https://github.com/thad0ctor/Gefen-X/tree/main/docs/benchmarks/logs/) · aggregated metrics as [CSV](https://github.com/thad0ctor/Gefen-X/blob/main/docs/benchmarks/optimizer_comparison_2000steps.csv) and [JSONL](https://github.com/thad0ctor/Gefen-X/blob/main/docs/benchmarks/optimizer_comparison_2000steps.jsonl).

## Hugging Face Trainer

Until native `optim="gefen"` support is released in Transformers, pass Gefen to the Trainer with `optimizer_cls_and_kwargs`:

```python
from gefen import Gefen
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="outputs",
    learning_rate=3e-5,   # ≈0.6× your AdamW LR — see "Learning rate" above; don't copy an AdamW-scale LR here
    weight_decay=0.0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    optimizer_cls_and_kwargs=(
        Gefen,
        {
            "lr": training_args.learning_rate,
            "betas": (training_args.adam_beta1, training_args.adam_beta2),
            "eps": training_args.adam_epsilon,
            "fused": True,
        },
    ),
)
```

### Distributed Training

Gefen drops into standard distributed training like any other PyTorch optimizer, with either `fused=True` or `fused=False`. Validated setups: single-GPU, PyTorch DDP, and FSDP2 (`fully_shard` / DTensor). DeepSpeed ZeRO is not yet validated.


### Extension: Gefen-Muon

`GefenMuon` adds a [Muon](https://kellerjordan.github.io/posts/muon/)-style pseudo-orthogonalization step (Keller Jordan et al., "Muon", 2024) on the first moment (skipping the second moment), then quantizes those first moments to 8-bit with Gefen's partitioning quantization.

> **Scope — read this first.** Like Muon, `GefenMuon` optimizes **2D hidden weight matrices only**. It has *no* code path for embeddings, the LM head, or 1D parameters (norms, biases) — feeding it those will fail or misbehave. For a full model, use [`GefenMuonHybrid`](#full-models-gefenmuonhybrid) below, which routes the non-Muon parameters to `Gefen` for you.
>
> **Pass `(name, param)` pairs, not bare tensors.** `GefenMuon` keys its 8-bit codebook cache on each parameter's name. If you strip the names (e.g. `[p for _, p in pairs]`) every parameter collapses to the name `"none"` and the cache is corrupted. Always pass the named pairs through:

```python
from gefen import GefenMuon

# muon_named_params: list of (name, param) for your 2D hidden weight matrices.
optimizer = GefenMuon(muon_named_params, lr=lr)   # note: raw GefenMuon defaults weight_decay=0.1 (the standard Muon recipe), unlike Gefen / the hybrid, which default 0.0
```

#### Full models: `GefenMuonHybrid`

`GefenMuonHybrid` is the drop-in for training a **whole model** with Muon. It routes 2D hidden weight matrices to `GefenMuon` and everything else (embeddings, LM head, norms, biases) to `Gefen`, behind a single `torch.optim.Optimizer` interface. Both sub-optimizers are 8-bit, so the whole optimizer-state footprint stays small — unlike a stock Muon+AdamW setup where the AdamW half is full precision.

Hand it a model and it splits and routes the parameters for you — no manual param lists:

```python
from gefen import GefenMuonHybrid

optimizer = GefenMuonHybrid(model, lr=ADAMW_LR)               # single-argument form
optimizer = GefenMuonHybrid.from_model(model, lr=ADAMW_LR)    # equivalent explicit classmethod
```

**Recommended configuration** (lowest loss vs AdamW at matched LR — this is what the `gefen_muon` benchmark row uses): keep the Muon half at AdamW scale and lower only the backup half, with per-element 2nd moment on its 1D tensors. Both are free on throughput.

```python
optimizer = GefenMuonHybrid(
    model,
    lr=ADAMW_LR,                   # Muon (2D) half stays at AdamW scale
    backup_lr=0.5 * ADAMW_LR,      # backup half ≈0.5× (tune 0.4–0.6×) — the main loss lever
    backup_1d_period_one=True,     # AdamW-like per-element 2nd moment on norms/biases
    adjust_lr_fn="match_rms_adamw",
    fused=True,
)
```

> **Manual control.** For a custom split, `gefen.split_params_for_muon(model)` returns the `(muon_named, backup_named)` `(name, param)` lists; pass them as the two positional arguments (`GefenMuonHybrid(muon_named, backup_named, lr=...)`) and adjust as needed. The single-argument and `from_model` forms cover the common case.

**Muon learning rate — keep `adjust_lr_fn="match_rms_adamw"` (the default).** The hybrid feeds a single `lr` to both halves, but Muon matrices and the Gefen backup want different LR scales, so the shared `lr` only works after a per-parameter rescale:

 - **`"match_rms_adamw"`** (the default) rescales Muon updates to AdamW-equivalent size, so one AdamW-scale `lr` works for the whole model.
 - **`None`** keeps Muon's native scaling — an AdamW-sized `lr` then under-trains the Muon matrices, and no single `lr` suits both halves. Don't combine `None` with an AdamW-scale `lr`.

**Optional knobs** (`GefenMuonHybrid`):

| Knob | Default | What it does | When to change it |
|---|---|---|---|
| `backup_lr` | `= lr` | learning rate for the Gefen backup half | set to ≈0.5× `lr` (tune 0.4–0.6×) — the recommended recipe and the main loss lever |
| `backup_1d_period_one` | `False` | AdamW-like per-element 2nd moment on 1D backup tensors (norms/biases); free on throughput | `True` in the recommended recipe |
| `ns_schedule` | `"tuned3"` | faster Newton-Schulz schedule (≈1.4× step throughput, loss-neutral) — [details](#experimental-lever-faster-newton-schulz-ns_schedule-fp8_ns) | `"standard"` for the bit-exact classic quintic |
| `normuon` | `True` | free per-neuron 2nd moment on the NS output; small consistent loss win — [details](#quality-lever-per-neuron-2nd-moment-on-the-newton-schulz-output-normuon) | `False` to disable |
| `backup_2d_period_one` | `False` | per-element 2nd moment on the embedding/LM head — matches AdamW's loss at extra memory — [details](#experimental-lever-match-adamw-2nd-moment-on-embed--lm-head-backup_2d_period_one) | turn on to close the last loss gap at 1.7B+ |
| `stochastic_round` | `False` | unbiased rounding for the 8-bit momentum (free, loss-neutral) | optional |
| `capturable` | `False` | CUDA-graph-capturable `step()`, like `torch.optim`'s `capturable` — [details](https://github.com/thad0ctor/Gefen-X/blob/main/COMPATIBILITY.md#cuda-graphs--torchcompile-capturable) | turn on to capture `step()` in a `torch.cuda.CUDAGraph` |

It supports `step()`, `zero_grad()`, `state_dict()`/`load_state_dict()`, and LR schedulers (e.g. `torch.optim.lr_scheduler.StepLR(optimizer, ...)`) like any optimizer. Because it splits params at construction rather than taking a single iterable, build it yourself and hand it to the Hugging Face `Trainer` via `optimizers=` (not `optimizer_cls_and_kwargs`):

```python
optimizer = GefenMuonHybrid(model, lr=training_args.learning_rate,
                            backup_lr=0.5 * training_args.learning_rate, backup_1d_period_one=True)
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                  optimizers=(optimizer, None))  # (optimizer, lr_scheduler)
```

## Using Gefen with Axolotl

[Axolotl](https://github.com/axolotl-ai-cloud/axolotl) can train with Gefen via its `optimizer: gefen` integration — see the (draft) PR [axolotl-ai-cloud/axolotl#3755](https://github.com/axolotl-ai-cloud/axolotl/pull/3755).

> ⚠️ **Install this fork from source, not the PyPI package.** PR #3755 documents  `pip install gefen`, which pulls **upstream** Gefen — *without* this fork's fixes (the modern-arch `period==1` memory fallback, the ≈2× fused kernels, the robustness guards). Install this fork from source (see [Installation](#installation)); axolotl's `optimizer: gefen` then imports whichever `gefen` is on the path..

**Basic** (memory-efficient AdamW drop-in):

```yaml
optimizer: gefen
learning_rate: 6.0e-6    # ≈ 0.6× the AdamW LR you'd use — Gefen needs a lower LR
weight_decay: 0.0
```

`6.0e-6` here is the `≈0.6×` *fallback heuristic*; to set it properly for your model, run the LR finder (`python -m gefen.tools.find_lr --model <path> --optimizer gefen --method sweep`) — see [`tools/README.md`](https://github.com/thad0ctor/Gefen-X/blob/main/src/gefen/tools/README.md).

**Fused kernels** (recommended — ≈2× faster `opt.step`, bit-exact, requires CUDA):

```yaml
optimizer: gefen
learning_rate: 6.0e-6
optim_args:
  fused: true
```

`learning_rate`, `weight_decay`, `adam_beta1`/`adam_beta2`, `adam_epsilon` are forwarded straight to the `Gefen` constructor; anything under `optim_args` (e.g. `fused`) is passed through as extra constructor kwargs.

**The levers, and where each lives:**

| Lever | How (axolotl) | Notes |
|---|---|---|
| Select Gefen | `optimizer: gefen` | added by PR #3755 |
| Fused kernels | `optim_args: { fused: true }` | ≈2× faster `opt.step`; bit-exact |
| Learning rate | `learning_rate: <≈0.6× AdamW>` | `≈0.6×` is a fallback heuristic — better, find it empirically with the LR finder (`python -m gefen.tools.find_lr …`, see [`tools/README.md`](https://github.com/thad0ctor/Gefen-X/blob/main/src/gefen/tools/README.md)). Background: [Learning rate](#learning-rate-when-porting-an-adamw-config) |
| Betas / eps / weight decay | standard `adam_beta1/2`, `adam_epsilon`, `weight_decay` | forwarded to Gefen |
| `period==1` memory fallback | *on by default in this fork* | restores ≈1 B/param on modern decoders; it's the module flag `MEMORY_SAFE_FALLBACK`, not a YAML key |

**Not (yet) exposed via axolotl config:** the **Muon hybrid** (`GefenMuonHybrid` + `adjust_lr_fn`) is **not** in PR #3755 (the PR notes Muon is "more involved") — use it directly in Python per [Gefen-Muon](#extension-gefen-muon) above. `MEMORY_SAFE_FALLBACK` is a module default (on in this fork), toggled in code rather than YAML.

> PR #3755 is an early draft and marked *untested* upstream; the config keys above may change — check the PR for the current state.

## Quality Lever: factored second moment on 2D params (`factored_v_2d`)

> **On by default** — this is what makes plain Gefen match AdamW's loss. Set `factored_v_2d=False` for the legacy behavior: ≈4% faster end-to-end at 1.7B (on par with fused AdamW's throughput), about 0.06 worse eval loss, and a lower best learning rate (`2e-5` instead of `3e-5` at the benchmarked scales).

Gefen historically shared one adaptive step size across each *block* of parameters, where AdamW keeps one per parameter — that granularity difference was the ≈0.06 loss gap. `factored_v_2d` gives every parameter of a weight matrix its own step size by tracking just one statistic per row and one per column (the [Adafactor](https://arxiv.org/abs/1804.04235) idea) and combining them on the fly inside the update kernel. The stored state is only rows + columns per matrix, so the ≈1 B/param memory story is unchanged.

```python
opt = Gefen(model.named_parameters(), lr=0.6 * ADAMW_LR, fused=True)  # factored-v is the default
```

| | **Gefen (default)** | AdamW bf16 | Gefen legacy (`factored_v_2d=False`) |
|---|---|---|---|
| Qwen3-0.6B eval loss | **1.420** | 1.433 | 1.497 |
| Qwen3-1.7B eval loss | **1.226** | 1.217 | 1.286 |
| Qwen3-1.7B tok/s | 2870 | 2985 | 2996 |
| Optimizer state | **1.01 B/param** | 4.00 | 1.01 |

![Gefen factored-v ablation — eval loss vs AdamW, factored-v on vs off](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/gefen_factored_v_loss.png)

> Chart from the original factored-v ablation run; the table above cites the current benchmark re-runs (per-cell logs, including the legacy-mode cell, in [`docs/benchmarks/logs/`](https://github.com/thad0ctor/Gefen-X/tree/main/docs/benchmarks/logs/)).

<details>
<summary>Details: validation, checkpoint migration, and limits</summary>

- Validated at both scales in the fair-LR regime, with a second-seed replication at 0.6B and a learning-rate sweep at 1.7B (best LR `3e-5` ≈ 0.6× AdamW's, matching Gefen's documented heuristic). The fused kernel computes the per-element step size in registers (no extra temporaries; step transients measured 0 MiB) and is covered by `tests/test_gefen_factored_v.py`.
- Checkpoints migrate automatically in both directions (old checkpoints work with the new default and vice versa; the second-moment statistics re-warm briefly and harmlessly).
- `GefenMuonHybrid`'s backup half stays on the legacy path (that combination hasn't been benchmarked), and under FSDP2 sharded 2D params fall back to the legacy path as well.
- Two sibling experiments from the same investigation ship off by default because they measured **no effect**: `period_one_substrings` (per-element state on name-matched tensors) and `codebook_refresh_every` (periodic codebook refit).

</details>

## Experimental Lever: faster Newton-Schulz (`ns_schedule`, `fp8_ns`)

> `GefenMuonHybrid` defaults to `ns_schedule="tuned3"` (≈1.4× faster steps, same loss). Pass `ns_schedule="standard"` for the classic schedule; `fp8_ns` is off by default.

The orthogonalization step is ≈90% of a Muon step, so it has two speed levers:

- **`ns_schedule`** — a tuned schedule that reaches the same result in fewer iterations. `"tuned4"` is ≈1.2× faster at equal quality; `"tuned3"` (the hybrid default) is ≈1.6× faster with a negligible quality cost.
- **`fp8_ns`** — run the matrix math in fp8 on GPUs that support it (RTX 40-series/Hopper and newer) for matrices large enough to benefit; everywhere else it silently falls back to normal precision, so it's always safe to enable.

```python
opt = GefenMuonHybrid(
    muon_named_params, backup_named_params,
    lr=5e-5, adjust_lr_fn="match_rms_adamw", fused=True,
    ns_schedule="tuned4",   # "tuned3" (hybrid default, ~1.6x) | "tuned4" (equal-quality, ~1.2x) | "standard" (classic quintic)
    fp8_ns=True,            # sm_89+ large matrices only; a no-op bf16 fallback elsewhere
)
```

Measured: no lever hurts loss — all variants track default Muon. `fp8_ns` pays off on larger models (≈1.2× end-to-end at Qwen3-4B) and its built-in size gate avoids the penalty on small ones; `tuned3` runs ≈1.4× faster end-to-end at the same loss, which is why it's the default.

![Gefen-Muon faster Newton-Schulz — real-training loss (AdamW + variants)](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/muon_ns_loss.png)
![Gefen-Muon faster Newton-Schulz — end-to-end throughput (tok/s)](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/muon_ns_throughput.png)

## Quality Lever: per-neuron 2nd moment on the Newton-Schulz output (`normuon`)

> On by default in `GefenMuonHybrid`; free on speed and memory. Disable with `normuon=False`.

Muon's orthogonalization step leaves some output neurons taking slightly bigger steps than others. `normuon` (after [NorMuon](https://arxiv.org/abs/2510.05491)) evens that out by tracking a running average per neuron and normalizing against it — at essentially zero cost in speed or memory.

Measured: a small but consistent loss win (−0.007 to −0.010 at 0.6B across two seeds, and about a third of the remaining gap to AdamW at 1.7B). A related experiment (`cautious=True`, sign-agreement masking) clearly hurt and ships off.

![Gefen-Muon normuon — eval loss vs AdamW and recommended](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/muon_normuon_loss.png)

## Experimental Lever: match-AdamW 2nd moment on embed / LM-head (`backup_2d_period_one`)

> Opt-in, default off. Unlike the levers above, this one **costs memory** — it's a deliberate loss-for-memory trade.

The hybrid's last sliver of loss gap at larger scales comes from the shared statistics on the embedding and LM head. This knob gives those two tensors full per-element statistics (like AdamW keeps everywhere), closing the gap at the cost of extra optimizer memory on them.

```python
opt = GefenMuonHybrid(
    model,
    lr=5e-5, backup_lr=2.5e-5, adjust_lr_fn="match_rms_adamw",
    backup_1d_period_one=True,
    backup_2d_period_one=True,   # per-element v on embed/LM-head — matches AdamW loss, costs memory
)
```

Measured at Qwen3-1.7B across two seeds: the gap to AdamW goes from ≈+0.02 to ≈0.00, at 2.45 B/param optimizer state (still well under AdamW's 4.0), with no speed cost. Leave it off to keep the ≈1 B/param memory-first footprint.

![Gefen-Muon — per-element 2nd moment on embed/LM-head closes the AdamW gap](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/muon_l5_loss_qwen1p7b.png)

A related opt-in, `stochastic_round=True`, switches the 8-bit momentum to unbiased rounding. It's free and validated but measured loss-neutral, so it stays optional.

## CUDA Graphs & torch.compile (`capturable`)

All three optimizers accept `capturable=True` (same meaning as `torch.optim`'s argument): `opt.step()` can then be captured in a `torch.cuda.CUDAGraph` or wrapped in `torch.compile(mode="reduce-overhead")` at no step-time cost — and the compiled hybrid step is about 10% faster than eager. Usage, caveats, and measured numbers: [COMPATIBILITY.md](https://github.com/thad0ctor/Gefen-X/blob/main/COMPATIBILITY.md#cuda-graphs--torchcompile-capturable).

## Experimental Lever: sharded Newton-Schulz under FSDP2 (`sharded_mode`)

> Opt-in. Default is `sharded_mode="exact"` (single-GPU parity).

Under FSDP2, Muon's orthogonalization needs each full weight matrix, but every GPU only holds a slice. Three strategies:

- **`"exact"`** (default) — every GPU rebuilds every matrix and does the same work. Identical to single-GPU results, but redundant.
- **`"distributed"`** (experimental) — each matrix is assigned by its stable position in the full distributed parameter set to one GPU, which does the work and shares the result. Bit-identical to `"exact"` on homogeneous GPUs and faster as you add them. Momentum ownership is stable when the active gradient set varies, and `state_dict()` collectively gathers owner-local momentum so rank 0 can write a complete checkpoint after all ranks call it.
- **`"approx"`** — each GPU works on just its slice. Fastest, but results genuinely differ — an accuracy trade.

```python
from gefen import GefenMuonHybrid

opt = GefenMuonHybrid(
    muon_named_params, backup_named_params,
    lr=5e-5, adjust_lr_fn="match_rms_adamw", fused=True,
    sharded_mode="distributed",  # "exact" (default, parity) | "distributed" (experimental, collective checkpoint) | "approx" (fastest, non-parity)
)
# only takes effect under FSDP2 (DTensor params); no-op single-GPU
```

> **`"distributed"` checkpointing is collective.** `state_dict()` gathers each owner's momentum across ranks, so **every rank must call it** (as in a standard FSDP full-state-dict flow). Calling `state_dict()` on rank 0 only — e.g. a rank-0-only save loop — **deadlocks**. Save and load also transiently materialize the full unsharded momentum on every rank, so peak memory at checkpoint time approaches `"exact"` mode's.

![Gefen-Muon exact / distributed / approx sharded — eval loss](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/muon_shard_loss.png)
![Gefen-Muon exact / distributed / approx sharded — throughput & VRAM](https://raw.githubusercontent.com/thad0ctor/Gefen-X/main/docs/benchmarks/muon_shard_perf.png)

Measured (Qwen3-0.6B, 2 and 4 GPUs): `"distributed"` is a free speedup at identical results (1.06× / 1.12×); `"approx"` is faster still (1.25× / 1.39×) but visibly costs training quality, and the cost grows with GPU count. The `"distributed"` win grows with model size.

## Param Groups and Integrations

`Gefen`, `GefenMuon`, and `GefenMuonHybrid` preserve the parameter groups you pass to the optimizer, so list-indexed layer-wise LR recipes, per-group LR logging, and `state_dict()["param_groups"]` see the same group boundaries as conventional `torch.optim` optimizers. Per-parameter names are stored in optimizer state and mirrored as each group's `param_names` list for integrations that need name-level routing. Checkpoints from the older one-group-per-parameter layout are migrated on load when the total parameter order still matches and the old per-param hyperparameters can be represented by the new group layout.

## Known limitations

- **Hybrid checkpoint schema.** `GefenMuonHybrid`'s `state_dict()` uses its own nested `{"muon": ..., "backup": ...}` layout. Resume from a checkpoint Gefen itself saved — not one consolidated or converted to the flat torch `{state, param_groups}` layout (those are rejected, not silently ignored).

## Citation:

If you found this library useful, please consider citing our work:

```bibtex
@article{benedek2026gefen,
  title={Gefen: Optimized Stochastic Optimizer},
  author={Benedek, Nadav and Koren, Tomer and Fried, Ohad},
  journal={arXiv preprint arXiv:2606.13894},
  year={2026}
}
```

## Paper:

# [Gefen: Optimized Stochastic Optimizer](https://arxiv.org/pdf/2606.13894)
