# Architecture Compatibility

Validation of the Gefen optimizer (`fused=True`, `factored_v_2d=True` — the defaults) across a broad range of model architecture families. Each family is validated the way that best fits it: models with a standard training task are trained end to end on a real dataset and their accuracy or mAP is compared against AdamW; models without one run a fixed-batch smoke test — a small batch trained for 40–300 steps that must show a sharp, finite loss drop (memorization). Both confirm the same thing: Gefen's fused step drives every parameter tensor the architecture produces. GPUs are 24 GB (RTX 3090) unless a Hardware column or note says otherwise.

> **Scope.** This is not an attempt to validate every model. It deliberately samples a wide range of architecture *families* — dense and Mixture-of-Experts LLMs, vision-language models, convolutional image classifiers and object detectors, image / video / audio diffusion and flow-matching DiTs, state-space (Mamba-hybrid) models, and speech recognition / text-to-speech models — to show that Gefen's optimizer path handles the parameter tensor shapes, ranks, and dtypes each family produces. Breadth of architectures is the goal; exhaustive coverage of individual checkpoints is not.

## Matrix

| Model | Kind | Method | Result | Loss (first → last, Δ%) | Peak VRAM | Hardware |
|---|---|---|---|---|---|---|
| LFM2.5-1.2B-Instruct | LLM | full-param | ✅ PASS | 5.52 → 0.029 (−99.5%) | 6.0 GiB | 1×24 GB |
| Qwen3.5-0.8B | LLM | full-param | ✅ PASS | 4.64 → 0.0005 (−100%) | 4.5 GiB | 1×24 GB |
| Llama-3.2-1B | LLM | full-param | ✅ PASS | 4.54 → 0.032 (−99.3%) | 6.8 GiB | 1×24 GB |
| Nemotron-3-Nano-4B | LLM (Mamba-hybrid) | full-param | ✅ PASS | 9.67 → 0.0 (−100%) | 18.8 GiB | 1×24 GB |
| GPT-OSS-20B | LLM (MoE) | LoRA | ✅ PASS | 3.84 → 0.0001 (−100%) | 39 GiB total | 3×24 GB |
| GLM-4.7-Flash (30B-A3B) | LLM (MoE) | LoRA | ✅ PASS | 4.69 → 1.11 (−76.4%) | 59 GiB total | 5×24 GB |
| Qwen3.6-35B-A3B | LLM (MoE) | LoRA | ✅ PASS | 3.45 → 0.0002 (−100%) | 65 GiB total | 4×24 GB |
| Qwen3-VL-2B-Instruct | VLM | full-param | ✅ PASS | 24.04 → 0.036 (−99.9%) | 11.2 GiB | 1×24 GB |
| MiniCPM-V-4.6 | VLM | full-param | ✅ PASS | 18.46 → 0.052 (−99.7%) | 7.2 GiB | 1×24 GB |
| Ministral-3-3B-Instruct-2512 | VLM (Mistral 3) | full-param | ✅ PASS | 5.94 → 0.022 (−99.6%) | 19.9 GiB | 1×24 GB |
| LFM2.5-VL-1.6B | VLM | full-param | ✅ PASS | 11.77 → 0.036 (−99.7%) | 8.1 GiB | 1×24 GB |
| InternVL3-2B | VLM | full-param | ✅ PASS | 21.2 → 0.033 (−99.8%) | 9.9 GiB | 1×24 GB |
| PaddleOCR-VL-1.6 | VLM (OCR) | full-param | ✅ PASS | 15.6 → 0.037 (−99.8%) | 4.3 GiB | 1×24 GB |
| gemma-3-4b-it | VLM | full-param | ✅ PASS | 20.99 → 0.059 (−99.7%) | 22.9 GiB | 1×96 GB |
| gemma-4-E2B-it | VLM (audio+vision) | full-param | ✅ PASS | 18.40 → 0.021 (−99.9%) | 24.5 GiB total | 2×24 GB |
| gemma-4-31B-it | VLM | LoRA | ✅ PASS | 16.1 → 0.0 (−100%) | 59.6 GiB total | 4×24 GB |
| Stable Diffusion XL (UNet) | image diffusion | full-param | ✅ PASS | 0.658 → 0.0007 (−99.9%, 300 steps) | 12.7 GiB | 1×24 GB |
| Stable Diffusion 3.5 Medium | image MMDiT | full-param | ✅ PASS | 2.97 → 0.001 (−100%, 150 steps) | 11.0 GiB | 1×24 GB |
| FLUX.2-klein-4B | image flow | full-param | ✅ PASS | 2.433 → 0.013 (−99.5%) | 18.2 GiB | 1×24 GB |
| FLUX.1-dev | image flow | full-param FSDP2 | ✅ PASS | 3.018 → 0.003 (−99.9%, 100 steps) | 16.3 GiB/GPU | 4×24 GB |
| Anima (Base v1.0) | anime image DiT | full-param | ✅ PASS | 2.06 → 0.64 (−68.8%, 150 steps) | 9.3 GiB | 1×24 GB |
| Z-Image | image S3-DiT | full-param FSDP2 | ✅ PASS | 3.00 → 0.018 (−99.4%, 120 steps) | 15.6 GiB/GPU | 2×24 GB |
| LTX-2 | video+audio DiT | full-param | ✅ PASS | 5.04 → 2.95 (−41.5%, 120 steps) | 88.1 GiB | 1×96 GB |
| ACE-Step 1.5 turbo | music flow | full-param | ✅ PASS | 2.828 → 0.171 (−94.0%, 100 steps) | 8.9 GiB | 1×24 GB |

Parameter ranks exercised: 1-D through 5-D — norms/biases, linears, Conv1d/patch embeds, Conv2d (SDXL UNet, vision towers), Mamba conv/SSM tensors, the 5-D Qwen3-VL patch embed. All route through Gefen's flattened block partitioning without special-casing.

## Vision

Every row is a real training run compared head to head with AdamW at matched model / data / LR / schedule / epochs. The MNIST CNN and CIFAR-10 ResNet-18 are trained from scratch; the other classifiers fine-tune ImageNet-pretrained weights on Imagenette; detectors fine-tune COCO-pretrained weights on COCO128 and report mAP@50-95 / @50. Higher is better. Gefen `fused=True` defaults.

| Model | Kind | Task (metric ↑) | Gefen | AdamW |
|---|---|---|---|---|
| CNN (2-conv, PyTorch example) | CNN | MNIST acc (3 seeds) | 98.96% | 99.04% |
| ResNet-18 | CNN — residual | CIFAR-10 acc | 91.82% | 91.94% |
| ResNet-50 | CNN — residual | Imagenette acc | 99.08% | 98.96% |
| ConvNeXt-Tiny | CNN — modern conv | Imagenette acc | 99.34% | 99.36% |
| EfficientNet-B0 | CNN — MBConv | Imagenette acc | 98.29% | 98.24% |
| MobileNetV3-Large | CNN — inverted residual | Imagenette acc | 98.29% | 98.42% |
| DenseNet-121 | CNN — dense | Imagenette acc | 97.63% | 97.91% |
| RegNet-Y-1.6GF | CNN — grouped | Imagenette acc | 99.16% | 99.21% |
| ViT-B/16 | vision transformer | Imagenette acc | 99.03% | 99.29% |
| Swin-T | vision transformer — windowed | Imagenette acc | 99.03% | 99.16% |
| Faster R-CNN (R50-FPN) | detector — two-stage conv | COCO128 mAP | 0.767 / 0.974 | 0.782 / 0.973 |
| RetinaNet (R50-FPN) | detector — one-stage conv | COCO128 mAP | 0.790 / 0.925 | 0.822 / 0.942 |
| SSD300-VGG16 | detector — one-stage conv | COCO128 mAP | 0.684 / 0.896 | 0.701 / 0.891 |
| FCOS (R50-FPN) | detector — anchor-free conv | COCO128 mAP | 0.849 / 0.958 | 0.866 / 0.970 |
| YOLO11n | detector — YOLO (ultralytics) | COCO128 mAP | 0.702 / 0.882 | 0.680 / 0.872 |
| RT-DETR | detector — DETR transformer | COCO128 mAP | 0.828 / 0.949 | 0.854 / 0.971 |
| YOLOS | detector — ViT transformer | COCO128 mAP | 0.361 / 0.586 | 0.388 / 0.622 |
| Deformable-DETR | detector — deformable transformer | COCO128 mAP | 0.729 / 0.931 | 0.800 / 0.966 |
| RF-DETR Nano | detector — DETR (roboflow) | COCO128-person mAP | 0.726 / 0.909 | 0.701 / 0.910 |

## Audio

Same head-to-head setup. ASR rows report Speech Commands keyword accuracy (higher is better) — the M5 baseline is trained from scratch, while Wav2Vec2 / HuBERT / Wav2Vec2-Conformer fine-tune pretrained encoders. TTS has no accuracy metric, so it reports held-out LJSpeech validation loss after fine-tuning the pretrained model (lower is better).

| Model | Kind | Task | Gefen | AdamW |
|---|---|---|---|---|
| M5 raw-waveform Conv1d | speech recognition — Conv1d | Speech Commands acc ↑ | 83.64% | 84.86% |
| Wav2Vec2 | ASR — conv + transformer | Speech Commands acc ↑ | 93.6% | 93.4% |
| HuBERT | ASR — conv + transformer | Speech Commands acc ↑ | 94.6% | 93.55% |
| Wav2Vec2-Conformer | ASR — conv + attention | Speech Commands acc ↑ | 77.1% | 80.1% |
| Tacotron2 | TTS — conv + LSTM + attention | LJSpeech val loss ↓ | 0.578 | 0.583 |
| Dia | TTS — transformer, audio codebooks | LJSpeech val loss ↓ | 4.75 | 4.72 |

Gefen ≈ AdamW on the classifiers, ASR, and TTS, and leads on YOLO11n and RF-DETR; it trails by 1–7 mAP points on the other detectors and the Conformer at matched LR. That gap is mostly hyperparameter: a per-optimizer LR grid (Gefen's optimum ≈0.6× AdamW's, at 3e-5) shrinks it to ≤0.01 on five of seven detectors — Gefen ahead on RetinaNet and RT-DETR — and ≤0.03 on the rest, e.g. Deformable-DETR's −0.21 at a shared 1e-4 collapses to −0.03. The `Gefen-Muon` hybrid matched or beat AdamW where run.

Setup: from scratch — MNIST (paper recipe, 3 seeds), CIFAR-10 ResNet-18; fine-tuned — classifiers→Imagenette, detectors→COCO128, ASR→Speech Commands, TTS→LJSpeech. RTX 5090 + RTX PRO 6000. Harnesses: [`benchmarks/arch-compat/`](benchmarks/arch-compat/).

## Method

- **Real-dataset rows** (the Vision and Audio tables) report held-out accuracy, COCO mAP, or — where no accuracy metric exists (TTS) — validation loss, after a full training run: a convergence comparison against AdamW at matched model / data / LR / schedule / epochs.
- **Smoke-test rows** (the LLM / VLM / diffusion matrix above) train a fixed batch for N optimizer steps and report the loss as `first → last (Δ%)`, where Δ% is the reduction from the first step's loss to the last (`(first − last) / first`); PASS if it exceeds 30% with no NaN/Inf. Peak VRAM from `torch.cuda.max_memory_allocated`.
- The **Method** column: `full-param` trains every native parameter tensor; `full-param FSDP2` / `device_map` shard that same full-parameter training for models over one card; `LoRA` covers models too large to full-fine-tune even sharded — a weaker claim, since only the adapter matrices are trained. The FSDP2 rows validate training and loss reduction; the narrower same-topology full-state optimizer checkpoint contract is described below.
- **Versions**: LLM / VLM / diffusion smokes on torch 2.12.0+cu133, transformers 5.10.2 (≥ 5.11 for PaddleOCR-VL), diffusers 0.39.0. Vision / audio / real-data runs on torch 2.13.0+cu133, torchvision 0.28, torchaudio 2.11, transformers 5.12, ultralytics 8.4, rfdetr 1.8. Harness and per-model recipes: [`benchmarks/arch-compat/`](benchmarks/arch-compat/).

## Megatron-LM integration scope

Megatron-LM integration tests ran tiny mock-data GPT training through the production pretraining entry point for plain Gefen, GefenMuon+AdamW, and GefenMuon+Gefen. Fused deterministic coverage on two homogeneous RTX 3090 Ti GPUs included DP2, TP2, PP2, CP2 with Transformer Engine, and EP2, with replica or tied-weight hashes appropriate to each topology; TP2 and PP2 also covered legacy optimizer checkpoint continuation for all three recipes. Unfused four-rank coverage included EP2×DP2, TP2×DP2, PP2×DP2, and TP2×PP2 for all three recipes, plus EP2×ETP2 and EP2 checkpoint continuation for GefenMuon+Gefen. This scope does not include Megatron's distributed optimizer, FSDP, optimizer CPU offload, fp16, or non-legacy optimizer checkpoint formats.

## DeepSpeed ZeRO integration scope

Plain Gefen runs as the DeepSpeed ZeRO client optimizer at stages 1-3. **Optimizer CPU-offload was verified at ZeRO-2 and ZeRO-3, and parameter CPU-offload at ZeRO-3** (`offload_param` is a ZeRO-3 facility). Under offload, DeepSpeed hands the client optimizer CPU-resident fp32 flat partitions and calls its step on them; Gefen's decomposed path steps CPU tensors directly (the fused CUDA kernels are bypassed per tensor when the partition is on CPU, so `fused=True` is safe). Raw `deepspeed.initialize` at ZeRO-2 and ZeRO-3 with `offload_optimizer` (and, at stage 3, `offload_param`) to CPU was verified to keep Gefen as the client optimizer, step CPU-resident partitions, and produce a loss curve matching the same run without offload. End-to-end axolotl training (`optimizer: gefenx`) was verified under ZeRO-2 and ZeRO-3 CPU-offload for both full fine-tuning and LoRA. This requires `"zero_allow_untested_optimizer": true` and, whenever any CPU-offload is enabled, `"zero_force_ds_cpu_optimizer": false`, with the config's own `optimizer` section left unset.

The Muon family (`GefenMuon`, `GefenMuonHybrid` / `optimizer: gefenx_muon`) cannot run under DeepSpeed ZeRO at any stage: ZeRO steps flattened 1-D partitions on which Newton-Schulz orthogonalization cannot apply. Those optimizers fail fast with an explanatory error rather than training incorrectly; use plain Gefen under ZeRO, or FSDP2/DDP for the Muon family.

## FSDP2 CPU offload

Plain Gefen supports FSDP2 training-time CPU offload via `CPUOffloadPolicy` (`fully_shard(module, offload_policy=CPUOffloadPolicy())`), which keeps each sharded parameter and its gradient on CPU and moves them to GPU only for the forward/backward compute. The optimizer then steps the CPU-resident local shards through the decomposed path (the fused CUDA kernels are bypassed per tensor for a CPU shard, so `fused=True` is safe). Verified on a single GPU and on two GPUs. Each rank steps its own CPU-resident local shard and learns its codebook rank-locally -- plain Gefen runs no cross-rank codebook collective -- and the two-GPU run completes on an NCCL-only process group as well as a mixed `cuda:nccl,cpu:gloo` group. This is separate from the FSDP2 *checkpoint* `get_optimizer_state_dict(..., cpu_offload=True)` path in the next section.

## Optimizer checkpoint scope

Native single-process optimizer `state_dict()`/`load_state_dict()` and the explicitly documented GefenMuon `distributed` state path remain separate from the rank-local DCP adapter. For plain Gefen and `GefenMuon(sharded_mode="approx")`, each rank can learn different local codebook and block geometry, so `state_dict()` collectively replaces ordinary rank-local tensors with one tagged, rank-indexed CPU payload that PyTorch full-state DCP preserves. Both optimizers' two-GPU FSDP2 `fully_shard` paths are exercised through `get_optimizer_state_dict(..., full_state_dict=True, cpu_offload=True)` and `set_optimizer_state_dict(..., full_state_dict=True, broadcast_from_rank0=True)`, with exact next-step continuation; the flattened optimizer-state form is also covered.

The native format is deliberately same-topology only and currently requires one 1-D DeviceMesh spanning the default process-group world. It is bit-exact on resume and is the recommended path when the world size does not change. Plain Gefen additionally exposes `GefenDCPState`, a standalone `torch.distributed.checkpoint` Stateful wrapper whose purpose is **resharding**: a checkpoint saved on N ranks can be loaded on M ranks for one-dimensional default-world `Shard(0)` DTensors. Save dequantizes the quantized momentum against each rank's learned codebook into dense `Shard(0)` DTensors, so DCP can reshard them without gathering the full optimizer state onto one rank. Load validates every field before mutating the live optimizer, then re-blocks the resharded dense momentum back into Gefen's compact per-block state on each new local shard — re-running the block-variance period search, relearning the exact codebook, and re-quantizing — so the restored optimizer keeps Gefen's ~1 byte/param footprint rather than expanding to per-element state.

Resume through `GefenDCPState` is a **correct, finite continuation, not a bit-exact restore** — even at the same world size. It approximates in two independent ways, and only the first is bounded by quantization noise:

- **Momentum re-quantization.** The momentum is routed through a dense reshard and re-quantized against a codebook relearned on the new shard, so it returns within 256-level quantization noise (measured ~2e-4 relative on the covered shards).
- **Second-moment re-aggregation.** Gefen never stores a per-element second moment: it keeps one `vmean` per block, and that `vmean` is an EMA of the block's mean squared gradient. Save can therefore only write each source block's `vmean` repeated across its elements, and load re-aggregates those values onto the *target* blocks. Because both the EMA and the block mean are linear, a target block that is an exact union of whole source blocks reproduces what a natively blocked run at the target period would hold, to fp32 round-off (measured ~3e-7) — **coarsening onto source boundaries costs nothing**. The lossy direction is the opposite one: when a target block falls strictly inside a source block, or straddles a source boundary, the single stored `vmean` can only be spread flat across sub-blocks that a native run would have given distinct values. That sub-block detail was never in the checkpoint, so it cannot be recovered (a 4x-finer blocking measured ~16% relative on a synthetic shard with realistic within-block variation). Its size is set by how much the true second moment varies *inside* a source block, which is data-dependent and can be far larger than the momentum quantization noise.

Block geometry is re-derived per shard, so a world-size change that moves a shard's chosen period off the source's block boundaries is what triggers the second effect; a reshard that keeps the period, or that only merges whole blocks, is subject only to the first. Same-topology resumes should use the bit-exact native `state_dict` path above.

`GefenDCPState` requires `factored_v_2d=False` and `capturable=False` (the factored row/column second moment is not shard-addressable, and capturable/compiled counter and seed semantics are not host-serializable); Muon, Hybrid, `sharded_mode`, multidimensional meshes, subgroups, and non-`Shard(0)` placements fail closed. The native full-state format is same-topology with one carve-out: a `GefenMuon(sharded_mode="distributed")` checkpoint consolidates every owner's complete state onto every rank, so it resumes under a different world size — including into a single-process optimizer that never initializes a process group. That portability is replication, not resharding: each rank materializes the full optimizer state instead of addressing its own shard, so it does not buy what `GefenDCPState` buys.

Pass `GefenSavePlanner` to `dcp.save` so the save stays memory-bounded. Resharding forces the dense expansion above — 4 bytes/param of momentum plus 4 bytes/param of second moment, eight times what the optimizer actually holds — and without the planner every slot's dense pair is expanded up front and kept live for the whole write, peaking higher than a training step. The planner expands one slot at a time, at the moment the writer asks for it, so the save peaks at the resident state plus a single slot's dense form:

```python
from gefen import Gefen, GefenDCPState, GefenSavePlanner
import torch.distributed.checkpoint as dcp

dcp.save(
    {"model": model, "optimizer": GefenDCPState(optimizer)},
    storage_writer=dcp.FileSystemWriter(path),
    planner=GefenSavePlanner(),          # required for the memory bound
)
dcp.load(                                 # load needs no planner
    {"model": model, "optimizer": GefenDCPState(optimizer)},
    storage_reader=dcp.FileSystemReader(path),
)
```

On a 64M-param local shard the peak over the optimizer's resident state falls from 12.0 to 6.0 bytes/param for one large tensor, and from 8.25 to 0.50 bytes/param for a 32-tensor model; the checkpoint bytes are unchanged. The bound is per *slot*, so the win grows as the state is split across more parameters, and the remaining transient is a single slot's dense form plus up to ~128 MiB of dequantize scratch. That scratch is a ceiling, not a fixed cost: it is bounded by the dequantize gather chunk (`GEFEN_DEQUANT_GATHER_CHUNK`, 8M elements), so only slots larger than the chunk pay the full ~128 MiB, and a smaller slot pays proportionally less — ~24 MiB per 2M-param slot in the 32-tensor case above, which is what makes 0.50 bytes/param reachable there. Omitting the planner is not a correctness bug — the save still writes exactly the same checkpoint — it only costs the old 8 bytes/param. `GefenSavePlanner` subclasses `DefaultSavePlanner` and defers to it for everything that is not Gefen optimizer state, so the model and any other Stateful items in the same state dict are unaffected.

`dcp.async_save` writes a `GefenDCPState` when given `storage_writer=GefenFileSystemWriter(path)`:

```python
from gefen import GefenFileSystemWriter

future = dcp.async_save(
    {"model": model, "optimizer": GefenDCPState(optimizer)},
    storage_writer=GefenFileSystemWriter(path),
    planner=GefenSavePlanner(),
)
# ... training continues ...
future.result()
```

Async save returns once the state dict has been *staged* — copied to CPU — and writes it from a background thread, which is what lets training resume against a checkpoint that still reflects the moment of the call. `GefenFileSystemWriter` stages Gefen's slots one at a time: it expands a slot's dense form, copies it to CPU, and releases it before expanding the next, so the device holds one slot's dense form during staging rather than every slot's at once. On the 32-tensor model above that is a **0.50 bytes/param** GPU peak over the optimizer's resident state — the same bound `GefenSavePlanner` gives the synchronous save. CPU RAM holds the full dense snapshot for the duration of the write, which is what async staging costs for any optimizer. Everything that is not Gefen optimizer state is staged by `FileSystemWriter` unchanged.

The writer is required because staging has to copy the state dict, and `GefenDCPState` hands DCP zero-storage stand-ins that have nothing to copy. Given the default writer, `async_save` raises a `RuntimeError` and writes nothing, on every supported torch version and on both CPU- and CUDA-resident state. The async checkpoint's file *layout* differs from the synchronous save's — staging sets the writer's `per_thread_copy_ahead` to 0, which orders the write items by size — and it restores identically.

## Transformers Trainer DDP

The `benchmarks.trainer_resume` gate exercises plain Gefen, GefenMuon+AdamW, and GefenMuon+Gefen through Trainer's internal Accelerate wrapper with tied weights, gradient accumulation, a changing scheduler, native Trainer checkpoint files, BF16 fused updates, and two-rank DDP replica hashes. All three recipes have passed its deterministic fused-BF16 two-rank configuration on homogeneous GPUs, which requires exact model, optimizer, scheduler, LR, and logged-loss agreement between uninterrupted and resumed runs. Run the gate with:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:src torchrun --standalone --nproc-per-node=2 \
  -m benchmarks.trainer_resume.run --output-dir benchmarks/trainer_resume/out/ddp \
  --device cuda --dtype bfloat16 --fused --deterministic --steps 3 --split-step 1 \
  --gradient-accumulation-steps 2
```

This covers Trainer's DDP frontend and internal Accelerate optimizer wrapper, not a standalone `Accelerator` loop, FSDP, or `model_init`.

## CUDA Graphs & torch.compile (`capturable`)

`Gefen`, `GefenMuon`, and `GefenMuonHybrid` accept `capturable=True` (same meaning as `torch.optim`'s argument): step counters, bias corrections, the optimizer-global checkpoint counter, and a tensor `lr` live on the GPU, so `opt.step()` stays correct inside a replayed CUDA graph or compiled regions. The default `capturable=False` preserves the established eager behavior, and capturing with it raises instead of silently freezing the step counters. Parity is bitwise for fixed-order components and tolerance-bounded where fused atomic reductions can vary at the ULP scale.

Measured step times (386M-parameter census, RTX 3090 Ti, tail-100 mean over 500 CUDA-event-timed steps):

| `opt.step()` | Gefen | GefenMuonHybrid |
|---|---|---|
| default eager (`capturable=False`) | 22.7 ms | 164.4 ms |
| eager, `capturable=True` | 22.8 ms | 173.0 ms |
| manual `torch.cuda.CUDAGraph` replay | 22.9 ms | 168.9 ms |
| `torch.compile(mode="reduce-overhead")` | 23.0 ms | **147.6 ms** |

These measurements are path- and shape-specific, not a zero-overhead guarantee: `capturable=True` and manual replay are nearly flat for plain Gefen but add modest overhead to the measured hybrid, while compilation improves the measured hybrid and is slightly slower for plain Gefen.

**Manual capture** — run a few real warmup steps first (codebook learning happens on the first step), then capture one `opt.step()` and drive training by refilling the static grad buffers (and the `lr` tensor, for schedules) before each `graph.replay()`:

```python
from gefen import GefenMuonHybrid

opt = GefenMuonHybrid(model, lr=torch.tensor(3e-5, device="cuda"), capturable=True)
```

**torch.compile** — the fused kernels are registered as torch custom ops, so they remain inside compiled regions. The step also contains deliberate `@torch._dynamo.disable` host helpers for static-address registration, batched scalar refresh, and global-step bookkeeping, so Dynamo may partition one optimizer call into multiple compiled regions rather than one end-to-end graph:

```python
compiled_step = torch.compile(opt.step, mode="reduce-overhead", dynamic=False)
opt.step(); opt.step()                      # first steps EAGER: codebook learning + static-address marking
# then, per iteration:
torch.compiler.cudagraph_mark_step_begin()
compiled_step()
```

Caveats:

- `dynamic=False` is required — dynamic shapes trip a dynamo symbolic-shapes bug (torch 2.12) on the per-param optimizer state.
- `capturable=True` rejects host-driven periodic codebook refresh and gradient-histogram output.
- Every parameter in a captured optimizer must live on the current CUDA capture device; multi-process distributed training should construct one optimizer per rank/device.
- A float `lr` bakes into the graph (as in `torch.optim`); pass a tensor `lr` and update it in place to drive an LR schedule.
- Checkpoints are portable across the `capturable` toggle in both directions; saving after graph replay synchronizes the true replayed global step so stochastic-rounding resumes continue from the correct seed.
- FSDP2 / DTensor row-shard capture is validated by `tests/test_capturable_fsdp2.py` on 2 ranks for plain Gefen plus GefenMuon `sharded_mode="approx"`, `"exact"`, and `"distributed"` (including empty-shard coverage); ranks must capture and replay in lockstep.
- Capturing the sharded `"exact"`/`"distributed"` modes records NCCL collectives (the in-`step()` all-gather) into the graph, which requires an NCCL/PyTorch build that supports collective graph capture — validated on PyTorch 2.12 / CUDA 13.3; on older stacks capture may raise (use `"approx"` or plain Gefen, which take no collectives, or `capturable=False`).
