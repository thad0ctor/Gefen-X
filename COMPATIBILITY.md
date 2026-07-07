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

Gefen ≈ AdamW on the classifiers, ASR, and TTS, and leads on YOLO11n and RF-DETR; it trails by ~1–7 mAP points on the other detectors and the Conformer at matched LR. That gap is mostly hyperparameter: a per-optimizer LR grid (Gefen's optimum ≈0.6× AdamW's, at 3e-5) shrinks it to ≤0.01 on five of seven detectors — Gefen ahead on RetinaNet and RT-DETR — and ≤0.03 on the rest, e.g. Deformable-DETR's −0.21 at a shared 1e-4 collapses to −0.03. The `Gefen-Muon` hybrid matched or beat AdamW where run.

Setup: from scratch — MNIST (paper recipe, 3 seeds), CIFAR-10 ResNet-18; fine-tuned — classifiers→Imagenette, detectors→COCO128, ASR→Speech Commands, TTS→LJSpeech. RTX 5090 + RTX PRO 6000. Harnesses: [`benchmarks/arch-compat/`](benchmarks/arch-compat/).

## Method

- **Real-dataset rows** (the Vision and Audio tables) report held-out accuracy, COCO mAP, or — where no accuracy metric exists (TTS) — validation loss, after a full training run: a convergence comparison against AdamW at matched model / data / LR / schedule / epochs.
- **Smoke-test rows** (the LLM / VLM / diffusion matrix above) train a fixed batch for N optimizer steps and report the loss as `first → last (Δ%)`, where Δ% is the reduction from the first step's loss to the last (`(first − last) / first`); PASS if it exceeds 30% with no NaN/Inf. Peak VRAM from `torch.cuda.max_memory_allocated`.
- The **Method** column: `full-param` trains every native parameter tensor; `full-param FSDP2` / `device_map` shard that same full-parameter training for models over one card; `LoRA` covers models too large to full-fine-tune even sharded — a weaker claim, since only the adapter matrices are trained.
- **Versions**: LLM / VLM / diffusion smokes on torch 2.12.0+cu133, transformers 5.10.2 (≥ 5.11 for PaddleOCR-VL), diffusers 0.39.0. Vision / audio / real-data runs on torch 2.13.0+cu133, torchvision 0.28, torchaudio 2.11, transformers 5.12, ultralytics 8.4, rfdetr 1.8. Harness and per-model recipes: [`benchmarks/arch-compat/`](benchmarks/arch-compat/).

## CUDA Graphs & torch.compile (`capturable`)

`Gefen`, `GefenMuon`, and `GefenMuonHybrid` accept `capturable=True` (same meaning as `torch.optim`'s argument): step counters, bias corrections, and a tensor `lr` live on the GPU, so `opt.step()` stays correct inside a replayed CUDA graph or a compiled region. The default `capturable=False` is bit-identical to previous behavior, and capturing with it raises instead of silently freezing the step counters.

Measured step times (386M-parameter census, RTX 3090 Ti, tail-100 mean over 500 CUDA-event-timed steps):

| `opt.step()` | Gefen | GefenMuonHybrid |
|---|---|---|
| default eager (`capturable=False`) | 22.7 ms | 164.4 ms |
| eager, `capturable=True` | 22.8 ms | 173.0 ms |
| manual `torch.cuda.CUDAGraph` replay | 22.9 ms | 168.9 ms |
| `torch.compile(mode="reduce-overhead")` | 23.0 ms | **147.6 ms** |

**Manual capture** — run a few real warmup steps first (codebook learning happens on the first step), then capture one `opt.step()` and drive training by refilling the static grad buffers (and the `lr` tensor, for schedules) before each `graph.replay()`:

```python
opt = GefenMuonHybrid(*split_params_for_muon(model), lr=torch.tensor(3e-5, device="cuda"),
                      capturable=True)
```

**torch.compile** — the fused kernels are registered as torch custom ops, so dynamo traces the whole step into one graph and CUDA-graphs it end to end:

```python
compiled_step = torch.compile(opt.step, mode="reduce-overhead", dynamic=False)
opt.step(); opt.step()                      # first steps EAGER: codebook learning + static-address marking
# then, per iteration:
torch.compiler.cudagraph_mark_step_begin()
compiled_step()
```

Caveats:

- `dynamic=False` is required — dynamic shapes trip a dynamo symbolic-shapes bug (torch 2.12) on the per-param optimizer state.
- `capturable=True` rejects the host-driven option `codebook_refresh_every > 0` at construction.
- A float `lr` bakes into the graph (as in `torch.optim`); pass a tensor `lr` and update it in place to drive an LR schedule.
- Checkpoints are portable across the `capturable` toggle in both directions.
- Validated single-process; sharded (FSDP2) capture is not.
