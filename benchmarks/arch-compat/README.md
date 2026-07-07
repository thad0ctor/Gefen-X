# Architecture compatibility smoke tests

Reusable harnesses that check the Gefen optimizer against a model architecture by
**memorizing a fixed batch**: a small synthetic batch is trained for N optimizer
steps and the loss must fall by more than 30% with no NaN/Inf. This exercises
every trainable parameter tensor of the architecture through Gefen's flattened
block partitioning and fused step — it is an optimizer/architecture check, not a
model-quality or text-fidelity benchmark.

These are the scripts behind the matrix in [`COMPATIBILITY.md`](../../COMPATIBILITY.md).

## Scripts

Smoke tests (memorize a fixed batch):

| Script | Covers |
|---|---|
| `validate_llm.py` | LLMs / VLMs via `transformers` (`AutoModelForCausalLM` / `AutoModelForImageTextToText`) |
| `validate_diffusion.py` | Diffusion / flow denoisers via `diffusers` (UNet or transformer), single GPU |
| `validate_fsdp.py` | Either family sharded with torch FSDP2 (`fully_shard`) for models too big for one card |
| `validate_vision.py` | torchvision classifiers + detectors, transformers DETR-family detectors, ultralytics YOLO, RF-DETR (`--arch all`) |
| `validate_audio.py` | torchaudio (Wav2Vec2, Conformer, Tacotron2) + transformers audio (Dia) speech models (`--arch all`) |

Real-dataset training (held-out accuracy / mAP / val loss vs AdamW — these back the Vision and Audio tables in `COMPATIBILITY.md`):

| Script | Covers |
|---|---|
| `validate_mnist_cnn.py` | official PyTorch MNIST CNN — real training, test accuracy over seeds (`--optimizers adamw,gefen,hybrid`) |
| `validate_cifar10.py` | ResNet-18 (CIFAR stem) on real CIFAR-10, test accuracy |
| `validate_finetune_cls.py` | ImageNet-pretrained torchvision classifiers (ResNet/ConvNeXt/EfficientNet/MobileNet/DenseNet/RegNet/ViT/Swin) fine-tuned on Imagenette, test accuracy (`--arch all`) |
| `train_tv_detectors.py` | COCO-pretrained torchvision detectors (Faster R-CNN, RetinaNet, SSD, FCOS) fine-tuned on COCO128, mAP via torchmetrics (`--arch all`) |
| `train_hf_detectors.py` | pretrained transformers detectors (RT-DETR, YOLOS, Deformable-DETR) fine-tuned on COCO128, mAP (`--arch all`) |
| `train_yolo_coco.py` | ultralytics YOLO11n fine-tuned on COCO128, mAP — Gefen injected via the trainer's `build_optimizer` |
| `train_rfdetr_coco.py` | RF-DETR Nano on COCO128 (COCO-format converter built in), mAP — Gefen injected via the Lightning module's `configure_optimizers`, reusing RF-DETR's layer-wise LR |
| `validate_speechcommands.py` | M5 raw-waveform Conv1d recognizer on real Speech Commands v2, test accuracy |
| `validate_asr_finetune.py` | pretrained Wav2Vec2 / HuBERT / Wav2Vec2-Conformer fine-tuned on Speech Commands, test accuracy (`--arch all`) |
| `validate_tts_finetune.py` | pretrained Tacotron2 fine-tuned on an LJSpeech subset, held-out validation loss |
| `validate_dia_tts.py` | pretrained Dia-1.6B fine-tuned on an LJSpeech subset (DiaProcessor encodes audio to DAC codes), held-out validation loss |

## Methods

Each run records a `method` (matching the value in `COMPATIBILITY.md`):

| `method` | Flag | What trains |
|---|---|---|
| `full-param` | (default) | every native parameter tensor, single GPU |
| `full-param FSDP2` | `validate_fsdp.py` | same, sharded across GPUs via `fully_shard` |
| `device_map` | `--device-map auto` | same, model-parallel across GPUs / CPU offload |
| `lora` | `--lora` | bf16 base frozen + LoRA adapters |
| `qlora` | `--qlora` | 4-bit NF4 base frozen + LoRA adapters |

`lora`/`qlora` are the fallback for models too large to full-fine-tune even
sharded: only the 2-D adapters are trained, so they are a weaker claim than the
full-parameter methods.

## Examples

```bash
# LLM, single GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python validate_llm.py --model <hf-id-or-path> --optimizer gefen --steps 40 --lr 1e-4

# VLM (image+text), single GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python validate_llm.py --model <hf-id> --vl --trust-remote-code --steps 40

# Diffusion / flow denoiser, single GPU
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python validate_diffusion.py --arch sd3 --model <hf-id> --steps 150 --lr 1e-4

# Full-parameter model-parallel across GPUs (large LLM/MoE)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,4,5,7 \
  python validate_llm.py --model <hf-id> --attn eager \
  --device-map auto --max-gpu-mem 22GiB --steps 40

# Full-parameter FSDP2 across GPUs
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,4,5,7 torchrun \
  --nproc_per_node=5 validate_fsdp.py --kind causal-lm --model <hf-id> --steps 40

# QLoRA fallback (weaker claim; see Methods above)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python validate_llm.py --model <hf-id> --qlora --attn eager --steps 60
```

Each run appends one JSON record to `--out` (default `results.jsonl`) with the
architecture, trained-parameter count, parameter tensor ranks exercised, the loss
trajectory, peak VRAM, and PASS/FAIL. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the
visible-device indices match `nvidia-smi`, and `TORCH_CUDA_ARCH_LIST` to your GPU
(e.g. `8.6` for RTX 3090) on the first run so the fused kernels build for it.

`--arch` builders in `validate_diffusion.py` construct fixed **random** text/image
conditioning of the shape each denoiser expects; add a new architecture by writing
one `setup_<arch>` that returns `(denoiser, loss_fn)` and registering it in `ARCHES`.
