# Architecture compatibility smoke tests

Reusable harnesses that check the Gefen optimizer against a model architecture by
**memorizing a fixed batch**: a small synthetic batch is trained for N optimizer
steps and the loss must fall by more than 30% with no NaN/Inf. This exercises
every trainable parameter tensor of the architecture through Gefen's flattened
block partitioning and fused step — it is an optimizer/architecture check, not a
model-quality or text-fidelity benchmark.

These are the scripts behind the matrix in [`COMPATIBILITY.md`](../../COMPATIBILITY.md).

## Scripts

| Script | Covers |
|---|---|
| `validate_llm.py` | LLMs / VLMs via `transformers` (`AutoModelForCausalLM` / `AutoModelForImageTextToText`) |
| `validate_diffusion.py` | Diffusion / flow denoisers via `diffusers` (UNet or transformer), single GPU |
| `validate_fsdp.py` | Either family sharded with torch FSDP2 (`fully_shard`) for models too big for one card |

## Coverage tiers

Pick the most faithful tier a model fits in. The tier is recorded in each result
as `method` so the matrix can label rows honestly.

1. **`full-param`** — every native parameter tensor is trained, single GPU.
   The default and strongest claim.
2. **`full-param FSDP2 xN` / `device_map`** — same full-parameter training,
   sharded across N GPUs (`validate_fsdp.py`, or `validate_llm.py --device-map auto`).
   Same claim as tier 1, just distributed.
3. **`qlora`** — 4-bit NF4 frozen base + LoRA adapters (`validate_llm.py --qlora`).
   Fallback for models that do not fit full-parameter even sharded. **Weaker
   claim**: Gefen only optimizes the 2-D adapter matrices, not the architecture's
   native tensors (patch embeds, convs, MoE experts). Use it to show an arch
   *loads, runs fwd/bwd, and trains on Gefen*, not that Gefen handles its full
   parameter set. Label these rows distinctly.

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

# QLoRA fallback (weaker claim; see tiers above)
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
