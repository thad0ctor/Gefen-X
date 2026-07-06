# Hardware Requirements

How much GPU memory do you need to fine-tune a model of a given size? Find your model's parameter count on the left and read across to your training method. Gefen keeps optimizer state in quantized codebooks — about **1 byte per parameter instead of AdamW's ≈6** — so its **full fine-tune** footprint is roughly **half** of AdamW's, which doubles the model size you can train on the same card.

This page is about *sizing*. For which specific architectures are known to train cleanly and on what hardware, see [COMPATIBILITY.md](../COMPATIBILITY.md).

## Assumptions

Every measured value is **peak VRAM** (`torch.cuda.max_memory_allocated`), in GiB, from a real fine-tune under a fixed workload:

- **Batch:** a short, two-sequence step (a memorization smoke workload).
- **Precision:** bf16 weights and gradients.
- **Gradient checkpointing:** **on** — this is why the numbers are a clean floor; with it off, peak VRAM is far higher and dominated by activations rather than optimizer state.
- **CPU / disk offload:** **off** — all weights, gradients, and optimizer state stay resident on the GPU(s); nothing is offloaded to host RAM or disk.
- **Optimizer:** Gefen with its defaults (`fused=True`, `factored_v_2d=True`) for the Gefen columns; `torch.optim.AdamW` (bf16 state) for the AdamW column. LoRA/QLoRA adapters are stepped by Gefen.

Because the batch and sequence length are small and gradient checkpointing is on, **activation memory is negligible here** — the numbers isolate what scales with model size: weights + gradients + optimizer state. Treat them as the memory *floor*; real training adds activation memory on top, which grows with batch size × sequence length (see [Estimating other sizes](#estimating-other-sizes)). Measured on Qwen models (dense 0.6B–72B and Qwen3 30B/35B MoE) on RTX PRO 6000 (96 GB) cards.

## Peak VRAM by model size

| Model | Full FT · **Gefen** | Full FT · AdamW | LoRA | QLoRA (4-bit) |
|---|---|---|---|---|
| 0.6B | **3.4** | **5.6** | **1.3** | **1.0** |
| 1.7B | **9.3** | **16.1** | **3.5** | **2.1** |
| 4B | **20.5** | 37.8 | **7.9** | **3.7** |
| 8B | 38.3 | 76.4 | **15.7** | **8.6** |
| 14B | 69.1 | 139 ᶠ | 28.2 | **13.0** |
| 27B | 121 ᵉ | 241 ᵉ | 51.4 | **22.6** |
| 30B-A3B (MoE) | 142 ᵉ | 284 ᵉ | 57.1 | 27 ᵉ |
| 35B-A3B (MoE) | 156 ᵉ | 312 ᵉ | 64.8 | 30 ᵉ |
| 72B | 339 ᵉ | 677 ᵉ | 138 ᵈ | **45.5** |

Values in GiB. **Bold** = fits a single **24 GB** card (3090 / 4090), with a little headroom left for activations. Plain values fit a larger single card (48 / 80 / 96 GB). Markers: **ᶠ** measured under FSDP2 on 2×96 GB (single-card OOM — see next table); **ᵈ** LoRA base sharded across 2×96 GB (`device_map`); **ᵉ** estimated from the per-parameter floor below (full fine-tune above 14B overflows even a 96 GB card, and MoE QLoRA hits a PEFT expert-upcast limit — use FSDP2 or the estimate). All other cells are single-GPU measurements.

The story is the two full-fine-tune columns against LoRA/QLoRA:

- **Full fine-tune:** Gefen trains **≈2× the model size AdamW can** on the same card — a single 24 GB card full-fine-tunes 4B with Gefen but only ≈2B with AdamW; a single 96 GB card reaches ≈19B versus ≈10B.
- **Adapters go much further on one card:** a **72B** model fine-tunes on a single 96 GB card with **QLoRA (45.5 GiB)**, or on a 24 GB card up to ≈14B (QLoRA 13.0 GiB).

## Full fine-tune under FSDP2 (2 × RTX PRO 6000)

When a full fine-tune overflows one card, FSDP2 shards weights + gradients + optimizer state across GPUs; per-GPU memory falls roughly linearly. Across two 96 GB cards:

| Model | Gefen — total / per-GPU | AdamW — total / per-GPU | Runs on 2×96 GB |
|---|---|---|---|
| 14B | **79 / 40** | **139 / 70** | Gefen ✅ · AdamW ✅ |
| 27B | 121 / 61 ᵉ | 241 / 121 ᵉ | Gefen ✅ · AdamW ❌ |
| 30B-A3B | 142 / 71 ᵉ | 284 / 142 ᵉ | Gefen ✅ · AdamW ❌ |
| 35B-A3B | 156 / 78 ᵉ | 312 / 156 ᵉ | Gefen ✅ · AdamW ❌ |
| 72B | 339 / 170 ᵉ | 677 / 339 ᵉ | ❌ · ❌ (needs 4+ cards) |

Bold rows measured; **ᵉ** estimated from the floor (the measured 14B row confirms it within ≈15%). **Two 96 GB cards full-fine-tune up to ≈35B with Gefen, where AdamW caps at ≈18B** — the same ≈2× advantage, now at the top of the range. Add cards to go further: FSDP2 across N GPUs raises these ceilings roughly N×.

## Estimating other sizes

Full fine-tuning holds three things per parameter, plus activations. Backing the measured floors out per parameter (asymptotic, from the 4B–14B rows):

| Component | AdamW | **Gefen** |
|---|---|---|
| Weights (bf16) | 2 B/param | 2 B/param |
| Gradients (bf16) | 2 B/param | 2 B/param |
| Optimizer state + step workspace | ≈ 6 B/param | **≈ 1 B/param** |
| **Full-FT floor (measured)** | **≈ 10 B/param** | **≈ 5 B/param** |

AdamW keeps two moment buffers plus a transient step workspace; Gefen replaces the moments with a quantized codebook (≈ 1 B/param). The classic mixed-precision AdamW recipe with **fp32** optimizer state is heavier still (≈ 8 B/param of state alone), so on that comparison Gefen's advantage is larger than the ≈2× shown here.

```text
full-FT floor (GiB) ≈ params_in_billions × bytes_per_param × 0.93
                    Gefen ≈ params_B × 5     AdamW ≈ params_B × 10
LoRA floor (GiB)  ≈ params_B × 2      (frozen bf16 base dominates)
QLoRA floor (GiB) ≈ params_B × 0.9    (frozen 4-bit base dominates)
```

The LoRA/QLoRA rates come straight from the measurements (LoRA 51.4 GiB at 27B, 138 at 72B → ≈2 B/param; QLoRA 22.6 at 27B, 45.5 at 72B → ≈0.9 B/param). Adapter methods barely touch the optimizer, so Gefen vs AdamW are within a rounding error there — those columns are what you need whichever optimizer you use.

**Worked example — a 7B model.** Gefen full-FT ≈ 7 × 5 × 0.93 ≈ **33 GiB** (fits a 48 GB card); AdamW ≈ **65 GiB** (needs 80 GB). LoRA ≈ **13 GiB**, QLoRA ≈ **7 GiB** — either fits a 24 GB card.

**Activations** are separate and depend on batch × sequence length, not the optimizer. With gradient checkpointing on they stay modest (a few GiB at batch 1, 512–2048 tokens for these sizes); longer sequences or larger batches add proportionally, so leave headroom above the floor.

## What a card unlocks

The largest model you can train each way, from the floors above (memory floor at a short sequence length — leave headroom for activations at your real batch and sequence length):

| GPU | Full FT · **Gefen** | Full FT · AdamW | LoRA | QLoRA |
|---|---|---|---|---|
| 24 GB — RTX 3090 / 4090 | **≈4B** | ≈2B | ≈11B | ≈24B |
| 48 GB — A6000 / L40S | **≈9B** | ≈4B | ≈23B | ≈50B |
| 80 GB — A100 / H100 | **≈16B** | ≈8B | ≈40B | ≈85B |
| 96 GB — RTX PRO 6000 | **≈19B** | ≈10B | ≈48B | ≈100B |
| 2 × 96 GB — FSDP2 | **≈38B** | ≈18B | ≈95B | ≈200B |

LoRA/QLoRA are optimizer-agnostic (same reach under Gefen or AdamW); the full-fine-tune columns are where Gefen roughly doubles what the card allows. Measured anchors: 72B QLoRA fits one 96 GB card (45.5 GiB); 72B LoRA fits two (138 GiB). FSDP2 across N cards raises the full-FT ceilings roughly N×.

## Reproducing these numbers

Every cell comes from the reusable harness in [`benchmarks/arch-compat/`](../benchmarks/arch-compat/), which records `peak_vram_GiB` per run:

```bash
# single GPU — full FT, LoRA, or QLoRA
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python validate_llm.py \
    --model Qwen/Qwen3-4B --optimizer gefen      # full FT, Gefen  (swap: --optimizer adamw)
    #                     --lora   /  --qlora     # adapter methods

# multi-GPU full fine-tune (FSDP2)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    validate_fsdp.py --kind causal-lm --model Qwen/Qwen3-14B --optimizer gefen
```

Versions: torch 2.12 (cu133), transformers 5.13, peft 0.19, bitsandbytes 0.49. Measured on RTX PRO 6000 (96 GB) cards; peak allocation is card-independent, so the same figures apply on any GPU large enough to hold them.
