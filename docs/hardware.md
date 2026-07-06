# Hardware Requirements

**How much GPU memory do you need to fine-tune a model of a given size?** Find your model size on the left, your method across the top. Gefen stores optimizer state in quantized codebooks — about **1 byte per parameter vs AdamW's ≈6** — so a **full fine-tune** costs roughly **half** of AdamW and fits **≈2× the model** on the same card.

Numbers are **measured peak VRAM** in GiB (`torch.cuda.max_memory_allocated`): bf16, batch 1, gradient checkpointing **on**, **no** CPU/disk offload (Gefen defaults; AdamW with bf16 state). That keeps activations tiny, so each cell is the *floor* — weights + gradients + optimizer state; add a few GiB for activations at longer sequences. Measured on Qwen (dense 0.6B–72B, Qwen3 30/35B MoE) on 96 GB cards. For which architectures train cleanly, see [COMPATIBILITY.md](../COMPATIBILITY.md).

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
| 72B | 339 ᵉ | 677 ᵉ | 138 ᵈ | 45.5 |

**Bold** = fits a 24 GB card (3090 / 4090). Markers: **ᶠ** measured under FSDP2 on 2×96 GB (overflows one card); **ᵈ** LoRA base sharded across 2×96 GB; **ᵉ** estimated (overflows a 96 GB card, or MoE QLoRA — see [Estimating any size](#estimating-any-size)). Everything else is a single-GPU measurement.

**The gist:** on a 24 GB card Gefen full-fine-tunes **4B** where AdamW manages only ≈2B; on a 96 GB card, **≈19B vs ≈10B**. Adapters reach far higher — a **72B** model QLoRA-fits a single 96 GB card (45.5 GiB).

## What fits on your GPU

Largest model you can train each way (memory floor — leave headroom for activations at your real batch and sequence length):

| GPU | Full FT · **Gefen** | Full FT · AdamW | LoRA | QLoRA |
|---|---|---|---|---|
| 24 GB — RTX 3090 / 4090 | **≈4B** | ≈2B | ≈11B | ≈24B |
| 48 GB — A6000 / L40S | **≈9B** | ≈4B | ≈23B | ≈50B |
| 80 GB — A100 / H100 | **≈16B** | ≈8B | ≈40B | ≈85B |
| 96 GB — RTX PRO 6000 | **≈19B** | ≈10B | ≈48B | ≈100B |
| 2 × 96 GB — FSDP2 | **≈38B** | ≈18B | ≈95B | ≈200B |

LoRA/QLoRA are optimizer-agnostic (same reach under either optimizer); the full-FT columns are where Gefen roughly doubles the reach.

## Bigger full fine-tunes: FSDP2

When a full fine-tune overflows one card, FSDP2 shards weights + gradients + optimizer state across GPUs (Gefen supports it) and per-GPU memory drops ≈linearly. Across two 96 GB cards:

| Model | Gefen — total / per-GPU | AdamW — total / per-GPU | Runs on 2×96 GB |
|---|---|---|---|
| 14B | **79 / 40** | **139 / 70** | Gefen ✅ · AdamW ✅ |
| 27B | 121 / 61 ᵉ | 241 / 121 ᵉ | Gefen ✅ · AdamW ❌ |
| 30B-A3B | 142 / 71 ᵉ | 284 / 142 ᵉ | Gefen ✅ · AdamW ❌ |
| 35B-A3B | 156 / 78 ᵉ | 312 / 156 ᵉ | Gefen ✅ · AdamW ❌ |
| 72B | 339 / 170 ᵉ | 677 / 339 ᵉ | ❌ · ❌ (needs 4+ cards) |

Bold row measured; **ᵉ** estimated from the floor (the 14B row confirms it within ≈15%). **Two 96 GB cards full-fine-tune up to ≈35B with Gefen, where AdamW caps at ≈18B.** Add cards to go further — FSDP2 across N GPUs raises the ceilings ≈N×.

## Estimating any size

Per parameter, the full-fine-tune floor (from the 4B–14B rows) breaks down as:

| Component | AdamW | **Gefen** |
|---|---|---|
| Weights + gradients (bf16) | 4 B/param | 4 B/param |
| Optimizer state + step workspace | ≈ 6 B/param | **≈ 1 B/param** |
| **Full-FT floor** | **≈ 10 B/param** | **≈ 5 B/param** |

```text
full-FT (GiB) ≈ params_B × 5  (Gefen)   or  × 10  (AdamW)
LoRA  ≈ params_B × 2          QLoRA ≈ params_B × 0.9
```

Gefen replaces AdamW's two moment buffers with a quantized codebook (≈1 B/param); the classic fp32-state AdamW recipe is heavier still (≈8 B/param of state alone), so Gefen's edge there is more than 2×. Adapter rates come straight from the table and barely touch the optimizer. **Example — 7B:** Gefen 33 GiB (48 GB card), AdamW 65 GiB (80 GB); LoRA 13, QLoRA 7 (either on a 24 GB card).

## Reproducing

Every cell comes from the harness in [`benchmarks/arch-compat/`](../benchmarks/arch-compat/) (`peak_vram_GiB` per run):

```bash
# single GPU — full FT, LoRA, or QLoRA
python validate_llm.py --model Qwen/Qwen3-4B --optimizer gefen   # or --optimizer adamw / --lora / --qlora

# multi-GPU full fine-tune (FSDP2)
torchrun --nproc_per_node=2 validate_fsdp.py --kind causal-lm --model Qwen/Qwen3-14B --optimizer gefen
```

Versions: torch 2.12 (cu133), transformers 5.13, peft 0.19, bitsandbytes 0.49. Peak allocation is card-independent, so the figures apply on any GPU large enough to hold them.
