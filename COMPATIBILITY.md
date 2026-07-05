# Architecture Compatibility

Validation of the Gefen optimizer (`fused=True`, `factored_v_2d=True` — the defaults) against the modern architectures requested in [issue #15](https://github.com/thad0ctor/Gefen-X/issues/15). Every entry is a real full-parameter fine-tune smoke test on RTX 3090 (24 GB) GPUs: a fixed small batch is trained for 40–300 steps and must show a sharp, finite loss drop (memorization).

## Matrix

| Model | Kind | Trained params | Result | Loss (first → last) | Peak VRAM | Hardware |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | VLM | 2.1B | ✅ PASS | 24.04 → 0.036 (−99.9%) | 11.2 GiB | 1×24 GB |
| MiniCPM-V-4.6 | VLM | 1.3B | ✅ PASS | 18.46 → 0.052 (−99.7%) | 7.2 GiB | 1×24 GB |
| Ministral-3-3B-Instruct-2512 | VLM (Mistral 3) | 3.8B | ✅ PASS | 5.940 → 0.022 (−99.6%) | 19.9 GiB | 1×24 GB |
| gemma-4-E2B-it | VLM (audio+vision) | 4.8B | ✅ PASS* | 18.40 → 0.021 (−99.9%) | 24.5 GiB total | 2×24 GB |
| Stable Diffusion XL (UNet) | image diffusion | 2.6B | ✅ PASS | 0.658 → 0.0007 (−99.9%, 300 steps) | 12.7 GiB | 1×24 GB |
| FLUX.2-klein-4B (transformer) | image flow | 3.9B | ✅ PASS | 2.433 → 0.013 (−99.5%) | 18.2 GiB | 1×24 GB |
| ACE-Step 1.5 turbo (DiT) | music flow | 1.6B of 2.4B | ✅ PASS | 2.828 → 0.171 (−94.0%, 100 steps) | 8.9 GiB | 1×24 GB |
| FLUX.1-dev (transformer) | image flow | 11.9B | ✅ PASS | 3.018 → 0.003 (−99.9%, 100 steps) | 16.3 GiB per GPU | 4×24 GB FSDP2 |

\* Requires the fixes on `fix/period-variance-grid-overflow` (see below) and does not fit a single 24 GB card: weights (10.2 GiB) + grads (9.6 GiB) + optimizer state put the peak near 25 GiB. It passes sharded across two GPUs with `device_map="sequential"` and `max_memory={0: "6GiB", 1: "22GiB"}` (the 2.35B-element per-layer embedding cannot be split, so the sequential split keeps it and its gradient on the first GPU).

Parameter ranks exercised: 1-D through 5-D (norms/biases, linears, Conv1d/patch embeds, Conv2d in the SDXL UNet and vision towers, the 5-D Qwen3-VL patch embed). All route through Gefen's flattened block partitioning without special-casing.

## Bugs found and fixed during this validation

Gemma 4's per-layer embedding table (`embed_tokens_per_layer`, 262144×8960 = 2.35e9 elements) is the first tensor past 2^31 elements Gefen-X met, and HF `device_map` sharding was previously untested. Three fixes, all with regression tests (`tests/test_large_tensor_paths.py`, `tests/test_codebook_multi_device.py`), branch `fix/period-variance-grid-overflow`:

- **Period-search kernel crashed on >2G-element tensors** — the CUDA grid was sized `numel/period`, which exceeds the 2^31−1 grid limit for small candidate periods. Now grid-strided; launches are unchanged for every tensor under 2^31 elements, and small-size results are bit-identical.
- **Exact-codebook histogram OOM'd on >2G-element tensors** — it upcast the whole flattened gradient to fp32 at once (8.75 GiB for the Gemma 4 embedding). Now chunked along block boundaries (`GEFEN_EXACT_HISTOGRAM_CHUNK`), which is bit-exact because per-block histogram counts are independent.
- **Multi-GPU `device_map` sharding failed on every fused step** — the learned codebook lived on one device and any param on another GPU raised `Expected all tensors on the same device as p.` Steps now use a cached per-device codebook copy; a second-device update is bit-identical to the single-device oracle.

## Method

- **Test**: fixed batch (synthetic image + caption for VLMs, fixed latents/noise/timestep for diffusion/flow models), N optimizer steps, PASS iff the loss falls by more than 30% with no NaN/Inf. Peak VRAM from `torch.cuda.max_memory_allocated`.
- **Trainable set**: one probe backward pass discovers the params that actually receive gradients (modality towers with no input are frozen); everything else full-parameter trains in bf16 with gradient checkpointing.
- **Conditioning for diffusion/flow models**: text encoders run once for prompt embeddings, then are freed; FLUX.1-dev uses fixed random embeddings of the correct shape (architecture/optimizer validation, not text fidelity).
- **FLUX.1-dev** shards with torch FSDP2 (`fully_shard` per block); Gefen steps the DTensor params directly — no unsharding, 12B full-parameter training in 16.3 GiB per 24 GB GPU.
- **Versions**: torch 2.12.0+cu133, transformers 5.10.2, diffusers 0.39.0.
- ACE-Step 1.5 loads the `acestep-v15-turbo` DiT via its HF remote code (needs `vector_quantize_pytorch`); its VQ tokenizer/detokenizer receive no gradients under the flow-matching loss, so 1.6B of 2.4B params train — same set its own trainer targets.
- GefenMuonHybrid spot-check: whole-model hybrid (`split_params_for_muon` + `GefenMuonHybrid`) passes on Qwen3-VL-2B — 24.04 → 0.188 (−99.2%) at the same 11.2 GiB peak, with the 5-D patch embed routed to the backup optimizer.

