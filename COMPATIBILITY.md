# Architecture Compatibility

Validation of the Gefen optimizer (`fused=True`, `factored_v_2d=True` — the defaults) against the modern architectures requested in [issue #15](https://github.com/thad0ctor/Gefen-X/issues/15). Every entry is a real full-parameter fine-tune smoke test on RTX 3090 (24 GB) GPUs: a fixed small batch is trained for 40–300 steps and must show a sharp, finite loss drop (memorization).

## Matrix

| Model | Kind | Trained params | Result | Loss (first → last) | Peak VRAM | Hardware |
|---|---|---|---|---|---|---|
| Qwen3-VL-2B-Instruct | VLM | 2.1B | ✅ PASS | 24.04 → 0.036 (−99.9%) | 11.2 GiB | 1×24 GB |
| MiniCPM-V-4.6 | VLM | 1.3B | ✅ PASS | 18.46 → 0.052 (−99.7%) | 7.2 GiB | 1×24 GB |
| Ministral-3-3B-Instruct-2512 | VLM (Mistral 3) | 3.8B | ✅ PASS | 5.940 → 0.022 (−99.6%) | 19.9 GiB | 1×24 GB |
| gemma-4-E2B-it | VLM (audio+vision) | 4.8B | ✅ PASS | 18.40 → 0.021 (−99.9%) | 24.5 GiB total | 2×24 GB |
| Stable Diffusion XL (UNet) | image diffusion | 2.6B | ✅ PASS | 0.658 → 0.0007 (−99.9%, 300 steps) | 12.7 GiB | 1×24 GB |
| FLUX.2-klein-4B (transformer) | image flow | 3.9B | ✅ PASS | 2.433 → 0.013 (−99.5%) | 18.2 GiB | 1×24 GB |
| ACE-Step 1.5 turbo (DiT) | music flow | 1.6B of 2.4B | ✅ PASS | 2.828 → 0.171 (−94.0%, 100 steps) | 8.9 GiB | 1×24 GB |
| FLUX.1-dev (transformer) | image flow | 11.9B | ✅ PASS | 3.018 → 0.003 (−99.9%, 100 steps) | 16.3 GiB per GPU | 4×24 GB FSDP2 |

Parameter ranks exercised: 1-D through 5-D (norms/biases, linears, Conv1d/patch embeds, Conv2d in the SDXL UNet and vision towers, the 5-D Qwen3-VL patch embed). All route through Gefen's flattened block partitioning without special-casing.

## Method

- **Test**: fixed batch (synthetic image + caption for VLMs, fixed latents/noise/timestep for diffusion/flow models), N optimizer steps, PASS iff the loss falls by more than 30% with no NaN/Inf. Peak VRAM from `torch.cuda.max_memory_allocated`.
- **Trainable set**: one probe backward pass discovers the params that actually receive gradients (modality towers with no input are frozen); everything else full-parameter trains in bf16 with gradient checkpointing.
- **Conditioning for diffusion/flow models**: text encoders run once for prompt embeddings, then are freed; FLUX.1-dev uses fixed random embeddings of the correct shape (architecture/optimizer validation, not text fidelity).
- **FLUX.1-dev** shards with torch FSDP2 (`fully_shard` per block); Gefen steps the DTensor params directly — no unsharding, 12B full-parameter training in 16.3 GiB per 24 GB GPU.
- **Versions**: torch 2.12.0+cu133, transformers 5.10.2, diffusers 0.39.0.
- ACE-Step 1.5 loads the `acestep-v15-turbo` DiT via its HF remote code (needs `vector_quantize_pytorch`); its VQ tokenizer/detokenizer receive no gradients under the flow-matching loss, so 1.6B of 2.4B params train — same set its own trainer targets.
- GefenMuonHybrid spot-check: whole-model hybrid (`split_params_for_muon` + `GefenMuonHybrid`) passes on Qwen3-VL-2B — 24.04 → 0.188 (−99.2%) at the same 11.2 GiB peak, with the 5-D patch embed routed to the backup optimizer.


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
