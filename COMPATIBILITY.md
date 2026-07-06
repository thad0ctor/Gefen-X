# Architecture Compatibility

Validation of the Gefen optimizer (`fused=True`, `factored_v_2d=True` — the defaults) across a broad range of model architecture families. Each entry is a real fine-tune smoke test: a fixed small batch is trained for 40–300 steps and must show a sharp, finite loss drop (memorization). GPUs are 24 GB (RTX 3090) unless the Hardware column says otherwise.

> **Scope.** This is not an attempt to validate every model. It deliberately samples a wide range of architecture *families* — dense and Mixture-of-Experts LLMs, vision-language models, image / video / audio diffusion and flow-matching DiTs, and state-space (Mamba-hybrid) models — to show that Gefen's optimizer path handles the parameter tensor shapes, ranks, and dtypes each family produces. Breadth of architectures is the goal; exhaustive coverage of individual checkpoints is not.

## Matrix

| Model | Kind | Method | Result | Loss (first → last) | Peak VRAM | Hardware |
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

## Method

- Fixed batch, N optimizer steps; PASS if the loss falls more than 30% with no NaN/Inf. Peak VRAM from `torch.cuda.max_memory_allocated`.
- The **Method** column: `full-param` trains every native parameter tensor; `full-param FSDP2` / `device_map` shard that same full-parameter training for models over one card; `LoRA` covers models too large to full-fine-tune even sharded — a weaker claim, since only the adapter matrices are trained.
- **Versions**: torch 2.12.0+cu133, transformers 5.10.2 (≥ 5.11 for PaddleOCR-VL), diffusers 0.39.0. Harness and per-model recipes: [`benchmarks/arch-compat/`](benchmarks/arch-compat/).

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
