# Sharding sweep: GefenMuon exact vs approx sharded Newton-Schulz (FSDP2)

A self-contained, reproducible benchmark that measures what GefenMuon's
`sharded_mode` does to convergence and speed under **FSDP2** (`fully_shard` /
DTensor) as the **shard count (world size) grows**. Anyone can clone this fork,
point the launcher at a model and some GPUs, and reproduce the
cells -> `results_fsdp2.jsonl` -> two charts pipeline with one command.

This is the multi-GPU sibling of `../optimizer-sweep` (single-GPU, optimizer
comparison). It reuses the same config/precedence/GPU-pinning patterns.

## 1. What it benchmarks (and the regime)

`GefenMuonHybrid` runs the Newton-Schulz (NS) orthogonalization of each hidden
matrix's gradient. Under FSDP2 every parameter is row-sharded across ranks, so
the optimizer can compute NS two ways:

- **`exact`** — every rank all-gathers the full gradient matrix, runs NS on the
  full matrix (**bit-for-bit single-GPU parity**), then slices its row-shard of
  the update back. Correct, but pays an all-gather + full-size NS per matrix.
- **`approx`** — each rank runs the whole pipeline on its **local row-shard
  only** (no all-gather, NS on a smaller matrix). **Faster, but non-parity** —
  the orthogonalization of a row-slice is not the slice of the full
  orthogonalization, and the error grows as the matrix is cut into more shards.

For each `(world_size, sharded_mode)` cell it records, under one fixed regime:

- **Final eval loss** on a held-out set, plus the full eval-loss curve
- **Avg training-step time** (s/step, incl. the periodic eval)
- **Peak VRAM** (`torch.cuda.max_memory_allocated`)

**Regime — identical across every cell (mirrors `optimizer-sweep`):**

- Full fine-tune, **bf16 master weights**, `gradient_checkpointing` ON,
  **micro-batch 1**, sequence length **2048**, constant LR, fixed seed
- Dataset **`tatsu-lab/alpaca`**, Alpaca prompt template (prompt tokens masked),
  greedily **packed into exact 2048-token blocks**
- **Data is REPLICATED across ranks** — every rank sees the identical packed
  block each step. So `exact` mode reproduces the single-GPU trajectory exactly,
  and any divergence in `approx` is **purely the NS approximation**, not a
  data-parallel batch-size change. This isolates the optimizer effect.

The only knob under test is `--sharded-mode {exact, approx}`; the only axis swept
is the FSDP2 world size (`--world-sizes`, default `2 4`).

## 2. Setup

Depends on the Gefen **fork** (the one this directory lives in). Install from
source on this fork (not PyPI), in an env whose `torch` has FSDP2
(`torch.distributed.fsdp.fully_shard`, i.e. torch >= 2.5):

```bash
git clone https://github.com/thad0ctor/Gefen-X     # this fork
cd Gefen-X
pip install -e .                                    # builds fused CUDA kernels on first run
pip install matplotlib datasets transformers
```

A CUDA toolkit + host compiler compatible with your PyTorch is required (Gefen
JIT-builds its fused kernels on the first CUDA run; set `cuda_home:` /
`--cuda-home` if `CUDA_HOME` isn't already in your environment).

## 3. Usage

Drive everything through `run.sh`. For each `world_size x mode` it launches
`fsdp2_convergence.py` under `torchrun --nproc_per_node=<world>`, pinned to the
**first `<world>` GPUs** of the pool, appends the cell's RESULT json to
`results_fsdp2.jsonl`, saves a per-cell log, then renders the two charts.

Settings come from CLI flags and/or a YAML config, with

```text
precedence:  CLI flag  >  config.yaml  >  built-in default
```

### Config file (recommended)

```bash
cd benchmarks/sharding-sweep
cp config.example.yaml config.yaml
$EDITOR config.yaml          # set venv, out, gpus, model
./run.sh                     # uses config.yaml for everything
./run.sh --steps 50 --world-sizes "2"   # CLI still overrides individual keys
```

`config.yaml` is gitignored, so your local venv/model paths never get committed.
Use `./run.sh --dry-run` to resolve and print the effective settings without
launching anything.

### GPU selection (PCI-ordered)

GPU indices are **PCI-bus-ordered** — `run.sh` exports
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the indices in `gpus:` / `--gpus` match
`nvidia-smi` and the physical cards torchrun pins. List them with
`./run.sh --list-gpus`. The pool must hold at least `max(world_sizes)` GPUs;
each cell pins the first `<world>` of them. Use one GPU type so step-time
comparisons across cells are apples-to-apples.

### Pure CLI

```bash
./run.sh \
  --venv  /path/to/your/venv/bin/python \
  --out   ./out \
  --gpus  "1,4,5,7" \
  --model /path/to/Qwen3-0.6B \
  --world-sizes "2 4" \
  --modes "exact approx" \
  --steps 2000 --arch 8.6
```

You can also run a single cell directly under torchrun:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4,5 TORCH_CUDA_ARCH_LIST=8.6 \
/path/to/venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
  fsdp2_convergence.py --model /path/to/Qwen3-0.6B --lr 5e-5 --steps 2000 \
  --sharded-mode approx --tag qwen0p6b --out ./out/results_fsdp2.jsonl
```

## 4. Outputs

Written to `--out`:

- `results_fsdp2.jsonl` — one RESULT json line per cell (world_size,
  sharded_mode, final_eval, peak_vram_gib, ...)
- `logs/<tag>_w<world>_<mode>.log` — per-cell stdout (the eval-loss curves and
  avg s/step are parsed from these)
- `muon_shard_loss.png` — held-out eval loss vs step, exact (solid) vs approx
  (dashed), colored by world size, each run's final eval annotated
- `muon_shard_perf.png` — avg step-time + peak VRAM bars per world size, with the
  approx speedup and loss-penalty annotated

Re-render the charts from existing data without re-running the sweep:

```bash
python plot_sharding.py --results ./out/results_fsdp2.jsonl \
  --logs ./out/logs --out-dir ./out
```

`plot_sharding.py` is data-driven (any set of world sizes / both naming schemes
for the logs) and matches each cell to its log by the RESULT json embedded in
the log, falling back to the log filename.

## 5. Caveats (honest)

- **`approx` is non-parity.** It is an approximation of the Newton-Schulz
  orthogonalization, and the **loss penalty grows with the shard count** — on
  Qwen3-0.6B (this regime) the measured final-eval gap vs `exact` was
  **+0.036 at 2 shards** and **+0.076 at 4 shards**. `exact` is the parity
  reference; reach for `approx` only when the step-time win is worth that drift,
  and re-measure the penalty at your shard count.
- **Single seed** by default (`--seed 0`). Re-run with other seeds to bound the
  noise on the eval-loss gap.
- **Data is replicated, not data-parallel.** This is deliberate, to isolate the
  optimizer; it is **not** a throughput/scaling benchmark (a real run would
  shard the data). The reported s/step is the optimizer-cost signal, not a
  tokens/sec scaling number.
- **Step-time is per GPU type.** Compare cells run on the same GPU type only.
- `exact`'s all-gather makes its step-time roughly flat in shard count while its
  per-step VRAM stays near the full-matrix NS cost; `approx`'s shrinks with the
  shard. The speedup therefore depends on model + shard count.

## 6. Using Gefen in training

To train with Gefen in [Axolotl](https://github.com/axolotl-ai-cloud/axolotl),
see the integration PR:
**https://github.com/axolotl-ai-cloud/axolotl/pull/3755** (draft). Note:
`GefenMuonHybrid` (this benchmark's optimizer) is **not yet exposed** through
that Axolotl integration — it is benchmarked here directly against the API.
