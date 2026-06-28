# Optimizer sweep: AdamW vs Gefen

A self-contained, reproducible benchmark that compares the **Gefen** optimizer
against the AdamW family (full-precision, 8-bit, 4-bit) on a small full
fine-tune. Anyone can clone this fork, point the launcher at a model and some
GPUs, and reproduce the LR sweep -> 2000-step finals -> plots -> table pipeline.

## 1. What it benchmarks (and the regime)

For each (model, optimizer) it measures, under one fixed regime:

- **Final eval loss** on a 32-example held-out set, plus the train-loss EMA
- **Throughput** (steady-state tokens/sec, warmup steps excluded)
- **Peak VRAM** — recorded two ways: `peak_vram_gib` =
  `torch.cuda.max_memory_allocated` (real tensor bytes; the apples-to-apples
  optimizer-memory metric and the one the charts use) and
  `peak_vram_reserved_gib` = `torch.cuda.max_memory_reserved` (caching-allocator
  pool incl. slack/fragmentation; closer to what `nvidia-smi` reports)
- **Optimizer-state bytes/param** (real packed bytes, recursing into tensor
  subclasses so torchao's 4-bit packed state isn't over-counted)

**Regime — identical across every cell:**

- Full fine-tune (all parameters trained), **bf16 master weights**
- `gradient_checkpointing` ON, **micro-batch 1**, sequence length **2048**
- Dataset **`tatsu-lab/alpaca`**, tokenized with the model's own tokenizer using
  the standard Alpaca prompt template (prompt tokens masked out of the loss),
  greedily **packed into exact 2048-token blocks**
- **Fixed seed**, identical packed-data order across all optimizers within a seed
- 32-example held-out eval, eval-every 50, constant LR, **2000-step** finals
- Each optimizer runs at **its own fair LR** (see below)

Optimizers (`--opt`): `adamw_bf16`, `adamw8bit` (bitsandbytes), `adamw4bit`
(torchao), `gefen_fused`, `gefen_nonfused`, `gefen_muon` (GefenMuonHybrid). The
default set run by `run.sh` is `adamw_bf16 adamw8bit adamw4bit gefen_fused
gefen_muon` — pass `--no-muon` to drop the (slow) Newton-Schulz `gefen_muon`.

The fair default LRs (175-step fair-sweep optima): AdamW family `5e-5`,
`gefen_fused` / `gefen_nonfused` `2e-5`, `gefen_muon` `5e-5` (with
`--muon-adjust match_rms_adamw`). Pass `--lr-sweep` to re-derive each
optimizer's optimum on your hardware instead of using these defaults.

## 2. Setup

This suite depends on the Gefen **fork** (the one this directory lives in), which
carries kernel/throughput fixes that are **not** in the PyPI `gefen` package.
Install Gefen **from source on this fork**, not from PyPI:

```bash
git clone https://github.com/thad0ctor/Gefen-X     # this fork
cd Gefen-X
pip install -e .                                    # builds fused CUDA kernels on first run
pip install bitsandbytes torchao matplotlib datasets transformers
```

A CUDA toolkit + host compiler compatible with your PyTorch is required (Gefen
JIT-builds its fused kernels on the first CUDA run).

## 3. Usage

Drive everything through `run.sh`. It runs the (optional) LR sweep, then the
finals for every optimizer x model, then the plots and the aggregated table.
Each (model, optimizer) job is pinned to one GPU round-robin, so within a model
all optimizers share a GPU type for a clean throughput comparison.

Settings come from CLI flags and/or a YAML config, with

```
precedence:  CLI flag  >  config.yaml  >  built-in default
```

### Config file (recommended)

Copy the template, edit it, and run with no flags — `run.sh` auto-loads
`./config.yaml` when present (or pass `--config <path>`):

```bash
cd benchmarks/optimizer-sweep
cp config.example.yaml config.yaml
$EDITOR config.yaml          # set venv, out, gpus, models
./run.sh                     # uses config.yaml for everything
./run.sh --steps 500 --no-muon   # CLI still overrides individual keys
```

`config.yaml` is gitignored, so your local venv/model paths never get committed.
See `config.example.yaml` for every key. Use `./run.sh --dry-run` to resolve and
print the effective settings without launching anything.

### GPU selection (PCI-ordered)

GPU indices are **PCI-bus-ordered** — `run.sh` exports
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the indices you put in `gpus:` / `--gpus`
match `nvidia-smi` and the physical cards the cells pin. List them with:

```bash
./run.sh --list-gpus
#   0  NVIDIA GeForce RTX 3090       00000000:21:00.0  used 18 MiB / 24576 MiB
#   1  NVIDIA GeForce RTX 3090 Ti    00000000:02:00.0  used 19 MiB / 24564 MiB
#   ...
```

Then set e.g. `gpus: "1,2,4,5,7"` (or `--gpus "1,2,4,5,7"`).

### Pure CLI

```bash
./run.sh \
  --venv   /path/to/your/venv/bin/python \
  --out    ./out \
  --gpus   "2,5,7" \
  --steps  2000 \
  --arch   8.6 \
  --model-0p6b /path/to/Qwen3-0.6B \
  --model-1p7b /path/to/Qwen3-1.7B
```

- Add `--lr-sweep` to run a short per-optimizer LR sweep first and use each
  optimizer's best LR for the finals (instead of the documented defaults).
- `gefen_muon` is in the default set; add `--no-muon` to skip it, or set `--opts "..."`.
- Use `--model <tag>=<path>` (repeatable) for arbitrary models/tags.
- `--arch` is your GPU's compute capability (`8.6` Ampere, `12.0` Blackwell).
- Run `./run.sh -h` for the full flag list. Missing required settings fail cleanly.

You can also run a single cell directly:

```bash
/path/to/venv/bin/python sweep_cell.py --opt gefen_fused --model /path/to/Qwen3-1.7B \
  --lr 2e-5 --seed 0 --steps 2000 --seq 2048 --gpu 0 --arch 8.6 \
  --tag qwen1p7b --dataset tatsu-lab/alpaca --out ./out/results.jsonl
```

## 4. Outputs

Written to `--out`:

- `results.jsonl` — one JSON line per cell (and `results_lrsweep.jsonl` if swept)
- `comparison_table.csv` / `comparison_table.md` — the aggregate table + a
  per-model headline (loss gap, throughput ratio, VRAM saved, opt-state)
- Six charts (per model x 3 figures, for 2 models):
  - `quad_<model>.png` — 2x2: eval loss / throughput / peak VRAM / opt-state
  - `loss_curve_<model>.png` — eval-loss curves (solid) + train-EMA (dashed)
  - `tput_vram_<model>.png` — throughput-vs-VRAM efficiency scatter
- `logs/` — per-cell stdout (the loss curves are parsed from these)

## 5. Caveats (honest)

- **Single seed** by default (`--seed 0`): 2000 steps is the signal, but seed
  noise is not estimated. Re-run with other seeds to bound it.
- **`gefen_muon` needs `--muon-adjust match_rms_adamw`** to share the AdamW LR
  scale; with the original Muon scaling its LR is on a different footing.
- **Muon is slower per step** — the Newton-Schulz orthogonalization adds work,
  so `gefen_muon` trades throughput for its update geometry.
- **Throughput is per-GPU.** Compare optimizers *within* a model (same GPU
  type); absolute tok/s across GPU types is not comparable.
- The fair default LRs are short-sweep optima; the true long-run optimum may
  differ, so use `--lr-sweep` for a hardware-honest comparison.
- `adamw4bit`'s low opt-state bytes do **not** translate to low peak VRAM
  (torchao's compiled step allocates large transient buffers).

## 6. Using Gefen in training

To actually train with Gefen in [Axolotl](https://github.com/axolotl-ai-cloud/axolotl),
see the integration PR: **https://github.com/axolotl-ai-cloud/axolotl/pull/3755**
(currently a **draft**). It wires Gefen in as an optimizer:

```yaml
optimizer: gefen
optim_args:
  fused: true
```

Note: `gefen_muon` / GefenMuonHybrid is **not yet exposed** through that Axolotl
integration — it is benchmarked here directly against the optimizer API.
