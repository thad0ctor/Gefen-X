# Optimizer sweep: AdamW vs Gefen

This suite compares Gefen with PyTorch fused AdamW, bitsandbytes 8-bit AdamW, and torchao 4-bit AdamW on a reproducible full fine-tune, including an optional learning-rate sweep, final runs, plots, and an aggregate table.

## Benchmark scope

Each model-and-optimizer cell records:

- Final evaluation loss on a 32-example held-out set and the training-loss EMA.
- Steady-state tokens per second after warmup.
- Peak allocated VRAM (`peak_vram_gib`, used by the charts) and peak reserved VRAM (`peak_vram_reserved_gib`, which includes caching-allocator slack and fragmentation).
- Measured optimizer-state bytes per parameter, including packed tensor subclasses.

All cells use the same full-fine-tuning regime: bf16 model weights, gradient checkpointing, micro-batch size 1, sequence length 2048, constant learning rate, 2,000 final steps, evaluation every 50 steps, a fixed seed, and identical packed-data order within each seed. The dataset is `tatsu-lab/alpaca`, formatted with the standard Alpaca prompt, masked so prompt tokens do not contribute to loss, and greedily packed into exact 2,048-token blocks.

## Supported optimizers

| `--opt` value | Configuration |
|---|---|
| `adamw_bf16` | PyTorch fused AdamW with native bf16 state |
| `adamw8bit` | bitsandbytes AdamW 8-bit |
| `adamw4bit` | torchao AdamW 4-bit |
| `gefen_fused` | Gefen with fused CUDA kernels and `factored_v_2d=True` |
| `gefen_nonfused` | Gefen without fused CUDA kernels and `factored_v_2d=True` |
| `gefen_muon` | `GefenMuonHybrid` balanced-SFT recipe: AdamW backup at full LR, tuned3 Newton–Schulz, and NorMuon |

`run.sh` defaults to `adamw_bf16 adamw8bit adamw4bit gefen_fused gefen_muon`. Use `--no-muon` to omit `gefen_muon`, or `--opts "..."` to select an explicit set.

Default learning rates are `5e-5` for the AdamW family and `3e-5` for Gefen and `gefen_muon`. Use `--lr-sweep` to select learning rates for a different model or training regime. The Muon recipe keeps `--muon-adjust match_rms_adamw`, which places its main updates on the AdamW LR scale.

For the lower-memory Muon recipe, pass `--muon-backup-optimizer gefen --muon-backup-lr-frac 0.5 --muon-backup-1d`. Use `--no-gefen-factored-v` only when intentionally measuring block-shared plain-Gefen second moments.

## Setup

Install this Gefen-X fork from source; the PyPI package named `gefen` does not contain this fork's CUDA kernels and fixes.

```bash
git clone https://github.com/thad0ctor/Gefen-X
cd Gefen-X
pip install -e .
pip install bitsandbytes torchao matplotlib datasets transformers
```

The fused path requires a CUDA toolkit and host compiler compatible with the installed PyTorch build. Gefen JIT-compiles its CUDA kernels on first use.

## Run the suite

Run commands from `benchmarks/optimizer-sweep`. The launcher optionally performs an LR sweep, runs every selected optimizer-and-model cell, and then generates plots and tables. Jobs are assigned to the selected GPUs round-robin.

Settings precedence is: CLI flag > YAML config > built-in default.

### YAML configuration

Copy the template to the gitignored `config.yaml`, set the Python interpreter, output directory, GPUs, and models, then run the launcher. Pass `--config <path>` to use another file.

```bash
cd benchmarks/optimizer-sweep
cp config.example.yaml config.yaml
$EDITOR config.yaml
./run.sh
./run.sh --steps 500 --no-muon
```

Use `./run.sh --dry-run` to print the resolved settings without launching jobs. See `config.example.yaml` for every key.

### GPU selection

`run.sh` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `--gpus` and the `gpus:` config key use the same PCI-ordered indices shown by `nvidia-smi` and this command:

```bash
./run.sh --list-gpus
```

Set the selection with `gpus: "0,1"` or `--gpus "0,1"`.

### CLI configuration

```bash
./run.sh \
  --venv /path/to/your/venv/bin/python \
  --out ./out \
  --gpus "0,1,2" \
  --steps 2000 \
  --arch 8.6 \
  --model-0p6b /path/to/Qwen3-0.6B \
  --model-1p7b /path/to/Qwen3-1.7B
```

- Add `--lr-sweep` to select each optimizer's final-run LR from its configured grid.
- Add `--no-muon` to skip `gefen_muon`, or use `--opts "..."` to choose the complete optimizer set.
- Use repeatable `--model <tag>=<path-or-hf-id>` arguments for arbitrary models.
- Set `--arch` to the GPU compute capability, such as `8.6` for Ampere or `12.0` for Blackwell.
- Run `./run.sh -h` for every option.

### Single cell

```bash
mkdir -p ./out
/path/to/venv/bin/python sweep_cell.py --opt gefen_fused --model /path/to/Qwen3-1.7B \
  --lr 3e-5 --seed 0 --steps 2000 --seq 2048 --gpu 0 --arch 8.6 \
  --tag qwen1p7b --dataset tatsu-lab/alpaca --out ./out/results.jsonl
```

## Outputs

The output directory contains:

- `results.jsonl`: one result row per final cell.
- `results_lrsweep.jsonl`: LR-sweep rows when `--lr-sweep` is enabled.
- `comparison_table.csv` and `comparison_table.md`: aggregate metrics and per-model comparisons.
- `quad_<model>.png`: evaluation loss, throughput, peak VRAM, and optimizer-state size.
- `loss_curve_<model>.png`: evaluation-loss curves and training-loss EMA.
- `tput_vram_<model>.png`: throughput versus peak allocated VRAM.
- `logs/`: per-cell stdout used to build the loss curves.

Result rows include optimizer recipe and provenance fields.

## Interpreting results

- Compare throughput only within the same model and GPU type; tokens per second are not comparable across different GPU models.
- The default run uses one seed. Repeat the suite with additional `--seed` values before treating small loss differences as meaningful.
- Retune with `--lr-sweep` when changing the model, data, batch regime, or training horizon.
- `gefen_muon` performs Newton–Schulz orthogonalization and is expected to be slower per step than plain Gefen or AdamW.
- Low optimizer-state bytes do not guarantee low peak VRAM; torchao AdamW 4-bit can allocate large temporary buffers during its step.
- Charts use allocated VRAM; inspect `peak_vram_reserved_gib` when allocator reservation is operationally relevant.

## Related documentation

For framework configuration and production recipes, see the top-level [Gefen-X integrations](../../README.md#gefen-x-integrations) and [Muon recipes](../../README.md#which-muon-recipe-should-i-use).
