# FSDP2 sharding sweep

This benchmark compares GefenMuon's Newton–Schulz execution modes across FSDP2 world sizes. It records held-out loss, average step time, and peak allocated VRAM for each `(world_size, sharded_mode)` cell, then renders comparison charts.

## Modes

| Mode | Behavior | Parity |
|---|---|---|
| `exact` | Every rank gathers each full matrix and runs Newton–Schulz locally. | Bit-identical to the single-GPU update. |
| `distributed` | One owner rank processes each full matrix and broadcasts the update. | Bit-identical to `exact` on homogeneous GPUs. |
| `approx` | Each rank processes only its local row shard. | Faster and lower-memory, but intentionally non-parity. |

The benchmark replicates the same packed training block across ranks so that changing the world size does not change the effective batch. It measures optimizer execution, not data-parallel throughput scaling.

## Setup

Install Gefen-X from the repository with the benchmark dependencies. PyTorch 2.5 or newer and a compatible CUDA toolkit are required for FSDP2 and the fused kernels.

```bash
git clone https://github.com/thad0ctor/Gefen-X
cd Gefen-X
pip install -e .
pip install matplotlib datasets transformers
```

## Run the sweep

The launcher accepts a YAML config, command-line flags, or both. Command-line values override the config, and config values override built-in defaults.

```bash
cd benchmarks/sharding-sweep
cp config.example.yaml config.yaml
$EDITOR config.yaml
./run.sh --dry-run
./run.sh
```

`config.yaml` is ignored by Git. Set its Python interpreter, output directory, GPU pool, model, world sizes, and modes for your system. Use `./run.sh --list-gpus` to print the PCI-ordered device indices accepted by `gpus` and `--gpus`.

The same sweep can be configured entirely from the command line:

```bash
./run.sh \
  --venv /path/to/venv/bin/python \
  --out ./out \
  --gpus "0,1,2,3" \
  --model /path/to/Qwen3-0.6B \
  --world-sizes "2 4" \
  --modes "exact distributed approx" \
  --steps 2000 \
  --arch 8.6
```

Use GPUs of the same model for timing comparisons. For a world size of `W`, the launcher uses the first `W` devices from the configured pool.

## Outputs

The output directory contains:

- `results_fsdp2.jsonl`: one result row per world-size/mode cell.
- `logs/<tag>_w<world>_<mode>.log`: the complete log for each cell.
- `muon_shard_loss.png`: held-out loss curves by mode and world size.
- `muon_shard_perf.png`: step-time and peak-VRAM comparisons.

Re-render the charts without rerunning training:

```bash
python plot_sharding.py \
  --results ./out/results_fsdp2.jsonl \
  --logs ./out/logs \
  --out-dir ./out
```

## Interpreting results

- Compare timing only across cells run on the same GPU model and software stack.
- Treat `exact` as the parity reference. `distributed` should match it while distributing matrix work across ranks.
- `approx` changes the optimizer update; evaluate its loss impact at the intended model, world size, and seed count before using it for training.
- Repeat important comparisons with multiple seeds.

For Axolotl usage, select `optimizer: gefenx_muon` and set `sharded_mode` under `optim_args`; see the main project README for optimizer recipes.
