# Gefen benchmarks

Reproducible benchmarks behind the numbers in the top-level [README](../README.md#benchmarks). Four suites, each self-contained:

| Suite | What it measures | GPUs | How to run |
|---|---|---|---|
| [`training_matrix/`](training_matrix/README.md) | Seven core AdamW/Muon recipes plus optional single-lever isolation cells; large-batch tiny-Qwen pretraining, checkpoint-crossed SFT, and full-model HF/Qwen SFT | RTX 3090 for retained throughput; CPU smoke | `python -m benchmarks.training_matrix.run_matrix` |
| [`optimizer-sweep/`](optimizer-sweep/README.md) | AdamW vs Gefen vs Gefen-Muon on a full fine-tune — eval loss, throughput, peak VRAM, optimizer-state bytes/param (the main table + quad / loss-curve / throughput-vs-VRAM plots) | single GPU per cell | `bash optimizer-sweep/run.sh` |
| [`sharding-sweep/`](sharding-sweep/README.md) | GefenMuon `sharded_mode` (exact / distributed / approx) convergence + throughput under FSDP2 as shard count grows | 2–4 GPUs | `bash sharding-sweep/run.sh` |
| `microbench/` | Newton-Schulz kernel-level studies — tuned coefficient schedules, fp8, the speed/quality of the NS speed levers | single GPU (fp8 needs sm_89+) | see scripts below |

## Quick start (optimizer sweep)

```bash
cd benchmarks/optimizer-sweep
cp config.example.yaml config.yaml      # edit models / GPUs / dataset
bash run.sh                             # LR-sweep (optional) → 2000-step finals → table → plots
```
Outputs land in the configured out-dir: per-cell logs, `results.jsonl`, `comparison_table.{md,csv}`, and the PNGs. Each suite README documents the regime, config precedence, GPU pinning, and the exact reproduce-it commands.

## Reproducing the published Gefen-Muon numbers

`run.sh` applies the **recommended Gefen-Muon config by default** — `adjust_lr_fn=match_rms_adamw`, `backup_lr = 0.5 × the cell LR`, `backup_1d_period_one`, and (via the `GefenMuonHybrid` default) `normuon`, the throughput-free per-neuron 2nd moment on the Newton-Schulz output (see the `GefenMuonHybrid` docstring). Override via config keys (`muon_adjust`, `muon_backup_lr_fraction`, `muon_backup_1d`, `muon_normuon`) or flags (`--muon-adjust`, `--muon-backup-lr-frac`, `--no-muon-backup-1d`, `--no-muon-normuon`). See [the optimizer-sweep README](optimizer-sweep/README.md) for the full flag list.

> The published `gefen_muon` numbers use the shipped defaults, including `normuon`. Pass `--no-muon-normuon` to disable it (results are then labeled `recommended-no-normuon`); the isolated normuon ablation (0.6B: −0.007/−0.010 final eval over two seeds; 1.7B: ≈⅓ of the residual gap to AdamW on the tail-mean of the last 10 evals) is documented in the main README's normuon section.

> **Note on the published `gefen_muon` throughput:** it reflects the current `GefenMuonHybrid` default `ns_schedule="tuned3"` (loss-neutral, ≈1.4× the classic quintic — measured +38%/+39% on the 3090-class GPUs used). `run.sh` reproduces it (tuned3 is the default); pass `--ns-schedule standard` to `sweep_cell` for the bit-identical classic quintic (≈0.7× the throughput, same loss). The other optimizers' rows are schedule-independent.

## microbench scripts

Run from `benchmarks/microbench/` with `gefen` importable (fp8 paths require an sm_89+ GPU):

- `bench_ns_schedule.py` / `plot_ns_schedule.py` — tuned NS coefficient schedules (tuned3 / tuned4) vs the standard 5-step quintic: per-shape step time + orthogonality.
- `bench_ns_fp8.py` / `plot_ns_fp8.py` — fp8 (e4m3) NS vs bf16: per-shape step time + orthogonality drift.
- `bench_batched_ns.py` — serial vs opt-in shape-batched bf16 NS, including
  speed, relative/max numerical delta, and the conservative workspace estimate;
  `--full-step` runs a paired tuned3 + NorMuon + Nesterov optimizer comparison.
- `bench_capturable_step.py` — 386M-param optimizer-step timing for eager, capturable eager, manual CUDA graph replay, and `torch.compile(mode="reduce-overhead")`, reporting tail-window CUDA and wall time plus peak allocated memory.
- `profile_capturable_cohorts.py` — CUDA-activity counts and step latency for uniform, decay/no-decay, and independently scheduled tensor-LR capturable cohorts in eager and CUDA-graph modes.
- `plot_ns_levers.py` — the consolidated "faster Newton-Schulz" charts (real-training loss + end-to-end throughput across NS variants).
- `bench_muon_fsdp2.py`, `profile_muon_step.py`, `derive_ns_schedule.py` — distributed-Muon timing, per-op Muon-step profiler, and the schedule-coefficient derivation.
