# Gefen benchmarks

Reproducible benchmarks behind the numbers in the top-level [README](../README.md#benchmarks). Three suites, each self-contained (point it at a model + some GPUs and it regenerates its data → tables/plots):

| Suite | What it measures | GPUs | How to run |
|---|---|---|---|
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

`run.sh` applies the **recommended Gefen-Muon config by default** — `adjust_lr_fn=match_rms_adamw`, `backup_lr = 0.5 × the cell LR`, and `backup_1d_period_one` — which is what the published table/plots use. Override via config keys (`muon_adjust`, `muon_backup_lr_fraction`, `muon_backup_1d`) or flags (`--muon-adjust`, `--muon-backup-lr-frac`, `--no-muon-backup-1d`). See [the optimizer-sweep README](optimizer-sweep/README.md) for the full flag list.

## microbench scripts

Run from `benchmarks/microbench/` with `gefen` importable (fp8 paths require an sm_89+ GPU):

- `bench_ns_schedule.py` / `plot_ns_schedule.py` — tuned NS coefficient schedules (tuned3 / tuned4) vs the standard 5-step quintic: per-shape step time + orthogonality.
- `bench_ns_fp8.py` / `plot_ns_fp8.py` — fp8 (e4m3) NS vs bf16: per-shape step time + orthogonality drift.
- `plot_ns_levers.py` — the consolidated "faster Newton-Schulz" charts (real-training loss + end-to-end throughput across NS variants).
- `bench_muon_fsdp2.py`, `profile_muon_step.py`, `derive_ns_schedule.py` — distributed-Muon timing, per-op Muon-step profiler, and the schedule-coefficient derivation.
