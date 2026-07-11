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

`run.sh` applies the **balanced SFT Gefen-Muon config by default** — `adjust_lr_fn=match_rms_adamw`, an AdamW backup at the full cell LR, tuned3, NorMuon, and no Gefen period-one backup override. Override via config keys (`muon_adjust`, `muon_backup_optimizer`, `muon_backup_lr_fraction`, `muon_backup_1d`, `muon_normuon`) or the corresponding CLI flags. The low-memory alternative selects the Gefen backup, half backup LR, and `backup_1d_period_one`; quality-first pretraining uses classic NS5 and disables NorMuon. See [the optimizer-sweep README](optimizer-sweep/README.md) for the full flag list.

> The published `gefen_muon` SFT numbers use tuned3 + NorMuon with the AdamW backup at full LR. Pass `--no-muon-normuon` for an ablation; use `--muon-backup-optimizer gefen --muon-backup-lr-frac 0.5 --muon-backup-1d` for the low-memory alternative. The effective recipe is recorded in every result line under `muon_flags`.
>
> **Note on the published `gefen_muon` throughput:** it reflects tuned3, which is the retained SFT schedule and was faster than classic NS5 in the SFT ablation. It is not universally loss-neutral: quality-first pretraining uses `ns_schedule="standard"`, five NS steps, and NorMuon off. The other optimizers' rows are schedule-independent.

## microbench scripts

Run from `benchmarks/microbench/` with `gefen` importable (fp8 paths require an sm_89+ GPU):

- `bench_ns_schedule.py` / `plot_ns_schedule.py` — tuned NS coefficient schedules (tuned3 / tuned4) vs the standard 5-step quintic: per-shape step time + orthogonality.
- `bench_ns_fp8.py` / `plot_ns_fp8.py` — fp8 (e4m3) NS vs bf16: per-shape step time + orthogonality drift.
- `bench_batched_ns.py` — serial vs opt-in shape-batched bf16 NS, including speed, relative/max numerical delta, and the conservative workspace estimate; `--full-step` runs a paired tuned3 + NorMuon + Nesterov optimizer comparison.
- `bench_capturable_step.py` — 386M-param optimizer-step timing for eager, capturable eager, manual CUDA graph replay, and `torch.compile(mode="reduce-overhead")`, reporting tail-window CUDA and wall time plus peak allocated memory.
- `profile_capturable_cohorts.py` — CUDA-activity counts and step latency for uniform, decay/no-decay, and independently scheduled tensor-LR capturable cohorts in eager and CUDA-graph modes.
- `plot_ns_levers.py` — the consolidated "faster Newton-Schulz" charts (real-training loss + end-to-end throughput across NS variants).
- `bench_muon_fsdp2.py`, `profile_muon_step.py`, `derive_ns_schedule.py` — distributed-Muon timing, per-op Muon-step profiler, and the schedule-coefficient derivation.
