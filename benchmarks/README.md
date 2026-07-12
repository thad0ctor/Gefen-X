# Gefen benchmarks

These suites measure optimizer quality, throughput, memory, and Muon-specific execution modes. Each suite README defines its inputs, outputs, and full command-line interface.

## Suites

| Suite | Purpose | Entry point |
|---|---|---|
| [`optimizer-sweep/`](optimizer-sweep/README.md) | Compare AdamW, Gefen, and Gefen-Muon on full-model SFT: validation loss, throughput, peak VRAM, and optimizer-state bytes per parameter | `bash benchmarks/optimizer-sweep/run.sh` |
| [`training_matrix/`](training_matrix/README.md) | Compare controlled AdamW/Muon recipes across full-model HF SFT, small-model pretraining, and checkpoint handoff | `PYTHONPATH=.:src python -m benchmarks.training_matrix.run_matrix` |
| [`trainer_resume/`](trainer_resume/README.md) | Validate Trainer's internal Accelerate wrapping, accumulation, scheduling, tied weights, DDP replicas, and exact native Trainer checkpoint continuation | `PYTHONPATH=.:src python -m benchmarks.trainer_resume.run` |
| [`sharding-sweep/`](sharding-sweep/README.md) | Compare Gefen-Muon `sharded_mode` choices under FSDP2 across world sizes | `bash benchmarks/sharding-sweep/run.sh` |
| [`microbench/`](microbench/) | Measure Newton–Schulz schedules, fp8 and batched kernels, capturable steps, and distributed Muon internals | `PYTHONPATH=.:src python benchmarks/microbench/bench_ns_schedule.py --help` |

## Canonical published results

The top-level [benchmark summary](../README.md#benchmarks) is the user-facing result overview. Canonical machine-readable optimizer-comparison rows are [`optimizer_comparison_2000steps.csv`](../docs/benchmarks/optimizer_comparison_2000steps.csv) and [`optimizer_comparison_2000steps.jsonl`](../docs/benchmarks/optimizer_comparison_2000steps.jsonl); supporting logs and plots live in [`docs/benchmarks/`](../docs/benchmarks/). Generated suite outputs remain local unless they appear in those checked-in artifacts.

## Run a suite

Run each block from the repository root.

Optimizer comparison:

```bash
(
  cd benchmarks/optimizer-sweep
  cp config.example.yaml config.yaml
  $EDITOR config.yaml
  bash run.sh
)
```

Training matrix help and CPU smoke:

```bash
PYTHONPATH=.:src python -m benchmarks.training_matrix.run_matrix --help
PYTHONPATH=.:src python -m benchmarks.training_matrix.run_matrix --cells adamw --execute \
  --output-dir /tmp/gefen-matrix-cpu -- \
  --phase pretrain --source synthetic --steps 1 --batch-size 1 \
  --gradient-accumulation-steps 1 --seq-len 16 --validation-blocks 1 \
  --eval-every 1 --tail-evals 1 --throughput-warmup 0 \
  --lr 1e-3 --weight-decay 0.01 --schedule constant \
  --hidden-size 16 --intermediate-size 32 --layers 1 --heads 2 --kv-heads 1 \
  --device cpu --dtype float32 --no-fused
```

Transformers Trainer checkpoint continuation:

```bash
PYTHONPATH=.:src python -m benchmarks.trainer_resume.run \
  --output-dir benchmarks/trainer_resume/out/cpu \
  --device cpu --dtype float32 --no-fused
```

FSDP2 sharding comparison:

```bash
(
  cd benchmarks/sharding-sweep
  cp config.example.yaml config.yaml
  $EDITOR config.yaml
  bash run.sh
)
```

Representative microbenchmarks:

```bash
PYTHONPATH=.:src python benchmarks/microbench/bench_ns_schedule.py --help
PYTHONPATH=.:src python benchmarks/microbench/bench_ns_fp8.py --help
PYTHONPATH=.:src python benchmarks/microbench/bench_batched_ns.py --help
PYTHONPATH=.:src python benchmarks/microbench/bench_capturable_step.py --help
PYTHONPATH=.:src python benchmarks/microbench/profile_capturable_cohorts.py --help
```

## Reproducibility requirements

- Record the exact commit and any local source diff; preserve suite-emitted provenance with the results.
- Keep model and dataset inputs immutable, and record the seed, initialization, optimizer recipe, update count, effective batch, schedule, and evaluation policy.
- Compare throughput only on the same physical GPU model and UUID with the same software stack, timing scope, warmup policy, and no competing workload.
- For `training_matrix`, compare result rows only when their recorded comparison context and data fingerprints match.
- Use repeated seeds for quality conclusions and forward/reverse or rotated cell order for sensitive throughput comparisons.

Use each suite's README for configuration precedence, GPU selection, output schemas, and complete reproduction commands.
