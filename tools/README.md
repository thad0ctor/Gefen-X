# Gefen LR tooling — finder + diagnostics

Standalone, non-invasive tools for setting and understanding the learning rate for the Gefen optimizer family (`Gefen`, `GefenMuon`, `GefenMuonHybrid`). Nothing here changes the optimizer's training math — each entry point either *measures* a short dry run or *recommends* a number you choose to apply. Safe to run against live setups; weights are snapshotted and restored.

## `find_lr` — the recommended way to set Gefen's LR

Instead of guessing a factor versus your AdamW LR, find Gefen's optimal LR empirically:

```python
from gefen import Gefen
from gefen.tools.find_lr import find_lr

res = find_lr(model, train_blocks, optimizer="gefen",
              eval_blocks=eval_blocks, method="sweep")   # res.lr is the recommendation
opt = Gefen(named_params, lr=res.lr, fused=True)
```

CLI: `python -m gefen.tools.find_lr --model <path> --optimizer gefen|gefen_muon|hybrid|adamw --method sweep`

Two modes:
- **`method="sweep"` (recommended, trustworthy):** runs short real fine-tunes over an LR grid and picks the lowest **held-out eval** loss. This is the gold standard.
- **`method="range_test"` (fast, rough):** a Leslie-Smith LR ramp. Cheap, but **only a bracketing aid** — across models it was run-to-run noisy and sometimes several-× off for the Gefen family. Use it to center the sweep grid, not as a final number.

### What the finder shows about Gefen LRs (RTX 5090 / Alpaca, 3 models)
- **Plain `Gefen`** wants a **low, roughly model-independent LR ≈ 1e-5**, i.e. **~0.1–0.4× your AdamW LR** (lower). The static "≈0.6–0.8×" head-dim heuristic overestimates this and varies by model — prefer the finder.
- **`GefenMuonHybrid`** roughly **reuses your AdamW LR** (~1×), with `adjust_lr_fn="match_rms_adamw"`.

## Running it (env, dataset, GPUs)

**Three GPU modes** — `--devices` parallelizes the *search*; `--shard` parallelizes the *model*:

| flags | mode | model placement | use when |
|---|---|---|---|
| `--device cuda:0` | sequential | full model on 1 GPU | small model, simplest |
| `--devices cuda:0 cuda:1 …` | parallel | **full copy on *each* GPU**; LR arms spread across them | model fits on 1 GPU; want a faster sweep |
| `--devices … --shard` | sharded (FSDP2) | **one model *split* across the GPUs**; arms sequential | model too big for 1 GPU |

Selecting multiple GPUs does **not** shard by itself — `--devices` alone replicates the whole model per GPU. Add `--shard` to split one model across them.

Worked example (single GPU):
# --dataset accepts a HF dataset name OR a local .txt/.json/.jsonl path
```bash
python -m gefen.tools.find_lr \
    --model /path/to/Qwen3-1.7B \
    --optimizer gefen --method sweep \
    --dataset tatsu-lab/alpaca \
    --device cuda:0 --seq 512 --bs 8 \
    --sweep-lrs 1e-5 3e-5 1e-4 3e-4
# -> RECOMMENDED base LR for gefen: <number>
```

**Parallel sweep across GPUs** (`--devices`). Each LR arm is independent, so a sweep fans across GPUs — one (or more) LRs per GPU — for a near-linear speedup, with the **same** lowest-eval selection as the sequential run:
# --devices spawns one worker per GPU; the LR grid is round-robin'd across them
```bash
python -m gefen.tools.find_lr \
    --model /path/to/Qwen3-1.7B --optimizer gefen --method sweep \
    --devices cuda:0 cuda:1 cuda:2 \
    --sweep-lrs 1e-5 3e-5 1e-4 3e-4 1e-3 --seq 512 --bs 8
# [find_lr] device cuda:0 <- LRs ['1.0e-05', '3.0e-04']  ... etc
# -> RECOMMENDED base LR for gefen: <number>   (identical to the sequential pick)
```
Measured: a 5-LR `gefen` sweep on Qwen3-0.6B took **202 s on 1 GPU vs 56 s across 3** (~3.6×), same best LR. Workers load the model from `--model` per process (CUDA needs `spawn`; the in-memory `find_lr(model, ...)` API stays single-device). Each LR arm is one short fine-tune, so the wall time is set by the busiest GPU (`ceil(#LRs / #GPUs)` arms). A single `--devices`/`--device` falls back to the sequential path. **Note: `--devices` replicates the full model on every GPU — each card must hold the whole model. For a model too big for one GPU, use `--shard` below.**

**Model-sharded sweep** (`--devices … --shard`, FSDP2). Splits ONE model across the selected GPUs via `fully_shard`; LR arms run sequentially, each arm sharded. Works for `gefen`, `adamw`, and `hybrid` (the Muon path all-gathers the full matrix per step, runs Newton-Schulz, then slices the update back to each rank's shard):
```bash
python -m gefen.tools.find_lr \
    --model /path/to/big-model --optimizer hybrid --method sweep --shard \
    --devices cuda:0 cuda:1 --sweep-lrs 3e-5 1e-4 3e-4 --seq 512 --bs 4
# [find_lr][shard] lr 3.00e-05 -> eval ...  (rank0 shard 0.63x, peak ~5.3 GiB/rank) ...
# -> RECOMMENDED base LR for gefen (sharded): <number>
```
Each LR spins up a torch.distributed (nccl) group across the GPUs, shards the model, fine-tunes `sweep_steps`, evals (rank 0), and tears the group down before the next LR. Per-rank params drop to ~1/N of the model (≈0.63× on 2 GPUs in practice — tied embeddings / small unwrapped modules don't shard evenly), so per-GPU memory is a fraction of the full model. Arms are sequential, so wall time ≈ `#LRs × per-arm time`. Validated for `gefen` and `hybrid` on a heterogeneous 3090+5090 pair and a homogeneous dual-RTX-6000 pair.

**Environment redirection** (`--venv` / `--cuda-home`). If you launch from an interpreter whose CUDA doesn't match (the kernel build needs an `nvcc` matching `torch.version.cuda`), point the CLI at the right venv and toolkit and it **re-execs** itself there before importing the kernels:
```bash
PYTHONPATH=<this-repo-or-shim> python -m gefen.tools.find_lr \
    --venv /path/to/cu130-venv --cuda-home /path/to/cuda-13.0 \
    --model /path/to/Qwen3-0.6B --method range_test --device cuda:0
# [find_lr] re-exec under /path/to/cu130-venv/bin/python (CUDA_HOME=/path/to/cuda-13.0)
# Initializing Gefen optimizer (fused=True).   <- runs the real fused path
```
The re-exec preserves the environment (incl. `PYTHONPATH`, `CUDA_VISIBLE_DEVICES`), so you don't have to install this repo into the target venv — just carry it on `PYTHONPATH`. Omit `--venv/--python` when you already run inside the right env.

- **Environment.** The fork must be importable (on `PYTHONPATH` or `pip install -e .`). The default `fused=True` uses Gefen's CUDA kernels, which **require a CUDA toolkit whose `nvcc` matches `torch.version.cuda`** (the kernel build guard enforces this; use `--venv/--cuda-home` above to satisfy it). If you don't have a matching toolkit, run with `fused=False` (programmatic) — pure-torch, no kernel build, just slower.
- **Dataset.** `--dataset` accepts an installed/cached **HuggingFace dataset name** *or* a path to a local **`.txt` / `.json` / `.jsonl`** file (alpaca-style `instruction/input/output`, or a `text`/`content` field, are auto-detected). Offline machines should use a local file or a pre-cached dataset. The programmatic API takes a `[N, seq]` token tensor directly if you'd rather tokenize yourself.
- **Large models.** Two options: (1) `--shard` (above) splits one model across GPUs via FSDP2 — works for `gefen`, `adamw`, *and* `hybrid` (all FSDP2-capable in this fork). (2) Or exploit that the good LR is largely **scale-transferable** (Gefen's factor tracks `head_dim`/norm-block structure, not parameter count — see the main README's LR section), so **find the LR on a size that fits one GPU (or a smaller same-`head_dim` proxy) and reuse it** — cheaper than a sharded sweep of the full model.

## Diagnostic probes

| entry point | what it measures |
|---|---|
| `lr_range_test` | LR-vs-loss ramp; suggested LR at steepest descent (also the basis of `find_lr` range_test) |
| `calibrate_relative` | per-group LR multipliers that equalize *applied update RMS* to a reference group |
| `calibrate_vs_adamw` | per-parameter Gefen-vs-AdamW update-RMS ratio, via a shadow AdamW fed identical gradients |
| `run_model_calibration.py` | runner that loads a HF model and reports the above per parameter-type (paths via CLI) |

These are diagnostics for *understanding* update magnitudes — useful, but see the finding below before treating any magnitude ratio as an LR prescription.

## Honest finding: matching AdamW *magnitude* is the wrong objective for Muon

`calibrate_vs_adamw` shows that `match_rms_adamw`'s flat `k=0.2` produces Muon updates **below** AdamW's magnitude (q/o_proj ~2.2× under, median ~1.3×). It is tempting to "fix" this by scaling the Muon updates up to match AdamW. We tried that (a per-type prior and an auto-calibration window) and validated it on real-dataset fine-tuning (tatsu-lab/alpaca, held-out eval): **it did not help and slightly hurt** — the original `match_rms_adamw` (k=0.2) tracked/beat AdamW, while the magnitude-matched variants were worse on train and eval.

Conclusion: the *measurement* is correct, but Muon's benefit is the orthogonalized update **direction**, not magnitude parity. So **`match_rms_adamw` stays the recommended default** for the hybrid, and the magnitude-matching machinery is intentionally **not shipped** (only the diagnostic probes that revealed this remain here). It may matter for heavier/from-scratch training — untested.

## Tests
CPU smoke (no CUDA needed): `tests/test_find_lr.py` (range_test across all four families + non-destructive restore) and `tests/test_lr_calibration.py` (the probes). The real-model finder validation runs on GPU.
