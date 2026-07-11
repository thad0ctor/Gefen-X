# Reproducible Muon training matrix

This directory provides reproducible full-model SFT and from-scratch training comparisons for AdamW, GefenMuon, and GefenMuonHybrid recipes. Runs keep the data order, evaluation protocol, and measurement metadata explicit so results can be compared safely.

## Tools

| Tool | Purpose |
|---|---|
| `hf_sft.py` | Run one full-model Hugging Face causal-LM SFT cell. |
| `train.py` | Run one dependency-light Qwen-style pretraining or SFT cell, with optional checkpoint handoff. |
| `run_matrix.py` | Plan or sequentially execute several cells with the same training arguments. |
| `consistency_2x2.py` | Cross two pretraining cells with two SFT cells and retain checkpoint lineage. |
| `summarize.py` | Render compatible JSONL result rows as a compact Markdown comparison. |

Generated results, checkpoints, and run-specific reports belong under `benchmarks/training_matrix/out/` and are intentionally untracked.

## Prerequisites

Run commands from the repository root. Install Gefen-X and the optional Hugging Face dependencies:

```bash
python -m pip install -e .
python -m pip install transformers datasets
export GEFEN_TRAINING_PY="${GEFEN_TRAINING_PY:-python}"
```

Full-model SFT and fused cells require CUDA. The `torch_muon_adamw` cell also requires a PyTorch release that provides `torch.optim.Muon`. Commands using `--cache-only` require the pinned model and dataset revisions to be present in the local Hugging Face cache; use `--no-cache-only` for the first download.

## Optimizer cells

Model-aware Muon cells route hidden 2D matrices to Muon and embeddings, the language-model head, norms, and biases to the auxiliary optimizer.

`run_matrix.py` uses these seven core cells by default:

| Cell | Recipe |
|---|---|
| `adamw` | `torch.optim.AdamW` over the full model. |
| `torch_muon_adamw` | Stock Muon NS5 on hidden matrices with an AdamW auxiliary at full LR. |
| `gefen_muon_classic_adamw` | GefenMuon NS5, NorMuon off, with an AdamW auxiliary at full LR. |
| `gefen_hybrid_period1_all` | GefenMuon tuned3 + NorMuon with a Gefen auxiliary, full backup LR, and period-one state on 1D and 2D auxiliary tensors. |
| `gefen_hybrid_recommended` | GefenMuon tuned3 + NorMuon with a Gefen auxiliary, half backup LR, and period-one state on 1D auxiliary tensors. |
| `gefen_hybrid_literal` | Current `GefenMuonHybrid` constructor defaults: Gefen auxiliary, shared LR, and no period-one override. |
| `gefen_hybrid_recommended_2d` | `gefen_hybrid_recommended` plus period-one state on 2D auxiliary tensors. |

Five additional cells are available by explicit name or with `--cells all`:

| Cell | Variant |
|---|---|
| `gefen_muon_tuned3_adamw` | GefenMuon tuned3, NorMuon off, with an AdamW auxiliary. |
| `gefen_muon_normuon_adamw` | GefenMuon tuned3 + NorMuon with an AdamW auxiliary at full LR. |
| `gefen_muon_recommended_adamw` | GefenMuon tuned3 + NorMuon with an AdamW auxiliary at half LR. |
| `gefen_hybrid_split_lr_only` | Literal Gefen hybrid with half backup LR. |
| `gefen_hybrid_period1_only` | Literal Gefen hybrid with period-one state on 1D auxiliary tensors. |

Cells default to `weight_decay=0`. Use `--weight-decay` for a common coefficient or `--muon-weight-decay` and `--backup-weight-decay` for explicit per-half values. `--muon-lr` and `--backup-lr` override recipe LRs. Every resolved value and override is stored in the result row.

Shape-batched Newton-Schulz is opt-in for Gefen Muon cells:

```bash
--batched-ns --batched-ns-workspace-bytes 268435456
```

## Full-model Qwen SFT

The following command runs a pinned Qwen3-0.6B comparison on an RTX 3090. For throughput comparisons, pin one physical device with `CUDA_VISIBLE_DEVICES=<GPU-UUID>` and pass the same UUID through `--expected-cuda-uuid`.

```bash
mkdir -p benchmarks/training_matrix/out/qwen06b

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.run_matrix \
  --backend hf_sft \
  --cells adamw,gefen_muon_normuon_adamw,gefen_hybrid_recommended \
  --execute --output-dir benchmarks/training_matrix/out/qwen06b -- \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dataset tatsu-lab/alpaca \
  --dataset-revision dce01c9b08f87459cf36a430d809084718273017 \
  --no-cache-only \
  --lr 5e-5 --weight-decay 0.01 --schedule constant \
  --steps 2000 --seq-len 2048 --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --validation-blocks 256 --validation-examples 8192 \
  --tail-evals 10
```

This is a common-LR comparison. Use separate runs or a sweep when each recipe needs its own tuned LR.

## From-scratch pretraining

`train.py` uses a byte-level Qwen-style model with tied embedding/head weights, RMSNorm, RoPE, grouped-query attention, and a SwiGLU MLP.

| Preset | Parameters | Shape | Intended use |
|---|---:|---|---|
| `screen_33m` | 33,169,408 | 512 hidden, 1365 MLP, 12 layers, 8:2 GQA | Fast functional and recipe screening. |
| `pretrain_134m` | 134,216,576 | 896 hidden, 2432 MLP, 16 layers, 14:2 GQA | Longer pretraining and checkpoint-handoff comparisons. |

This pinned Tiny Shakespeare command is a short functional smoke, not a quality benchmark:

```bash
mkdir -p benchmarks/training_matrix/out/screen33m

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.run_matrix \
  --cells adamw,torch_muon_adamw,gefen_muon_classic_adamw,gefen_hybrid_recommended \
  --execute --output-dir benchmarks/training_matrix/out/screen33m -- \
  --phase pretrain --source hf \
  --hf-dataset winglian/tiny-shakespeare \
  --hf-revision 619106eee01474d8eaa5dd400b4b405eb3734ebe \
  --no-cache-only \
  --preset screen_33m --seq-len 1024 --batch-size 1 \
  --gradient-accumulation-steps 128 --steps 20 \
  --lr 1e-3 --weight-decay 0.01 --schedule warmup_cosine \
  --validation-blocks 256
```

For substantive pretraining comparisons, use a pinned multi-document dataset with a separate validation split, such as the WikiText-103 command in the checkpoint example below.

## Reproducibility guarantees

- `--steps` means optimizer updates, not microsteps.
- A seed fixes model initialization, held-out selection, packed blocks, and the epoch-cycled training order. Result rows include data and order SHA-256 fingerprints.
- Validation loss is token weighted, and `--validation-blocks` is exact: the run fails before training if the source cannot supply the requested disjoint held-out blocks while retaining training data.
- Gradient accumulation normalizes by the total supervised-token count, including SFT microbatches with different response lengths.
- `constant` and `warmup_cosine` schedules update once per optimizer step and preserve each parameter group's base-LR ratio.
- Result rows include resolved optimizer settings, batch and token counts, evaluation policy, persistent optimizer-state bytes, measured training-step throughput, software versions, device identity, and source provenance.
- Hub model runs record requested and resolved model revisions. Dataset inputs record the requested revision and resolved dataset fingerprint. Local model directories are fingerprinted from relative paths and full file contents.
- `run_matrix.py` fingerprints relevant source before launching children and stops if that source changes between cells.
- Seeded CUDA execution is best-effort by default. `--deterministic-kernels` enables PyTorch's strictest supported checks, but custom fused kernels still require empirical replay verification before claiming bitwise determinism.
- Throughput covers the measured training step, including host-to-device transfer, forward/backward, gradient checks, optimizer step, and zero-grad; evaluation is excluded.

## Checkpoint handoff

Pretraining checkpoints contain model weights and lineage metadata. They are weights-only by default and do not restore optimizer, scheduler, or update-counter state; `--save-optimizer` adds archival state for inspection, not resumable training. SFT requires a checkpoint marked `phase=pretrain` unless an explicit initialization override is supplied. Checkpoint writes are atomic and refuse to overwrite an existing file unless `--overwrite-checkpoint` is passed.

Create a pinned 134M pretraining checkpoint and use it for SFT:

```bash
mkdir -p benchmarks/training_matrix/out

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.train \
  --cell gefen_muon_classic_adamw --phase pretrain \
  --source hf --hf-dataset EleutherAI/wikitext_document_level \
  --hf-config wikitext-103-raw-v1 \
  --hf-revision 647234772b9554e208af6c826f23b99e3cac88c8 \
  --hf-split train --hf-validation-split validation --text-column page \
  --max-train-bytes 167772160 --no-cache-only \
  --preset pretrain_134m --seq-len 1024 --batch-size 1 \
  --gradient-accumulation-steps 128 --steps 1024 \
  --lr 1e-3 --weight-decay 0.01 --schedule warmup_cosine \
  --checkpoint-out benchmarks/training_matrix/out/pretrain-gefen.pt \
  --results benchmarks/training_matrix/out/results.jsonl

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.train \
  --cell adamw --phase sft \
  --source hf --hf-dataset tatsu-lab/alpaca \
  --hf-revision dce01c9b08f87459cf36a430d809084718273017 \
  --no-cache-only \
  --seq-len 1024 --batch-size 1 --gradient-accumulation-steps 1 \
  --steps 500 --lr 1e-4 --schedule warmup_cosine \
  --init-checkpoint benchmarks/training_matrix/out/pretrain-gefen.pt \
  --results benchmarks/training_matrix/out/results.jsonl
```

## Two-by-two consistency matrix

`consistency_2x2.py` creates two pretraining checkpoints and crosses each with two SFT optimizers. Copy the checked example, adapt its data, cache, and device settings, then inspect the six planned commands before execution:

```bash
mkdir -p benchmarks/training_matrix/out/consistency
cp benchmarks/training_matrix/consistency.example.json \
  benchmarks/training_matrix/out/consistency/config.json

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.consistency_2x2 \
  --config benchmarks/training_matrix/out/consistency/config.json \
  --output-dir benchmarks/training_matrix/out/consistency

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.consistency_2x2 \
  --config benchmarks/training_matrix/out/consistency/config.json \
  --output-dir benchmarks/training_matrix/out/consistency --execute
```

Each SFT result records its parent checkpoint's phase, optimizer cell, seed, content SHA-256, and data fingerprint.

## Reading results

Each run appends one self-describing JSON row. Render a Markdown table with speed ratios against compatible AdamW rows:

```bash
PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.summarize \
  --results benchmarks/training_matrix/out/qwen06b/results.jsonl
```

Only compare rows with matching model, phase, seed, data fingerprint, schedule, update count, evaluation cadence, tail window, throughput warmup, and initialization. Use multiple seeds before treating small loss differences as signal.

## CPU smoke and tests

Run a one-step synthetic CPU check without fused kernels:

```bash
PYTHONPATH=.:src python -m benchmarks.training_matrix.run_matrix \
  --cells adamw,gefen_hybrid_literal --execute \
  --output-dir /tmp/gefen-matrix-cpu -- \
  --phase pretrain --source synthetic --steps 1 --batch-size 1 \
  --gradient-accumulation-steps 1 --seq-len 16 --validation-blocks 1 \
  --eval-every 1 --tail-evals 1 --throughput-warmup 0 \
  --lr 1e-3 --weight-decay 0.01 --schedule constant \
  --hidden-size 16 --intermediate-size 32 --layers 1 --heads 2 --kv-heads 1 \
  --device cpu --dtype float32 --no-fused

PYTHONPATH=.:src python -m pytest -q tests/test_training_matrix_harness.py
```
