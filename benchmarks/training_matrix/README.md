# Reproducible Muon training matrix

This harness answers two different questions without conflating them:

1. Does an optimizer recipe preserve full-model Qwen SFT loss and how fast is
   it on the same RTX 3090?
2. Does Muon's result depend on using the same optimizer in pretraining and
   SFT, or on the effective token batch used during pretraining?

`hf_sft.py` is the full-model Qwen benchmark path. `train.py` is a
dependency-light, from-scratch Qwen-style model that supports pretraining,
checkpoint handoff, and instruction tuning. `run_matrix.py` runs the same data
order through multiple optimizer cells. `consistency_2x2.py` crosses two
pretraining optimizers with two SFT optimizers.

The older [`optimizer-sweep`](../optimizer-sweep/README.md) remains the source
for the published historical rows. This matrix makes the control recipes and
current literal-vs-recommended behavior explicit.

## Seven core cells

Every non-AdamW cell uses Gefen's public model-aware split: hidden 2D matrices
go to Muon; tied embeddings/LM head, norms, and biases go to the auxiliary
optimizer.

| Cell | Hidden matrices | Auxiliary parameters | Default backup LR | Backup period-one | Recipe label |
|---|---|---|---:|---|---|
| `adamw` | `torch.optim.AdamW` | same optimizer | n/a | n/a | native parameter-dtype-state control |
| `torch_muon_adamw` | stock `torch.optim.Muon`, standard NS5 | native-dtype-state AdamW | `1.0 × lr` | n/a | control |
| `gefen_muon_classic_adamw` | `GefenMuon`, standard NS5, NorMuon off | native-dtype-state AdamW | `1.0 × lr` | n/a | classic |
| `gefen_hybrid_period1_all` | `GefenMuon`, tuned3 (3 NS iterations), NorMuon on | Gefen | `1.0 × lr` | 1D and 2D | ablation |
| `gefen_hybrid_recommended` | `GefenMuon`, tuned3 (3 NS iterations), NorMuon on | Gefen | `0.5 × lr` | 1D | recommended |
| `gefen_hybrid_literal` | current `GefenMuonHybrid` defaults | Gefen | `1.0 × lr` | none | literal constructor defaults |
| `gefen_hybrid_recommended_2d` | recommended Muon half | Gefen | `0.5 × lr` | 1D and 2D | recommended + 2D |

`run_matrix` defaults to exactly these seven cells. Five isolation controls are
available with `--cells all` or by explicit name; they are never silently added
to the expensive default run:

| Isolation cell | Exact single contrast |
|---|---|
| `gefen_muon_tuned3_adamw` | classic GefenMuon+AdamW → tuned3 (3 NS iterations); NorMuon remains off and backup LR remains `1.0 × lr` |
| `gefen_muon_normuon_adamw` | tuned3 control → NorMuon on; real AdamW and `1.0 × lr` backup remain fixed |
| `gefen_muon_recommended_adamw` | NorMuon control → real-AdamW backup LR `0.5 × lr`; this then matches the recommended hidden recipe while retaining real AdamW aux |
| `gefen_hybrid_split_lr_only` | literal hybrid → backup LR `0.5 × lr`; period-one remains off (strictly LR-only at WD=0) |
| `gefen_hybrid_period1_only` | literal hybrid → 1D period-one on; backup LR remains shared |

The clean auxiliary-only edges are
`gefen_muon_normuon_adamw → gefen_hybrid_literal` at shared LR/no period-one,
and `gefen_muon_recommended_adamw → gefen_hybrid_split_lr_only` at 0.5 backup
LR/no period-one. The classic → tuned3 → NorMuon chain separately isolates the
two hidden-update levers. The literal/split-only/period-one-only/recommended
quartet is a 2×2 factorial for the two recommended backup levers.

Run these single-lever LR/period isolation edges with `--weight-decay 0`.
Decoupled shrink is `lr × weight_decay`, so at nonzero common WD, halving the
backup LR also halves its effective decay and the edge is no longer strictly
LR-only. Complete-recipe comparisons may keep a common nonzero WD, but should
call it a matched WD coefficient—not matched effective shrink.

The two NS5 controls intentionally use
`adjust_lr_fn="match_rms_adamw"`, so `--lr` is on the same scale as the AdamW
baseline. Native/reference scaling remains an explicit ablation:

```bash
--muon-adjust original
```

All cells default to matched `weight_decay=0`. Set one common value with
`--weight-decay`, or deliberately separate the halves with
`--muon-weight-decay` and `--backup-weight-decay`. `--muon-lr` and
`--backup-lr` similarly override the recorded recipe values. The JSON result
contains every resolved value; an override cannot silently masquerade as the
literal or recommended recipe.

Models are cast to the requested parameter dtype before optimizer creation.
Consequently, the BF16 CUDA AdamW control keeps BF16 `exp_avg`/`exp_avg_sq`
(`zeros_like` parameter state); it is not an FP32-master/FP32-moment mixed-
precision reference. `--dtype float32` is available as a full-FP32 run, but the
harness does not silently represent it as BF16-autocast with FP32 masters.

## Reproducibility contract

- `--steps` always means optimizer updates, not microsteps.
- A seed fixes model initialization, held-out selection, packed blocks, and the
  complete epoch-cycled training order. Results record both data and order
  SHA-256 fingerprints.
- Seeded CUDA replay is best-effort by default: `--no-deterministic-kernels` is
  the default, and custom fused kernels may not be covered by PyTorch's
  deterministic-algorithm registry. Use `--deterministic-kernels` for the
  strictest supported mode, still compare saved fingerprints, and do not claim
  bitwise replay without verifying it empirically.
- Gradient accumulation uses summed token NLL and divides accumulated gradients
  by the total supervised-token count before clipping and stepping. This is
  equivalent to one effective batch even when SFT microbatches have unequal
  response lengths.
- Scheduler updates happen once per optimizer update. Both `constant` and
  `warmup_cosine` multiply each group's own base LR, preserving a split backup
  LR.
- Validation loss is token weighted. Defaults reserve 256 exact blocks; the HF
  path reserves 8,192 disjoint examples to build them (4,096 cached Alpaca
  examples produce only about 208 blocks at Qwen sequence length 2,048).
- `tail_eval_mean` is the mean of up to the final `--tail-evals` post-update
  points (default 10), alongside the single final point. Step-0 initialization
  loss is recorded but never included in the tail.
- Results record microbatch size, accumulation, effective batch,
  tokens/optimizer-update, serialized persistent optimizer-state bytes,
  requested evaluation cadence/tail window/throughput warmup, their resolved
  tail and measured-update counts, device-pipeline training-step throughput,
  device, visible GPU UUID, NVIDIA driver, Python, Transformers, Datasets,
  PyTorch/CUDA versions, and an immutable commit+source-diff fingerprint. Sequential launchers
  capture that fingerprint once for every child result, then recompute it before
  each later cell and abort if relevant source changed; generated files under
  `benchmarks/training_matrix/out/` remain excluded.
- Pinned Hub model IDs record the requested and resolved revisions. A local model
  directory instead gets a streaming SHA-256 manifest over every model,
  tokenizer, configuration, and ancillary file's path and full contents; local
  shard replacement cannot hide behind an unchanged filename or byte count.

## RTX 3090 policy

Retained throughput comparisons use the same physical RTX 3090 by UUID, never
an ordinal that can be remapped:

```text
GPU-22becb18-bef9-4f98-4eee-5fc4bb8c7a53
```

Activate the cu13.3/PyTorch 2.12 environment (or point
`GEFEN_TRAINING_PY` at its interpreter) and serialize access with the shared
lock:

```bash
cd /path/to/Gefen
export GEFEN_TRAINING_PY="${GEFEN_TRAINING_PY:-python}"
mkdir -p benchmarks/training_matrix/out/qwen06b

flock /tmp/gefen-main-3090.lock -c '
  OUT=benchmarks/training_matrix/out/qwen06b
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader | tee "$OUT/census-before.txt"
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-22becb18-bef9-4f98-4eee-5fc4bb8c7a53 \
  PYTHONPATH=.:src \
  "$GEFEN_TRAINING_PY" \
  -m benchmarks.training_matrix.run_matrix --backend hf_sft --execute \
  --output-dir benchmarks/training_matrix/out/qwen06b -- \
  --model Qwen/Qwen3-0.6B \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --dataset tatsu-lab/alpaca \
  --dataset-revision dce01c9b08f87459cf36a430d809084718273017 --cache-only \
  --lr 5e-5 --weight-decay 0.01 --schedule constant \
  --steps 2000 --seq-len 2048 --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --validation-blocks 256 --validation-examples 8192 \
  --tail-evals 10 --expected-device-name "RTX 3090" \
  --expected-cuda-uuid GPU-22becb18-bef9-4f98-4eee-5fc4bb8c7a53
  status=$?
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader | tee "$OUT/census-after.txt"
  exit $status
'
```

Capture the same `nvidia-smi` compute-app query during the run in another
terminal. Do not compare throughput from a 3090 Ti, RTX 5090, or RTX PRO 6000
row. The launcher also refuses the reserved RTX PRO 6000 by name.

The timed scope starts after CPU order lookup/stack/cast and includes H2D,
forward/backward, finite/gradient checks, optimizer step, and zero-grad; it
excludes evaluation. It is a measured device-pipeline training-step rate, not
full input-pipeline end-to-end throughput. Reduce thermal/order bias with
separate forward and reverse runs (`--order forward` / `--order reverse`) in
separate output directories—an ABBA pair—or rotate the starting cell with
`--order rotate --rotation N` across seeds.

Functional and loss-only jobs may run in parallel on the otherwise identical,
currently idle RTX 3090
`GPU-61d1ade4-4ce0-cfed-2b99-ae742967acbf`. Record its census and UUID in the
result, but do not merge its throughput with the retained main-UUID benchmark.
Avoid `GPU-0a7d50c4-ed16-645c-d31a-d7ce9f0af241` while its external JRIP
process is present.

The machine already has cached `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`,
`tatsu-lab/alpaca`, `winglian/tiny-shakespeare`, and the pinned document-level
WikiText-103 configuration; `--cache-only` makes an
accidental network fetch fail rather than changing the data underneath a run.
With cached Alpaca and Qwen tokenization, the default
`--training-pool-blocks 8192` is a cap: the remaining records yield roughly 2.2–2.4k unique packed
blocks, which the recorded deterministic order cycles as needed.

## From-scratch pretraining

The built-in model is byte-level but keeps the topology relevant to routing:
tied embedding/LM head, RMSNorm, RoPE, grouped-query attention with separate
Q/K/V/O matrices, and a SwiGLU MLP. Two 3090-friendly presets are included:

Multi-row text sources preserve every row/document whole, group exact duplicate
rows by content hash, and assign each entire group to one side before packing.
A single long text row instead uses recorded, non-overlapping contiguous byte
ranges. Order within each side is preserved. The cached Tiny Shakespeare has
472 HF rows and, at sequence 1024, yields 551 training blocks plus 256 held-out
blocks (`whole_row_content_hash_groups`); row IDs and whole-row hashes are
disjoint, but about 910 exact nonblank line strings still recur across rows
because it is one literary corpus. Treat it as an offline consistency/smoke
source, not credible pretraining generalization. Use a provenance-rich,
multi-document corpus with document IDs for a credible quality run.

The stronger small-pretraining pilot uses `EleutherAI/wikitext_document_level`, config
`wikitext-103-raw-v1`, pinned revision
`647234772b9554e208af6c826f23b99e3cac88c8`, column `page`, and official
`train`/`validation` splits. The cached fingerprints are
`57fbbccacc214a80` / `db990d651d8c2e83` (29,444 / 60 documents). Rows remain
in source order within each split; `--max-train-bytes` stops only at a whole-
document boundary and the result records selected document IDs/hashes plus raw
and packed bytes. Use at least 80 MiB unique train bytes for a 500-update 33M
triage and 160 MiB for the 1,024-update 134M run. License metadata is
inconsistent across the derivative dataset (CC-BY-SA 3.0) and underlying card
(4.0); treat CC-BY-SA 3.0 as the conservative requirement and verify before
redistributing derived artifacts.

| Preset | Exact parameters | Shape | Intended use |
|---|---:|---|---|
| `screen_33m` | 33,169,408 | 512 hidden / 1365 MLP / 12 layers / 8:2 GQA | fast recipe screening |
| `pretrain_134m` | 134,216,576 | 896 hidden / 2432 MLP / 16 layers / 14:2 GQA | stronger small-pretraining confirmation/pilot |

The 134M preset's head dimension is 64; its repeated 128×896 K/V shapes also
exercise batched Newton-Schulz. At sequence 1024, microbatch 1, accumulation
128, the effective batch is exactly 131,072 input tokens per optimizer update.

A 20-update compile/divergence/throughput smoke:

```bash
PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.run_matrix \
  --cells adamw,torch_muon_adamw,gefen_muon_classic_adamw,gefen_hybrid_recommended \
  --execute --output-dir benchmarks/training_matrix/out/screen33m -- \
  --phase pretrain --source hf --hf-dataset winglian/tiny-shakespeare --cache-only \
  --preset screen_33m --seq-len 1024 --batch-size 1 \
  --gradient-accumulation-steps 128 --steps 20 \
  --lr 1e-3 --weight-decay 0.01 --schedule warmup_cosine \
  --validation-blocks 256
```

For real checkpoint outputs, run cells separately (or use the 2×2 driver) so
each cell gets its own checkpoint filename. The checked-in
[`consistency.example.json`](consistency.example.json) config uses
`pretrain_134m`, 1×1024×128 = 131,072 tokens/update, 160 MiB of pinned cached
document-level WikiText-103 with its official validation split, and pinned
cached Alpaca SFT.

## LR, decay, schedule, and batch grids

Do not use one under-tuned point as the loss verdict. A practical two-stage
matrix is:

1. Hold data, seed, weight decay, schedule, and effective tokens/update fixed;
   sweep a common LR grid to expose scale errors.
2. Report both the matched-LR row and each recipe's own best validation LR.

Suggested starting grids (refine around the winner):

| Knob | From-scratch screen | Qwen HF SFT |
|---|---|---|
| main LR | `3e-4, 6e-4, 1e-3, 2e-3` | `2e-5, 3e-5, 5e-5, 8e-5` |
| matched weight decay | `0, 0.01, 0.1` | `0, 0.01, 0.1` |
| backup LR fraction | `0.25, 0.5, 0.75, 1.0` | same |
| schedule | `constant`, `warmup_cosine` | same |
| HF accumulation | n/a | `1, 8, 32, 64` (batch-regime ablation) |

Warmup-cosine defaults to a resolved 3% warmup and `min_lr_ratio=0.1`; set
`--warmup-steps` to make a cross-run value exact. Keep both half-decays matched
for the primary comparison. Per-half decay, `--muon-adjust original`, 2D
period-one, and different backup fractions are labeled ablations, not extra
unannounced degrees of freedom.

## Checkpoint handoff and 2×2 consistency

A direct pretrain-to-SFT handoff is weights-only by default, preventing the SFT
optimizer from accidentally inheriting the pretraining optimizer state:

SFT requires a checkpoint whose metadata says `phase=pretrain`; random SFT and
non-pretrain initialization require the explicit `--allow-random-sft` or
`--allow-nonpretrain-init` escape hatch. Pretraining rejects an init checkpoint
by default. Checkpoint writes are atomic, refuse overwrite unless
`--overwrite-checkpoint` is passed, and the parent result records the saved
file's SHA-256.

```bash
PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.train \
  --cell gefen_hybrid_recommended --phase pretrain \
  --source hf --hf-dataset EleutherAI/wikitext_document_level \
  --hf-config wikitext-103-raw-v1 \
  --hf-revision 647234772b9554e208af6c826f23b99e3cac88c8 \
  --hf-split train --hf-validation-split validation --text-column page \
  --max-train-bytes 167772160 --cache-only \
  --preset pretrain_134m --seq-len 1024 --batch-size 1 \
  --gradient-accumulation-steps 128 --steps 1024 \
  --lr 1e-3 --schedule warmup_cosine \
  --checkpoint-out benchmarks/training_matrix/out/pretrain-gefen.pt \
  --results benchmarks/training_matrix/out/results.jsonl

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.train \
  --cell adamw --phase sft --source hf --hf-dataset tatsu-lab/alpaca \
  --hf-revision dce01c9b08f87459cf36a430d809084718273017 --cache-only \
  --seq-len 1024 --batch-size 1 --gradient-accumulation-steps 1 --steps 500 \
  --lr 1e-4 --schedule warmup_cosine \
  --init-checkpoint benchmarks/training_matrix/out/pretrain-gefen.pt \
  --results benchmarks/training_matrix/out/results.jsonl
```

Plan the full two-pretrain × two-SFT cross without launching anything:

```bash
PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.consistency_2x2 \
  --config benchmarks/training_matrix/consistency.example.json \
  --output-dir benchmarks/training_matrix/out/consistency
```

The checked config itself requires the main UUID/name. Execute the six jobs
under the same lock (and take the census only after acquiring it):

```bash
flock /tmp/gefen-main-3090.lock -c '
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=GPU-22becb18-bef9-4f98-4eee-5fc4bb8c7a53 \
  PYTHONPATH=.:src \
  "$GEFEN_TRAINING_PY" \
  -m benchmarks.training_matrix.consistency_2x2 \
  --config benchmarks/training_matrix/consistency.example.json \
  --output-dir benchmarks/training_matrix/out/consistency --execute
'
```

Each of the four SFT result rows records its parent checkpoint's phase, cell,
seed, content SHA-256, and data fingerprint.

## CPU smoke and tests

Use accumulation 1 and a deliberately tiny custom model for a fast local
functional check:

```bash
PYTHONPATH=.:src python -m benchmarks.training_matrix.run_matrix --execute \
  --output-dir /tmp/gefen-matrix-cpu -- \
  --phase pretrain --source synthetic --steps 1 --batch-size 1 \
  --gradient-accumulation-steps 1 --seq-len 16 --validation-blocks 1 \
  --eval-every 1 --tail-evals 1 --throughput-warmup 0 \
  --lr 1e-3 --weight-decay 0.01 --schedule constant \
  --hidden-size 16 --intermediate-size 32 --layers 1 --heads 2 --kv-heads 1 \
  --device cpu --dtype float32 --no-fused

PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m pytest -q tests/test_training_matrix_harness.py
```

One-step smoke throughput is startup-dominated and is not a benchmark.
Likewise, 20 updates is not a loss-quality screen: prior Muon transients can
take roughly 500 updates to catch up. Use about 500 updates for first-pass
quality triage, then confirm survivors with multiple seeds and a longer horizon.

## Reading results

Each run appends one self-describing JSON row. A compact Markdown comparison,
including device-pipeline training-step speed ratio against a matching AdamW row, is:

```bash
PYTHONPATH=.:src "$GEFEN_TRAINING_PY" -m benchmarks.training_matrix.summarize \
  --results benchmarks/training_matrix/out/qwen06b/results.jsonl
```

Only rows with the same model, phase, seed, data fingerprint, schedule, update
count, evaluation cadence, tail window, throughput warmup, and initialization
are valid speedup comparisons. These requested measurement-policy values are
part of every new row's comparison ID. Run multiple seeds before treating small
tail-loss differences as signal.

## Runtime and storage planning

Runtime scales with `steps × gradient_accumulation_steps × micro_batch_size ×
seq_len`. The 20-update smoke command processes 2.62M input tokens per
cell; the 134M example at 1×1024×128 processes 131,072 tokens per optimizer
update. Run a 1–2 update CUDA pilot after JIT warmup and extrapolate from the
recorded tokens/sec before launching the full matrix. First fused Gefen use may
spend minutes compiling CUDA extensions, and that startup must not be counted
as steady-state throughput. A stronger 134M multi-seed confirmation is a multi-hour
job; none is launched by the smoke instructions.

Weights-only checkpoints are approximately 64 MiB for `screen_33m` and 256 MiB
for `pretrain_134m` in BF16, plus small serialization metadata. The default does
not save optimizer state. `--save-optimizer` adds two native-dtype AdamW moments
(about 0.5 GiB at 134M with BF16 parameters) and recipe-dependent Gefen state,
so enable it only for archival inspection. The harness does not restore
optimizer, scheduler, or update-counter state, so this payload is not a
training-resume checkpoint. JSONL result rows are small; packed
blocks live in host memory and are not written to disk. The cached Qwen3-0.6B
checkpoint itself occupies about 1.5 GiB on this machine.
