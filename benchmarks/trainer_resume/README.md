# Transformers Trainer resume harness

This harness validates plain Gefen, Gefen-Muon with an AdamW backup, and Gefen-Muon with a Gefen backup through the Hugging Face `Trainer` lifecycle. It compares an uninterrupted deterministic run against a native Trainer checkpoint continuation and fails unless model, optimizer, scheduler, learning-rate, and logged-loss state agree exactly.

The run also verifies that Trainer creates the optimizer from the model instance passed to it, Trainer's internal Accelerate layer wraps the optimizer, gradient accumulation produces one optimizer step per logical update, tied parameters are routed once, checkpoint hooks reach the underlying optimizer, and DDP replicas retain identical state hashes. This is not a standalone Accelerate harness: it does not exercise a direct `Accelerator` loop or `Accelerator.save_state()`/`load_state()`. It also does not exercise FSDP or a `model_init` callback.

Install the optional framework dependencies and Gefen-X from the repository root:

```bash
python -m pip install -e .
python -m pip install 'transformers>=5,<6' 'accelerate>=1.1'
```

Run the fast CPU matrix:

```bash
PYTHONPATH=.:src python -m benchmarks.trainer_resume.run \
  --output-dir benchmarks/trainer_resume/out/cpu \
  --device cpu --dtype float32 --no-fused
```

Run the fused BF16 matrix against a local model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:src python -m benchmarks.trainer_resume.run \
  --output-dir benchmarks/trainer_resume/out/qwen3-06b \
  --model /path/to/Qwen3-0.6B --attn-implementation eager \
  --device cuda --dtype bfloat16 --fused --gradient-checkpointing \
  --steps 6 --split-step 3 --batch-size 1 --gradient-accumulation-steps 2 --seq-len 64
```

Launch the same validation through Trainer's DDP frontend with homogeneous GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=.:src torchrun --standalone --nproc-per-node=2 \
  -m benchmarks.trainer_resume.run \
  --output-dir benchmarks/trainer_resume/out/qwen3-06b-ddp \
  --model /path/to/Qwen3-0.6B --attn-implementation eager \
  --device cuda --dtype bfloat16 --fused --gradient-checkpointing \
  --steps 6 --split-step 3 --batch-size 1 --gradient-accumulation-steps 2 --seq-len 64
```

Each recipe retains the staged Trainer checkpoint and writes a machine-readable `summary.json` under the selected output directory. Use a new output directory for each run; the harness refuses to mix results into a non-empty phase directory.

The tiny-model CPU command is the fast lifecycle gate. The local-model BF16 and DDP commands exercise the additional options shown; their exact-continuation assertion is a test condition for the selected model, attention implementation, hardware, and framework versions, not a claim that every Transformers architecture is bitwise reproducible across restart. The harness uses native Trainer checkpoints and does not validate FSDP/DCP optimizer checkpoint conversion.
