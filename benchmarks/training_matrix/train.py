#!/usr/bin/env python3
"""Run one reproducible tiny-Qwen optimizer cell.

The same entry point supports from-scratch byte-level pretraining and
instruction tuning from a saved pretraining checkpoint.  Real text/JSONL/HF
data and an explicitly synthetic smoke source share the exact same code path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .cells import CELL_RECIPES, CellBuildConfig, build_optimizer
from .comparison import attach_comparison, immutable_source_fingerprint
from .data import DatasetBundle, build_dataset
from .schedule import GroupLRSchedule
from .tiny_qwen import TINY_QWEN_PRESETS, TinyQwenConfig, TinyQwenForCausalLM, preset_config


CHECKPOINT_FORMAT = "gefen-training-matrix-v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True, choices=tuple(CELL_RECIPES))
    parser.add_argument("--phase", choices=("pretrain", "sft"), default="pretrain")
    parser.add_argument("--source", choices=("synthetic", "text", "jsonl", "hf"), default="synthetic")
    parser.add_argument("--text-file")
    parser.add_argument("--jsonl-file")
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-validation-split")
    parser.add_argument("--hf-revision")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--max-train-blocks",
        type=_positive_int,
        help="hard packed-byte/block cap; may truncate the final selected document",
    )
    parser.add_argument(
        "--max-train-bytes",
        type=_positive_int,
        help="soft raw-byte target that stops at the next whole-document boundary",
    )
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--steps", type=_positive_int, default=1000)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        default=128,
        help="microbatches per optimizer update (default gives 131k tokens/update at batch=4, seq=256)",
    )
    parser.add_argument("--seq-len", type=_positive_int, default=256)
    parser.add_argument("--validation-blocks", type=_positive_int, default=256)
    parser.add_argument("--eval-every", type=_positive_int, default=50)
    parser.add_argument("--tail-evals", type=_positive_int, default=10)
    parser.add_argument("--throughput-warmup", type=_nonnegative_int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--muon-lr", type=float)
    parser.add_argument("--backup-lr", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-weight-decay", type=float)
    parser.add_argument("--backup-weight-decay", type=float)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--muon-eps", type=float, default=1e-7)
    parser.add_argument("--backup-eps", "--eps", dest="backup_eps", type=float, default=1e-8)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--muon-adjust",
        choices=("original", "match_rms_adamw"),
        help="override the cell's explicit Muon LR scaling (default: fair AdamW-scale recipes)",
    )
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--schedule", choices=("constant", "warmup_cosine"), default="warmup_cosine")
    parser.add_argument("--warmup-steps", type=_nonnegative_int)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)

    parser.add_argument(
        "--preset",
        choices=("custom", *TINY_QWEN_PRESETS),
        default="custom",
        help="screen_33m (33,169,408 params), pretrain_134m (134,216,576 params), or custom dimensions",
    )
    parser.add_argument("--hidden-size", type=_positive_int, default=192)
    parser.add_argument("--intermediate-size", type=_positive_int, default=512)
    parser.add_argument("--layers", type=_positive_int, default=6)
    parser.add_argument("--heads", type=_positive_int, default=6)
    parser.add_argument("--kv-heads", type=_positive_int, default=2)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--tie-embeddings", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--deterministic-kernels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-device-name", help="abort unless CUDA device name contains this text")
    parser.add_argument("--expected-cuda-uuid", help="abort unless CUDA_VISIBLE_DEVICES exactly matches this UUID")

    parser.add_argument("--init-checkpoint", help="weights-only handoff from a pretraining cell")
    parser.add_argument(
        "--allow-random-sft",
        action="store_true",
        help="allow SFT without a pretraining checkpoint (off by default)",
    )
    parser.add_argument(
        "--allow-nonpretrain-init",
        action="store_true",
        help="allow a non-pretrain checkpoint or initialized pretraining continuation",
    )
    parser.add_argument("--checkpoint-out")
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    parser.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "archive optimizer state for external inspection only; this harness "
            "does not restore optimizer/scheduler/update state for training resume"
        ),
    )
    parser.add_argument("--results", help="append the complete result to this JSONL file")
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _select_dtype(requested: str, device: torch.device) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[requested]


def _load_checkpoint(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if "model_config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError(f"{path} lacks model_config/model_state")
    return checkpoint


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_config(args: argparse.Namespace, initialization: dict[str, Any] | None) -> TinyQwenConfig:
    if initialization is not None:
        config = TinyQwenConfig(**initialization["model_config"])
        if args.seq_len > config.max_seq_len:
            raise ValueError(
                f"requested seq_len={args.seq_len} exceeds checkpoint max_seq_len={config.max_seq_len}"
            )
        return config
    if args.preset != "custom":
        return preset_config(
            args.preset,
            max_seq_len=args.seq_len,
            rope_theta=args.rope_theta,
            tie_word_embeddings=args.tie_embeddings,
        )
    return TinyQwenConfig(
        max_seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        rope_theta=args.rope_theta,
        tie_word_embeddings=args.tie_embeddings,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_gradients(parameters, supervised_tokens: int) -> None:
    """Turn accumulated summed-NLL gradients into one token-weighted mean."""

    if supervised_tokens <= 0:
        raise RuntimeError("an optimizer update accumulated no supervised tokens")
    scale = 1.0 / supervised_tokens
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)


def materialize_finite_update_loss(
    loss_sum: torch.Tensor, supervised_tokens: int, device: torch.device
) -> float:
    """One synchronized host check per optimizer update, before mutation."""

    _sync(device)
    value = float((loss_sum / supervised_tokens).item())
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite accumulated training loss: {value}")
    return value


def validate_and_clip_gradients(parameters, max_norm: float) -> None:
    """Reject non-finite grads even when clipping is disabled (max_norm=0)."""

    effective_max_norm = max_norm if max_norm > 0 else float("inf")
    torch.nn.utils.clip_grad_norm_(
        parameters,
        effective_max_norm,
        error_if_nonfinite=True,
    )


def training_batch_metadata(
    *, micro_batch_size: int, gradient_accumulation_steps: int, sequence_length: int, optimizer_updates: int
) -> dict[str, int]:
    return {
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": micro_batch_size * gradient_accumulation_steps,
        "tokens_per_optimizer_update": (
            micro_batch_size * gradient_accumulation_steps * sequence_length
        ),
        "optimizer_updates": optimizer_updates,
    }


def throughput_measurement_metadata(
    *,
    measured_tokens: int,
    measured_seconds: float,
    optimizer_updates: int,
    warmup_updates: int,
) -> dict[str, int | float | None]:
    """Return the shared throughput schema for both training backends."""

    return {
        "training_step_tokens_per_second": (
            measured_tokens / measured_seconds if measured_seconds else None
        ),
        "throughput_measured_updates": max(0, optimizer_updates - warmup_updates),
    }


def measurement_policy_metadata(
    *, eval_every: int, tail_evals: int, throughput_warmup: int
) -> dict[str, int]:
    """Return requested evaluation and timing knobs shared by both backends."""

    return {
        "eval_every": eval_every,
        "tail_evals": tail_evals,
        "throughput_warmup": throughput_warmup,
    }


def prepare_tiny_microbatches(
    data: DatasetBundle, update: int, accumulation_steps: int
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    prepared = []
    for microstep in range(accumulation_steps):
        data_index = update * accumulation_steps + microstep
        input_ids, labels = data.training_batch(data_index)
        supervised = int((labels[:, 1:] != -100).sum().item())  # CPU scalar: no device sync
        if supervised == 0:
            raise RuntimeError(
                f"training update {update + 1}, microstep {microstep + 1} has no supervised tokens"
            )
        prepared.append((input_ids, labels, supervised))
    return prepared


def accumulate_tiny_microbatches(
    model: TinyQwenForCausalLM,
    microbatches: list[tuple[torch.Tensor, torch.Tensor, int]],
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """Backprop one token-weighted effective batch without host scalar reads."""

    update_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    total_supervised = 0
    processed_tokens = 0
    for input_ids_cpu, labels_cpu, supervised in microbatches:
        input_ids = input_ids_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        output = model(input_ids, labels=labels)
        output.loss_sum.backward()
        update_loss_sum = update_loss_sum + output.loss_sum.detach()
        total_supervised += supervised
        processed_tokens += input_ids_cpu.numel()
    normalize_gradients(model.parameters(), total_supervised)
    return update_loss_sum, total_supervised, processed_tokens


@torch.inference_mode()
def evaluate(
    model: TinyQwenForCausalLM,
    blocks: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batch_size: int,
) -> tuple[float, int]:
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_tokens = 0
    for start in range(0, len(blocks), batch_size):
        rows = blocks[start : start + batch_size]
        input_ids = torch.stack([row[0] for row in rows]).long().to(device, non_blocking=True)
        labels = torch.stack([row[1] for row in rows]).long().to(device, non_blocking=True)
        output = model(input_ids, labels=labels)
        total_loss = total_loss + output.loss_sum.detach()
        total_tokens += sum(int((row[1][1:] != -100).sum().item()) for row in rows)
    model.train()
    if total_tokens == 0:
        raise RuntimeError("validation set contains no supervised tokens")
    _sync(device)
    return float((total_loss / total_tokens).item()), total_tokens


def _tensor_bytes(value: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        pointer = id(value)
        if pointer in seen:
            return 0
        seen.add(pointer)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item, seen) for item in value)
    return 0


def snapshot_peak_then_measure_serialized_state(
    optimizer: Any, device: torch.device
) -> tuple[int | None, int | None, int]:
    """Freeze training peak before state_dict may clone non-owning views."""

    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
    else:
        peak_allocated = None
        peak_reserved = None
    serialized_state_bytes = _tensor_bytes(optimizer.state_dict())
    return peak_allocated, peak_reserved, serialized_state_bytes


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        return immutable_source_fingerprint(root)
    except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError):
        return {"commit": None, "diff_sha256": None, "dirty": None}


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _nvidia_driver_version(cuda_visible_devices: str | None) -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        rows = [row.strip().split(", ", 1) for row in output.splitlines() if row.strip()]
        if cuda_visible_devices:
            for uuid, version in rows:
                if uuid == cuda_visible_devices:
                    return version
        return rows[0][1] if rows else None
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_checkpoint(
    path: str,
    model: TinyQwenForCausalLM,
    optimizer: Any,
    result: dict[str, Any],
    save_optimizer: bool,
    overwrite: bool,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite checkpoint {destination}; pass --overwrite-checkpoint explicitly"
        )
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "model_config": model.config.to_dict(),
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "metadata": {
            "run_name": result["run_name"],
            "phase": result["phase"],
            "cell": result["cell"],
            "recipe_class": result["optimizer"]["recipe_class"],
            "seed": result["seed"],
            "data_sha256": result["data"]["data_sha256"],
            "parent_checkpoint": result["initialization"],
            "comparison_id": result["comparison_id"],
        },
    }
    if save_optimizer:
        payload["optimizer_state"] = optimizer.state_dict()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=destination.name + ".tmp-",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.deterministic_kernels:
        torch.use_deterministic_algorithms(True)

    device = _select_device(args.device)
    dtype = _select_dtype(args.dtype, device)
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.expected_cuda_uuid and visible_uuid != args.expected_cuda_uuid:
        raise RuntimeError(
            f"expected CUDA_VISIBLE_DEVICES={args.expected_cuda_uuid}, got {visible_uuid!r}"
        )
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    if args.expected_device_name and args.expected_device_name not in device_name:
        raise RuntimeError(
            f"expected device name containing {args.expected_device_name!r}, got {device_name!r}"
        )
    if "RTX PRO 6000" in device_name:
        raise RuntimeError(f"refusing reserved GPU {device_name}")

    initialization = _load_checkpoint(args.init_checkpoint) if args.init_checkpoint else None
    initialization_sha256 = _file_sha256(args.init_checkpoint) if args.init_checkpoint else None
    if args.phase == "pretrain" and initialization is not None and not args.allow_nonpretrain_init:
        raise ValueError(
            "pretraining starts from scratch by default; pass --allow-nonpretrain-init "
            "to continue from --init-checkpoint"
        )
    if args.phase == "sft" and initialization is None and not args.allow_random_sft:
        raise ValueError(
            "SFT requires --init-checkpoint from pretraining; pass --allow-random-sft "
            "for an intentional random-initialization control"
        )
    if (
        args.phase == "sft"
        and initialization is not None
        and initialization.get("metadata", {}).get("phase") != "pretrain"
        and not args.allow_nonpretrain_init
    ):
        raise ValueError(
            "SFT init checkpoint metadata.phase must be 'pretrain'; pass "
            "--allow-nonpretrain-init for an intentional exception"
        )
    config = _model_config(args, initialization)
    model = TinyQwenForCausalLM(config)
    if initialization is not None:
        model.load_state_dict(initialization["model_state"], strict=True)
    model.to(device=device, dtype=dtype)

    data: DatasetBundle = build_dataset(
        phase=args.phase,
        source=args.source,
        sequence_length=args.seq_len,
        validation_blocks=args.validation_blocks,
        updates=args.steps * args.gradient_accumulation_steps,
        batch_size=args.batch_size,
        seed=args.seed,
        text_file=args.text_file,
        jsonl_file=args.jsonl_file,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        hf_config=args.hf_config,
        hf_revision=args.hf_revision,
        hf_validation_split=args.hf_validation_split,
        text_column=args.text_column,
        cache_only=args.cache_only,
        max_train_blocks=args.max_train_blocks,
        max_train_bytes=args.max_train_bytes,
    )

    fused = device.type == "cuda" if args.fused is None else args.fused
    if fused and device.type != "cuda":
        raise ValueError("--fused requires CUDA; pass --no-fused for CPU")
    build_config = CellBuildConfig(
        lr=args.lr,
        muon_lr=args.muon_lr,
        backup_lr=args.backup_lr,
        weight_decay=args.weight_decay,
        muon_weight_decay=args.muon_weight_decay,
        backup_weight_decay=args.backup_weight_decay,
        betas=(args.beta1, args.beta2),
        muon_eps=args.muon_eps,
        backup_eps=args.backup_eps,
        momentum=args.momentum,
        nesterov=args.nesterov,
        fused=fused,
        adjust_lr_fn=args.muon_adjust,
    )
    optimizer, optimizer_recipe = build_optimizer(model, args.cell, build_config)
    optimizer.zero_grad(set_to_none=True)

    if args.warmup_steps is None:
        warmup_steps = 0 if args.schedule == "constant" else min(args.steps - 1, max(1, round(args.steps * 0.03)))
    else:
        warmup_steps = args.warmup_steps
    scheduler = GroupLRSchedule(
        optimizer,
        schedule=args.schedule,
        total_steps=args.steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    evaluation: list[dict[str, Any]] = []
    initial_loss, validation_tokens = evaluate(
        model, data.validation_blocks, device, args.batch_size
    )
    evaluation.append({"step": 0, "loss": initial_loss})
    print(f"[eval] step=0 loss={initial_loss:.8f} tokens={validation_tokens}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    measured_seconds = 0.0
    measured_tokens = 0
    train_losses: list[float] = []
    for update in range(args.steps):
        lrs = scheduler.apply(update)
        microbatches = prepare_tiny_microbatches(
            data, update, args.gradient_accumulation_steps
        )
        _sync(device)
        started = time.perf_counter()
        update_loss_sum, update_supervised_tokens, update_tokens = (
            accumulate_tiny_microbatches(model, microbatches, device)
        )
        try:
            train_loss = materialize_finite_update_loss(
                update_loss_sum, update_supervised_tokens, device
            )
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"update {update + 1}: {exc}") from exc
        try:
            validate_and_clip_gradients(model.parameters(), args.grad_clip)
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(f"update {update + 1}: non-finite gradients") from exc
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        elapsed = time.perf_counter() - started
        train_losses.append(train_loss)
        if update >= args.throughput_warmup:
            measured_seconds += elapsed
            measured_tokens += update_tokens

        completed = update + 1
        if completed % args.eval_every == 0 or completed == args.steps:
            eval_loss, tokens = evaluate(model, data.validation_blocks, device, args.batch_size)
            evaluation.append({"step": completed, "loss": eval_loss})
            print(
                f"[eval] step={completed} loss={eval_loss:.8f} tokens={tokens} "
                f"train_loss={train_losses[-1]:.8f} lrs={','.join(f'{lr:.4g}' for lr in lrs)}",
                flush=True,
            )

    post_update_evaluation = [point for point in evaluation if point["step"] > 0]
    tail_count = min(args.tail_evals, len(post_update_evaluation))
    tail_mean = (
        sum(point["loss"] for point in post_update_evaluation[-tail_count:]) / tail_count
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    peak_vram_bytes, peak_vram_reserved_bytes, optimizer_bytes = (
        snapshot_peak_then_measure_serialized_state(optimizer, device)
    )
    initialization_metadata = (
        {
            "path": str(Path(args.init_checkpoint).resolve()),
            "checkpoint_sha256": initialization_sha256,
            "metadata": initialization.get("metadata", {}),
        }
        if initialization is not None
        else None
    )
    model_metadata = {
        **config.to_dict(),
        "preset": args.preset if initialization is None else "checkpoint",
        "parameter_count": parameter_count,
        "dtype": str(dtype),
    }
    data_metadata = {
        **data.source_metadata,
        "data_sha256": data.data_sha256,
        "order_sha256": data.order_sha256,
        "validation_supervised_tokens": validation_tokens,
    }
    schedule_metadata = scheduler.metadata()
    batch_metadata = training_batch_metadata(
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sequence_length=args.seq_len,
        optimizer_updates=args.steps,
    )
    measurement_policy = measurement_policy_metadata(
        eval_every=args.eval_every,
        tail_evals=args.tail_evals,
        throughput_warmup=args.throughput_warmup,
    )
    runtime_metadata = {
        "device": str(device),
        "device_name": device_name,
        "cuda_visible_devices": visible_uuid,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "transformers_version": _package_version("transformers"),
        "datasets_version": _package_version("datasets"),
        "nvidia_driver_version": _nvidia_driver_version(visible_uuid),
        "deterministic_kernels": args.deterministic_kernels,
        "git": _git_metadata(),
    }
    result: dict[str, Any] = {
        "format": "gefen-training-matrix-result-v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": args.run_name or f"{args.phase}-{args.cell}-seed{args.seed}",
        "phase": args.phase,
        "cell": args.cell,
        "seed": args.seed,
        "optimizer": optimizer_recipe,
        "schedule": schedule_metadata,
        "model": model_metadata,
        "data": data_metadata,
        "evaluation": evaluation,
        "measurement_policy": measurement_policy,
        "final_eval_loss": evaluation[-1]["loss"],
        "tail_eval_count": tail_count,
        "tail_eval_mean": tail_mean,
        "final_train_loss": train_losses[-1],
        **throughput_measurement_metadata(
            measured_tokens=measured_tokens,
            measured_seconds=measured_seconds,
            optimizer_updates=args.steps,
            warmup_updates=args.throughput_warmup,
        ),
        "training_step_timing_scope": (
            "H2D transfer + forward/backward + finite/grad checks + optimizer step + zero_grad; "
            "excludes CPU order lookup/stack/cast and evaluation"
        ),
        "training_batch": batch_metadata,
        "optimizer_serialized_state_bytes": optimizer_bytes,
        "optimizer_serialized_state_bytes_per_parameter": optimizer_bytes / parameter_count,
        "optimizer_state_measurement": (
            "serialized persistent tensor payload; not live-resident allocator delta"
        ),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
        "initialization": initialization_metadata,
        "runtime": runtime_metadata,
    }
    comparison_context = {
        "backend": "tiny_qwen",
        "phase": args.phase,
        "seed": args.seed,
        "model": model_metadata,
        "data": data_metadata,
        "training_batch": batch_metadata,
        "measurement_policy": measurement_policy,
        "schedule": {
            key: schedule_metadata[key]
            for key in ("name", "total_steps", "warmup_steps", "min_lr_ratio")
        },
        "shared_optimizer_args": {
            "lr": args.lr,
            "muon_lr": args.muon_lr,
            "backup_lr": args.backup_lr,
            "weight_decay": args.weight_decay,
            "muon_weight_decay": args.muon_weight_decay,
            "backup_weight_decay": args.backup_weight_decay,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "muon_eps": args.muon_eps,
            "backup_eps": args.backup_eps,
            "momentum": args.momentum,
            "nesterov": args.nesterov,
            "muon_adjust": args.muon_adjust,
            "fused": fused,
            "grad_clip": args.grad_clip,
        },
        "initialization": (
            {
                "checkpoint_sha256": initialization_sha256,
                "metadata": initialization.get("metadata", {}),
            }
            if initialization is not None
            else None
        ),
        "runtime": runtime_metadata,
    }
    attach_comparison(result, comparison_context)
    if args.checkpoint_out:
        _save_checkpoint(
            args.checkpoint_out,
            model,
            optimizer,
            result,
            args.save_optimizer,
            args.overwrite_checkpoint,
        )
        result["checkpoint_out"] = str(Path(args.checkpoint_out).resolve())
        result["checkpoint_sha256"] = _file_sha256(args.checkpoint_out)
    if args.results:
        _append_jsonl(args.results, result)
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.throughput_warmup >= args.steps:
        raise SystemExit("--throughput-warmup must be smaller than --steps")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise SystemExit("--min-lr-ratio must be in [0, 1]")
    if args.grad_clip < 0:
        raise SystemExit("--grad-clip must be >= 0")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
