#!/usr/bin/env python3
"""Full-model Hugging Face SFT backend for core and isolation optimizer cells.

This is the RTX-3090 benchmark path.  It uses a cached/local Qwen checkpoint,
Alpaca-style instruction records, exact packed blocks, a disjoint large
validation set, token-weighted loss, and final-N evaluation means.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from .cells import (
    BATCHED_NS_DEFAULT_WORKSPACE_BYTES,
    CELL_RECIPES,
    CellBuildConfig,
    build_optimizer,
)
from .comparison import attach_comparison, canonical_json
from .data import ALPACA_NO_INPUT, ALPACA_WITH_INPUT, deterministic_order
from .schedule import GroupLRSchedule
from .train import (
    _append_jsonl,
    _git_metadata,
    _nvidia_driver_version,
    _package_version,
    _sync,
    normalize_gradients,
    materialize_finite_update_loss,
    measurement_policy_metadata,
    snapshot_peak_then_measure_serialized_state,
    training_batch_metadata,
    throughput_measurement_metadata,
    validate_and_clip_gradients,
)


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
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cell", required=True, choices=tuple(CELL_RECIPES))
    parser.add_argument("--model", required=True, help="local checkpoint path or Hugging Face model id")
    parser.add_argument("--model-revision", help="requested immutable model revision/commit")
    parser.add_argument("--tokenizer-revision", help="defaults to --model-revision")
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--dataset-revision", help="requested immutable dataset revision/commit")
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=_positive_int, default=2000)
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        default=1,
        help="microbatches per optimizer update; raise explicitly to test the large-batch regime",
    )
    parser.add_argument("--seq-len", type=_positive_int, default=2048)
    parser.add_argument(
        "--training-pool-blocks",
        type=_positive_int,
        default=8192,
        help="maximum disjoint packed training blocks before deterministic epoch cycling",
    )
    parser.add_argument(
        "--validation-blocks",
        type=_positive_int,
        default=256,
        help="exact held-out block count; abort if the reserved examples cannot supply it",
    )
    parser.add_argument(
        "--validation-examples",
        type=_positive_int,
        default=8192,
        help="disjoint shuffled candidate pool used to build the exact validation-block count",
    )
    parser.add_argument("--eval-every", type=_positive_int, default=100)
    parser.add_argument("--tail-evals", type=_positive_int, default=10)
    parser.add_argument("--throughput-warmup", type=_nonnegative_int, default=15)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lr", type=float, required=True)
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
        help="override the fair AdamW-scale Muon adjustment for a native-scaling ablation",
    )
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--batched-ns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="opt into shape-batched Newton-Schulz for Gefen Muon cells only",
    )
    parser.add_argument(
        "--batched-ns-workspace-bytes",
        type=_positive_int,
        default=BATCHED_NS_DEFAULT_WORKSPACE_BYTES,
        help="temporary-memory cap for the opt-in batched Newton-Schulz path",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--schedule", choices=("constant", "warmup_cosine"), default="constant")
    parser.add_argument("--warmup-steps", type=_nonnegative_int)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--deterministic-kernels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-device-name", default="RTX 3090")
    parser.add_argument("--expected-cuda-uuid")
    parser.add_argument("--results", required=True)
    parser.add_argument("--run-name")
    return parser.parse_args(argv)


def _load_records(args: argparse.Namespace):
    import datasets

    if Path(args.dataset).is_dir():
        records = datasets.load_from_disk(args.dataset)
    else:
        download = datasets.DownloadConfig(local_files_only=True) if args.cache_only else None
        records = datasets.load_dataset(
            args.dataset,
            split=args.dataset_split,
            download_config=download,
            revision=args.dataset_revision,
        )
    return records


def _render(record: dict[str, Any], tokenizer) -> tuple[list[int], list[int]]:
    if "input_ids" in record and "labels" in record:
        return list(record["input_ids"]), list(record["labels"])
    try:
        instruction = str(record["instruction"]).strip()
        response = str(record["output"]).strip()
    except KeyError as exc:
        raise ValueError(
            "dataset rows must contain input_ids/labels or Alpaca instruction/input/output fields"
        ) from exc
    input_text = str(record.get("input") or "").strip()
    template = ALPACA_WITH_INPUT if input_text else ALPACA_NO_INPUT
    prompt = template.format(instruction=instruction, input=input_text) if input_text else template.format(instruction=instruction)
    eos = tokenizer.eos_token or ""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response + eos, add_special_tokens=False)["input_ids"]
    return prompt_ids + response_ids, [-100] * len(prompt_ids) + response_ids


def _pack(
    records,
    indices: Iterable[int],
    tokenizer,
    sequence_length: int,
    limit: int | None,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    blocks: list[tuple[torch.Tensor, torch.Tensor]] = []
    ids_buffer: list[int] = []
    labels_buffer: list[int] = []
    for index in indices:
        ids, labels = _render(dict(records[int(index)]), tokenizer)
        ids_buffer.extend(ids)
        labels_buffer.extend(labels)
        while len(ids_buffer) >= sequence_length:
            ids_chunk = ids_buffer[:sequence_length]
            labels_chunk = labels_buffer[:sequence_length]
            ids_buffer = ids_buffer[sequence_length:]
            labels_buffer = labels_buffer[sequence_length:]
            if any(label != -100 for label in labels_chunk[1:]):
                blocks.append(
                    (
                        torch.tensor(ids_chunk, dtype=torch.long),
                        torch.tensor(labels_chunk, dtype=torch.long),
                    )
                )
                if limit is not None and len(blocks) == limit:
                    return blocks
    return blocks


def _block_hash(blocks: list[tuple[torch.Tensor, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for ids, labels in blocks:
        digest.update(ids.numpy().tobytes())
        digest.update(labels.numpy().tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _local_model_manifest_sha256(model_path: str) -> str | None:
    path = Path(model_path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )
    for child in files:
        relative = child.relative_to(path).as_posix()
        size = child.stat().st_size
        digest.update(b"file\0")
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        bytes_read = 0
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                digest.update(chunk)
        if bytes_read != size:
            raise RuntimeError(
                f"local model file changed while hashing: {child} "
                f"({size} bytes before read, {bytes_read} bytes read)"
            )
    return digest.hexdigest()


def prepare_hf_microbatches(
    training,
    training_order: list[int],
    *,
    update: int,
    accumulation_steps: int,
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor, int]]:
    prepared = []
    for microstep in range(accumulation_steps):
        microbatch = update * accumulation_steps + microstep
        order_start = microbatch * batch_size
        block_indices = training_order[order_start : order_start + batch_size]
        rows = [training[index] for index in block_indices]
        ids = torch.stack([row[0] for row in rows])
        labels = torch.stack([row[1] for row in rows])
        supervised = int((labels[:, 1:] != -100).sum().item())  # CPU only
        if supervised == 0:
            raise RuntimeError(
                f"training update {update + 1}, microstep {microstep + 1} has no labels"
            )
        prepared.append((ids, labels, supervised))
    return prepared


def accumulate_hf_microbatches(model, microbatches, device: torch.device):
    """Backprop summed token NLL without a host scalar read per microbatch."""

    update_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    total_supervised = 0
    processed_tokens = 0
    for ids_cpu, labels_cpu, supervised in microbatches:
        ids = ids_cpu.to(device, non_blocking=True)
        labels = labels_cpu.to(device, non_blocking=True)
        loss = model(ids, labels=labels).loss
        weighted_loss = loss * supervised
        weighted_loss.backward()
        update_loss_sum = update_loss_sum + weighted_loss.detach()
        total_supervised += supervised
        processed_tokens += ids_cpu.numel()
    normalize_gradients(model.parameters(), total_supervised)
    return update_loss_sum, total_supervised, processed_tokens


@torch.inference_mode()
def evaluate(model, blocks, batch_size: int) -> tuple[float, int]:
    model.eval()
    loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
    supervised = 0
    for start in range(0, len(blocks), batch_size):
        rows = blocks[start : start + batch_size]
        ids = torch.stack([row[0] for row in rows]).cuda(non_blocking=True)
        labels = torch.stack([row[1] for row in rows]).cuda(non_blocking=True)
        token_count = sum(int((row[1][1:] != -100).sum().item()) for row in rows)
        output = model(ids, labels=labels)
        loss_sum = loss_sum + output.loss.detach() * token_count
        supervised += token_count
    model.train()
    if not supervised:
        raise RuntimeError("validation data contains no supervised response tokens")
    torch.cuda.synchronize()
    return float((loss_sum / supervised).item()), supervised


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("hf_sft is the CUDA full-model path; use train.py for CPU smoke tests")
    if args.throughput_warmup >= args.steps:
        raise ValueError("throughput_warmup must be smaller than steps")
    visible_uuid = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.expected_cuda_uuid and visible_uuid != args.expected_cuda_uuid:
        raise RuntimeError(
            f"expected CUDA_VISIBLE_DEVICES={args.expected_cuda_uuid}, got {visible_uuid!r}"
        )
    device_name = torch.cuda.get_device_name(0)
    if args.expected_device_name and args.expected_device_name not in device_name:
        raise RuntimeError(
            f"expected a device containing {args.expected_device_name!r}, got {device_name!r}"
        )
    if "RTX PRO 6000" in device_name:
        raise RuntimeError(f"refusing reserved GPU {device_name}")

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.deterministic_kernels:
        torch.use_deterministic_algorithms(True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_revision = args.tokenizer_revision or args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=tokenizer_revision,
        local_files_only=args.cache_only,
    )
    records = _load_records(args)
    dataset_fingerprint = getattr(records, "_fingerprint", None)
    if len(records) <= args.validation_examples:
        raise ValueError(
            f"dataset has {len(records)} examples; validation_examples={args.validation_examples} leaves no training set"
        )
    order = list(range(len(records)))
    random.Random(args.seed).shuffle(order)
    validation_indices = order[: args.validation_examples]
    training_indices = order[args.validation_examples :]
    validation = _pack(
        records, validation_indices, tokenizer, args.seq_len, args.validation_blocks
    )
    if len(validation) != args.validation_blocks:
        raise ValueError(
            f"validation split packed only {len(validation)}/{args.validation_blocks} blocks; "
            "increase --validation-examples or lower --validation-blocks"
        )
    training = _pack(
        records,
        training_indices,
        tokenizer,
        args.seq_len,
        args.training_pool_blocks,
    )
    if not training:
        raise ValueError(
            "training split produced no supervised blocks; use more data or fewer validation examples"
        )
    training_order = deterministic_order(
        len(training),
        args.steps * args.gradient_accumulation_steps,
        args.batch_size,
        args.seed,
    )
    data_sha256 = _block_hash(validation + training)
    record_order_sha256 = hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()
    order_sha256 = hashlib.sha256(",".join(map(str, training_order)).encode()).hexdigest()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            revision=args.model_revision,
            local_files_only=args.cache_only,
        )
    except TypeError:  # transformers<4.56 used torch_dtype
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            revision=args.model_revision,
            local_files_only=args.cache_only,
        )
    model.cuda()
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    optimizer, recipe = build_optimizer(
        model,
        args.cell,
        CellBuildConfig(
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
            fused=args.fused,
            batched_ns=args.batched_ns,
            batched_ns_workspace_bytes=args.batched_ns_workspace_bytes,
            adjust_lr_fn=args.muon_adjust,
        ),
    )
    optimizer.zero_grad(set_to_none=True)
    warmup_steps = (
        (0 if args.schedule == "constant" else min(args.steps - 1, max(1, round(args.steps * 0.03))))
        if args.warmup_steps is None
        else args.warmup_steps
    )
    scheduler = GroupLRSchedule(
        optimizer,
        schedule=args.schedule,
        total_steps=args.steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    evaluation: list[dict[str, Any]] = []
    initial_loss, validation_tokens = evaluate(model, validation, args.batch_size)
    evaluation.append({"step": 0, "loss": initial_loss})
    print(f"[eval] step=0 loss={initial_loss:.8f} tokens={validation_tokens}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    measured_time = 0.0
    measured_tokens = 0
    train_loss = math.nan
    for update in range(args.steps):
        current_lrs = scheduler.apply(update)
        microbatches = prepare_hf_microbatches(
            training,
            training_order,
            update=update,
            accumulation_steps=args.gradient_accumulation_steps,
            batch_size=args.batch_size,
        )
        _sync(torch.device("cuda"))
        started = time.perf_counter()
        update_loss_sum, update_supervised_tokens, update_tokens = (
            accumulate_hf_microbatches(model, microbatches, torch.device("cuda"))
        )
        try:
            train_loss = materialize_finite_update_loss(
                update_loss_sum, update_supervised_tokens, torch.device("cuda")
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
        _sync(torch.device("cuda"))
        elapsed = time.perf_counter() - started
        if update >= args.throughput_warmup:
            measured_time += elapsed
            measured_tokens += update_tokens
        step = update + 1
        if step % args.eval_every == 0 or step == args.steps:
            eval_loss, _ = evaluate(model, validation, args.batch_size)
            evaluation.append({"step": step, "loss": eval_loss})
            print(
                f"[eval] step={step} loss={eval_loss:.8f} train_loss={train_loss:.8f} "
                f"lrs={','.join(f'{lr:.4g}' for lr in current_lrs)}",
                flush=True,
            )

    post_update_evaluation = [point for point in evaluation if point["step"] > 0]
    tail_count = min(args.tail_evals, len(post_update_evaluation))
    tail_mean = (
        sum(point["loss"] for point in post_update_evaluation[-tail_count:]) / tail_count
    )
    peak_vram_bytes, peak_vram_reserved_bytes, state_bytes = (
        snapshot_peak_then_measure_serialized_state(
            optimizer, torch.device("cuda")
        )
    )
    model_metadata = {
        "id": args.model,
        "requested_revision": args.model_revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "config_sha256": _json_sha256(model.config.to_dict()),
        "local_manifest_sha256": _local_model_manifest_sha256(args.model),
        "parameter_count": parameter_count,
        "dtype": str(dtype),
        "gradient_checkpointing": args.gradient_checkpointing,
        "tokenizer": {
            "requested_revision": tokenizer_revision,
            "resolved_revision": tokenizer.init_kwargs.get("_commit_hash"),
            "vocab_sha256": _json_sha256(tokenizer.get_vocab()),
            "init_kwargs_sha256": _json_sha256(tokenizer.init_kwargs),
        },
    }
    data_metadata = {
        "source": "hf_or_local_sft",
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "requested_revision": args.dataset_revision,
        "resolved_fingerprint": dataset_fingerprint,
        "cache_only": args.cache_only,
        "data_sha256": data_sha256,
        "order_sha256": order_sha256,
        "record_order_sha256": record_order_sha256,
        "sequence_length": args.seq_len,
        "training_pool_blocks": len(training),
        "validation_blocks": len(validation),
        "validation_source_examples": args.validation_examples,
        "validation_supervised_tokens": validation_tokens,
    }
    batch_metadata = training_batch_metadata(
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        sequence_length=args.seq_len,
        optimizer_updates=args.steps,
    )
    schedule_metadata = scheduler.metadata()
    measurement_policy = measurement_policy_metadata(
        eval_every=args.eval_every,
        tail_evals=args.tail_evals,
        throughput_warmup=args.throughput_warmup,
    )
    runtime_metadata = {
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
    result = {
        "format": "gefen-hf-sft-matrix-result-v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_name": args.run_name or f"hf-sft-{args.cell}-seed{args.seed}",
        "phase": "sft",
        "cell": args.cell,
        "seed": args.seed,
        "optimizer": recipe,
        "schedule": schedule_metadata,
        "model": model_metadata,
        "data": data_metadata,
        "training_batch": batch_metadata,
        "evaluation": evaluation,
        "measurement_policy": measurement_policy,
        "final_eval_loss": evaluation[-1]["loss"],
        "tail_eval_count": tail_count,
        "tail_eval_mean": tail_mean,
        "final_train_loss": train_loss,
        **throughput_measurement_metadata(
            measured_tokens=measured_tokens,
            measured_seconds=measured_time,
            optimizer_updates=args.steps,
            warmup_updates=args.throughput_warmup,
        ),
        "training_step_timing_scope": (
            "H2D transfer + forward/backward + finite/grad checks + optimizer step + zero_grad; "
            "excludes CPU order lookup/stack/cast and evaluation"
        ),
        "optimizer_serialized_state_bytes": state_bytes,
        "optimizer_serialized_state_bytes_per_parameter": state_bytes / parameter_count,
        "optimizer_state_measurement": (
            "serialized persistent tensor payload; not live-resident allocator delta"
        ),
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_reserved_bytes": peak_vram_reserved_bytes,
        "initialization": None,
        "runtime": runtime_metadata,
    }
    comparison_context = {
        "backend": "hf_sft",
        "phase": "sft",
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
            "fused": args.fused,
            "batched_ns": args.batched_ns,
            "batched_ns_workspace_bytes": args.batched_ns_workspace_bytes,
            "grad_clip": args.grad_clip,
        },
        "initialization": None,
        "runtime": runtime_metadata,
    }
    attach_comparison(result, comparison_context)
    _append_jsonl(args.results, result)
    print("RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.grad_clip < 0:
        raise SystemExit("--grad-clip must be >= 0")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
