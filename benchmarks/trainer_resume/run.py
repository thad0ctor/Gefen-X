#!/usr/bin/env python3
"""Validate Gefen optimizer recipes through Hugging Face Trainer save/resume.

The harness compares an uninterrupted run with a stopped-and-resumed run. It
uses Trainer's model-aware optimizer-factory lifecycle, Accelerate wrapping,
gradient accumulation, a changing learning-rate schedule, and native Trainer
checkpoints. The default tiny tied-weight GPT-2 model is dependency-light;
``--model`` accepts a local causal-LM checkpoint for a release gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from gefen import Gefen, GefenMuonHybrid, __version__ as gefen_version


RECIPES = ("gefen", "gefen_muon_adamw", "gefen_muon_gefen")


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


def _recipe_list(value: str) -> tuple[str, ...]:
    recipes = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(recipes) - set(RECIPES))
    if not recipes or unknown:
        choices = ", ".join(RECIPES)
        detail = "no recipes supplied" if not recipes else f"unknown recipes: {', '.join(unknown)}"
        raise argparse.ArgumentTypeError(f"{detail}; choose from {choices}")
    if len(recipes) != len(set(recipes)):
        raise argparse.ArgumentTypeError("recipes must not contain duplicates")
    return recipes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="local causal-LM checkpoint; omit for a tiny random GPT-2")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", choices=("auto", "eager", "sdpa"), default="auto")
    parser.add_argument("--recipes", type=_recipe_list, default=RECIPES)
    parser.add_argument("--steps", type=_positive_int, default=6)
    parser.add_argument("--split-step", type=_positive_int, default=3)
    parser.add_argument("--warmup-steps", type=_nonnegative_int, default=1)
    parser.add_argument("--batch-size", type=_positive_int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=_positive_int, default=2)
    parser.add_argument("--seq-len", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backup-lr", type=float, help="Gefen-backup LR; defaults to half --lr")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--ns-steps", type=_positive_int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


class FixedTokenDataset(Dataset):
    """Deterministic causal-LM rows with no online tokenization or transforms."""

    def __init__(self, *, samples: int, seq_len: int, vocab_size: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(0, vocab_size, (samples, seq_len), generator=generator)

    def __len__(self) -> int:
        return self.input_ids.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.input_ids[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


@dataclass
class HookCounts:
    steps: int = 0
    state_dicts: int = 0
    loads: int = 0


def _increment(tracker: HookCounts, field: str) -> None:
    setattr(tracker, field, getattr(tracker, field) + 1)


class GefenTrainerOptimizerFactory:
    """Transformers 5 model-aware optimizer factory used by ``Trainer``.

    Trainer recognizes non-Optimizer classes as factories, constructs one with
    no arguments, then calls it with the model it owns. This preserves names,
    tied-weight/module-aware Muon routing, and wrapper lifecycle ordering.
    """

    def __call__(
        self,
        model: torch.nn.Module,
        *,
        recipe: str,
        lr: float,
        backup_lr: float,
        weight_decay: float,
        fused: bool,
        deterministic: bool,
        ns_steps: int,
        hook_counts: HookCounts,
    ) -> torch.optim.Optimizer:
        if recipe == "gefen":
            optimizer = Gefen(
                model.named_parameters(),
                lr=lr,
                weight_decay=weight_decay,
                fused=fused,
                deterministic=deterministic,
            )
        elif recipe in ("gefen_muon_adamw", "gefen_muon_gefen"):
            backup_optimizer = recipe.removeprefix("gefen_muon_")
            optimizer = GefenMuonHybrid.from_model(
                model,
                lr=lr,
                muon_lr=lr,
                backup_lr=lr if backup_optimizer == "adamw" else backup_lr,
                backup_optimizer=backup_optimizer,
                weight_decay=weight_decay,
                fused=fused,
                deterministic=deterministic,
                ns_steps=ns_steps,
                ns_schedule="tuned3" if ns_steps == 3 else "standard",
                normuon=True,
                backup_1d_period_one=backup_optimizer == "gefen",
            )
        else:  # pragma: no cover - guarded by argparse and direct-call validation
            raise ValueError(f"unknown recipe {recipe!r}")

        optimizer.register_step_post_hook(lambda *_args, **_kwargs: _increment(hook_counts, "steps"))
        optimizer.register_state_dict_post_hook(
            lambda _optimizer, state_dict: (_increment(hook_counts, "state_dicts"), state_dict)[1]
        )
        optimizer.register_load_state_dict_post_hook(lambda _optimizer: _increment(hook_counts, "loads"))
        return optimizer


class StopAfterStep:
    """Callback implementation is completed lazily after Transformers imports."""

    def __new__(cls, stop_step: int):
        from transformers import TrainerCallback

        class _StopAfterStep(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= stop_step:
                    control.should_training_stop = True
                return control

        return _StopAfterStep()


def _feed_digest(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().contiguous().cpu()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        keys = sorted(value, key=lambda item: (type(item).__qualname__, repr(item)))
        for key in keys:
            _feed_digest(digest, key)
            _feed_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list\0" if isinstance(value, list) else b"tuple\0")
        for item in value:
            _feed_digest(digest, item)
    elif isinstance(value, bool):
        digest.update(b"bool\1" if value else b"bool\0")
    elif isinstance(value, int):
        encoded = str(value).encode()
        digest.update(b"int\0" + encoded + b"\0")
    elif isinstance(value, float):
        digest.update(b"float\0" + struct.pack("!d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8", errors="surrogateescape")
        digest.update(b"str\0" + str(len(encoded)).encode() + b"\0" + encoded)
    elif value is None:
        digest.update(b"none\0")
    else:
        raise TypeError(f"unsupported digest value {type(value).__qualname__}")


def state_digest(value: Any) -> str:
    digest = hashlib.sha256()
    _feed_digest(digest, value)
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ("git", *args), check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
    }


def _distributed_values(value: Any) -> list[Any]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [value]
    values: list[Any] = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(values, value)
    return values


def _assert_replicated(label: str, value: Any) -> None:
    values = _distributed_values(value)
    if any(item != values[0] for item in values[1:]):
        raise AssertionError(f"{label} differs across ranks: {values}")


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def _wait() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        kwargs = {}
        if "nccl" in str(torch.distributed.get_backend()).lower():
            kwargs["device_ids"] = [torch.cuda.current_device()]
        torch.distributed.barrier(**kwargs)


def _device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _dtype(args: argparse.Namespace) -> torch.dtype:
    name = args.dtype
    if name == "auto":
        name = "bfloat16" if args.device == "cuda" else "float32"
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _load_model(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel, set_seed

    set_seed(args.seed)
    if args.model:
        kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": Path(args.model).exists(),
        }
        if args.attn_implementation != "auto":
            kwargs["attn_implementation"] = args.attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, **kwargs)
        except TypeError:  # transformers<4.56 used torch_dtype
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, **kwargs)
    else:
        config = GPT2Config(
            vocab_size=128,
            n_positions=max(args.seq_len, 16),
            n_ctx=max(args.seq_len, 16),
            n_embd=32,
            n_layer=1,
            n_head=4,
            n_inner=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
            tie_word_embeddings=True,
        )
        model = GPT2LMHeadModel(config).to(dtype=dtype)
    model.config.use_cache = False
    model.to(device)
    return model


def _routing_census(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    routed = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(routed) != len({id(parameter) for parameter in routed}):
        raise AssertionError("optimizer routes a trainable parameter more than once")
    if {id(parameter) for parameter in routed} != {id(parameter) for parameter in trainable}:
        raise AssertionError("optimizer routing does not cover every trainable parameter exactly once")

    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            aliases.setdefault(id(parameter), []).append(name)
    tied = [names for names in aliases.values() if len(names) > 1]
    census: dict[str, Any] = {
        "parameter_tensors": len(trainable),
        "parameters": sum(parameter.numel() for parameter in trainable),
        "optimizer_groups": len(optimizer.param_groups),
        "tied_parameter_aliases": tied,
    }
    if isinstance(optimizer, GefenMuonHybrid):
        census.update(
            {
                "muon_tensors": sum(len(group["params"]) for group in optimizer.muon.param_groups)
                if optimizer.muon is not None
                else 0,
                "backup_tensors": sum(len(group["params"]) for group in optimizer.backup.param_groups)
                if optimizer.backup is not None
                else 0,
                "backup_optimizer": optimizer.backup_optimizer,
            }
        )
    return census


def _unwrap_optimizer(optimizer: Any) -> torch.optim.Optimizer:
    current = optimizer
    while type(current).__module__.startswith("accelerate.") and hasattr(current, "optimizer"):
        current = current.optimizer
    if not isinstance(current, torch.optim.Optimizer):
        raise TypeError(f"Trainer produced an unexpected optimizer wrapper {type(current)!r}")
    return current


def _loss_history(history: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("step", "loss", "grad_norm", "learning_rate", "epoch")
    return [{key: row[key] for key in keys if key in row} for row in history if "loss" in row]


def _checkpoint_files(path: Path) -> list[str]:
    files = sorted(item.name for item in path.iterdir() if item.is_file())
    required = {"optimizer.pt", "scheduler.pt", "trainer_state.json"}
    missing = sorted(required - set(files))
    if missing:
        raise AssertionError(f"Trainer checkpoint is missing {missing}: {path}")
    if not any(name.startswith("rng_state") and name.endswith(".pth") for name in files):
        raise AssertionError(f"Trainer checkpoint has no RNG state: {path}")
    if not any(name in files for name in ("model.safetensors", "pytorch_model.bin")):
        raise AssertionError(f"Trainer checkpoint has no model weights: {path}")
    return files


def _phase_dir(output_dir: Path, recipe: str, phase: str) -> Path:
    path = output_dir / recipe / phase
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to mix with non-empty phase directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_phase(
    args: argparse.Namespace,
    *,
    recipe: str,
    phase: str,
    dataset: Dataset,
    dtype: torch.dtype,
    device: torch.device,
    resume_from: Path | None = None,
    stop_step: int | None = None,
) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    model = _load_model(args, dtype, device)
    tracker = HookCounts()
    callbacks = [StopAfterStep(stop_step)] if stop_step is not None else []
    phase_dir = _phase_dir(args.output_dir, recipe, phase)
    training_args = TrainingArguments(
        output_dir=str(phase_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.steps,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=dtype == torch.bfloat16,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="steps" if stop_step is not None else "no",
        save_steps=args.split_step,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
        disable_tqdm=True,
        full_determinism=args.deterministic,
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=args.device == "cpu",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=args.device == "cuda",
    )
    optimizer_kwargs = {
        "recipe": recipe,
        "lr": args.lr,
        "backup_lr": args.backup_lr,
        "weight_decay": args.weight_decay,
        "fused": args.fused,
        "deterministic": args.deterministic,
        "ns_steps": args.ns_steps,
        "hook_counts": tracker,
    }
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks,
        optimizer_cls_and_kwargs=(GefenTrainerOptimizerFactory, optimizer_kwargs),
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
    wrapped_optimizer = trainer.optimizer
    optimizer = _unwrap_optimizer(wrapped_optimizer)
    if not type(wrapped_optimizer).__module__.startswith("accelerate."):
        raise AssertionError(f"Trainer optimizer was not wrapped by Accelerate: {type(wrapped_optimizer)!r}")
    expected_steps = stop_step if stop_step is not None else args.steps - (args.split_step if resume_from else 0)
    if tracker.steps != expected_steps:
        raise AssertionError(
            f"{recipe}/{phase}: optimizer step hook fired {tracker.steps} times, expected {expected_steps}"
        )
    if resume_from is not None and tracker.loads < 1:
        raise AssertionError(f"{recipe}/{phase}: Trainer did not load the underlying optimizer state")

    framework_hook_counts = asdict(tracker)
    routing = _routing_census(model, optimizer)
    model_hash = state_digest(model.state_dict())
    optimizer_hash = state_digest(optimizer.state_dict())
    scheduler_hash = state_digest(trainer.lr_scheduler.state_dict())
    _assert_replicated(f"{recipe}/{phase} model hash", model_hash)
    _assert_replicated(f"{recipe}/{phase} optimizer hash", optimizer_hash)
    _assert_replicated(f"{recipe}/{phase} scheduler hash", scheduler_hash)
    result = {
        "global_step": trainer.state.global_step,
        "model_sha256": model_hash,
        "optimizer_sha256": optimizer_hash,
        "scheduler_sha256": scheduler_hash,
        "loss_history": _loss_history(trainer.state.log_history),
        "hook_counts": framework_hook_counts,
        "routing": routing,
        "optimizer_wrapper": f"{type(wrapped_optimizer).__module__}.{type(wrapped_optimizer).__qualname__}",
        "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
    }
    del trainer, optimizer, wrapped_optimizer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _wait()
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if not 0 < args.split_step < args.steps:
        raise ValueError(f"--split-step must satisfy 0 < split < steps, got {args.split_step}/{args.steps}")
    if args.warmup_steps >= args.steps:
        raise ValueError("--warmup-steps must be smaller than --steps")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be > 0")
    if args.fused is None:
        args.fused = args.device == "cuda"
    if args.device == "cpu" and args.fused:
        raise ValueError("--fused requires --device cuda; pass --no-fused for a CPU run")
    if args.backup_lr is None:
        args.backup_lr = 0.5 * args.lr
    if args.backup_lr <= 0:
        raise ValueError("--backup-lr must be > 0")


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import accelerate
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError(
            "this harness requires transformers and accelerate; install them with "
            "`python -m pip install transformers 'accelerate>=1.1'`"
        ) from exc

    _validate_args(args)
    device = _device(args)
    dtype = _dtype(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probe_model = _load_model(args, dtype, device)
    vocab_size = int(probe_model.config.vocab_size)
    del probe_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    samples = max(
        16,
        args.steps * args.batch_size * args.gradient_accumulation_steps * _world_size() * 2,
    )
    dataset = FixedTokenDataset(samples=samples, seq_len=args.seq_len, vocab_size=vocab_size, seed=args.seed + 1)

    results: dict[str, Any] = {}
    for recipe in args.recipes:
        baseline = _run_phase(
            args, recipe=recipe, phase="baseline", dataset=dataset, dtype=dtype, device=device
        )
        staged = _run_phase(
            args,
            recipe=recipe,
            phase="staged",
            dataset=dataset,
            dtype=dtype,
            device=device,
            stop_step=args.split_step,
        )
        checkpoint = args.output_dir / recipe / "staged" / f"checkpoint-{args.split_step}"
        _wait()
        checkpoint_files = _checkpoint_files(checkpoint)
        resumed = _run_phase(
            args,
            recipe=recipe,
            phase="resumed",
            dataset=dataset,
            dtype=dtype,
            device=device,
            resume_from=checkpoint,
        )

        if staged["global_step"] != args.split_step:
            raise AssertionError(f"{recipe}: staged Trainer stopped at {staged['global_step']}")
        if baseline["global_step"] != args.steps or resumed["global_step"] != args.steps:
            raise AssertionError(f"{recipe}: baseline/resumed Trainer did not reach step {args.steps}")
        for key in ("model_sha256", "optimizer_sha256", "scheduler_sha256", "learning_rates"):
            if baseline[key] != resumed[key]:
                raise AssertionError(f"{recipe}: baseline/resumed {key} mismatch")
        if baseline["loss_history"][: args.split_step] != staged["loss_history"]:
            raise AssertionError(f"{recipe}: pre-checkpoint loss/schedule history mismatch")
        if baseline["loss_history"] != resumed["loss_history"]:
            raise AssertionError(f"{recipe}: post-resume loss/schedule history mismatch")
        if staged["hook_counts"]["state_dicts"] < 1:
            raise AssertionError(f"{recipe}: Trainer checkpoint did not call optimizer.state_dict()")
        if baseline["routing"] != resumed["routing"]:
            raise AssertionError(f"{recipe}: tied-weight/parameter routing changed after resume")

        results[recipe] = {
            "passed": True,
            "baseline": baseline,
            "staged": staged,
            "resumed": resumed,
            "checkpoint": str(checkpoint),
            "checkpoint_files": checkpoint_files,
        }

    cuda_devices = []
    if args.device == "cuda":
        cuda_devices = _distributed_values(torch.cuda.get_device_name(device))
    summary = {
        "schema_version": 1,
        "passed": True,
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir),
            "recipes": list(args.recipes),
            "dtype": str(dtype).removeprefix("torch."),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "gefen_x": gefen_version,
            "world_size": _world_size(),
            "cuda_devices": cuda_devices,
            "git": _git_metadata(),
        },
        "recipes": results,
    }
    _wait()
    if _rank() == 0:
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        _wait()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
