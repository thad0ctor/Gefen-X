"""Benchmark issue #44 param-group behavior across branches.

Run this script from any checkout and select the Gefen implementation with
PYTHONPATH:

    PYTHONPATH=/path/to/gefen/src python benchmarks/issue44_param_groups.py \
      --variant baseline --output /tmp/issue44_baseline.json

It emits JSON so the same harness can compare "was" and "is" runs without
importing two Gefen versions into the same Python process.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
Gefen: Any = None
GefenMuon: Any = None
GefenMuonHybrid: Any = None


@dataclass(frozen=True)
class StepCase:
    name: str
    warmup: int
    steps: int
    repeats: int


class TinyLanguageModel(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = 1024,
        dim: int = 192,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_classes: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.Linear(hidden_dim, dim, bias=True),
                    nn.LayerNorm(dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(dim, num_classes, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(tokens)
        x = embeddings[:, 0] + 0.25 * embeddings.mean(dim=1)
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


def ensure_gefen_imported() -> None:
    global Gefen, GefenMuon, GefenMuonHybrid
    if Gefen is not None:
        return
    from gefen import Gefen as _Gefen
    from gefen import GefenMuon as _GefenMuon
    from gefen import GefenMuonHybrid as _GefenMuonHybrid

    Gefen = _Gefen
    GefenMuon = _GefenMuon
    GefenMuonHybrid = _GefenMuonHybrid


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def now(device: torch.device) -> float:
    sync(device)
    return time.perf_counter()


def manual_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def optimizer_params(opt: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    return [p for group in opt.param_groups for p in group["params"]]


def state_dict_size_and_time(opt: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    t0 = now(device)
    state = opt.state_dict()
    t1 = now(device)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    t2 = now(device)
    state_param_groups, state_entries = count_state_dict_groups_and_entries(state)
    return {
        "state_dict_ms": (t1 - t0) * 1000.0,
        "torch_save_ms": (t2 - t1) * 1000.0,
        "serialized_bytes": float(buffer.tell()),
        "state_param_groups": float(state_param_groups),
        "state_entries": float(state_entries),
    }


def count_state_dict_groups_and_entries(state: dict[str, Any]) -> tuple[int, int]:
    if "param_groups" in state:
        return len(state.get("param_groups", [])), len(state.get("state", {}))

    groups = 0
    entries = 0
    for child_key in ("muon", "backup"):
        child = state.get(child_key)
        if child is None:
            continue
        child_groups, child_entries = count_state_dict_groups_and_entries(child)
        groups += child_groups
        entries += child_entries
    return groups, entries


def build_step_model(device: torch.device, *, seed: int = 1001) -> TinyLanguageModel:
    manual_seed(seed, device)
    return TinyLanguageModel().to(device=device, dtype=torch.float32)


def assign_deterministic_grads(params: list[torch.nn.Parameter], device: torch.device, seed: int) -> None:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    for p in params:
        p.grad = torch.randn(
            p.shape,
            dtype=p.dtype,
            device=device,
            generator=generator,
        ).mul_(1.0e-3)


def build_optimizer(case_name: str, model: TinyLanguageModel) -> torch.optim.Optimizer:
    if case_name == "gefen_fused":
        return Gefen(
            list(model.named_parameters()),
            lr=3e-4,
            weight_decay=0.01,
            fused=True,
        )
    if case_name == "gefen_fused_explicit_groups":
        embed_head: list[tuple[str, torch.nn.Parameter]] = []
        matrix_body: list[tuple[str, torch.nn.Parameter]] = []
        norms_biases: list[tuple[str, torch.nn.Parameter]] = []
        for name, param in model.named_parameters():
            if "embed" in name or "head" in name:
                embed_head.append((name, param))
            elif param.ndim == 2:
                matrix_body.append((name, param))
            else:
                norms_biases.append((name, param))
        return Gefen(
            [
                {
                    "params": embed_head,
                    "lr": 2e-4,
                    "weight_decay": 0.0,
                },
                {
                    "params": matrix_body,
                    "lr": 3e-4,
                    "weight_decay": 0.01,
                },
                {
                    "params": norms_biases,
                    "lr": 1e-4,
                    "weight_decay": 0.0,
                },
            ],
            lr=3e-4,
            weight_decay=0.01,
            fused=True,
        )
    if case_name == "gefen_unfused":
        return Gefen(
            list(model.named_parameters()),
            lr=3e-4,
            weight_decay=0.01,
            fused=False,
        )
    if case_name == "gefen_muon_fused":
        named_2d = [
            (name, param)
            for name, param in model.named_parameters()
            if param.ndim == 2 and "embed" not in name and "head" not in name
        ]
        return GefenMuon(
            named_2d,
            lr=3e-4,
            weight_decay=0.01,
            fused=True,
            adjust_lr_fn="match_rms_adamw",
        )
    if case_name == "hybrid_fused":
        return GefenMuonHybrid(
            model,
            lr=3e-4,
            weight_decay=0.01,
            fused=True,
        )
    raise ValueError(f"Unknown step case: {case_name}")


def run_step_case(case: StepCase, device: torch.device) -> dict[str, Any]:
    repeats: list[dict[str, Any]] = []
    for repeat in range(case.repeats):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = build_step_model(device, seed=1001 + repeat)

        construct_t0 = now(device)
        opt = build_optimizer(case.name, model)
        construct_t1 = now(device)

        params = optimizer_params(opt)
        assign_deterministic_grads(params, device, seed=2001 + repeat)

        for _ in range(case.warmup):
            opt.step()
        sync(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(case.steps):
                opt.step()
            end.record()
            sync(device)
            step_ms = start.elapsed_time(end) / case.steps
            peak_memory = float(torch.cuda.max_memory_allocated(device))
        else:
            t0 = time.perf_counter()
            for _ in range(case.steps):
                opt.step()
            t1 = time.perf_counter()
            step_ms = (t1 - t0) * 1000.0 / case.steps
            peak_memory = 0.0

        state_metrics = state_dict_size_and_time(opt, device)
        repeats.append(
            {
                "construct_ms": (construct_t1 - construct_t0) * 1000.0,
                "step_ms": step_ms,
                "param_groups": len(opt.param_groups),
                "param_count": len(params),
                "peak_memory_bytes": peak_memory,
                **state_metrics,
            }
        )

    def med(key: str) -> float:
        return float(statistics.median(float(row[key]) for row in repeats))

    return {
        "name": case.name,
        "warmup_steps": case.warmup,
        "measured_steps": case.steps,
        "repeats": repeats,
        "median": {
            "construct_ms": med("construct_ms"),
            "step_ms": med("step_ms"),
            "param_groups": med("param_groups"),
            "param_count": med("param_count"),
            "peak_memory_bytes": med("peak_memory_bytes"),
            "state_dict_ms": med("state_dict_ms"),
            "torch_save_ms": med("torch_save_ms"),
            "serialized_bytes": med("serialized_bytes"),
            "state_param_groups": med("state_param_groups"),
            "state_entries": med("state_entries"),
        },
    }


def make_dataset(
    *,
    seed: int,
    n_train: int,
    n_val: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x_train = torch.randint(
        vocab_size, (n_train, seq_len), generator=generator, device=device
    )
    x_val = torch.randint(vocab_size, (n_val, seq_len), generator=generator, device=device)

    y_train = x_train[:, 0] % num_classes
    y_val = x_val[:, 0] % num_classes
    return x_train, y_train.long(), x_val, y_val.long()


def build_training_optimizer(name: str, model: TinyLanguageModel) -> torch.optim.Optimizer:
    if name == "gefen_fused":
        return Gefen(
            list(model.named_parameters()),
            lr=1.0e-3,
            weight_decay=0.01,
            fused=True,
        )
    if name == "hybrid_fused":
        return GefenMuonHybrid(
            model,
            lr=7.5e-4,
            weight_decay=0.01,
            fused=True,
        )
    raise ValueError(f"Unknown training optimizer: {name}")


def run_training_case(
    optimizer_name: str,
    *,
    seeds: list[int],
    device: torch.device,
    steps: int,
    batch_size: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        manual_seed(seed, device)
        model = TinyLanguageModel(
            vocab_size=512,
            dim=128,
            hidden_dim=384,
            num_layers=3,
            num_classes=32,
        ).to(device=device, dtype=torch.float32)
        opt = build_training_optimizer(optimizer_name, model)
        x_train, y_train, x_val, y_val = make_dataset(
            seed=seed + 17,
            n_train=4096,
            n_val=1024,
            seq_len=24,
            vocab_size=512,
            num_classes=32,
            device=device,
        )
        batch_generator = torch.Generator(device=device)
        batch_generator.manual_seed(seed + 29)
        losses: list[float] = []

        t0 = now(device)
        for _ in range(steps):
            idx = torch.randint(
                x_train.shape[0],
                (batch_size,),
                device=device,
                generator=batch_generator,
            )
            opt.zero_grad(set_to_none=True)
            logits = model(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        train_t1 = now(device)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = F.cross_entropy(val_logits, y_val)
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean()
        sync(device)

        runs.append(
            {
                "seed": seed,
                "steps": steps,
                "batch_size": batch_size,
                "param_groups": len(opt.param_groups),
                "state_param_groups": count_state_dict_groups_and_entries(
                    opt.state_dict()
                )[0],
                "last20_train_loss": float(statistics.mean(losses[-20:])),
                "val_loss": float(val_loss.detach().cpu()),
                "val_accuracy": float(val_acc.detach().cpu()),
                "train_ms_per_step": (train_t1 - t0) * 1000.0 / steps,
            }
        )

    def mean(key: str) -> float:
        return float(statistics.mean(float(row[key]) for row in runs))

    return {
        "optimizer": optimizer_name,
        "runs": runs,
        "mean": {
            "last20_train_loss": mean("last20_train_loss"),
            "val_loss": mean("val_loss"),
            "val_accuracy": mean("val_accuracy"),
            "train_ms_per_step": mean("train_ms_per_step"),
            "param_groups": mean("param_groups"),
            "state_param_groups": mean("state_param_groups"),
        },
    }


def compare_results(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    def pct(before_value: float, after_value: float) -> float:
        if before_value == 0:
            return float("nan")
        return (after_value - before_value) * 100.0 / before_value

    step_rows = []
    before_steps = {row["name"]: row for row in before["step_benchmarks"]}
    after_steps = {row["name"]: row for row in after["step_benchmarks"]}
    for name in sorted(before_steps):
        b = before_steps[name]["median"]
        a = after_steps[name]["median"]
        step_rows.append(
            {
                "name": name,
                "before_step_ms": b["step_ms"],
                "after_step_ms": a["step_ms"],
                "step_ms_pct": pct(b["step_ms"], a["step_ms"]),
                "before_param_groups": b["param_groups"],
                "after_param_groups": a["param_groups"],
                "before_state_param_groups": b["state_param_groups"],
                "after_state_param_groups": a["state_param_groups"],
                "before_serialized_bytes": b["serialized_bytes"],
                "after_serialized_bytes": a["serialized_bytes"],
                "serialized_bytes_pct": pct(
                    b["serialized_bytes"], a["serialized_bytes"]
                ),
            }
        )

    training_rows = []
    before_training = {row["optimizer"]: row for row in before["training_benchmarks"]}
    after_training = {row["optimizer"]: row for row in after["training_benchmarks"]}
    for name in sorted(before_training):
        b = before_training[name]["mean"]
        a = after_training[name]["mean"]
        training_rows.append(
            {
                "optimizer": name,
                "before_val_loss": b["val_loss"],
                "after_val_loss": a["val_loss"],
                "val_loss_delta": a["val_loss"] - b["val_loss"],
                "before_val_accuracy": b["val_accuracy"],
                "after_val_accuracy": a["val_accuracy"],
                "val_accuracy_delta": a["val_accuracy"] - b["val_accuracy"],
                "before_train_ms_per_step": b["train_ms_per_step"],
                "after_train_ms_per_step": a["train_ms_per_step"],
                "train_ms_per_step_pct": pct(
                    b["train_ms_per_step"], a["train_ms_per_step"]
                ),
                "before_param_groups": b["param_groups"],
                "after_param_groups": a["param_groups"],
            }
        )

    return {
        "before_variant": before["variant"],
        "after_variant": after["variant"],
        "step_benchmarks": step_rows,
        "training_benchmarks": training_rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_gefen_imported()
    device_arg = "cuda:0" if args.device == "cuda" else args.device
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    step_cases = [
        StepCase("gefen_fused", warmup=6, steps=30, repeats=args.repeats),
        StepCase(
            "gefen_fused_explicit_groups",
            warmup=6,
            steps=30,
            repeats=args.repeats,
        ),
        StepCase("gefen_unfused", warmup=4, steps=12, repeats=args.repeats),
        StepCase("gefen_muon_fused", warmup=6, steps=30, repeats=args.repeats),
        StepCase("hybrid_fused", warmup=6, steps=30, repeats=args.repeats),
    ]

    return {
        "variant": args.variant,
        "device": str(device),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            "capability": torch.cuda.get_device_capability(device)
            if device.type == "cuda"
            else None,
        },
        "step_benchmarks": [run_step_case(case, device) for case in step_cases],
        "training_benchmarks": [
            run_training_case(
                name,
                seeds=[11, 13, 17],
                device=device,
                steps=args.training_steps,
                batch_size=args.batch_size,
            )
            for name in ["gefen_fused", "hybrid_fused"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="unknown")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--training-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output")
    parser.add_argument("--compare-before")
    parser.add_argument("--compare-after")
    args = parser.parse_args()

    if args.compare_before or args.compare_after:
        if not args.compare_before or not args.compare_after:
            raise SystemExit("--compare-before and --compare-after must be used together")
        with open(args.compare_before, "r", encoding="utf-8") as f:
            before = json.load(f)
        with open(args.compare_after, "r", encoding="utf-8") as f:
            after = json.load(f)
        result = compare_results(before, after)
    else:
        result = run(args)

    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(encoded)
            f.write("\n")
    print(encoded)


if __name__ == "__main__":
    main()
