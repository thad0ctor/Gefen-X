"""Measure opt-in shape-batched Newton--Schulz against the serial path.

The batched path changes GEMM reduction order, so this reports numerical deltas
alongside speed and the conservative workspace estimate. ``--full-step`` runs a
paired GefenMuon comparison with the shipped hybrid-side settings (tuned3,
NorMuon and Nesterov). Retained numbers must be run on an otherwise-idle GPU;
record the physical GPU UUID with the retained artifact.
"""
import argparse
import math
import statistics

import torch

from gefen import GefenMuon
from gefen.gefen_muon import (
    NS_SCHEDULE_3STEP,
    _batched_ns_shape_eligible,
    _batched_ns_workspace_bytes,
    _zeropower_via_newtonschulz,
    _zeropower_via_newtonschulz_batched,
)


SHAPES = (
    (128, 1024),
    (256, 1024),
    (512, 1024),
    (128, 2048),
    (256, 2048),
    (512, 2048),
    (256, 4096),
    (1024, 2048),  # excluded by the min-dimension gate
    (512, 4096),   # excluded by the per-matrix-numel gate
)


def cuda_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / iters


def transformer_shapes(hidden, inter, layers, kv_groups):
    kv = hidden // kv_groups
    return [
        (hidden, hidden),
        (kv, hidden),
        (kv, hidden),
        (hidden, hidden),
        (inter, hidden),
        (inter, hidden),
        (hidden, inter),
    ] * layers


def full_step_benchmark(args):
    shapes = transformer_shapes(
        args.hidden, args.inter, args.layers, args.kv_groups
    )
    torch.manual_seed(20260710)
    initial = [
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02)
        for shape in shapes
    ]
    grads = [
        (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 1e-3)
        for shape in shapes
    ]

    def build(batched):
        params = [
            (f"matrix{i}", torch.nn.Parameter(value.clone()))
            for i, value in enumerate(initial)
        ]
        opt = GefenMuon(
            params,
            lr=1e-3,
            weight_decay=0.0,
            fused=True,
            ns_schedule="tuned3",
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            batched_ns=batched,
            batched_ns_workspace_bytes=args.workspace_mib << 20,
        )

        def step():
            for (_, p), grad in zip(params, grads):
                p.grad = grad
            opt.step()

        return params, opt, step

    serial_params, serial_opt, serial_step = build(False)
    batch_params, batch_opt, batch_step = build(True)
    for _ in range(args.full_step_warmup):
        serial_step()
        batch_step()
    torch.cuda.synchronize()

    serial_samples = []
    batch_samples = []
    for block in range(args.blocks):
        if block % 2 == 0:
            serial_samples.append(cuda_ms(serial_step, 0, args.full_step_iters))
            batch_samples.append(cuda_ms(batch_step, 0, args.full_step_iters))
        else:
            batch_samples.append(cuda_ms(batch_step, 0, args.full_step_iters))
            serial_samples.append(cuda_ms(serial_step, 0, args.full_step_iters))

    def peak_delta(step):
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        step()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - live) / (1 << 20)

    serial_peak = peak_delta(serial_step)
    batch_peak = peak_delta(batch_step)

    delta_sq = 0.0
    serial_update_sq = 0.0
    max_abs = 0.0
    equal = 0
    total = 0
    state_equal = True
    for start, (_, serial), (_, batched) in zip(
        initial, serial_params, batch_params
    ):
        delta = batched.float() - serial.float()
        serial_update = serial.float() - start.float()
        delta_sq += delta.square().sum().item()
        serial_update_sq += serial_update.square().sum().item()
        max_abs = max(max_abs, delta.abs().max().item())
        equal += torch.eq(batched, serial).sum().item()
        total += serial.numel()
        serial_state = serial_opt.state[serial]
        batch_state = batch_opt.state[batched]
        state_equal &= torch.equal(
            serial_state["m_codebook"], batch_state["m_codebook"]
        )
        state_equal &= torch.equal(
            serial_state["m_magnitude"], batch_state["m_magnitude"]
        )

    serial_ms = statistics.median(serial_samples)
    batch_ms = statistics.median(batch_samples)
    rel_update_l2 = math.sqrt(delta_sq) / max(
        math.sqrt(serial_update_sq), 1e-30
    )
    print("device:", torch.cuda.get_device_name(0))
    print(
        "full-step: tuned3 normuon=1 nesterov=1 weight_decay=0 "
        f"matrices={len(shapes)} params={sum(r*c for r, c in shapes)/1e6:.1f}M"
    )
    print(
        f"serial_ms={serial_ms:.6f} range={min(serial_samples):.6f}-"
        f"{max(serial_samples):.6f} peak_mib={serial_peak:.1f}"
    )
    print(
        f"batched_ms={batch_ms:.6f} range={min(batch_samples):.6f}-"
        f"{max(batch_samples):.6f} peak_mib={batch_peak:.1f} "
        f"speedup={serial_ms / batch_ms:.6f}"
    )
    print(
        f"param_rel_update_l2={rel_update_l2:.6g} max_abs={max_abs:.6g} "
        f"equal_fraction={equal / total:.6g} momentum_state_equal={state_equal}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="8,32,64")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--full-step", action="store_true")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--inter", type=int, default=6144)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kv-groups", type=int, default=4)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument("--full-step-warmup", type=int, default=3)
    parser.add_argument("--full-step-iters", type=int, default=5)
    args = parser.parse_args()
    if args.full_step:
        full_step_benchmark(args)
        return
    batches = [int(x) for x in args.batches.split(",")]

    print("device:", torch.cuda.get_device_name(0))
    print("schedule: tuned3; dtype: bf16")
    print(
        "shape batch gate serial_ms batched_ms speedup rel_l2 max_abs "
        "measured_peak_mib workspace_cap_model_mib"
    )
    for rows, cols in SHAPES:
        for batch in batches:
            generator = torch.Generator(device="cuda").manual_seed(
                rows * 100000 + cols * 10 + batch
            )
            inputs = (
                torch.randn(
                    batch, rows, cols, device="cuda", dtype=torch.bfloat16,
                    generator=generator,
                )
                * 1e-3
            )

            def serial_outputs():
                return [
                    _zeropower_via_newtonschulz(
                        x, NS_SCHEDULE_3STEP, 3, 1e-7
                    )
                    for x in inputs
                ]

            def serial_timed():
                # The real serial optimizer consumes each result immediately;
                # do not charge it a full-batch torch.stack copy.
                for x in inputs:
                    _zeropower_via_newtonschulz(
                        x, NS_SCHEDULE_3STEP, 3, 1e-7
                    )

            def batched():
                return _zeropower_via_newtonschulz_batched(
                    inputs.clone(), NS_SCHEDULE_3STEP, 3, 1e-7
                )

            ref = torch.stack(serial_outputs())
            got = batched()
            delta = got.float() - ref.float()
            rel_l2 = delta.norm() / ref.float().norm().clamp(min=1e-12)
            max_abs = delta.abs().max().item()
            del ref, got, delta
            serial_samples = []
            batched_samples = []
            for block in range(args.blocks):
                if block % 2 == 0:
                    serial_samples.append(
                        cuda_ms(serial_timed, args.warmup, args.iters)
                    )
                    batched_samples.append(
                        cuda_ms(batched, args.warmup, args.iters)
                    )
                else:
                    batched_samples.append(
                        cuda_ms(batched, args.warmup, args.iters)
                    )
                    serial_samples.append(
                        cuda_ms(serial_timed, args.warmup, args.iters)
                    )
            serial_ms = statistics.median(serial_samples)
            batched_ms = statistics.median(batched_samples)
            torch.cuda.synchronize()
            base_allocated = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            peak_result = batched()
            torch.cuda.synchronize()
            measured_peak = (
                torch.cuda.max_memory_allocated() - base_allocated
            ) / (1 << 20)
            del peak_result
            workspace = _batched_ns_workspace_bytes(batch, rows, cols) / (1 << 20)
            eligible = (
                batch >= 8
                and _batched_ns_shape_eligible(rows, cols)
            )
            print(
                f"{rows}x{cols} {batch} {int(eligible)} "
                f"{serial_ms:.4f} {batched_ms:.4f} "
                f"{serial_ms / batched_ms:.3f} {rel_l2.item():.6g} "
                f"{max_abs:.6g} {measured_peak:.1f} {workspace:.1f}"
            )


if __name__ == "__main__":
    main()
