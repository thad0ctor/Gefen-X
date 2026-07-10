"""Decompose the single-GPU GefenMuon step into its component costs.

Builds a GefenMuon over a representative set of 2D transformer weight matrices,
warms up, then CUDA-event times:

  * full opt.step()                      -- end-to-end steady-state cost
  * _zeropower_via_newtonschulz only     -- the NS orthogonalization
  * the quantized-momentum update only   -- the per-param quantized-momentum
                                            update that emits the NS input

and sweeps ns_steps in {3,4,5}. Prints per-step ms and the NS fraction so we can
confirm (or refute) "Newton-Schulz dominates steady state" before optimizing.

The momentum-only timer calls GefenMuon._fused_quantized_momentum_update over
every primed param. It is signature-tolerant so the SAME script can be pointed
(via PYTHONPATH) at either the baseline (which takes a `p` scratch + does a
separate dequant gather) or this branch (single-pass emit) for before/after.

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<3090> \
  TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda-13.3 \
  python profile_muon_step.py
"""
import argparse
import inspect
from collections import OrderedDict

import torch

from gefen import GefenMuon
from gefen.gefen_muon import (
    NS_SCHEDULES,
    _batched_ns_chunk_sizes,
    _batched_ns_shape_eligible,
    _zeropower_via_newtonschulz,
    _zeropower_via_newtonschulz_batched,
)


def make_shapes(hidden, inter, n_layers, n_kv_groups=1):
    """Per-layer 2D weights that Muon would own (q/k/v/o + gate/up/down)."""
    kv = hidden // n_kv_groups
    per_layer = [
        (hidden, hidden),   # q_proj
        (kv, hidden),       # k_proj
        (kv, hidden),       # v_proj
        (hidden, hidden),   # o_proj
        (inter, hidden),    # gate_proj
        (inter, hidden),    # up_proj
        (hidden, inter),    # down_proj
    ]
    return per_layer * n_layers


def cuda_time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=6144)
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-groups", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--batched-ns", type=int, choices=(0, 1), default=0)
    ap.add_argument("--batched-ns-workspace-mib", type=int, default=256)
    ap.add_argument("--ns-schedule", choices=("tuned3", "tuned4"), default=None)
    ap.add_argument("--normuon", type=int, choices=(0, 1), default=0)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    args = ap.parse_args()

    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    shapes = make_shapes(args.hidden, args.inter, args.layers, args.kv_groups)
    n_params = sum(r * c for r, c in shapes)
    print(f"=== {name}  fused={bool(args.fused)} batched_ns={bool(args.batched_ns)} ===")
    print(f"shapes: {len(shapes)} matrices, {n_params/1e6:.1f}M params "
          f"(hidden={args.hidden} inter={args.inter} layers={args.layers})")

    torch.manual_seed(0)
    params = []
    for i, (r, c) in enumerate(shapes):
        p = torch.randn(r, c, device=dev, dtype=torch.bfloat16) * 0.02
        p.requires_grad_(True)
        params.append((f"layer{i}", p))
    grads = [torch.randn(r, c, device=dev, dtype=torch.bfloat16) * 1e-3 for r, c in shapes]

    variants = (
        [(len(NS_SCHEDULES[args.ns_schedule]), args.ns_schedule)]
        if args.ns_schedule is not None
        else [(5, None), (4, None), (3, None)]
    )

    for ns_steps, ns_schedule in variants:
        opt = GefenMuon(params, lr=1e-3, ns_steps=ns_steps, fused=bool(args.fused),
                        adjust_lr_fn="match_rms_adamw",
                        weight_decay=args.weight_decay,
                        ns_schedule=ns_schedule,
                        normuon=bool(args.normuon),
                        batched_ns=bool(args.batched_ns),
                        batched_ns_workspace_bytes=args.batched_ns_workspace_mib << 20)
        coeffs = opt.param_groups[0]["ns_coefficients"]
        resolved_steps = opt.param_groups[0]["ns_steps"]

        def full_step(opt=opt, params=params, grads=grads):
            for (_, p), g in zip(params, grads):
                p.grad = g
            opt.step()

        # prime state (codebook / period) before timing
        for _ in range(3):
            full_step()

        step_ms = cuda_time(full_step, args.iters, args.warmup)
        torch.cuda.synchronize()
        live_bytes = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        full_step()
        torch.cuda.synchronize()
        step_peak_mib = (
            torch.cuda.max_memory_allocated() - live_bytes
        ) / (1 << 20)

        # NS-only: run the same NS over all matrices once per "step". Keep the
        # bf16 param dtype -- NS casts to bf16 internally, so .float() here would
        # charge the timer an fp32->bf16 cast the real step never pays.
        ns_inputs = [g.clone() for g in grads]
        batched_ns_active = bool(args.batched_ns and opt.fused)

        def ns_only(
            ns_inputs=ns_inputs,
            coeffs=coeffs,
            ns_steps=resolved_steps,
            batched_ns_active=batched_ns_active,
        ):
            if not batched_ns_active:
                for x in ns_inputs:
                    _zeropower_via_newtonschulz(x, coeffs, ns_steps, 1e-7)
                return

            buckets = OrderedDict()
            for position, x in enumerate(ns_inputs):
                rows, cols = x.shape
                if _batched_ns_shape_eligible(rows, cols):
                    key = (rows, cols)
                else:
                    key = ("serial", position)
                buckets.setdefault(key, []).append(x)

            workspace_bytes = args.batched_ns_workspace_mib << 20
            for key, bucket in buckets.items():
                if key[0] == "serial":
                    _zeropower_via_newtonschulz(
                        bucket[0], coeffs, ns_steps, 1e-7
                    )
                    continue
                sizes, serial_tail = _batched_ns_chunk_sizes(
                    len(bucket), key[0], key[1], workspace_bytes
                )
                offset = 0
                transposed = key[0] > key[1]
                for size in sizes:
                    chunk = bucket[offset : offset + size]
                    stack = torch.stack(
                        [x.T for x in chunk] if transposed else chunk
                    )
                    _zeropower_via_newtonschulz_batched(
                        stack, coeffs, ns_steps, 1e-7
                    )
                    del stack
                    offset += size
                for x in bucket[offset : offset + serial_tail]:
                    _zeropower_via_newtonschulz(
                        x, coeffs, ns_steps, 1e-7
                    )

        ns_ms = cuda_time(ns_only, args.iters, args.warmup)

        # Momentum-only: the per-param quantized-momentum update that produces the
        # NS input, isolated from NS and the weight write. State (codebook /
        # period / m_codebook / m_magnitude) is already primed by full_step above.
        # NB: this is a TIMING probe -- each call advances the quantized momentum
        # state (the op isn't idempotent), so the measured cost reflects a steadily
        # evolving state, not a fixed one. That's fine for wall-clock timing (the
        # work per call is identical); don't read correctness into the values.
        # Only meaningful for fused runs: full_step() uses the non-fused dequant
        # path when --fused 0, so _fused_quantized_momentum_update is NOT part of
        # the measured step there -- skip it to avoid a misleading decomposition.
        mom_ms = None
        if args.fused:
            mom_fn = opt._fused_quantized_momentum_update
            # baseline signature: (p, state, grad_view, momentum); this branch drops p.
            # Detect the historical positional scratch explicitly: optional
            # keyword-only controls (for example exact fused Nesterov emission)
            # must not make the modern signature look like the four-positional
            # baseline again.
            parameters = tuple(inspect.signature(mom_fn).parameters)
            takes_p = bool(parameters) and parameters[0] == "p"
            primed = []  # (p_or_None, state, grad_view, momentum) precomputed once
            for (_, p), g in zip(params, grads):
                state = opt.state[p]
                gv = opt._automatic_view(g.reshape(-1), state["automatic_period"])
                primed.append((p if takes_p else None, state, gv))
            momentum = opt.param_groups[0]["momentum"]

            def momentum_only(primed=primed, takes_p=takes_p, mom_fn=mom_fn,
                              momentum=momentum):
                for p_arg, state, gv in primed:
                    if takes_p:
                        mom_fn(p_arg, state, gv, momentum)
                    else:
                        mom_fn(state, gv, momentum)

            mom_ms = cuda_time(momentum_only, args.iters, args.warmup)

        frac = 100.0 * ns_ms / step_ms if step_ms else 0.0
        label = (
            f"ns_schedule={ns_schedule}"
            if ns_schedule is not None
            else f"ns_steps={ns_steps}"
        )
        line = (f"{label}: step={step_ms:7.3f} ms  "
                f"NS-only={ns_ms:7.3f} ms  NS≈{frac:4.1f}% of step  "
                f"(rest≈{step_ms-ns_ms:6.3f} ms)  "
                f"step-peak=+{step_peak_mib:.1f} MiB")
        if mom_ms is not None:
            mom_frac = 100.0 * mom_ms / step_ms if step_ms else 0.0
            line += f"  momentum-only={mom_ms:6.3f} ms ({mom_frac:4.1f}% of step)"
        print(line)


if __name__ == "__main__":
    main()
