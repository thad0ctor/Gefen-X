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
import os
import sys

import torch


def _strip_shadowing_gefen():
    # A flat `gefen.py` on sys.path shadows the installed `gefen/` package and
    # breaks `from gefen import ...`; drop any such entry (the real package,
    # reached via the PYTHONPATH symlink, survives).
    sys.path[:] = [
        p for p in sys.path
        if not os.path.isfile(os.path.join(p or ".", "gefen.py"))
    ]


_strip_shadowing_gefen()

from gefen import GefenMuon  # noqa: E402
from gefen.gefen_muon import (  # noqa: E402
    _zeropower_via_newtonschulz, DEFAULT_A, DEFAULT_B, DEFAULT_C)


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
    args = ap.parse_args()

    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    shapes = make_shapes(args.hidden, args.inter, args.layers, args.kv_groups)
    n_params = sum(r * c for r, c in shapes)
    print(f"=== {name}  fused={bool(args.fused)} ===")
    print(f"shapes: {len(shapes)} matrices, {n_params/1e6:.1f}M params "
          f"(hidden={args.hidden} inter={args.inter} layers={args.layers})")

    torch.manual_seed(0)
    params = []
    for i, (r, c) in enumerate(shapes):
        p = torch.randn(r, c, device=dev, dtype=torch.bfloat16) * 0.02
        p.requires_grad_(True)
        params.append((f"layer{i}", p))
    grads = [torch.randn(r, c, device=dev, dtype=torch.bfloat16) * 1e-3 for r, c in shapes]

    coeffs = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

    for ns_steps in (5, 4, 3):
        opt = GefenMuon(params, lr=1e-3, ns_steps=ns_steps, fused=bool(args.fused),
                        adjust_lr_fn="match_rms_adamw")

        def full_step(opt=opt, params=params, grads=grads):
            for (_, p), g in zip(params, grads):
                p.grad = g
            opt.step()

        # prime state (codebook / period) before timing
        for _ in range(3):
            full_step()

        step_ms = cuda_time(full_step, args.iters, args.warmup)

        # NS-only: run the same NS over all matrices once per "step". Keep the
        # bf16 param dtype -- NS casts to bf16 internally, so .float() here would
        # charge the timer an fp32->bf16 cast the real step never pays.
        ns_inputs = [g.clone() for g in grads]

        def ns_only(ns_inputs=ns_inputs, coeffs=coeffs, ns_steps=ns_steps):
            for x in ns_inputs:
                _zeropower_via_newtonschulz(x, coeffs, ns_steps, 1e-7)

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
            takes_p = len(inspect.signature(mom_fn).parameters) == 4
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
        line = (f"ns_steps={ns_steps}: step={step_ms:7.3f} ms  "
                f"NS-only={ns_ms:7.3f} ms  NS≈{frac:4.1f}% of step  "
                f"(rest≈{step_ms-ns_ms:6.3f} ms)")
        if mom_ms is not None:
            mom_frac = 100.0 * mom_ms / step_ms if step_ms else 0.0
            line += f"  momentum-only={mom_ms:6.3f} ms ({mom_frac:4.1f}% of step)"
        print(line)


if __name__ == "__main__":
    main()
