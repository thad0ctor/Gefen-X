"""Speed + quality of the fp8 (e4m3) Newton-Schulz path vs bf16.

Runs the ACTUAL NS kernel (`_zeropower_via_newtonschulz`) over the real Muon
weight shapes and reports, per shape:

  * SPEED: CUDA-event ms per NS call for the production bf16 path (eager addmm)
    vs the fp8 path (`use_fp8=True`, torch.compile-fused -- the win only
    materializes once the per-matmul quant is fused, which compile does).
  * QUALITY: orthogonality error ||G - I||_F (fp32 metric) of the fp32 reference
    NS, the bf16 output, and the fp8 output, plus the relative deviation of the
    fp8 output from bf16  ||O_fp8 - O_bf16||_F / ||O_bf16||_F.

fp8 tensor-core GEMMs need sm_89+ (Ada / Hopper / Blackwell) AND a large enough
matrix (smaller dim >= FP8_MIN_DIM); on anything else the kernel transparently
falls back to bf16, so shapes that fall back show bf16==fp8 (flagged below).

Run (Blackwell / Ada / Hopper only for a real fp8 measurement):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  python bench_ns_fp8.py [--json out.json]
"""
import argparse
import json
import os
import sys

import torch


def _strip_shadowing_gefen():
    sys.path[:] = [
        p for p in sys.path
        if not os.path.isfile(os.path.join(p or ".", "gefen.py"))
    ]


_strip_shadowing_gefen()

from gefen.gefen_muon import (  # noqa: E402
    _zeropower_via_newtonschulz,
    _fp8_supported, FP8_MIN_DIM,
    DEFAULT_A, DEFAULT_B, DEFAULT_C,
)

STD5 = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

SHAPES = [
    ("q/k/v/o 2048x2048", 2048, 2048),
    ("kv 512x2048", 512, 2048),
    ("gate/up 6144x2048", 6144, 2048),
    ("down 2048x6144", 2048, 6144),
]


def ortho_error(O):
    """||G - I||_F on the smaller dimension, computed in fp32."""
    O = O.float()
    r, c = O.shape
    G = O @ O.t() if r <= c else O.t() @ O
    m = G.shape[0]
    I = torch.eye(m, device=G.device, dtype=G.dtype)
    return (G - I).norm().item()


def ns_fp32_ref(X, coeffs, nsteps, eps):
    """fp32 Newton-Schulz reference (same recurrence, full precision)."""
    a, b, c = coeffs
    O = X.float()
    transposed = O.size(0) > O.size(1)
    if transposed:
        O = O.T
    O = O / O.norm().clamp(min=eps)
    for _ in range(nsteps):
        G = O @ O.T
        O = a * O + (b * G + c * (G @ G)) @ O
    return O.T if transposed else O


def cuda_time(fn, iters=50, warmup=15):
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
    ap.add_argument("--json", default=None,
                    help="write structured speed+quality results here")
    ap.add_argument("--ns-steps", type=int, default=5)
    args = ap.parse_args()

    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    supported = _fp8_supported()
    print(f"=== fp8 vs bf16 Newton-Schulz on {name} (sm_{cap[0]}{cap[1]}) ===")
    print(f"fp8 GEMM supported: {supported}   FP8_MIN_DIM gate: {FP8_MIN_DIM}\n")
    if not supported:
        print("WARNING: this GPU has no fp8 GEMM; the fp8 path falls back to "
              "bf16 (numbers below will show no win). Use sm_89+ for real data.\n")

    eps = 1e-7
    ns = args.ns_steps
    data = {"device": name, "sm": f"{cap[0]}{cap[1]}",
            "fp8_supported": bool(supported), "fp8_min_dim": FP8_MIN_DIM,
            "ns_steps": ns, "shapes": {}}

    # ---------- QUALITY ----------
    print("QUALITY: ||G-I||_F (fp32 metric) for fp32-ref / bf16 / fp8, and the "
          "fp8-vs-bf16 relative deviation\n")
    header = (f"{'shape':22s} {'gated':>6s} {'fp32':>9s} {'bf16':>9s} "
              f"{'fp8':>9s} {'rel(fp8-bf16)':>14s}")
    print(header)
    print("-" * len(header))
    for label, r, c in SHAPES:
        gated_off = (not supported) or (min(r, c) < FP8_MIN_DIM)
        g = torch.Generator(device=dev).manual_seed(1234)
        X = torch.randn(r, c, device=dev, dtype=torch.bfloat16, generator=g) * 1e-3
        O_ref = ns_fp32_ref(X, STD5, ns, eps)
        O_bf16 = _zeropower_via_newtonschulz(X, STD5, ns, eps, use_fp8=False)
        O_fp8 = _zeropower_via_newtonschulz(
            X, STD5, ns, eps, use_fp8=True, compile_fp8=True)
        e_ref, e_bf16, e_fp8 = (ortho_error(O_ref), ortho_error(O_bf16),
                                ortho_error(O_fp8))
        rel = ((O_fp8.float() - O_bf16.float()).norm()
               / O_bf16.float().norm().clamp(min=eps)).item()
        flag = "bf16" if gated_off else "fp8"
        print(f"{label:22s} {flag:>6s} {e_ref:9.4f} {e_bf16:9.4f} "
              f"{e_fp8:9.4f} {rel*100:13.2f}%")
        data["shapes"].setdefault(label, {"rows": r, "cols": c})
        data["shapes"][label].update({
            "gated_off": bool(gated_off),
            "ortho_err": {"fp32": e_ref, "bf16": e_bf16, "fp8": e_fp8},
            "rel_fp8_bf16": rel,
        })

    # ---------- SPEED ----------
    print("\nSPEED: CUDA-event ms per NS call (warmup excluded). bf16 = eager "
          "production path; fp8 = torch.compile-fused\n")
    print(f"{'shape':22s} {'gated':>6s} {'bf16':>9s} {'fp8':>9s} {'fp8 spd':>8s}")
    print("-" * 60)
    for label, r, c in SHAPES:
        gated_off = (not supported) or (min(r, c) < FP8_MIN_DIM)
        g = torch.Generator(device=dev).manual_seed(42)
        X = torch.randn(r, c, device=dev, dtype=torch.bfloat16, generator=g) * 1e-3
        t_bf16 = cuda_time(
            lambda: _zeropower_via_newtonschulz(X, STD5, ns, eps, use_fp8=False))
        t_fp8 = cuda_time(
            lambda: _zeropower_via_newtonschulz(
                X, STD5, ns, eps, use_fp8=True, compile_fp8=True))
        flag = "bf16" if gated_off else "fp8"
        print(f"{label:22s} {flag:>6s} {t_bf16:9.4f} {t_fp8:9.4f} "
              f"{t_bf16/t_fp8:7.2f}x")
        data["shapes"][label]["speed_ms"] = {"bf16": t_bf16, "fp8": t_fp8}

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
