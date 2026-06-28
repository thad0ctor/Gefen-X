"""Quality + speed of tuned Newton-Schulz schedules vs the standard 5-step.

Runs the ACTUAL bf16 NS kernel (`_zeropower_via_newtonschulz`) over the real
Muon weight shapes and reports, per shape:

  * QUALITY: orthogonality error ||G - I||_F where G is the Gram on the smaller
    dimension (G = O O^T if rows<=cols else O^T O), measured in fp32, plus the
    output singular-value min/max/mean (perfect orthogonalization -> all 1).
  * SAFETY: a near-low-rank adversarial input (normalized top singular value ~1)
    -- checks the schedule does not blow up (no NaN/Inf, bounded output).
  * SPEED: CUDA-event ms per NS call for each schedule.

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
  python bench_ns_schedule.py [--json out.json]

Pass --json to also dump the per-shape quality + speed numbers as a structured
file that plot_ns_schedule.py renders into the committed PNGs.
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
    DEFAULT_A, DEFAULT_B, DEFAULT_C,
    NS_SCHEDULE_3STEP, NS_SCHEDULE_4STEP,
)

STD5 = (DEFAULT_A, DEFAULT_B, DEFAULT_C)

SHAPES = [
    ("q/k/v/o 2048x2048", 2048, 2048),
    ("kv 512x2048", 512, 2048),
    ("gate/up 6144x2048", 6144, 2048),
    ("down 2048x6144", 2048, 6144),
]

SCHEDULES = [
    ("std-5", STD5, 5),
    ("tuned-3", NS_SCHEDULE_3STEP, None),
    ("tuned-4", NS_SCHEDULE_4STEP, None),
]


def ortho_error(O):
    """||G - I||_F on the smaller dimension, computed in fp32."""
    O = O.float()
    r, c = O.shape
    if r <= c:
        G = O @ O.t()
    else:
        G = O.t() @ O
    m = G.shape[0]
    I = torch.eye(m, device=G.device, dtype=G.dtype)
    return (G - I).norm().item()


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
                    help="write structured quality+speed results here")
    args = ap.parse_args()

    dev = "cuda"
    name = torch.cuda.get_device_name(0)
    print(f"=== Newton-Schulz schedule quality + speed on {name} ===\n")
    seeds = [0, 1, 2, 3, 4]
    data = {"device": name, "seeds": seeds, "shapes": {}}

    # ---------- QUALITY ----------
    print("QUALITY: orthogonality error ||G-I||_F (mean +/- std over seeds), "
          "fp32 metric; output singular-value [min,mean,max]\n")
    header = f"{'shape':22s} {'schedule':9s} {'||G-I||_F':>12s}  {'sv[min/mean/max]':>26s}"
    print(header)
    print("-" * len(header))
    quality = {}
    for label, r, c in SHAPES:
        for sname, coeffs, nsteps in SCHEDULES:
            errs = []
            svmins, svmeans, svmaxs = [], [], []
            for sd in seeds:
                g = torch.Generator(device=dev).manual_seed(1000 + sd)
                X = torch.randn(r, c, device=dev, dtype=torch.bfloat16,
                                generator=g) * 1e-3
                ns = nsteps if nsteps is not None else len(coeffs)
                O = _zeropower_via_newtonschulz(X, coeffs, ns, 1e-7)
                errs.append(ortho_error(O))
                sv = torch.linalg.svdvals(O.float())
                svmins.append(sv.min().item())
                svmeans.append(sv.mean().item())
                svmaxs.append(sv.max().item())
            import statistics as st
            em, es = st.mean(errs), (st.pstdev(errs))
            quality[(label, sname)] = em
            d = data["shapes"].setdefault(label, {"rows": r, "cols": c})
            d.setdefault("quality", {})[sname] = {
                "ortho_err_mean": em, "ortho_err_std": es,
                "sv_min": min(svmins), "sv_mean": st.mean(svmeans),
                "sv_max": max(svmaxs),
            }
            print(f"{label:22s} {sname:9s} {em:9.4f}+/-{es:6.4f}  "
                  f"[{min(svmins):.3f}/{st.mean(svmeans):.3f}/{max(svmaxs):.3f}]")
        # relative vs std-5
        base = quality[(label, 'std-5')]
        t3 = quality[(label, 'tuned-3')]
        t4 = quality[(label, 'tuned-4')]
        print(f"{'':22s} {'-> vs std-5: tuned-3 ' + f'{t3/base:.2f}x':32s} "
              f"tuned-4 {t4/base:.2f}x  (lower=better)\n")

    # ---------- SAFETY: near-low-rank adversarial input ----------
    print("SAFETY: near-low-rank input (normalized top sv ~1). "
          "max|O| should stay bounded (no blow-up); finite check.\n")
    print(f"{'schedule':9s} {'max|O|':>10s} {'finite':>8s} {'||G-I||_F':>12s}")
    print("-" * 42)
    r, c = 2048, 2048
    g = torch.Generator(device=dev).manual_seed(7)
    u = torch.randn(r, 1, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, c, device=dev, dtype=torch.bfloat16, generator=g)
    noise = torch.randn(r, c, device=dev, dtype=torch.bfloat16, generator=g)
    X_lr = (u @ v) + 0.01 * noise  # dominant rank-1 + small noise
    for sname, coeffs, nsteps in SCHEDULES:
        ns = nsteps if nsteps is not None else len(coeffs)
        O = _zeropower_via_newtonschulz(X_lr, coeffs, ns, 1e-7)
        finite = bool(torch.isfinite(O).all().item())
        maxo = O.float().abs().max().item()
        err = ortho_error(O) if finite else float('nan')
        print(f"{sname:9s} {maxo:10.3f} {str(finite):>8s} {err:12.4f}")

    # ---------- SPEED ----------
    print("\nSPEED: CUDA-event ms per NS call (warmup excluded)\n")
    print(f"{'shape':22s} {'std-5':>9s} {'tuned-3':>9s} {'tuned-4':>9s}  "
          f"{'t3 spd':>7s} {'t4 spd':>7s}")
    print("-" * 72)
    for label, r, c in SHAPES:
        g = torch.Generator(device=dev).manual_seed(42)
        X = torch.randn(r, c, device=dev, dtype=torch.bfloat16, generator=g) * 1e-3
        times = {}
        for sname, coeffs, nsteps in SCHEDULES:
            ns = nsteps if nsteps is not None else len(coeffs)
            times[sname] = cuda_time(
                lambda c=coeffs, n=ns: _zeropower_via_newtonschulz(X, c, n, 1e-7))
        b = times['std-5']
        d = data["shapes"].setdefault(label, {"rows": r, "cols": c})
        d["speed_ms"] = dict(times)
        print(f"{label:22s} {b:9.4f} {times['tuned-3']:9.4f} "
              f"{times['tuned-4']:9.4f}  {b/times['tuned-3']:6.2f}x "
              f"{b/times['tuned-4']:6.2f}x")

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
