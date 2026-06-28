"""2-GPU FSDP2 (DTensor) harness for GefenMuon: parity + step timing.

Under FSDP2 each rank holds a row shard (Shard(0)) of every 2D weight. GefenMuon
all-gathers the gradient to the full matrix per param (full_tensor()), runs the
quantized-momentum + Newton-Schulz pipeline on the full matrix on every rank
(single-GPU parity), then slices the orthogonalized update back to the local
shard (_shard_like). distribute_tensor(Shard(0)) produces exactly the DTensor
params fully_shard creates, so the optimizer code path exercised here is
identical to a real fully_shard model (the optimizer never sees the module).

This harness:
  (a) validates the gathered sharded parameter matches a single-GPU plain-tensor
      reference after N steps (tight tolerance), and
  (b) CUDA-event times the optimizer step, decomposed into:
        - gather-only   : sum of grad.full_tensor()+wait over all muon params
        - NS-only       : Newton-Schulz over the full matrices
        - full step     : opt.step() end-to-end
      (apply+momentum is inferred as step - gather - NS).

Run (2x 3090 Ti at nvidia-smi indices 1,7):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,7 \
  TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda-13.3 \
  PYTHONPATH=<symlink-dir> \
  torchrun --nproc_per_node=2 bench_muon_fsdp2.py [--fused 1] [--steps 20] \
           [--approx 0]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist


def _strip_shadowing_gefen():
    # A flat `gefen.py` on sys.path shadows the installed `gefen/` package and
    # breaks `from gefen.X import ...`; drop any such entry (the real package,
    # reached via the PYTHONPATH symlink, survives).
    sys.path[:] = [
        p for p in sys.path
        if not os.path.isfile(os.path.join(p or ".", "gefen.py"))
    ]


_strip_shadowing_gefen()

from gefen.gefen_muon import (  # noqa: E402
    GefenMuon,
    _zeropower_via_newtonschulz,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_C,
)

LR = 5e-3
WD = 0.1


def make_shapes(hidden, inter, n_layers, n_kv_groups=4):
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
    dist.barrier()
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
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--kv-groups", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--fused", type=int, default=1)
    ap.add_argument("--ns-steps", type=int, default=5)
    ap.add_argument("--approx", type=int, default=0,
                    help="use approximate sharded Muon (non-parity)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    mesh = init_device_mesh("cuda", (world,))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    shapes = make_shapes(args.hidden, args.inter, args.layers, args.kv_groups)
    n_params = sum(r * c for r, c in shapes)
    if rank == 0:
        print(f"=== {torch.cuda.get_device_name(0)} x{world}  "
              f"fused={bool(args.fused)} approx={bool(args.approx)} "
              f"dtype={args.dtype} ns_steps={args.ns_steps} ===")
        print(f"shapes: {len(shapes)} matrices, {n_params/1e6:.1f}M params "
              f"(hidden={args.hidden} inter={args.inter} layers={args.layers})")

    # Identical full init + grad on every rank (same seed) so the gathered full
    # grad equals the single-GPU reference's grad exactly.
    torch.manual_seed(0)
    full_inits = [(torch.randn(r, c, device=dev, dtype=dtype) * 0.02)
                  for r, c in shapes]
    full_grads = [(torch.randn(r, c, device=dev, dtype=dtype) * 1e-3)
                  for r, c in shapes]

    params = []
    for i, w in enumerate(full_inits):
        import torch.nn as nn
        p = nn.Parameter(distribute_tensor(w.clone(), mesh, [Shard(0)]))
        params.append((f"layer{i}", p))

    kwargs = dict(lr=LR, ns_steps=args.ns_steps, fused=bool(args.fused),
                  weight_decay=WD, adjust_lr_fn="match_rms_adamw")
    if args.approx:
        kwargs["sharded_mode"] = "approx"
    opt = GefenMuon(params, **kwargs)

    def set_grads():
        for (_, p), g in zip(params, full_grads):
            p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])

    def full_step():
        set_grads()
        opt.step()

    # ---- timing ----
    step_ms = cuda_time(full_step, args.iters, args.warmup)

    # The gather-only / NS-only breakdown is EXACT-mode only (approx has no
    # all-gather and runs NS on local shards), so skip this discarded work in
    # approx mode rather than paying the exact-mode cost on every approx run.
    gather_ms = ns_ms = 0.0
    if not args.approx:
        # gather-only: pure per-param all-gather (full_tensor) over pre-sharded
        # grads -- isolates the serial collective cost from distribute/compute.
        sharded_grads = [distribute_tensor(g.clone(), mesh, [Shard(0)])
                         for g in full_grads]

        def gather_only():
            for dg in sharded_grads:
                fg = dg.full_tensor()
                if hasattr(fg, "wait"):
                    fg = fg.wait()
        gather_ms = cuda_time(gather_only, args.iters, args.warmup)

        # NS-only over full matrices (rank-redundant compute). Keep the param
        # dtype (bf16) -- NS casts to bf16 internally, so .float() here would
        # charge the timer an fp32->bf16 cast the real step never pays.
        ns_inputs = [g.clone() for g in full_grads]

        def ns_only():
            for x in ns_inputs:
                _zeropower_via_newtonschulz(
                    x, (DEFAULT_A, DEFAULT_B, DEFAULT_C), args.ns_steps, 1e-7)
        ns_ms = cuda_time(ns_only, args.iters, args.warmup)

    if rank == 0:
        if args.approx:
            # gather_only / ns_only measure the EXACT full-matrix costs, which do
            # not apply to approx (no gather; NS on local shards) -- so only the
            # end-to-end step is meaningful here.
            print(f"step={step_ms:8.3f} ms  (approx: no all-gather, "
                  f"NS on local shard; full-matrix breakdown N/A)")
        else:
            rest = step_ms - gather_ms - ns_ms
            print(f"step={step_ms:8.3f} ms | gather={gather_ms:8.3f} ms "
                  f"({100*gather_ms/step_ms:4.1f}%) | NS={ns_ms:8.3f} ms "
                  f"({100*ns_ms/step_ms:4.1f}%) | rest(momentum+apply)~{rest:7.3f} ms")

    # ---- parity vs single-GPU reference (skip in approx mode) ----
    # Fresh optimizer + state, one deterministic step, compare gathered param.
    dist.barrier()
    torch.manual_seed(0)
    p_inits = [(torch.randn(r, c, device=dev, dtype=dtype) * 0.02)
               for r, c in shapes]
    p_grads = [(torch.randn(r, c, device=dev, dtype=dtype) * 1e-3)
               for r, c in shapes]
    import torch.nn as nn
    sharded = [nn.Parameter(distribute_tensor(w.clone(), mesh, [Shard(0)]))
               for w in p_inits]
    named = [(f"L{i}", p) for i, p in enumerate(sharded)]
    kwargs2 = dict(lr=LR, ns_steps=args.ns_steps, fused=bool(args.fused),
                   weight_decay=WD, adjust_lr_fn="match_rms_adamw")
    if args.approx:
        kwargs2["sharded_mode"] = "approx"
    opt2 = GefenMuon(named, **kwargs2)
    nsteps = 3
    for _ in range(nsteps):
        for (_, p), g in zip(named, p_grads):
            p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
        opt2.step()
    gathered = []
    for p in sharded:
        g = p.detach().full_tensor()
        if hasattr(g, "wait"):  # mirror gather_only(): resolve any async gather
            g = g.wait()
        gathered.append(g)

    if rank == 0:
        # Single-GPU oracle: ONE optimizer over ALL full params (Gefen's codebook
        # is global per-optimizer, so a per-matrix optimizer would not match the
        # single sharded optimizer that owns every matrix).
        refs = [nn.Parameter(p_inits[i].clone()) for i in range(len(shapes))]
        named_ref = [(f"L{i}", refs[i]) for i in range(len(shapes))]
        opt_ref = GefenMuon(named_ref, lr=LR, ns_steps=args.ns_steps,
                            fused=bool(args.fused), weight_decay=WD,
                            adjust_lr_fn="match_rms_adamw")
        for _ in range(nsteps):
            for i, pr in enumerate(refs):
                pr.grad = p_grads[i].clone()
            opt_ref.step()
        max_abs = max(
            (gathered[i].float() - refs[i].detach().float()).abs().max().item()
            for i in range(len(shapes))
        )
        tag = "approx(non-parity)" if args.approx else "exact"
        print(f"parity[{tag}] vs single-GPU after {nsteps} steps: "
              f"max|dist-oracle|={max_abs:.3e}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
