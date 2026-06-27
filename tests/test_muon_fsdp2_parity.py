"""
FSDP2 (DTensor) parity for GefenMuon's Muon 2D-matrix path.

Under FSDP2 each rank holds only a row slice of a 2D weight, but Newton-Schulz
must orthogonalize the FULL matrix. GefenMuon._step_automatic all-gathers the
gradient to the full matrix, runs the quantized-momentum + Newton-Schulz pipeline
on the full matrix, then slices the orthogonalized update back to the local shard.

This test runs a real 2-rank process group (CPU / gloo, non-fused path -- no CUDA
kernels required) and asserts that, after one optimizer step, the gathered sharded
parameter matches a single-process plain-tensor reference within tolerance. It
covers three shardings on dim 0:

  - even   (M divisible by world_size)
  - uneven (M not divisible -> ranks own different row counts)
  - empty  (M < world_size -> at least one rank owns 0 rows)

The empty/uneven cases also prove no deadlock: every rank reaches the
full_tensor() all-gather (in codebook learning AND in the step) and the final
full_tensor() gather, regardless of whether its local shard is empty.

Run: python tests/test_muon_fsdp2_parity.py
"""
import os

import torch
import torch.nn as nn

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

LR = 5e-3
WD = 0.1
TOL = 1e-5

# (M, N) per case. world_size is always 2.
CASES = {
    "even": (8, 6),
    "uneven": (5, 4),
    "empty": (1, 4),
}


def _reference(full_init, full_grad):
    """Single-process plain-tensor GefenMuon: the single-GPU oracle."""
    from gefen.gefen_muon import GefenMuon

    p_ref = nn.Parameter(full_init.clone())
    p_ref.grad = full_grad.clone()
    opt = GefenMuon([("w", p_ref)], lr=LR, fused=False, weight_decay=WD)
    opt.step()
    return p_ref.detach()


def _worker(rank, world, case, q):
    # A spawned child inherits the parent's sys.path. Under pytest that can
    # include the worktree root, where a flat `gefen.py` shadows the installed
    # `gefen/` package and breaks `from gefen.X import ...`. Drop any path entry
    # that exposes such a shadowing module; the real package (a `gefen/` dir,
    # reached via PYTHONPATH) survives.
    import os as _os
    import sys as _sys

    _sys.path[:] = [
        pth
        for pth in _sys.path
        if not _os.path.isfile(_os.path.join(pth or ".", "gefen.py"))
    ]

    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    from gefen.gefen_muon import GefenMuon

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29613"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        M, N = CASES[case]

        # Identical full init + grad on every rank (same seed) so the gathered
        # full grad equals the reference's grad exactly.
        torch.manual_seed(1234)
        full_init = torch.randn(M, N, dtype=torch.float32)
        full_grad = torch.randn(M, N, dtype=torch.float32) * 0.01

        p = nn.Parameter(distribute_tensor(full_init.clone(), mesh, [Shard(0)]))
        p.grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
        local_rows = p.to_local().shape[0]

        opt = GefenMuon([("w", p)], lr=LR, fused=False, weight_decay=WD)
        opt.step()

        gathered = p.detach().full_tensor()  # collective; all ranks must call

        if rank == 0:
            ref = _reference(full_init, full_grad)
            changed = not torch.equal(gathered, full_init)
            finite = torch.isfinite(gathered).all().item()
            max_abs = (gathered - ref).abs().max().item()
            q.put(("result", case, local_rows, changed, finite, max_abs))
        else:
            q.put(("rank", case, rank, local_rows))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_case(case):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, 2, case, q)) for r in range(2)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join(timeout=180)

    results = {}
    rank_rows = {}
    while not q.empty():
        item = q.get()
        if item[0] == "result":
            results["main"] = item[1:]
        else:
            rank_rows[item[2]] = item[3]

    for pr in procs:
        assert pr.exitcode == 0, f"[{case}] worker exited with {pr.exitcode}"
    assert "main" in results, f"[{case}] no result from rank 0"

    _case, local_rows, changed, finite, max_abs = results["main"]
    print(
        f"[{case}] rank0_rows={local_rows} rank1_rows={rank_rows.get(1)} "
        f"changed={changed} finite={finite} max|dist-oracle|={max_abs:.3e}"
    )
    ok = changed and finite and (max_abs <= TOL)
    print(f"[{case}] {'PASS' if ok else 'FAIL'}")
    return ok


def case_fused_cuda():
    """Single-rank DTensor on CUDA, fused=True: exercises the is_sharded *fused*
    branch (full-matrix scratch param for the momentum kernel + redistribute
    apply) and asserts parity with a fused plain-tensor oracle. world_size=1 so
    full_tensor()/redistribute are identities, but the sharded code path is taken
    in full -- which is what we want to cover on GPU."""
    import torch.distributed as dist
    from torch.distributed.tensor import (
        Shard,
        distribute_tensor,
        init_device_mesh,
    )

    from gefen.gefen_muon import GefenMuon

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29714")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cuda", (1,))
        torch.manual_seed(7)
        M, N = 64, 96
        init = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.1
        grad = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 0.01

        p = nn.Parameter(distribute_tensor(init.clone(), mesh, [Shard(0)]))
        p.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
        opt = GefenMuon([("w", p)], lr=LR, fused=True, weight_decay=WD)
        opt.step()
        got = p.detach().full_tensor()

        p_ref = nn.Parameter(init.clone())
        p_ref.grad = grad.clone()
        opt_ref = GefenMuon([("w", p_ref)], lr=LR, fused=True, weight_decay=WD)
        opt_ref.step()

        changed = not torch.equal(got, init)
        max_abs = (got.float() - p_ref.detach().float()).abs().max().item()
        ok = changed and (max_abs <= TOL)
        print(f"[fused_cuda] changed={changed} max|dist-oracle|={max_abs:.3e} "
              f"{'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run():
    results = {c: _run_case(c) for c in CASES}
    if torch.cuda.is_available():
        results["fused_cuda"] = case_fused_cuda()
    return all(results.values())


if pytest is not None:

    @pytest.mark.parametrize("case", list(CASES))
    def test_muon_fsdp2_parity(case):
        assert _run_case(case)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="fused Muon path requires a CUDA device",
    )
    def test_muon_fsdp2_fused_cuda_parity():
        assert case_fused_cuda()


if __name__ == "__main__":
    ok = run()
    print("\nMUON FSDP2 PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
