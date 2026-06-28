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
import queue
import socket

import torch
import torch.distributed  # noqa: F401  (availability checks at import time)
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


def _get_free_port():
    """A free localhost port, bound per-case to avoid rendezvous collisions
    under parallel pytest workers or after a stale failed run."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _worker(rank, world, case, port, q):
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
    os.environ["MASTER_PORT"] = port
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
    port = _get_free_port()
    procs = [ctx.Process(target=_worker, args=(r, 2, case, port, q)) for r in range(2)]
    for pr in procs:
        pr.start()

    # Drain the two expected messages with a timeout. Queue.empty() is
    # unreliable across processes (it can miss rank 1 and leave rank_rows unset),
    # and reading the pipe before join() avoids a large put() blocking the child.
    results = {}
    rank_rows = {}
    for _ in range(2):
        try:
            item = q.get(timeout=180)
        except queue.Empty:
            break
        if item[0] == "result":
            results["main"] = item[1:]
        else:
            rank_rows[item[2]] = item[3]

    for pr in procs:
        pr.join(timeout=180)
    # A deadlocked rank leaves exitcode=None; terminate it so the test session
    # can't hang on the assertion below.
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
            pr.join(timeout=10)

    for pr in procs:
        assert pr.exitcode == 0, f"[{case}] worker exited with {pr.exitcode}"
    assert "main" in results, f"[{case}] no result from rank 0"
    assert 1 in rank_rows, f"[{case}] no shard metadata from rank 1"

    _case, local_rows, changed, finite, max_abs = results["main"]
    # Make the shard split explicit so the empty/uneven coverage can't silently
    # regress (rank0_rows, rank1_rows), world_size=2.
    expected_rows = {"even": (4, 4), "uneven": (3, 2), "empty": (1, 0)}[case]
    assert (local_rows, rank_rows[1]) == expected_rows, (
        f"[{case}] shard rows {(local_rows, rank_rows[1])} != {expected_rows}"
    )
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
    # Runs in the main process: drop any sys.path entry exposing a flat `gefen.py`
    # that would shadow the installed `gefen/` package (same guard as _worker).
    import os as _os
    import sys as _sys

    # Scope the mutation: this runs in the pytest process, so restore sys.path
    # afterwards to avoid leaking into later tests.
    _original_sys_path = _sys.path[:]
    try:
        _sys.path[:] = [
            pth
            for pth in _sys.path
            if not _os.path.isfile(_os.path.join(pth or ".", "gefen.py"))
        ]

        import torch.distributed as dist
        from torch.distributed.tensor import (
            Shard,
            distribute_tensor,
            init_device_mesh,
        )

        from gefen.gefen_muon import GefenMuon
    finally:
        _sys.path[:] = _original_sys_path

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


# Multi-rank fused-CUDA parity: closes the gap that the only >1-rank coverage was
# CPU/non-fused and the only fused coverage was world=1. Runs the *fused* Muon
# kernel under a real 2-GPU shard and checks the gathered param against a fused
# single-GPU oracle. The full-matrix gather is exact and the shard-slice is
# byte-identical, so a single sharded matrix should match the oracle tightly; the
# tolerance is a touch looser than the world=1 1e-5 only to absorb cross-rank bf16
# NS noise (a wrong shard-slice would error by O(1), not ~1e-3).
MULTI_FUSED_TOL = 1e-3
# (M, N) on dim-0 shard; world_size=2. "even" splits 32/32, "uneven" splits 33/32
# (exercises the ceil shard-slice arithmetic on the fused path).
FUSED_CASES = {"even": (64, 96), "uneven": (65, 96)}


def _worker_fused_cuda(rank, world, case, port, q):
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
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cuda", (world,))
        M, N = FUSED_CASES[case]
        # identical full init+grad on every rank (same seed) so the gathered full
        # grad equals the single-GPU oracle's exactly
        torch.manual_seed(1234)
        full_init = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.1)
        full_grad = (torch.randn(M, N, device=dev, dtype=torch.bfloat16) * 0.01)

        p = nn.Parameter(distribute_tensor(full_init.clone(), mesh, [Shard(0)]))
        p.grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
        local_rows = p.to_local().shape[0]

        opt = GefenMuon([("w", p)], lr=LR, fused=True, weight_decay=WD)
        opt.step()
        gathered = p.detach().full_tensor()  # collective; every rank must call

        if rank == 0:
            p_ref = nn.Parameter(full_init.clone())
            p_ref.grad = full_grad.clone()
            opt_ref = GefenMuon([("w", p_ref)], lr=LR, fused=True, weight_decay=WD)
            opt_ref.step()
            changed = not torch.equal(gathered, full_init)
            finite = torch.isfinite(gathered).all().item()
            max_abs = (gathered.float() - p_ref.detach().float()).abs().max().item()
            q.put(("result", case, local_rows, changed, finite, max_abs))
        else:
            q.put(("rank", case, rank, local_rows))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_fused_cuda_case(case):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(target=_worker_fused_cuda, args=(r, 2, case, port, q))
        for r in range(2)
    ]
    for pr in procs:
        pr.start()

    results, rank_rows = {}, {}
    for _ in range(2):
        try:
            item = q.get(timeout=300)
        except queue.Empty:
            break
        if item[0] == "result":
            results["main"] = item[1:]
        else:
            rank_rows[item[2]] = item[3]

    for pr in procs:
        pr.join(timeout=300)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
            pr.join(timeout=10)

    for pr in procs:
        assert pr.exitcode == 0, f"[fused {case}] worker exited with {pr.exitcode}"
    assert "main" in results, f"[fused {case}] no result from rank 0"
    assert 1 in rank_rows, f"[fused {case}] no shard metadata from rank 1"

    _case, local_rows, changed, finite, max_abs = results["main"]
    expected_rows = {"even": (32, 32), "uneven": (33, 32)}[case]
    assert (local_rows, rank_rows[1]) == expected_rows, (
        f"[fused {case}] shard rows {(local_rows, rank_rows[1])} != {expected_rows}"
    )
    ok = changed and finite and (max_abs <= MULTI_FUSED_TOL)
    print(
        f"[fused {case}] rank0_rows={local_rows} rank1_rows={rank_rows.get(1)} "
        f"changed={changed} finite={finite} max|dist-oracle|={max_abs:.3e} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


# ---------------------------------------------------------------------------
# "distributed" mode parity ("Parallel Muon"): each 2D matrix is round-robin
# assigned to ONE owner rank that runs Newton-Schulz on the full matrix and
# broadcasts the orthogonalized update. This must be EXACT vs (a) the single-GPU
# oracle and (b) the existing "exact" mode (which runs NS redundantly on every
# rank). The test uses SEVERAL matrices so the round-robin owner actually cycles
# over ranks, runs MULTIPLE steps (so each owner's momentum state must persist),
# and deliberately includes a matrix whose owner rank holds an EMPTY shard (owner
# still all-gathers the full grad, runs NS, and broadcasts) plus matrices smaller
# than world_size (empty non-owner shards) -- proving no collective deadlocks.
DIST_TOL = 1e-3  # absorbs cross-rank bf16 NS noise vs the single-GPU oracle
DIST_EXACT_TOL = 0.0  # distributed must be BIT-IDENTICAL to exact mode
DIST_STEPS = 3


def _dist_shapes(world):
    # Shapes chosen so round-robin owner = idx % world hits, in order:
    #   idx0 -> rank0 (normal), idx1 -> rank1 (normal),
    #   idx2 -> rank2%world (normal-ish), idx3 -> rank3%world owns an EMPTY shard
    # The (1, 8) matrix has a single row, so every rank except rank0 owns 0 rows;
    # placing it at idx3 makes rank (3 % world) an empty OWNER when world>1.
    return [(64, 96), (48, 64), (65, 96), (1, 8), (2, 12)]


def _worker_distributed(rank, world, port, q, ns_schedule=None):
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
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cuda", (world,))
        shapes = _dist_shapes(world)

        def make_grads(step):
            # Identical full grads on every rank (same seed) per step.
            g = torch.Generator(device="cpu").manual_seed(100 + step)
            return [
                (torch.randn(M, N, generator=g, dtype=torch.float32) * 0.01).to(
                    dev, torch.bfloat16
                )
                for (M, N) in shapes
            ]

        torch.manual_seed(1234)
        inits = [
            (torch.randn(M, N, dtype=torch.float32) * 0.1).to(dev, torch.bfloat16)
            for (M, N) in shapes
        ]

        def build(mode):
            params = []
            for i, init in enumerate(inits):
                params.append(
                    (
                        f"w{i}",
                        nn.Parameter(distribute_tensor(init.clone(), mesh, [Shard(0)])),
                    )
                )
            opt = GefenMuon(
                params,
                lr=LR,
                fused=True,
                weight_decay=WD,
                sharded_mode=mode,
                ns_schedule=ns_schedule,
            )
            return params, opt

        ex_params, ex_opt = build("exact")
        di_params, di_opt = build("distributed")

        owner_empty_seen = False
        for step in range(DIST_STEPS):
            grads = make_grads(step)
            for (_, p), g in zip(ex_params, grads):
                p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
            for (_, p), g in zip(di_params, grads):
                p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)])
            ex_opt.step()
            di_opt.step()

        # Record whether this rank was an empty OWNER for some matrix (idx3 -> the
        # (1,8) row matrix; rank (3%world) owns it and, for world>1, owns 0 rows).
        owner_idx3 = 3 % world
        if rank == owner_idx3:
            local_rows_idx3 = di_params[3][1].to_local().shape[0]
            owner_empty_seen = local_rows_idx3 == 0

        ex_gathered = [p.detach().full_tensor() for _, p in ex_params]
        di_gathered = [p.detach().full_tensor() for _, p in di_params]

        if rank == 0:
            # Single-GPU oracle: plain GefenMuon, same inits + grads, no mesh.
            ref_params = [
                (f"w{i}", nn.Parameter(init.clone())) for i, init in enumerate(inits)
            ]
            ref_opt = GefenMuon(
                ref_params,
                lr=LR,
                fused=True,
                weight_decay=WD,
                ns_schedule=ns_schedule,
            )
            for step in range(DIST_STEPS):
                grads = make_grads(step)
                for (_, p), g in zip(ref_params, grads):
                    p.grad = g.clone()
                ref_opt.step()
            ref = [p.detach() for _, p in ref_params]

            max_vs_oracle = max(
                (di_gathered[i].float() - ref[i].float()).abs().max().item()
                for i in range(len(shapes))
            )
            max_vs_exact = max(
                (di_gathered[i].float() - ex_gathered[i].float()).abs().max().item()
                for i in range(len(shapes))
            )
            changed = all(
                not torch.equal(di_gathered[i], inits[i]) for i in range(len(shapes))
            )
            finite = all(torch.isfinite(g).all().item() for g in di_gathered)
            q.put(
                ("result", max_vs_oracle, max_vs_exact, changed, finite)
            )
        else:
            q.put(("owner_empty", rank, owner_empty_seen))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_case(world, ns_schedule=None):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(
            target=_worker_distributed, args=(r, world, port, q, ns_schedule)
        )
        for r in range(world)
    ]
    for pr in procs:
        pr.start()

    result = None
    owner_empty = {}
    for _ in range(world):
        try:
            item = q.get(timeout=300)
        except queue.Empty:
            break
        if item[0] == "result":
            result = item[1:]
        else:
            owner_empty[item[1]] = item[2]

    for pr in procs:
        pr.join(timeout=300)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
            pr.join(timeout=10)

    for pr in procs:
        assert pr.exitcode == 0, f"[dist x{world}] worker exited with {pr.exitcode}"
    assert result is not None, f"[dist x{world}] no result from rank 0"

    max_vs_oracle, max_vs_exact, changed, finite = result
    # For world>1, rank (3%world) owns the (1,8) matrix and holds 0 rows -> proves
    # an empty owner still gathers+NS+broadcasts without deadlock.
    empty_owner_rank = 3 % world
    empty_owner_ok = (
        empty_owner_rank == 0 or owner_empty.get(empty_owner_rank, False)
    )
    ok = (
        changed
        and finite
        and (max_vs_oracle <= DIST_TOL)
        and (max_vs_exact <= DIST_EXACT_TOL)
        and empty_owner_ok
    )
    sched_label = ns_schedule if ns_schedule is not None else "standard"
    print(
        f"[dist x{world} sched={sched_label}] changed={changed} finite={finite} "
        f"max|dist-oracle|={max_vs_oracle:.3e} max|dist-exact|={max_vs_exact:.3e} "
        f"empty_owner(rank{empty_owner_rank})={'n/a' if empty_owner_rank==0 else owner_empty.get(empty_owner_rank)} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def run():
    results = {c: _run_case(c) for c in CASES}
    if (
        torch.cuda.is_available()
        and torch.distributed.is_available()
        and torch.distributed.is_nccl_available()
    ):
        results["fused_cuda"] = case_fused_cuda()
        if torch.cuda.device_count() >= 2:
            for c in FUSED_CASES:
                results[f"fused_multirank_{c}"] = _run_fused_cuda_case(c)
            results["distributed_x2"] = _run_distributed_case(2)
            # Combined-lever coverage: distributed mode WITH the tuned4 schedule
            # must still be bit-identical to exact mode (also tuned4) and within
            # tol of the single-GPU tuned4 oracle.
            results["distributed_x2_tuned4"] = _run_distributed_case(2, "tuned4")
        if torch.cuda.device_count() >= 4:
            results["distributed_x4"] = _run_distributed_case(4)
            results["distributed_x4_tuned4"] = _run_distributed_case(4, "tuned4")
    return all(results.values())


if pytest is not None:

    @pytest.mark.parametrize("case", list(CASES))
    def test_muon_fsdp2_parity(case):
        assert _run_case(case)

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available(),
        reason="fused Muon path requires CUDA and NCCL",
    )
    def test_muon_fsdp2_fused_cuda_parity():
        assert case_fused_cuda()

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
        or torch.cuda.device_count() < 2,
        reason="multi-rank fused Muon path requires >=2 CUDA GPUs + NCCL",
    )
    @pytest.mark.parametrize("case", list(FUSED_CASES))
    def test_muon_fsdp2_fused_multirank_parity(case):
        assert _run_fused_cuda_case(case)

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
        or torch.cuda.device_count() < 2,
        reason="distributed Muon mode requires >=2 CUDA GPUs + NCCL",
    )
    @pytest.mark.parametrize(
        "world", [2] + ([4] if torch.cuda.device_count() >= 4 else [])
    )
    @pytest.mark.parametrize("ns_schedule", [None, "tuned4"])
    def test_muon_fsdp2_distributed_parity(world, ns_schedule):
        assert _run_distributed_case(world, ns_schedule)


if __name__ == "__main__":
    ok = run()
    print("\nMUON FSDP2 PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
