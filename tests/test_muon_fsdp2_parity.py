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

# (M, N) per case. world_size is always 2. The normuon-* cases re-run the
# even/uneven shardings with normuon=True (the GefenMuonHybrid default), so the
# per-row 2nd-moment state and the row-normalized update are covered under
# sharding: rank parity vs the single-process oracle must hold exactly like the
# plain path (the normalization runs on the full-matrix NS output every rank
# reconstructs, so it is deterministic and identical across ranks).
CASES = {
    "even": (8, 6),
    "uneven": (5, 4),
    "empty": (1, 4),
    "normuon-even": (8, 6),
    "normuon-uneven": (5, 4),
}


def _case_opt_kwargs(case):
    return {"normuon": True} if case.startswith("normuon-") else {}


def _reference(full_init, full_grad, opt_kwargs=None):
    """Single-process plain-tensor GefenMuon: the single-GPU oracle."""
    from gefen.gefen_muon import GefenMuon

    p_ref = nn.Parameter(full_init.clone())
    p_ref.grad = full_grad.clone()
    opt = GefenMuon(
        [("w", p_ref)], lr=LR, fused=False, weight_decay=WD, **(opt_kwargs or {})
    )
    opt.step()
    return p_ref.detach()


def _get_free_port():
    """A free localhost port, bound per-case to avoid rendezvous collisions
    under parallel pytest workers or after a stale failed run."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _worker(rank, world, case, port, q):
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

        opt = GefenMuon(
            [("w", p)], lr=LR, fused=False, weight_decay=WD,
            **_case_opt_kwargs(case),
        )
        opt.step()

        gathered = p.detach().full_tensor()  # collective; all ranks must call

        if rank == 0:
            ref = _reference(full_init, full_grad, _case_opt_kwargs(case))
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
    expected_rows = {"even": (4, 4), "uneven": (3, 2), "empty": (1, 0)}[
        case.replace("normuon-", "")
    ]
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
# "distributed" mode parity ("Parallel Muon"): each 2D matrix is assigned by
# stable full-param-set position to ONE owner rank that runs Newton-Schulz and
# broadcasts the orthogonalized update. This must be EXACT vs (a) the single-GPU
# oracle and (b) the existing "exact" mode (which runs NS redundantly on every
# rank). The test uses SEVERAL matrices so stable owners cover multiple ranks,
# runs MULTIPLE steps (so each owner's momentum state must persist),
# and deliberately includes a matrix whose owner rank holds an EMPTY shard (owner
# still all-gathers the full grad, runs NS, and broadcasts) plus matrices smaller
# than world_size (empty non-owner shards) -- proving no collective deadlocks.
DIST_TOL = 1e-3  # absorbs cross-rank bf16 NS noise vs the single-GPU oracle
DIST_EXACT_TOL = 0.0  # distributed must be BIT-IDENTICAL to exact mode
# ... but ONLY on a homogeneous rig: exact mode runs Newton-Schulz on every
# rank's own GPU while distributed broadcasts one owner's result, so with mixed
# GPU architectures (e.g. a 3090 + 5090 box) the cross-arch bf16 GEMMs differ at
# ULP scale and bit-identity is unattainable by design (verified: 4x3090 passes
# at 0.0, any 5090+3090 mix fails). On a mixed rig the exact-vs-distributed
# comparison is relaxed to this ULP-scale bound instead.
DIST_MIXED_ARCH_EXACT_TOL = 2e-3
DIST_STEPS = 3


def _cuda_archs_homogeneous(world):
    """True iff the first `world` visible GPUs share one compute capability."""
    return len({torch.cuda.get_device_capability(i) for i in range(world)}) == 1


def _dist_specs(world):
    # The (1, 8) matrix has a single row, so every rank except rank0 owns 0 rows.
    # Put it at stable full-param index 1, whose owner is rank1 for the 2/4-GPU
    # worlds this test uses, proving an empty owner still gathers+NS+broadcasts
    # without relying on active-gradient-list ownership.
    return [
        ("w0", 64, 96),
        ("empty_owner", 1, 8),
        ("w2", 65, 96),
        ("w3", 48, 64),
        ("w4", 2, 12),
    ]


def _worker_distributed(rank, world, port, q, ns_schedule=None):
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
        specs = _dist_specs(world)
        names = [name for name, _, _ in specs]
        shapes = [(M, N) for _, M, N in specs]

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
            for name, init in zip(names, inits):
                params.append(
                    (
                        name,
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

        # Record whether rank1 was the empty owner for the one-row matrix.
        empty_owner_idx = 1
        empty_owner_rank = 1
        if rank == empty_owner_rank:
            local_rows = di_params[empty_owner_idx][1].to_local().shape[0]
            owner_empty_seen = local_rows == 0

        ex_gathered = [p.detach().full_tensor() for _, p in ex_params]
        di_gathered = [p.detach().full_tensor() for _, p in di_params]

        if rank == 0:
            # Single-GPU oracle: plain GefenMuon, same inits + grads, no mesh.
            ref_params = [
                (name, nn.Parameter(init.clone())) for name, init in zip(names, inits)
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
    # Rank1 owns the (1,8) matrix by stable full-param index and holds 0 rows -> proves
    # an empty owner still gathers+NS+broadcasts without deadlock.
    empty_owner_rank = 1
    empty_owner_ok = owner_empty.get(empty_owner_rank, False)
    # Bit-identity vs exact mode is only enforceable on a homogeneous rig; on
    # mixed GPU archs relax to the ULP-scale bound (see DIST_MIXED_ARCH_EXACT_TOL).
    homogeneous = _cuda_archs_homogeneous(world)
    exact_tol = DIST_EXACT_TOL if homogeneous else DIST_MIXED_ARCH_EXACT_TOL
    if not homogeneous:
        caps = [torch.cuda.get_device_capability(i) for i in range(world)]
        print(
            f"[dist x{world}] MIXED GPU archs {caps}: exact<->distributed "
            f"bit-identity is unattainable cross-arch; relaxing the exact-mode "
            f"comparison to {DIST_MIXED_ARCH_EXACT_TOL:.0e}"
        )
    ok = (
        changed
        and finite
        and (max_vs_oracle <= DIST_TOL)
        and (max_vs_exact <= exact_tol)
        and empty_owner_ok
    )
    sched_label = ns_schedule if ns_schedule is not None else "standard"
    print(
        f"[dist x{world} sched={sched_label}] changed={changed} finite={finite} "
        f"max|dist-oracle|={max_vs_oracle:.3e} max|dist-exact|={max_vs_exact:.3e} "
        f"(tol={exact_tol:.0e}) "
        f"empty_owner(rank{empty_owner_rank})={'n/a' if empty_owner_rank==0 else owner_empty.get(empty_owner_rank)} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


def _active_for_fluctuating_step(step, idx):
    # Initialize every matrix, then vary the active set. The old positional owner
    # scheme shifted owners for later matrices on steps 1/2 and reset momentum.
    if step == 1 and idx == 0:
        return False
    if step == 2 and idx == 2:
        return False
    return True


def _worker_distributed_fluctuating(rank, world, port, q):
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
        specs = _dist_specs(world)
        names = [name for name, _, _ in specs]
        shapes = [(M, N) for _, M, N in specs]

        def make_grads(step):
            g = torch.Generator(device="cpu").manual_seed(500 + step)
            return [
                (torch.randn(M, N, generator=g, dtype=torch.float32) * 0.01).to(
                    dev, torch.bfloat16
                )
                for (M, N) in shapes
            ]

        torch.manual_seed(5678)
        inits = [
            (torch.randn(M, N, dtype=torch.float32) * 0.1).to(dev, torch.bfloat16)
            for (M, N) in shapes
        ]

        def build(mode):
            params = [
                (name, nn.Parameter(distribute_tensor(init.clone(), mesh, [Shard(0)])))
                for name, init in zip(names, inits)
            ]
            opt = GefenMuon(
                params,
                lr=LR,
                fused=False,
                weight_decay=WD,
                sharded_mode=mode,
            )
            return params, opt

        ex_params, ex_opt = build("exact")
        di_params, di_opt = build("distributed")

        for step in range(DIST_STEPS + 1):
            grads = make_grads(step)
            for idx, ((_, p), g) in enumerate(zip(ex_params, grads)):
                p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)]) if _active_for_fluctuating_step(step, idx) else None
            for idx, ((_, p), g) in enumerate(zip(di_params, grads)):
                p.grad = distribute_tensor(g.clone(), mesh, [Shard(0)]) if _active_for_fluctuating_step(step, idx) else None
            ex_opt.step()
            di_opt.step()

        ex_gathered = [p.detach().full_tensor() for _, p in ex_params]
        di_gathered = [p.detach().full_tensor() for _, p in di_params]

        if rank == 0:
            ref_params = [
                (name, nn.Parameter(init.clone())) for name, init in zip(names, inits)
            ]
            ref_opt = GefenMuon(
                ref_params,
                lr=LR,
                fused=False,
                weight_decay=WD,
            )
            for step in range(DIST_STEPS + 1):
                grads = make_grads(step)
                for idx, ((_, p), g) in enumerate(zip(ref_params, grads)):
                    p.grad = g.clone() if _active_for_fluctuating_step(step, idx) else None
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
            finite = all(torch.isfinite(g).all().item() for g in di_gathered)
            q.put(("result", max_vs_oracle, max_vs_exact, finite))
        else:
            q.put(("done", rank))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_fluctuating_case(world=2):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(target=_worker_distributed_fluctuating, args=(r, world, port, q))
        for r in range(world)
    ]
    for pr in procs:
        pr.start()

    result = None
    for _ in range(world):
        try:
            item = q.get(timeout=300)
        except queue.Empty:
            break
        if item[0] == "result":
            result = item[1:]

    for pr in procs:
        pr.join(timeout=300)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
            pr.join(timeout=10)
    for pr in procs:
        assert pr.exitcode == 0, f"[fluctuating x{world}] worker exited with {pr.exitcode}"
    assert result is not None, f"[fluctuating x{world}] no result from rank 0"

    max_vs_oracle, max_vs_exact, finite = result
    homogeneous = _cuda_archs_homogeneous(world)
    exact_tol = DIST_EXACT_TOL if homogeneous else DIST_MIXED_ARCH_EXACT_TOL
    ok = finite and (max_vs_oracle <= DIST_TOL) and (max_vs_exact <= exact_tol)
    print(
        f"[fluctuating x{world}] finite={finite} "
        f"max|dist-oracle|={max_vs_oracle:.3e} "
        f"max|dist-exact|={max_vs_exact:.3e} (tol={exact_tol:.0e}) "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return ok


CKPT_SPLIT_STEP = 2
CKPT_TOTAL_STEPS = 5


def _checkpoint_specs():
    return [
        ("ckpt_w0", 32, 48),
        ("ckpt_w1", 40, 32),
        ("ckpt_w2", 33, 64),
        ("ckpt_w3", 2, 16),
    ]


def _checkpoint_grads(step, shapes, dev):
    g = torch.Generator(device="cpu").manual_seed(900 + step)
    return [
        (torch.randn(M, N, generator=g, dtype=torch.float32) * 0.01).to(
            dev, torch.bfloat16
        )
        for (M, N) in shapes
    ]


def _worker_distributed_checkpoint_save(rank, world, port, path, q):
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
        specs = _checkpoint_specs()
        names = [name for name, _, _ in specs]
        shapes = [(M, N) for _, M, N in specs]
        gen = torch.Generator(device="cpu").manual_seed(2468)
        init_cpu = [
            (torch.randn(M, N, generator=gen, dtype=torch.float32) * 0.1).to(torch.bfloat16)
            for (M, N) in shapes
        ]
        params = [
            (
                name,
                nn.Parameter(distribute_tensor(init.to(dev).clone(), mesh, [Shard(0)])),
            )
            for name, init in zip(names, init_cpu)
        ]
        opt = GefenMuon(
            params,
            lr=LR,
            fused=False,
            weight_decay=WD,
            sharded_mode="distributed",
        )

        for step in range(CKPT_SPLIT_STEP):
            grads = _checkpoint_grads(step, shapes, dev)
            for (_, p), grad in zip(params, grads):
                p.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
            opt.step()

        opt_state = opt.state_dict()  # collective consolidation
        complete = all(
            "m_codebook" in opt_state["state"].get(pid, {})
            for group in opt_state["param_groups"]
            for pid in group["params"]
        )
        full_params = [p.detach().full_tensor().cpu() for _, p in params]
        if rank == 0:
            torch.save(
                {
                    "init": init_cpu,
                    "params": full_params,
                    "optimizer": opt_state,
                    "names": names,
                    "shapes": shapes,
                },
                path,
            )
            q.put(("saved", complete))
        else:
            q.put(("done", rank))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_distributed_checkpoint_load(rank, world, port, path, q):
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    from gefen.gefen_muon import GefenMuon, _stable_distributed_owner

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        checkpoint = torch.load(path, map_location=dev, weights_only=False)
        mesh = init_device_mesh("cuda", (world,))
        names = checkpoint["names"]
        shapes = checkpoint["shapes"]
        params = [
            (
                name,
                nn.Parameter(distribute_tensor(full.to(dev).clone(), mesh, [Shard(0)])),
            )
            for name, full in zip(names, checkpoint["params"])
        ]
        opt = GefenMuon(
            params,
            lr=LR,
            fused=False,
            weight_decay=WD,
            sharded_mode="distributed",
        )
        opt.load_state_dict(checkpoint["optimizer"])

        owner_state_ok = all(
            "m_codebook" in opt.state[p]
            for idx, (_, p) in enumerate(params)
            if _stable_distributed_owner(idx, world) == rank
        )
        gathered_owner_state_ok = [None] * world
        dist.all_gather_object(gathered_owner_state_ok, owner_state_ok)

        for step in range(CKPT_SPLIT_STEP, CKPT_TOTAL_STEPS):
            grads = _checkpoint_grads(step, shapes, dev)
            for (_, p), grad in zip(params, grads):
                p.grad = distribute_tensor(grad.clone(), mesh, [Shard(0)])
            opt.step()

        gathered = [p.detach().full_tensor() for _, p in params]
        if rank == 0:
            ref_params = [
                (name, nn.Parameter(init.to(dev).clone()))
                for name, init in zip(names, checkpoint["init"])
            ]
            ref_opt = GefenMuon(
                ref_params,
                lr=LR,
                fused=False,
                weight_decay=WD,
            )
            for step in range(CKPT_TOTAL_STEPS):
                grads = _checkpoint_grads(step, shapes, dev)
                for (_, p), grad in zip(ref_params, grads):
                    p.grad = grad.clone()
                ref_opt.step()
            ref = [p.detach() for _, p in ref_params]
            max_vs_oracle = max(
                (gathered[i].float() - ref[i].float()).abs().max().item()
                for i in range(len(shapes))
            )
            finite = all(torch.isfinite(g).all().item() for g in gathered)
            q.put(("result", max_vs_oracle, finite, gathered_owner_state_ok))
        else:
            q.put(("done", rank))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_checkpoint_phase(worker, world, path):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(target=worker, args=(r, world, port, str(path), q))
        for r in range(world)
    ]
    for pr in procs:
        pr.start()

    items = []
    for _ in range(world):
        try:
            items.append(q.get(timeout=300))
        except queue.Empty:
            break

    for pr in procs:
        pr.join(timeout=300)
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
            pr.join(timeout=10)
    for pr in procs:
        assert pr.exitcode == 0, f"[checkpoint x{world}] worker exited with {pr.exitcode}"
    return items


def _run_distributed_checkpoint_case(path, save_world=2, load_world=2):
    save_items = _spawn_checkpoint_phase(
        _worker_distributed_checkpoint_save, save_world, path
    )
    saved = [item for item in save_items if item[0] == "saved"]
    assert saved and saved[0][1], "[checkpoint] consolidated state missing owner momentum"

    load_items = _spawn_checkpoint_phase(
        _worker_distributed_checkpoint_load, load_world, path
    )
    result = [item[1:] for item in load_items if item[0] == "result"]
    assert result, "[checkpoint] no load result from rank 0"
    max_vs_oracle, finite, owner_state_ok = result[0]
    ok = finite and all(owner_state_ok) and max_vs_oracle <= DIST_TOL
    print(
        f"[checkpoint {save_world}->{load_world}] finite={finite} "
        f"owner_state_ok={owner_state_ok} max|resume-oracle|={max_vs_oracle:.3e} "
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
            results["distributed_fluctuating_x2"] = _run_distributed_fluctuating_case(2)
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

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
        or torch.cuda.device_count() < 2,
        reason="distributed Muon mode requires >=2 CUDA GPUs + NCCL",
    )
    def test_muon_fsdp2_distributed_fluctuating_grad_set():
        assert _run_distributed_fluctuating_case(2)

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
        or torch.cuda.device_count() < 2,
        reason="distributed Muon checkpoint test requires >=2 CUDA GPUs + NCCL",
    )
    def test_muon_fsdp2_distributed_checkpoint_restore(tmp_path):
        load_world = 4 if torch.cuda.device_count() >= 4 else 2
        assert _run_distributed_checkpoint_case(
            tmp_path / "gefen_muon_distributed.pt",
            save_world=2,
            load_world=load_world,
        )


if __name__ == "__main__":
    ok = run()
    print("\nMUON FSDP2 PARITY OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
