"""FSDP2 (DTensor) APPROX sharded-mode smoke test for GefenMuon.

GefenMuon(sharded_mode="approx") is an opt-in, explicitly NON-PARITY mode: each
rank runs the whole pipeline (period/codebook, quantized momentum, Newton-Schulz)
on its LOCAL shard only -- no all-gather, and NS on the smaller row-sharded
matrix. This makes it cheaper than the default exact mode but it does NOT match
the single-GPU result (NS of a row block != orthogonalization of the full matrix).

This test (2-rank CPU/gloo, non-fused) asserts that approx mode:
  - runs to completion on every rank, including a rank that owns an EMPTY shard
    (it has no collectives, so the empty rank returns early without deadlocking),
  - produces finite, changed parameters, and
  - genuinely diverges from the exact single-GPU oracle (proving it is not
    silently falling back to the exact path).

Run: python tests/test_muon_fsdp2_approx.py
"""
import os
import queue
import socket

import torch
import torch.distributed  # noqa: F401
import torch.nn as nn

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

LR = 5e-3
WD = 0.1
CASES = {"even": (8, 6), "uneven": (5, 4), "empty": (1, 4)}


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _worker(rank, world, case, port, q):
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
        torch.manual_seed(1234)
        full_init = torch.randn(M, N, dtype=torch.float32)
        full_grad = torch.randn(M, N, dtype=torch.float32) * 0.01

        p = nn.Parameter(distribute_tensor(full_init.clone(), mesh, [Shard(0)]))
        opt = GefenMuon([("w", p)], lr=LR, fused=False, weight_decay=WD,
                        sharded_mode="approx")
        for _ in range(2):
            p.grad = distribute_tensor(full_grad.clone(), mesh, [Shard(0)])
            opt.step()
        local_rows = p.to_local().shape[0]
        gathered = p.detach().full_tensor()  # collective; all ranks must call

        if rank == 0:
            # exact single-GPU oracle for the same init/grad
            p_ref = nn.Parameter(full_init.clone())
            opt_ref = GefenMuon([("w", p_ref)], lr=LR, fused=False, weight_decay=WD)
            for _ in range(2):
                p_ref.grad = full_grad.clone()
                opt_ref.step()
            changed = not torch.equal(gathered, full_init)
            finite = torch.isfinite(gathered).all().item()
            diff = (gathered - p_ref.detach()).abs().max().item()
            q.put(("result", case, local_rows, changed, finite, diff))
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

    results, rank_rows = {}, {}
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
    for pr in procs:
        if pr.is_alive():
            pr.terminate()
            pr.join(timeout=10)
    for pr in procs:
        assert pr.exitcode == 0, f"[{case}] worker exited with {pr.exitcode}"
    assert "main" in results, f"[{case}] no result from rank 0 (deadlock?)"

    _case, local_rows, changed, finite, diff = results["main"]
    print(f"[approx {case}] rank0_rows={local_rows} changed={changed} "
          f"finite={finite} |approx-exact|={diff:.3e}")
    # approx must run, stay finite, change the param, and (by construction) NOT
    # match the exact oracle -- except the degenerate empty case where rank 0
    # owns the whole tensor (world-1 split puts everything on rank 0), so approx
    # == exact there. For real multi-row shards the two must differ.
    ok = changed and finite
    if case != "empty":
        ok = ok and diff > 1e-6
    return ok


if pytest is not None:

    @pytest.mark.parametrize("case", list(CASES))
    def test_muon_fsdp2_approx(case):
        assert _run_case(case)


def run():
    return all(_run_case(c) for c in CASES)


if __name__ == "__main__":
    ok = run()
    print("\nMUON FSDP2 APPROX OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
