"""FSDP2 training-time CPU offload (`CPUOffloadPolicy`) with plain Gefen.

`fully_shard(module, offload_policy=CPUOffloadPolicy())` keeps each sharded
parameter (and its gradient) on CPU, moving it to GPU only for the forward /
backward compute; the optimizer then steps the CPU-resident local shard. This
is the CPU-resident non-fused path exercised through FSDP2's own offload, and
distinct from the FSDP2 *checkpoint* `get_optimizer_state_dict(cpu_offload=True)`
path covered in test_gefen_fsdp2_checkpoint.py.

Covered here: single-rank (1 GPU) and two-rank (2 GPU) offloaded stepping --
Gefen steps its CPU-resident local shard on each rank (rank-locally; plain Gefen
runs no cross-rank codebook collective). Checks are rank-local (`to_local()`, not
the `full_tensor()` all-gather, which would need a collective over CPU-resident
shards). Checkpoint/resume of the optimizer state is covered separately by
test_gefen_fsdp2_checkpoint.py (DCP) and test_cpu_resident_step.py (paged resume).

Run (needs one CUDA GPU + NCCL): pytest tests/test_fsdp2_cpu_offload.py
"""
import os
import socket
import time

import pytest
import torch


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _spawn(target, world, timeout=180):
    """Spawn `world` workers; return the object rank 0 puts on the queue. Fails
    fast if a worker dies before producing a result (no long queue-timeout hang)."""
    import queue as _queue

    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    procs = [
        context.Process(target=target, args=(rank, world, port, result_queue))
        for rank in range(world)
    ]
    for proc in procs:
        proc.start()
    result = None
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result = result_queue.get(timeout=2)
                break
            except _queue.Empty:
                # Fail fast if any worker already died without a result.
                dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
                if dead:
                    raise AssertionError(f"worker exited early with {dead}")
    finally:
        for proc in procs:
            proc.join(timeout=15)
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    assert result is not None, "no result from rank 0 (timeout)"
    assert all(proc.exitcode == 0 for proc in procs), [p.exitcode for p in procs]
    return result


def _offload_step_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        torch.manual_seed(0)
        d = 64
        model = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).cuda()
        for module in model:
            if isinstance(module, nn.Linear):
                fully_shard(module, offload_policy=CPUOffloadPolicy())
        fully_shard(model, offload_policy=CPUOffloadPolicy())

        # fused=True is axolotl's default; under CPU offload the shard is a CPU
        # tensor, so the per-tensor gate must downgrade to the pure-torch path.
        optimizer = Gefen(list(model.named_parameters()), lr=1e-3, fused=True)

        gen = torch.Generator().manual_seed(123 + rank)
        w = (torch.randn(d, d, generator=gen) / (d ** 0.5)).cuda()
        x = torch.randn(32, d, generator=gen).cuda()
        y = x @ w

        # Rank-local checks: to_local() is the local shard on CPU (no collective);
        # full_tensor() would all-gather CPU shards over NCCL and is unnecessary.
        def _locals():
            return [
                (p.to_local() if hasattr(p, "to_local") else p).detach().cpu().clone()
                for p in model.parameters()
            ]

        before = _locals()
        losses = []
        for _ in range(8):
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        shard_devices, state_devices = set(), set()
        for p in model.parameters():
            local = p.to_local() if hasattr(p, "to_local") else p
            shard_devices.add(local.device.type)
            for v in optimizer.state[p].values():
                if torch.is_tensor(v):
                    state_devices.add(v.device.type)

        after = _locals()
        checks = {
            "shards_cpu": shard_devices == {"cpu"},
            "state_cpu": state_devices == {"cpu"},
            "params_moved": all(not torch.equal(b, a) for b, a in zip(before, after)),
            "finite": all(torch.isfinite(a).all().item() for a in after),
            "loss_decreased": losses[-1] < losses[0],
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, checks)
        if rank == 0:
            result_queue.put({k: all(g[k] for g in gathered) for k in checks})
    finally:
        dist.destroy_process_group()


_NCCL = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.distributed.is_nccl_available(),
    reason="FSDP2 CPUOffloadPolicy needs a CUDA GPU and NCCL",
)
_TWO_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_nccl_available(),
    reason="requires two CUDA GPUs and NCCL",
)


@_NCCL
def test_gefen_steps_fsdp2_cpu_offload_shards_single_rank():
    checks = _spawn(_offload_step_worker, world=1)
    failed = {k: v for k, v in checks.items() if not v}
    assert not failed, f"FSDP2 CPU-offload checks failed: {failed}"


@_TWO_GPU
def test_gefen_steps_fsdp2_cpu_offload_shards_two_rank():
    checks = _spawn(_offload_step_worker, world=2)
    failed = {k: v for k, v in checks.items() if not v}
    assert not failed, f"2-GPU FSDP2 CPU-offload checks failed: {failed}"
