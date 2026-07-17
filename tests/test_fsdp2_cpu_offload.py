"""FSDP2 training-time CPU offload (`CPUOffloadPolicy`) with plain Gefen.

`fully_shard(module, offload_policy=CPUOffloadPolicy())` keeps each sharded
parameter (and its gradient) on CPU, moving it to GPU only for the forward /
backward compute; the optimizer then steps the CPU-resident local shard. This
is the CPU-resident non-fused path exercised through FSDP2's own offload, and
distinct from the FSDP2 *checkpoint* `get_optimizer_state_dict(cpu_offload=True)`
path covered in test_gefen_fsdp2_checkpoint.py (which shards without CPU offload).

Coverage here:
  * single-rank (1 GPU) and two-rank (2 GPU) offloaded stepping -- Gefen steps
    its CPU-resident local shard on each rank (rank-locally; plain Gefen runs no
    cross-rank codebook collective). Checks are rank-local (`to_local()`, not the
    `full_tensor()` all-gather, which would need a collective over CPU-resident
    shards);
  * single-rank resume of model+optimizer through DCP while the model stays
    under CPUOffloadPolicy -- save with get_state_dict, restore into a fresh
    model/optimizer, and continue with exact next-step parity against the
    uninterrupted run. DCP gathers the CPU-resident state, so this worker uses a
    gloo-capable process group; the step workers use NCCL-only to prove the step
    itself needs no gloo backend. (Multi-rank DCP resume under CPUOffloadPolicy
    deadlocks in DCP's gather and is left to the non-offload multi-rank coverage
    in test_gefen_fsdp2_checkpoint.py.)

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


def _init(rank, world, port, backend):
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=world)


def _build_offloaded(d=64):
    """A tiny model fully_sharded under CPUOffloadPolicy + a plain Gefen. Seeded
    so independent builds get identical initial weights."""
    import torch.nn as nn
    try:
        from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy  # >= 2.6
    except ImportError:
        from torch.distributed._composable.fsdp import (  # torch 2.5
            fully_shard,
            CPUOffloadPolicy,
        )

    from gefen import Gefen

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).cuda()
    for module in model:
        if isinstance(module, nn.Linear):
            fully_shard(module, offload_policy=CPUOffloadPolicy())
    fully_shard(model, offload_policy=CPUOffloadPolicy())
    return model, Gefen(list(model.named_parameters()), lr=1e-3, fused=False)


def _locals(model):
    return [
        (p.to_local() if hasattr(p, "to_local") else p).detach().cpu().clone()
        for p in model.parameters()
    ]


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

    # NCCL-only on purpose: the offloaded step itself is rank-local and needs no
    # CPU collective, so it must work without a gloo backend.
    from gefen import Gefen

    _init(rank, world, port, "nccl")
    try:
        model, _ = _build_offloaded()
        # fused=True is axolotl's default; under CPU offload the shard is a CPU
        # tensor, so the per-tensor gate must downgrade to the pure-torch path.
        optimizer = Gefen(list(model.named_parameters()), lr=1e-3, fused=True)

        gen = torch.Generator().manual_seed(123 + rank)
        w = (torch.randn(64, 64, generator=gen) / 8.0).cuda()
        x = torch.randn(32, 64, generator=gen).cuda()
        y = x @ w

        before = _locals(model)
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

        after = _locals(model)
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


def _offload_resume_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_state_dict,
        set_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    try:
        from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy  # >= 2.6
    except ImportError:
        from torch.distributed._composable.fsdp import (  # torch 2.5
            fully_shard,
            CPUOffloadPolicy,
        )

    from gefen import Gefen

    # DCP gathers the CPU-resident optimizer state, so route CPU collectives to
    # gloo (CUDA collectives still use NCCL).
    _init(rank, world, port, "cuda:nccl,cpu:gloo")
    try:
        d = 64
        # One shared mesh for every build so the rank-local checkpoint's recorded
        # mesh matches on resume (mirrors test_gefen_fsdp2_checkpoint).
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

        def build():
            torch.manual_seed(0)
            m = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).cuda()
            for module in m:
                if isinstance(module, nn.Linear):
                    fully_shard(module, mesh=mesh, offload_policy=CPUOffloadPolicy())
            fully_shard(m, mesh=mesh, offload_policy=CPUOffloadPolicy())
            return m, Gefen(list(m.named_parameters()), lr=1e-3, fused=False)

        gen = torch.Generator().manual_seed(7)
        w = (torch.randn(d, d, generator=gen) / 8.0).cuda()
        x = torch.randn(32, d, generator=gen).cuda()
        y = x @ w

        def step(m, o):
            loss = ((m(x) - y) ** 2).mean()
            loss.backward()
            o.step()
            o.zero_grad(set_to_none=True)

        full = StateDictOptions(full_state_dict=True, cpu_offload=True)
        broadcast = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)

        # Uninterrupted reference: 3 steps.
        ref_model, ref_opt = build()
        for _ in range(3):
            step(ref_model, ref_opt)
        reference = _locals(ref_model)

        # Interrupted: 2 steps, capture model+optimizer state, restore into a
        # separate model/optimizer, 1 more step. Uses the in-memory
        # get_state_dict/set_state_dict path (not dcp.save/load): Gefen's FSDP2
        # optimizer state is a variable-size rank-local payload that DCP's
        # fixed-shape save/load planner rejects, so the whole suite -- including
        # test_gefen_fsdp2_checkpoint -- resumes through get/set_state_dict.
        model, opt = build()
        step(model, opt)
        step(model, opt)
        msd, osd = get_state_dict(model, opt, options=full)

        resumed_model, resumed_opt = build()
        set_state_dict(
            resumed_model,
            resumed_opt,
            model_state_dict=msd,
            optim_state_dict=osd,
            options=broadcast,
        )
        step(resumed_model, resumed_opt)
        resumed = _locals(resumed_model)

        # A lossless resume must reproduce the uninterrupted run exactly (both
        # take the deterministic decomposed CPU path).
        exact = all(torch.equal(a, b) for a, b in zip(reference, resumed))
        gathered = [None] * world
        dist.all_gather_object(gathered, bool(exact))
        if rank == 0:
            result_queue.put({"resume_matches_uninterrupted": all(gathered)})
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


@_NCCL
def test_fsdp2_cpu_offload_dcp_resume_is_exact_single_rank():
    checks = _spawn(_offload_resume_worker, world=1)
    assert checks["resume_matches_uninterrupted"]


# NOTE: multi-rank DCP resume *while under CPUOffloadPolicy* is intentionally not
# covered here -- the two-rank get_state_dict/set_state_dict gather over the
# CPU-resident, offloaded rank-local state deadlocks in DCP. Multi-rank DCP
# optimizer-state resume without CPU offload is covered by
# test_gefen_fsdp2_checkpoint.py; the single-rank test above exercises the
# resume-under-offload path itself.
