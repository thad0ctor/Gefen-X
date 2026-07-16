"""Multi-GPU (DDP) coverage for CPU-authoritative optimizer-state offload.

Offload is scoped to plain replicated-CUDA ``Gefen``; DistributedDataParallel is
exactly that replicated regime. These tests drive a live NCCL process group with
DDP gradient all-reduce and assert the offloaded run stays bit-identical to a
non-offloaded run stepping the same all-reduced gradients, with per-parameter
state remaining CPU-resident throughout.
"""

import copy
import os
import queue
import socket
import traceback

import pytest
import torch
import torch.nn as nn


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _build_model() -> nn.Module:
    # Fixed init so every rank starts identically; DDP additionally broadcasts
    # rank 0's weights, so the reference and offloaded models agree everywhere.
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 4)).cuda()


def _ddp_offload_worker(rank, world, port, factored, result_queue):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        # Inside the try so a set_device/init failure still reaches the
        # result_queue reporting path instead of surfacing as a bare timeout.
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        reference_model = _build_model()
        offloaded_model = copy.deepcopy(reference_model)
        reference_ddp = DDP(reference_model, device_ids=[rank])
        offloaded_ddp = DDP(offloaded_model, device_ids=[rank])

        def make_optimizer(module):
            optimizer = Gefen(
                module.named_parameters(),
                lr=3e-3,
                fused=False,
                factored_v_2d=factored,
            )
            optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
            return optimizer

        reference_optimizer = make_optimizer(reference_ddp.module)
        offloaded_optimizer = make_optimizer(offloaded_ddp.module)
        offloaded_optimizer.offload_state_()

        # Per-rank-distinct but deterministic batches so the all-reduce actually
        # mixes gradients across ranks; identical within a rank for both models.
        generator = torch.Generator(device="cuda").manual_seed(100 + rank)
        bit_exact = True
        cpu_authoritative = True
        for _ in range(5):
            inputs = torch.randn(16, 8, generator=generator, device="cuda")
            targets = torch.randn(16, 4, generator=generator, device="cuda")
            for ddp_model, optimizer in (
                (reference_ddp, reference_optimizer),
                (offloaded_ddp, offloaded_optimizer),
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = ((ddp_model(inputs) - targets) ** 2).mean()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                for reference_param, offloaded_param in zip(
                    reference_ddp.module.parameters(),
                    offloaded_ddp.module.parameters(),
                ):
                    if not torch.equal(reference_param, offloaded_param):
                        bit_exact = False
            cpu_authoritative = (
                cpu_authoritative
                and offloaded_optimizer.state_offload_active
                and all(
                    not torch.is_tensor(value) or value.device.type == "cpu"
                    for parameter_state in offloaded_optimizer.state.values()
                    for value in parameter_state.values()
                )
            )

        gathered = [None] * world
        dist.all_gather_object(gathered, (bit_exact, cpu_authoritative))
        if rank == 0:
            result_queue.put(gathered)
    except BaseException:  # surface worker failures to the parent process
        if rank == 0:
            result_queue.put(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available(),
    reason="DDP offload parity requires two CUDA GPUs and NCCL",
)
@pytest.mark.parametrize("factored", [False, True])
def test_ddp_offload_is_bit_exact_and_keeps_cpu_authority(factored):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_ddp_offload_worker,
            args=(rank, 2, port, factored, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        result = result_queue.get(timeout=180)
    except queue.Empty:
        result = None
    for process in processes:
        process.join(timeout=180)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    assert result is not None, "DDP offload workers timed out"
    if isinstance(result, str):
        pytest.fail("DDP offload worker raised:\n" + result)
    assert all(process.exitcode == 0 for process in processes)
    assert all(bit_exact for bit_exact, _ in result), result
    assert all(cpu_authoritative for _, cpu_authoritative in result), result
