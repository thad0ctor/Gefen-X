"""FSDP2 training-time CPU offload (`CPUOffloadPolicy`) with plain Gefen.

`fully_shard(module, offload_policy=CPUOffloadPolicy())` keeps each sharded
parameter (and its gradient) on CPU, moving it to GPU only for the forward /
backward compute; the optimizer then steps the CPU-resident local shard. This
is the CPU-resident non-fused path exercised through FSDP2's own offload, and
distinct from the FSDP2 *checkpoint* `get_optimizer_state_dict(cpu_offload=True)`
path covered in test_gefen_fsdp2_checkpoint.py.

The single-rank case here runs on one GPU (no cross-rank collectives). Multi-GPU
CPU-offload -- where Gefen's cross-rank codebook collectives run against CPU
shards -- is validated on multi-GPU hardware separately; both single- and
multi-GPU (NCCL-only) offload were confirmed to keep Gefen stepping CPU shards.

Run (needs one CUDA GPU + NCCL): pytest tests/test_fsdp2_cpu_offload.py
"""
import os
import socket

import pytest
import torch


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _fsdp2_cpu_offload_worker(rank, world, port, result_queue):
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

        gen = torch.Generator().manual_seed(123)
        w = (torch.randn(d, d, generator=gen) / (d ** 0.5)).cuda()
        x = torch.randn(32, d, generator=gen).cuda()
        y = x @ w

        before = [p.detach().full_tensor().clone() for p in model.parameters()]
        losses = []
        for _ in range(8):
            loss = ((model(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        # The local shard the optimizer stepped, and its momentum state, are CPU.
        shard_devices = set()
        state_devices = set()
        for p in model.parameters():
            local = p.to_local() if hasattr(p, "to_local") else p
            shard_devices.add(local.device.type)
            for v in optimizer.state[p].values():
                if torch.is_tensor(v):
                    state_devices.add(v.device.type)

        after = [p.detach().full_tensor().clone() for p in model.parameters()]
        moved = all(not torch.equal(b, a) for b, a in zip(before, after))
        finite = all(torch.isfinite(a).all().item() for a in after)

        checks = {
            "shards_cpu": shard_devices == {"cpu"},
            "state_cpu": state_devices == {"cpu"},
            "params_moved": moved,
            "finite": finite,
            "loss_decreased": losses[-1] < losses[0],
        }
        if rank == 0:
            result_queue.put(checks)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.distributed.is_nccl_available(),
    reason="FSDP2 CPUOffloadPolicy needs a CUDA GPU and NCCL",
)
def test_gefen_steps_fsdp2_cpu_offload_shards_single_rank():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    proc = context.Process(
        target=_fsdp2_cpu_offload_worker, args=(0, 1, port, result_queue)
    )
    proc.start()
    checks = result_queue.get(timeout=180)
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
    assert proc.exitcode == 0
    failed = {k: v for k, v in checks.items() if not v}
    assert not failed, f"FSDP2 CPU-offload checks failed: {failed}"
