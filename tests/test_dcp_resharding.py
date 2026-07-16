"""Real DCP save/load and 2-to-4 resharding for standalone Gefen state."""

from __future__ import annotations

from datetime import timedelta
import multiprocessing as mp
import os
import queue
import socket
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from gefen import Gefen, GefenDCPState


_SHAPE = (8, 6)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _local(value):
    value = value.to_local() if hasattr(value, "to_local") else value
    return value.wait() if hasattr(value, "wait") else value


def _full_values():
    numel = _SHAPE[0] * _SHAPE[1]
    parameter = torch.linspace(-0.8, 0.7, numel).reshape(_SHAPE)
    momentum = torch.linspace(-1.25, 1.5, numel).reshape(_SHAPE)
    second = torch.arange(1, numel + 1, dtype=torch.float32).reshape(_SHAPE) / 19
    gradient = torch.arange(numel, dtype=torch.float32).reshape(_SHAPE).sin()
    return parameter, momentum, second, gradient


def _build(mesh, full_parameter):
    parameter = nn.Parameter(
        distribute_tensor(full_parameter.clone(), mesh, [Shard(0)])
    )
    optimizer = Gefen(
        [("weight", parameter)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2e-8,
        weight_decay=0.03,
        fused=False,
        factored_v_2d=False,
        deterministic=True,
    )
    return parameter, optimizer


def _local_slice(full, rank, world):
    return torch.chunk(full, world, dim=0)[rank]


def _seed_projected(optimizer, parameter, momentum, second, rank, world):
    local_momentum = _local_slice(momentum, rank, world).clone()
    local_second = _local_slice(second, rank, world).clone()
    flat = local_momentum.reshape(-1)
    optimizer._gefen_codebook = torch.linspace(
        -1, 1, 256, device=flat.device
    )
    optimizer._gefen_global_step = 9
    optimizer.state[parameter].update(
        {
            "name": "weight",
            "automatic_period": 1,
            "step": 7,
            "m_codebook": torch.where(
                torch.signbit(flat),
                torch.zeros(
                    flat.numel(), dtype=torch.uint8, device=flat.device
                ),
                torch.full(
                    (flat.numel(),),
                    255,
                    dtype=torch.uint8,
                    device=flat.device,
                ),
            ).reshape(-1, 1),
            "m_magnitude": flat.abs().reshape(-1, 1),
            "vmean": local_second.reshape(-1, 1),
            "vmean_step": 6,
        }
    )


def _snapshot(optimizer, parameter):
    state = optimizer.state[parameter]
    codebook = optimizer._gefen_codebook
    momentum = codebook[state["m_codebook"].long()].mul(state["m_magnitude"])
    return {
        "parameter": _local(parameter).detach().clone(),
        "momentum": momentum.reshape(_local(parameter).shape).clone(),
        "second": state["vmean"].reshape(_local(parameter).shape).clone(),
        "step": state["step"],
        "vmean_step": state["vmean_step"],
        "global_step": optimizer._gefen_global_step,
    }


def _worker(rank, world, port, checkpoint_dir, mode, device_type, result_queue):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            WORLD_SIZE=str(world),
        )
        if device_type == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(
            "cuda:nccl,cpu:gloo" if device_type == "cuda" else "gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh(device_type, (world,), mesh_dim_names=("dp",))
        parameter_value, momentum, second, gradient = _full_values()
        parameter_value = parameter_value.to(device_type)
        momentum = momentum.to(device_type)
        second = second.to(device_type)
        gradient = gradient.to(device_type)
        parameter, optimizer = _build(mesh, parameter_value)
        wrapper = GefenDCPState(optimizer)
        if mode == "save":
            _seed_projected(
                optimizer, parameter, momentum, second, rank, world
            )
            torch.distributed.checkpoint.save(
                {"optimizer": wrapper},
                storage_writer=FileSystemWriter(checkpoint_dir),
            )
            result_queue.put({"rank": rank, "saved": True})
            return

        torch.distributed.checkpoint.load(
            {"optimizer": wrapper},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        reference_parameter, reference = _build(mesh, parameter_value)
        _seed_projected(
            reference, reference_parameter, momentum, second, rank, world
        )
        before = _snapshot(optimizer, parameter)
        expected_before = _snapshot(reference, reference_parameter)
        parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])
        reference_parameter.grad = distribute_tensor(
            gradient.clone(), mesh, [Shard(0)]
        )
        optimizer.step()
        reference.step()
        after = _snapshot(optimizer, parameter)
        expected_after = _snapshot(reference, reference_parameter)
        result_queue.put(
            {
                "rank": rank,
                "restored_exact": all(
                    torch.equal(before[key], expected_before[key])
                    if torch.is_tensor(before[key])
                    else before[key] == expected_before[key]
                    for key in before
                ),
                "continued_exact": all(
                    torch.equal(after[key], expected_after[key])
                    if torch.is_tensor(after[key])
                    else after[key] == expected_after[key]
                    for key in after
                ),
                "shape": tuple(_local(parameter).shape),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run(world, checkpoint_dir, mode, *, device_type="cpu", timeout=120):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_worker,
            args=(
                rank,
                world,
                port,
                checkpoint_dir,
                mode,
                device_type,
                result_queue,
            ),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    results = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=timeout))
    except queue.Empty:
        pass
    finally:
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
    assert len(results) == world, (results, [p.exitcode for p in processes])
    assert all(p.exitcode == 0 for p in processes), [p.exitcode for p in processes]
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_save_on_two_ranks_loads_and_continues_on_four(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-2-to-4")
    saved = _run(2, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run(4, checkpoint_dir, "load")
    assert all("fatal_error" not in item for item in loaded), loaded
    assert [item["shape"] for item in loaded] == [
        (2, 6),
        (2, 6),
        (2, 6),
        (2, 6),
    ]
    assert all(item["restored_exact"] for item in loaded), loaded
    assert all(item["continued_exact"] for item in loaded), loaded


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 4
    or not dist.is_nccl_available(),
    reason="GPU DCP resharding coverage requires four CUDA devices and NCCL",
)
def test_dcp_save_on_two_gpus_loads_and_continues_on_four(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-gpu-2-to-4")
    saved = _run(2, checkpoint_dir, "save", device_type="cuda")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run(4, checkpoint_dir, "load", device_type="cuda")
    assert all("fatal_error" not in item for item in loaded), loaded
    assert all(item["restored_exact"] for item in loaded), loaded
    assert all(item["continued_exact"] for item in loaded), loaded


def test_rejects_non_dtensor_optimizer(tmp_path):
    init_file = tmp_path / "single-rank-init"
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_file),
        rank=0,
        world_size=1,
    )
    try:
        parameter = nn.Parameter(torch.ones(4, 4))
        optimizer = Gefen([("weight", parameter)], lr=1e-3, fused=False)
        with pytest.raises(RuntimeError, match="every parameter to be a DTensor"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()
