"""Real DCP save/load and N-to-M resharding for standalone Gefen state.

These tests exercise the *real* save path -- a genuine ``Gefen.step`` learns the
exact codebook and picks a block period > 1 -- and check that after a DCP reshard
the restored optimizer (a) keeps Gefen's compact ~1 byte/param block state
instead of collapsing to per-element period one, and (b) continues within a
quantization-noise tolerance of a native run at the *target* topology. The
continuation is tolerance-based, not exact, because routing the momentum through
a dense DCP reshard re-blocks it against a freshly learned per-shard codebook --
a different (and inherently lossy) blocking than the native target-topology run.
"""

from __future__ import annotations

from datetime import timedelta
import multiprocessing as mp
import os
import queue
import socket
import traceback

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from gefen import Gefen, GefenDCPState


# A 2D weight large enough that the block-variance period search returns a real
# period > 1 (so the compact vs per-element distinction is actually tested) and
# that 2, 4 both divide the row count for clean N-to-M resharding.
_SHAPE = (256, 128)
_SAVE_STEPS = 3


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _local(value):
    value = value.to_local() if hasattr(value, "to_local") else value
    return value.wait() if hasattr(value, "wait") else value


def _global_param():
    numel = _SHAPE[0] * _SHAPE[1]
    return torch.linspace(-0.8, 0.7, numel).reshape(_SHAPE)


def _global_grad(step, device):
    # Deterministic global gradient shared by every topology, so the momentum /
    # second-moment history is identical across world sizes up to quantization.
    generator = torch.Generator().manual_seed(1000 + step)
    grad = torch.randn(_SHAPE, generator=generator)
    return grad.to(device)


def _build(mesh, device):
    parameter = nn.Parameter(
        distribute_tensor(_global_param().to(device), mesh, [Shard(0)])
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


def _step(parameter, optimizer, mesh, step, device):
    parameter.grad = distribute_tensor(
        _global_grad(step, device), mesh, [Shard(0)]
    )
    optimizer.step()


def _period_info(optimizer, parameter):
    state = optimizer.state[parameter]
    period = int(state["automatic_period"])
    local_numel = int(_local(parameter).numel())
    return {
        "period": period,
        "vmean_numel": int(state["vmean"].numel()),
        "mmag_numel": int(state["m_magnitude"].numel()),
        "mcodebook_numel": int(state["m_codebook"].numel()),
        "local_numel": local_numel,
        # Compact iff the second-moment / magnitude state is one value per block,
        # i.e. numel == local_numel / period, not one-per-element (period one).
        "compact": (
            period > 1
            and int(state["vmean"].numel()) * period == local_numel
            and int(state["m_magnitude"].numel()) * period == local_numel
        ),
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
            torch.cuda.set_device(rank % torch.cuda.device_count())
            device = torch.device("cuda", rank % torch.cuda.device_count())
        else:
            device = torch.device("cpu")
        dist.init_process_group(
            "cuda:nccl,cpu:gloo" if device_type == "cuda" else "gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=120),
        )
        mesh = init_device_mesh(device_type, (world,), mesh_dim_names=("dp",))
        parameter, optimizer = _build(mesh, device)

        if mode == "save":
            for step in range(_SAVE_STEPS):
                _step(parameter, optimizer, mesh, step, device)
            saved = _period_info(optimizer, parameter)
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
            )
            result_queue.put({"rank": rank, "saved": True, **saved})
            return

        # mode == "load": reshard the saved state onto this (possibly different)
        # world, then compare a one-step continuation against a native reference
        # trained at THIS topology on the identical global gradient history.
        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(optimizer)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        restored = _period_info(optimizer, parameter)

        reference_param, reference = _build(mesh, device)
        for step in range(_SAVE_STEPS):
            _step(reference_param, reference, mesh, step, device)
        # Continue both from the identical parameter so the delta isolates the
        # restored optimizer STATE (momentum / second moment), not the parameter
        # trajectory that already diverged by per-shard quantization noise.
        reference_snapshot = _local(reference_param).detach().clone()
        with torch.no_grad():
            _local(parameter).copy_(reference_snapshot)

        _step(parameter, optimizer, mesh, _SAVE_STEPS, device)
        _step(reference_param, reference, mesh, _SAVE_STEPS, device)

        restored_next = _local(parameter).detach()
        reference_next = _local(reference_param).detach()
        finite = bool(torch.isfinite(restored_next).all())
        max_abs_diff = float((restored_next - reference_next).abs().max())
        result_queue.put(
            {
                "rank": rank,
                "shape": tuple(_local(parameter).shape),
                "restored": restored,
                "finite": finite,
                "continuation_max_abs_diff": max_abs_diff,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run(world, checkpoint_dir, mode, *, device_type="cpu", timeout=240):
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


# Continuation tolerance. The resharded momentum is re-blocked against a codebook
# relearned on the new local shard -- a different, lossy blocking than the native
# target-topology run -- so the one-step continuation matches only up to
# accumulated 256-level quantization noise (empirically ~1e-3 on these shards),
# not bit-for-bit.
_CONTINUE_TOL = 5e-3


def _assert_loaded(loaded, expected_shapes=None):
    assert all("fatal_error" not in item for item in loaded), loaded
    if expected_shapes is not None:
        assert [item["shape"] for item in loaded] == expected_shapes, loaded
    for item in loaded:
        assert item["finite"], item
        # The whole point of Gefen: the restored state must stay compact
        # (~1 byte/param), not collapse to per-element period one.
        assert item["restored"]["compact"], item
        assert item["restored"]["period"] > 1, item
        assert item["continuation_max_abs_diff"] < _CONTINUE_TOL, item


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_same_topology_stays_compact_and_continues(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-2-to-2")
    saved = _run(2, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    assert all(item["compact"] and item["period"] > 1 for item in saved), saved
    loaded = _run(2, checkpoint_dir, "load")
    _assert_loaded(loaded, expected_shapes=[(128, 128), (128, 128)])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_save_on_two_ranks_loads_and_continues_on_four(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-2-to-4")
    saved = _run(2, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run(4, checkpoint_dir, "load")
    _assert_loaded(
        loaded,
        expected_shapes=[(64, 128), (64, 128), (64, 128), (64, 128)],
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_save_on_four_ranks_loads_and_continues_on_two(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-4-to-2")
    saved = _run(4, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run(2, checkpoint_dir, "load")
    _assert_loaded(loaded, expected_shapes=[(128, 128), (128, 128)])


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 4
    or not dist.is_nccl_available(),
    reason="GPU DCP resharding coverage requires four CUDA devices and NCCL",
)
def test_dcp_save_on_two_gpus_loads_and_continues_on_four(tmp_path):
    # NCCL forbids two ranks per physical device in one communicator, so the
    # four-rank load needs four distinct GPUs (rank -> set_device(rank)).
    checkpoint_dir = str(tmp_path / "gefen-dcp-gpu-2-to-4")
    saved = _run(2, checkpoint_dir, "save", device_type="cuda")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run(4, checkpoint_dir, "load", device_type="cuda")
    _assert_loaded(loaded)


def _fully_shard_worker(rank, world, port, checkpoint_dir, mode, result_queue):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            WORLD_SIZE=str(world),
        )
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            "cuda:nccl,cpu:gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=120),
        )
        from torch.distributed.fsdp import fully_shard

        torch.manual_seed(0)
        model = nn.Linear(128, 64, bias=False).to(device)
        fully_shard(model)
        parameter = model.weight
        optimizer = Gefen(
            [("weight", parameter)],
            lr=2.5e-3,
            betas=(0.8, 0.97),
            eps=2e-8,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )

        def real_step(seed):
            generator = torch.Generator(device=device).manual_seed(seed)
            inputs = torch.randn(32, 128, generator=generator, device=device)
            targets = torch.randn(32, 64, generator=generator, device=device)
            optimizer.zero_grad()
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            optimizer.step()

        if mode == "save":
            for step in range(_SAVE_STEPS):
                real_step(2000 + step)
            info = _period_info(optimizer, parameter)
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
            )
            result_queue.put({"rank": rank, "saved": True, **info})
            return

        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(optimizer)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        restored = _period_info(optimizer, parameter)
        real_step(2000 + _SAVE_STEPS)
        finite = bool(torch.isfinite(_local(parameter)).all())
        result_queue.put({"rank": rank, "restored": restored, "finite": finite})
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_fully_shard(world, checkpoint_dir, mode, timeout=240):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_fully_shard_worker,
            args=(rank, world, port, checkpoint_dir, mode, result_queue),
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
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not dist.is_nccl_available(),
    reason="fully_shard DCP coverage requires two CUDA devices and NCCL",
)
def test_dcp_fully_shard_real_model_stays_compact(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-fully-shard")
    saved = _run_fully_shard(2, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    assert all(item["compact"] and item["period"] > 1 for item in saved), saved
    loaded = _run_fully_shard(2, checkpoint_dir, "load")
    assert all("fatal_error" not in item for item in loaded), loaded
    for item in loaded:
        assert item["finite"] and item["restored"]["compact"], item


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
        optimizer = Gefen(
            [("weight", parameter)], lr=1e-3, fused=False, factored_v_2d=False
        )
        with pytest.raises(RuntimeError, match="every parameter to be a DTensor"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


def test_rejects_capturable(tmp_path):
    init_file = tmp_path / "capturable-init"
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_file),
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        parameter = nn.Parameter(distribute_tensor(_global_param(), mesh, [Shard(0)]))
        optimizer = Gefen(
            [("weight", parameter)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
            capturable=True,
        )
        with pytest.raises(RuntimeError, match="capturable"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


def test_rejects_factored_v_2d(tmp_path):
    init_file = tmp_path / "factored-init"
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_file),
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        parameter = nn.Parameter(distribute_tensor(_global_param(), mesh, [Shard(0)]))
        optimizer = Gefen(
            [("weight", parameter)],
            lr=1e-3,
            fused=False,
            factored_v_2d=True,
        )
        with pytest.raises(RuntimeError, match="factored_v_2d"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()
