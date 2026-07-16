"""Real DCP save/load and N-to-M resharding for standalone Gefen state.

These tests exercise the *real* save path -- a genuine ``Gefen.step`` learns the
exact codebook and picks a block period > 1 -- and check that after a DCP reshard
the restored optimizer (a) keeps Gefen's compact ~1 byte/param block state
instead of collapsing to per-element period one, and (b) continues within
tolerance of a native run at the *target* topology. The continuation is
tolerance-based, not exact, because routing the momentum through a dense DCP
reshard re-quantizes it against a freshly learned per-shard codebook.

These reshards all re-derive the same block period, so they isolate that
re-quantization error; the second, larger approximation a reshard can incur --
re-aggregating the per-block second moment when the target blocking coarsens --
is pinned separately by test_second_moment_reblocking_regimes. See _CONTINUE_TOL.
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
from torch.distributed.tensor import (
    Replicate,
    Shard,
    distribute_tensor,
    init_device_mesh,
)

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


def _build(mesh, device, fused=False):
    parameter = nn.Parameter(
        distribute_tensor(_global_param().to(device), mesh, [Shard(0)])
    )
    optimizer = Gefen(
        [("weight", parameter)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2e-8,
        weight_decay=0.03,
        fused=fused,
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


def _worker(rank, world, port, checkpoint_dir, mode, device_type, fused, result_queue):
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
        parameter, optimizer = _build(mesh, device, fused)

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

        reference_param, reference = _build(mesh, device, fused)
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


def _run(world, checkpoint_dir, mode, *, device_type="cpu", fused=False, timeout=240):
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
                fused,
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


# Continuation tolerance for the reshards below. Every one of them re-derives the
# SAME block period (2048) on the target shard, so source and target block
# boundaries align: the second moment survives the round trip to fp32 round-off
# (expanding vmean per element and re-averaging each block of identical values
# returns the original), and the only error left is re-quantizing the momentum
# against a codebook relearned on the new shard -- measured ~2e-4 here.
#
# This tolerance is therefore NOT a general bound on resharded continuation
# error. A reshard whose target period differs (a shard size whose block-variance
# search lands elsewhere) *coarsens* the blocking, and load then averages
# together source blocks whose per-element second-moment history never existed
# and cannot be recovered. That aggregation error is bounded only by how much
# vmean varies across the merged blocks -- unrelated to, and measurably larger
# than, quantization noise. See test_second_moment_reblocking_regimes.
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


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not dist.is_nccl_available(),
    reason="fused DCP coverage requires two CUDA devices and NCCL",
)
def test_dcp_fused_state_reshards_and_stays_compact(tmp_path):
    # The fused CUDA kernels (fused=True) produce the same optimizer-state layout
    # as the decomposed path, so GefenDCPState must save/reshard fused-trained
    # state too. Same-topology (2->2) on CUDA is enough to prove the fused state
    # round-trips compactly and continues; the reshard math itself is covered by
    # the non-fused N-to-M tests and is fused-independent.
    checkpoint_dir = str(tmp_path / "gefen-dcp-fused")
    saved = _run(2, checkpoint_dir, "save", device_type="cuda", fused=True)
    assert all(item.get("saved") for item in saved), saved
    assert all(item["compact"] and item["period"] > 1 for item in saved), saved
    loaded = _run(2, checkpoint_dir, "load", device_type="cuda", fused=True)
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


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
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


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
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


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
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


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_rejects_non_shard0_placement(tmp_path):
    # The PR advertises that non-Shard(0) placements fail closed.
    init_file = tmp_path / "replicate-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        parameter = nn.Parameter(distribute_tensor(_global_param(), mesh, [Replicate()]))
        optimizer = Gefen(
            [("weight", parameter)], lr=1e-3, fused=False, factored_v_2d=False
        )
        with pytest.raises(RuntimeError, match="Shard"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_rejects_multidimensional_mesh(tmp_path):
    # Multidimensional meshes fail closed.
    init_file = tmp_path / "mesh2d-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1, 1), mesh_dim_names=("dp", "tp"))
        parameter = nn.Parameter(
            distribute_tensor(_global_param(), mesh, [Shard(0), Replicate()])
        )
        optimizer = Gefen(
            [("weight", parameter)], lr=1e-3, fused=False, factored_v_2d=False
        )
        with pytest.raises(RuntimeError, match="one-dimensional"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
def test_rejects_muon_sharded_mode(tmp_path):
    # A group carrying a Muon sharded_mode fails closed.
    init_file = tmp_path / "shardedmode-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        parameter = nn.Parameter(distribute_tensor(_global_param(), mesh, [Shard(0)]))
        optimizer = Gefen(
            [("weight", parameter)], lr=1e-3, fused=False, factored_v_2d=False
        )
        optimizer.param_groups[0]["sharded_mode"] = "approx"
        with pytest.raises(RuntimeError, match="sharded mode"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


def _reordered_mesh_worker(rank, world, port, result_queue):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            WORLD_SIZE=str(world),
        )
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, timeout=timedelta(seconds=60)
        )
        from torch.distributed.device_mesh import DeviceMesh

        # A full-world 1-D mesh whose ranks are reordered ([1, 0] on world 2):
        # it passes the size check but permutes shard->rank ownership.
        mesh = DeviceMesh("cpu", torch.tensor([1, 0]))
        parameter = nn.Parameter(distribute_tensor(_global_param(), mesh, [Shard(0)]))
        optimizer = Gefen(
            [("weight", parameter)], lr=1e-3, fused=False, factored_v_2d=False
        )
        raised = False
        try:
            GefenDCPState(optimizer)
        except RuntimeError as exc:
            raised = "reordered" in str(exc) or "canonical" in str(exc)
        result_queue.put({"rank": rank, "raised": raised})
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires Gloo",
)
def test_rejects_reordered_full_world_mesh():
    # A reordered full-world mesh must fail closed on every rank.
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(target=_reordered_mesh_worker, args=(rank, 2, port, result_queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=120))
    except queue.Empty:
        pass
    finally:
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all("fatal_error" not in item for item in results), results
    assert len(results) == 2 and all(item["raised"] for item in results), results


# --- Review-finding coverage (Codex PR #83) -------------------------------
#
# The tests below cover the fail-closed / semantics fixes. Findings 1, 2, 3 and 5
# do not need a real topology change, so they run in-process on a one-rank Gloo
# group (save, then load into a differently configured optimizer). Finding 4
# (an N->M reshard whose dim-0 is smaller than the target world) needs multiple
# ranks and reuses the spawn/worker harness.


def _shaped_dparam(shape, mesh, device="cpu"):
    numel = 1
    for dim in shape:
        numel *= dim
    tensor = torch.linspace(-0.8, 0.7, numel).reshape(shape).to(device)
    return nn.Parameter(distribute_tensor(tensor, mesh, [Shard(0)]))


def _shaped_grad(shape, step):
    generator = torch.Generator().manual_seed(1000 + step)
    return torch.randn(shape, generator=generator)


def _save_optimizer(optimizer, param, shape, mesh, checkpoint_dir):
    for step in range(_SAVE_STEPS):
        param.grad = distribute_tensor(_shaped_grad(shape, step), mesh, [Shard(0)])
        optimizer.step()
    torch.distributed.checkpoint.save(
        {"optimizer": GefenDCPState(optimizer)},
        storage_writer=FileSystemWriter(checkpoint_dir),
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_restores_checkpoint_group_hyperparameters(tmp_path):
    init_file = tmp_path / "hyper-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)],
            lr=2.5e-3,
            betas=(0.8, 0.97),
            eps=2e-8,
            weight_decay=0.03,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )
        checkpoint_dir = str(tmp_path / "hyper-ckpt")
        _save_optimizer(optimizer, param, _SHAPE, mesh, checkpoint_dir)

        # Resume into an optimizer built with DIFFERENT hyperparameters, as if an
        # LR scheduler had advanced or the run was reconfigured. The restore must
        # adopt the checkpoint's hyperparameters, not silently keep these.
        param2 = _shaped_dparam(_SHAPE, mesh)
        resumed = Gefen(
            [("weight", param2)],
            lr=9.9e-9,
            betas=(0.1, 0.2),
            eps=7e-7,
            weight_decay=0.5,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )
        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(resumed)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        group = resumed.param_groups[0]
        assert abs(group["lr"] - 2.5e-3) < 1e-12, group["lr"]
        assert abs(group["beta1"] - 0.8) < 1e-12, group["beta1"]
        assert abs(group["beta2"] - 0.97) < 1e-12, group["beta2"]
        assert abs(group["eps"] - 2e-8) < 1e-15, group["eps"]
        assert abs(group["weight_decay"] - 0.03) < 1e-12, group["weight_decay"]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_swapped_parameter_identities(tmp_path):
    init_file = tmp_path / "identity-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        first = _shaped_dparam(shape, mesh)
        second = _shaped_dparam(shape, mesh)
        optimizer = Gefen(
            [("alpha", first), ("beta", second)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        for step in range(_SAVE_STEPS):
            for param in (first, second):
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step), mesh, [Shard(0)]
                )
            optimizer.step()
        checkpoint_dir = str(tmp_path / "identity-ckpt")
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(checkpoint_dir),
        )

        # The same two same-shaped parameters, registered in SWAPPED name order.
        # Slot-index addressing alone would load "successfully" and hand each
        # parameter the other's momentum; the identity guard must reject it.
        first_swapped = _shaped_dparam(shape, mesh)
        second_swapped = _shaped_dparam(shape, mesh)
        swapped = Gefen(
            [("beta", second_swapped), ("alpha", first_swapped)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        with pytest.raises(ValueError, match="parameter identities"):
            torch.distributed.checkpoint.load(
                {"optimizer": GefenDCPState(swapped)},
                storage_reader=FileSystemReader(checkpoint_dir),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_deterministic_mismatch(tmp_path):
    init_file = tmp_path / "deterministic-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )
        checkpoint_dir = str(tmp_path / "deterministic-ckpt")
        _save_optimizer(optimizer, param, _SHAPE, mesh, checkpoint_dir)

        param2 = _shaped_dparam(_SHAPE, mesh)
        other = Gefen(
            [("weight", param2)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
            deterministic=False,
        )
        with pytest.raises(ValueError, match="deterministic"):
            torch.distributed.checkpoint.load(
                {"optimizer": GefenDCPState(other)},
                storage_reader=FileSystemReader(checkpoint_dir),
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_restore_honors_force_2d_period_one(tmp_path):
    init_file = tmp_path / "period-init"
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        checkpoint_dir = str(tmp_path / "period-ckpt")
        _save_optimizer(optimizer, param, _SHAPE, mesh, checkpoint_dir)

        # Control: an unconstrained resume re-derives the compact block period.
        control_param = _shaped_dparam(_SHAPE, mesh)
        control = Gefen(
            [("weight", control_param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(control)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        assert int(control.state[control_param]["automatic_period"]) > 1

        # A resume that explicitly forces 2D params to period one must honor that
        # gate rather than running the raw search and restoring period > 1.
        forced_param = _shaped_dparam(_SHAPE, mesh)
        forced = Gefen(
            [("weight", forced_param)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
            force_2d_period_one=True,
        )
        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(forced)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        assert int(forced.state[forced_param]["automatic_period"]) == 1, forced.state[
            forced_param
        ]["automatic_period"]
    finally:
        dist.destroy_process_group()


def _empty_shard_worker(rank, world, port, checkpoint_dir, mode, result_queue):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            WORLD_SIZE=str(world),
        )
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, timeout=timedelta(seconds=120)
        )
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        # dim-0 == 2: fits two ranks fully, but reshards to empty local shards on
        # ranks >= 2 of a four-rank target.
        shape = (2, 128)
        parameter = nn.Parameter(
            distribute_tensor(
                torch.linspace(-0.8, 0.7, shape[0] * shape[1]).reshape(shape),
                mesh,
                [Shard(0)],
            )
        )
        optimizer = Gefen(
            [("weight", parameter)],
            lr=2.5e-3,
            betas=(0.8, 0.97),
            eps=2e-8,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )

        if mode == "save":
            for step in range(_SAVE_STEPS):
                parameter.grad = distribute_tensor(
                    _shaped_grad(shape, step), mesh, [Shard(0)]
                )
                optimizer.step()
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
            )
            result_queue.put({"rank": rank, "saved": True})
            return

        # The load must not crash on ranks whose local shard resharded to empty
        # (a None per-rank codebook previously dereferenced codebook.device).
        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(optimizer)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        local_numel = int(_local(parameter).numel())
        state = optimizer.state.get(parameter, {})
        materialized = all(
            key in state
            for key in ("automatic_period", "m_codebook", "m_magnitude", "vmean")
        )
        finite = True
        if materialized:
            finite = bool(
                torch.isfinite(state["m_magnitude"]).all()
                and torch.isfinite(state["vmean"]).all()
            )
        result_queue.put(
            {
                "rank": rank,
                "local_numel": local_numel,
                "materialized": materialized,
                "finite": finite,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_empty(world, checkpoint_dir, mode, timeout=240):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_empty_shard_worker,
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
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_reshard_to_empty_target_shard_does_not_crash(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-empty-shard")
    saved = _run_empty(2, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run_empty(4, checkpoint_dir, "load")
    assert all("fatal_error" not in item for item in loaded), loaded
    by_rank = {item["rank"]: item for item in loaded}
    # dim-0 == 2 over world 4: ranks 0 and 1 own a row, ranks 2 and 3 reshard to
    # empty local shards. The empty ranks must load as unmaterialized state (like
    # native, which never materializes an empty local shard) instead of crashing.
    assert by_rank[0]["materialized"] and by_rank[1]["materialized"], loaded
    assert by_rank[0]["local_numel"] == 128 and by_rank[1]["local_numel"] == 128, loaded
    assert by_rank[2]["local_numel"] == 0 and by_rank[3]["local_numel"] == 0, loaded
    assert not by_rank[2]["materialized"] and not by_rank[3]["materialized"], loaded
    for item in loaded:
        assert item["finite"], item


# --- Second review round (CodeRabbit + Codex on 7ea3f52) -------------------
#
# These harden the v2 additions themselves: unique/caller-stable identities
# (finding 1), pre-commit hyperparameter validation (finding 2), stale-slot
# refresh after add_param_group (finding 3), mixed-device codebook co-location
# (finding 4), cross-rank commit agreement (finding 5), and coherent
# initialization metadata (finding 6). Most run on a single-rank Gloo group and,
# where a full distributed repro is impractical, exercise the validation via a
# direct GefenDCPState.state_dict() -> mutate -> load_state_dict() round trip
# (which is the exact code path DCP drives per rank).


def _single_rank_group(init_file):
    dist.init_process_group(
        "gloo", init_method="file://{}".format(init_file), rank=0, world_size=1
    )


def _live_state_signature(optimizer, param):
    state = optimizer.state.get(param, {})
    return {
        "period": int(state["automatic_period"]) if "automatic_period" in state else None,
        "step": int(state["step"]) if "step" in state else None,
        "vmean_step": int(state["vmean_step"]) if "vmean_step" in state else None,
        "m_magnitude_sum": (
            float(state["m_magnitude"].double().sum())
            if "m_magnitude" in state
            else None
        ),
    }


def _group_hyper_signature(optimizer):
    return [
        {
            key: float(
                group[key].item()
                if torch.is_tensor(group[key])
                else group[key]
            )
            for key in ("lr", "beta1", "beta2", "eps", "weight_decay")
        }
        for group in optimizer.param_groups
    ]


# --- Finding 1: caller-stable, UNIQUE parameter identities -----------------


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_unnamed_synthesized_identities(tmp_path):
    _single_rank_group(tmp_path / "unnamed-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        # Bare (unnamed) parameters: Gefen synthesizes positional names
        # (group_0_param_0/1) that encode registration order, not identity, so
        # two reorderings would cross-assign momentum. Construction must refuse.
        first = _shaped_dparam((128, 64), mesh)
        second = _shaped_dparam((128, 64), mesh)
        optimizer = Gefen(
            [first, second], lr=1e-3, fused=False, factored_v_2d=False
        )
        with pytest.raises(RuntimeError, match="synthesized"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_duplicate_identities(tmp_path):
    _single_rank_group(tmp_path / "dup-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        first = _shaped_dparam((128, 64), mesh)
        second = _shaped_dparam((128, 64), mesh)
        optimizer = Gefen(
            [("alpha", first), ("beta", second)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        # Force two identical (name, group, shape) identities, as a corrupted or
        # adversarial caller could. Same-shaped same-group same-named slots cannot
        # be disambiguated on a resharded load, so construction must reject them.
        optimizer.param_groups[0]["param_names"] = ["dup", "dup"]
        with pytest.raises(RuntimeError, match="unique parameter identities"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


# --- Finding 2: validate group hyperparameters BEFORE committing -----------


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
@pytest.mark.parametrize(
    "key,bad_value,match",
    [
        ("lr", float("nan"), "not\\s+finite"),
        ("beta1", 1.0, "invalid beta1"),
        ("eps", -1e-8, "invalid eps"),
        ("weight_decay", -0.1, "invalid weight_decay"),
    ],
)
def test_dcp_rejects_corrupt_hyper_fail_atomic(tmp_path, key, bad_value, match):
    _single_rank_group(tmp_path / "corrupt-hyper-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)],
            lr=2.5e-3,
            betas=(0.8, 0.97),
            eps=2e-8,
            weight_decay=0.03,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()

        saved = GefenDCPState(optimizer).state_dict()
        # Corrupt one committed-in hyper. It is finite-but-invalid or NaN, and it
        # would silently poison the next update if committed.
        saved["param_group_hypers"][0][key] = bad_value

        state_before = _live_state_signature(optimizer, param)
        hypers_before = _group_hyper_signature(optimizer)
        with pytest.raises(ValueError, match=match):
            GefenDCPState(optimizer).load_state_dict(saved)
        # Fail-atomic: neither optimizer state nor param-group hypers moved.
        assert _live_state_signature(optimizer, param) == state_before
        assert _group_hyper_signature(optimizer) == hypers_before
    finally:
        dist.destroy_process_group()


# --- Finding 3: refresh cached _slots before save/load ---------------------


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_retained_wrapper_saves_added_param_group(tmp_path):
    _single_rank_group(tmp_path / "addgroup-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        first = _shaped_dparam(shape, mesh)
        optimizer = Gefen(
            [("alpha", first)], lr=1e-3, fused=False, factored_v_2d=False
        )
        # Retain the wrapper (as a training loop would), THEN grow the optimizer.
        wrapper = GefenDCPState(optimizer)
        second = _shaped_dparam(shape, mesh)
        optimizer.add_param_group({"params": [("beta", second)]})
        for step in range(_SAVE_STEPS):
            for param in (first, second):
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step), mesh, [Shard(0)]
                )
            optimizer.step()

        checkpoint_dir = str(tmp_path / "addgroup-ckpt")
        # Saving through the RETAINED wrapper must include the added parameter,
        # not the construction-time single-slot snapshot.
        saved = wrapper.state_dict()
        assert int(saved["slot_count"]) == 2, saved["slot_count"]
        torch.distributed.checkpoint.save(
            {"optimizer": wrapper},
            storage_writer=FileSystemWriter(checkpoint_dir),
        )

        # The resumed optimizer must mirror the saved group topology (alpha in
        # group 0, beta added as group 1) for the identity/group guard to accept.
        first2 = _shaped_dparam(shape, mesh)
        second2 = _shaped_dparam(shape, mesh)
        resumed = Gefen(
            [("alpha", first2)], lr=1e-3, fused=False, factored_v_2d=False
        )
        resumed.add_param_group({"params": [("beta", second2)]})
        torch.distributed.checkpoint.load(
            {"optimizer": GefenDCPState(resumed)},
            storage_reader=FileSystemReader(checkpoint_dir),
        )
        for param in (first2, second2):
            assert "automatic_period" in resumed.state[param], resumed.state[param]
            assert int(resumed.state[param]["automatic_period"]) >= 1
    finally:
        dist.destroy_process_group()


# --- Finding 4: co-locate the codebook with each slot's device -------------


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="mixed-device codebook coverage requires CUDA",
)
def test_reblock_relocates_codebook_to_operand_device():
    # Direct unit test (no distributed needed): a codebook learned on CUDA must
    # re-quantize a CPU-resident momentum shard. gefen_nearest_codebook_indices
    # rejects a codebook whose device differs from the operand, so without the
    # per-slot co-location _reblock would abort on mixed CPU/CUDA shards.
    import gefen.dcp as dcp_mod

    torch.manual_seed(0)
    period = 4
    cuda_momentum = torch.randn(256, device="cuda")
    codebook = dcp_mod._learn_codebook(
        [("w", cuda_momentum.reshape(-1), period)], torch.device("cuda")
    )
    assert codebook is not None and codebook.is_cuda

    cpu_momentum = torch.randn(256)
    cpu_second = torch.rand(256)
    m_codebook, m_magnitude, vmean = dcp_mod._reblock(
        codebook, cpu_momentum, cpu_second, period
    )
    assert m_codebook.device.type == "cpu"
    assert m_magnitude.device.type == "cpu" and vmean.device.type == "cpu"
    assert bool(torch.isfinite(m_magnitude).all())
    assert bool(torch.isfinite(vmean).all())
    # The original CUDA codebook must be untouched (co-location returns a copy).
    assert codebook.is_cuda


# --- Finding 5: cross-rank commit agreement --------------------------------


def _fail_sync_worker(rank, world, port, checkpoint_dir, mode, result_queue):
    try:
        os.environ.update(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=port,
            RANK=str(rank),
            WORLD_SIZE=str(world),
        )
        dist.init_process_group(
            "gloo", rank=rank, world_size=world, timeout=timedelta(seconds=120)
        )
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        device = torch.device("cpu")
        parameter, optimizer = _build(mesh, device)

        if mode == "save":
            for step in range(_SAVE_STEPS):
                _step(parameter, optimizer, mesh, step, device)
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
            )
            result_queue.put({"rank": rank, "saved": True})
            return

        # Give the live optimizer real state, then inject a one-rank re-block
        # failure. The cross-rank success sync must make EVERY rank raise and
        # leave EVERY rank's live state untouched (fail-atomic, no partial
        # restore) rather than committing on the ranks that did not fail.
        _step(parameter, optimizer, mesh, 0, device)
        before = _live_state_signature(optimizer, parameter)

        import gefen.dcp as dcp_mod

        if rank == 0:
            def _boom(*args, **kwargs):
                raise RuntimeError("injected one-rank re-block failure")

            dcp_mod._reblock = _boom

        raised = False
        try:
            torch.distributed.checkpoint.load(
                {"optimizer": GefenDCPState(optimizer)},
                storage_reader=FileSystemReader(checkpoint_dir),
            )
        except Exception:
            raised = True
        after = _live_state_signature(optimizer, parameter)
        result_queue.put(
            {
                "rank": rank,
                "raised": raised,
                "state_unchanged": after == before,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_fail_sync(world, checkpoint_dir, mode, timeout=240):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_fail_sync_worker,
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
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_one_rank_failure_aborts_every_rank(tmp_path):
    checkpoint_dir = str(tmp_path / "gefen-dcp-fail-sync")
    saved = _run_fail_sync(2, checkpoint_dir, "save")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run_fail_sync(2, checkpoint_dir, "load")
    assert all("fatal_error" not in item for item in loaded), loaded
    # A failure injected only on rank 0 must abort BOTH ranks and commit on
    # neither: every rank raised and every rank's live state is unchanged.
    for item in loaded:
        assert item["raised"], item
        assert item["state_unchanged"], item


# --- Finding 6: reject inconsistent initialization metadata/counters -------


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_initialized_slot_with_zero_step(tmp_path):
    _single_rank_group(tmp_path / "coherent-step-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()

        saved = GefenDCPState(optimizer).state_dict()
        # An initialized slot whose age counter collapsed to zero is incoherent
        # (native requires initialized momentum to carry a positive step).
        saved["slot_00000000.step"] = torch.tensor(0, dtype=torch.int64)

        before = _live_state_signature(optimizer, param)
        with pytest.raises(ValueError, match="initialized momentum requires step"):
            GefenDCPState(optimizer).load_state_dict(saved)
        assert _live_state_signature(optimizer, param) == before
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_uninitialized_slot_with_nonzero_counters(tmp_path):
    _single_rank_group(tmp_path / "coherent-uninit-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
            deterministic=True,
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()

        saved = GefenDCPState(optimizer).state_dict()
        # Flip the slot to uninitialized while it still carries a nonzero step and
        # dense momentum: native never materializes an uninitialized slot with
        # history, so the load must reject rather than silently discard it.
        saved["slot_00000000.initialized"] = torch.tensor(0, dtype=torch.uint8)

        before = _live_state_signature(optimizer, param)
        with pytest.raises(ValueError, match="uninitialized"):
            GefenDCPState(optimizer).load_state_dict(saved)
        assert _live_state_signature(optimizer, param) == before
    finally:
        dist.destroy_process_group()


# --- Post-merge review round (Codex, after PR #83) -------------------------


def _reference_dense_momentum(optimizer, parameter, state):
    """The pre-fix ``_dense_momentum``: whole-tensor advanced indexing.

    Kept verbatim as the numerics oracle. The chunked implementation that
    replaced it is a *memory* fix and must reproduce this bit-for-bit.
    """
    indices = state["m_codebook"].reshape(-1).long()
    magnitude = state["m_magnitude"].reshape(-1).float()
    codebook = optimizer._gefen_codebook.detach().to(
        device=indices.device, dtype=torch.float32
    )
    period = int(state["automatic_period"])
    return (
        codebook[indices]
        .reshape(-1, period)
        .mul(magnitude.reshape(-1, 1))
        .reshape(_local(parameter).shape)
    )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
@pytest.mark.parametrize("chunk", [None, 97])
def test_dense_momentum_is_bit_identical_to_advanced_indexing(tmp_path, chunk, monkeypatch):
    # The chunked dequantize must not move the saved momentum at all -- neither
    # in one chunk (the default constant covers these shards whole) nor across
    # many. The 97-element chunk is deliberately not a divisor of the shard or of
    # the block period, so chunk boundaries fall mid-block.
    import gefen.dcp as dcp_mod
    import gefen.gefen as gefen_mod

    _single_rank_group(tmp_path / "dense-parity-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()

        state = optimizer.state[param]
        expected = _reference_dense_momentum(optimizer, param, state)

        if chunk is not None:
            monkeypatch.setattr(gefen_mod, "GEFEN_DEQUANT_GATHER_CHUNK", chunk)
        actual = dcp_mod._dense_momentum(optimizer, param, state)

        assert actual.shape == expected.shape
        assert actual.dtype == torch.float32
        assert torch.equal(actual, expected), (
            (actual - expected).abs().max().item()
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="peak-allocation measurement requires CUDA",
)
def test_dense_momentum_transient_is_bounded():
    # Direct unit test (no distributed needed). The pre-fix form built a
    # full-size int64 index copy (8 bytes/param) and a full-size fp32 gather on
    # top of the 4-byte/param result -- ~16 bytes/param of transient, which OOMs
    # an otherwise-viable save on a large shard. The chunked form keeps the
    # int64/gather scratch bounded to one chunk regardless of shard size, so the
    # transient is the 4-byte/param result plus a constant.
    import gefen.dcp as dcp_mod

    class _FakeOptimizer:
        def __init__(self, codebook):
            self._gefen_codebook = codebook

    numel = 1 << 26  # 64M elements: many times the bounded chunk
    period = 64
    torch.manual_seed(0)
    optimizer = _FakeOptimizer(torch.sort(torch.randn(256, device="cuda"))[0].float())
    state = {
        "m_codebook": torch.randint(
            0, 256, (numel // period, period), dtype=torch.uint8, device="cuda"
        ),
        "m_magnitude": torch.rand(numel // period, 1, device="cuda"),
        "automatic_period": period,
    }
    param = torch.empty(numel // 128, 128, device="cuda", dtype=torch.bfloat16)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    dense = dcp_mod._dense_momentum(optimizer, param, state)
    torch.cuda.synchronize()
    peak_bytes_per_elem = (torch.cuda.max_memory_allocated() - base) / numel

    assert dense.numel() == numel
    # The 4-byte/param fp32 result is *kept*, so it is the floor; the pre-fix
    # form peaked at ~16. Anything at/above 8 means a full-size int64 index copy
    # (or a second full-size gather) is live again.
    assert peak_bytes_per_elem < 8.0, peak_bytes_per_elem


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_orphaned_vmean_step_at_save(tmp_path):
    _single_rank_group(tmp_path / "orphan-vmean-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()

        # A slot carrying only its name and an orphaned vmean_step: a partial
        # materialize. It was classified as wholly fresh, so save wrote an
        # "uninitialized" slot with a nonzero counter -- a checkpoint every later
        # load rejects as incoherent. It must be caught at SAVE instead.
        optimizer.state[param] = {"name": "weight", "vmean_step": 5}
        with pytest.raises(ValueError, match="partial optimizer state"):
            GefenDCPState(optimizer).state_dict()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_saves_legacy_state_without_vmean_step(tmp_path):
    _single_rank_group(tmp_path / "legacy-vmean-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()

        # A state restored from a checkpoint written before the vmean_step counter
        # existed carries vmean without it; native backfills it from `step` at
        # step time. vmean_step marks a materialize but is NOT required, so such a
        # slot must still save as a complete, initialized slot.
        del optimizer.state[param]["vmean_step"]
        saved = GefenDCPState(optimizer).state_dict()
        assert int(saved["slot_00000000.initialized"]) == 1
        assert int(saved["slot_00000000.vmean_step"]) == int(
            saved["slot_00000000.step"]
        )
        GefenDCPState(optimizer).load_state_dict(saved)
        assert "automatic_period" in optimizer.state[param]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_integral_tensor_lr_before_commit(tmp_path):
    _single_rank_group(tmp_path / "int-lr-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()
        saved = GefenDCPState(optimizer).state_dict()

        # An integral one-element lr tensor cannot hold the checkpoint's 1e-3:
        # the in-place fill would silently truncate it to 0 and freeze every
        # later update -- and it ran after the state was already replaced.
        optimizer.param_groups[0]["lr"] = torch.tensor(1, dtype=torch.int64)
        state_before = _live_state_signature(optimizer, param)
        hypers_before = _group_hyper_signature(optimizer)

        with pytest.raises(ValueError, match="cannot represent"):
            GefenDCPState(optimizer).load_state_dict(saved)

        # Fail-atomic: rejected during staging, so neither the optimizer state nor
        # the param groups moved, and the lr was never truncated.
        assert _live_state_signature(optimizer, param) == state_before
        assert _group_hyper_signature(optimizer) == hypers_before
        assert int(optimizer.param_groups[0]["lr"]) == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_accepts_float_tensor_lr(tmp_path):
    _single_rank_group(tmp_path / "float-lr-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=2.5e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(_shaped_grad(_SHAPE, step), mesh, [Shard(0)])
            optimizer.step()
        saved = GefenDCPState(optimizer).state_dict()

        # A float tensor lr is the supported live form (fused kernels hold a
        # reference to it): it must still restore in place, identity preserved.
        live_lr = torch.tensor(9.9e-9, dtype=torch.float32)
        optimizer.param_groups[0]["lr"] = live_lr
        GefenDCPState(optimizer).load_state_dict(saved)
        assert optimizer.param_groups[0]["lr"] is live_lr
        assert abs(float(live_lr) - 2.5e-3) < 1e-9, float(live_lr)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_accepts_caller_named_param_0(tmp_path):
    _single_rank_group(tmp_path / "real-param0-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        first = _shaped_dparam(shape, mesh)
        second = _shaped_dparam(shape, mesh)
        # A model may genuinely declare a parameter called "param_0". Its name
        # comes from named_parameters(), so it IS caller-stable and unique --
        # rejecting it on spelling alone locked such an optimizer out of
        # GefenDCPState entirely. Provenance, not the pattern, decides.
        optimizer = Gefen(
            [("param_0", first), ("group_1_param_0", second)],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        wrapper = GefenDCPState(optimizer)
        assert [slot.name for slot in wrapper._slots] == ["param_0", "group_1_param_0"]

        for step in range(_SAVE_STEPS):
            for param in (first, second):
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step), mesh, [Shard(0)]
                )
            optimizer.step()
        # It must round-trip, not merely construct.
        saved = wrapper.state_dict()
        GefenDCPState(optimizer).load_state_dict(saved)
        for param in (first, second):
            assert "automatic_period" in optimizer.state[param]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_still_rejects_genuinely_synthesized_names_in_named_group(tmp_path):
    _single_rank_group(tmp_path / "synth-group-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        first = _shaped_dparam((128, 64), mesh)
        second = _shaped_dparam((128, 64), mesh)
        # Unnamed params inside an EXPLICIT group get group_0_param_N. Gefen
        # generated those, so they still encode registration order and must be
        # refused however they are spelled.
        optimizer = Gefen(
            [{"params": [first, second]}], lr=1e-3, fused=False, factored_v_2d=False
        )
        assert optimizer.param_groups[0]["param_names"] == [
            "group_0_param_0",
            "group_0_param_1",
        ]
        with pytest.raises(RuntimeError, match="synthesized"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_synthesized_name_added_after_construction(tmp_path):
    _single_rank_group(tmp_path / "synth-added-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        first = _shaped_dparam(shape, mesh)
        optimizer = Gefen(
            [("alpha", first)], lr=1e-3, fused=False, factored_v_2d=False
        )
        # A group added later without names is synthesized just the same, and the
        # provenance record must follow it through add_param_group.
        second = _shaped_dparam(shape, mesh)
        optimizer.add_param_group({"params": [second]})
        assert optimizer._param_name_is_synthesized(second)
        assert not optimizer._param_name_is_synthesized(first)
        with pytest.raises(RuntimeError, match="synthesized"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


def test_second_moment_reblocking_regimes():
    # Pins the two regimes the resume guarantee rests on (direct unit test, no
    # distributed needed). The saved dense second moment is only each source
    # block's vmean repeated per element -- the per-element history never existed.
    #
    #   aligned / refining (target period divides the source period): every target
    #     block sits inside one source block, so averaging identical repeated
    #     values returns them unchanged, to fp32 round-off.
    #   coarsening (target block spans several source blocks): their distinct
    #     vmeans are averaged together, irreversibly. Bounded only by the spread
    #     of vmean across the merged blocks, NOT by quantization noise.
    import gefen.dcp as dcp_mod

    source_period = 64
    blocks = 16
    numel = source_period * blocks
    # A vmean that genuinely varies block to block, as a real layer's does.
    vmean = torch.logspace(-6, 0, blocks).reshape(-1, 1)
    dense_second = vmean.expand(-1, source_period).reshape(-1).clone()
    momentum = torch.linspace(-1.0, 1.0, numel)
    codebook = dcp_mod._learn_codebook(
        [("w", momentum, source_period)], torch.device("cpu")
    )

    def reblocked_second(period):
        _, _, new_vmean = dcp_mod._reblock(codebook, momentum, dense_second, period)
        return new_vmean.reshape(-1, 1).expand(-1, period).reshape(-1)

    # Aligned: recovered to fp32 round-off (the block mean sums then divides).
    assert torch.allclose(
        reblocked_second(source_period), dense_second, rtol=1e-6, atol=0
    )
    # Refining: still exact -- each sub-block sees one constant value.
    assert torch.allclose(
        reblocked_second(source_period // 4), dense_second, rtol=1e-6, atol=0
    )

    # Coarsening: four source blocks per target block get averaged into one
    # value. The result is materially wrong, in a way no re-quantization
    # tolerance covers, and cannot be undone.
    coarse = reblocked_second(source_period * 4)
    relative = (coarse - dense_second).abs().max() / dense_second.abs().max()
    assert relative > 0.1, relative
