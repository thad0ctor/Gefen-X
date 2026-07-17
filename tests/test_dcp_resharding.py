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
re-aggregating the per-block second moment when the target blocking lands finer
than, or off the boundaries of, the source's -- is pinned separately by
test_second_moment_reblocking_regimes. See _CONTINUE_TOL.
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

from gefen import Gefen, GefenDCPState, GefenSavePlanner


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
# search lands elsewhere) can *refine* the blocking, or land off the source's
# boundaries; load must then spread one stored vmean across sub-blocks a natively
# blocked run would have given distinct values, and that detail was never in the
# checkpoint. Bounded only by how much the second moment varies inside a source
# block -- unrelated to, and measurably larger than, quantization noise.
# (Coarsening onto source boundaries is not lossy: block means average into the
# mean over their union.) See test_second_moment_reblocking_regimes.
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
        # adversarial caller could. The global name rule catches this first: a
        # duplicate identity implies a duplicate name, so it reports the name.
        optimizer.param_groups[0]["param_names"] = ["dup", "dup"]
        with pytest.raises(RuntimeError, match="unique across every group"):
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


def _fail_sync_worker(
    rank, world, port, checkpoint_dir, mode, result_queue, failure="reblock"
):
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

        if mode == "save_fail":
            # Malform ONE rank's rank-local state, then save. state_dict() runs
            # while dcp.save is still converting Stateful objects -- before its
            # planning collectives -- so an unsynchronized raise here drops rank 0
            # out while rank 1 walks into collectives rank 0 will never join.
            for step in range(_SAVE_STEPS):
                _step(parameter, optimizer, mesh, step, device)
            if rank == 0:
                if failure == "geometry":
                    # Block geometry that no longer tiles the local shard.
                    state = optimizer.state[parameter]
                    state["m_magnitude"] = state["m_magnitude"][:-1].clone()
                else:
                    optimizer._gefen_codebook = None
            raised = False
            error = None
            try:
                torch.distributed.checkpoint.save(
                    {"optimizer": GefenDCPState(optimizer)},
                    storage_writer=FileSystemWriter(checkpoint_dir),
                )
            except Exception as exc:
                raised = True
                # Keep the message: "some rank raised" is not the claim under
                # test -- WHICH error, on WHICH rank, is.
                error = repr(exc)
            result_queue.put({"rank": rank, "raised": raised, "error": error})
            return

        # Give the live optimizer real state, then inject a one-rank re-block
        # failure. The cross-rank success sync must make EVERY rank raise and
        # leave EVERY rank's live state untouched (fail-atomic, no partial
        # restore) rather than committing on the ranks that did not fail.
        _step(parameter, optimizer, mesh, 0, device)
        before = _live_state_signature(optimizer, parameter)

        import gefen.dcp as dcp_mod

        if rank == 0:
            if failure == "reblock":
                # Fails INSIDE the synchronized staging block.
                def _boom(*args, **kwargs):
                    raise RuntimeError("injected one-rank re-block failure")

                dcp_mod._reblock = _boom
            else:
                # Fails in rank-local validation, which runs BEFORE staging: an
                # integral tensor LR destination cannot hold the checkpoint's
                # float lr. Only this rank sees it -- the peers' destinations are
                # fine -- so the validation must still participate in the success
                # collective. If it raises outside it, rank 0 exits while rank 1
                # blocks in the collective forever and this test hangs.
                optimizer.param_groups[0]["lr"] = torch.tensor(0, dtype=torch.long)

        raised = False
        error = None
        try:
            torch.distributed.checkpoint.load(
                {"optimizer": GefenDCPState(optimizer)},
                storage_reader=FileSystemReader(checkpoint_dir),
            )
        except Exception as exc:
            raised = True
            # Keep the message: "some rank raised" is not the claim under test --
            # WHICH error, on WHICH rank, is.
            error = repr(exc)
        after = _live_state_signature(optimizer, parameter)
        result_queue.put(
            {
                "rank": rank,
                "raised": raised,
                "error": error,
                "state_unchanged": after == before,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_fail_sync(world, checkpoint_dir, mode, timeout=240, failure="reblock"):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_fail_sync_worker,
            args=(rank, world, port, checkpoint_dir, mode, result_queue, failure),
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


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_one_rank_validation_failure_aborts_every_rank_cleanly(tmp_path):
    # Rank-local validation (it reads the LIVE optimizer) can fail on ONE rank
    # and pass on the others, so it has to participate in the success collective.
    # Hoisted out of the synchronized block it raises while the peers are still
    # heading into that collective -- which the failing rank then never joins.
    #
    # Asserting only "every rank raised" does NOT catch that: with the check
    # hoisted out, rank 1 still raises, but with
    #   RuntimeError(gloo ... Connection closed by peer)
    # -- it escapes the collective only because rank 0 tore its process group
    # down on the way out. That is the *lucky* end of this bug; a failing rank
    # that keeps its group alive (a training loop that catches the error, or
    # NCCL) leaves its peers blocked in the collective instead. So pin the thing
    # that is actually load-bearing: WHICH error each rank gets.
    #
    # The reblock-injection test above cannot cover this -- it fails inside
    # staging, which is already synchronized.
    checkpoint_dir = str(tmp_path / "gefen-dcp-fail-sync-validation")
    saved = _run_fail_sync(2, checkpoint_dir, "save", failure="destination")
    assert all(item.get("saved") for item in saved), saved
    loaded = _run_fail_sync(2, checkpoint_dir, "load", timeout=90, failure="destination")
    assert all("fatal_error" not in item for item in loaded), loaded
    for item in loaded:
        assert item["raised"], item
        assert item["state_unchanged"], item
    by_rank = {item["rank"]: item for item in loaded}
    # The rank that actually has the bad destination reports the real cause...
    assert "cannot represent it" in by_rank[0]["error"], by_rank[0]
    # ...and its peer gets the group-wide abort, not a transport-level symptom.
    assert "aborted before commit" in by_rank[1]["error"], by_rank[1]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
@pytest.mark.parametrize("failure", ["codebook", "geometry"])
def test_dcp_one_rank_save_validation_failure_aborts_every_rank_cleanly(
    tmp_path, failure
):
    # The SAVE mirror of the load-side test above. state_dict()'s checks read
    # rank-local state (the live codebook, each slot's block geometry), so they
    # can fail on one rank and pass on the others -- and dcp.save calls
    # state_dict() while converting Stateful objects, BEFORE entering its own
    # planning collectives. Unsynchronized, rank 0 raises and leaves; rank 1
    # proceeds into planning collectives that rank 0 will never join.
    #
    # Asserting only "every rank raised" does NOT catch that. Unsynchronized,
    # rank 1 still raises here -- but with
    #   RuntimeError(gloo ... Connection closed by peer)
    # because this worker's `finally` tears rank 0's process group down on the
    # way out. That teardown is an artifact of the test, not a guarantee: a
    # training loop that catches the error and keeps its group alive leaves the
    # peers blocked in the collective instead. So pin WHICH error each rank gets.
    checkpoint_dir = str(tmp_path / "gefen-dcp-save-fail-sync")
    saved = _run_fail_sync(2, checkpoint_dir, "save_fail", timeout=90, failure=failure)
    assert all("fatal_error" not in item for item in saved), saved
    for item in saved:
        assert item["raised"], item
    by_rank = {item["rank"]: item for item in saved}
    # The rank that actually holds the malformed state reports the real cause...
    expected = (
        "requires an initialized codebook"
        if failure == "codebook"
        else "block geometry is invalid"
    )
    assert expected in by_rank[0]["error"], by_rank[0]
    # ...and its peer gets the group-wide abort, not a transport-level symptom.
    assert "aborted before any rank proceeds" in by_rank[1]["error"], by_rank[1]


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
    not torch.cuda.is_available(),
    reason="peak-allocation measurement requires CUDA",
)
def test_dense_second_moment_transient_is_bounded():
    # Direct unit test (no distributed needed). The pre-fix form was
    # expand(...).reshape(...).clone(): reshaping a stride-0 expand cannot return
    # a view, so it already built a full 4-byte/param copy that the clone then
    # copied a SECOND time -- 8 bytes/param live to produce a 4-byte/param
    # result. Filling one preallocated buffer is single-copy.
    import gefen.dcp as dcp_mod

    numel = 1 << 24
    period = 64
    torch.manual_seed(0)
    state = {
        "vmean": torch.rand(numel // period, device="cuda"),
        "automatic_period": period,
    }
    param = torch.empty(numel // 128, 128, device="cuda", dtype=torch.bfloat16)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    dense = dcp_mod._dense_second_moment(param, state)
    torch.cuda.synchronize()
    peak_bytes_per_elem = (torch.cuda.max_memory_allocated() - base) / numel

    assert dense.numel() == numel
    # The 4-byte/param fp32 result is kept, so it is the floor; the pre-fix form
    # peaked at exactly 8. Anything at/above 6 means a second full-size copy is
    # live again.
    assert peak_bytes_per_elem < 6.0, peak_bytes_per_elem


def test_dense_second_moment_does_not_alias_live_vmean_at_period_one():
    # At period 1 the expand is a no-op and the reshape returns a VIEW of the
    # live vmean state -- the clone the pre-fix form ended with was what kept the
    # dense save from aliasing (and a later in-place re-block from corrupting)
    # the optimizer's own second moment. The preallocated fill must own its
    # memory at every period, including this one.
    import gefen.dcp as dcp_mod

    original = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state = {"vmean": original.clone(), "automatic_period": 1}
    param = torch.empty(4)

    dense = dcp_mod._dense_second_moment(param, state)
    assert torch.equal(dense, original)
    dense.add_(1.0)
    # Mutating the dense expansion must not touch the optimizer's live state.
    assert torch.equal(state["vmean"], original), state["vmean"]


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_available() or not dist.is_gloo_available(),
    reason="peak-allocation measurement requires CUDA and Gloo",
)
def test_dcp_save_peak_is_bounded_by_one_slot(tmp_path):
    # The save must expand Gefen's ~1 byte/param block state to a dense 4
    # byte/param momentum + 4 byte/param second moment, because the momentum
    # codebook is learned per rank. The pre-fix state_dict() built that dense
    # pair for EVERY slot up front and handed DCP a finished dict, so all of it
    # -- 8 bytes/param, eight times the optimizer's resident state -- stayed live
    # for the whole write, peaking higher than a training step.
    #
    # DCP resolves one write item at a time, so GefenSavePlanner caps the
    # transient at a single slot's dense form (plus the dequantize's gather
    # scratch, which is bounded by the gather chunk rather than by the total
    # state, so it does not scale with the slot count). Sized so that a
    # regression to all-slots-live is unambiguous: 8 equal slots means eager
    # retention is ~8x this bound.
    _single_rank_group(tmp_path / "save-peak-init")
    try:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        slots = 8
        shape = (2048, 1024)  # 2M params per slot; 16M total
        named = []
        for index in range(slots):
            named.append(
                ("w{}".format(index), _shaped_dparam(shape, mesh, device="cuda"))
            )
        optimizer = Gefen(
            named, lr=1e-3, fused=False, factored_v_2d=False, deterministic=True
        )
        numel = sum(param.numel() for _, param in named)
        for step in range(_SAVE_STEPS):
            for _, param in named:
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step).cuda(), mesh, [Shard(0)]
                )
            optimizer.step()
        for _, param in named:
            param.grad = None

        # state_dict() itself must materialize nothing at all.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        saved = GefenDCPState(optimizer).state_dict()
        torch.cuda.synchronize()
        held = (torch.cuda.memory_allocated() - base) / numel
        assert held < 0.1, held
        del saved

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(str(tmp_path / "save-peak-ckpt")),
            planner=GefenSavePlanner(),
        )
        torch.cuda.synchronize()
        peak_bytes_per_elem = (torch.cuda.max_memory_allocated() - base) / numel

        # Eager retention is >= 8 (dense momentum + dense second moment for every
        # slot). One slot's dense pair is 8/slots = 1, and the gather scratch
        # adds ~24 MiB (12 bytes per element of a 2M-element slot, under the 8M
        # chunk) = 1.5 here; 4 separates the two regimes with room to spare on
        # either side.
        assert peak_bytes_per_elem < 4.0, peak_bytes_per_elem
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_available() or not dist.is_gloo_available(),
    reason="peak-allocation measurement requires CUDA and Gloo",
)
def test_dcp_save_peak_is_bounded_under_a_multi_threaded_writer(tmp_path):
    # thread_count is a public FileSystemWriter option, and it splits the plan
    # into one file bucket per thread, so N threads call resolve_data at once.
    # Resolving is what expands a slot, so without serialization the bound is one
    # slot's dense form PER THREAD: at thread_count=8 this same model peaked at
    # ~8 bytes/param, the full eager cost the planner exists to avoid, silently
    # undoing the bound for anyone who set a throughput option.
    #
    # Serializing alone would not fix it -- the writer keeps what resolve_data
    # returns until it has written that item, so a device-side return outlives
    # any lock. resolve_data returns the shard already on the CPU, which is what
    # makes the device-side expansion dead before the next thread's begins.
    #
    # Deliberately thread_count=8 rather than 2: whether two threads overlap in
    # resolve_data at all is a race, so a small count measures the scheduler. At
    # 8 the unbounded peak is unambiguous, while the bounded one stays flat.
    _single_rank_group(tmp_path / "save-peak-threads-init")
    try:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        slots = 8
        shape = (2048, 1024)  # 2M params per slot; 16M total
        named = []
        for index in range(slots):
            named.append(
                ("w{}".format(index), _shaped_dparam(shape, mesh, device="cuda"))
            )
        optimizer = Gefen(
            named, lr=1e-3, fused=False, factored_v_2d=False, deterministic=True
        )
        numel = sum(param.numel() for _, param in named)
        for step in range(_SAVE_STEPS):
            for _, param in named:
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step).cuda(), mesh, [Shard(0)]
                )
            optimizer.step()
        for _, param in named:
            param.grad = None

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(
                str(tmp_path / "save-peak-threads-ckpt"), thread_count=8
            ),
            planner=GefenSavePlanner(),
        )
        torch.cuda.synchronize()
        peak_bytes_per_elem = (torch.cuda.max_memory_allocated() - base) / numel

        # The same threshold the single-threaded bound is held to above: it
        # separates "one slot's dense form plus gather scratch" from any regime
        # that keeps more than one slot live.
        assert peak_bytes_per_elem < 4.0, peak_bytes_per_elem
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not dist.is_available()
    or not dist.is_gloo_available()
    or not hasattr(torch.cuda, "host_memory_stats"),
    # host_memory_stats lands after the 2.5.0 floor, and it is the only way to
    # read page-locked bytes: cudaHostAlloc'd pages are not accounted in
    # /proc VmLck or VmPin, so there is nothing to fall back to.
    reason="page-locked accounting requires CUDA, Gloo, and torch.cuda.host_memory_stats",
)
def test_dcp_save_page_locks_one_slot_at_a_time(tmp_path):
    # The writer decides how long it keeps what resolve_data returns, and torch's
    # keeps every tensor it has written until it closes the file. So whatever the
    # save hands back is held for the whole write, and handing back page-locked
    # memory page-locks the entire dense state dict -- 8 bytes/param the host
    # cannot swap, and cannot reclaim afterwards either, since the caching host
    # allocator never returns blocks to the OS. Copies go through one reusable
    # page-locked buffer instead and come back pageable, which keeps the
    # page-locked high-water at a single slot's dense form.
    #
    # thread_count=8 rather than the default 1 because torch pins on its own at
    # thread_count == 1: _OverlappingCpuLoader copies with
    # `.to("cpu", non_blocking=True)`, whose destination is page-locked, so the
    # measurement could not tell torch's page-locking from Gefen's. Above 1 torch
    # uses _SerialCpuLoader, whose `.cpu()` is pageable and leaves this
    # attributable to the save path alone.
    _single_rank_group(tmp_path / "save-pinned-init")
    try:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        slots = 8
        shape = (2048, 1024)  # 2M params per slot; 16M total
        named = [
            ("w{}".format(index), _shaped_dparam(shape, mesh, device="cuda"))
            for index in range(slots)
        ]
        optimizer = Gefen(
            named, lr=1e-3, fused=False, factored_v_2d=False, deterministic=True
        )
        numel = sum(param.numel() for _, param in named)
        for step in range(_SAVE_STEPS):
            for _, param in named:
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step).cuda(), mesh, [Shard(0)]
                )
            optimizer.step()
        for _, param in named:
            param.grad = None

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_host_memory_stats()
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(
                str(tmp_path / "save-pinned-ckpt"), thread_count=8
            ),
            planner=GefenSavePlanner(),
        )
        torch.cuda.synchronize()
        pinned_peak = torch.cuda.host_memory_stats().get("allocated_bytes.peak", 0)
        pinned_bytes_per_elem = pinned_peak / numel

        # One slot's dense form is 4 bytes/param over that slot, which is 0.5
        # bytes/param spread over this 8-slot model; page-locking every slot's
        # instead is the full 8. The threshold separates the two regimes without
        # pinning the exact staging buffer size.
        assert pinned_bytes_per_elem < 4.0, pinned_bytes_per_elem
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not dist.is_available()
    or not dist.is_gloo_available()
    or not hasattr(torch.cuda, "host_memory_stats"),
    reason="page-locked accounting requires CUDA, Gloo, and torch.cuda.host_memory_stats",
)
def test_dcp_save_page_locks_one_slot_with_ascending_slot_sizes(tmp_path):
    # The one-slot page-locked bound must not depend on the slots happening to be
    # the same size. Sizing the staging buffer on demand does depend on it: torch's
    # CachingHostAllocator keeps freed pinned blocks in its own cache rather than
    # returning them to the OS, so every fresh maximum ADDS a page-locked block and
    # the footprint climbs toward the sum of the successive maxima.
    #
    # Ascending is not a contrived order -- it is the only order this writer uses.
    # At the default thread_count=1 torch loads through _OverlappingCpuLoader, which
    # sorts every write item by size and resolves them smallest first, so a model
    # with N distinct parameter sizes walks the maxima in exactly the order that
    # allocates all N of them. (Above 1 thread, _split_by_size_and_type sorts
    # DESCENDING into per-thread buckets, so the largest slot lands first and sizes
    # the buffer immediately -- which is why the uniform, thread_count=8 test above
    # cannot see this, and why this one pins thread_count=1.)
    #
    # Asserts the allocation count, not just the high-water: the pinned high-water
    # alone is measured against an allocator whose cached blocks are shared with
    # every other test in this process, so it can be satisfied by reuse. "Exactly
    # one buffer, sized to the largest slot" is the property itself, and is
    # deterministic.
    import gefen.dcp as dcp_mod

    _single_rank_group(tmp_path / "save-pinned-ascending-init")
    try:
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        # Dense fp32 slots of 4, 8, 16 and 32 MiB: strictly ascending, so growth
        # on demand pins 4+8+16+32 = 60 MiB against a 32 MiB largest slot.
        shapes = [(1024, 1024), (2048, 1024), (4096, 1024), (8192, 1024)]
        named = [
            ("w{}".format(index), _shaped_dparam(shape, mesh, device="cuda"))
            for index, shape in enumerate(shapes)
        ]
        optimizer = Gefen(
            named, lr=1e-3, fused=False, factored_v_2d=False, deterministic=True
        )
        for step in range(_SAVE_STEPS):
            for (_, param), shape in zip(named, shapes):
                param.grad = distribute_tensor(
                    _shaped_grad(shape, step).cuda(), mesh, [Shard(0)]
                )
            optimizer.step()
        for _, param in named:
            param.grad = None

        largest_slot_bytes = max(param.numel() for _, param in named) * 4

        # Record every pinned block the staging buffer allocates during the save.
        pinned_blocks = []
        real_expand = dcp_mod._PinnedStaging.expand

        def traced_expand(self, dense):
            before = self._buffer.data_ptr() if self._buffer is not None else None
            result = real_expand(self, dense)
            if self._buffer is not None and self._buffer.data_ptr() != before:
                pinned_blocks.append(self._buffer.numel())
            return result

        dcp_mod._PinnedStaging.expand = traced_expand
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_host_memory_stats()
        before_pinned = torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)
        try:
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(
                    str(tmp_path / "save-pinned-ascending-ckpt"), thread_count=1
                ),
                planner=GefenSavePlanner(),
            )
        finally:
            dcp_mod._PinnedStaging.expand = real_expand
        torch.cuda.synchronize()

        # One buffer for the whole save, sized to the largest slot -- not one per
        # new maximum. Growth on demand allocates [4, 8, 16, 32] MiB here.
        assert pinned_blocks == [largest_slot_bytes], [
            block // 1024**2 for block in pinned_blocks
        ]

        # And the allocator agrees: the page-locked high-water over the save is one
        # slot's dense form, not the 60 MiB sum of the maxima.
        pinned_peak = torch.cuda.host_memory_stats().get("allocated_bytes.peak", 0)
        assert pinned_peak - before_pinned <= largest_slot_bytes, (
            pinned_peak - before_pinned,
            largest_slot_bytes,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_save_without_planner_is_still_correct(tmp_path):
    # GefenSavePlanner is what makes the save memory-BOUNDED, not what makes it
    # correct. A caller who forgets it must still get a checkpoint that restores
    # exactly what the planner-bounded save would have written -- the stand-ins
    # fall back to expanding and retaining their dense form (the pre-fix memory
    # profile). A silently wrong checkpoint would be far worse than a hungry one.
    _single_rank_group(tmp_path / "no-planner-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        planned_dir = str(tmp_path / "planned")
        plain_dir = str(tmp_path / "plain")

        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()

        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(planned_dir),
            planner=GefenSavePlanner(),
        )
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(plain_dir),
        )

        # Both checkpoints must restore byte-identical optimizer state.
        signatures = []
        for checkpoint_dir in (planned_dir, plain_dir):
            target_param = _shaped_dparam(_SHAPE, mesh)
            target = Gefen(
                [("weight", target_param)], lr=1e-3, fused=False, factored_v_2d=False
            )
            wrapper = GefenDCPState(target)
            state = wrapper.state_dict()
            torch.distributed.checkpoint.load(
                {"optimizer": wrapper},
                storage_reader=FileSystemReader(checkpoint_dir),
            )
            del state
            restored = target.state[target_param]
            signatures.append(
                (
                    int(restored["automatic_period"]),
                    int(restored["step"]),
                    restored["m_codebook"].clone(),
                    restored["m_magnitude"].clone(),
                    restored["vmean"].clone(),
                )
            )

        planned, plain = signatures
        assert planned[:2] == plain[:2], (planned[:2], plain[:2])
        for left, right in zip(planned[2:], plain[2:]):
            assert torch.equal(left, right), (left - right).abs().max()
    finally:
        dist.destroy_process_group()


def test_dcp_save_planner_rejects_unflattened_state_dict():
    # flatten_state_dict is a public DefaultSavePlanner option GefenSavePlanner
    # inherits, but a GefenDCPState cannot be saved without flattening: DCP only
    # descends into a nested mapping when it flattens it, so unflattened the whole
    # {"optimizer": ...} mapping becomes ONE BYTE_IO write item that DCP pickles
    # wholesale. The stand-ins' own write items are never created, and pickling
    # reaches _LazyDenseShard.__reduce_ex__, which refuses. Reject at construction
    # rather than let that surface as an opaque TypeError mid-write.
    with pytest.raises(ValueError) as excinfo:
        GefenSavePlanner(flatten_state_dict=False)
    message = str(excinfo.value)
    # The error has to name the real constraint, not just restate the flag.
    assert "flatten_state_dict=True" in message, message
    assert "stand-in" in message, message

    # Positional passing is the same option and must be rejected identically.
    with pytest.raises(ValueError):
        GefenSavePlanner(False)

    # The default, and an explicit True, must both still construct.
    assert GefenSavePlanner().flatten_state_dict is True
    assert GefenSavePlanner(flatten_state_dict=True).flatten_state_dict is True


def _has_checkpoint_bytes(checkpoint_dir):
    # A DCP checkpoint is loadable only with its .metadata; .distcp files are the
    # shard payloads. Either one means the save got past planning and put
    # something on disk.
    if not os.path.isdir(checkpoint_dir):
        return False
    names = os.listdir(checkpoint_dir)
    return any(
        name == ".metadata" or name.endswith(".distcp") for name in names
    )


def _restored_signature(mesh, checkpoint_dir):
    """Load ``checkpoint_dir`` into a fresh optimizer and snapshot its state."""
    target_param = _shaped_dparam(_SHAPE, mesh)
    target = Gefen(
        [("weight", target_param)], lr=1e-3, fused=False, factored_v_2d=False
    )
    torch.distributed.checkpoint.load(
        {"optimizer": GefenDCPState(target)},
        storage_reader=FileSystemReader(checkpoint_dir),
    )
    restored = target.state[target_param]
    return (
        int(restored["automatic_period"]),
        int(restored["step"]),
        restored["m_codebook"].clone(),
        restored["m_magnitude"].clone(),
        restored["vmean"].clone(),
    )


def _await_async_save(response):
    # torch 2.5 returns a bare Future; newer torch returns an AsyncSaveResponse
    # holding one. Wait on whichever it is, so a writer-thread failure surfaces
    # here instead of after the test has moved on.
    future = getattr(response, "upload_completion", response)
    future.result()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_async_save_without_the_gefen_writer_is_refused(tmp_path):
    # async_save stages the state dict to the CPU and returns, then writes from a
    # background thread; GefenFileSystemWriter is what stages a GefenDCPState
    # correctly (test_dcp_async_save_with_gefen_writer_restores_like_the_sync_save
    # and ..._snapshots_the_state_a_later_step_mutates cover that path). This
    # pins the other half of the contract: the DEFAULT writer must fail loudly
    # and leave nothing behind, on every torch version and every device.
    #
    # It has to be a refusal rather than "either outcome is safe", because the
    # one combination that used to get through staging untouched was the
    # dangerous one, not the benign one. Torch 2.5 stages with `.to(cpu_device)`,
    # which short-circuits to the same object for CPU-resident state -- so the
    # stand-ins entered the "staged" dict by reference, nothing was copied, and
    # the writer thread expanded them against LIVE optimizer state while training
    # continued. That writes a tear: save-time counters beside post-step
    # momentum, silently. It looked correct only while the state was quiescent,
    # which is exactly the condition async_save exists to lift.
    #
    # __torch_dispatch__ cannot catch it (a same-device `.to()` never dispatches
    # an operator), so the refusal lives in __torch_function__ and this test
    # holds it there.
    _single_rank_group(tmp_path / "async-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        # The baseline: what the supported synchronous save restores to.
        sync_dir = str(tmp_path / "sync-ckpt")
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(sync_dir),
            planner=GefenSavePlanner(),
        )

        checkpoint_dir = str(tmp_path / "async-ckpt")
        refusal = None
        try:
            _await_async_save(
                torch.distributed.checkpoint.async_save(
                    {"optimizer": GefenDCPState(optimizer)},
                    storage_writer=FileSystemWriter(checkpoint_dir),
                    planner=GefenSavePlanner(),
                )
            )
        except RuntimeError as error:
            # Only the stand-in's own refusal is an expected outcome; anything
            # else propagates as the failure it is.
            refusal = error

        # Getting through staging is the failure, not an alternative outcome:
        # nothing would have been copied, and the writer would race step().
        assert refusal is not None, (
            "async_save accepted the default writer; the stand-ins passed "
            "through staging by reference and the write would read live state"
        )
        message = str(refusal)
        # The message must name the writer that stages these correctly...
        assert "GefenFileSystemWriter" in message, message
        # ...and must NOT tell them to pass a planner. This call already passes
        # one, so advice to pass it is advice to redo what just failed.
        assert "GefenSavePlanner" not in message, message
        # Refusing is only worth anything if it refused *before* writing.
        assert not _has_checkpoint_bytes(checkpoint_dir), os.listdir(
            checkpoint_dir
        )
        # The supported path must still land the same checkpoint the sync save
        # would, so the refusal above is a routing error and not a dead end.
        from gefen import GefenFileSystemWriter

        good_dir = str(tmp_path / "async-ok-ckpt")
        _await_async_save(
            torch.distributed.checkpoint.async_save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=GefenFileSystemWriter(good_dir),
                planner=GefenSavePlanner(),
            )
        )
        expected = _restored_signature(mesh, sync_dir)
        actual = _restored_signature(mesh, good_dir)
        assert actual[:2] == expected[:2], (actual[:2], expected[:2])
        for left, right in zip(expected[2:], actual[2:]):
            assert torch.equal(left, right), (left - right).abs().max()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_save_rejects_codebook_that_does_not_tile_the_shard(tmp_path):
    # The block-count checks are not redundant with each other: shrinking
    # m_magnitude and m_codebook together by one block keeps the ratio
    # m_codebook == m_magnitude * period intact and leaves vmean alone, so both
    # of the other invariants still hold -- but the blocks no longer tile the
    # local shard, which is what _dense_momentum's final reshape requires. Since
    # the dense expansion is deferred to write time, missing this check does not
    # merely lose an error message: it turns a clean up-front rejection into a
    # failure from inside the write.
    _single_rank_group(tmp_path / "tile-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        state = optimizer.state[param]
        period = int(state["automatic_period"])
        blocks = state["m_magnitude"].reshape(-1).numel()
        assert blocks > 1 and period > 1, (blocks, period)
        state["m_magnitude"] = state["m_magnitude"].reshape(-1)[: blocks - 1].clone()
        state["m_codebook"] = (
            state["m_codebook"].reshape(-1)[: (blocks - 1) * period].clone()
        )

        with pytest.raises(ValueError, match="momentum block geometry is invalid"):
            GefenDCPState(optimizer).state_dict()

        # ...and the save that would have consumed it stops before writing a byte.
        checkpoint_dir = str(tmp_path / "tile-ckpt")
        with pytest.raises(ValueError, match="momentum block geometry is invalid"):
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=GefenSavePlanner(),
            )
        assert not _has_checkpoint_bytes(checkpoint_dir)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_one_name_shared_by_two_parameters(tmp_path):
    # A name has to identify a parameter by itself. The saved identity is
    # (name, group, shape) and the group is a positional index, so two
    # same-shaped parameters sharing a name produce an identity list that does
    # not change when their groups are rebuilt in the opposite order: validation
    # passes and each slot's state lands on the other parameter. The index cannot
    # break the tie precisely because a reorder is what changes it.
    #
    # Both names here are caller-provided, so the synthesized-provenance guard
    # does not fire -- this is a distinct hole from the collision-suffix one, and
    # `param_0` being a legitimate caller name is what makes it reachable.
    _single_rank_group(tmp_path / "dupe-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        first = _shaped_dparam(_SHAPE, mesh)
        second = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [{"params": [("weight", first)]}, {"params": [("weight", second)]}],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        for step in range(_SAVE_STEPS):
            for param in (first, second):
                param.grad = distribute_tensor(
                    _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
                )
            optimizer.step()
        for param in (first, second):
            param.grad = None

        with pytest.raises(RuntimeError, match="unique across every group"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_accepts_distinct_names_in_separate_groups(tmp_path):
    # The guard above must not cost the ordinary case: distinct caller names in
    # separate groups are exactly how a real model arrives, and rejecting them
    # would lock legitimate optimizers out of resharding entirely.
    _single_rank_group(tmp_path / "distinct-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        first = _shaped_dparam(_SHAPE, mesh)
        second = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [
                {"params": [("encoder.weight", first)]},
                {"params": [("decoder.weight", second)]},
            ],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        for step in range(_SAVE_STEPS):
            for param in (first, second):
                param.grad = distribute_tensor(
                    _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
                )
            optimizer.step()
        for param in (first, second):
            param.grad = None

        names = [slot["name"] for slot in GefenDCPState(optimizer)._identities()]
        assert names == ["encoder.weight", "decoder.weight"], names
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="a device mismatch needs a second device to straddle",
)
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_save_rejects_state_straddling_devices(tmp_path):
    # Counts and dtypes say nothing about WHERE the block state lives, and the
    # expansion is deferred to write time: _dense_momentum scales the gathered
    # values by m_magnitude in place, so a slot whose tensors straddle devices
    # raises from inside the write, after part of the checkpoint is on disk.
    # The check cannot cost a valid save -- that same multiply would fail on any
    # mismatch it rejects -- so it only moves the failure ahead of the first byte.
    _single_rank_group(tmp_path / "device-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        # Strand one tensor on the GPU while the shard and its peers stay on CPU.
        state = optimizer.state[param]
        state["m_magnitude"] = state["m_magnitude"].cuda()

        with pytest.raises(ValueError, match="state device is inconsistent"):
            GefenDCPState(optimizer).state_dict()

        checkpoint_dir = str(tmp_path / "device-ckpt")
        with pytest.raises(ValueError, match="state device is inconsistent"):
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=GefenSavePlanner(),
            )
        assert not _has_checkpoint_bytes(checkpoint_dir)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
@pytest.mark.parametrize(
    "malformation, message",
    [
        ("short_codebook", "momentum codebook geometry is invalid"),
        ("codebook_2d", "momentum codebook geometry is invalid"),
        ("index_dtype", "momentum index dtype is invalid"),
    ],
)
def test_dcp_save_rejects_indices_the_codebook_cannot_address(
    tmp_path, malformation, message
):
    # The block-count checks say the arrays TILE the shard; they say nothing about
    # whether the indices can ADDRESS the codebook. _dense_momentum gathers with
    # index_select, so a codebook too short for an index the shard holds raises
    # IndexError -- and because the expansion is deferred to write time, that
    # arrives from inside the write, leaving a partial .distcp and no .metadata,
    # rather than as an up-front rejection.
    #
    # The checks this exercises are O(1) metadata, deliberately not a max() over
    # the indices: they are uint8, so a 1-D codebook of at least 256 rows covers
    # every index that can exist, whatever the shard holds.
    _single_rank_group(tmp_path / "address-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        state = optimizer.state[param]
        if malformation == "short_codebook":
            # Truncated below the largest index the shard actually holds.
            highest = int(state["m_codebook"].max())
            assert highest > 1, highest
            optimizer._gefen_codebook = optimizer._gefen_codebook[:highest].clone()
        elif malformation == "codebook_2d":
            # Still 256 values, so a numel check alone would pass it, but
            # index_select(0) now addresses 128 rows.
            optimizer._gefen_codebook = optimizer._gefen_codebook.reshape(128, 2)
        else:
            # Indices outside uint8's range are no longer bounded by construction.
            state["m_codebook"] = state["m_codebook"].long() * 1000

        with pytest.raises(ValueError, match=message):
            GefenDCPState(optimizer).state_dict()

        # ...and the save that would have consumed it stops before writing a byte.
        checkpoint_dir = str(tmp_path / "address-ckpt")
        with pytest.raises(ValueError, match=message):
            torch.distributed.checkpoint.save(
                {"optimizer": GefenDCPState(optimizer)},
                storage_writer=FileSystemWriter(checkpoint_dir),
                planner=GefenSavePlanner(),
            )
        assert not _has_checkpoint_bytes(checkpoint_dir)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_lazy_shard_repr_is_informative_and_never_dispatches(tmp_path):
    # torch.Tensor.__repr__ formats by reading elements, which the stand-in
    # refuses -- so before the __repr__ override, printing the state dict (or
    # logging it from an exception handler) raised a SECONDARY error that buried
    # whatever was actually being debugged. A repr must never raise.
    _single_rank_group(tmp_path / "repr-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        state_dict = GefenDCPState(optimizer).state_dict()
        standin = state_dict["slot_00000000.momentum"]

        text = repr(standin)
        assert "_LazyDenseShard" in text, text
        assert "unmaterialized" in text, text
        # The global shape, not the local shard's: that is what the stand-in
        # reports to DCP, so a repr showing anything else would mislead.
        assert str(tuple(param.shape)) in text, text
        assert "torch.float32" in text, text
        assert str(_local(param).device) in text, text
        # str/format go through the same override, and a dict of them -- the way
        # this is actually printed -- must format too.
        assert str(standin) == text
        assert "{}".format(standin) == text
        assert "_LazyDenseShard" in repr(state_dict)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_lazy_shard_pickle_names_the_real_constraint(tmp_path):
    # torch.save of a GefenDCPState state_dict used to fail inside pickle with
    # "Can't get local object '_lazy_momentum.<locals>.materialize'", which names
    # an implementation detail instead of the constraint. The stand-in references
    # live optimizer state rather than carrying data, so it is not serializable
    # outside DCP at all; the error has to say so and point somewhere useful.
    _single_rank_group(tmp_path / "pickle-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        state_dict = GefenDCPState(optimizer).state_dict()
        with pytest.raises(TypeError, match="cannot be pickled"):
            torch.save(state_dict, str(tmp_path / "standin.pt"))

        # The optimizer's own state_dict stays the serializable path, and saying
        # so is the point of the message.
        torch.save(optimizer.state_dict(), str(tmp_path / "native.pt"))
    finally:
        dist.destroy_process_group()


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
def test_dcp_rejects_overflowing_tensor_lr_before_commit(tmp_path):
    _single_rank_group(tmp_path / "overflow-lr-init")
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
        saved["param_group_hypers"][0]["lr"] = 128.0

        # 128.0 is integral, so the truncation check passes it -- but it overflows
        # int8. fill_ raises on that overflow, and it runs only AFTER the state was
        # swapped in, which is the half-applied restore staging exists to prevent.
        optimizer.param_groups[0]["lr"] = torch.tensor(1, dtype=torch.int8)
        state_before = _live_state_signature(optimizer, param)
        hypers_before = _group_hyper_signature(optimizer)

        with pytest.raises(ValueError, match="representable range"):
            GefenDCPState(optimizer).load_state_dict(saved)

        assert _live_state_signature(optimizer, param) == state_before
        assert _group_hyper_signature(optimizer) == hypers_before
        assert int(optimizer.param_groups[0]["lr"]) == 1
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_rejects_out_of_range_bool_tensor_lr_before_commit(tmp_path):
    _single_rank_group(tmp_path / "bool-lr-init")
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
        saved["param_group_hypers"][0]["lr"] = 128.0

        # bool is the one integral destination whose fill_ never raises: it would
        # coerce 128.0 to True and commit lr=1.0, a silently wrong value rather
        # than a loud one. The range check is what catches it.
        optimizer.param_groups[0]["lr"] = torch.tensor(True)
        state_before = _live_state_signature(optimizer, param)
        hypers_before = _group_hyper_signature(optimizer)

        with pytest.raises(ValueError, match="representable range"):
            GefenDCPState(optimizer).load_state_dict(saved)

        assert _live_state_signature(optimizer, param) == state_before
        assert _group_hyper_signature(optimizer) == hypers_before
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
def test_dcp_rejects_collision_suffixed_group_name(tmp_path):
    _single_rank_group(tmp_path / "collide-group-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        first = _shaped_dparam(shape, mesh)
        second = _shaped_dparam(shape, mesh)
        # Two unnamed single-parameter groups given the SAME group name. The
        # first keeps it; the second is disambiguated to "weight_1" -- and which
        # group gets the suffix is decided by registration order alone. That
        # makes the suffixed name positional despite descending from a caller's
        # name, so it must be refused: it would otherwise validate cleanly on
        # spelling and cross-assign the two parameters' momentum on a reordered
        # load.
        optimizer = Gefen(
            [
                {"params": [first], "name": "weight"},
                {"params": [second], "name": "weight"},
            ],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        assert [group["param_names"] for group in optimizer.param_groups] == [
            ["weight"],
            ["weight_1"],
        ]
        assert not optimizer._param_name_is_synthesized(first)
        assert optimizer._param_name_is_synthesized(second)
        with pytest.raises(RuntimeError, match="synthesized"):
            GefenDCPState(optimizer)
    finally:
        dist.destroy_process_group()


def _named_pair(names, make_param):
    first = make_param()
    second = make_param()
    optimizer = Gefen(
        [
            {"params": [first], "name": names[0]},
            {"params": [second], "name": names[1]},
        ],
        lr=1e-3,
        fused=False,
        factored_v_2d=False,
    )
    return first, second, optimizer


def _rename_source_checkpoint(shape):
    """A checkpoint whose two colliding groups spell out "weight"/"weight_1".

    Saved from PLAIN (non-DTensor) parameters on purpose: that is the path a
    rename can actually travel. A checkpoint saved from DTensor parameters
    carries a rank-local topology marker, and load rejects any rename against it
    ("topology differs") long before provenance is consulted. Without that marker
    -- a single-device run whose checkpoint is being resumed into a sharded one,
    which is exactly when a caller then reaches for DCP resharding -- the rename
    goes through and provenance is all that stands between the checkpoint and a
    cross-assigned reshard.
    """
    _, _, source = _named_pair(
        ("weight", "weight"), lambda: nn.Parameter(torch.randn(*shape))
    )
    assert [group["param_names"] for group in source.param_groups] == [
        ["weight"],
        ["weight_1"],
    ]
    return source.state_dict()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_native_load_rename_preserves_collision_provenance(tmp_path):
    # A native load that CHANGES a live parameter's name drops the flag recorded
    # at registration -- that flag describes the OLD name. Provenance therefore
    # travels with the name, in the checkpoint.
    #
    # Without it, this is silent momentum cross-assignment: a checkpoint from two
    # unnamed single-parameter groups both called "weight" holds the positional
    # collision result "weight"/"weight_1"; loaded into an optimizer named
    # alpha/beta, neither new name matches the group_N_param_N / param_N spelling,
    # so both are taken as caller-provided. GefenDCPState then accepts "weight_1"
    # as a stable identity, and a later DCP load with the colliding groups
    # reconstructed in the other order validates cleanly while handing each
    # parameter the other's momentum. Widening the spelling regex cannot fix this:
    # no regex separates a positional "weight_1" from a legitimate "layer_1".
    _single_rank_group(tmp_path / "rename-provenance-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        checkpoint = _rename_source_checkpoint(shape)

        # The target's own names are legitimate and caller-provided...
        alpha, beta, target = _named_pair(
            ("alpha", "beta"), lambda: _shaped_dparam(shape, mesh)
        )
        assert not target._param_name_is_synthesized(alpha)
        assert not target._param_name_is_synthesized(beta)

        # ...until the checkpoint renames them to the colliding pair.
        target.load_state_dict(checkpoint)
        assert [group["param_names"] for group in target.param_groups] == [
            ["weight"],
            ["weight_1"],
        ]
        # Provenance is restored exactly, not guessed: "weight" was the caller's
        # own group name and stays resharding-safe; only the order-dependent
        # "weight_1" is refused.
        assert not target._param_name_is_synthesized(alpha)
        assert target._param_name_is_synthesized(beta)
        with pytest.raises(RuntimeError, match="synthesized"):
            GefenDCPState(target)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_native_load_rename_without_provenance_fails_closed(tmp_path):
    # A checkpoint written before provenance was serialized carries names but no
    # flags. Their provenance is then genuinely unknowable, and the old spelling
    # fallback is exactly the hole above -- so fail closed instead: a refused
    # reshard is recoverable (re-save the checkpoint), cross-assigned momentum is
    # not. The cost is narrow, and bounded by the companion test below: only a
    # legacy checkpoint that also CHANGES a name is affected.
    _single_rank_group(tmp_path / "legacy-provenance-init")
    try:
        from gefen.gefen import _SYNTHESIZED_PARAM_NAMES_KEY

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        checkpoint = _rename_source_checkpoint(shape)
        for group in checkpoint["param_groups"]:
            group.pop(_SYNTHESIZED_PARAM_NAMES_KEY, None)

        alpha, beta, target = _named_pair(
            ("alpha", "beta"), lambda: _shaped_dparam(shape, mesh)
        )
        target.load_state_dict(checkpoint)
        # Both renamed names are unknowable, so both are refused -- including
        # "weight", which really was caller-provided. That over-refusal is the
        # deliberate price of a legacy checkpoint carrying no provenance.
        assert target._param_name_is_synthesized(alpha)
        assert target._param_name_is_synthesized(beta)
        with pytest.raises(RuntimeError, match="synthesized"):
            GefenDCPState(target)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_legacy_load_without_rename_keeps_registration_provenance(tmp_path):
    # The bound on the fail-closed rule above: it triggers on a CHANGED name, not
    # on a missing provenance key. A legacy checkpoint loaded into an optimizer
    # with the same names leaves every name exactly where registration put it, so
    # the registration-time flags remain authoritative and resharding still works.
    _single_rank_group(tmp_path / "legacy-no-rename-init")
    try:
        from gefen.gefen import _SYNTHESIZED_PARAM_NAMES_KEY

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        _, _, source = _named_pair(
            ("encoder", "decoder"), lambda: nn.Parameter(torch.randn(*shape))
        )
        checkpoint = source.state_dict()
        for group in checkpoint["param_groups"]:
            group.pop(_SYNTHESIZED_PARAM_NAMES_KEY, None)

        target_first, target_second, target = _named_pair(
            ("encoder", "decoder"), lambda: _shaped_dparam(shape, mesh)
        )
        target.load_state_dict(checkpoint)
        assert not target._param_name_is_synthesized(target_first)
        assert not target._param_name_is_synthesized(target_second)
        assert [slot.name for slot in GefenDCPState(target)._slots] == [
            "encoder",
            "decoder",
        ]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_accepts_distinct_group_names(tmp_path):
    _single_rank_group(tmp_path / "distinct-group-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        shape = (128, 64)
        first = _shaped_dparam(shape, mesh)
        second = _shaped_dparam(shape, mesh)
        # The companion to the collision case: a caller's group name that needed
        # no suffix is still caller-provided and still resharding-safe. Rejecting
        # these would lock every named-group optimizer out of GefenDCPState.
        optimizer = Gefen(
            [
                {"params": [first], "name": "encoder"},
                {"params": [second], "name": "decoder"},
            ],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        assert not optimizer._param_name_is_synthesized(first)
        assert not optimizer._param_name_is_synthesized(second)
        wrapper = GefenDCPState(optimizer)
        assert [slot.name for slot in wrapper._slots] == ["encoder", "decoder"]

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
    # Pins the two re-blocking regimes the resume guarantee rests on (direct unit
    # test, no distributed needed), measured against the only baseline that means
    # anything: what a NATIVE run at the TARGET blocking would hold.
    #
    # Scoring against the *source* values instead inverts the conclusion -- it
    # credits "gave back the number we started with" as fidelity, which is
    # precisely what the lossy direction does.
    #
    # Gefen never stores a per-element second moment: it keeps one vmean per
    # block, and that vmean is an EMA of the block's mean grad**2. Save can only
    # write each source block's vmean repeated across its elements. Both the EMA
    # and the block mean are linear, so:
    #
    #   coarsening onto source boundaries (target block = a union of whole source
    #     blocks): averaging equal-size source block means IS the mean over their
    #     union -- exactly what a native coarse run holds, to fp32 round-off.
    #   refining (target block strictly inside a source block): the one source
    #     vmean can only be spread flat over the sub-blocks, while a native fine
    #     run holds a distinct value per sub-block. That detail was never stored,
    #     so this -- not coarsening -- is the direction that loses history.
    import gefen.dcp as dcp_mod

    source_period = 64
    blocks = 16
    numel = source_period * blocks
    torch.manual_seed(0)

    # Per-element grad**2 carrying BOTH block-to-block spread (6 orders of
    # magnitude, as a real layer has) and genuine within-block variation. The
    # latter is what a refine must reconstruct and cannot; a fixture that is
    # constant within each source block would hide the effect entirely.
    scale = torch.logspace(-6, 0, blocks).reshape(-1, 1)
    grad_sq = (scale * torch.rand(blocks, source_period).add(0.5)).reshape(-1)

    def native_vmean(period):
        # What a run natively blocked at `period` holds: the mean grad**2 per
        # target block. The EMA drops out -- it is linear and identical on both
        # sides -- leaving the block mean as the thing re-blocking must match.
        return grad_sq.reshape(-1, period).mean(dim=1)

    # What save can write: each source block's vmean, repeated per element.
    dense_second = (
        native_vmean(source_period)
        .reshape(-1, 1)
        .expand(-1, source_period)
        .reshape(-1)
        .clone()
    )
    momentum = torch.linspace(-1.0, 1.0, numel)
    codebook = dcp_mod._learn_codebook(
        [("w", momentum, source_period)], torch.device("cpu")
    )

    def relative_to_native(period):
        _, _, new_vmean = dcp_mod._reblock(codebook, momentum, dense_second, period)
        # _reblock returns vmean as (nblocks, 1); flatten it or the subtraction
        # below broadcasts into an (nblocks, nblocks) outer difference.
        new_vmean = new_vmean.reshape(-1)
        native = native_vmean(period)
        return ((new_vmean - native).abs() / native.abs()).max().item()

    # Aligned, and coarsening onto source boundaries: reproduces a native run at
    # the target blocking to fp32 round-off. Coarsening is NOT the lossy case.
    assert relative_to_native(source_period) < 1e-5, relative_to_native(source_period)
    assert relative_to_native(source_period * 4) < 1e-5, relative_to_native(
        source_period * 4
    )

    # Refining: materially wrong versus a native fine run, in a way no
    # re-quantization tolerance covers -- the sub-block detail the target
    # blocking wants was never in the checkpoint to begin with.
    assert relative_to_native(source_period // 4) > 0.1, relative_to_native(
        source_period // 4
    )


def _same_signature(left, right):
    """Compare two :func:`_restored_signature` tuples."""
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and torch.equal(left[2], right[2])
        and torch.equal(left[3], right[3])
        and torch.equal(left[4], right[4])
    )


def _async_saved(state, checkpoint_dir, storage_writer, planner=None):
    """Run an async_save to completion, whatever shape the future takes."""
    _await_async_save(
        torch.distributed.checkpoint.async_save(
            state,
            storage_writer=storage_writer,
            planner=planner,
        )
    )
    return checkpoint_dir


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_async_save_with_gefen_writer_restores_like_the_sync_save(tmp_path):
    # The checkpoint an async save lands must be worth the same as the one the
    # synchronous save lands. Not the same *bytes*: staging sets the writer's
    # per_thread_copy_ahead to 0, which sorts the write items differently, so the
    # file layout legitimately differs. What must match is what comes back out.
    from gefen import GefenFileSystemWriter

    _single_rank_group(tmp_path / "async-ok-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        sync_dir = str(tmp_path / "sync-ckpt")
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(sync_dir),
            planner=GefenSavePlanner(),
        )

        async_dir = str(tmp_path / "async-ckpt")
        _async_saved(
            {"optimizer": GefenDCPState(optimizer)},
            async_dir,
            GefenFileSystemWriter(async_dir),
            GefenSavePlanner(),
        )
        assert _has_checkpoint_bytes(async_dir), async_dir

        expected = _restored_signature(mesh, sync_dir)
        actual = _restored_signature(mesh, async_dir)
        assert _same_signature(actual, expected), (actual[:2], expected[:2])
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DCP resharding coverage requires Gloo",
)
def test_dcp_async_save_snapshots_the_state_a_later_step_mutates(tmp_path):
    # The property async_save exists to provide: it returns once the state is
    # staged, and training may resume immediately without changing what lands on
    # disk. That is only true if staging *copies* the dense form; a save that
    # expanded the stand-ins in the writer thread would read whatever the
    # optimizer holds by then and write a checkpoint that is neither the state at
    # save time nor the state after the step, but a tear across both.
    #
    # Gating the writer is what makes this deterministic rather than a race the
    # test usually wins: the background write is parked until the step has
    # already landed, so an unstaged save could not accidentally pass.
    import threading

    from gefen import GefenFileSystemWriter

    class _GatedWriter(GefenFileSystemWriter):
        def __init__(self, path):
            super().__init__(path)
            self.entered = threading.Event()
            self.release = threading.Event()

        def write_data(self, plan, planner):
            self.entered.set()
            if not self.release.wait(timeout=120):
                raise RuntimeError("gated write was never released")
            return super().write_data(plan, planner)

    _single_rank_group(tmp_path / "async-snap-init")
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))
        param = _shaped_dparam(_SHAPE, mesh)
        optimizer = Gefen(
            [("weight", param)], lr=1e-3, fused=False, factored_v_2d=False
        )
        for step in range(_SAVE_STEPS):
            param.grad = distribute_tensor(
                _shaped_grad(_SHAPE, step), mesh, [Shard(0)]
            )
            optimizer.step()
        param.grad = None

        # What the optimizer holds at the moment of the async_save call.
        before_dir = str(tmp_path / "before-ckpt")
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(before_dir),
            planner=GefenSavePlanner(),
        )

        async_dir = str(tmp_path / "async-ckpt")
        writer = _GatedWriter(async_dir)
        response = torch.distributed.checkpoint.async_save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=writer,
            planner=GefenSavePlanner(),
        )
        # async_save has returned, so staging is done and the write is parked.
        assert writer.entered.wait(timeout=120)

        # Resume training while the checkpoint is still unwritten.
        param.grad = distribute_tensor(
            _shaped_grad(_SHAPE, _SAVE_STEPS + 7), mesh, [Shard(0)]
        )
        optimizer.step()
        param.grad = None
        after_dir = str(tmp_path / "after-ckpt")
        torch.distributed.checkpoint.save(
            {"optimizer": GefenDCPState(optimizer)},
            storage_writer=FileSystemWriter(after_dir),
            planner=GefenSavePlanner(),
        )

        writer.release.set()
        _await_async_save(response)

        at_save = _restored_signature(mesh, before_dir)
        after_step = _restored_signature(mesh, after_dir)
        # The step has to have moved the state, or the assertions below are vacuous.
        assert not _same_signature(at_save, after_step)

        landed = _restored_signature(mesh, async_dir)
        assert _same_signature(landed, at_save), (landed[:2], at_save[:2])
        assert not _same_signature(landed, after_step)
    finally:
        dist.destroy_process_group()
