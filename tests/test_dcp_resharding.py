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
