"""Focused acceptance coverage for the plain Gefen DTensor data plane."""

from __future__ import annotations

import copy
from datetime import timedelta
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import time
import traceback

import pytest
import torch
import torch.distributed as dist


_CANONICAL_STATE_KEYS = {
    "name",
    "automatic_period",
    "step",
    "m_codebook",
    "m_magnitude",
    "vmean",
    "vmean_step",
    "v_row",
    "v_col",
    "factored_step",
}


def _local(value):
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "wait"):
        value = value.wait()
    return value


def _raw_bytes(value):
    value = _local(value).detach().contiguous()
    return value.reshape(-1).view(torch.uint8).clone()


def _clone(value):
    if torch.is_tensor(value):
        return _local(value).detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _assert_exact(actual, expected, path="value"):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), path
        actual = _local(actual)
        expected = _local(expected)
        assert actual.dtype == expected.dtype, path
        assert actual.layout == expected.layout, path
        assert tuple(actual.shape) == tuple(expected.shape), path
        assert torch.equal(_raw_bytes(actual), _raw_bytes(expected)), path
        return
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (path, tuple(actual), tuple(expected))
        for key in expected:
            _assert_exact(actual[key], expected[key], "{}.{}".format(path, key))
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_exact(actual_item, expected_item, "{}[{}]".format(path, index))
    else:
        assert actual == expected, path


def _region(shape, coordinate, parts, dimension=0):
    from gefen import LogicalRegion

    global_length = shape[dimension]
    chunk = (global_length + parts - 1) // parts
    offset = min(coordinate * chunk, global_length)
    length = max(0, min(global_length, offset + chunk) - offset)
    offsets = [0] * len(shape)
    lengths = list(shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _manifest(fqn, shape, world, *, axis_lengths=None):
    from gefen import (
        LogicalRegion,
        ParameterIdentity,
        ParameterLayout,
        PlacementKind,
        ProcessGroupIdentity,
        ShardIdentity,
        ShardPlacement,
        ShardingManifest,
    )

    members = tuple("rank:{}".format(rank) for rank in range(world))
    identity = ParameterIdentity(fqn, tuple(shape))
    process_group = ProcessGroupIdentity("data_parallel", members)
    regions = []
    if axis_lengths is None:
        regions = [_region(shape, rank, world) for rank in range(world)]
    else:
        cursor = 0
        for length in axis_lengths:
            regions.append(LogicalRegion((cursor, 0), (length, shape[1])))
            cursor += length
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
            regions[rank],
            placements=(
                ShardPlacement(
                    "dp",
                    PlacementKind.DIMENSION_SHARD,
                    rank,
                    world,
                    parameter_dimension=0,
                ),
            ),
            process_group=process_group,
            local_member=members[rank],
        )
        for rank in range(world)
    )
    return ShardingManifest(shards), shards


def _optimizer(parameter, *, factored_v_2d=False):
    from gefen import Gefen

    return Gefen(
        [("legacy.weight", parameter)],
        lr=3e-3,
        betas=(0.8, 0.97),
        eps=2e-8,
        weight_decay=0.02,
        fused=False,
        deterministic=True,
        force_2d_period_one=True,
        factored_v_2d=factored_v_2d,
    )


def _make_finalized(mesh, rank, world, full, *, fqn="Model.Weight", factored_v_2d=False):
    from gefen import ParameterRebinding
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor

    parameter = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
    optimizer = _optimizer(parameter, factored_v_2d=factored_v_2d)
    manifest, shards = _manifest(fqn, full.shape, world)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shards[rank]),),
        manifest=manifest,
    )
    return optimizer, parameter, manifest, shards[rank]


def _assign_gradient(parameter, mesh, gradient):
    from torch.distributed.tensor import Shard, distribute_tensor

    parameter.grad = distribute_tensor(gradient.clone(), mesh, [Shard(0)])


def _persistent_snapshot(optimizer, parameter):
    return {
        "global_step": optimizer._gefen_global_step,
        "codebook": _clone(optimizer._gefen_codebook),
        "parameter": _clone(parameter),
        "state": _clone(
            {
                key: value
                for key, value in optimizer.state[parameter].items()
                if key in _CANONICAL_STATE_KEYS
            }
        ),
    }


def _observable_snapshot(optimizer, parameter):
    parameter_state = optimizer.state.get(parameter)
    return {
        "containers": (
            id(optimizer.__dict__),
            id(optimizer.state),
            id(optimizer.param_groups),
            tuple(id(group) for group in optimizer.param_groups),
            tuple(id(group.get("params")) for group in optimizer.param_groups),
            id(optimizer._param_names),
            id(optimizer._gefen_shard_bindings),
            None if parameter_state is None else id(parameter_state),
        ),
        "parameter": _clone(parameter),
        "global_step": optimizer._gefen_global_step,
        "codebook": _clone(optimizer._gefen_codebook),
        "state": None if parameter_state is None else _clone(dict(parameter_state)),
        "groups": tuple(
            (
                tuple(id(item) for item in group["params"]),
                _clone({key: value for key, value in group.items() if key != "params"}),
            )
            for group in optimizer.param_groups
        ),
        "names": tuple((id(item), value) for item, value in optimizer._param_names.items()),
        "bindings": tuple((id(item), value) for item, value in optimizer._gefen_shard_bindings.items()),
        "local_bindings": tuple(
            (None if item is None else id(item), shard)
            for item, shard in optimizer._gefen_local_shard_bindings
        ),
        "manifest": optimizer._gefen_sharding_manifest,
        "finalized": optimizer._gefen_post_sharding_finalized,
    }


def _zero_tensors(value):
    changed = False
    if torch.is_tensor(value):
        changed = bool(value.numel() and torch.count_nonzero(value).item())
        value.zero_()
        return changed
    if isinstance(value, dict):
        for item in value.values():
            changed = _zero_tensors(item) or changed
        return changed
    if isinstance(value, (list, tuple)):
        for item in value:
            changed = _zero_tensors(item) or changed
        return changed
    return False


def _init_process_group(rank, world, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_file),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=30),
    )


def _finish_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _routes_worker(rank, world, init_file, checkpoint_dir, result_queue):
    del checkpoint_dir
    from torch.distributed.tensor import init_device_mesh

    try:
        _init_process_group(rank, world, init_file)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        full = torch.linspace(-0.75, 0.5, 20, dtype=torch.float32).reshape(5, 4)
        gradient = torch.linspace(0.6, -0.4, 20, dtype=torch.float32).reshape(5, 4)

        identity_optimizer, identity_parameter, manifest, shard = _make_finalized(mesh, rank, world, full)
        identity_binding = (
            identity_optimizer.param_groups[0]["params"][0] is identity_parameter
            and identity_optimizer.shard_identity(identity_parameter) == shard
            and identity_optimizer.shard_bindings() == ((identity_parameter, shard),)
            and identity_optimizer.sharding_manifest() == manifest
        )
        _assign_gradient(identity_parameter, mesh, gradient)
        identity_optimizer.step()
        identity_stepped = identity_optimizer.state[identity_parameter].get("step") == 1

        factored_optimizer, factored_parameter, _, _ = _make_finalized(
            mesh,
            rank,
            world,
            full.neg(),
            fqn="Model.FactoredWeight",
            factored_v_2d=True,
        )
        _assign_gradient(factored_parameter, mesh, gradient.cos())
        factored_optimizer.step()
        local = _local(factored_parameter)
        state = factored_optimizer.state[factored_parameter]
        factored_uses_block_state = (
            "vmean" in state
            and "vmean_step" in state
            and tuple(state["vmean"].shape) == (local.numel(), 1)
            and state["vmean_step"] == 1
            and "v_row" not in state
            and "v_col" not in state
            and "factored_step" not in state
        )
        result_queue.put(
            {
                "rank": rank,
                "identity_binding": identity_binding,
                "identity_stepped": identity_stepped,
                "factored_uses_block_state": factored_uses_block_state,
                "factored_keys": sorted(key for key in state if not key.startswith("_gefen_rank_local_")),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal": traceback.format_exc()})
    finally:
        _finish_process_group()


def _remote_manifest_worker(rank, world, init_file, checkpoint_dir, result_queue):
    del checkpoint_dir
    from gefen import ParameterRebinding
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        _init_process_group(rank, world, init_file)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        full = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        parameter = nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))
        optimizer = _optimizer(parameter)
        manifest, shards = _manifest("Model.Weight", full.shape, world, axis_lengths=(2, 1, 2))
        local_descriptor_matches = tuple(_local(parameter).shape) == shards[rank].logical_region.lengths
        before = _observable_snapshot(optimizer, parameter)
        error = None
        try:
            optimizer.post_sharding(
                (ParameterRebinding(parameter, parameter, shards[rank]),),
                manifest=manifest,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            error = "{}: {}".format(type(exc).__name__, exc)
        unchanged = True
        try:
            _assert_exact(_observable_snapshot(optimizer, parameter), before, "remote manifest rejection")
        except AssertionError:
            unchanged = False
        result_queue.put(
            {
                "rank": rank,
                "rejected": error is not None,
                "error": error,
                "unchanged": unchanged,
                "local_descriptor_matches": local_descriptor_matches,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal": traceback.format_exc()})
    finally:
        _finish_process_group()


def _checkpoint_worker(rank, world, init_file, checkpoint_dir, result_queue):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch import nn
    from torch.distributed.tensor import init_device_mesh

    try:
        _init_process_group(rank, world, init_file)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        full = torch.linspace(-1.0, 1.0, 20, dtype=torch.float32).reshape(5, 4)
        first_gradient = torch.arange(20, dtype=torch.float32).reshape(5, 4).sin()
        second_gradient = torch.arange(20, dtype=torch.float32).reshape(5, 4).cos()
        source, source_parameter, _, _ = _make_finalized(mesh, rank, world, full)
        _assign_gradient(source_parameter, mesh, first_gradient)
        source.step()
        native_state = source.state_dict()

        current_full = source_parameter.detach().full_tensor().clone()
        mismatch, mismatch_parameter, _, _ = _make_finalized(
            mesh,
            rank,
            world,
            current_full,
            fqn="Model.DifferentWeight",
        )
        before_mismatch = _observable_snapshot(mismatch, mismatch_parameter)
        mismatch_error = None
        try:
            mismatch.load_state_dict(native_state)
        except (TypeError, ValueError, RuntimeError) as exc:
            mismatch_error = "{}: {}".format(type(exc).__name__, exc)
        mismatch_unchanged = True
        try:
            _assert_exact(
                _observable_snapshot(mismatch, mismatch_parameter),
                before_mismatch,
                "stable identity mismatch",
            )
        except AssertionError:
            mismatch_unchanged = False

        source_model = nn.Module()
        source_model.register_parameter("weight", source_parameter)
        public_state = get_optimizer_state_dict(
            source_model,
            source,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        public_handoff = [public_state if rank == 0 else None]
        dist.broadcast_object_list(public_handoff, src=0)
        persisted = {"optimizer": public_handoff[0]}
        metadata = dcp.save(
            state_dict=persisted,
            storage_writer=FileSystemWriter(checkpoint_dir),
            process_group=dist.group.WORLD,
        )
        on_disk_metadata = FileSystemReader(checkpoint_dir).read_metadata()
        load_target = _clone(persisted)
        zeroing_changed_payload = _zero_tensors(load_target)
        dcp.load(
            state_dict=load_target,
            storage_reader=FileSystemReader(checkpoint_dir),
            process_group=dist.group.WORLD,
        )
        _assert_exact(load_target, persisted, "filesystem DCP round trip")

        resumed, resumed_parameter, _, _ = _make_finalized(mesh, rank, world, current_full)
        resumed_model = nn.Module()
        resumed_model.register_parameter("weight", resumed_parameter)
        set_optimizer_state_dict(
            resumed_model,
            resumed,
            load_target["optimizer"] if rank == 0 else {},
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
        restored = True
        try:
            _assert_exact(
                _persistent_snapshot(resumed, resumed_parameter),
                _persistent_snapshot(source, source_parameter),
                "DCP restored optimizer",
            )
        except AssertionError:
            restored = False

        _assign_gradient(source_parameter, mesh, second_gradient)
        _assign_gradient(resumed_parameter, mesh, second_gradient)
        source.step()
        resumed.step()
        continued = True
        try:
            _assert_exact(
                _persistent_snapshot(resumed, resumed_parameter),
                _persistent_snapshot(source, source_parameter),
                "DCP exact continuation",
            )
        except AssertionError:
            continued = False
        result_queue.put(
            {
                "rank": rank,
                "mismatch_rejected": mismatch_error is not None,
                "mismatch_error": mismatch_error,
                "mismatch_unchanged": mismatch_unchanged,
                "dcp_metadata": bool(metadata.state_dict_metadata),
                "reader_metadata": set(on_disk_metadata.state_dict_metadata) == set(metadata.state_dict_metadata),
                "zeroing_changed_payload": zeroing_changed_payload,
                "restored": restored,
                "continued": continued,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal": traceback.format_exc()})
    finally:
        _finish_process_group()


def _checkpoint_failure_worker(rank, world, init_file, checkpoint_dir, result_queue):
    del checkpoint_dir
    from torch import nn
    from torch.distributed.tensor import init_device_mesh

    try:
        _init_process_group(rank, world, init_file)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        full = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        optimizer, parameter, _, _ = _make_finalized(mesh, rank, world, full)

        def fail_pre_hook(_optimizer):
            raise RuntimeError("injected rank-zero state-dict pre-hook failure")

        handle = optimizer.register_state_dict_pre_hook(fail_pre_hook) if rank == 0 else None
        pre_hook_error = None
        try:
            optimizer.state_dict()
        except RuntimeError as exc:
            pre_hook_error = str(exc)
        if handle is not None:
            handle.remove()
        dist.barrier()

        original_parameter = optimizer.param_groups[0]["params"][0]
        if rank == 0:
            optimizer.param_groups[0]["params"][0] = nn.Parameter(torch.zeros_like(_local(parameter)))
        layout_error = None
        try:
            optimizer.state_dict()
        except RuntimeError as exc:
            layout_error = str(exc)
        if rank == 0:
            optimizer.param_groups[0]["params"][0] = original_parameter
        result_queue.put(
            {
                "rank": rank,
                "pre_hook_error": pre_hook_error,
                "layout_error": layout_error,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal": traceback.format_exc()})
    finally:
        _finish_process_group()


def _spawn(worker, world, *, timeout):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-dtensor-acceptance-")
    init_file = os.path.join(directory, "store")
    checkpoint_dir = os.path.join(directory, "checkpoint")
    processes = [
        context.Process(
            target=worker,
            args=(rank, world, init_file, checkpoint_dir, result_queue),
        )
        for rank in range(world)
    ]
    results = []
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.start()
        while len(results) < world:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                results.append(result_queue.get(timeout=remaining))
            except queue_module.Empty:
                break
        for process in processes:
            process.join(timeout=max(0.0, min(5.0, deadline - time.monotonic())))
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
        shutil.rmtree(directory, ignore_errors=True)
    assert len(results) == world, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), (results, [process.exitcode for process in processes])
    results = sorted(results, key=lambda item: item["rank"])
    assert all("fatal" not in result for result in results), results
    return results


@pytest.fixture(scope="module")
def _route_results():
    return _spawn(_routes_worker, 2, timeout=60)


@pytest.fixture(scope="module")
def _checkpoint_results():
    return _spawn(_checkpoint_worker, 2, timeout=90)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="DTensor acceptance coverage requires Gloo")
def test_identity_only_rebinding_finalizes_an_existing_dtensor_slot(_route_results):
    assert all(result["identity_binding"] and result["identity_stepped"] for result in _route_results), _route_results


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="DTensor acceptance coverage requires Gloo")
def test_factored_v_2d_routes_sharded_matrices_to_local_block_vmean(_route_results):
    assert all(result["factored_uses_block_state"] for result in _route_results), _route_results


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="DTensor acceptance coverage requires Gloo")
def test_malformed_remote_manifest_region_fails_locally_before_mutation():
    results = _spawn(_remote_manifest_worker, 3, timeout=60)
    assert results[0]["local_descriptor_matches"], results
    assert all(result["rejected"] and result["unchanged"] for result in results), results


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="DTensor acceptance coverage requires Gloo")
def test_rank_local_load_requires_exact_stable_binding(_checkpoint_results):
    assert all(result["mismatch_rejected"] and result["mismatch_unchanged"] for result in _checkpoint_results), _checkpoint_results


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="DTensor acceptance coverage requires Gloo")
def test_rank_local_state_dict_failures_raise_symmetrically_without_hanging():
    results = _spawn(_checkpoint_failure_worker, 2, timeout=45)
    assert all(result["pre_hook_error"] is not None and result["layout_error"] is not None for result in results), results
    assert "failed on this rank" in results[0]["pre_hook_error"], results
    assert "failed on another rank" in results[1]["pre_hook_error"], results
    assert "failed on this rank" in results[0]["layout_error"], results
    assert "failed on another rank" in results[1]["layout_error"], results


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="DTensor acceptance coverage requires Gloo")
def test_rank_local_checkpoint_persists_through_filesystem_dcp(_checkpoint_results):
    assert all(
        result["dcp_metadata"]
        and result["reader_metadata"]
        and result["zeroing_changed_payload"]
        and result["restored"]
        and result["continued"]
        for result in _checkpoint_results
    ), _checkpoint_results
