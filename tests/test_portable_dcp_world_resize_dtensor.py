"""Focused 2-to-4 world-size coverage for sharded portable DTensor DCP."""

from __future__ import annotations

from datetime import timedelta
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import traceback
import warnings

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

from gefen import (
    CheckpointProcessGroupBinding,
    CodebookProcessGroupBinding,
    Gefen,
    LogicalRegion,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    PlacementKind,
    PortableStateLimits,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
    load_portable_dcp,
    save_portable_dcp,
)


_FQN = "model.weight"
_SHAPE = (5, 3)


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=2 << 20,
        max_collective_tensor_bytes=8 << 20,
        max_collective_metadata_bytes=8 << 20,
        chunk_bytes=29,
        max_members=4,
        max_metadata_bytes=2 << 20,
        max_tree_nodes=20_000,
        max_tree_depth=32,
        max_container_items=20_000,
        max_string_bytes=16 << 10,
        max_integer_bytes=128,
        max_tensors=512,
        max_tensor_rank=8,
        diagnostic_bytes=1024,
    )


def _strict_worker_warnings():
    warnings.simplefilter("error")
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
    )


def _values():
    numel = _SHAPE[0] * _SHAPE[1]
    return (
        torch.linspace(-0.8, 0.7, numel, dtype=torch.float32).reshape(_SHAPE),
        torch.linspace(-1.5, 1.25, numel, dtype=torch.float32).reshape(_SHAPE),
        torch.arange(1, numel + 1, dtype=torch.float32).reshape(_SHAPE).div_(7.0),
        torch.arange(numel, dtype=torch.float32).reshape(_SHAPE).cos(),
    )


def _standard_region(shape, coordinate, dimension, world_size):
    chunk = (shape[dimension] + world_size - 1) // world_size
    offset = min(coordinate * chunk, shape[dimension])
    length = max(0, min(shape[dimension], offset + chunk) - offset)
    offsets = [0] * len(shape)
    lengths = list(shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _make_optimizer(
    rank,
    world_size,
    mesh,
    group,
    dimension,
    full_parameter,
    *,
    deterministic,
):
    identity = ParameterIdentity(_FQN, tuple(full_parameter.shape))
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
            _standard_region(
                identity.global_shape,
                coordinate,
                dimension,
                world_size,
            ),
            placements=(
                ShardPlacement(
                    "dp",
                    PlacementKind.DIMENSION_SHARD,
                    coordinate,
                    world_size,
                    parameter_dimension=dimension,
                ),
            ),
            process_group=group,
            local_member=member,
        )
        for coordinate, member in enumerate(group.ordered_members)
    )
    manifest = ShardingManifest(shards)
    parameter = nn.Parameter(
        distribute_tensor(full_parameter.clone(), mesh, [Shard(dimension)])
    )
    optimizer = Gefen(
        [("weight", parameter)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        force_2d_period_one=True,
        factored_v_2d=False,
        deterministic=deterministic,
    )
    member = group.ordered_members[rank]
    codebook_binding = CodebookProcessGroupBinding(
        group,
        member,
        dist.group.WORLD,
        torch.device("cpu"),
    )
    checkpoint_binding = CheckpointProcessGroupBinding(
        group,
        member,
        dist.group.WORLD,
        torch.device("cpu"),
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shards[rank]),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, parameter, shards[rank], checkpoint_binding


def _region_index(region):
    return tuple(
        slice(offset, offset + length)
        for offset, length in zip(region.offsets, region.lengths)
    )


def _local_tensor(parameter):
    local = parameter.to_local()
    if hasattr(local, "wait"):
        local = local.wait()
    return local


def _seed_state(optimizer, parameter, shard, momentum, second_moment):
    optimizer._gefen_global_step = 9
    optimizer._gefen_codebook = torch.linspace(
        -1.0,
        1.0,
        256,
        dtype=torch.float32,
    )
    local_momentum = momentum[_region_index(shard.logical_region)].clone()
    if local_momentum.numel() == 0:
        return
    flat = local_momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8),
        torch.full((flat.numel(),), 255, dtype=torch.uint8),
    )
    local_second = second_moment[_region_index(shard.logical_region)].clone()
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 7,
            "m_codebook": indices.reshape(-1, 1),
            "m_magnitude": flat.abs().reshape(-1, 1).clone(),
            "vmean": local_second.reshape(-1, 1),
            "vmean_step": 6,
        }
    )


def _assign_gradient(parameter, mesh, dimension, full_gradient):
    parameter.grad = distribute_tensor(
        full_gradient.clone(),
        mesh,
        [Shard(dimension)],
    )


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if type(value) is dict:
        return {key: _clone(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone(item) for item in value]
    if type(value) is tuple:
        return tuple(_clone(item) for item in value)
    return value


def _snapshot(optimizer, parameter):
    local_state = optimizer.state.get(parameter)
    return {
        "global_step": optimizer._gefen_global_step,
        "codebook": _clone(optimizer._gefen_codebook),
        "deterministic": optimizer._deterministic,
        "parameter": _local_tensor(parameter).detach().clone(),
        "state": None if local_state is None else _clone(local_state),
    }


def _assert_exact(actual, expected, path="value"):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual), path
        assert actual.dtype == expected.dtype, path
        assert tuple(actual.shape) == tuple(expected.shape), path
        assert torch.equal(
            actual.detach().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().contiguous().reshape(-1).view(torch.uint8),
        ), path
        return
    assert type(actual) is type(expected), path
    if type(expected) is dict:
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_exact(actual[key], expected[key], "{}.{}".format(path, key))
    elif type(expected) in {list, tuple}:
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_exact(
                actual_item,
                expected_item,
                "{}[{}]".format(path, index),
            )
    else:
        assert actual == expected, path


def _save_worker(rank, world_size, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        group = ProcessGroupIdentity(
            "portable_dtensor_save_2",
            tuple("save:{}".format(index) for index in range(world_size)),
        )
        parameter_value, momentum, second_moment, _gradient = _values()
        optimizer, parameter, shard, binding = _make_optimizer(
            rank,
            world_size,
            mesh,
            group,
            0,
            parameter_value,
            deterministic=True,
        )
        _seed_state(optimizer, parameter, shard, momentum, second_moment)
        save_portable_dcp(
            optimizer,
            checkpoint_process_group=binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="portable-dtensor-save-2",
            limits=_limits(),
        )

        checkpoint_metadata = FileSystemReader(checkpoint_dir).read_metadata()
        entries = checkpoint_metadata.state_dict_metadata
        field_entries = [
            entry
            for key, entry in entries.items()
            if key.startswith("optimizer.__gefen_portable_field_")
        ]
        v2_sharded = (
            "optimizer.__gefen_portable_metadata_v2__" in entries
            and "optimizer.__gefen_portable_metadata_v1__" not in entries
            and bool(field_entries)
            and all(len(entry.chunks) == world_size for entry in field_entries)
        )
        result_queue.put(
            {
                "rank": rank,
                "local_shape": tuple(_local_tensor(parameter).shape),
                "v2_sharded": v2_sharded,
                "source_chunk_counts": tuple(
                    len(entry.chunks) for entry in field_entries
                ),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _load_worker(rank, world_size, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("dp",))
        group = ProcessGroupIdentity(
            "portable_dtensor_load_4",
            tuple("load:{}".format(index) for index in range(world_size)),
        )
        parameter_value, momentum, second_moment, gradient = _values()
        target, target_parameter, target_shard, target_binding = _make_optimizer(
            rank,
            world_size,
            mesh,
            group,
            1,
            parameter_value,
            deterministic=False,
        )
        load_portable_dcp(
            target,
            checkpoint_process_group=target_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="portable-dtensor-load-4",
            limits=_limits(),
        )

        reference, reference_parameter, reference_shard, _reference_binding = (
            _make_optimizer(
                rank,
                world_size,
                mesh,
                group,
                1,
                parameter_value,
                deterministic=True,
            )
        )
        _seed_state(
            reference,
            reference_parameter,
            reference_shard,
            momentum,
            second_moment,
        )
        _assert_exact(
            _snapshot(target, target_parameter),
            _snapshot(reference, reference_parameter),
            "restored optimizer",
        )

        _assign_gradient(target_parameter, mesh, 1, gradient)
        _assign_gradient(reference_parameter, mesh, 1, gradient)
        target.step()
        reference.step()
        _assert_exact(
            _snapshot(target, target_parameter),
            _snapshot(reference, reference_parameter),
            "continued optimizer",
        )
        result_queue.put(
            {
                "rank": rank,
                "local_shape": tuple(_local_tensor(target_parameter).shape),
                "empty": target_shard.logical_region.numel == 0,
                "restored_exact": True,
                "next_step_exact": True,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_phase(worker, world_size, checkpoint_dir, *, timeout=180):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(
        prefix="gefen-portable-dtensor-world-resize-"
    )
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=worker,
            args=(
                rank,
                world_size,
                init_file,
                checkpoint_dir,
                result_queue,
            ),
        )
        for rank in range(world_size)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _process in processes:
                results.append(result_queue.get(timeout=timeout))
        except queue_module.Empty:
            pass
        for process in processes:
            process.join(timeout=10)
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
        if os.path.exists(init_file):
            os.unlink(init_file)
    assert len(results) == world_size, (
        results,
        [process.exitcode for process in processes],
    )
    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DTensor world-resize coverage requires Gloo",
)
def test_sharded_portable_dtensor_checkpoint_resumes_from_two_to_four_ranks(
    tmp_path,
):
    checkpoint_dir = str(tmp_path / "portable-dtensor-2-to-4")
    save_results = _run_phase(_save_worker, 2, checkpoint_dir)
    assert all("fatal_error" not in result for result in save_results), save_results
    assert [result["local_shape"] for result in save_results] == [(3, 3), (2, 3)]
    assert all(result["v2_sharded"] for result in save_results), save_results
    assert all(
        result["source_chunk_counts"]
        and set(result["source_chunk_counts"]) == {2}
        for result in save_results
    ), save_results

    load_results = _run_phase(_load_worker, 4, checkpoint_dir)
    assert all("fatal_error" not in result for result in load_results), load_results
    assert [result["local_shape"] for result in load_results] == [
        (5, 1),
        (5, 1),
        (5, 1),
        (5, 0),
    ]
    assert [result["empty"] for result in load_results] == [False, False, False, True]
    assert all(
        result["restored_exact"] and result["next_step_exact"]
        for result in load_results
    ), load_results
