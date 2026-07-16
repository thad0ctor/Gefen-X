"""Two-rank same-topology Hybrid rebinding with a DTensor AdamW backup."""

from datetime import timedelta
import copy
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist


_WORLD = 2
_MEMBERS = ("rank:0", "rank:1")
_DTYPES = (
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
    ("float32", torch.float32),
    ("float64", torch.float64),
)


def _region(shape, coordinate, dimension):
    from gefen import LogicalRegion

    global_length = shape[dimension]
    chunk = (global_length + _WORLD - 1) // _WORLD
    offset = min(coordinate * chunk, global_length)
    length = max(0, min(global_length, offset + chunk) - offset)
    offsets = [0] * len(shape)
    lengths = list(shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _manifest():
    from gefen import (
        ParameterIdentity,
        ParameterLayout,
        PlacementKind,
        ProcessGroupIdentity,
        ShardIdentity,
        ShardPlacement,
        ShardingManifest,
    )

    group = ProcessGroupIdentity("data_parallel", _MEMBERS)
    shards = []
    by_fqn = {}
    for fqn, shape in (("Model.Bias", (4,)),):
        identity = ParameterIdentity(fqn, shape)
        parameter_shards = []
        for coordinate, member in enumerate(_MEMBERS):
            shard = ShardIdentity(
                identity,
                ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
                _region(shape, coordinate, 0),
                placements=(
                    ShardPlacement(
                        "dp",
                        PlacementKind.DIMENSION_SHARD,
                        coordinate,
                        _WORLD,
                        parameter_dimension=0,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
            shards.append(shard)
            parameter_shards.append(shard)
        by_fqn[fqn] = tuple(parameter_shards)
    return ShardingManifest(tuple(shards)), by_fqn


def _optimizer(old_bias):
    from gefen import GefenMuonHybrid

    return GefenMuonHybrid(
        [],
        [("layer.bias", old_bias)],
        lr=1e-3,
        fused=False,
        backup_optimizer="adamw",
    )


def _local(value):
    value = value.to_local()
    return value.wait() if hasattr(value, "wait") else value


def _state_exact(left, left_parameter, right, right_parameter):
    left_state = left.backup.state[left_parameter]
    right_state = right.backup.state[right_parameter]
    if set(left_state) != set(right_state):
        return False
    for key in left_state:
        left_value = left_state[key]
        right_value = right_state[key]
        if torch.is_tensor(left_value):
            if hasattr(left_value, "to_local"):
                left_value = _local(left_value)
                right_value = _local(right_value)
            if left_value.dtype != right_value.dtype or not torch.equal(left_value, right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def _exercise(rank, mesh, dtype):
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor

    from gefen import ParameterRebinding

    full_bias = torch.arange(4, dtype=dtype)
    manifest, shards = _manifest()

    old_bias = nn.Parameter(full_bias.clone())
    bias = nn.Parameter(distribute_tensor(full_bias.clone(), mesh, [Shard(0)]))
    optimizer = _optimizer(old_bias)
    optimizer.post_sharding(
        (
            ParameterRebinding(old_bias, bias, shards["Model.Bias"][rank]),
        ),
        manifest=manifest,
    )

    first_gradient = torch.tensor([0.5, -0.25, 0.75, -1.0], dtype=dtype)
    bias.grad = distribute_tensor(first_gradient, mesh, [Shard(0)])
    optimizer.step()
    optimizer.zero_grad()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    current_bias = bias.detach().full_tensor().clone()

    resumed_old_bias = nn.Parameter(current_bias.clone())
    resumed_bias = nn.Parameter(distribute_tensor(current_bias, mesh, [Shard(0)]))
    resumed = _optimizer(resumed_old_bias)
    resumed.post_sharding(
        (
            ParameterRebinding(resumed_old_bias, resumed_bias, shards["Model.Bias"][rank]),
        ),
        manifest=manifest,
    )
    resumed.load_state_dict(checkpoint)
    assert _state_exact(optimizer, bias, resumed, resumed_bias)

    second_gradient = torch.tensor([-0.5, 0.125, 0.25, 0.875], dtype=dtype)
    bias.grad = distribute_tensor(second_gradient, mesh, [Shard(0)])
    resumed_bias.grad = distribute_tensor(second_gradient, mesh, [Shard(0)])
    optimizer.step()
    resumed.step()
    assert torch.equal(_local(bias.detach()), _local(resumed_bias.detach()))
    assert _state_exact(optimizer, bias, resumed, resumed_bias)


def _worker(rank, init_file, result_queue):
    from torch.distributed.tensor import init_device_mesh

    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=60),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        completed = []
        for dtype_name, dtype in _DTYPES:
            _exercise(rank, mesh, dtype)
            completed.append(dtype_name)
        result_queue.put({"rank": rank, "completed": tuple(completed)})
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-hybrid-adamw-dtensor-")
    init_file = os.path.join(directory, "store")
    processes = [
        context.Process(target=_worker, args=(rank, init_file, result_queue))
        for rank in range(_WORLD)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=120))
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
        shutil.rmtree(directory, ignore_errors=True)
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), results
    return results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="DTensor Hybrid rebinding requires Gloo",
)
def test_hybrid_adamw_dtensor_rebinding_and_native_continuation_are_exact():
    results = _run()
    assert all("traceback" not in result for result in results), results
    expected = tuple(name for name, _dtype in _DTYPES)
    assert all(result["completed"] == expected for result in results), results
