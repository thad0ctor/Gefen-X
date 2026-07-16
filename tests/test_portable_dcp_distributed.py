"""CPU-only distributed coverage for portable optimizer DCP persistence."""

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
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from gefen import load_portable_dcp, save_portable_dcp
from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.gefen import Gefen
from gefen.portable import _decode_quantized_momentum
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_SAVE_WORLD = 2
_FQN = "model.weight"


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=4 << 20,
        max_collective_metadata_bytes=4 << 20,
        chunk_bytes=11,
        max_members=4,
        max_metadata_bytes=1 << 20,
        max_tree_nodes=10_000,
        max_tree_depth=32,
        max_container_items=10_000,
        max_string_bytes=16 << 10,
        max_integer_bytes=128,
        max_tensors=256,
        max_tensor_rank=8,
        diagnostic_bytes=1024,
    )


def _save_members():
    return tuple("save:{}".format(rank) for rank in range(_SAVE_WORLD))


def _optimizer(parameter, *, deterministic):
    return Gefen(
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


def _bindings(identity, local_member, process_group):
    return (
        CodebookProcessGroupBinding(
            identity,
            local_member,
            process_group,
            torch.device("cpu"),
        ),
        CheckpointProcessGroupBinding(
            identity,
            local_member,
            process_group,
            torch.device("cpu"),
        ),
    )


def _flat_manifest(identity, group, lengths):
    if len(lengths) != len(group.ordered_members) or sum(lengths) != identity.numel:
        raise AssertionError("invalid flat test partition")
    shards = []
    offset = 0
    for coordinate, (member, length) in enumerate(zip(group.ordered_members, lengths)):
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, length),
                placements=(
                    ShardPlacement(
                        "checkpoint",
                        PlacementKind.FLAT_SHARD,
                        coordinate,
                        len(group.ordered_members),
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return ShardingManifest(tuple(shards)), tuple(shards)


def _replicated_manifest(identity, group):
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(
            ShardPlacement(
                "checkpoint",
                PlacementKind.REPLICATE,
                0,
                1,
            ),
        ),
        process_group=group,
        local_member=group.ordered_members[0],
    )
    return ShardingManifest((shard,)), shard


def _make_flat_source(rank, group):
    identity = ParameterIdentity(_FQN, (2, 4))
    lengths = (3, 5)
    original = torch.nn.Parameter(torch.zeros(identity.global_shape, dtype=torch.float32))
    local = torch.nn.Parameter(torch.zeros(lengths[rank], dtype=torch.float32))
    optimizer = _optimizer(original, deterministic=True)
    manifest, shards = _flat_manifest(identity, group, lengths)
    codebook_binding, checkpoint_binding = _bindings(group, _save_members()[rank], dist.group.WORLD)
    optimizer.post_sharding(
        (ParameterRebinding(original, local, shards[rank]),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, local, shards[rank], checkpoint_binding


def _make_replicated_target(initial_parameter, *, deterministic):
    identity = ParameterIdentity(_FQN, (2, 4))
    group = ProcessGroupIdentity("portable_dcp_singleton", ("load:0",))
    parameter = torch.nn.Parameter(initial_parameter.clone())
    optimizer = _optimizer(parameter, deterministic=deterministic)
    manifest, shard = _replicated_manifest(identity, group)
    codebook_binding, checkpoint_binding = _bindings(group, "load:0", None)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, parameter, checkpoint_binding


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _momentum_values():
    return (
        torch.tensor(
            [
                -2147483648,
                0,
                1,
                -2147483647,
                1056964608,
                -1090519040,
                1078984704,
                -1058013184,
            ],
            dtype=torch.int32,
        )
        .view(torch.float32)
        .reshape(2, 4)
        .clone()
    )


def _second_moment_values():
    return (
        torch.tensor(
            [
                -2147483648,
                0,
                1,
                8388608,
                1048576000,
                1065353216,
                1073741824,
                2139095039,
            ],
            dtype=torch.int32,
        )
        .view(torch.float32)
        .reshape(2, 4)
        .clone()
    )


def _quantized_period_one(momentum):
    flat = momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8),
        torch.full((flat.numel(),), 255, dtype=torch.uint8),
    )
    return indices.reshape(-1, 1), flat.abs().reshape(-1, 1).clone()


def _seed_state(optimizer, parameter, momentum, second_moment):
    indices, magnitudes = _quantized_period_one(momentum)
    optimizer._gefen_global_step = 9
    optimizer._gefen_codebook = _codebook()
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 7,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "vmean": second_moment.reshape(-1, 1).clone(),
            "vmean_step": 6,
        }
    )


def _bits_equal(left, right):
    return (
        type(left) is torch.Tensor
        and type(right) is torch.Tensor
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


def _capture_runtime_error(operation):
    try:
        operation()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("portable DCP operation unexpectedly succeeded")


def _strict_worker_warnings():
    warnings.simplefilter("error")
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
    )


def _save_worker(
    rank,
    init_file,
    checkpoint_dir,
    mismatch_dir,
    divergent_dir,
    invalid_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_SAVE_WORLD,
            timeout=timedelta(seconds=60),
        )
        group = ProcessGroupIdentity("portable_dcp_save", _save_members())
        optimizer, parameter, shard, checkpoint_binding = _make_flat_source(rank, group)
        momentum = _momentum_values().reshape(-1)
        second_moment = _second_moment_values().reshape(-1)
        start = shard.logical_slice.flat_offset
        stop = start + shard.logical_slice.length
        _seed_state(
            optimizer,
            parameter,
            momentum[start:stop].clone(),
            second_moment[start:stop].clone(),
        )

        save_portable_dcp(
            optimizer,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="portable-dcp-save-2-to-1",
            limits=_limits(),
            namespace="optimizer",
        )
        dist.barrier()

        token = optimizer._canonical_import_live_token()
        namespace_message = _capture_runtime_error(
            lambda: save_portable_dcp(
                optimizer,
                checkpoint_process_group=checkpoint_binding,
                storage_writer=FileSystemWriter(mismatch_dir),
                transaction_id="portable-dcp-mismatched-namespace",
                limits=_limits(),
                namespace="optimizer-a" if rank == 0 else "optimizer-b",
            )
        )
        namespace_unchanged = optimizer._canonical_import_live_token() == token
        dist.barrier()

        token = optimizer._canonical_import_live_token()
        storage_message = _capture_runtime_error(
            lambda: save_portable_dcp(
                optimizer,
                checkpoint_process_group=checkpoint_binding,
                storage_writer=FileSystemWriter("{}-{}".format(divergent_dir, rank)),
                transaction_id="portable-dcp-mismatched-storage",
                limits=_limits(),
                namespace="optimizer",
            )
        )
        storage_unchanged = optimizer._canonical_import_live_token() == token
        dist.barrier()

        token = optimizer._canonical_import_live_token()
        storage = object() if rank == 0 else FileSystemWriter(invalid_dir)
        invalid_message = _capture_runtime_error(
            lambda: save_portable_dcp(
                optimizer,
                checkpoint_process_group=checkpoint_binding,
                storage_writer=storage,
                transaction_id="portable-dcp-invalid-writer",
                limits=_limits(),
                namespace="optimizer",
            )
        )
        invalid_unchanged = optimizer._canonical_import_live_token() == token
        dist.barrier()

        result_queue.put(
            {
                "rank": rank,
                "saved": True,
                "namespace_message": namespace_message,
                "namespace_unchanged": namespace_unchanged,
                "storage_message": storage_message,
                "storage_unchanged": storage_unchanged,
                "invalid_message": invalid_message,
                "invalid_unchanged": invalid_unchanged,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _loaded_state_is_exact(optimizer, parameter):
    state = optimizer.state[parameter]
    decoded = _decode_quantized_momentum(
        optimizer._gefen_codebook,
        state["m_codebook"],
        state["m_magnitude"],
        logical_shape=(2, 4),
        period=1,
        step=state["step"],
    )
    return (
        _bits_equal(decoded, _momentum_values())
        and _bits_equal(state["vmean"].reshape(2, 4), _second_moment_values())
        and _bits_equal(optimizer._gefen_codebook, _codebook())
        and optimizer._gefen_global_step == 9
        and state["automatic_period"] == 1
        and state["step"] == 7
        and state["vmean_step"] == 6
        and optimizer._deterministic is True
    )


def _next_step_is_exact(optimizer, parameter, initial_parameter):
    oracle, oracle_parameter, _ = _make_replicated_target(initial_parameter, deterministic=True)
    _seed_state(oracle, oracle_parameter, _momentum_values(), _second_moment_values())
    gradient = torch.tensor(
        [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
        dtype=torch.float32,
    ).reshape(2, 4)
    parameter.grad = gradient.clone()
    oracle_parameter.grad = gradient.clone()
    optimizer.step()
    oracle.step()
    state = optimizer.state[parameter]
    oracle_state = oracle.state[oracle_parameter]
    return (
        _bits_equal(parameter.detach(), oracle_parameter.detach())
        and optimizer._gefen_global_step == oracle._gefen_global_step == 10
        and state["step"] == oracle_state["step"] == 8
        and state["vmean_step"] == oracle_state["vmean_step"] == 7
        and all(_bits_equal(state[key], oracle_state[key]) for key in ("m_codebook", "m_magnitude", "vmean"))
    )


def _load_worker(rank, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=1,
            timeout=timedelta(seconds=60),
        )
        initial_parameter = torch.linspace(-0.4, 0.3, 8, dtype=torch.float32).reshape(2, 4)
        optimizer, parameter, checkpoint_binding = _make_replicated_target(
            initial_parameter,
            deterministic=False,
        )
        load_portable_dcp(
            optimizer,
            checkpoint_process_group=checkpoint_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="portable-dcp-load-2-to-1",
            limits=_limits(),
            namespace="optimizer",
        )
        restored_exact = _loaded_state_is_exact(optimizer, parameter)
        next_step_exact = _next_step_is_exact(optimizer, parameter, initial_parameter)
        result_queue.put(
            {
                "rank": rank,
                "restored_exact": restored_exact,
                "next_step_exact": next_step_exact,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_phase(worker, world_size, *worker_args):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-dcp-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=worker,
            args=(rank, init_file, *worker_args, result_queue),
        )
        for rank in range(world_size)
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
        if os.path.exists(init_file):
            os.unlink(init_file)
    assert len(results) == world_size, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP topology-change coverage requires Gloo",
)
def test_portable_dcp_saves_flattened_world_and_loads_replicated_singleton(tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoint")
    save_results = _run_phase(
        _save_worker,
        _SAVE_WORLD,
        checkpoint_dir,
        str(tmp_path / "mismatched-namespace"),
        str(tmp_path / "mismatched-storage"),
        str(tmp_path / "invalid-writer"),
    )

    assert all("fatal_error" not in result for result in save_results), save_results
    assert all(result["saved"] for result in save_results)
    namespace_messages = [result["namespace_message"] for result in save_results]
    assert namespace_messages[0] == namespace_messages[1]
    assert "context" in namespace_messages[0]
    storage_messages = [result["storage_message"] for result in save_results]
    assert storage_messages[0] == storage_messages[1]
    assert "context" in storage_messages[0]
    invalid_messages = [result["invalid_message"] for result in save_results]
    assert invalid_messages[0] == invalid_messages[1]
    assert "StorageWriter" in invalid_messages[0]
    assert all(
        result["namespace_unchanged"] and result["storage_unchanged"] and result["invalid_unchanged"]
        for result in save_results
    )

    load_results = _run_phase(_load_worker, 1, checkpoint_dir)
    assert all("fatal_error" not in result for result in load_results), load_results
    assert load_results == [{"rank": 0, "restored_exact": True, "next_step_exact": True}]
