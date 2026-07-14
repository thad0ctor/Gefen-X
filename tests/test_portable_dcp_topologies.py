"""Warning-strict CPU/Gloo topology coverage for portable DCP."""

import copy
from dataclasses import replace
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
    CheckpointProjectionQualifier,
    CheckpointStateRepresentation,
    CheckpointStateTransitionKind,
    CheckpointTransport,
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
from gefen.gefen_muon import GefenMuon
from gefen.hybrid import GefenMuonHybrid
from gefen.portable import _decode_quantized_momentum
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_PLAIN_FQN = "model.weight"
_MUON_FQN = "model.matrix"


class _NoTouchFileSystemWriter(FileSystemWriter):
    def __init__(self, path):
        super().__init__(path)
        self.touched = False

    def _unexpected_io(self):
        self.touched = True
        raise AssertionError("portable DCP writer was touched during save preparation")

    def reset(self, *_args, **_kwargs):
        self._unexpected_io()

    def set_up_storage_writer(self, *_args, **_kwargs):
        self._unexpected_io()

    def prepare_local_plan(self, *_args, **_kwargs):
        self._unexpected_io()

    def prepare_global_plan(self, *_args, **_kwargs):
        self._unexpected_io()

    def write_data(self, *_args, **_kwargs):
        self._unexpected_io()

    def finish(self, *_args, **_kwargs):
        self._unexpected_io()


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=4 << 20,
        max_collective_metadata_bytes=4 << 20,
        chunk_bytes=13,
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


def _container_limits(max_container_items):
    return replace(_limits(), max_container_items=max_container_items)


def _strict_worker_warnings():
    warnings.simplefilter("error")
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
    )


def _bindings(group, local_member, process_group):
    return (
        CodebookProcessGroupBinding(
            group,
            local_member,
            process_group,
            torch.device("cpu"),
        ),
        CheckpointProcessGroupBinding(
            group,
            local_member,
            process_group,
            torch.device("cpu"),
        ),
    )


def _replicated_manifest(identity, group):
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
            placements=(
                ShardPlacement(
                    "checkpoint",
                    PlacementKind.REPLICATE,
                    coordinate,
                    len(group.ordered_members),
                ),
            ),
            process_group=group,
            local_member=member,
        )
        for coordinate, member in enumerate(group.ordered_members)
    )
    return ShardingManifest(shards), shards


def _flat_manifest(identity, group, lengths):
    if len(lengths) != len(group.ordered_members) or sum(lengths) != identity.numel:
        raise AssertionError("invalid flattened test partition")
    offset = 0
    shards = []
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
    shards = tuple(shards)
    return ShardingManifest(shards), shards


def _owner_manifest(identity, group, owner):
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            (LogicalSlice.full(identity) if member == owner else LogicalSlice(0, 0)),
            placements=(
                ShardPlacement(
                    "checkpoint",
                    PlacementKind.WHOLE_PARAMETER_OWNER,
                    coordinate,
                    len(group.ordered_members),
                ),
            ),
            process_group=group,
            local_member=member,
            owner=owner,
        )
        for coordinate, member in enumerate(group.ordered_members)
    )
    return ShardingManifest(shards), shards


def _plain_optimizer(parameter, *, deterministic):
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


def _scalar_optimizer(parameter, *, deterministic):
    return Gefen(
        [("scalar", parameter)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        factored_v_2d=False,
        period_one_substrings=("scalar",),
        deterministic=deterministic,
    )


def _projection_optimizer(parameter, *, factored, deterministic):
    return Gefen(
        [("weight", parameter)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        force_1d_period_one=True,
        force_2d_period_one=True,
        factored_v_2d=factored,
        deterministic=deterministic,
    )


def _muon_optimizer(parameter, *, deterministic, normuon):
    return GefenMuon(
        [("matrix", parameter)],
        lr=3.0e-3,
        weight_decay=0.02,
        momentum=0.85,
        nesterov=False,
        ns_steps=2,
        fused=False,
        sharded_mode="distributed",
        deterministic=deterministic,
        normuon=normuon,
        normuon_beta2=0.9,
        normuon_eps=3.0e-8,
    )


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _plain_initial_parameter():
    return torch.linspace(-0.4, 0.3, 8, dtype=torch.float32).reshape(2, 4)


def _plain_momentum():
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


def _plain_second_moment():
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


def _plain_gradient():
    return torch.tensor(
        [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
        dtype=torch.float32,
    ).reshape(2, 4)


def _muon_initial_parameter():
    return torch.linspace(-0.3, 0.2, 6, dtype=torch.float32).reshape(3, 2)


def _muon_momentum():
    return torch.tensor(
        [[-0.75, 0.5], [1.25, -1.5], [2.0, -2.5]],
        dtype=torch.float32,
    )


def _muon_normuon_v():
    return torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32)


def _muon_gradient():
    return torch.tensor(
        [[0.4, -0.7], [1.1, -1.3], [1.7, -1.9]],
        dtype=torch.float32,
    )


def _quantized_period_one(momentum):
    flat = momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8),
        torch.full((flat.numel(),), 255, dtype=torch.uint8),
    )
    return indices.reshape(-1, 1), flat.abs().reshape(-1, 1).clone()


def _seed_plain_state(optimizer, parameter, momentum, second_moment):
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


def _seed_scalar_state(optimizer, parameter):
    optimizer._gefen_global_step = 4
    optimizer._gefen_codebook = _codebook()
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 4,
            "m_codebook": torch.tensor([[255]], dtype=torch.uint8),
            "m_magnitude": torch.tensor([[2.0]], dtype=torch.float32),
            "vmean": torch.tensor([[3.0]], dtype=torch.float32),
            "vmean_step": 3,
        }
    )


def _seed_factored_state(optimizer, parameter, momentum, row, column):
    indices, magnitudes = _quantized_period_one(momentum)
    optimizer._gefen_global_step = 9
    optimizer._gefen_codebook = _codebook()
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 7,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "v_row": row.clone(),
            "v_col": column.clone(),
            "factored_step": 6,
        }
    )


def _seed_muon_state(optimizer, parameter, *, normuon):
    indices, magnitudes = _quantized_period_one(_muon_momentum())
    optimizer._gefen_global_step = 13
    optimizer._gefen_codebook = _codebook()
    state = optimizer.state[parameter]
    state.update(
        {
            "automatic_period": 1,
            "step": 11,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
        }
    )
    if normuon:
        state.update(
            {
                "normuon_v": _muon_normuon_v(),
                "normuon_step": 10,
            }
        )


def _bits_equal(left, right):
    return (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().contiguous().reshape(-1).view(torch.uint8),
        )
    )


def _state_equal(left, right):
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if torch.is_tensor(left_value) or torch.is_tensor(right_value):
            if not _bits_equal(left_value, right_value):
                return False
        elif type(left_value) is not type(right_value) or left_value != right_value:
            return False
    return True


def _plain_reference(local_parameter, momentum, second_moment):
    parameter = torch.nn.Parameter(local_parameter.clone())
    optimizer = _plain_optimizer(parameter, deterministic=True)
    _seed_plain_state(optimizer, parameter, momentum, second_moment)
    return optimizer, parameter


def _muon_reference(*, normuon):
    parameter = torch.nn.Parameter(_muon_initial_parameter().clone())
    optimizer = _muon_optimizer(
        parameter,
        deterministic=True,
        normuon=normuon,
    )
    _seed_muon_state(optimizer, parameter, normuon=normuon)
    return optimizer, parameter


def _capture_runtime_error(operation):
    try:
        operation()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("portable DCP operation unexpectedly succeeded")


def _run_phase(worker, world_size, *worker_args, timeout=180):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-dcp-topology-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=worker,
            args=(rank, world_size, init_file, *worker_args, result_queue),
        )
        for rank in range(world_size)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
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
    assert len(results) == world_size, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
    return sorted(results, key=lambda item: item["rank"])


def _plain_singleton_save_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        member = "save:0"
        group = ProcessGroupIdentity("plain_singleton_save", (member,))
        identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
        parameter = torch.nn.Parameter(_plain_initial_parameter().clone())
        optimizer = _plain_optimizer(parameter, deterministic=True)
        manifest, shards = _replicated_manifest(identity, group)
        codebook_binding, checkpoint_binding = _bindings(group, member, None)
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, shards[0]),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        _seed_plain_state(
            optimizer,
            parameter,
            _plain_momentum(),
            _plain_second_moment(),
        )
        save_portable_dcp(
            optimizer,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="plain-singleton-to-flat-save",
            limits=_limits(),
        )
        result_queue.put({"rank": rank, "saved": True})
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _plain_flat_load_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("load:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("plain_flat_load", members)
        identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
        lengths = (3, 5)
        manifest, shards = _flat_manifest(identity, group, lengths)
        shard = shards[rank]
        start = shard.logical_slice.flat_offset
        stop = start + shard.logical_slice.length
        original = torch.nn.Parameter(_plain_initial_parameter().clone())
        local_initial = _plain_initial_parameter().reshape(-1)[start:stop].clone()
        local = torch.nn.Parameter(local_initial.clone())
        optimizer = _plain_optimizer(original, deterministic=False)
        codebook_binding, checkpoint_binding = _bindings(
            group,
            members[rank],
            dist.group.WORLD,
        )
        optimizer.post_sharding(
            (ParameterRebinding(original, local, shard),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        load_portable_dcp(
            optimizer,
            checkpoint_process_group=checkpoint_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="plain-singleton-to-flat-load",
            limits=_limits(),
        )

        expected_momentum = _plain_momentum().reshape(-1)[start:stop].clone()
        expected_second = _plain_second_moment().reshape(-1)[start:stop].clone()
        state = optimizer.state[local]
        decoded = _decode_quantized_momentum(
            optimizer._gefen_codebook,
            state["m_codebook"],
            state["m_magnitude"],
            logical_shape=(lengths[rank],),
            period=1,
            step=state["step"],
        )
        restored_exact = (
            _bits_equal(decoded, expected_momentum)
            and _bits_equal(state["vmean"].reshape(-1), expected_second)
            and state["vmean_step"] == 6
            and state["step"] == 7
            and optimizer._gefen_global_step == 9
            and optimizer._deterministic is True
        )

        reference, reference_parameter = _plain_reference(
            local_initial,
            expected_momentum,
            expected_second,
        )
        gradient = _plain_gradient().reshape(-1)[start:stop].clone()
        local.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        optimizer.step()
        reference.step()
        continuation = {
            "parameter": _bits_equal(local, reference_parameter),
            "state": _state_equal(
                optimizer.state[local],
                reference.state[reference_parameter],
            ),
            "global_step": optimizer._gefen_global_step == reference._gefen_global_step,
            "codebook": _bits_equal(
                optimizer._gefen_codebook,
                reference._gefen_codebook,
            ),
        }
        next_step_exact = all(continuation.values())
        result_queue.put(
            {
                "rank": rank,
                "restored_exact": restored_exact,
                "next_step_exact": next_step_exact,
                "continuation": continuation,
                "length": lengths[rank],
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _muon_singleton_save_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    normuon,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        member = "save:0"
        group = ProcessGroupIdentity("muon_singleton_save", (member,))
        identity = ParameterIdentity(_MUON_FQN, (3, 2))
        parameter = torch.nn.Parameter(_muon_initial_parameter().clone())
        optimizer = _muon_optimizer(
            parameter,
            deterministic=True,
            normuon=normuon,
        )
        manifest, shards = _replicated_manifest(identity, group)
        codebook_binding, checkpoint_binding = _bindings(group, member, None)
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, shards[0]),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        _seed_muon_state(optimizer, parameter, normuon=normuon)
        save_portable_dcp(
            optimizer,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="muon-singleton-to-owner-save-{}".format(int(normuon)),
            limits=_limits(),
        )
        result_queue.put({"rank": rank, "saved": True})
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _muon_owner_load_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    normuon,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("load:{}".format(index) for index in range(world_size))
        owner = members[-1]
        group = ProcessGroupIdentity("muon_owner_load", members)
        identity = ParameterIdentity(_MUON_FQN, (3, 2))
        manifest, shards = _owner_manifest(identity, group, owner)
        original = torch.nn.Parameter(_muon_initial_parameter().clone())
        optimizer = _muon_optimizer(
            original,
            deterministic=False,
            normuon=normuon,
        )
        local = original if members[rank] == owner else None
        codebook_binding, checkpoint_binding = _bindings(
            group,
            members[rank],
            dist.group.WORLD,
        )
        optimizer.post_sharding(
            (ParameterRebinding(original, local, shards[rank]),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        load_portable_dcp(
            optimizer,
            checkpoint_process_group=checkpoint_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="muon-singleton-to-owner-load-{}".format(int(normuon)),
            limits=_limits(),
        )

        if local is None:
            restored_exact = (
                not optimizer.param_groups[0]["params"]
                and not optimizer.state
                and optimizer._gefen_global_step == 13
                and optimizer._deterministic is True
                and _bits_equal(optimizer._gefen_codebook, _codebook())
            )
        else:
            state = optimizer.state[local]
            decoded = _decode_quantized_momentum(
                optimizer._gefen_codebook,
                state["m_codebook"],
                state["m_magnitude"],
                logical_shape=(3, 2),
                period=1,
                step=state["step"],
            )
            restored_exact = (
                _bits_equal(decoded, _muon_momentum())
                and state["step"] == 11
                and optimizer._gefen_global_step == 13
                and optimizer._deterministic is True
                and (
                    not normuon or (_bits_equal(state["normuon_v"], _muon_normuon_v()) and state["normuon_step"] == 10)
                )
            )

        reference = None
        reference_parameter = None
        if local is not None:
            reference, reference_parameter = _muon_reference(normuon=normuon)
            local.grad = _muon_gradient().clone()
            reference_parameter.grad = _muon_gradient().clone()
        optimizer.step()
        if reference is not None:
            reference.step()
            continuation = {
                "parameter": _bits_equal(local, reference_parameter),
                "state": _state_equal(
                    optimizer.state[local],
                    reference.state[reference_parameter],
                ),
                "global_step": optimizer._gefen_global_step == reference._gefen_global_step,
                "codebook": _bits_equal(
                    optimizer._gefen_codebook,
                    reference._gefen_codebook,
                ),
            }
            next_step_exact = all(continuation.values())
        else:
            continuation = {
                "empty_nonowner": not optimizer.param_groups[0]["params"] and not optimizer.state,
                "global_step": optimizer._gefen_global_step == 14,
            }
            next_step_exact = all(continuation.values())
        result_queue.put(
            {
                "rank": rank,
                "owner": members[rank] == owner,
                "restored_exact": restored_exact,
                "next_step_exact": next_step_exact,
                "continuation": continuation,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _scalar_empty_chunk_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("rank:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("portable_dcp_scalar_empty_chunk", members)
        identity = ParameterIdentity("model.scalar", ())
        manifest, shards = _replicated_manifest(identity, group)
        codebook_binding, checkpoint_binding = _bindings(
            group,
            members[rank],
            dist.group.WORLD,
        )

        source_parameter = torch.nn.Parameter(torch.tensor(2.0))
        source = _scalar_optimizer(source_parameter, deterministic=True)
        source.post_sharding(
            (ParameterRebinding(source_parameter, source_parameter, shards[rank]),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        _seed_scalar_state(source, source_parameter)
        save_portable_dcp(
            source,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="scalar-empty-chunk-save",
            limits=_limits(),
        )
        dist.barrier()

        entries = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
        field_entries = [
            entry
            for key, entry in entries.items()
            if key.startswith("optimizer.__gefen_portable_field_")
        ]
        small_geometries = [
            sorted(
                (int(chunk.offsets[0]), int(chunk.sizes[0]))
                for chunk in entry.chunks
            )
            for entry in field_entries
            if tuple(entry.size) == (1,)
        ]
        v2_sharded = (
            "optimizer.__gefen_portable_metadata_v2__" in entries
            and "optimizer.__gefen_portable_metadata_v1__" not in entries
            and bool(field_entries)
            and all(len(entry.chunks) == world_size for entry in field_entries)
        )
        trailing_empty_chunks = bool(small_geometries) and all(
            geometry == [(0, 1), (1, 0)] for geometry in small_geometries
        )

        target_parameter = torch.nn.Parameter(torch.tensor(2.0))
        target = _scalar_optimizer(target_parameter, deterministic=False)
        target.post_sharding(
            (ParameterRebinding(target_parameter, target_parameter, shards[rank]),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        load_portable_dcp(
            target,
            checkpoint_process_group=checkpoint_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="scalar-empty-chunk-load",
            limits=_limits(),
        )
        restored_exact = (
            _state_equal(target.state[target_parameter], source.state[source_parameter])
            and target._gefen_global_step == source._gefen_global_step == 4
            and target._deterministic is source._deterministic is True
            and _bits_equal(target._gefen_codebook, source._gefen_codebook)
        )

        gradient = torch.tensor(0.25)
        target_parameter.grad = gradient.clone()
        source_parameter.grad = gradient.clone()
        target.step()
        source.step()
        continuation_exact = (
            _bits_equal(target_parameter, source_parameter)
            and _state_equal(target.state[target_parameter], source.state[source_parameter])
            and target._gefen_global_step == source._gefen_global_step == 5
            and _bits_equal(target._gefen_codebook, source._gefen_codebook)
        )

        mismatch_parameter = torch.nn.Parameter(torch.tensor(-7.0))
        mismatch = _scalar_optimizer(mismatch_parameter, deterministic=False)
        mismatch.post_sharding(
            (ParameterRebinding(mismatch_parameter, mismatch_parameter, shards[rank]),),
            manifest=manifest,
            codebook_process_group=codebook_binding,
        )
        if rank == 0:
            mismatch.param_groups[0]["eps"] = 1.0e-7
        before = mismatch._canonical_import_live_token()
        state_object = mismatch.state[mismatch_parameter]
        mismatch_message = _capture_runtime_error(
            lambda: load_portable_dcp(
                mismatch,
                checkpoint_process_group=checkpoint_binding,
                storage_reader=FileSystemReader(checkpoint_dir),
                transaction_id="scalar-empty-chunk-asymmetric-target",
                limits=_limits(),
            )
        )
        mismatch_unchanged = (
            mismatch._canonical_import_live_token() == before
            and mismatch.state[mismatch_parameter] is state_object
            and mismatch.state[mismatch_parameter] == {"name": "scalar"}
            and mismatch._gefen_global_step == 0
            and mismatch._gefen_codebook is None
        )

        identity_mismatch = ParameterIdentity("model.scalar", (1,))
        identity_mismatch_manifest, identity_mismatch_shards = _replicated_manifest(
            identity_mismatch,
            group,
        )
        identity_mismatch_parameter = torch.nn.Parameter(torch.tensor([-11.0]))
        identity_mismatch_target = _scalar_optimizer(
            identity_mismatch_parameter,
            deterministic=False,
        )
        identity_mismatch_target.post_sharding(
            (
                ParameterRebinding(
                    identity_mismatch_parameter,
                    identity_mismatch_parameter,
                    identity_mismatch_shards[rank],
                ),
            ),
            manifest=identity_mismatch_manifest,
            codebook_process_group=codebook_binding,
        )
        identity_before = identity_mismatch_target._canonical_import_live_token()
        identity_state_object = identity_mismatch_target.state[
            identity_mismatch_parameter
        ]
        identity_parameter_before = identity_mismatch_parameter.detach().clone()
        identity_mismatch_message = _capture_runtime_error(
            lambda: load_portable_dcp(
                identity_mismatch_target,
                checkpoint_process_group=checkpoint_binding,
                storage_reader=FileSystemReader(checkpoint_dir),
                transaction_id="scalar-empty-chunk-identity-mismatch",
                limits=_limits(),
            )
        )
        identity_mismatch_unchanged = (
            identity_mismatch_target._canonical_import_live_token()
            == identity_before
            and identity_mismatch_target.state[identity_mismatch_parameter]
            is identity_state_object
            and identity_mismatch_target.state[identity_mismatch_parameter]
            == {"name": "scalar"}
            and _bits_equal(
                identity_mismatch_parameter,
                identity_parameter_before,
            )
            and identity_mismatch_target._gefen_global_step == 0
            and identity_mismatch_target._gefen_codebook is None
        )
        result_queue.put(
            {
                "rank": rank,
                "v2_sharded": v2_sharded,
                "small_field_count": len(small_geometries),
                "trailing_empty_chunks": trailing_empty_chunks,
                "restored_exact": restored_exact,
                "continuation_exact": continuation_exact,
                "mismatch_message": mismatch_message,
                "mismatch_unchanged": mismatch_unchanged,
                "identity_mismatch_message": identity_mismatch_message,
                "identity_mismatch_unchanged": identity_mismatch_unchanged,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _factored_projection_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("rank:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("portable_dcp_factored_projection", members)
        identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
        replicated_manifest, replicated_shards = _replicated_manifest(identity, group)
        codebook_binding, checkpoint_binding = _bindings(
            group,
            members[rank],
            dist.group.WORLD,
        )
        full_parameter = _plain_initial_parameter()
        momentum = torch.tensor(
            [[-0.5, 0.75, -1.0, 1.25], [1.5, -1.75, 2.0, -2.25]],
            dtype=torch.float32,
        )
        row = torch.tensor([1.0, 3.0], dtype=torch.float32)
        column = torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32)
        gradient = _plain_gradient()

        source_parameter = torch.nn.Parameter(full_parameter.clone())
        source = _projection_optimizer(
            source_parameter,
            factored=True,
            deterministic=True,
        )
        source.post_sharding(
            (
                ParameterRebinding(
                    source_parameter,
                    source_parameter,
                    replicated_shards[rank],
                ),
            ),
            manifest=replicated_manifest,
            codebook_process_group=codebook_binding,
        )
        _seed_factored_state(source, source_parameter, momentum, row, column)

        canonical_support = next(
            support
            for support in source.optimizer_contract().capabilities.checkpoints
            if support.transport is CheckpointTransport.CANONICAL_GLOBAL
        )
        transition = next(
            item
            for item in canonical_support.state_transitions
            if item.source is CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
            and item.target is CheckpointStateRepresentation.BLOCK_SECOND_MOMENT
        )
        declared_projection = (
            transition.kind is CheckpointStateTransitionKind.DEFINED_PROJECTION
            and transition.qualifier
            is CheckpointProjectionQualifier.FACTORED_TO_BLOCK_LIVE_FP32_TARGET_PERIOD_ONE_V1
            and ParameterLayout.FLATTENED_ELEMENT_SHARD
            in transition.topology_changing
        )
        save_portable_dcp(
            source,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="factored-to-flat-block-save",
            limits=_limits(),
        )
        dist.barrier()
        entries = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
        v2_sharded = (
            "optimizer.__gefen_portable_metadata_v2__" in entries
            and "optimizer.__gefen_portable_metadata_v1__" not in entries
            and any(
                key.startswith("optimizer.__gefen_portable_field_")
                for key in entries
            )
        )

        lengths = (3, 5)
        flat_manifest, flat_shards = _flat_manifest(identity, group, lengths)
        shard = flat_shards[rank]
        start = shard.logical_slice.flat_offset
        stop = start + shard.logical_slice.length
        local_initial = full_parameter.reshape(-1)[start:stop].clone()
        original = torch.nn.Parameter(full_parameter.clone())
        target_parameter = torch.nn.Parameter(local_initial.clone())
        target = _projection_optimizer(
            original,
            factored=False,
            deterministic=False,
        )
        target.post_sharding(
            (ParameterRebinding(original, target_parameter, shard),),
            manifest=flat_manifest,
            codebook_process_group=codebook_binding,
        )

        from gefen.portable_dcp_sharded import (
            _load_sharded_payloads,
            _metadata_storage_spec,
            _normalize_sharded_metadata,
            _read_sharded_metadata,
            _stage_sharded_import,
        )
        from gefen.portable_schema import portable_state_digest
        from gefen.portable_state import _SECOND_MOMENT_PROJECTION_EXACT

        invalid_reader = FileSystemReader(checkpoint_dir)
        checkpoint_metadata = invalid_reader.read_metadata()
        source_metadata = _read_sharded_metadata(
            storage_reader=invalid_reader,
            checkpoint_metadata=checkpoint_metadata,
            binding=checkpoint_binding,
            namespace="optimizer",
            limits=_limits(),
            transaction_id="factored-to-flat-block-invalid-qualifier-metadata",
        )
        source_factored_state = source_metadata["parameters"][_PLAIN_FQN]["state"]
        denominator_field = source_metadata["fields"][
            source_factored_state["v_row_denominator_field"]
        ]
        v2_denominator = (
            denominator_field["name"] == "v_row_denominator"
            and denominator_field["shape"] == [1]
            and denominator_field["kind"] == "special"
            and denominator_field["nonnegative"] is True
        )
        metadata_shape = _metadata_storage_spec(
            checkpoint_metadata,
            namespace="optimizer",
            limits=_limits(),
        )
        invalid_local_fields = _load_sharded_payloads(
            storage_reader=invalid_reader,
            checkpoint_metadata=checkpoint_metadata,
            metadata=source_metadata,
            metadata_tensor=torch.empty(metadata_shape, dtype=torch.uint8),
            binding=checkpoint_binding,
            namespace="optimizer",
            limits=_limits(),
            transaction_id="factored-to-flat-block-invalid-qualifier-payload",
            context_digest=bytes.fromhex(
                source_metadata["completion"]["metadata_digest"]
            ),
        )
        invalid_metadata = copy.deepcopy(source_metadata)
        invalid_metadata["policy"][
            "second_moment_projection"
        ] = _SECOND_MOMENT_PROJECTION_EXACT
        invalid_semantic = {
            key: value
            for key, value in invalid_metadata.items()
            if key != "completion"
        }
        invalid_metadata["completion"]["metadata_digest"] = portable_state_digest(
            invalid_semantic
        )
        invalid_metadata = _normalize_sharded_metadata(
            invalid_metadata,
            limits=_limits(),
        )
        invalid_before = target._canonical_import_live_token()
        invalid_state_object = target.state[target_parameter]
        invalid_parameter_before = target_parameter.detach().clone()
        invalid_message = _capture_runtime_error(
            lambda: _stage_sharded_import(
                target,
                invalid_metadata,
                invalid_local_fields,
                binding=checkpoint_binding,
                limits=_limits(),
                transaction_id="factored-to-flat-block-invalid-qualifier",
            )
        )
        invalid_unchanged = (
            target._canonical_import_live_token() == invalid_before
            and target.state[target_parameter] is invalid_state_object
            and target.state[target_parameter] == {"name": "weight"}
            and _bits_equal(target_parameter, invalid_parameter_before)
            and target._gefen_global_step == 0
            and target._gefen_codebook is None
            and target._deterministic is False
        )

        load_portable_dcp(
            target,
            checkpoint_process_group=checkpoint_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="factored-to-flat-block-load",
            limits=_limits(),
        )

        expected_second = torch.outer(row, column).div_(
            row.mean().clamp_(min=torch.finfo(torch.float32).tiny)
        )
        expected_local_momentum = momentum.reshape(-1)[start:stop].clone()
        expected_local_second = expected_second.reshape(-1)[start:stop].clone()
        target_state = target.state[target_parameter]
        decoded = _decode_quantized_momentum(
            target._gefen_codebook,
            target_state["m_codebook"],
            target_state["m_magnitude"],
            logical_shape=(lengths[rank],),
            period=1,
            step=target_state["step"],
        )
        projection_exact = (
            _bits_equal(decoded, expected_local_momentum)
            and _bits_equal(
                target_state["vmean"].reshape(-1),
                expected_local_second,
            )
            and target_state["step"] == 7
            and target_state["vmean_step"] == 6
            and target._gefen_global_step == 9
            and target._deterministic is True
        )

        oracle_original = torch.nn.Parameter(full_parameter.clone())
        oracle_parameter = torch.nn.Parameter(local_initial.clone())
        oracle = _projection_optimizer(
            oracle_original,
            factored=False,
            deterministic=True,
        )
        oracle.post_sharding(
            (ParameterRebinding(oracle_original, oracle_parameter, shard),),
            manifest=flat_manifest,
            codebook_process_group=codebook_binding,
        )
        _seed_plain_state(
            oracle,
            oracle_parameter,
            expected_local_momentum,
            expected_local_second,
        )
        imported_state_exact = _state_equal(
            target.state[target_parameter],
            oracle.state[oracle_parameter],
        )
        local_gradient = gradient.reshape(-1)[start:stop].clone()
        target_parameter.grad = local_gradient.clone()
        oracle_parameter.grad = local_gradient.clone()
        target.step()
        oracle.step()
        continuation_exact = (
            _bits_equal(target_parameter, oracle_parameter)
            and _state_equal(
                target.state[target_parameter],
                oracle.state[oracle_parameter],
            )
            and target._gefen_global_step == oracle._gefen_global_step == 10
            and _bits_equal(target._gefen_codebook, oracle._gefen_codebook)
        )
        result_queue.put(
            {
                "rank": rank,
                "v2_sharded": v2_sharded,
                "v2_denominator": v2_denominator,
                "declared_projection": declared_projection,
                "length": lengths[rank],
                "invalid_message": invalid_message,
                "invalid_unchanged": invalid_unchanged,
                "projection_exact": projection_exact,
                "imported_state_exact": imported_state_exact,
                "continuation_exact": continuation_exact,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _stage_projection_failure_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    failure_site,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("rank:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("portable_dcp_stage_failure_{}".format(failure_site), members)
        identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
        replicated_manifest, replicated_shards = _replicated_manifest(identity, group)
        codebook_binding, checkpoint_binding = _bindings(group, members[rank], dist.group.WORLD)
        full_parameter = _plain_initial_parameter()
        momentum = _plain_momentum()

        source_parameter = torch.nn.Parameter(full_parameter.clone())
        source = _projection_optimizer(
            source_parameter,
            factored=failure_site == "factored_projection",
            deterministic=True,
        )
        source.post_sharding(
            (ParameterRebinding(source_parameter, source_parameter, replicated_shards[rank]),),
            manifest=replicated_manifest,
            codebook_process_group=codebook_binding,
        )
        if failure_site == "factored_projection":
            _seed_factored_state(
                source,
                source_parameter,
                momentum,
                torch.tensor([1.0, 3.0], dtype=torch.float32),
                torch.tensor([2.0, 4.0, 8.0, 16.0], dtype=torch.float32),
            )
        else:
            _seed_plain_state(source, source_parameter, momentum, _plain_second_moment())
        save_portable_dcp(
            source,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="stage-failure-{}-save".format(failure_site),
            limits=_limits(),
        )
        dist.barrier()

        flat_manifest, flat_shards = _flat_manifest(identity, group, (3, 5))
        shard = flat_shards[rank]
        start = shard.logical_slice.flat_offset
        stop = start + shard.logical_slice.length
        original = torch.nn.Parameter(full_parameter.clone())
        target_parameter = torch.nn.Parameter(full_parameter.reshape(-1)[start:stop].clone())
        target = _projection_optimizer(original, factored=False, deterministic=False)
        target.post_sharding(
            (ParameterRebinding(original, target_parameter, shard),),
            manifest=flat_manifest,
            codebook_process_group=codebook_binding,
        )
        before_token = target._canonical_import_live_token()
        before_state = target.state[target_parameter]
        before_parameter = target_parameter.detach().clone()

        if rank == 0:
            def injected_failure(*_args, **_kwargs):
                raise ValueError("injected {} failure".format(failure_site))

            if failure_site == "momentum_recompress":
                from gefen import portable as portable_module

                portable_module._recompress_dense_momentum = injected_failure
            else:
                from gefen import portable_dcp_sharded as sharded_module

                sharded_module._local_factored_projection = injected_failure

        message = _capture_runtime_error(
            lambda: load_portable_dcp(
                target,
                checkpoint_process_group=checkpoint_binding,
                storage_reader=FileSystemReader(checkpoint_dir),
                transaction_id="stage-failure-{}-load".format(failure_site),
                limits=_limits(),
            )
        )
        heartbeat = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(heartbeat)
        unchanged = (
            target._canonical_import_live_token() == before_token
            and target.state[target_parameter] is before_state
            and target.state[target_parameter] == {"name": "weight"}
            and _bits_equal(target_parameter, before_parameter)
            and target._gefen_global_step == 0
            and target._gefen_codebook is None
            and target._deterministic is False
        )
        result_queue.put(
            {
                "rank": rank,
                "message": message,
                "unchanged": unchanged,
                "heartbeat": heartbeat.item() == 3,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _projected_target_limit_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("rank:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("portable_dcp_projected_target_limit", members)
        identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
        replicated_manifest, replicated_shards = _replicated_manifest(identity, group)
        codebook_binding, checkpoint_binding = _bindings(group, members[rank], dist.group.WORLD)
        full_parameter = _plain_initial_parameter()
        source_parameter = torch.nn.Parameter(full_parameter.clone())
        source = _plain_optimizer(source_parameter, deterministic=True)
        source.post_sharding(
            (ParameterRebinding(source_parameter, source_parameter, replicated_shards[rank]),),
            manifest=replicated_manifest,
            codebook_process_group=codebook_binding,
        )
        _seed_plain_state(source, source_parameter, _plain_momentum(), _plain_second_moment())
        save_portable_dcp(
            source,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="projected-target-limit-save",
            limits=_limits(),
        )
        dist.barrier()

        flat_manifest, flat_shards = _flat_manifest(identity, group, (8, 0))
        shard = flat_shards[rank]
        start = shard.logical_slice.flat_offset
        stop = start + shard.logical_slice.length
        original = torch.nn.Parameter(full_parameter.clone())
        target_parameter = torch.nn.Parameter(full_parameter.reshape(-1)[start:stop].clone())
        target = _plain_optimizer(original, deterministic=False)
        target.post_sharding(
            (ParameterRebinding(original, target_parameter, shard),),
            manifest=flat_manifest,
            codebook_process_group=codebook_binding,
        )
        before_token = target._canonical_import_live_token()
        before_state = target.state[target_parameter]
        before_parameter = target_parameter.detach().clone()
        message = _capture_runtime_error(
            lambda: load_portable_dcp(
                target,
                checkpoint_process_group=checkpoint_binding,
                storage_reader=FileSystemReader(checkpoint_dir),
                transaction_id="projected-target-limit-load",
                limits=replace(_limits(), max_fragment_tensor_bytes=800),
            )
        )
        heartbeat = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(heartbeat)
        result_queue.put(
            {
                "rank": rank,
                "message": message,
                "unchanged": (
                    target._canonical_import_live_token() == before_token
                    and target.state[target_parameter] is before_state
                    and target.state[target_parameter] == {"name": "weight"}
                    and _bits_equal(target_parameter, before_parameter)
                    and target._gefen_global_step == 0
                    and target._gefen_codebook is None
                ),
                "heartbeat": heartbeat.item() == 3,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _save_chunk_limit_worker(
    rank,
    world_size,
    init_file,
    checkpoint_root,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        members = tuple("rank:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("portable_dcp_save_chunk_limits", members)
        codebook_binding, checkpoint_binding = _bindings(
            group,
            members[rank],
            dist.group.WORLD,
        )

        first = torch.nn.Parameter(
            torch.tensor([[0.5, -1.0], [1.5, -2.0]], dtype=torch.float32)
        )
        second = torch.nn.Parameter(
            torch.tensor([[2.5, -3.0], [3.5, -4.0]], dtype=torch.float32)
        )
        plain = Gefen(
            [("first", first), ("second", second)],
            lr=2.5e-3,
            betas=(0.8, 0.97),
            eps=2.0e-8,
            weight_decay=0.03,
            fused=False,
            force_2d_period_one=True,
            factored_v_2d=False,
            deterministic=True,
        )
        first_manifest, first_shards = _replicated_manifest(
            ParameterIdentity("model.first", (2, 2)),
            group,
        )
        second_manifest, second_shards = _replicated_manifest(
            ParameterIdentity("model.second", (2, 2)),
            group,
        )
        plain.post_sharding(
            (
                ParameterRebinding(first, first, first_shards[rank]),
                ParameterRebinding(second, second, second_shards[rank]),
            ),
            manifest=ShardingManifest(
                first_manifest.shards + second_manifest.shards
            ),
            codebook_process_group=codebook_binding,
        )
        _seed_plain_state(
            plain,
            first,
            torch.tensor(
                [[-0.5, 0.75], [-1.0, 1.25]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[0.25, 0.5], [0.75, 1.0]],
                dtype=torch.float32,
            ),
        )
        _seed_plain_state(
            plain,
            second,
            torch.tensor(
                [[1.5, -1.75], [2.0, -2.25]],
                dtype=torch.float32,
            ),
            torch.tensor(
                [[1.25, 1.5], [1.75, 2.0]],
                dtype=torch.float32,
            ),
        )
        plain_writer_path = "{}-plain".format(checkpoint_root)
        plain_writer = _NoTouchFileSystemWriter(plain_writer_path)
        plain_before = plain._canonical_import_live_token()
        plain_message = _capture_runtime_error(
            lambda: save_portable_dcp(
                plain,
                checkpoint_process_group=checkpoint_binding,
                storage_writer=plain_writer,
                transaction_id="plain-save-expected-chunk-limit",
                limits=_container_limits(10),
            )
        )
        plain_unchanged = plain._canonical_import_live_token() == plain_before
        dist.barrier()

        matrix = torch.nn.Parameter(_muon_initial_parameter().clone())
        bias_a = torch.nn.Parameter(
            torch.tensor([0.5, -1.0, 1.5, -2.0], dtype=torch.float32)
        )
        bias_b = torch.nn.Parameter(
            torch.tensor([2.5, -3.0, 3.5, -4.0], dtype=torch.float32)
        )
        hybrid = GefenMuonHybrid(
            [("matrix", matrix)],
            [("bias_a", bias_a), ("bias_b", bias_b)],
            lr=2.5e-3,
            muon_lr=3.0e-3,
            backup_lr=2.5e-3,
            weight_decay=0.03,
            muon_weight_decay=0.02,
            backup_weight_decay=0.03,
            backup_optimizer="gefen",
            backup_1d_period_one=True,
            betas=(0.8, 0.97),
            eps=2.0e-8,
            fused=False,
            momentum=0.85,
            nesterov=False,
            ns_steps=2,
            sharded_mode="distributed",
            deterministic=True,
            normuon=True,
            normuon_beta2=0.9,
            normuon_eps=3.0e-8,
        )
        matrix_manifest, matrix_shards = _replicated_manifest(
            ParameterIdentity("model.matrix", (3, 2)),
            group,
        )
        bias_a_manifest, bias_a_shards = _replicated_manifest(
            ParameterIdentity("model.bias_a", (4,)),
            group,
        )
        bias_b_manifest, bias_b_shards = _replicated_manifest(
            ParameterIdentity("model.bias_b", (4,)),
            group,
        )
        hybrid.post_sharding(
            (
                ParameterRebinding(matrix, matrix, matrix_shards[rank]),
                ParameterRebinding(bias_a, bias_a, bias_a_shards[rank]),
                ParameterRebinding(bias_b, bias_b, bias_b_shards[rank]),
            ),
            manifest=ShardingManifest(
                matrix_manifest.shards
                + bias_a_manifest.shards
                + bias_b_manifest.shards
            ),
            codebook_process_group=codebook_binding,
        )
        _seed_muon_state(hybrid.muon, matrix, normuon=True)
        _seed_plain_state(
            hybrid.backup,
            bias_a,
            torch.tensor([-0.5, 0.75, -1.0, 1.25], dtype=torch.float32),
            torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float32),
        )
        _seed_plain_state(
            hybrid.backup,
            bias_b,
            torch.tensor([1.5, -1.75, 2.0, -2.25], dtype=torch.float32),
            torch.tensor([1.25, 1.5, 1.75, 2.0], dtype=torch.float32),
        )
        hybrid.backup._gefen_global_step = 13
        from gefen import portable_dcp_sharded as sharded_module

        original_decode = sharded_module._decode_metadata_tensor
        if rank == 0:
            def fail_first_child_decode(*_args, **_kwargs):
                raise ValueError("injected first Hybrid child decode failure")

            sharded_module._decode_metadata_tensor = fail_first_child_decode
        phase_writer_path = "{}-hybrid-phase".format(checkpoint_root)
        phase_writer = _NoTouchFileSystemWriter(phase_writer_path)
        phase_before = (
            hybrid.muon._canonical_import_live_token(),
            hybrid.backup._canonical_import_live_token(),
        )
        hybrid_phase_message = _capture_runtime_error(
            lambda: save_portable_dcp(
                hybrid,
                checkpoint_process_group=checkpoint_binding,
                storage_writer=phase_writer,
                transaction_id="hybrid-save-first-child-decode-failure",
                limits=_limits(),
            )
        )
        sharded_module._decode_metadata_tensor = original_decode
        phase_heartbeat = torch.tensor([rank + 1], dtype=torch.int64)
        dist.all_reduce(phase_heartbeat)
        hybrid_phase_unchanged = phase_before == (
            hybrid.muon._canonical_import_live_token(),
            hybrid.backup._canonical_import_live_token(),
        )

        projected_path = "{}-hybrid-projected".format(checkpoint_root)
        save_portable_dcp(
            hybrid,
            checkpoint_process_group=checkpoint_binding,
            storage_writer=FileSystemWriter(projected_path),
            transaction_id="hybrid-projected-limit-save",
            limits=_limits(),
        )
        dist.barrier()
        projected_before = (
            hybrid.muon._canonical_import_live_token(),
            hybrid.backup._canonical_import_live_token(),
        )
        hybrid_projected_message = _capture_runtime_error(
            lambda: load_portable_dcp(
                hybrid,
                checkpoint_process_group=checkpoint_binding,
                storage_reader=FileSystemReader(projected_path),
                transaction_id="hybrid-projected-limit-load",
                limits=replace(_limits(), max_fragment_tensor_bytes=1500),
            )
        )
        hybrid_projected_unchanged = projected_before == (
            hybrid.muon._canonical_import_live_token(),
            hybrid.backup._canonical_import_live_token(),
        )

        hybrid_writer_path = "{}-hybrid".format(checkpoint_root)
        hybrid_writer = _NoTouchFileSystemWriter(hybrid_writer_path)
        hybrid_before = (
            hybrid.muon._canonical_import_live_token(),
            hybrid.backup._canonical_import_live_token(),
        )
        hybrid_message = _capture_runtime_error(
            lambda: save_portable_dcp(
                hybrid,
                checkpoint_process_group=checkpoint_binding,
                storage_writer=hybrid_writer,
                transaction_id="hybrid-save-expected-chunk-limit",
                limits=_container_limits(18),
            )
        )
        hybrid_unchanged = hybrid_before == (
            hybrid.muon._canonical_import_live_token(),
            hybrid.backup._canonical_import_live_token(),
        )
        result_queue.put(
            {
                "rank": rank,
                "plain_message": plain_message,
                "plain_writer_untouched": (
                    not plain_writer.touched
                    and not os.path.exists(plain_writer_path)
                ),
                "plain_unchanged": plain_unchanged,
                "hybrid_message": hybrid_message,
                "hybrid_phase_message": hybrid_phase_message,
                "hybrid_phase_writer_untouched": (
                    not phase_writer.touched
                    and not os.path.exists(phase_writer_path)
                ),
                "hybrid_phase_unchanged": hybrid_phase_unchanged,
                "hybrid_phase_heartbeat": phase_heartbeat.item() == 3,
                "hybrid_projected_message": hybrid_projected_message,
                "hybrid_projected_unchanged": hybrid_projected_unchanged,
                "hybrid_writer_untouched": (
                    not hybrid_writer.touched
                    and not os.path.exists(hybrid_writer_path)
                ),
                "hybrid_unchanged": hybrid_unchanged,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _subgroup_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        subgroup_ranks = (0, 2)
        subgroup = dist.new_group(
            ranks=list(subgroup_ranks),
            backend="gloo",
            timeout=timedelta(seconds=60),
        )
        if rank not in subgroup_ranks:
            dist.barrier()
            result_queue.put({"rank": rank, "excluded": True})
            return

        coordinate = subgroup_ranks.index(rank)
        members = tuple("subgroup:{}".format(item) for item in subgroup_ranks)
        group = ProcessGroupIdentity("portable_dcp_subgroup", members)
        identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
        manifest, shards = _replicated_manifest(identity, group)
        source_parameter = torch.nn.Parameter(_plain_initial_parameter().clone())
        source = _plain_optimizer(source_parameter, deterministic=True)
        source_codebook, source_checkpoint = _bindings(
            group,
            members[coordinate],
            subgroup,
        )
        source.post_sharding(
            (
                ParameterRebinding(
                    source_parameter,
                    source_parameter,
                    shards[coordinate],
                ),
            ),
            manifest=manifest,
            codebook_process_group=source_codebook,
        )
        _seed_plain_state(
            source,
            source_parameter,
            _plain_momentum(),
            _plain_second_moment(),
        )
        save_portable_dcp(
            source,
            checkpoint_process_group=source_checkpoint,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="supported-subgroup-save",
            limits=_limits(),
        )

        target_parameter = torch.nn.Parameter(_plain_initial_parameter().clone())
        target = _plain_optimizer(target_parameter, deterministic=False)
        target_codebook, target_checkpoint = _bindings(
            group,
            members[coordinate],
            subgroup,
        )
        target.post_sharding(
            (
                ParameterRebinding(
                    target_parameter,
                    target_parameter,
                    shards[coordinate],
                ),
            ),
            manifest=manifest,
            codebook_process_group=target_codebook,
        )
        load_portable_dcp(
            target,
            checkpoint_process_group=target_checkpoint,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="supported-subgroup-load",
            limits=_limits(),
        )
        restored_exact = (
            _state_equal(target.state[target_parameter], source.state[source_parameter])
            and target._gefen_global_step == source._gefen_global_step
            and target._deterministic is True
            and _bits_equal(target._gefen_codebook, source._gefen_codebook)
        )

        reference, reference_parameter = _plain_reference(
            _plain_initial_parameter(),
            _plain_momentum(),
            _plain_second_moment(),
        )
        target_parameter.grad = _plain_gradient().clone()
        reference_parameter.grad = _plain_gradient().clone()
        target.step()
        reference.step()
        continuation = {
            "parameter": _bits_equal(target_parameter, reference_parameter),
            "state": _state_equal(
                target.state[target_parameter],
                reference.state[reference_parameter],
            ),
            "global_step": target._gefen_global_step == reference._gefen_global_step,
        }
        next_step_exact = all(continuation.values())
        dist.barrier()
        result_queue.put(
            {
                "rank": rank,
                "excluded": False,
                "restored_exact": restored_exact,
                "next_step_exact": next_step_exact,
                "continuation": continuation,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP topology coverage requires Gloo",
)
def test_portable_dcp_replicated_singleton_to_uneven_flattened_world(tmp_path):
    checkpoint_dir = str(tmp_path / "plain-singleton-to-flat")
    save_results = _run_phase(
        _plain_singleton_save_worker,
        1,
        checkpoint_dir,
    )
    assert save_results == [{"rank": 0, "saved": True}]

    load_results = _run_phase(
        _plain_flat_load_worker,
        2,
        checkpoint_dir,
    )
    assert all("fatal_error" not in result for result in load_results), load_results
    assert [result["length"] for result in load_results] == [3, 5]
    assert all(result["restored_exact"] and result["next_step_exact"] for result in load_results), load_results


@pytest.mark.parametrize("normuon", [False, True])
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP topology coverage requires Gloo",
)
def test_portable_dcp_muon_replicated_singleton_to_world_owner(tmp_path, normuon):
    checkpoint_dir = str(tmp_path / "muon-singleton-to-owner-{}".format(int(normuon)))
    save_results = _run_phase(
        _muon_singleton_save_worker,
        1,
        checkpoint_dir,
        normuon,
    )
    assert save_results == [{"rank": 0, "saved": True}]

    load_results = _run_phase(
        _muon_owner_load_worker,
        2,
        checkpoint_dir,
        normuon,
    )
    assert all("fatal_error" not in result for result in load_results), load_results
    assert [result["owner"] for result in load_results] == [False, True]
    assert all(result["restored_exact"] and result["next_step_exact"] for result in load_results), load_results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP empty-chunk coverage requires Gloo",
)
def test_portable_dcp_two_rank_scalar_fields_accept_trailing_empty_chunks_and_continue(tmp_path):
    results = _run_phase(
        _scalar_empty_chunk_worker,
        2,
        str(tmp_path / "scalar-empty-chunk"),
    )
    assert all("fatal_error" not in result for result in results), results
    assert all(result["v2_sharded"] for result in results), results
    assert all(result["small_field_count"] == 2 for result in results), results
    assert all(result["trailing_empty_chunks"] for result in results), results
    assert all(result["restored_exact"] and result["continuation_exact"] for result in results), results
    assert all(result["mismatch_unchanged"] for result in results), results
    mismatch_messages = [result["mismatch_message"] for result in results]
    assert mismatch_messages[0] == mismatch_messages[1]
    assert "algorithm options do not match" in mismatch_messages[0]
    assert all(result["identity_mismatch_unchanged"] for result in results), results
    identity_mismatch_messages = [
        result["identity_mismatch_message"] for result in results
    ]
    assert identity_mismatch_messages[0] == identity_mismatch_messages[1]
    assert "parameter identity does not match the target" in identity_mismatch_messages[0]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP defined-projection coverage requires Gloo",
)
def test_portable_dcp_projects_replicated_factored_state_to_uneven_flat_block_shards(tmp_path):
    results = _run_phase(
        _factored_projection_worker,
        2,
        str(tmp_path / "factored-to-flat-block"),
    )
    assert all("fatal_error" not in result for result in results), results
    assert [result["length"] for result in results] == [3, 5]
    invalid_messages = [result["invalid_message"] for result in results]
    assert invalid_messages[0] == invalid_messages[1]
    assert "source policy does not authorize factored-to-block" in invalid_messages[0]
    assert all(
        result["v2_sharded"]
        and result["v2_denominator"]
        and result["declared_projection"]
        and result["invalid_unchanged"]
        and result["projection_exact"]
        and result["imported_state_exact"]
        and result["continuation_exact"]
        for result in results
    ), results


@pytest.mark.parametrize("failure_site", ["momentum_recompress", "factored_projection"])
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP stage failure-protocol coverage requires Gloo",
)
def test_portable_dcp_stage_projection_failures_are_collective_and_atomic(tmp_path, failure_site):
    results = _run_phase(
        _stage_projection_failure_worker,
        2,
        str(tmp_path / failure_site),
        failure_site,
    )
    assert all("fatal_error" not in result for result in results), results
    messages = [result["message"] for result in results]
    assert messages[0] == messages[1]
    assert "injected {} failure".format(failure_site) in messages[0]
    assert all(result["unchanged"] and result["heartbeat"] for result in results), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP projected target-limit coverage requires Gloo",
)
def test_portable_dcp_rejects_uneven_target_projected_state_before_payload_read(tmp_path):
    results = _run_phase(
        _projected_target_limit_worker,
        2,
        str(tmp_path / "projected-target-limit"),
    )
    assert all("fatal_error" not in result for result in results), results
    messages = [result["message"] for result in results]
    assert messages[0] == messages[1]
    assert "projected target state exceeds max_fragment_tensor_bytes" in messages[0]
    assert all(result["unchanged"] and result["heartbeat"] for result in results), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP save-limit coverage requires Gloo",
)
def test_portable_dcp_save_rejects_expected_plain_and_hybrid_chunks_before_writer_io(tmp_path):
    results = _run_phase(
        _save_chunk_limit_worker,
        2,
        str(tmp_path / "save-chunk-limits"),
        timeout=240,
    )
    assert all("fatal_error" not in result for result in results), results
    plain_messages = [result["plain_message"] for result in results]
    assert plain_messages[0] == plain_messages[1]
    assert "sharded portable DCP chunks exceed aggregate limits" in plain_messages[0]
    hybrid_messages = [result["hybrid_message"] for result in results]
    assert hybrid_messages[0] == hybrid_messages[1]
    assert "sharded Hybrid DCP chunks exceed aggregate limits" in hybrid_messages[0]
    phase_messages = [result["hybrid_phase_message"] for result in results]
    assert phase_messages[0] == phase_messages[1]
    assert "injected first Hybrid child decode failure" in phase_messages[0]
    projected_messages = [result["hybrid_projected_message"] for result in results]
    assert projected_messages[0] == projected_messages[1]
    assert "Hybrid projected target state exceeds max_fragment_tensor_bytes" in projected_messages[0]
    assert all(
        result["plain_writer_untouched"]
        and result["plain_unchanged"]
        and result["hybrid_writer_untouched"]
        and result["hybrid_unchanged"]
        and result["hybrid_phase_writer_untouched"]
        and result["hybrid_phase_unchanged"]
        and result["hybrid_phase_heartbeat"]
        and result["hybrid_projected_unchanged"]
        for result in results
    ), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable DCP subgroup coverage requires Gloo",
)
def test_portable_dcp_supported_adapter_defined_subgroup(tmp_path):
    results = _run_phase(
        _subgroup_worker,
        3,
        str(tmp_path / "supported-subgroup"),
        timeout=240,
    )
    assert all("fatal_error" not in result for result in results), results
    assert results[1] == {"rank": 1, "excluded": True}
    participants = (results[0], results[2])
    assert all(not result["excluded"] for result in participants)
    assert all(result["restored_exact"] and result["next_step_exact"] for result in participants), results
