"""Warning-strict distributed topology coverage for portable Hybrid state."""

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

from gefen import GefenMuonHybrid, load_portable_dcp, save_portable_dcp
from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    CheckpointTransport,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
    TopologyChange,
)
from gefen.portable_hybrid import _hybrid_portable_live_token
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_WORLD = 2
_MUON_FQN = "model.matrix"
_BACKUP_FQN = "model.vector"
_SOURCE_OWNER = "rank:0"
_TARGET_OWNER = "rank:1"
_SOURCE_LENGTHS = (3, 5)
_TARGET_LENGTHS = (5, 3)


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


def _members():
    return tuple("rank:{}".format(rank) for rank in range(_WORLD))


def _bindings(group, rank):
    member = _members()[rank]
    return (
        CodebookProcessGroupBinding(
            group,
            member,
            dist.group.WORLD,
            torch.device("cpu"),
        ),
        CheckpointProcessGroupBinding(
            group,
            member,
            dist.group.WORLD,
            torch.device("cpu"),
        ),
    )


def _owner_shards(identity, group, owner):
    return tuple(
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


def _replicated_shards(identity, group):
    return tuple(
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


def _flat_shards(identity, group, lengths):
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
    return tuple(shards)


def _muon_initial():
    return torch.linspace(-0.3, 0.2, 6, dtype=torch.float32).reshape(3, 2)


def _backup_initial():
    return torch.linspace(-0.4, 0.3, 8, dtype=torch.float32)


def _muon_momentum():
    return torch.tensor(
        [[-0.75, 0.5], [1.25, -1.5], [2.0, -2.5]],
        dtype=torch.float32,
    )


def _muon_normuon_v():
    return torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32)


def _backup_momentum():
    return torch.tensor(
        [-0.25, 0.5, -0.75, 1.0, -1.25, 1.5, -1.75, 2.0],
        dtype=torch.float32,
    )


def _backup_second_moment():
    return torch.linspace(0.2, 0.9, 8, dtype=torch.float32)


def _muon_gradient():
    return torch.tensor(
        [[0.4, -0.7], [1.1, -1.3], [1.7, -1.9]],
        dtype=torch.float32,
    )


def _backup_gradient():
    return torch.tensor(
        [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
        dtype=torch.float32,
    )


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _quantized_period_one(momentum):
    flat = momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8),
        torch.full((flat.numel(),), 255, dtype=torch.uint8),
    )
    return indices.reshape(-1, 1), flat.abs().reshape(-1, 1).clone()


def _make_hybrid(rank, group, *, owner, lengths, deterministic):
    muon_identity = ParameterIdentity(_MUON_FQN, (3, 2))
    backup_identity = ParameterIdentity(_BACKUP_FQN, (8,))
    old_muon = torch.nn.Parameter(_muon_initial().clone())
    old_backup = torch.nn.Parameter(_backup_initial().clone())
    optimizer = GefenMuonHybrid(
        [("matrix", old_muon)],
        [("vector", old_backup)],
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
        deterministic=deterministic,
        normuon=True,
        normuon_beta2=0.9,
        normuon_eps=3.0e-8,
    )
    muon_shards = _owner_shards(muon_identity, group, owner)
    backup_shards = _flat_shards(backup_identity, group, lengths)
    muon_local = old_muon if _members()[rank] == owner else None
    backup_shard = backup_shards[rank]
    start = backup_shard.logical_slice.flat_offset
    stop = start + backup_shard.logical_slice.length
    backup_local = torch.nn.Parameter(_backup_initial()[start:stop].clone())
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (
            ParameterRebinding(old_muon, muon_local, muon_shards[rank]),
            ParameterRebinding(old_backup, backup_local, backup_shard),
        ),
        manifest=ShardingManifest(muon_shards + backup_shards),
        codebook_process_group=codebook_binding,
    )
    return (
        optimizer,
        muon_local,
        backup_local,
        backup_shard,
        checkpoint_binding,
    )


def _seed_hybrid_state(optimizer, muon_parameter, backup_parameter, backup_shard):
    for child in (optimizer.muon, optimizer.backup):
        child._gefen_global_step = 13
        child._gefen_codebook = _codebook()
    if muon_parameter is not None:
        indices, magnitudes = _quantized_period_one(_muon_momentum())
        optimizer.muon.state[muon_parameter].update(
            {
                "automatic_period": 1,
                "step": 11,
                "m_codebook": indices,
                "m_magnitude": magnitudes,
                "normuon_v": _muon_normuon_v(),
                "normuon_step": 10,
            }
        )
    start = backup_shard.logical_slice.flat_offset
    stop = start + backup_shard.logical_slice.length
    momentum = _backup_momentum()[start:stop]
    indices, magnitudes = _quantized_period_one(momentum)
    optimizer.backup.state[backup_parameter].update(
        {
            "automatic_period": 1,
            "step": 11,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "vmean": _backup_second_moment()[start:stop].reshape(-1, 1).clone(),
            "vmean_step": 10,
        }
    )


def _bits_equal(left, right):
    return (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
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


def _local_child_exact(left, left_parameter, right, right_parameter):
    if left_parameter is None or right_parameter is None:
        return (
            left_parameter is right_parameter is None
            and not left.param_groups[0]["params"]
            and not right.param_groups[0]["params"]
            and not left.state
            and not right.state
        )
    return _state_equal(
        left.state[left_parameter],
        right.state[right_parameter],
    )


def _hybrids_exact(
    left,
    left_muon,
    left_backup,
    right,
    right_muon,
    right_backup,
):
    return (
        left._canonical_identity_ready()
        and right._canonical_identity_ready()
        and left._hybrid_fqn_roles == right._hybrid_fqn_roles == ((_MUON_FQN, "muon"), (_BACKUP_FQN, "backup"))
        and left._deterministic is right._deterministic is True
        and left.muon._deterministic is right.muon._deterministic is True
        and left.backup._deterministic is right.backup._deterministic is True
        and left.muon._gefen_global_step == right.muon._gefen_global_step
        and left.backup._gefen_global_step == right.backup._gefen_global_step
        and _bits_equal(
            left.muon._gefen_codebook,
            right.muon._gefen_codebook,
        )
        and _bits_equal(
            left.backup._gefen_codebook,
            right.backup._gefen_codebook,
        )
        and _local_child_exact(
            left.muon,
            left_muon,
            right.muon,
            right_muon,
        )
        and _local_child_exact(
            left.backup,
            left_backup,
            right.backup,
            right_backup,
        )
    )


def _step_against_reference(
    target,
    target_muon,
    target_backup,
    target_backup_shard,
    reference,
    reference_muon,
    reference_backup,
):
    if target_muon is not None:
        target_muon.grad = _muon_gradient().clone()
        reference_muon.grad = _muon_gradient().clone()
    start = target_backup_shard.logical_slice.flat_offset
    stop = start + target_backup_shard.logical_slice.length
    gradient = _backup_gradient()[start:stop]
    target_backup.grad = gradient.clone()
    reference_backup.grad = gradient.clone()
    target.step()
    reference.step()
    parameters_exact = (
        (target_muon is reference_muon is None) or _bits_equal(target_muon, reference_muon)
    ) and _bits_equal(target_backup, reference_backup)
    return parameters_exact and _hybrids_exact(
        target,
        target_muon,
        target_backup,
        reference,
        reference_muon,
        reference_backup,
    )


def _fresh_target_and_reference(rank, group):
    target = _make_hybrid(
        rank,
        group,
        owner=_TARGET_OWNER,
        lengths=_TARGET_LENGTHS,
        deterministic=False,
    )
    reference = _make_hybrid(
        rank,
        group,
        owner=_TARGET_OWNER,
        lengths=_TARGET_LENGTHS,
        deterministic=True,
    )
    _seed_hybrid_state(
        reference[0],
        reference[1],
        reference[2],
        reference[3],
    )
    return target, reference


def _successful_import_result(rank, group, document):
    target, reference = _fresh_target_and_reference(rank, group)
    target[0].import_portable_state(
        document,
        checkpoint_process_group=target[4],
        transaction_id="hybrid-direct-import-v1",
        limits=_limits(),
    )
    restored_exact = _hybrids_exact(
        target[0],
        target[1],
        target[2],
        reference[0],
        reference[1],
        reference[2],
    )
    next_step_exact = _step_against_reference(
        target[0],
        target[1],
        target[2],
        target[3],
        reference[0],
        reference[1],
        reference[2],
    )
    return {
        "restored_exact": restored_exact,
        "next_step_exact": next_step_exact,
        "target_owns_muon": target[1] is not None,
        "target_backup_length": target[3].logical_slice.length,
    }


def _asymmetric_failure_result(rank, group, document):
    target = _make_hybrid(
        rank,
        group,
        owner=_TARGET_OWNER,
        lengths=_TARGET_LENGTHS,
        deterministic=False,
    )
    optimizer, muon_parameter, backup_parameter, _shard, binding = target
    if rank == 0:
        optimizer.backup.param_groups[0]["rank_local_extension"] = "reject-me"
    token_before = _hybrid_portable_live_token(optimizer)
    child_state_objects = (optimizer.muon.state, optimizer.backup.state)
    parameter_values = (
        None if muon_parameter is None else muon_parameter.detach().clone(),
        backup_parameter.detach().clone(),
    )
    try:
        optimizer.import_portable_state(
            document,
            checkpoint_process_group=binding,
            transaction_id="hybrid-asymmetric-import-failure-v1",
            limits=_limits(),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = None
    parameters_unchanged = (
        (muon_parameter is None and parameter_values[0] is None) or _bits_equal(muon_parameter, parameter_values[0])
    ) and _bits_equal(backup_parameter, parameter_values[1])
    return {
        "message": message,
        "token_unchanged": _hybrid_portable_live_token(optimizer) == token_before,
        "state_objects_unchanged": (
            optimizer.muon.state is child_state_objects[0] and optimizer.backup.state is child_state_objects[1]
        ),
        "parameters_unchanged": parameters_unchanged,
        "common_unchanged": (
            optimizer._deterministic is False
            and optimizer.muon._gefen_global_step == 0
            and optimizer.backup._gefen_global_step == 0
            and optimizer.muon._gefen_codebook is None
            and optimizer.backup._gefen_codebook is None
        ),
    }


def _dcp_result(rank, group, checkpoint_dir):
    target, reference = _fresh_target_and_reference(rank, group)
    load_portable_dcp(
        target[0],
        checkpoint_process_group=target[4],
        storage_reader=FileSystemReader(checkpoint_dir),
        transaction_id="hybrid-dcp-load-v1",
        limits=_limits(),
    )
    restored_exact = _hybrids_exact(
        target[0],
        target[1],
        target[2],
        reference[0],
        reference[1],
        reference[2],
    )
    next_step_exact = _step_against_reference(
        target[0],
        target[1],
        target[2],
        target[3],
        reference[0],
        reference[1],
        reference[2],
    )
    return {
        "restored_exact": restored_exact,
        "next_step_exact": next_step_exact,
    }


def _strict_worker_warnings():
    warnings.simplefilter("error")
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
    )


def _distributed_worker(rank, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=120),
        )
        group = ProcessGroupIdentity("hybrid_portable_checkpoint", _members())
        source = _make_hybrid(
            rank,
            group,
            owner=_SOURCE_OWNER,
            lengths=_SOURCE_LENGTHS,
            deterministic=True,
        )
        _seed_hybrid_state(source[0], source[1], source[2], source[3])
        document = source[0].export_portable_state(
            checkpoint_process_group=source[4],
            transaction_id="hybrid-direct-export-v1",
            limits=_limits(),
        )
        direct = _successful_import_result(rank, group, document)
        dist.barrier()
        asymmetric = _asymmetric_failure_result(rank, group, document)
        dist.barrier()
        save_portable_dcp(
            source[0],
            checkpoint_process_group=source[4],
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="hybrid-dcp-save-v1",
            limits=_limits(),
        )
        dist.barrier()
        dcp = _dcp_result(rank, group, checkpoint_dir)
        dist.barrier()
        result_queue.put(
            {
                "rank": rank,
                "digest": document["completion"]["digest"],
                "document_exact": (
                    document["routing"] == {_MUON_FQN: "muon", _BACKUP_FQN: "backup"}
                    and all(
                        document["children"][role]["common"]["gefen_global_step"] == 13
                        and document["children"][role]["common"]["gefen_deterministic"] is True
                        and _bits_equal(
                            document["children"][role]["common"]["gefen_codebook"],
                            _codebook(),
                        )
                        for role in ("muon", "backup")
                    )
                ),
                "direct": direct,
                "asymmetric": asymmetric,
                "dcp": dcp,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_workers(checkpoint_dir):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-hybrid-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, init_file, checkpoint_dir, result_queue),
        )
        for rank in range(_WORLD)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=240))
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
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
    return sorted(results, key=lambda item: item["rank"])


def _make_singleton_hybrid(*, deterministic=True, replicated=False):
    member = "load:0" if replicated else "save:0"
    group = ProcessGroupIdentity(
        (
            "hybrid_portable_reverse_singleton"
            if replicated
            else "hybrid_portable_singleton"
        ),
        (member,),
    )
    muon_identity = ParameterIdentity(_MUON_FQN, (3, 2))
    backup_identity = ParameterIdentity(_BACKUP_FQN, (8,))
    muon_parameter = torch.nn.Parameter(_muon_initial().clone())
    backup_parameter = torch.nn.Parameter(_backup_initial().clone())
    optimizer = GefenMuonHybrid(
        [("matrix", muon_parameter)],
        [("vector", backup_parameter)],
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
        deterministic=deterministic,
        normuon=True,
        normuon_beta2=0.9,
        normuon_eps=3.0e-8,
    )
    muon_shard = (
        _replicated_shards(muon_identity, group)[0]
        if replicated
        else _owner_shards(muon_identity, group, member)[0]
    )
    backup_shard = (
        _replicated_shards(backup_identity, group)[0]
        if replicated
        else _flat_shards(backup_identity, group, (8,))[0]
    )
    optimizer.post_sharding(
        (
            ParameterRebinding(muon_parameter, muon_parameter, muon_shard),
            ParameterRebinding(backup_parameter, backup_parameter, backup_shard),
        ),
        manifest=ShardingManifest((muon_shard, backup_shard)),
        codebook_process_group=CodebookProcessGroupBinding(
            group,
            member,
            None,
            torch.device("cpu"),
        ),
    )
    return (
        optimizer,
        muon_parameter,
        backup_parameter,
        backup_shard,
        CheckpointProcessGroupBinding(
            group,
            member,
            None,
            torch.device("cpu"),
        ),
    )


def _has_replicated_portable_intersection(optimizer, *, same_topology):
    supports = tuple(
        support
        for support in optimizer.optimizer_contract().capabilities.checkpoints
        if support.transport is CheckpointTransport.CANONICAL_GLOBAL
    )
    return (
        len(supports) == 1
        and supports[0].same_topology
        == (
            frozenset({ParameterLayout.REPLICATED})
            if same_topology
            else frozenset()
        )
        and supports[0].topology_changing
        == frozenset({ParameterLayout.REPLICATED})
        and supports[0].topology_change_kinds
        == frozenset({TopologyChange.PLACEMENT_RESHARD})
        and supports[0].requires_collective
        and supports[0].atomic_load
    )


def _singleton_save_worker(_rank, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=120),
        )
        source = _make_singleton_hybrid()
        _seed_hybrid_state(source[0], source[1], source[2], source[3])
        save_portable_dcp(
            source[0],
            checkpoint_process_group=source[4],
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="hybrid-world-change-save",
            limits=_limits(),
        )
        keys = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
        result_queue.put(
            {
                "rank": 0,
                "v2": (
                    "optimizer.__gefen_portable_composite_metadata_v2__" in keys
                    and "optimizer.__gefen_portable_metadata_v1__" not in keys
                ),
            }
        )
    except BaseException:
        result_queue.put({"rank": 0, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _world_change_load_worker(rank, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=120),
        )
        group = ProcessGroupIdentity(
            "hybrid_portable_world_change_target",
            _members(),
        )
        target, reference = _fresh_target_and_reference(rank, group)
        load_portable_dcp(
            target[0],
            checkpoint_process_group=target[4],
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="hybrid-world-change-load",
            limits=_limits(),
        )
        restored_exact = _hybrids_exact(
            target[0],
            target[1],
            target[2],
            reference[0],
            reference[1],
            reference[2],
        )
        next_step_exact = _step_against_reference(
            target[0],
            target[1],
            target[2],
            target[3],
            reference[0],
            reference[1],
            reference[2],
        )
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


def _reverse_world_change_save_worker(rank, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=120),
        )
        group = ProcessGroupIdentity(
            "hybrid_portable_reverse_source",
            _members(),
        )
        source = _make_hybrid(
            rank,
            group,
            owner=_SOURCE_OWNER,
            lengths=_SOURCE_LENGTHS,
            deterministic=True,
        )
        _seed_hybrid_state(source[0], source[1], source[2], source[3])
        intersection_exact = _has_replicated_portable_intersection(
            source[0],
            same_topology=False,
        )
        save_portable_dcp(
            source[0],
            checkpoint_process_group=source[4],
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="hybrid-reverse-world-change-save",
            limits=_limits(),
        )
        keys = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
        result_queue.put(
            {
                "rank": rank,
                "intersection_exact": intersection_exact,
                "v2": (
                    "optimizer.__gefen_portable_composite_metadata_v2__" in keys
                    and "optimizer.__gefen_portable_metadata_v1__" not in keys
                ),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _reverse_world_change_load_worker(_rank, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=120),
        )
        target = _make_singleton_hybrid(deterministic=False, replicated=True)
        reference = _make_singleton_hybrid(deterministic=True, replicated=True)
        _seed_hybrid_state(
            reference[0],
            reference[1],
            reference[2],
            reference[3],
        )
        intersection_exact = _has_replicated_portable_intersection(
            target[0],
            same_topology=True,
        )
        load_portable_dcp(
            target[0],
            checkpoint_process_group=target[4],
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="hybrid-reverse-world-change-load",
            limits=_limits(),
        )
        restored_exact = _hybrids_exact(
            target[0],
            target[1],
            target[2],
            reference[0],
            reference[1],
            reference[2],
        )
        next_step_exact = _step_against_reference(
            target[0],
            target[1],
            target[2],
            target[3],
            reference[0],
            reference[1],
            reference[2],
        )
        result_queue.put(
            {
                "rank": 0,
                "intersection_exact": intersection_exact,
                "restored_exact": restored_exact,
                "next_step_exact": next_step_exact,
            }
        )
    except BaseException:
        result_queue.put({"rank": 0, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_phase(worker, world_size, checkpoint_dir):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-hybrid-phase-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=worker,
            args=(rank, init_file, checkpoint_dir, result_queue),
        )
        for rank in range(world_size)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=180))
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
    assert all(process.exitcode == 0 for process in processes)
    results = sorted(results, key=lambda item: item["rank"])
    assert all("fatal_error" not in result for result in results), results
    return results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable Hybrid topology coverage requires Gloo",
)
def test_two_process_portable_hybrid_topology_change_and_dcp(tmp_path):
    results = _run_distributed_workers(str(tmp_path / "hybrid-portable-dcp"))

    assert all("fatal_error" not in result for result in results), results
    assert len({result["digest"] for result in results}) == 1
    assert all(result["document_exact"] for result in results)
    assert [result["direct"]["target_owns_muon"] for result in results] == [
        False,
        True,
    ]
    assert [result["direct"]["target_backup_length"] for result in results] == [
        5,
        3,
    ]
    assert all(
        result["direct"]["restored_exact"]
        and result["direct"]["next_step_exact"]
        and result["dcp"]["restored_exact"]
        and result["dcp"]["next_step_exact"]
        for result in results
    ), results
    messages = [result["asymmetric"]["message"] for result in results]
    assert messages[0] == messages[1]
    assert messages[0] is not None and "unknown keys" in messages[0]
    assert all(
        result["asymmetric"]["token_unchanged"]
        and result["asymmetric"]["state_objects_unchanged"]
        and result["asymmetric"]["parameters_unchanged"]
        and result["asymmetric"]["common_unchanged"]
        for result in results
    ), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable Hybrid world-size coverage requires Gloo",
)
def test_sharded_hybrid_dcp_changes_world_size_and_continues_exactly(tmp_path):
    checkpoint_dir = str(tmp_path / "hybrid-world-change")
    saved = _run_phase(_singleton_save_worker, 1, checkpoint_dir)
    assert saved == [{"rank": 0, "v2": True}]
    loaded = _run_phase(_world_change_load_worker, _WORLD, checkpoint_dir)
    assert all(
        result["restored_exact"] and result["next_step_exact"]
        for result in loaded
    ), loaded


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable Hybrid reverse world-size coverage requires Gloo",
)
def test_sharded_hybrid_dcp_collapses_to_singleton_and_continues_exactly(tmp_path):
    checkpoint_dir = str(tmp_path / "hybrid-reverse-world-change")
    saved = _run_phase(
        _reverse_world_change_save_worker,
        _WORLD,
        checkpoint_dir,
    )
    assert all(
        result["intersection_exact"] and result["v2"]
        for result in saved
    ), saved
    loaded = _run_phase(
        _reverse_world_change_load_worker,
        1,
        checkpoint_dir,
    )
    assert loaded == [
        {
            "rank": 0,
            "intersection_exact": True,
            "restored_exact": True,
            "next_step_exact": True,
        }
    ]
