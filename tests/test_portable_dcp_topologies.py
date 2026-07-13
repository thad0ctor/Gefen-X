"""Warning-strict CPU/Gloo topology coverage for portable DCP."""

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
from gefen.gefen_muon import GefenMuon
from gefen.portable import _decode_quantized_momentum
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_PLAIN_FQN = "model.weight"
_MUON_FQN = "model.matrix"


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
