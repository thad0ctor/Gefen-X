"""Warning-strict distributed coverage for topology-neutral optimizer state."""

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


_WORLD = 2
_PLAIN_FQN = "model.weight"
_MUON_FQN = "model.matrix"


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=4 << 20,
        max_collective_metadata_bytes=4 << 20,
        chunk_bytes=7,
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
        CodebookProcessGroupBinding(group, member, dist.group.WORLD, torch.device("cpu")),
        CheckpointProcessGroupBinding(group, member, dist.group.WORLD, torch.device("cpu")),
    )


def _replicated_manifest(identity, group):
    members = group.ordered_members
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
                    len(members),
                ),
            ),
            process_group=group,
            local_member=member,
        )
        for coordinate, member in enumerate(members)
    )
    return ShardingManifest(shards), shards


def _flat_manifest(identity, group, lengths):
    members = group.ordered_members
    if len(lengths) != len(members) or sum(lengths) != identity.numel:
        raise AssertionError("invalid test partition")
    offset = 0
    shards = []
    for coordinate, (member, length) in enumerate(zip(members, lengths)):
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
                        len(members),
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return ShardingManifest(tuple(shards)), tuple(shards)


def _owner_manifest(identity, group, owner):
    members = group.ordered_members
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            LogicalSlice.full(identity) if member == owner else LogicalSlice(0, 0),
            placements=(
                ShardPlacement(
                    "checkpoint",
                    PlacementKind.WHOLE_PARAMETER_OWNER,
                    coordinate,
                    len(members),
                ),
            ),
            process_group=group,
            local_member=member,
            owner=owner,
        )
        for coordinate, member in enumerate(members)
    )
    return ShardingManifest(shards), shards


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _momentum_values(shape):
    bits = torch.tensor(
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
    return bits.view(torch.float32).reshape(shape).clone()


def _second_moment_values(shape):
    bits = torch.tensor(
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
    return bits.view(torch.float32).reshape(shape).clone()


def _quantized_period_one(momentum):
    indices = torch.where(
        torch.signbit(momentum.reshape(-1)),
        torch.zeros(momentum.numel(), dtype=torch.uint8),
        torch.full((momentum.numel(),), 255, dtype=torch.uint8),
    )
    return indices.reshape(-1, 1), momentum.abs().reshape(-1, 1).clone()


def _bits_equal(left, right):
    return (
        left.dtype == right.dtype == torch.float32
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))
    )


def _make_plain_replicated(rank, group, *, deterministic):
    identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
    parameter = torch.nn.Parameter(torch.zeros(identity.global_shape, dtype=torch.float32))
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
    manifest, shards = _replicated_manifest(identity, group)
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shards[rank]),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, parameter, checkpoint_binding


def _make_plain_flat(rank, group, lengths, *, deterministic):
    identity = ParameterIdentity(_PLAIN_FQN, (2, 4))
    original = torch.nn.Parameter(torch.zeros(identity.global_shape, dtype=torch.float32))
    local = torch.nn.Parameter(torch.zeros(lengths[rank], dtype=torch.float32))
    optimizer = Gefen(
        [("weight", original)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        force_2d_period_one=True,
        factored_v_2d=False,
        deterministic=deterministic,
    )
    manifest, shards = _flat_manifest(identity, group, lengths)
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (ParameterRebinding(original, local, shards[rank]),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, local, shards[rank], checkpoint_binding


def _seed_plain_state(optimizer, parameter):
    momentum = _momentum_values((2, 4))
    second_moment = _second_moment_values((2, 4))
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
    return momentum, second_moment


def _make_muon_owner(rank, group, *, owner, deterministic):
    identity = ParameterIdentity(_MUON_FQN, (3, 2))
    original = torch.nn.Parameter(torch.zeros(identity.global_shape, dtype=torch.float32))
    optimizer = GefenMuon(
        [("matrix", original)],
        lr=3.0e-3,
        weight_decay=0.02,
        momentum=0.85,
        nesterov=False,
        ns_steps=2,
        fused=False,
        sharded_mode="distributed",
        deterministic=deterministic,
        normuon=True,
        normuon_beta2=0.9,
        normuon_eps=3.0e-8,
    )
    manifest, shards = _owner_manifest(identity, group, owner)
    local = original if _members()[rank] == owner else None
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (ParameterRebinding(original, local, shards[rank]),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, local, checkpoint_binding


def _seed_muon_state(optimizer, parameter):
    optimizer._gefen_global_step = 13
    optimizer._gefen_codebook = _codebook()
    if parameter is None:
        return
    momentum = _momentum_values((2, 4)).reshape(-1)[:6].reshape(3, 2).clone()
    indices, magnitudes = _quantized_period_one(momentum)
    normuon_bits = torch.tensor([-2147483648, 1, 1082130432], dtype=torch.int32)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 11,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "normuon_v": normuon_bits.view(torch.float32).reshape(3, 1).clone(),
            "normuon_step": 10,
        }
    )


def _plain_result(rank, group):
    source, source_parameter, source_binding = _make_plain_replicated(rank, group, deterministic=True)
    expected_momentum, expected_second = _seed_plain_state(source, source_parameter)
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="plain-replicated-export-v1",
        limits=_limits(),
    )

    lengths = (3, 5)
    target, target_parameter, target_shard, target_binding = _make_plain_flat(
        rank,
        group,
        lengths,
        deterministic=False,
    )
    target.import_portable_state(
        document,
        checkpoint_process_group=target_binding,
        transaction_id="plain-flat-import-v1",
        limits=_limits(),
    )
    state = target.state[target_parameter]
    decoded = _decode_quantized_momentum(
        target._gefen_codebook,
        state["m_codebook"],
        state["m_magnitude"],
        logical_shape=(lengths[rank],),
        period=1,
        step=state["step"],
    )
    start = target_shard.logical_slice.flat_offset
    stop = start + target_shard.logical_slice.length
    expected_local_momentum = expected_momentum.reshape(-1)[start:stop].clone()
    expected_local_second = expected_second.reshape(-1)[start:stop].clone()
    parameter_record = document["parameters"][_PLAIN_FQN]["state"]
    return {
        "digest": document["completion"]["digest"],
        "document_momentum_exact": _bits_equal(parameter_record["momentum"], expected_momentum),
        "document_second_exact": _bits_equal(parameter_record["second_moment"], expected_second),
        "local_momentum_exact": _bits_equal(decoded, expected_local_momentum),
        "local_second_exact": _bits_equal(state["vmean"].reshape(-1), expected_local_second),
        "counters_exact": target._gefen_global_step == 9 and state["step"] == 7 and state["vmean_step"] == 6,
        "deterministic_imported": target._deterministic is True,
        "document": document,
    }


def _asymmetric_failure_result(rank, group, document):
    target, target_parameter, _, binding = _make_plain_flat(rank, group, (5, 3), deterministic=False)
    if rank == 0:
        target.param_groups[0]["rank_local_extension"] = "reject-me"
    token_before = target._canonical_import_live_token()
    state_object_before = target.state[target_parameter]
    try:
        target.import_portable_state(
            document,
            checkpoint_process_group=binding,
            transaction_id="plain-asymmetric-import-failure-v1",
            limits=_limits(),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = None
    return {
        "message": message,
        "token_unchanged": target._canonical_import_live_token() == token_before,
        "state_object_unchanged": target.state[target_parameter] is state_object_before,
        "still_pristine": target.state[target_parameter] == {"name": "weight"},
        "common_unchanged": target._gefen_global_step == 0 and target._gefen_codebook is None,
    }


def _muon_result(rank, group):
    source_owner = _members()[0]
    target_owner = _members()[1]
    source, source_parameter, source_binding = _make_muon_owner(
        rank,
        group,
        owner=source_owner,
        deterministic=True,
    )
    _seed_muon_state(source, source_parameter)
    document = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="muon-owner-export-v1",
        limits=_limits(),
    )

    target, target_parameter, target_binding = _make_muon_owner(
        rank,
        group,
        owner=target_owner,
        deterministic=False,
    )
    target.import_portable_state(
        document,
        checkpoint_process_group=target_binding,
        transaction_id="muon-owner-import-v1",
        limits=_limits(),
    )
    expected_momentum = _momentum_values((2, 4)).reshape(-1)[:6].reshape(3, 2).clone()
    expected_normuon = torch.tensor([-2147483648, 1, 1082130432], dtype=torch.int32).view(torch.float32).reshape(3, 1)
    record = document["parameters"][_MUON_FQN]["state"]
    if target_parameter is None:
        local_exact = len(target.state) == 0 and len(target.param_groups[0]["params"]) == 0
        counters_exact = target._gefen_global_step == 13
    else:
        state = target.state[target_parameter]
        decoded = _decode_quantized_momentum(
            target._gefen_codebook,
            state["m_codebook"],
            state["m_magnitude"],
            logical_shape=(3, 2),
            period=1,
            step=state["step"],
        )
        local_exact = _bits_equal(decoded, expected_momentum) and _bits_equal(state["normuon_v"], expected_normuon)
        counters_exact = target._gefen_global_step == 13 and state["step"] == 11 and state["normuon_step"] == 10
    return {
        "digest": document["completion"]["digest"],
        "document_momentum_exact": _bits_equal(record["momentum"], expected_momentum),
        "document_normuon_exact": _bits_equal(record["normuon_v"], expected_normuon),
        "local_exact": local_exact,
        "counters_exact": counters_exact,
        "deterministic_imported": target._deterministic is True,
        "target_has_storage": target_parameter is not None,
    }


def _distributed_worker(rank, init_file, result_queue):
    warnings.simplefilter("error")
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=90),
        )
        group = ProcessGroupIdentity("portable_checkpoint", _members())
        plain = _plain_result(rank, group)
        dist.barrier()
        asymmetric = _asymmetric_failure_result(rank, group, plain.pop("document"))
        dist.barrier()
        muon = _muon_result(rank, group)
        dist.barrier()
        result_queue.put(
            {
                "rank": rank,
                "plain": plain,
                "asymmetric": asymmetric,
                "muon": muon,
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_workers():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-runtime-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(target=_distributed_worker, args=(rank, init_file, result_queue)) for rank in range(_WORLD)
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
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable runtime topology-change coverage requires Gloo",
)
def test_two_process_portable_runtime_reshards_relocates_and_fails_atomically():
    results = _run_distributed_workers()

    assert all("fatal_error" not in result for result in results), results
    assert len({result["plain"]["digest"] for result in results}) == 1
    assert len({result["muon"]["digest"] for result in results}) == 1
    for result in results:
        plain = result["plain"]
        assert all(
            plain[key]
            for key in (
                "document_momentum_exact",
                "document_second_exact",
                "local_momentum_exact",
                "local_second_exact",
                "counters_exact",
                "deterministic_imported",
            )
        ), result
        muon = result["muon"]
        assert all(
            muon[key]
            for key in (
                "document_momentum_exact",
                "document_normuon_exact",
                "local_exact",
                "counters_exact",
                "deterministic_imported",
            )
        ), result
        assert muon["target_has_storage"] is (result["rank"] == 1)

    messages = [result["asymmetric"]["message"] for result in results]
    assert messages[0] == messages[1]
    assert messages[0] is not None and "unknown keys" in messages[0]
    assert all(
        all(
            result["asymmetric"][key]
            for key in (
                "token_unchanged",
                "state_object_unchanged",
                "still_pristine",
                "common_unchanged",
            )
        )
        for result in results
    ), results
