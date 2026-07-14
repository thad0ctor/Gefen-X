"""CPU/Gloo coverage for bounded sharded portable-DCP redistribution."""

from datetime import timedelta
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    LogicalRegion,
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
from gefen.portable_dcp_sharded import (
    _DCP_SHARDED_METADATA_KEY,
    _allocate_canonical_dtensor,
    _as_canonical_dtensor,
    _canonical_local_tensor,
    _canonical_projection_send_numel,
    _canonical_segments,
    _decode_metadata_tensor,
    _dense_local_numel,
    _dense_projection_send_numel,
    _dense_segments_for_shard,
    _factored_axis_segments,
    _full_segment,
    _load_sharded_payloads,
    _local_factored_projection,
    _prepare_sharded_save_state,
    _read_sharded_metadata,
    _redistribute_canonical_to_dense,
    _redistribute_dense_to_canonical,
    _redistribute_flat_field,
    _segment_global_indices,
    _validate_sharded_dcp_entries,
)
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=4 << 20,
        max_collective_metadata_bytes=4 << 20,
        max_metadata_bytes=1 << 20,
        max_members=4,
        max_tree_nodes=10_000,
        max_tree_depth=32,
        max_container_items=10_000,
        max_string_bytes=16 << 10,
        max_integer_bytes=128,
        max_tensors=256,
        max_tensor_rank=8,
        diagnostic_bytes=1024,
    )


def _dtensor_shards(identity, group, *, dimension):
    chunk = (identity.global_shape[dimension] + len(group.ordered_members) - 1) // len(
        group.ordered_members
    )
    shards = []
    for coordinate, member in enumerate(group.ordered_members):
        offset = min(coordinate * chunk, identity.global_shape[dimension])
        length = max(
            0,
            min(identity.global_shape[dimension], offset + chunk) - offset,
        )
        offsets = [0] * len(identity.global_shape)
        lengths = list(identity.global_shape)
        offsets[dimension] = offset
        lengths[dimension] = length
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
                LogicalRegion(tuple(offsets), tuple(lengths)),
                placements=(
                    ShardPlacement(
                        "model",
                        PlacementKind.DIMENSION_SHARD,
                        coordinate,
                        len(group.ordered_members),
                        dimension,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
    return tuple(shards)


def _flat_shards(identity, group, lengths):
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
                        "model",
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


def test_factored_axis_routes_only_compact_flat_head_and_tail():
    group = ProcessGroupIdentity("factored_flat_segments", ("rank:0", "rank:1", "rank:2"))
    identity = ParameterIdentity("model.matrix", (3, 5))
    shard = _flat_shards(identity, group, (4, 3, 8))[1]
    row_segments = _factored_axis_segments(shard, axis=0)
    column_segments = _factored_axis_segments(shard, axis=1)
    assert _segment_global_indices(row_segments).tolist() == [0, 1]
    assert _segment_global_indices(column_segments).tolist() == [0, 1, 4]
    assert sum(segment.length for segment in column_segments) < identity.global_shape[1]


@pytest.mark.parametrize(
    ("dimension", "coordinate", "expected_rows", "expected_columns"),
    [
        (0, 0, [0, 1], [0, 1, 2, 3, 4, 5]),
        (1, 1, [0, 1, 2, 3], [3, 4, 5]),
    ],
)
def test_factored_axis_routes_dtensor_dimension_slices(
    dimension,
    coordinate,
    expected_rows,
    expected_columns,
):
    group = ProcessGroupIdentity("factored_dtensor_segments", ("rank:0", "rank:1"))
    identity = ParameterIdentity("model.matrix", (4, 6))
    shard = _dtensor_shards(identity, group, dimension=dimension)[coordinate]
    assert _segment_global_indices(_factored_axis_segments(shard, axis=0)).tolist() == expected_rows
    assert _segment_global_indices(_factored_axis_segments(shard, axis=1)).tolist() == expected_columns


def test_factored_axis_routes_empty_dtensor_shard_without_factors():
    group = ProcessGroupIdentity("factored_empty_segments", ("rank:0", "rank:1"))
    identity = ParameterIdentity("model.matrix", (2, 1))
    empty = _dtensor_shards(identity, group, dimension=1)[1]
    assert _factored_axis_segments(empty, axis=0) == ()
    assert _factored_axis_segments(empty, axis=1) == ()


def test_compact_factored_projection_uses_saved_fp32_denominator_exactly():
    group = ProcessGroupIdentity("factored_projection_segments", ("rank:0", "rank:1", "rank:2"))
    identity = ParameterIdentity("model.matrix", (3, 5))
    shard = _flat_shards(identity, group, (4, 3, 8))[1]
    full_row = torch.tensor([1.0, 1.0000001192092896, 16.0], dtype=torch.float32)
    full_column = torch.tensor([2.0, 4.0, 8.0, 16.0, 32.0], dtype=torch.float32)
    row_indices = _segment_global_indices(_factored_axis_segments(shard, axis=0))
    column_indices = _segment_global_indices(_factored_axis_segments(shard, axis=1))
    denominator = full_row.mean().clamp_(min=torch.finfo(torch.float32).tiny).reshape(1)
    projected = _local_factored_projection(
        full_row[row_indices],
        row_indices,
        full_column[column_indices],
        column_indices,
        denominator,
        shard,
    )
    expected = torch.outer(full_row, full_column).div_(denominator.reshape(())).reshape(-1)[4:7]
    assert torch.equal(projected.view(torch.uint8), expected.view(torch.uint8))


def test_factored_router_keeps_one_by_four_thousand_flat_scratch_below_fragment_limit():
    members = tuple("rank:{}".format(index) for index in range(4))
    group = ProcessGroupIdentity("factored_large_flat_segments", members)
    identity = ParameterIdentity("model.matrix", (1, 4000))
    shards = _flat_shards(identity, group, (1000, 1000, 1000, 1000))
    scratch_bytes = []
    for shard in shards:
        row_numel = sum(segment.length for segment in _factored_axis_segments(shard, axis=0))
        column_numel = sum(segment.length for segment in _factored_axis_segments(shard, axis=1))
        local_dense_numel = _dense_local_numel(shard, unique_replicas=False)
        scratch_bytes.append((row_numel + column_numel + 1 + local_dense_numel) * 4)
    assert scratch_bytes == [8008, 8008, 8008, 8008]
    assert max(scratch_bytes) < 9000
    assert identity.global_shape[1] * 4 > 9000


def test_projection_send_plan_counts_replicated_scalar_amplification():
    parts = 1024
    counts = _canonical_projection_send_numel(
        1,
        target_segments_by_rank=tuple(_full_segment(1) for _coordinate in range(parts)),
        parts=parts,
    )
    assert counts[0] * 4 == 4096
    assert all(value == 0 for value in counts[1:])
    assert counts[0] * 4 > 2048


def test_dense_send_plan_is_closed_form_for_large_nonleading_dtensor_shards():
    group = ProcessGroupIdentity("dense_large_dtensor_send", ("rank:0", "rank:1"))
    identity = ParameterIdentity("model.matrix", (1_000_000, 2))
    shards = _dtensor_shards(identity, group, dimension=1)
    assert _dense_projection_send_numel(
        identity.numel,
        target_shards=shards,
        parts=2,
    ) == (1_000_000, 1_000_000)


def _worker(rank, world_size, init_file, checkpoint_dir, result_queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        members = tuple("rank:{}".format(index) for index in range(world_size))
        group = ProcessGroupIdentity("portable_sharded_segments", members)
        binding = CheckpointProcessGroupBinding(
            group,
            members[rank],
            dist.group.WORLD,
            torch.device("cpu"),
        )

        identity = ParameterIdentity("model.weight", (3, 5))
        dtensor_shards = _dtensor_shards(identity, group, dimension=1)
        source_segments = tuple(
            _dense_segments_for_shard(shard, unique_replicas=True)
            for shard in dtensor_shards
        )
        canonical_segments = _canonical_segments(identity.numel, world_size)
        global_value = torch.arange(identity.numel, dtype=torch.float32).reshape(identity.global_shape)
        region = dtensor_shards[rank].logical_region
        index = tuple(
            slice(offset, offset + length)
            for offset, length in zip(region.offsets, region.lengths)
        )
        local = global_value[index].contiguous()
        canonical = _redistribute_flat_field(
            local,
            source_segments_by_rank=source_segments,
            target_segments_by_rank=canonical_segments,
            global_numel=identity.numel,
            binding=binding,
        )
        canonical_segment = canonical_segments[rank][0]
        expected_canonical = global_value.reshape(-1)[
            canonical_segment.global_offset : canonical_segment.global_offset + canonical_segment.length
        ]
        checkpoint_value = _as_canonical_dtensor(
            canonical,
            global_numel=identity.numel,
            binding=binding,
        )
        dcp.save(
            {"field": checkpoint_value},
            storage_writer=FileSystemWriter(checkpoint_dir),
            process_group=binding.process_group,
        )
        loaded_value = _allocate_canonical_dtensor(
            global_numel=identity.numel,
            dtype=torch.float32,
            binding=binding,
        )
        dcp.load(
            {"field": loaded_value},
            storage_reader=FileSystemReader(checkpoint_dir),
            process_group=binding.process_group,
        )
        loaded_canonical = _canonical_local_tensor(loaded_value)

        flat_shards = _flat_shards(identity, group, (4, 11))
        flat_segments = tuple(
            _dense_segments_for_shard(shard, unique_replicas=False)
            for shard in flat_shards
        )
        projected = _redistribute_flat_field(
            canonical,
            source_segments_by_rank=canonical_segments,
            target_segments_by_rank=flat_segments,
            global_numel=identity.numel,
            binding=binding,
        )
        flat_slice = flat_shards[rank].logical_slice
        expected_projected = global_value.reshape(-1)[
            flat_slice.flat_offset : flat_slice.flat_offset + flat_slice.length
        ]

        empty_identity = ParameterIdentity("model.tiny", (2, 1))
        empty_shards = _dtensor_shards(empty_identity, group, dimension=1)
        empty_sources = tuple(
            _dense_segments_for_shard(shard, unique_replicas=True)
            for shard in empty_shards
        )
        empty_global = torch.tensor([[3.0], [7.0]])
        empty_region = empty_shards[rank].logical_region
        empty_index = tuple(
            slice(offset, offset + length)
            for offset, length in zip(empty_region.offsets, empty_region.lengths)
        )
        empty_local = empty_global[empty_index].contiguous()
        empty_canonical = _redistribute_flat_field(
            empty_local,
            source_segments_by_rank=empty_sources,
            target_segments_by_rank=_canonical_segments(empty_identity.numel, world_size),
            global_numel=empty_identity.numel,
            binding=binding,
        )

        large_identity = ParameterIdentity("model.large", (100_000, 2))
        large_shards = _dtensor_shards(large_identity, group, dimension=1)
        large_region = large_shards[rank].logical_region
        large_index = tuple(
            slice(offset, offset + length)
            for offset, length in zip(
                large_region.offsets,
                large_region.lengths,
            )
        )
        large_global = torch.arange(
            large_identity.numel,
            dtype=torch.float32,
        ).reshape(large_identity.global_shape)
        large_local = large_global[large_index].contiguous()
        large_canonical = _redistribute_dense_to_canonical(
            large_local,
            source_shard=large_shards[rank],
            global_numel=large_identity.numel,
            binding=binding,
        )
        large_round_trip = _redistribute_canonical_to_dense(
            large_canonical,
            target_shards=large_shards,
            global_numel=large_identity.numel,
            binding=binding,
        )

        flat_manifest_shards = _flat_shards(identity, group, (8, 7))
        source_shard = flat_manifest_shards[rank]
        source_start = source_shard.logical_slice.flat_offset
        source_stop = source_start + source_shard.logical_slice.length
        original_parameter = torch.nn.Parameter(global_value.clone())
        local_parameter = torch.nn.Parameter(
            global_value.reshape(-1)[source_start:source_stop].clone()
        )
        optimizer = Gefen(
            [("weight", original_parameter)],
            fused=False,
            factored_v_2d=False,
            force_2d_period_one=True,
            deterministic=True,
        )
        optimizer.post_sharding(
            (ParameterRebinding(original_parameter, local_parameter, source_shard),),
            manifest=ShardingManifest(flat_manifest_shards),
            codebook_process_group=CodebookProcessGroupBinding(
                group,
                members[rank],
                dist.group.WORLD,
                torch.device("cpu"),
            ),
        )
        optimizer._gefen_global_step = 4
        optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256)
        local_momentum = torch.linspace(-0.5, 0.75, identity.numel)[source_start:source_stop]
        optimizer.state[local_parameter].update(
            {
                "automatic_period": 1,
                "step": 3,
                "m_codebook": torch.where(
                    torch.signbit(local_momentum),
                    torch.zeros(local_momentum.numel(), dtype=torch.uint8),
                    torch.full((local_momentum.numel(),), 255, dtype=torch.uint8),
                ).reshape(-1, 1),
                "m_magnitude": local_momentum.abs().reshape(-1, 1),
                "vmean": torch.linspace(0.25, 1.25, identity.numel)[
                    source_start:source_stop
                ].reshape(-1, 1),
                "vmean_step": 2,
            }
        )
        prepared_state = _prepare_sharded_save_state(
            optimizer,
            binding=binding,
            transaction_id="prepare-sharded-save",
            limits=_limits(),
            namespace="optimizer",
        )
        prepared_envelope = prepared_state["optimizer"]
        prepared_field_keys = [
            key for key in prepared_envelope if key != _DCP_SHARDED_METADATA_KEY
        ]
        prepared_local_sizes = [
            _canonical_local_tensor(prepared_envelope[key]).numel()
            for key in prepared_field_keys
        ]
        decoded_metadata = _decode_metadata_tensor(
            prepared_envelope[_DCP_SHARDED_METADATA_KEY],
            limits=_limits(),
        )
        optimizer_checkpoint_dir = "{}-optimizer".format(checkpoint_dir)
        dcp.save(
            prepared_state,
            storage_writer=FileSystemWriter(optimizer_checkpoint_dir),
            process_group=binding.process_group,
        )
        optimizer_reader = FileSystemReader(optimizer_checkpoint_dir)
        optimizer_checkpoint_metadata = optimizer_reader.read_metadata()
        loaded_metadata = _read_sharded_metadata(
            storage_reader=optimizer_reader,
            checkpoint_metadata=optimizer_checkpoint_metadata,
            binding=binding,
            namespace="optimizer",
            limits=_limits(),
            transaction_id="sharded-segment-metadata",
        )
        _validate_sharded_dcp_entries(
            optimizer_checkpoint_metadata,
            loaded_metadata,
            namespace="optimizer",
            binding=binding,
            limits=_limits(),
        )
        loaded_fields = _load_sharded_payloads(
            storage_reader=optimizer_reader,
            checkpoint_metadata=optimizer_checkpoint_metadata,
            metadata=loaded_metadata,
            metadata_tensor=prepared_envelope[_DCP_SHARDED_METADATA_KEY].clone(),
            binding=binding,
            namespace="optimizer",
            limits=_limits(),
            transaction_id="sharded-segment-load",
            context_digest=bytes.fromhex(
                loaded_metadata["completion"]["metadata_digest"]
            ),
        )

        result_queue.put(
            {
                "rank": rank,
                "canonical": torch.equal(canonical.cpu(), expected_canonical),
                "dcp": torch.equal(loaded_canonical.cpu(), expected_canonical),
                "projected": torch.equal(projected.cpu(), expected_projected),
                "empty_source": empty_local.numel() == (2 if rank == 0 else 0),
                "empty_canonical": torch.equal(
                    empty_canonical.cpu(),
                    empty_global.reshape(-1)[rank : rank + 1],
                ),
                "large_shard1": torch.equal(
                    large_round_trip.cpu(),
                    large_local.reshape(-1),
                ),
                "prepared_schema": (
                    len(prepared_field_keys) == 3
                    and prepared_envelope[_DCP_SHARDED_METADATA_KEY].dtype is torch.uint8
                    and prepared_envelope[_DCP_SHARDED_METADATA_KEY].numel() > 0
                    and all(size <= 256 for size in prepared_local_sizes)
                    and len(decoded_metadata["fields"]) == 3
                    and decoded_metadata["parameters"]["model.weight"]["state_variant"]
                    == "initialized_dense"
                    and loaded_metadata["completion"] == decoded_metadata["completion"]
                    and len(loaded_fields) == 3
                ),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="sharded portable-DCP redistribution requires Gloo",
)
def test_dtensor_dimension_shards_redistribute_through_canonical_chunks(tmp_path):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-sharded-segments-")
    os.close(descriptor)
    os.unlink(init_file)
    checkpoint_dir = str(tmp_path / "checkpoint")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, 2, init_file, checkpoint_dir, result_queue),
        )
        for rank in range(2)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=90))
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

    assert len(results) == 2, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes)
    assert all("fatal_error" not in result for result in results), results
    assert all(all(value is True for key, value in result.items() if key != "rank") for result in results)
