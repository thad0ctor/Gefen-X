"""Strict resource and DCP-metadata validation for sharded portable state."""

import copy
from dataclasses import replace

import pytest
import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

from gefen import (
    CheckpointProcessGroupBinding,
    CodebookProcessGroupBinding,
    Gefen,
    GefenMuon,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.portable_dcp import _flat_key
from gefen.portable_dcp_sharded import (
    _DCP_SHARDED_METADATA_KEY,
    _decode_metadata_tensor,
    _field_key,
    _normalize_sharded_metadata,
    _prepare_sharded_save_state,
    _source_chunk_assignments,
    _tensor_sha256,
    _validate_field_byte_limits,
    _validate_integrity_assignment_limits,
    _validate_sharded_dcp_entries,
    _validate_source_storage_chunk_limit,
)
from gefen.portable_state import PortableStateLimits
from gefen.portable_schema import portable_state_digest


def _binding():
    identity = ProcessGroupIdentity("sharded_validation", ("rank:0",))
    return CheckpointProcessGroupBinding(
        identity,
        "rank:0",
        None,
        torch.device("cpu"),
    )


def _limits(*, fragment=4096, collective=4096, containers=16):
    return PortableStateLimits(
        max_fragment_tensor_bytes=fragment,
        max_collective_tensor_bytes=collective,
        max_collective_metadata_bytes=4096,
        max_metadata_bytes=4096,
        max_container_items=containers,
        max_tree_nodes=64,
        max_tensors=16,
    )


def _semantic_limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=4096,
        max_collective_tensor_bytes=4096,
        max_collective_metadata_bytes=16 << 10,
        max_metadata_bytes=16 << 10,
        max_container_items=1024,
        max_tree_nodes=4096,
        max_tensors=16,
    )


def _field(index, numel):
    return {
        "index": index,
        "key": _field_key(index),
        "fqn": "model.parameter{}".format(index),
        "name": "momentum",
        "shape": [numel],
        "numel": numel,
        "dtype": "float32",
        "kind": "dense",
        "nonnegative": False,
        "source_chunk_sha256": ["00" * 32],
    }


def _entry(dtype, numel, chunks):
    return TensorStorageMetadata(
        properties=TensorProperties(dtype=dtype),
        size=torch.Size((numel,)),
        chunks=chunks,
    )


def _checkpoint(fields, field_chunks):
    namespace = "optimizer"
    metadata_key = _flat_key(namespace, _DCP_SHARDED_METADATA_KEY)
    entries = {
        metadata_key: _entry(
            torch.uint8,
            1,
            [
                ChunkStorageMetadata(
                    offsets=torch.Size((0,)),
                    sizes=torch.Size((1,)),
                )
            ],
        )
    }
    for field, chunks in zip(fields, field_chunks):
        entries[_flat_key(namespace, field["key"])] = _entry(
            torch.float32,
            field["numel"],
            chunks,
        )
    return Metadata(
        state_dict_metadata=entries,
        planner_data={key: tuple(key.split(".", 1)) for key in entries},
    )


@pytest.fixture(scope="module")
def _initialized_v2_metadata():
    group = ProcessGroupIdentity("sharded_validation", ("rank:0",))
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = Gefen(
        [("weight", parameter)],
        fused=False,
        factored_v_2d=False,
        period_one_substrings=("weight",),
        deterministic=True,
    )
    identity = ParameterIdentity("weight", (2,))
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
        local_member="rank:0",
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=CodebookProcessGroupBinding(
            group,
            "rank:0",
            None,
            torch.device("cpu"),
        ),
    )
    optimizer._gefen_global_step = 4
    optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 4,
            "m_codebook": torch.tensor([[0], [255]], dtype=torch.uint8),
            "m_magnitude": torch.tensor([[1.0], [2.0]]),
            "vmean": torch.tensor([[3.0], [4.0]]),
            "vmean_step": 3,
        }
    )
    limits = _semantic_limits()
    state = _prepare_sharded_save_state(
        optimizer,
        binding=CheckpointProcessGroupBinding(
            group,
            "rank:0",
            None,
            torch.device("cpu"),
        ),
        transaction_id="sharded-validation-v2",
        limits=limits,
        namespace="optimizer",
    )
    return _decode_metadata_tensor(
        state["optimizer"][_DCP_SHARDED_METADATA_KEY],
        limits=limits,
    )


def _different_digest(value):
    replacement = "0" if value[0] != "0" else "1"
    return replacement + value[1:]


def _refresh_metadata_digest(metadata):
    metadata["completion"]["metadata_digest"] = portable_state_digest(
        {key: value for key, value in metadata.items() if key != "completion"}
    )


@pytest.fixture(scope="module")
def _factored_v2_metadata():
    group = ProcessGroupIdentity("sharded_factored_validation", ("rank:0",))
    parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
    optimizer = Gefen(
        [("matrix", parameter)],
        fused=False,
        factored_v_2d=True,
        force_2d_period_one=True,
        deterministic=True,
    )
    identity = ParameterIdentity("matrix", (2, 3))
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", PlacementKind.REPLICATE, 0, 1),),
        process_group=group,
        local_member="rank:0",
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=CodebookProcessGroupBinding(group, "rank:0", None, torch.device("cpu")),
    )
    optimizer._gefen_global_step = 4
    optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 4,
            "m_codebook": torch.tensor([[0], [255], [0], [255], [0], [255]], dtype=torch.uint8),
            "m_magnitude": torch.arange(1, 7, dtype=torch.float32).reshape(-1, 1),
            "v_row": torch.tensor([1.0, 3.0], dtype=torch.float32),
            "v_col": torch.tensor([2.0, 4.0, 8.0], dtype=torch.float32),
            "factored_step": 3,
        }
    )
    limits = _semantic_limits()
    state = _prepare_sharded_save_state(
        optimizer,
        binding=CheckpointProcessGroupBinding(group, "rank:0", None, torch.device("cpu")),
        transaction_id="sharded-factored-validation-v2",
        limits=limits,
        namespace="optimizer",
    )
    return _decode_metadata_tensor(state["optimizer"][_DCP_SHARDED_METADATA_KEY], limits=limits)


@pytest.fixture(scope="module")
def _muon_pristine_v2_metadata():
    group = ProcessGroupIdentity("sharded_muon_validation", ("rank:0",))
    parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
    optimizer = GefenMuon(
        [("matrix", parameter)],
        fused=False,
        sharded_mode="exact",
        deterministic=True,
    )
    identity = ParameterIdentity("matrix", (2, 3))
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", PlacementKind.REPLICATE, 0, 1),),
        process_group=group,
        local_member="rank:0",
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=CodebookProcessGroupBinding(group, "rank:0", None, torch.device("cpu")),
    )
    limits = _semantic_limits()
    state = _prepare_sharded_save_state(
        optimizer,
        binding=CheckpointProcessGroupBinding(group, "rank:0", None, torch.device("cpu")),
        transaction_id="sharded-muon-validation-v2",
        limits=limits,
        namespace="optimizer",
    )
    return _decode_metadata_tensor(state["optimizer"][_DCP_SHARDED_METADATA_KEY], limits=limits)


def _mutate_v2_metadata(metadata, case):
    parameter = metadata["parameters"]["weight"]
    if case == "unknown-key":
        metadata["unknown"] = None
    elif case == "negative-global-step":
        metadata["common"]["gefen_global_step"] = -1
    elif case == "step-exceeds-global":
        parameter["state"]["step"] = metadata["common"]["gefen_global_step"] + 1
    elif case == "unsupported-variant":
        parameter["state_variant"] = "initialized_unknown"
    elif case == "duplicate-field-reference":
        parameter["state"]["second_moment_field"] = parameter["state"]["momentum_field"]
    elif case == "unreferenced-field":
        field = copy.deepcopy(metadata["fields"][1])
        field.update(
            {
                "index": len(metadata["fields"]),
                "key": _field_key(len(metadata["fields"])),
                "name": "unused",
            }
        )
        metadata["fields"].append(field)
    elif case == "incompatible-field-reference":
        parameter["state"]["momentum_field"] = parameter["state"]["second_moment_field"]
    elif case == "manifest-process-group-mismatch":
        metadata["source_manifest"]["shards"][0]["process_group"]["semantic_name"] = "different_group"
    elif case == "short-chunk-digest":
        metadata["fields"][1]["source_chunk_sha256"][0] = "00" * 31
    elif case == "nonhex-chunk-digest":
        metadata["fields"][1]["source_chunk_sha256"][0] = "gg" * 32
    elif case == "metadata-digest-mismatch":
        completion = metadata["completion"]
        completion["metadata_digest"] = _different_digest(completion["metadata_digest"])
    elif case == "payload-digest-mismatch":
        completion = metadata["completion"]
        completion["payload_digest"] = _different_digest(completion["payload_digest"])
    else:
        raise AssertionError("unknown mutation case")


def test_v2_semantic_metadata_fixture_is_valid(_initialized_v2_metadata):
    metadata = copy.deepcopy(_initialized_v2_metadata)
    assert _normalize_sharded_metadata(metadata, limits=_semantic_limits()) == metadata


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("unknown-key", "invalid schema"),
        ("negative-global-step", "bounded exact counter"),
        ("step-exceeds-global", "counters are invalid"),
        ("unsupported-variant", "state_variant is unsupported"),
        ("duplicate-field-reference", "referenced exactly once"),
        ("unreferenced-field", "unreferenced fields"),
        ("incompatible-field-reference", "incompatible field"),
        ("manifest-process-group-mismatch", "different process group"),
        ("short-chunk-digest", "chunk digest is invalid"),
        ("nonhex-chunk-digest", "chunk digest is invalid"),
        ("metadata-digest-mismatch", "metadata digest mismatch"),
        ("payload-digest-mismatch", "payload digest mismatch"),
    ],
)
def test_v2_semantic_metadata_rejects_malformed_envelopes(
    _initialized_v2_metadata,
    case,
    message,
):
    metadata = copy.deepcopy(_initialized_v2_metadata)
    _mutate_v2_metadata(metadata, case)
    with pytest.raises(ValueError, match=message):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing-reference", "invalid schema"),
        ("duplicate-reference", "referenced exactly once"),
        ("wrong-name", "incompatible field"),
        ("wrong-shape", "incompatible field"),
    ],
)
def test_v2_factored_denominator_reference_is_strict(
    _factored_v2_metadata,
    mutation,
    message,
):
    metadata = copy.deepcopy(_factored_v2_metadata)
    state = metadata["parameters"]["matrix"]["state"]
    denominator_index = state["v_row_denominator_field"]
    if mutation == "missing-reference":
        state.pop("v_row_denominator_field")
    elif mutation == "duplicate-reference":
        state["v_row_denominator_field"] = state["v_row_field"]
    elif mutation == "wrong-name":
        metadata["fields"][denominator_index]["name"] = "other_denominator"
    elif mutation == "wrong-shape":
        metadata["fields"][denominator_index]["shape"] = [2]
        metadata["fields"][denominator_index]["numel"] = 2
    else:
        raise AssertionError("unknown denominator mutation")
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match=message):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_v2_factored_metadata_emits_digest_protected_row_denominator(_factored_v2_metadata):
    metadata = copy.deepcopy(_factored_v2_metadata)
    state = metadata["parameters"]["matrix"]["state"]
    field = metadata["fields"][state["v_row_denominator_field"]]
    assert field["name"] == "v_row_denominator"
    assert field["shape"] == [1]
    assert field["kind"] == "special"
    assert field["nonnegative"] is True
    assert _normalize_sharded_metadata(metadata, limits=_semantic_limits()) == metadata


def test_v2_plain_source_rejects_whole_owner_layout(_initialized_v2_metadata):
    metadata = copy.deepcopy(_initialized_v2_metadata)
    shard = metadata["source_manifest"]["shards"][0]
    shard["layout"] = "whole_parameter_owner"
    shard["placements"][0]["kind"] = "whole_parameter_owner"
    shard["owner"] = "rank:0"
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match="source layout is unsupported"):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_v2_plain_factored_source_requires_replicated_layout(_factored_v2_metadata):
    metadata = copy.deepcopy(_factored_v2_metadata)
    shard = metadata["source_manifest"]["shards"][0]
    shard["layout"] = "flattened_element_shard"
    shard["placements"][0]["kind"] = "flat_shard"
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match="factored logical matrices|factored sharded portable"):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_v2_plain_factored_source_policy_must_match_catalog(_factored_v2_metadata):
    metadata = copy.deepcopy(_factored_v2_metadata)
    metadata["policy"]["factored_v_2d"] = False
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match="second_moment_policy conflicts"):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_v2_muon_source_rejects_dtensor_even_when_pristine(_muon_pristine_v2_metadata):
    metadata = copy.deepcopy(_muon_pristine_v2_metadata)
    shard = metadata["source_manifest"]["shards"][0]
    shard["layout"] = "dtensor_1d_default_world"
    shard["logical_extent"] = {
        "kind": "logical_region",
        "offsets": [0, 0],
        "lengths": [2, 3],
    }
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match="source layout is unsupported"):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_v2_muon_whole_owner_source_requires_distributed_mode(_muon_pristine_v2_metadata):
    metadata = copy.deepcopy(_muon_pristine_v2_metadata)
    shard = metadata["source_manifest"]["shards"][0]
    shard["layout"] = "whole_parameter_owner"
    shard["logical_extent"] = {
        "kind": "logical_slice",
        "flat_offset": 0,
        "length": 6,
    }
    shard["placements"][0]["kind"] = "whole_parameter_owner"
    shard["owner"] = "rank:0"
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match="requires sharded_mode='distributed'"):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_v2_muon_source_parameters_must_be_logical_matrices(_muon_pristine_v2_metadata):
    metadata = copy.deepcopy(_muon_pristine_v2_metadata)
    metadata["parameters"]["matrix"]["identity"]["global_shape"] = [6]
    shard = metadata["source_manifest"]["shards"][0]
    shard["parameter"]["global_shape"] = [6]
    shard["logical_extent"]["length"] = 6
    _refresh_metadata_digest(metadata)
    with pytest.raises(ValueError, match="logical matrices"):
        _normalize_sharded_metadata(metadata, limits=_semantic_limits())


def test_sharded_save_rejects_int64_index_route_above_fragment_limit():
    group = ProcessGroupIdentity("sharded_save_route_validation", ("rank:0",))
    parameter = torch.nn.Parameter(torch.ones((20, 20), dtype=torch.float32))
    optimizer = GefenMuon(
        [("matrix", parameter)],
        fused=False,
        sharded_mode="exact",
        deterministic=True,
    )
    identity = ParameterIdentity("matrix", (20, 20))
    shard = ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", PlacementKind.REPLICATE, 0, 1),),
        process_group=group,
        local_member="rank:0",
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=CodebookProcessGroupBinding(group, "rank:0", None, torch.device("cpu")),
    )
    optimizer._gefen_global_step = 2
    optimizer._gefen_codebook = torch.linspace(-1.0, 1.0, 256)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 2,
            "m_codebook": torch.zeros((400, 1), dtype=torch.uint8),
            "m_magnitude": torch.ones((400, 1), dtype=torch.float32),
        }
    )
    limits = replace(_semantic_limits(), max_fragment_tensor_bytes=2800)
    with pytest.raises(RuntimeError, match="save routing scratch"):
        _prepare_sharded_save_state(
            optimizer,
            binding=CheckpointProcessGroupBinding(group, "rank:0", None, torch.device("cpu")),
            transaction_id="sharded-save-route-limit",
            limits=limits,
            namespace="optimizer",
        )


def test_field_byte_limits_are_aggregate_not_per_tensor():
    fields = [_field(0, 200), _field(1, 200)]
    with pytest.raises(ValueError, match="max_collective_tensor_bytes"):
        _validate_field_byte_limits(
            fields,
            binding=_binding(),
            limits=_limits(fragment=1024, collective=1024),
        )


def test_dcp_chunk_limit_is_aggregate_across_fields():
    fields = [_field(0, 4), _field(1, 4)]
    split = [
        ChunkStorageMetadata(
            offsets=torch.Size((0,)),
            sizes=torch.Size((2,)),
        ),
        ChunkStorageMetadata(
            offsets=torch.Size((2,)),
            sizes=torch.Size((2,)),
        ),
    ]
    checkpoint = _checkpoint(fields, [split, split])
    with pytest.raises(ValueError, match="aggregate limits"):
        _validate_sharded_dcp_entries(
            checkpoint,
            {"fields": fields},
            namespace="optimizer",
            binding=_binding(),
            limits=_limits(containers=3),
        )


def test_dcp_chunk_limit_includes_mandatory_metadata_chunk():
    fields = [_field(0, 4), _field(1, 4)]
    complete = [
        ChunkStorageMetadata(
            offsets=torch.Size((0,)),
            sizes=torch.Size((4,)),
        )
    ]
    checkpoint = _checkpoint(fields, [complete, complete])
    with pytest.raises(ValueError, match="aggregate limits"):
        _validate_sharded_dcp_entries(
            checkpoint,
            {"fields": fields},
            namespace="optimizer",
            binding=_binding(),
            limits=_limits(containers=2),
        )


def test_dcp_chunk_geometry_accepts_trailing_empty_shards():
    fields = [_field(0, 1)]
    chunks = [
        ChunkStorageMetadata(
            offsets=torch.Size((0,)),
            sizes=torch.Size((1,)),
        ),
        ChunkStorageMetadata(
            offsets=torch.Size((1,)),
            sizes=torch.Size((0,)),
        ),
    ]
    checkpoint = _checkpoint(fields, [chunks])
    _validate_sharded_dcp_entries(
        checkpoint,
        {"fields": fields},
        namespace="optimizer",
        binding=_binding(),
        limits=_limits(),
    )


@pytest.mark.parametrize(
    ("geometry", "message"),
    [
        (((0, -1),), "exceeds its tensor"),
        (((-1, 0), (0, 4)), "exceeds its tensor"),
        (((0, 4), (5, 0)), "exceeds its tensor"),
        (((0, 2), (3, 1)), "cover each field exactly once"),
        (((0, 3), (2, 2)), "cover each field exactly once"),
    ],
)
def test_dcp_chunk_geometry_rejects_invalid_boundaries(
    geometry,
    message,
):
    fields = [_field(0, 4)]
    chunks = [
        ChunkStorageMetadata(
            offsets=torch.Size((offset,)),
            sizes=torch.Size((length,)),
        )
        for offset, length in geometry
    ]
    checkpoint = _checkpoint(fields, [chunks])
    with pytest.raises(ValueError, match=message):
        _validate_sharded_dcp_entries(
            checkpoint,
            {"fields": fields},
            namespace="optimizer",
            binding=_binding(),
            limits=_limits(),
        )


def test_dcp_chunk_geometry_requires_exact_torch_size_and_ints():
    fields = [_field(0, 4)]
    inexact = [
        ChunkStorageMetadata(
            offsets=[0.0],
            sizes=[4.0],
        )
    ]
    checkpoint = _checkpoint(fields, [inexact])
    with pytest.raises(ValueError, match="chunk is invalid"):
        _validate_sharded_dcp_entries(
            checkpoint,
            {"fields": fields},
            namespace="optimizer",
            binding=_binding(),
            limits=_limits(),
        )


def test_integrity_reassembly_rejects_packed_assignment_above_fragment_limit():
    field = _field(0, 200)
    field["source_chunk_sha256"] = ["00" * 32, "11" * 32, "22" * 32]
    with pytest.raises(ValueError, match="integrity scratch"):
        _validate_integrity_assignment_limits(
            field,
            target_parts=2,
            limits=_limits(fragment=400, collective=4096),
        )


def test_integrity_assignment_accounts_for_six_element_packed_reassembly():
    assignments = _source_chunk_assignments(10, 3, 2)
    packed_numel = [
        max(
            (segment.local_offset + segment.length for _source, segment in items),
            default=0,
        )
        for items in assignments
    ]
    assert packed_numel == [6, 4]


def test_dcp_source_storage_chunk_must_fit_current_fragment_limit():
    with pytest.raises(ValueError, match="source storage chunk"):
        _validate_source_storage_chunk_limit(
            8,
            limits=_limits(fragment=16, collective=4096),
        )


def test_integrity_hash_accepts_scalar_and_empty_tensors():
    assert len(_tensor_sha256(torch.tensor(3.5))) == 32
    assert len(_tensor_sha256(torch.empty(0))) == 32
