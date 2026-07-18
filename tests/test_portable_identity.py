"""Strict primitive wire coverage for portable optimizer identities."""

import copy
import io

import pytest
import torch

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
from gefen.portable_identity import (
    _normalize_parameter_identity,
    _normalize_process_group_identity,
    _normalize_shard_identity,
    _normalize_sharding_manifest,
    _serialize_parameter_identity,
    _serialize_process_group_identity,
    _serialize_shard_identity,
    _serialize_sharding_manifest,
)


def _group():
    return ProcessGroupIdentity("pipeline:1/data_parallel", ("worker:c", "worker:a", "worker:b"))


def _placement(group, member, kind, *, dimension=None):
    return ShardPlacement(
        "data_parallel",
        kind,
        group.ordered_members.index(member),
        len(group.ordered_members),
        dimension,
    )


def _replicated_shards(parameter, group):
    return tuple(
        ShardIdentity(
            parameter,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(parameter),
            placements=(_placement(group, member, PlacementKind.REPLICATE),),
            process_group=group,
            local_member=member,
        )
        for member in group.ordered_members
    )


def _flat_shards(parameter, group):
    boundaries = ((0, 2), (2, 2), (4, 1))
    return tuple(
        ShardIdentity(
            parameter,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            LogicalSlice(offset, length),
            placements=(_placement(group, member, PlacementKind.FLAT_SHARD),),
            process_group=group,
            local_member=member,
        )
        for member, (offset, length) in zip(group.ordered_members, boundaries)
    )


def _owner_shards(parameter, group, owner):
    return tuple(
        ShardIdentity(
            parameter,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            (LogicalSlice.full(parameter) if member == owner else LogicalSlice(0, 0)),
            placements=(_placement(group, member, PlacementKind.WHOLE_PARAMETER_OWNER),),
            process_group=group,
            local_member=member,
            owner=owner,
        )
        for member in group.ordered_members
    )


def _dtensor_shards(parameter, group, lengths, *, dimension):
    offset = 0
    shards = []
    for member, length in zip(group.ordered_members, lengths):
        offsets = [0] * len(parameter.global_shape)
        region_lengths = list(parameter.global_shape)
        offsets[dimension] = offset
        region_lengths[dimension] = length
        shards.append(
            ShardIdentity(
                parameter,
                ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
                LogicalRegion(offsets, region_lengths),
                placements=(
                    _placement(
                        group,
                        member,
                        PlacementKind.DIMENSION_SHARD,
                        dimension=dimension,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return tuple(shards)


def _dtensor_replicated_shards(parameter, group):
    return tuple(
        ShardIdentity(
            parameter,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
            LogicalRegion.full(parameter),
            placements=(_placement(group, member, PlacementKind.REPLICATE),),
            process_group=group,
            local_member=member,
        )
        for member in group.ordered_members
    )


def _complete_manifest():
    group = _group()
    shards = (
        ShardIdentity(
            ParameterIdentity("local.weight", (2, 3)),
            ParameterLayout.REPLICATED,
            LogicalSlice(0, 6),
        ),
    )
    shards += _replicated_shards(ParameterIdentity("replicated.weight", (2, 3)), group)
    shards += _flat_shards(ParameterIdentity("flat.weight", (5,)), group)
    shards += _owner_shards(ParameterIdentity("owner.weight", (2, 3)), group, "worker:a")
    shards += _dtensor_shards(
        ParameterIdentity("row.weight", (5, 4)),
        group,
        (2, 2, 1),
        dimension=0,
    )
    shards += _dtensor_shards(
        ParameterIdentity("column_with_empty.weight", (2, 2)),
        group,
        (1, 1, 0),
        dimension=1,
    )
    shards += _dtensor_replicated_shards(ParameterIdentity("dtensor_replicated.weight", (2, 3)), group)
    return ShardingManifest(tuple(reversed(shards)))


def _assert_primitive_tree(value):
    assert type(value) in {dict, list, str, int, type(None)}
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_primitive_tree(item)
    elif type(value) is list:
        for item in value:
            _assert_primitive_tree(item)


def test_parameter_and_process_group_codecs_have_exact_canonical_wire_shapes():
    parameter = ParameterIdentity("Encoder.Block.Weight", (0, 4, 7))
    group = _group()

    parameter_wire = _serialize_parameter_identity(parameter)
    group_wire = _serialize_process_group_identity(group)

    assert parameter_wire == {
        "schema_version": 1,
        "fqn": "Encoder.Block.Weight",
        "global_shape": [0, 4, 7],
    }
    assert group_wire == {
        "schema_version": 1,
        "semantic_name": "pipeline:1/data_parallel",
        "ordered_members": ["worker:c", "worker:a", "worker:b"],
    }
    assert _normalize_parameter_identity(parameter_wire) == parameter_wire
    assert _normalize_process_group_identity(group_wire) == group_wire
    _assert_primitive_tree(parameter_wire)
    _assert_primitive_tree(group_wire)


@pytest.mark.parametrize(
    "identity,extent_kind",
    (
        (
            ShardIdentity(
                ParameterIdentity("local.weight", (2, 3)),
                ParameterLayout.REPLICATED,
                LogicalSlice(0, 6),
            ),
            "logical_slice",
        ),
        (_flat_shards(ParameterIdentity("flat.weight", (5,)), _group())[1], "logical_slice"),
        (
            _owner_shards(
                ParameterIdentity("owner.weight", (2, 3)),
                _group(),
                "worker:a",
            )[0],
            "logical_slice",
        ),
        (
            _dtensor_shards(
                ParameterIdentity("row.weight", (5, 4)),
                _group(),
                (2, 2, 1),
                dimension=0,
            )[2],
            "logical_region",
        ),
        (
            _dtensor_shards(
                ParameterIdentity("empty.weight", (2, 2)),
                _group(),
                (1, 1, 0),
                dimension=1,
            )[2],
            "logical_region",
        ),
        (
            _dtensor_replicated_shards(
                ParameterIdentity("dtensor_replicated.weight", (2, 3)),
                _group(),
            )[1],
            "logical_region",
        ),
    ),
)
def test_shard_codec_losslessly_round_trips_every_supported_identity(identity, extent_kind):
    wire = _serialize_shard_identity(identity)

    assert wire["logical_extent"]["kind"] == extent_kind
    assert _normalize_shard_identity(wire) == wire
    _assert_primitive_tree(wire)


def test_shard_wire_explicitly_discriminates_slice_and_region_fields():
    legacy = _serialize_shard_identity(_flat_shards(ParameterIdentity("flat.weight", (5,)), _group())[0])
    dtensor = _serialize_shard_identity(
        _dtensor_shards(
            ParameterIdentity("row.weight", (5, 4)),
            _group(),
            (2, 2, 1),
            dimension=0,
        )[0]
    )

    assert legacy["logical_extent"] == {
        "kind": "logical_slice",
        "flat_offset": 0,
        "length": 2,
    }
    assert dtensor["logical_extent"] == {
        "kind": "logical_region",
        "offsets": [0, 0],
        "lengths": [2, 4],
    }


def test_manifest_codec_is_deterministic_complete_and_weights_only_safe():
    manifest = _complete_manifest()
    wire = _serialize_sharding_manifest(manifest)
    reversed_wire = copy.deepcopy(wire)
    reversed_wire["shards"].reverse()

    assert [record["parameter"]["fqn"] for record in wire["shards"]] == [
        shard.parameter.fqn for shard in manifest.shards
    ]
    assert _normalize_sharding_manifest(wire) == wire
    assert _normalize_sharding_manifest(reversed_wire) == wire
    assert any(
        record["logical_extent"]
        == {
            "kind": "logical_region",
            "offsets": [0, 2],
            "lengths": [2, 0],
        }
        for record in wire["shards"]
    )
    _assert_primitive_tree(wire)

    buffer = io.BytesIO()
    torch.save(wire, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert _normalize_sharding_manifest(loaded) == wire


def test_normalizers_rebuild_fresh_canonical_container_trees():
    source = _serialize_sharding_manifest(_complete_manifest())
    normalized = _normalize_sharding_manifest(source)

    assert normalized == source
    assert normalized is not source
    assert normalized["shards"] is not source["shards"]
    assert normalized["shards"][0] is not source["shards"][0]

    source["shards"][0]["parameter"]["global_shape"].append(99)
    assert normalized != source


@pytest.mark.parametrize(
    "corruption",
    (
        "not_dict",
        "missing_key",
        "extra_key",
        "bool_version",
        "tuple_shape",
        "bool_dimension",
        "invalid_fqn",
    ),
)
def test_parameter_identity_schema_corruption_is_rejected(corruption):
    wire = _serialize_parameter_identity(ParameterIdentity("layer.weight", (2, 3)))
    if corruption == "not_dict":
        wire = []
    elif corruption == "missing_key":
        wire.pop("fqn")
    elif corruption == "extra_key":
        wire["global_rank"] = 0
    elif corruption == "bool_version":
        wire["schema_version"] = True
    elif corruption == "tuple_shape":
        wire["global_shape"] = (2, 3)
    elif corruption == "bool_dimension":
        wire["global_shape"][0] = True
    else:
        wire["fqn"] = ".layer"

    with pytest.raises((TypeError, ValueError)):
        _normalize_parameter_identity(wire)


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_key",
        "extra_handle",
        "bool_version",
        "tuple_members",
        "non_string_member",
        "duplicate_member",
        "invalid_name",
    ),
)
def test_process_group_identity_schema_corruption_is_rejected(corruption):
    wire = _serialize_process_group_identity(_group())
    if corruption == "missing_key":
        wire.pop("semantic_name")
    elif corruption == "extra_handle":
        wire["process_group_handle"] = 123
    elif corruption == "bool_version":
        wire["schema_version"] = True
    elif corruption == "tuple_members":
        wire["ordered_members"] = tuple(wire["ordered_members"])
    elif corruption == "non_string_member":
        wire["ordered_members"][0] = 0
    elif corruption == "duplicate_member":
        wire["ordered_members"][1] = wire["ordered_members"][0]
    else:
        wire["semantic_name"] = " dp"

    with pytest.raises((TypeError, ValueError)):
        _normalize_process_group_identity(wire)


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_key",
        "runtime_rank",
        "bool_version",
        "enum_layout",
        "tuple_placements",
        "bad_process_group_type",
        "non_string_member",
        "non_string_owner",
        "unknown_extent_kind",
        "mixed_slice_region_fields",
        "tuple_region_offsets",
        "bool_region_length",
        "slice_for_dtensor",
        "region_for_flat",
        "placement_missing_key",
        "enum_placement_kind",
        "bool_coordinate",
        "bool_parts",
        "bool_parameter_dimension",
        "coordinate_mismatch",
    ),
)
def test_shard_identity_schema_and_semantic_corruption_is_rejected(corruption):
    identity = _dtensor_shards(
        ParameterIdentity("row.weight", (5, 4)),
        _group(),
        (2, 2, 1),
        dimension=0,
    )[0]
    wire = _serialize_shard_identity(identity)
    if corruption == "missing_key":
        wire.pop("owner")
    elif corruption == "runtime_rank":
        wire["global_rank"] = 7
    elif corruption == "bool_version":
        wire["schema_version"] = True
    elif corruption == "enum_layout":
        wire["layout"] = ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
    elif corruption == "tuple_placements":
        wire["placements"] = tuple(wire["placements"])
    elif corruption == "bad_process_group_type":
        wire["process_group"] = []
    elif corruption == "non_string_member":
        wire["local_member"] = 0
    elif corruption == "non_string_owner":
        wire["owner"] = 0
    elif corruption == "unknown_extent_kind":
        wire["logical_extent"]["kind"] = "flat"
    elif corruption == "mixed_slice_region_fields":
        wire["logical_extent"]["flat_offset"] = 0
    elif corruption == "tuple_region_offsets":
        wire["logical_extent"]["offsets"] = (0, 0)
    elif corruption == "bool_region_length":
        wire["logical_extent"]["lengths"][0] = True
    elif corruption == "slice_for_dtensor":
        wire["logical_extent"] = {
            "kind": "logical_slice",
            "flat_offset": 0,
            "length": 8,
        }
    elif corruption == "region_for_flat":
        wire["layout"] = "flattened_element_shard"
    elif corruption == "placement_missing_key":
        wire["placements"][0].pop("parts")
    elif corruption == "enum_placement_kind":
        wire["placements"][0]["kind"] = PlacementKind.DIMENSION_SHARD
    elif corruption == "bool_coordinate":
        wire["placements"][0]["coordinate"] = True
    elif corruption == "bool_parts":
        wire["placements"][0]["parts"] = True
    elif corruption == "bool_parameter_dimension":
        wire["placements"][0]["parameter_dimension"] = True
    else:
        wire["placements"][0]["coordinate"] = 1

    with pytest.raises((TypeError, ValueError)):
        _normalize_shard_identity(wire)


@pytest.mark.parametrize(
    "corruption",
    (
        "not_dict",
        "missing_key",
        "extra_key",
        "bool_version",
        "tuple_shards",
        "empty_shards",
        "duplicate_shard",
        "incomplete_group",
    ),
)
def test_sharding_manifest_schema_and_completeness_corruption_is_rejected(
    corruption,
):
    wire = _serialize_sharding_manifest(_complete_manifest())
    if corruption == "not_dict":
        wire = []
    elif corruption == "missing_key":
        wire.pop("schema_version")
    elif corruption == "extra_key":
        wire["default_process_group"] = "world"
    elif corruption == "bool_version":
        wire["schema_version"] = True
    elif corruption == "tuple_shards":
        wire["shards"] = tuple(wire["shards"])
    elif corruption == "empty_shards":
        wire["shards"] = []
    elif corruption == "duplicate_shard":
        wire["shards"].append(copy.deepcopy(wire["shards"][0]))
    else:
        fqn = "flat.weight"
        wire["shards"] = [
            shard
            for shard in wire["shards"]
            if not (shard["parameter"]["fqn"] == fqn and shard["local_member"] == "worker:b")
        ]

    with pytest.raises((TypeError, ValueError)):
        _normalize_sharding_manifest(wire)


@pytest.mark.parametrize(
    "serializer,value",
    (
        (_serialize_parameter_identity, object()),
        (_serialize_process_group_identity, object()),
        (_serialize_shard_identity, object()),
        (_serialize_sharding_manifest, object()),
    ),
)
def test_serializers_require_contract_dataclasses(serializer, value):
    with pytest.raises(TypeError):
        serializer(value)
