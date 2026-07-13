"""CPU coverage for canonical parameter and shard identity descriptors."""

from dataclasses import FrozenInstanceError

import pytest

import gefen
from gefen import (
    IDENTITY_SCHEMA_VERSION,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)


def _group():
    return ProcessGroupIdentity("data_parallel", ("rank:2", "rank:0", "rank:1"))


def _flat_shard(parameter, group, member, offset, length):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        LogicalSlice(offset, length),
        placements=(
            ShardPlacement(
                "data_parallel",
                PlacementKind.FLAT_SHARD,
                coordinate,
                len(group.ordered_members),
            ),
        ),
        process_group=group,
        local_member=member,
    )


def _owner_shard(parameter, group, member, owner):
    coordinate = group.ordered_members.index(member)
    logical_slice = LogicalSlice.full(parameter) if member == owner else LogicalSlice(0, 0)
    return ShardIdentity(
        parameter,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
        logical_slice,
        placements=(
            ShardPlacement(
                "data_parallel",
                PlacementKind.WHOLE_PARAMETER_OWNER,
                coordinate,
                len(group.ordered_members),
            ),
        ),
        process_group=group,
        local_member=member,
        owner=owner,
    )


def test_parameter_and_process_group_identities_are_exact_and_immutable():
    parameter = ParameterIdentity("Encoder.Block.Weight", [4, 8])
    group = ProcessGroupIdentity("pipeline:1/dp", ["worker:b", "worker:a"])

    assert parameter.schema_version == IDENTITY_SCHEMA_VERSION
    assert parameter.fqn == "Encoder.Block.Weight"
    assert parameter.global_shape == (4, 8)
    assert parameter.numel == 32
    assert group.ordered_members == ("worker:b", "worker:a")
    with pytest.raises(FrozenInstanceError):
        parameter.fqn = "changed"
    with pytest.raises(FrozenInstanceError):
        group.semantic_name = "changed"


@pytest.mark.parametrize(
    "fqn", ["", " layer.weight", "layer.weight ", ".layer", "layer.", "layer..weight", "layer\x00weight"]
)
def test_parameter_identity_rejects_noncanonical_fqns(fqn):
    with pytest.raises(ValueError):
        ParameterIdentity(fqn, (4, 4))


@pytest.mark.parametrize("shape", [(4, -1), (4, True), (4, 1.5)])
def test_parameter_identity_rejects_invalid_global_shapes(shape):
    with pytest.raises(ValueError):
        ParameterIdentity("layer.weight", shape)


@pytest.mark.parametrize("shape", ["48", b"\x04\x08", bytearray((4, 8))])
def test_parameter_identity_rejects_string_and_bytes_shape_containers(shape):
    with pytest.raises(TypeError, match="sequence"):
        ParameterIdentity("layer.weight", shape)


def test_process_group_identity_preserves_authoritative_member_order():
    group = _group()
    assert group.ordered_members == ("rank:2", "rank:0", "rank:1")
    with pytest.raises(ValueError, match="unique"):
        ProcessGroupIdentity("dp", ("rank:0", "rank:0"))
    with pytest.raises(ValueError, match="non-empty"):
        ProcessGroupIdentity("dp", ())
    with pytest.raises(TypeError, match="sequence"):
        ProcessGroupIdentity("dp", "ab")


def test_placement_and_logical_slice_validation():
    placement = ShardPlacement("dp", PlacementKind.DIMENSION_SHARD, 1, 2, 0)
    assert placement.parameter_dimension == 0
    with pytest.raises(ValueError, match="within"):
        ShardPlacement("dp", PlacementKind.FLAT_SHARD, 2, 2)
    with pytest.raises(ValueError, match="only"):
        ShardPlacement("dp", PlacementKind.REPLICATE, 0, 1, 0)
    with pytest.raises(ValueError, match="requires"):
        ShardPlacement("dp", PlacementKind.DIMENSION_SHARD, 0, 1)
    with pytest.raises(ValueError):
        LogicalSlice(-1, 1)
    with pytest.raises(ValueError):
        LogicalSlice(0, -1)


def test_replicated_identity_can_be_local_or_process_group_scoped():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    local = ShardIdentity(
        parameter,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(parameter),
    )
    assert local.process_group is None

    group = _group()
    scoped = [
        ShardIdentity(
            parameter,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(parameter),
            placements=(
                ShardPlacement(
                    "data_parallel",
                    PlacementKind.REPLICATE,
                    group.ordered_members.index(member),
                    len(group.ordered_members),
                ),
            ),
            process_group=group,
            local_member=member,
        )
        for member in group.ordered_members
    ]
    manifest = ShardingManifest(tuple(reversed(scoped)))
    assert {item.local_member for item in manifest.shards} == set(group.ordered_members)


def test_contiguous_slice_schema_rejects_dtensor_identity_until_regions_exist():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    group = ProcessGroupIdentity("dp", ("rank:0", "rank:1"))
    with pytest.raises(ValueError, match="logical-region"):
        ShardIdentity(
            parameter,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
            LogicalSlice(0, 8),
            placements=(ShardPlacement("dp", PlacementKind.DIMENSION_SHARD, 0, 2, 0),),
            process_group=group,
            local_member="rank:0",
        )


@pytest.mark.parametrize("version", [True, 1.0, 0, 2])
def test_parameter_identity_schema_version_requires_exact_supported_int(version):
    with pytest.raises(ValueError, match="schema version"):
        ParameterIdentity("layer.weight", (4, 4), schema_version=version)


def test_every_versioned_identity_descriptor_checks_its_schema_version():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    local = ShardIdentity(
        parameter,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(parameter),
    )
    with pytest.raises(ValueError, match="schema version"):
        ProcessGroupIdentity("dp", ("rank:0",), schema_version=True)
    with pytest.raises(ValueError, match="schema version"):
        ShardIdentity(
            parameter,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(parameter),
            schema_version=1.0,
        )
    with pytest.raises(ValueError, match="schema version"):
        ShardingManifest((local,), schema_version=False)


def test_flattened_manifest_normalizes_order_and_requires_exact_coverage():
    parameter = ParameterIdentity("Encoder.Weight", (4, 4))
    group = _group()
    shards = (
        _flat_shard(parameter, group, "rank:1", 16, 0),
        _flat_shard(parameter, group, "rank:0", 8, 8),
        _flat_shard(parameter, group, "rank:2", 0, 8),
    )

    manifest = ShardingManifest(shards)

    assert tuple((item.logical_slice.flat_offset, item.local_member) for item in manifest.shards) == (
        (0, "rank:2"),
        (8, "rank:0"),
        (16, "rank:1"),
    )
    assert manifest.for_parameter("Encoder.Weight") == manifest.shards
    assert manifest.for_parameter("missing") == ()

    with pytest.raises(ValueError, match="gapless"):
        ShardingManifest(
            (
                _flat_shard(parameter, group, "rank:2", 0, 7),
                _flat_shard(parameter, group, "rank:0", 8, 8),
                _flat_shard(parameter, group, "rank:1", 16, 0),
            )
        )
    with pytest.raises(ValueError, match="gapless"):
        ShardingManifest(
            (
                _flat_shard(parameter, group, "rank:2", 0, 9),
                _flat_shard(parameter, group, "rank:0", 8, 8),
                _flat_shard(parameter, group, "rank:1", 16, 0),
            )
        )
    with pytest.raises(ValueError, match="each process-group member"):
        ShardingManifest(shards[:-1])
    inconsistent_axis = ShardIdentity(
        parameter,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        LogicalSlice(8, 8),
        placements=(ShardPlacement("zero", PlacementKind.FLAT_SHARD, 1, 3),),
        process_group=group,
        local_member="rank:0",
    )
    with pytest.raises(ValueError, match="one topology"):
        ShardingManifest((shards[0], inconsistent_axis, shards[2]))
    with pytest.raises(ValueError, match="unique"):
        ShardingManifest((shards[0], shards[0]))


def test_flattened_manifest_empty_shard_is_order_independent_at_middle_boundary():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    group = _group()
    manifest = ShardingManifest(
        (
            _flat_shard(parameter, group, "rank:2", 0, 8),
            _flat_shard(parameter, group, "rank:0", 8, 8),
            _flat_shard(parameter, group, "rank:1", 8, 0),
        )
    )
    assert sum(item.logical_slice.length for item in manifest.shards) == 16
    with pytest.raises(ValueError, match="partition boundary"):
        ShardingManifest(
            (
                _flat_shard(parameter, group, "rank:2", 0, 8),
                _flat_shard(parameter, group, "rank:0", 8, 8),
                _flat_shard(parameter, group, "rank:1", 7, 0),
            )
        )


def test_flattened_shard_requires_explicit_group_and_in_bounds_slice():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    group = _group()
    placement = ShardPlacement("dp", PlacementKind.FLAT_SHARD, 0, 3)
    with pytest.raises(ValueError, match="process group"):
        ShardIdentity(
            parameter,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            LogicalSlice(0, 8),
            placements=(placement,),
        )
    with pytest.raises(ValueError, match="exceeds"):
        _flat_shard(parameter, group, "rank:2", 12, 8)
    with pytest.raises(ValueError, match="coordinates"):
        ShardIdentity(
            parameter,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            LogicalSlice(0, 8),
            placements=(ShardPlacement("dp", PlacementKind.FLAT_SHARD, 1, 3),),
            process_group=group,
            local_member="rank:2",
        )


def test_replicated_identity_rejects_ambiguous_multi_axis_placements():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    group = ProcessGroupIdentity("mesh", ("rank:0",))
    placements = (
        ShardPlacement("tensor", PlacementKind.REPLICATE, 0, 1),
        ShardPlacement("data", PlacementKind.REPLICATE, 0, 1),
    )
    with pytest.raises(ValueError, match="one placement"):
        ShardIdentity(
            parameter,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(parameter),
            placements=placements,
            process_group=group,
            local_member="rank:0",
        )


def test_whole_parameter_owner_manifest_has_one_complete_owner():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    group = _group()
    shards = tuple(_owner_shard(parameter, group, member, "rank:0") for member in group.ordered_members)

    manifest = ShardingManifest(tuple(reversed(shards)))

    owner = next(item for item in manifest.shards if item.local_member == "rank:0")
    assert owner.logical_slice == LogicalSlice.full(parameter)
    assert all(item.logical_slice.length == 0 for item in manifest.shards if item.local_member != "rank:0")
    with pytest.raises(ValueError, match="non-owner"):
        ShardIdentity(
            parameter,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            LogicalSlice.full(parameter),
            placements=shards[0].placements,
            process_group=group,
            local_member="rank:2",
            owner="rank:0",
        )


def test_zero_numel_whole_parameter_still_distinguishes_owner_by_member():
    parameter = ParameterIdentity("empty.weight", (0, 4))
    group = ProcessGroupIdentity("dp", ("rank:0", "rank:1"))
    manifest = ShardingManifest(
        tuple(_owner_shard(parameter, group, member, "rank:0") for member in group.ordered_members)
    )
    assert tuple(item.owner for item in manifest.shards) == ("rank:0", "rank:0")


def test_whole_owner_manifest_rejects_owner_disagreement():
    parameter = ParameterIdentity("layer.weight", (4, 4))
    group = ProcessGroupIdentity("dp", ("rank:0", "rank:1"))
    with pytest.raises(ValueError, match="one owner"):
        ShardingManifest(
            (
                _owner_shard(parameter, group, "rank:0", "rank:0"),
                _owner_shard(parameter, group, "rank:1", "rank:1"),
            )
        )


def test_manifest_rejects_parameter_or_layout_disagreement():
    group = _group()
    first = ParameterIdentity("layer.weight", (4, 4))
    different = ParameterIdentity("layer.weight", (2, 8))
    with pytest.raises(ValueError, match="parameter identity"):
        ShardingManifest(
            (
                _flat_shard(first, group, "rank:2", 0, 8),
                _flat_shard(different, group, "rank:0", 8, 8),
                _flat_shard(first, group, "rank:1", 16, 0),
            )
        )
    flat = tuple(
        _flat_shard(first, group, member, offset, length)
        for member, offset, length in (
            ("rank:2", 0, 8),
            ("rank:0", 8, 8),
            ("rank:1", 16, 0),
        )
    )
    replicated = ShardIdentity(
        first,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(first),
        placements=(
            ShardPlacement("data_parallel", PlacementKind.REPLICATE, 0, 3),
        ),
        process_group=group,
        local_member="rank:2",
    )
    with pytest.raises(ValueError, match="layout or process group"):
        ShardingManifest((replicated, flat[1], flat[2]))

    other_group = ProcessGroupIdentity(
        "other_data_parallel", group.ordered_members
    )
    with pytest.raises(ValueError, match="layout or process group"):
        ShardingManifest(
            (
                flat[0],
                _flat_shard(first, other_group, "rank:0", 8, 8),
                flat[2],
            )
        )


def test_identity_descriptors_copy_mutable_input_sequences_deeply_enough():
    shape = [4, 4]
    members = ["rank:0"]
    placements = [
        ShardPlacement("dp", PlacementKind.REPLICATE, 0, 1),
    ]
    parameter = ParameterIdentity("layer.weight", shape)
    group = ProcessGroupIdentity("dp", members)
    shard = ShardIdentity(
        parameter,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(parameter),
        placements=placements,
        process_group=group,
        local_member="rank:0",
    )
    source_shards = [shard]
    manifest = ShardingManifest(source_shards)

    shape.append(8)
    members.append("rank:1")
    placements.clear()
    source_shards.clear()

    assert parameter.global_shape == (4, 4)
    assert group.ordered_members == ("rank:0",)
    assert shard.placements == (
        ShardPlacement("dp", PlacementKind.REPLICATE, 0, 1),
    )
    assert manifest.shards == (shard,)


def test_manifest_order_is_fqn_then_logical_slice_then_member():
    group = ProcessGroupIdentity("dp", ("rank:1", "rank:0"))
    alpha = ParameterIdentity("Alpha.Weight", (0,))
    zeta = ParameterIdentity("Zeta.Weight", (2,))
    shards = (
        _flat_shard(zeta, group, "rank:0", 2, 0),
        _flat_shard(alpha, group, "rank:0", 0, 0),
        _flat_shard(zeta, group, "rank:1", 0, 2),
        _flat_shard(alpha, group, "rank:1", 0, 0),
    )

    manifest = ShardingManifest(tuple(reversed(shards)))

    assert tuple(
        (
            item.parameter.fqn,
            item.logical_slice.flat_offset,
            item.local_member,
        )
        for item in manifest.shards
    ) == (
        ("Alpha.Weight", 0, "rank:1"),
        ("Alpha.Weight", 0, "rank:0"),
        ("Zeta.Weight", 0, "rank:1"),
        ("Zeta.Weight", 2, "rank:0"),
    )


def test_identity_contracts_are_public_lazy_exports():
    for name in (
        "ParameterIdentity",
        "ProcessGroupIdentity",
        "PlacementKind",
        "ShardPlacement",
        "LogicalSlice",
        "ShardIdentity",
        "ShardingManifest",
    ):
        assert name in gefen.__all__
        assert getattr(gefen, name).__module__ == "gefen.contracts"
    assert "ParameterRebinding" in gefen.__all__
    assert gefen.ParameterRebinding.__module__ == "gefen.rebinding"
