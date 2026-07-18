"""CPU coverage for pure dense logical-field assembly and projection."""

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
from gefen.portable_fields import (
    _assemble_dense_logical_field,
    _project_dense_logical_field,
)


def _group(members=("worker:c", "worker:a", "worker:b")):
    return ProcessGroupIdentity("data_parallel", members)


def _placement(group, member, kind, *, dimension=None):
    return ShardPlacement(
        "data_parallel",
        kind,
        group.ordered_members.index(member),
        len(group.ordered_members),
        dimension,
    )


def _local_replicated(parameter):
    return (
        ShardIdentity(
            parameter,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(parameter),
        ),
    )


def _grouped_replicated(parameter, group):
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


def _flat_shards(parameter, group, lengths):
    offset = 0
    result = []
    for member, length in zip(group.ordered_members, lengths):
        result.append(
            ShardIdentity(
                parameter,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, length),
                placements=(_placement(group, member, PlacementKind.FLAT_SHARD),),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return tuple(result)


def _owner_shards(parameter, group, owner):
    return tuple(
        ShardIdentity(
            parameter,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            (LogicalSlice.full(parameter) if member == owner else LogicalSlice(0, 0)),
            placements=(
                _placement(
                    group,
                    member,
                    PlacementKind.WHOLE_PARAMETER_OWNER,
                ),
            ),
            process_group=group,
            local_member=member,
            owner=owner,
        )
        for member in group.ordered_members
    )


def _dtensor_shards(parameter, group, lengths, *, dimension):
    offset = 0
    result = []
    for member, length in zip(group.ordered_members, lengths):
        offsets = [0] * len(parameter.global_shape)
        region_lengths = list(parameter.global_shape)
        offsets[dimension] = offset
        region_lengths[dimension] = length
        result.append(
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
    return tuple(result)


def _dtensor_replicated(parameter, group):
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


def _assert_tight_cpu_fp32(value, shape):
    assert type(value) is torch.Tensor
    assert value.device.type == "cpu"
    assert value.dtype == torch.float32
    assert tuple(value.shape) == tuple(shape)
    assert not value.requires_grad
    assert value.is_contiguous()
    assert value.storage_offset() == 0
    assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert bool(torch.isfinite(value).all())


def _sparse_tensor(shape):
    with torch.sparse.check_sparse_tensor_invariants(enable=True):
        return torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]),
            torch.tensor([1.0]),
            size=shape,
        )


def _round_trip(parameter, shards, logical):
    manifest = ShardingManifest(tuple(reversed(shards)))
    fragments = tuple(
        (
            shard,
            _project_dense_logical_field(parameter, logical, shard),
        )
        for shard in reversed(shards)
    )
    assembled = _assemble_dense_logical_field(
        manifest,
        parameter,
        fragments,
    )
    _assert_tight_cpu_fp32(assembled, parameter.global_shape)
    assert torch.equal(assembled, logical)
    return assembled, fragments


def test_local_and_grouped_replicated_fields_round_trip_without_aliasing():
    parameter = ParameterIdentity("layer.weight", (2, 3))
    logical = torch.arange(12, dtype=torch.float32).reshape(2, 6)[:, ::2]
    assert not logical.is_contiguous()

    for shards in (
        _local_replicated(parameter),
        _grouped_replicated(parameter, _group()),
    ):
        assembled, fragments = _round_trip(parameter, shards, logical)
        snapshots = tuple(payload.clone() for _, payload in fragments if payload is not None)
        assembled.add_(1000)
        assert torch.equal(logical, torch.tensor([[0.0, 2.0, 4.0], [6.0, 8.0, 10.0]]))
        assert all(
            torch.equal(payload, snapshot)
            for (_, payload), snapshot in zip(
                ((shard, payload) for shard, payload in fragments if payload is not None),
                snapshots,
            )
        )


def test_flattened_uneven_and_empty_shards_preserve_row_major_order():
    parameter = ParameterIdentity("layer.weight", (2, 3))
    group = _group(("rank:0", "rank:1", "rank:2", "rank:3"))
    shards = _flat_shards(parameter, group, (2, 0, 3, 1))
    logical = torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]])

    assembled, fragments = _round_trip(parameter, shards, logical)

    assert torch.equal(assembled, logical)
    by_member = {shard.local_member: payload for shard, payload in fragments}
    assert torch.equal(by_member["rank:0"], torch.tensor([10.0, 11.0]))
    assert by_member["rank:1"] is None
    assert torch.equal(by_member["rank:2"], torch.tensor([12.0, 20.0, 21.0]))
    assert torch.equal(by_member["rank:3"], torch.tensor([22.0]))


def test_whole_parameter_owner_projects_only_to_owner_and_round_trips():
    parameter = ParameterIdentity("layer.weight", (2, 3))
    group = _group()
    shards = _owner_shards(parameter, group, "worker:a")
    logical = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()

    assembled, fragments = _round_trip(parameter, shards, logical)

    assert torch.equal(assembled, logical.detach())
    by_member = {shard.local_member: payload for shard, payload in fragments}
    assert by_member["worker:c"] is None
    assert by_member["worker:b"] is None
    _assert_tight_cpu_fp32(by_member["worker:a"], parameter.global_shape)


@pytest.mark.parametrize(
    "shape,lengths,dimension",
    (
        ((5, 4), (2, 0, 3), 0),
        ((3, 5), (2, 2, 1), 1),
    ),
)
def test_dtensor_row_and_column_regions_round_trip_uneven_and_empty_shards(
    shape,
    lengths,
    dimension,
):
    parameter = ParameterIdentity("layer.weight", shape)
    group = _group()
    shards = _dtensor_shards(
        parameter,
        group,
        lengths,
        dimension=dimension,
    )
    base = torch.arange(shape[0] * (shape[1] * 2), dtype=torch.float32).reshape(
        shape[0],
        shape[1] * 2,
    )
    logical = base[:, ::2]
    assert not logical.is_contiguous()

    assembled, fragments = _round_trip(parameter, shards, logical)

    assert torch.equal(assembled, logical)
    for shard, payload in fragments:
        region = shard.logical_region
        if region.numel == 0:
            assert payload is None
        else:
            _assert_tight_cpu_fp32(payload, region.lengths)


def test_replicated_dtensor_fields_require_equal_full_replicas():
    parameter = ParameterIdentity("layer.weight", (2, 3))
    shards = _dtensor_replicated(parameter, _group())
    logical = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    assembled, fragments = _round_trip(parameter, shards, logical)

    assert torch.equal(assembled, logical)
    assert all(torch.equal(payload, logical) for _, payload in fragments)


def test_manifest_may_contain_other_parameters_but_fragments_must_not():
    parameter = ParameterIdentity("wanted.weight", (4,))
    shards = _flat_shards(parameter, _group(), (1, 1, 2))
    other_parameter = ParameterIdentity("other.weight", (1,))
    other_shard = _local_replicated(other_parameter)[0]
    manifest = ShardingManifest(shards + (other_shard,))
    logical = torch.arange(4, dtype=torch.float32)
    fragments = tuple((shard, _project_dense_logical_field(parameter, logical, shard)) for shard in shards)

    assert torch.equal(
        _assemble_dense_logical_field(manifest, parameter, fragments),
        logical,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        _assemble_dense_logical_field(
            manifest,
            parameter,
            fragments + ((other_shard, torch.ones(1)),),
        )


def test_fragments_reject_missing_extra_duplicate_and_malformed_identities():
    parameter = ParameterIdentity("layer.weight", (3,))
    shards = _flat_shards(parameter, _group(), (1, 1, 1))
    manifest = ShardingManifest(shards)
    logical = torch.arange(3, dtype=torch.float32)
    fragments = tuple((shard, _project_dense_logical_field(parameter, logical, shard)) for shard in shards)

    with pytest.raises(ValueError, match="exactly cover"):
        _assemble_dense_logical_field(manifest, parameter, fragments[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        _assemble_dense_logical_field(manifest, parameter, fragments + (fragments[0],))
    different = _local_replicated(ParameterIdentity("different.weight", (3,)))[0]
    with pytest.raises(ValueError, match="exactly cover"):
        _assemble_dense_logical_field(
            manifest,
            parameter,
            fragments[:-1] + ((different, torch.arange(3, dtype=torch.float32)),),
        )
    with pytest.raises(TypeError, match="pair"):
        _assemble_dense_logical_field(manifest, parameter, ((shards[0],),))
    with pytest.raises(TypeError, match="ShardIdentity"):
        _assemble_dense_logical_field(manifest, parameter, (("not-a-shard", None),))


def test_requested_parameter_must_match_the_manifest_identity_exactly():
    parameter = ParameterIdentity("layer.weight", (3,))
    shard = _local_replicated(parameter)[0]
    manifest = ShardingManifest((shard,))

    with pytest.raises(ValueError, match="does not contain"):
        _assemble_dense_logical_field(
            manifest,
            ParameterIdentity("other.weight", (3,)),
            ((shard, torch.arange(3, dtype=torch.float32)),),
        )
    with pytest.raises(ValueError, match="does not match"):
        _assemble_dense_logical_field(
            manifest,
            ParameterIdentity("layer.weight", (4,)),
            ((shard, torch.arange(3, dtype=torch.float32)),),
        )


def test_replicas_reject_absent_and_disagreeing_payloads():
    parameter = ParameterIdentity("layer.weight", (2, 2))
    shards = _grouped_replicated(parameter, _group())
    manifest = ShardingManifest(shards)
    logical = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    fragments = [(shard, logical.clone()) for shard in shards]

    fragments[1] = (shards[1], None)
    with pytest.raises(ValueError, match="every shard"):
        _assemble_dense_logical_field(manifest, parameter, fragments)
    fragments[1] = (shards[1], logical.clone())
    fragments[2][1][1, 1] += 1
    with pytest.raises(ValueError, match="disagree"):
        _assemble_dense_logical_field(manifest, parameter, fragments)


def test_replicas_require_bit_exact_signed_zero_payloads():
    parameter = ParameterIdentity("layer.weight", (1,))
    shards = _grouped_replicated(parameter, _group())
    manifest = ShardingManifest(shards)
    fragments = [(shard, torch.tensor([0.0])) for shard in shards]
    fragments[1] = (shards[1], torch.tensor([-0.0]))

    with pytest.raises(ValueError, match="disagree"):
        _assemble_dense_logical_field(manifest, parameter, fragments)


def test_flattened_and_dtensor_empty_payloads_are_explicit_none():
    flat_parameter = ParameterIdentity("flat.weight", (2,))
    flat_group = _group()
    flat_shards = _flat_shards(flat_parameter, flat_group, (1, 0, 1))
    flat_manifest = ShardingManifest(flat_shards)
    flat_fragments = (
        (flat_shards[0], torch.tensor([1.0])),
        (flat_shards[1], torch.empty(0)),
        (flat_shards[2], torch.tensor([2.0])),
    )
    with pytest.raises(ValueError, match="empty flattened"):
        _assemble_dense_logical_field(
            flat_manifest,
            flat_parameter,
            flat_fragments,
        )
    missing_flat = list(flat_fragments)
    missing_flat[0] = (flat_shards[0], None)
    missing_flat[1] = (flat_shards[1], None)
    with pytest.raises(ValueError, match="nonempty flattened"):
        _assemble_dense_logical_field(
            flat_manifest,
            flat_parameter,
            missing_flat,
        )

    dtensor_parameter = ParameterIdentity("dtensor.weight", (2, 2))
    dtensor_shards = _dtensor_shards(
        dtensor_parameter,
        _group(),
        (1, 1, 0),
        dimension=1,
    )
    dtensor_manifest = ShardingManifest(dtensor_shards)
    logical = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    dtensor_fragments = [
        (
            shard,
            _project_dense_logical_field(dtensor_parameter, logical, shard),
        )
        for shard in dtensor_shards
    ]
    dtensor_fragments[2] = (dtensor_shards[2], torch.empty((2, 0)))
    with pytest.raises(ValueError, match="empty DTensor"):
        _assemble_dense_logical_field(
            dtensor_manifest,
            dtensor_parameter,
            dtensor_fragments,
        )
    dtensor_fragments[2] = (dtensor_shards[2], None)
    dtensor_fragments[0] = (dtensor_shards[0], None)
    with pytest.raises(ValueError, match="nonempty DTensor"):
        _assemble_dense_logical_field(
            dtensor_manifest,
            dtensor_parameter,
            dtensor_fragments,
        )


def test_whole_parameter_payload_is_present_exactly_on_the_owner():
    parameter = ParameterIdentity("layer.weight", (2, 2))
    shards = _owner_shards(parameter, _group(), "worker:a")
    manifest = ShardingManifest(shards)
    logical = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    fragments = [
        (
            shard,
            logical.clone() if shard.local_member == shard.owner else None,
        )
        for shard in shards
    ]
    owner_index = next(index for index, shard in enumerate(shards) if shard.local_member == shard.owner)

    missing = list(fragments)
    missing[owner_index] = (shards[owner_index], None)
    with pytest.raises(ValueError, match="owner requires"):
        _assemble_dense_logical_field(manifest, parameter, missing)
    nonowner_index = next(index for index, shard in enumerate(shards) if shard.local_member != shard.owner)
    extra = list(fragments)
    extra[nonowner_index] = (shards[nonowner_index], torch.empty(0))
    with pytest.raises(ValueError, match="non-owners"):
        _assemble_dense_logical_field(manifest, parameter, extra)


@pytest.mark.parametrize(
    "bad_value,error",
    (
        (torch.ones((2, 3), dtype=torch.float64), TypeError),
        (torch.ones(6, dtype=torch.float32), ValueError),
        (torch.tensor([[1.0, 2.0, 3.0], [4.0, float("inf"), 6.0]]), ValueError),
        (torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float32)), TypeError),
        (_sparse_tensor((2, 3)), TypeError),
        (torch.empty((2, 3), device="meta"), TypeError),
    ),
)
def test_assembly_rejects_wrong_dtype_shape_nonfinite_subclass_and_sparse(
    bad_value,
    error,
):
    parameter = ParameterIdentity("layer.weight", (2, 3))
    shard = _local_replicated(parameter)[0]
    with pytest.raises(error):
        _assemble_dense_logical_field(
            ShardingManifest((shard,)),
            parameter,
            ((shard, bad_value),),
        )


@pytest.mark.parametrize(
    "bad_value,error",
    (
        (torch.ones((2, 3), dtype=torch.float64), TypeError),
        (torch.ones(6, dtype=torch.float32), ValueError),
        (torch.tensor([[1.0, 2.0, 3.0], [4.0, float("nan"), 6.0]]), ValueError),
        (torch.nn.Parameter(torch.ones((2, 3), dtype=torch.float32)), TypeError),
        (_sparse_tensor((2, 3)), TypeError),
    ),
)
def test_projection_rejects_wrong_dtype_shape_nonfinite_subclass_and_sparse(
    bad_value,
    error,
):
    parameter = ParameterIdentity("layer.weight", (2, 3))
    shard = _local_replicated(parameter)[0]
    with pytest.raises(error):
        _project_dense_logical_field(parameter, bad_value, shard)


def test_projection_requires_an_exact_target_parameter_identity():
    parameter = ParameterIdentity("layer.weight", (2, 3))
    logical = torch.ones((2, 3), dtype=torch.float32)
    other_shard = _local_replicated(ParameterIdentity("other.weight", (2, 3)))[0]

    with pytest.raises(ValueError, match="does not match"):
        _project_dense_logical_field(parameter, logical, other_shard)
    with pytest.raises(TypeError, match="ShardIdentity"):
        _project_dense_logical_field(parameter, logical, "not-a-shard")
    with pytest.raises(TypeError, match="ParameterIdentity"):
        _project_dense_logical_field("layer.weight", logical, other_shard)


def test_api_requires_valid_manifest_parameter_and_fragment_containers():
    parameter = ParameterIdentity("layer.weight", (1,))
    shard = _local_replicated(parameter)[0]
    manifest = ShardingManifest((shard,))

    with pytest.raises(TypeError, match="ShardingManifest"):
        _assemble_dense_logical_field(
            "not-a-manifest",
            parameter,
            ((shard, torch.ones(1)),),
        )
    with pytest.raises(TypeError, match="ParameterIdentity"):
        _assemble_dense_logical_field(
            manifest,
            "layer.weight",
            ((shard, torch.ones(1)),),
        )
    with pytest.raises(TypeError, match="sequence"):
        _assemble_dense_logical_field(manifest, parameter, None)
    with pytest.raises(TypeError, match="sequence"):
        _assemble_dense_logical_field(manifest, parameter, "fragment")
