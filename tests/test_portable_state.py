"""Warning-strict CPU coverage for portable Gefen optimizer-state semantics."""

import copy
import dataclasses
import math

import pytest
import torch

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
from gefen.portable import _decode_quantized_momentum
from gefen.portable_fields import _project_dense_logical_field
from gefen.portable_identity import (
    _serialize_parameter_identity,
    _serialize_shard_identity,
)
from gefen.portable_schema import build_portable_state_document
from gefen.portable_state import (
    PortableStateLimits,
    _SECOND_MOMENT_PROJECTION_EXACT,
    _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK,
    _assemble_portable_state_fragments,
    _build_portable_state_fragment,
    _normalize_gefen_portable_state_document,
    _normalize_portable_state_fragment,
    _project_portable_parameter_state,
)


def _limits(**changes):
    limits = PortableStateLimits(
        max_fragment_tensor_bytes=4 << 20,
        max_collective_tensor_bytes=16 << 20,
        max_collective_metadata_bytes=64 << 20,
    )
    return dataclasses.replace(limits, **changes)


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _policy(*, factored=False):
    return {
        "schema_version": 1,
        "factored_v_2d": factored,
        "force_1d_period_one": False,
        "force_2d_period_one": False,
        "period_one_substrings": [],
        "stochastic_round": False,
        "codebook_refresh_every": 0,
        "momentum_projection": "dense_fp32_target_period_one_v1",
        "second_moment_projection": (
            _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK if factored else _SECOND_MOMENT_PROJECTION_EXACT
        ),
    }


def _plain_options(*, second="block"):
    return {
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "second_moment_policy": second,
    }


def _muon_options(*, normuon=False, mode="distributed"):
    return {
        "lr": 0.01,
        "weight_decay": 0.1,
        "momentum": 0.95,
        "nesterov": True,
        "ns_schedule": [[3.4445, -4.775, 2.0315]],
        "ns_eps": 1e-7,
        "adjust_lr_fn": "match_rms_adamw",
        "sharded_mode": mode,
        "fp8_ns": False,
        "fp8_ns_compile": True,
        "batched_ns": False,
        "batched_ns_workspace_bytes": 1 << 20,
        "normuon": normuon,
        "normuon_beta2": 0.95,
        "normuon_eps": 1e-8,
        "cautious": False,
    }


def _group(members=("rank:0", "rank:1")):
    return ProcessGroupIdentity("checkpoint", members)


def _placement(group, member, kind):
    return ShardPlacement(
        "checkpoint",
        kind,
        group.ordered_members.index(member),
        len(group.ordered_members),
    )


def _replicas(parameter, group):
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


def _flat(parameter, group, lengths):
    offset = 0
    shards = []
    for member, length in zip(group.ordered_members, lengths):
        shards.append(
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
    return tuple(shards)


def _owners(parameter, group, owner):
    return tuple(
        ShardIdentity(
            parameter,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            LogicalSlice.full(parameter) if member == owner else LogicalSlice(0, 0),
            placements=(_placement(group, member, PlacementKind.WHOLE_PARAMETER_OWNER),),
            process_group=group,
            local_member=member,
            owner=owner,
        )
        for member in group.ordered_members
    )


def _role(shard):
    if shard.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
        return "whole_owner" if shard.local_member == shard.owner else "whole_nonowner"
    if shard.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD and shard.logical_slice.length == 0:
        return "empty_flat"
    if shard.parameter.numel == 0:
        return "empty_replicated"
    return "live"


def _fragments(
    implementation,
    parameter,
    shards,
    *,
    options,
    policy,
    variant,
    momentum=None,
    second=None,
    v_row=None,
    v_col=None,
    normuon_v=None,
):
    manifest = ShardingManifest(shards)
    common = {
        "gefen_global_step": 4 if variant != "pristine" else 0,
        "gefen_codebook": _codebook() if variant != "pristine" else None,
        "gefen_deterministic": True,
    }
    catalog = {
        parameter.fqn: {
            "identity": _serialize_parameter_identity(parameter),
            "algorithm_options": options,
        }
    }
    result = []
    for shard in shards:
        payload = _role(shard) in {"live", "whole_owner"} and parameter.numel > 0
        local_variant = variant if payload else "pristine"
        state = {}
        source_second = None
        period = None
        if local_variant == "period_selected":
            period = 1
        elif local_variant.startswith("initialized_"):
            period = 1
            state = {
                "step": 4,
                "momentum": _project_dense_logical_field(parameter, momentum, shard),
            }
            if implementation == "gefen.Gefen" and local_variant == "initialized_dense":
                source_second = "block"
                state.update(
                    {
                        "second_moment": _project_dense_logical_field(parameter, second, shard),
                        "second_moment_step": 3,
                    }
                )
            elif local_variant == "initialized_factored":
                source_second = "factored"
                state.update({"v_row": v_row.clone(), "v_col": v_col.clone(), "factored_step": 3})
            elif local_variant == "initialized_dense_normuon":
                state.update({"normuon_v": normuon_v.clone(), "normuon_step": 2})
        slot = {
            "group_index": 0,
            "original_slot_index": 0,
            "compatibility_name": "weight",
            "shard": _serialize_shard_identity(shard),
            "algorithm_options": options,
            "role": _role(shard),
            "source_period": period,
            "source_second_moment": source_second,
            "state_variant": local_variant,
            "state": state,
        }
        result.append(
            _build_portable_state_fragment(
                implementation=implementation,
                member=shard.local_member,
                policy=policy,
                common=common,
                manifest=manifest,
                catalog=catalog,
                logical_slots=[slot],
                limits=_limits(),
            )
        )
    return result


@pytest.mark.parametrize("variant", ["pristine", "period_selected"])
def test_plain_pristine_and_period_selected_round_trip(variant):
    parameter = ParameterIdentity("layer.weight", (2, 3))
    group = _group()
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        _replicas(parameter, group),
        options=_plain_options(),
        policy=_policy(),
        variant=variant,
    )
    document = _assemble_portable_state_fragments(fragments, process_group_identity=group, limits=_limits())
    record = document["parameters"][parameter.fqn]
    assert record["state_variant"] == variant
    assert record["state"] == {}
    assert record["projection_hints"]["source_periods"] == ([] if variant == "pristine" else [1])
    normalized = _normalize_gefen_portable_state_document(document, limits=_limits())
    assert normalized["completion"] == document["completion"]


@pytest.mark.parametrize("layout", ["replicated", "flat"])
def test_plain_initialized_block_assembles_and_projects_exact_period_one(layout):
    parameter = ParameterIdentity("layer.weight", (2, 3))
    group = _group()
    shards = _replicas(parameter, group) if layout == "replicated" else _flat(parameter, group, (2, 4))
    momentum = torch.tensor([[-0.0, 0.0, float.fromhex("0x1p-149")], [1.25, -3.5, torch.finfo(torch.float32).max]])
    second = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        shards,
        options=_plain_options(),
        policy=_policy(),
        variant="initialized_dense",
        momentum=momentum,
        second=second,
    )
    document = _assemble_portable_state_fragments(fragments, process_group_identity=group, limits=_limits())
    record = document["parameters"][parameter.fqn]
    assert torch.equal(record["state"]["second_moment"], second)
    assert torch.equal(record["state"]["momentum"].view(torch.int32), momentum.view(torch.int32))

    for target in _flat(parameter, group, (3, 3)):
        projected = _project_portable_parameter_state(
            record,
            target,
            implementation="gefen.Gefen",
            global_step=4,
            codebook=document["common"]["gefen_codebook"],
            source_second_moment_projection=document["policy"]["second_moment_projection"],
            target_algorithm_options=_plain_options(),
            target_second_moment="block",
        )
        local_momentum = _decode_quantized_momentum(
            document["common"]["gefen_codebook"],
            projected["m_codebook"],
            projected["m_magnitude"],
            logical_shape=(target.logical_slice.length,),
            period=1,
            step=4,
        )
        expected = _project_dense_logical_field(parameter, momentum, target)
        assert torch.equal(local_momentum.view(torch.int32), expected.view(torch.int32))
        assert projected["automatic_period"] == 1
        assert torch.equal(projected["vmean"].reshape(-1), _project_dense_logical_field(parameter, second, target))


def test_empty_flat_fragment_is_pristine_and_ownership_is_enforced():
    parameter = ParameterIdentity("layer.weight", (2,))
    group = _group(("rank:0", "rank:1", "rank:2"))
    shards = _flat(parameter, group, (1, 0, 1))
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        shards,
        options=_plain_options(),
        policy=_policy(),
        variant="initialized_dense",
        momentum=torch.tensor([-0.0, 2.0]),
        second=torch.tensor([1.0, 2.0]),
    )
    assert fragments[1]["logical_slots"][0]["role"] == "empty_flat"
    assert fragments[1]["logical_slots"][0]["state_variant"] == "pristine"
    corrupted = copy.deepcopy(fragments[1])
    corrupted["logical_slots"][0]["role"] = "live"
    with pytest.raises(ValueError, match="role"):
        _normalize_portable_state_fragment(corrupted, limits=_limits())


def test_factored_replicas_preserve_factor_bytes_and_define_live_fp32_block_projection():
    parameter = ParameterIdentity("matrix.weight", (2, 3))
    group = _group()
    shards = _replicas(parameter, group)
    momentum = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    v_row = torch.tensor([-0.0, 4.0])
    v_col = torch.tensor([1.0, 2.0, 3.0])
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        shards,
        options=_plain_options(second="factored"),
        policy=_policy(factored=True),
        variant="initialized_factored",
        momentum=momentum,
        v_row=v_row,
        v_col=v_col,
    )
    document = _assemble_portable_state_fragments(fragments, process_group_identity=group, limits=_limits())
    record = document["parameters"][parameter.fqn]
    assert torch.equal(record["state"]["v_row"].view(torch.int32), v_row.view(torch.int32))
    projected = _project_portable_parameter_state(
        record,
        shards[0],
        implementation="gefen.Gefen",
        global_step=4,
        codebook=_codebook(),
        source_second_moment_projection=document["policy"]["second_moment_projection"],
        target_algorithm_options=_plain_options(second="factored"),
        target_second_moment="factored",
    )
    assert torch.equal(projected["v_row"].view(torch.int32), v_row.view(torch.int32))
    assert torch.equal(projected["v_col"].view(torch.int32), v_col.view(torch.int32))
    block = _project_portable_parameter_state(
        record,
        shards[0],
        implementation="gefen.Gefen",
        global_step=4,
        codebook=_codebook(),
        source_second_moment_projection=document["policy"]["second_moment_projection"],
        target_algorithm_options=_plain_options(second="block"),
        target_second_moment="block",
    )
    denominator = v_row.mean().clamp_(min=torch.finfo(torch.float32).tiny)
    expected = torch.outer(v_row, v_col).div_(denominator)
    assert torch.equal(block["vmean"].reshape(2, 3).view(torch.int32), expected.view(torch.int32))
    assert block["vmean_step"] == 3
    assert block["step"] == 4
    assert block["automatic_period"] == 1
    fragments[1]["logical_slots"][0]["state"]["v_row"][0] = 0.0
    with pytest.raises(ValueError, match="v_row disagree"):
        _assemble_portable_state_fragments(fragments, process_group_identity=group, limits=_limits())


def test_second_moment_projection_rejects_reverse_and_exact_only_factored_migration():
    parameter = ParameterIdentity("matrix.weight", (2, 3))
    group = _group()
    shards = _replicas(parameter, group)
    block_document = _assemble_portable_state_fragments(
        _fragments(
            "gefen.Gefen",
            parameter,
            shards,
            options=_plain_options(second="block"),
            policy=_policy(),
            variant="initialized_dense",
            momentum=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            second=torch.arange(1, 7, dtype=torch.float32).reshape(2, 3),
        ),
        process_group_identity=group,
        limits=_limits(),
    )
    with pytest.raises(ValueError, match="block-to-factored"):
        _project_portable_parameter_state(
            block_document["parameters"][parameter.fqn],
            shards[0],
            implementation="gefen.Gefen",
            global_step=4,
            codebook=block_document["common"]["gefen_codebook"],
            source_second_moment_projection=block_document["policy"]["second_moment_projection"],
            target_algorithm_options=_plain_options(second="factored"),
            target_second_moment="factored",
        )

    exact_policy = _policy(factored=True)
    exact_policy["second_moment_projection"] = _SECOND_MOMENT_PROJECTION_EXACT
    exact_factored_document = _assemble_portable_state_fragments(
        _fragments(
            "gefen.Gefen",
            parameter,
            shards,
            options=_plain_options(second="factored"),
            policy=exact_policy,
            variant="initialized_factored",
            momentum=torch.arange(6, dtype=torch.float32).reshape(2, 3),
            v_row=torch.tensor([2.0, 3.0]),
            v_col=torch.tensor([5.0, 7.0, 11.0]),
        ),
        process_group_identity=group,
        limits=_limits(),
    )
    with pytest.raises(ValueError, match="does not authorize"):
        _project_portable_parameter_state(
            exact_factored_document["parameters"][parameter.fqn],
            shards[0],
            implementation="gefen.Gefen",
            global_step=4,
            codebook=exact_factored_document["common"]["gefen_codebook"],
            source_second_moment_projection=exact_factored_document["policy"]["second_moment_projection"],
            target_algorithm_options=_plain_options(second="block"),
            target_second_moment="block",
        )


def test_period_selected_projection_binds_target_second_moment_policy():
    parameter = ParameterIdentity("matrix.weight", (2, 3))
    group = _group()
    shards = _replicas(parameter, group)
    document = _assemble_portable_state_fragments(
        _fragments(
            "gefen.Gefen",
            parameter,
            shards,
            options=_plain_options(second="factored"),
            policy=_policy(factored=True),
            variant="period_selected",
        ),
        process_group_identity=group,
        limits=_limits(),
    )
    projected = _project_portable_parameter_state(
        document["parameters"][parameter.fqn],
        shards[0],
        implementation="gefen.Gefen",
        global_step=0,
        codebook=document["common"]["gefen_codebook"],
        source_second_moment_projection=document["policy"]["second_moment_projection"],
        target_algorithm_options=_plain_options(second="block"),
        target_second_moment="block",
    )
    assert projected == {"automatic_period": 1}
    with pytest.raises(ValueError, match="conflicts with target algorithm options"):
        _project_portable_parameter_state(
            document["parameters"][parameter.fqn],
            shards[0],
            implementation="gefen.Gefen",
            global_step=0,
            codebook=document["common"]["gefen_codebook"],
            source_second_moment_projection=document["policy"]["second_moment_projection"],
            target_algorithm_options=_plain_options(second="factored"),
            target_second_moment="block",
        )


def test_factored_matrix_rejects_flat_fragments_before_initialization():
    parameter = ParameterIdentity("matrix.weight", (2, 3))
    group = _group()
    with pytest.raises(ValueError, match="factored logical matrices require replicated"):
        _fragments(
            "gefen.Gefen",
            parameter,
            _flat(parameter, group, (2, 4)),
            options=_plain_options(second="factored"),
            policy=_policy(factored=True),
            variant="period_selected",
        )
    document = _assemble_portable_state_fragments(
        _fragments(
            "gefen.Gefen",
            parameter,
            _replicas(parameter, group),
            options=_plain_options(second="factored"),
            policy=_policy(factored=True),
            variant="period_selected",
        ),
        process_group_identity=group,
        limits=_limits(),
    )
    with pytest.raises(ValueError, match="replicated target shards"):
        _project_portable_parameter_state(
            document["parameters"][parameter.fqn],
            _flat(parameter, group, (2, 4))[0],
            implementation="gefen.Gefen",
            global_step=document["common"]["gefen_global_step"],
            codebook=document["common"]["gefen_codebook"],
            source_second_moment_projection=document["policy"]["second_moment_projection"],
            target_algorithm_options=_plain_options(second="factored"),
            target_second_moment="factored",
        )


@pytest.mark.parametrize("normuon", [False, True])
def test_muon_whole_owner_assembles_and_projects_after_owner_move(normuon):
    parameter = ParameterIdentity("muon.weight", (2, 3))
    group = _group()
    source = _owners(parameter, group, "rank:0")
    momentum = torch.tensor([[-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]])
    normuon_v = torch.tensor([[1.0], [2.0]])
    variant = "initialized_dense_normuon" if normuon else "initialized_dense"
    fragments = _fragments(
        "gefen.GefenMuon",
        parameter,
        source,
        options=_muon_options(normuon=normuon),
        policy=_policy(),
        variant=variant,
        momentum=momentum,
        normuon_v=normuon_v,
    )
    document = _assemble_portable_state_fragments(fragments, process_group_identity=group, limits=_limits())
    target = _owners(parameter, group, "rank:1")
    nonowner = _project_portable_parameter_state(
        document["parameters"][parameter.fqn],
        target[0],
        implementation="gefen.GefenMuon",
        global_step=4,
        codebook=_codebook(),
        source_second_moment_projection=document["policy"]["second_moment_projection"],
        target_algorithm_options=_muon_options(normuon=normuon),
    )
    owner = _project_portable_parameter_state(
        document["parameters"][parameter.fqn],
        target[1],
        implementation="gefen.GefenMuon",
        global_step=4,
        codebook=_codebook(),
        source_second_moment_projection=document["policy"]["second_moment_projection"],
        target_algorithm_options=_muon_options(normuon=normuon),
    )
    assert nonowner == {}
    assert owner["automatic_period"] == 1
    if normuon:
        assert torch.equal(owner["normuon_v"], normuon_v)


def test_muon_whole_owner_projection_requires_distributed_mode():
    parameter = ParameterIdentity("muon.weight", (2, 3))
    group = _group()
    document = _assemble_portable_state_fragments(
        _fragments(
            "gefen.GefenMuon",
            parameter,
            _replicas(parameter, group),
            options=_muon_options(mode="exact"),
            policy=_policy(),
            variant="period_selected",
        ),
        process_group_identity=group,
        limits=_limits(),
    )
    target = _owners(parameter, group, "rank:0")[0]
    with pytest.raises(ValueError, match="sharded_mode='distributed'"):
        _project_portable_parameter_state(
            document["parameters"][parameter.fqn],
            target,
            implementation="gefen.GefenMuon",
            global_step=document["common"]["gefen_global_step"],
            codebook=document["common"]["gefen_codebook"],
            source_second_moment_projection=document["policy"]["second_moment_projection"],
            target_algorithm_options=_muon_options(mode="exact"),
        )


def test_strict_document_schema_completion_digest_counters_and_period_gate():
    parameter = ParameterIdentity("layer.weight", (2,))
    record = {
        "identity": _serialize_parameter_identity(parameter),
        "algorithm_options": _plain_options(),
        "state_variant": "period_selected",
        "state": {},
        "projection_hints": {
            "source_periods": [1],
            "source_second_moment": None,
            "target_period": 1,
        },
    }
    document = build_portable_state_document(
        implementation="gefen.Gefen",
        policy=_policy(),
        common={"gefen_global_step": 1, "gefen_codebook": _codebook(), "gefen_deterministic": False},
        parameters={parameter.fqn: record},
        provenance=None,
    )
    _normalize_gefen_portable_state_document(document, limits=_limits())
    corrupt = copy.deepcopy(document)
    corrupt["completion"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        _normalize_gefen_portable_state_document(corrupt, limits=_limits())
    bad_period = copy.deepcopy(record)
    bad_period["projection_hints"]["source_periods"] = [2]
    with pytest.raises(ValueError, match="period one"):
        _normalize_gefen_portable_state_document(
            build_portable_state_document(
                implementation="gefen.Gefen",
                policy=_policy(),
                common=document["common"],
                parameters={parameter.fqn: bad_period},
                provenance=None,
            ),
            limits=_limits(),
        )
    too_large = copy.deepcopy(record)
    too_large["state_variant"] = "pristine"
    too_large["projection_hints"]["source_periods"] = []
    with pytest.raises(ValueError, match="at most"):
        _normalize_gefen_portable_state_document(
            build_portable_state_document(
                implementation="gefen.Gefen",
                policy=_policy(),
                common={
                    "gefen_global_step": (1 << 53),
                    "gefen_codebook": None,
                    "gefen_deterministic": False,
                },
                parameters={parameter.fqn: too_large},
                provenance=None,
            ),
            limits=_limits(),
        )


def test_global_step_zero_allows_period_selection_only_with_a_valid_codebook():
    parameter = ParameterIdentity("layer.weight", (2,))
    record = {
        "identity": _serialize_parameter_identity(parameter),
        "algorithm_options": _plain_options(),
        "state_variant": "period_selected",
        "state": {},
        "projection_hints": {
            "source_periods": [1],
            "source_second_moment": None,
            "target_period": 1,
        },
    }

    def document(codebook):
        return build_portable_state_document(
            implementation="gefen.Gefen",
            policy=_policy(),
            common={
                "gefen_global_step": 0,
                "gefen_codebook": codebook,
                "gefen_deterministic": False,
            },
            parameters={parameter.fqn: record},
            provenance=None,
        )

    normalized = _normalize_gefen_portable_state_document(document(_codebook()), limits=_limits())
    assert normalized["common"]["gefen_global_step"] == 0
    with pytest.raises(ValueError, match="requires a codebook"):
        _normalize_gefen_portable_state_document(document(None), limits=_limits())


def test_fragment_rejects_ungrouped_shards_and_empty_period_projection_stays_empty():
    parameter = ParameterIdentity("layer.weight", (2,))
    ungrouped = ShardIdentity(
        parameter,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(parameter),
    )
    catalog = {
        parameter.fqn: {
            "identity": _serialize_parameter_identity(parameter),
            "algorithm_options": _plain_options(),
        }
    }
    with pytest.raises(ValueError, match="ungrouped"):
        _build_portable_state_fragment(
            implementation="gefen.Gefen",
            member="rank:0",
            policy=_policy(),
            common={
                "gefen_global_step": 0,
                "gefen_codebook": None,
                "gefen_deterministic": False,
            },
            manifest=ShardingManifest((ungrouped,)),
            catalog=catalog,
            logical_slots=[
                {
                    "group_index": 0,
                    "original_slot_index": 0,
                    "compatibility_name": "weight",
                    "shard": _serialize_shard_identity(ungrouped),
                    "algorithm_options": _plain_options(),
                    "role": "live",
                    "source_period": None,
                    "source_second_moment": None,
                    "state_variant": "pristine",
                    "state": {},
                }
            ],
            limits=_limits(),
        )

    group = _group(("rank:0", "rank:1", "rank:2"))
    source = _replicas(parameter, group)
    document = _assemble_portable_state_fragments(
        _fragments(
            "gefen.Gefen",
            parameter,
            source,
            options=_plain_options(),
            policy=_policy(),
            variant="period_selected",
        ),
        process_group_identity=group,
        limits=_limits(),
    )
    empty_target = _flat(parameter, group, (1, 0, 1))[1]
    assert (
        _project_portable_parameter_state(
            document["parameters"][parameter.fqn],
            empty_target,
            implementation="gefen.Gefen",
            global_step=4,
            codebook=document["common"]["gefen_codebook"],
            source_second_moment_projection=document["policy"]["second_moment_projection"],
            target_algorithm_options=_plain_options(),
            target_second_moment="block",
        )
        == {}
    )


@pytest.mark.parametrize("implementation", ["gefen.Gefen", "gefen.GefenMuon"])
def test_complete_zero_numel_parameter_must_remain_pristine(implementation):
    parameter = ParameterIdentity("empty.weight", (0, 3))
    options = _plain_options(second="factored") if implementation == "gefen.Gefen" else _muon_options()
    policy = _policy(factored=implementation == "gefen.Gefen")
    hints = {"source_periods": [1], "target_period": 1}
    if implementation == "gefen.Gefen":
        hints["source_second_moment"] = None
    record = {
        "identity": _serialize_parameter_identity(parameter),
        "algorithm_options": options,
        "state_variant": "period_selected",
        "state": {},
        "projection_hints": hints,
    }
    document = build_portable_state_document(
        implementation=implementation,
        policy=policy,
        common={
            "gefen_global_step": 0,
            "gefen_codebook": _codebook(),
            "gefen_deterministic": False,
        },
        parameters={parameter.fqn: record},
        provenance=None,
    )
    with pytest.raises(ValueError, match="empty logical parameters must remain pristine"):
        _normalize_gefen_portable_state_document(document, limits=_limits())


def test_assembly_authenticates_full_process_group_identity():
    parameter = ParameterIdentity("layer.weight", (2,))
    source_group = _group()
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        _replicas(parameter, source_group),
        options=_plain_options(),
        policy=_policy(),
        variant="pristine",
    )
    wrong_scope = ProcessGroupIdentity("different-checkpoint", source_group.ordered_members)
    with pytest.raises(ValueError, match="exactly match process_group_identity"):
        _assemble_portable_state_fragments(
            fragments,
            process_group_identity=wrong_scope,
            limits=_limits(),
        )


def test_muon_schedule_respects_native_length_limit():
    parameter = ParameterIdentity("muon.weight", (2, 2))
    options = _muon_options()
    options["ns_schedule"] = [[1.0, 2.0, 3.0] for _ in range(100)]
    record = {
        "identity": _serialize_parameter_identity(parameter),
        "algorithm_options": options,
        "state_variant": "pristine",
        "state": {},
        "projection_hints": {"source_periods": [], "target_period": 1},
    }
    document = build_portable_state_document(
        implementation="gefen.GefenMuon",
        policy=_policy(),
        common={
            "gefen_global_step": 0,
            "gefen_codebook": None,
            "gefen_deterministic": False,
        },
        parameters={parameter.fqn: record},
        provenance=None,
    )
    with pytest.raises(ValueError, match="between 1 and 99"):
        _normalize_gefen_portable_state_document(document, limits=_limits())


def test_limits_are_strict_and_bound_tensor_and_member_aggregation():
    with pytest.raises(TypeError):
        PortableStateLimits(True, 2, 64 << 20)
    with pytest.raises(ValueError):
        PortableStateLimits(2, 1, 64 << 20)
    parameter = ParameterIdentity("layer.weight", (2,))
    group = _group()
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        _replicas(parameter, group),
        options=_plain_options(),
        policy=_policy(),
        variant="initialized_dense",
        momentum=torch.ones(2),
        second=torch.ones(2),
    )
    with pytest.raises(ValueError, match="max_fragment_tensor_bytes"):
        _normalize_portable_state_fragment(
            fragments[0],
            limits=PortableStateLimits(1, 64, 64 << 20),
        )
    with pytest.raises(ValueError, match="max_members"):
        _assemble_portable_state_fragments(
            fragments,
            process_group_identity=group,
            limits=dataclasses.replace(_limits(), max_members=1),
        )
    with pytest.raises(ValueError, match="max_collective_tensor_bytes"):
        _assemble_portable_state_fragments(
            fragments,
            process_group_identity=group,
            limits=PortableStateLimits(
                max_fragment_tensor_bytes=2048,
                max_collective_tensor_bytes=2048,
                max_collective_metadata_bytes=64 << 20,
            ),
        )


def test_signed_zero_is_bitwise_consensus_not_numeric_consensus():
    parameter = ParameterIdentity("layer.weight", (1,))
    group = _group()
    fragments = _fragments(
        "gefen.Gefen",
        parameter,
        _replicas(parameter, group),
        options=_plain_options(),
        policy=_policy(),
        variant="initialized_dense",
        momentum=torch.tensor([-0.0]),
        second=torch.tensor([1.0]),
    )
    fragments[1]["logical_slots"][0]["state"]["momentum"][0] = 0.0
    with pytest.raises(ValueError, match="replicas disagree"):
        _assemble_portable_state_fragments(fragments, process_group_identity=group, limits=_limits())


def test_logical_slot_positions_must_agree_across_members():
    first = ParameterIdentity("first.weight", (1,))
    second = ParameterIdentity("second.weight", (1,))
    group = _group()
    shards = _replicas(first, group) + _replicas(second, group)
    manifest = ShardingManifest(shards)
    common = {
        "gefen_global_step": 0,
        "gefen_codebook": None,
        "gefen_deterministic": False,
    }
    catalog = {
        parameter.fqn: {
            "identity": _serialize_parameter_identity(parameter),
            "algorithm_options": _plain_options(),
        }
        for parameter in (first, second)
    }
    fragments = []
    for member_index, member in enumerate(group.ordered_members):
        member_shards = [shard for shard in shards if shard.local_member == member]
        if member_index:
            member_shards.reverse()
        logical_slots = []
        for slot_index, shard in enumerate(member_shards):
            logical_slots.append(
                {
                    "group_index": 0,
                    "original_slot_index": slot_index,
                    "compatibility_name": shard.parameter.fqn,
                    "shard": _serialize_shard_identity(shard),
                    "algorithm_options": _plain_options(),
                    "role": "live",
                    "source_period": None,
                    "source_second_moment": None,
                    "state_variant": "pristine",
                    "state": {},
                }
            )
        fragments.append(
            _build_portable_state_fragment(
                implementation="gefen.Gefen",
                member=member,
                policy=_policy(),
                common=common,
                manifest=manifest,
                catalog=catalog,
                logical_slots=logical_slots,
                limits=_limits(),
            )
        )
    with pytest.raises(ValueError, match="logical slot position disagree"):
        _assemble_portable_state_fragments(
            fragments,
            process_group_identity=group,
            limits=_limits(),
        )


def test_public_limits_are_frozen_and_only_public_export():
    limits = _limits()
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.chunk_bytes = 1
    assert math.isfinite(float(limits.max_collective_tensor_bytes))
