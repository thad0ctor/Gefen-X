"""Focused capability coverage for directional checkpoint-state projections."""

from dataclasses import replace

import pytest
import torch

import gefen
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION,
    CheckpointProjectionQualifier,
    CheckpointStateRepresentation,
    CheckpointStateTransition,
    CheckpointStateTransitionKind,
    CheckpointSupport,
    CheckpointTransport,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ProcessGroupScope,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
    TopologyChange,
)
from gefen.gefen import Gefen
from gefen.portable_state import _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK
from gefen.rebinding import ParameterRebinding


_MEMBER = "rank:0"


def _finalized_plain(*, factored, include_block_bias=False):
    weight = torch.nn.Parameter(
        torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    )
    named_parameters = [("layer.weight", weight)]
    if include_block_bias:
        named_parameters.append(
            ("layer.bias", torch.nn.Parameter(torch.arange(3, dtype=torch.float32)))
        )
    optimizer = Gefen(
        named_parameters,
        fused=False,
        factored_v_2d=factored,
        force_2d_period_one=True,
    )
    group = ProcessGroupIdentity("checkpoint", (_MEMBER,))
    shards = tuple(
        ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
            placements=(
                ShardPlacement("checkpoint", PlacementKind.REPLICATE, 0, 1),
            ),
            process_group=group,
            local_member=_MEMBER,
        )
        for identity in (
            ParameterIdentity(name, tuple(parameter.shape))
            for name, parameter in named_parameters
        )
    )
    optimizer.post_sharding(
        tuple(
            ParameterRebinding(parameter, parameter, shard)
            for (_, parameter), shard in zip(named_parameters, shards)
        ),
        manifest=ShardingManifest(shards),
        codebook_process_group=CodebookProcessGroupBinding(
            group, _MEMBER, None, torch.device("cpu")
        ),
    )
    support = next(
        item
        for item in optimizer.optimizer_contract().capabilities.checkpoints
        if item.transport is CheckpointTransport.CANONICAL_GLOBAL
    )
    return support


def _transition(support, source, target):
    matches = tuple(
        transition
        for transition in support.state_transitions
        if transition.source is source and transition.target is target
    )
    assert len(matches) == 1
    return matches[0]


def test_factored_source_separates_exact_restore_from_versioned_block_projection():
    support = _finalized_plain(factored=True)
    factored = CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
    block = CheckpointStateRepresentation.BLOCK_SECOND_MOMENT

    assert support.same_topology == frozenset({ParameterLayout.REPLICATED})
    assert not support.topology_changing
    exact = _transition(support, factored, factored)
    assert exact.kind is CheckpointStateTransitionKind.EXACT
    assert exact.same_topology == support.same_topology
    assert not exact.topology_changing
    assert exact.qualifier is None

    projection = _transition(support, factored, block)
    assert projection.kind is CheckpointStateTransitionKind.DEFINED_PROJECTION
    assert (
        projection.qualifier
        is CheckpointProjectionQualifier.FACTORED_TO_BLOCK_LIVE_FP32_TARGET_PERIOD_ONE_V1
    )
    assert (
        projection.qualifier.value
        == _SECOND_MOMENT_PROJECTION_FACTORED_TO_BLOCK
    )
    assert projection.schema_version == CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION
    assert projection.same_topology == frozenset({ParameterLayout.REPLICATED})
    assert projection.topology_changing == frozenset(
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
    )
    assert projection.target_layouts == frozenset(
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
    )
    assert projection.topology_change_kinds == frozenset(
        {TopologyChange.PLACEMENT_RESHARD}
    )


def test_mixed_replicated_factored_weight_and_block_bias_declare_both_exact_paths():
    support = _finalized_plain(factored=True, include_block_bias=True)
    factored = CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
    block = CheckpointStateRepresentation.BLOCK_SECOND_MOMENT

    replicated = frozenset({ParameterLayout.REPLICATED})
    assert support.same_topology == replicated
    factored_exact = _transition(support, factored, factored)
    block_exact = _transition(support, block, block)
    assert factored_exact.kind is CheckpointStateTransitionKind.EXACT
    assert factored_exact.same_topology == replicated
    assert block_exact.kind is CheckpointStateTransitionKind.EXACT
    assert block_exact.same_topology == replicated
    projection = _transition(support, factored, block)
    assert (
        projection.qualifier
        is CheckpointProjectionQualifier.FACTORED_TO_BLOCK_LIVE_FP32_TARGET_PERIOD_ONE_V1
    )


def test_block_source_advertises_only_exact_block_restore():
    support = _finalized_plain(factored=False)
    block = CheckpointStateRepresentation.BLOCK_SECOND_MOMENT
    factored = CheckpointStateRepresentation.FACTORED_SECOND_MOMENT

    assert support.same_topology == frozenset({ParameterLayout.REPLICATED})
    assert support.topology_changing == frozenset(
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
        }
    )
    exact = _transition(support, block, block)
    assert exact.kind is CheckpointStateTransitionKind.EXACT
    assert exact.same_topology == support.same_topology
    assert exact.topology_changing == support.topology_changing
    assert exact.topology_change_kinds == support.topology_change_kinds
    assert all(
        transition.target is not factored
        for transition in support.state_transitions
    )


def test_transition_schema_is_additive_immutable_and_validates_direction():
    legacy = CheckpointSupport(
        CheckpointTransport.NATIVE_OPTIMIZER,
        frozenset({ParameterLayout.REPLICATED}),
        frozenset(),
        ProcessGroupScope.NONE,
    )
    assert legacy.state_transitions == ()
    assert gefen.CheckpointStateTransition is CheckpointStateTransition
    assert (
        gefen.CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION
        == CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION
    )

    factored = CheckpointStateRepresentation.FACTORED_SECOND_MOMENT
    block = CheckpointStateRepresentation.BLOCK_SECOND_MOMENT
    source_layouts = {ParameterLayout.REPLICATED}
    projection = CheckpointStateTransition(
        factored,
        block,
        CheckpointStateTransitionKind.DEFINED_PROJECTION,
        source_layouts,
        set(),
        qualifier=(
            CheckpointProjectionQualifier.FACTORED_TO_BLOCK_LIVE_FP32_TARGET_PERIOD_ONE_V1
        ),
    )
    source_layouts.clear()
    assert projection.same_topology == frozenset({ParameterLayout.REPLICATED})
    with pytest.raises(TypeError, match="require a CheckpointProjectionQualifier"):
        replace(projection, qualifier=None)
    with pytest.raises(ValueError, match="declared direction and kind"):
        replace(projection, source=block, target=factored)
    with pytest.raises(ValueError, match="unsupported checkpoint state-transition schema"):
        replace(projection, schema_version=CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION + 1)


def test_exact_transition_must_cover_legacy_exact_layout_claims():
    exact = CheckpointStateTransition(
        CheckpointStateRepresentation.BLOCK_SECOND_MOMENT,
        CheckpointStateRepresentation.BLOCK_SECOND_MOMENT,
        CheckpointStateTransitionKind.EXACT,
        frozenset({ParameterLayout.REPLICATED}),
        frozenset(),
    )
    with pytest.raises(ValueError, match="exact state transitions must cover"):
        CheckpointSupport(
            CheckpointTransport.CANONICAL_GLOBAL,
            frozenset({ParameterLayout.FLATTENED_ELEMENT_SHARD}),
            frozenset(),
            ProcessGroupScope.ADAPTER_DEFINED,
            state_transitions=(exact,),
        )
