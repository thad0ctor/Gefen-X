"""CPU coverage for public optimizer capability and state-layout contracts."""

import copy
from dataclasses import FrozenInstanceError

import pytest
import torch

import gefen
from gefen import (
    CONTRACT_SCHEMA_VERSION,
    CheckpointTransport,
    Gefen,
    GefenMuon,
    GefenMuonHybrid,
    OptimizerContractProvider,
    OptimizerStateLayout,
    ParameterLayout,
    ParameterStateRole,
    Precision,
    ProcessGroupScope,
    StateExtent,
    StateField,
    StateGeometry,
    StateKeyMatch,
    StateScope,
    StateVariant,
    TopologyChange,
)


_DTENSOR = ParameterLayout.DTENSOR_1D_DEFAULT_WORLD


def _values_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(
            left, right
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _declares_state_key(contract, key):
    try:
        contract.state_layout.field(key)
    except KeyError:
        return False
    return True


def _step_with_fixed_grad(optimizer, params):
    for index, param in enumerate(params):
        values = torch.arange(param.numel(), dtype=param.dtype).reshape_as(param)
        param.grad = (values + index + 1) / max(1, param.numel())
    optimizer.step()


def _persistent_parameter_keys(contract, state):
    return {
        key
        for key in state
        if _declares_state_key(contract, key)
        and contract.state_layout.field(key).scope is StateScope.PARAMETER
    }


def _matching_variants(
    contract,
    state,
    *,
    layout=ParameterLayout.REPLICATED,
    parameter_rank=2,
    sharded_mode=None,
    role=ParameterStateRole.ANY,
):
    keys = _persistent_parameter_keys(contract, state)
    return [
        variant
        for variant in contract.state_layout.parameter_variants
        if set(variant.fields) == keys
        and layout in variant.layouts
        and (
            variant.parameter_ranks is None
            or parameter_rank in variant.parameter_ranks
        )
        and parameter_rank not in variant.excluded_parameter_ranks
        and variant.sharded_mode == sharded_mode
        and (variant.role is ParameterStateRole.ANY or variant.role is role)
    ]


def _training_support(contract, layout, mode=None):
    matches = [
        item
        for item in contract.capabilities.training
        if item.layout is layout
        and (mode is None or item.sharded_mode == mode)
    ]
    assert len(matches) == 1
    return matches[0]


def _checkpoint_support(contract, transport, layout):
    return [
        item
        for item in contract.capabilities.checkpoints
        if item.transport is transport
        and (layout in item.same_topology or layout in item.topology_changing)
    ]


@pytest.mark.parametrize("factored_v_2d", [False, True])
def test_plain_contract_matches_live_persistent_state(factored_v_2d):
    param = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    optimizer = Gefen(
        [("layer.weight", param)],
        fused=False,
        factored_v_2d=factored_v_2d,
    )
    assert isinstance(optimizer, OptimizerContractProvider)
    contract = optimizer.optimizer_contract()

    assert contract.schema_version == CONTRACT_SCHEMA_VERSION
    assert contract.implementation == "gefen.Gefen"
    assert contract.capabilities.supported_parameter_ranks is None
    assert {
        item.layout for item in contract.capabilities.training
    } == {
        ParameterLayout.REPLICATED,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        _DTENSOR,
    }
    dtensor_training = _training_support(contract, _DTENSOR)
    assert dtensor_training.mesh_dimensions == (1,)
    assert dtensor_training.process_group_scope is ProcessGroupScope.INFERRED_DEVICE_MESH
    dcp = _checkpoint_support(
        contract, CheckpointTransport.PYTORCH_RANK_LOCAL, _DTENSOR
    )
    assert len(dcp) == 1
    assert dcp[0].process_group_scope is ProcessGroupScope.DEFAULT_WORLD
    assert dcp[0].mesh_dimensions == (1,)
    assert dcp[0].requires_collective
    assert dcp[0].atomic_load
    assert not dcp[0].topology_changing
    native = next(
        item
        for item in contract.capabilities.checkpoints
        if item.transport is CheckpointTransport.NATIVE_OPTIMIZER
    )
    assert native.atomic_load
    assert ParameterLayout.FLATTENED_ELEMENT_SHARD not in native.same_topology
    assert contract.capabilities.accepts_semantic_parameter_names
    assert not contract.capabilities.canonical_parameter_fqns
    assert not contract.capabilities.stable_shard_identity
    assert contract.capabilities.explicit_process_group_codebook_scope
    assert contract.capabilities.shard_rebinding
    assert contract.capabilities.post_sharding
    assert not contract.capabilities.canonical_state_io
    assert not contract.capabilities.atomic_state_movement
    assert not contract.capabilities.state_offload
    assert Precision.FLOAT64 in contract.capabilities.precisions
    flattened = _training_support(
        contract, ParameterLayout.FLATTENED_ELEMENT_SHARD
    )
    assert flattened.process_group_scope is ProcessGroupScope.NONE

    assert optimizer.state[param] == {"name": "layer.weight"}
    name_only = next(
        item
        for item in contract.state_layout.parameter_variants
        if item.name == "name_only"
    )
    assert name_only.fields == ("name",)
    assert name_only.extent is StateExtent.METADATA_ONLY
    assert not name_only.initialized

    _step_with_fixed_grad(optimizer, [param])
    state = optimizer.state[param]
    assert all(_declares_state_key(contract, key) for key in state)
    expected_variant = (
        "factored_second_moment" if factored_v_2d else "block_second_moment"
    )
    variant = next(
        item
        for item in contract.state_layout.parameter_variants
        if item.name == expected_variant
    )
    persistent_keys = {
        key
        for key in state
        if contract.state_layout.field(key).scope is StateScope.PARAMETER
    }
    assert persistent_keys == set(variant.fields)

    state_dict = optimizer.state_dict()
    common = {
        field.name
        for field in contract.state_layout.fields_for_scope(
            StateScope.OPTIMIZER_COMMON
        )
    }
    assert common == {
        "gefen_global_step",
        "gefen_codebook",
        "gefen_deterministic",
        "gefen_codebook_scope",
    }
    required_common = {
        field.name
        for field in contract.state_layout.fields_for_scope(
            StateScope.OPTIMIZER_COMMON
        )
        if not field.optional
    }
    assert required_common.issubset(state_dict)
    assert contract.state_layout.field("gefen_codebook_scope").optional
    assert "gefen_codebook_scope" not in state_dict


@pytest.mark.parametrize(
    ("source_factored", "target_factored", "migrated_variant"),
    [
        (True, False, "block_with_retained_factored_state"),
        (False, True, "factored_with_retained_block_state"),
    ],
)
def test_plain_contract_declares_factored_v_checkpoint_migration_states(
    source_factored, target_factored, migrated_variant
):
    initial = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    source_param = torch.nn.Parameter(initial.clone())
    source = Gefen(
        [("layer.weight", source_param)],
        fused=False,
        factored_v_2d=source_factored,
    )
    _step_with_fixed_grad(source, [source_param])
    checkpoint = copy.deepcopy(source.state_dict())

    target_param = torch.nn.Parameter(initial.clone())
    target = Gefen(
        [("layer.weight", target_param)],
        fused=False,
        factored_v_2d=target_factored,
    )
    target.load_state_dict(checkpoint)
    contract = target.optimizer_contract()
    pending = _matching_variants(contract, target.state[target_param])
    expected_pending = (
        "factored_state_pending_block_initialization"
        if source_factored
        else "block_state_pending_factored_initialization"
    )
    assert [variant.name for variant in pending] == [expected_pending]
    assert pending[0].migration_only

    _step_with_fixed_grad(target, [target_param])
    matches = _matching_variants(contract, target.state[target_param])
    assert [variant.name for variant in matches] == [migrated_variant]
    assert matches[0].inactive_fields


@pytest.mark.parametrize("target_factored", [False, True])
def test_plain_contract_declares_legacy_vmean_counter_transition(target_factored):
    initial = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    source_param = torch.nn.Parameter(initial.clone())
    source = Gefen(
        [("layer.weight", source_param)],
        fused=False,
        factored_v_2d=False,
    )
    _step_with_fixed_grad(source, [source_param])
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["state"][0].pop("vmean_step")

    target_param = torch.nn.Parameter(initial.clone())
    target = Gefen(
        [("layer.weight", target_param)],
        fused=False,
        factored_v_2d=target_factored,
    )
    target.load_state_dict(checkpoint)
    contract = target.optimizer_contract()
    pending = _matching_variants(contract, target.state[target_param])
    expected_pending = (
        "legacy_block_state_pending_factored_initialization"
        if target_factored
        else "legacy_block_without_vmean_counter"
    )
    assert [variant.name for variant in pending] == [expected_pending]

    _step_with_fixed_grad(target, [target_param])
    matches = _matching_variants(contract, target.state[target_param])
    expected_after = (
        "factored_with_retained_legacy_block_state"
        if target_factored
        else "block_second_moment"
    )
    assert [variant.name for variant in matches] == [expected_after]


def test_plain_contract_classifies_legacy_m_codebook_shape_as_inert():
    initial = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    source_param = torch.nn.Parameter(initial.clone())
    source = Gefen([("layer.weight", source_param)], fused=False)
    _step_with_fixed_grad(source, [source_param])
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["state"][0]["m_codebook_shape"] = tuple(
        checkpoint["state"][0]["m_codebook"].shape
    )

    target_param = torch.nn.Parameter(initial.clone())
    target = Gefen([("layer.weight", target_param)], fused=False)
    target.load_state_dict(checkpoint)
    contract = target.optimizer_contract()
    field = contract.state_layout.field("m_codebook_shape")
    assert field.scope is StateScope.DERIVED
    assert not field.authoritative
    assert "m_codebook_shape" in target.state[target_param]
    assert _matching_variants(contract, target.state[target_param])


@pytest.mark.parametrize(
    ("sharded_mode", "normuon", "same_topology_dtensor", "reshard_dtensor"),
    [
        ("exact", False, False, False),
        ("approx", False, True, False),
        ("distributed", True, True, True),
    ],
)
def test_muon_contract_separates_mode_topology_and_state_extent(
    sharded_mode, normuon, same_topology_dtensor, reshard_dtensor
):
    param = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    optimizer = GefenMuon(
        [("layer.weight", param)],
        fused=False,
        sharded_mode=sharded_mode,
        normuon=normuon,
    )
    contract = optimizer.optimizer_contract()

    assert contract.implementation == "gefen.GefenMuon"
    assert contract.capabilities.supported_parameter_ranks == (2,)
    assert contract.capabilities.explicit_process_group_codebook_scope
    native = next(
        item
        for item in contract.capabilities.checkpoints
        if item.transport is CheckpointTransport.NATIVE_OPTIMIZER
        and ParameterLayout.REPLICATED in item.same_topology
    )
    assert native.atomic_load
    replicated = _training_support(contract, ParameterLayout.REPLICATED)
    assert replicated.requires_complete_logical_matrix
    assert not replicated.requires_complete_parameter_storage
    dtensor = _training_support(contract, _DTENSOR, sharded_mode)
    assert dtensor.mesh_dimensions == (1,)
    assert dtensor.requires_complete_logical_matrix == (
        sharded_mode in ("exact", "distributed")
    )
    assert not dtensor.requires_complete_parameter_storage

    same_topology = any(
        _DTENSOR in support.same_topology
        for support in contract.capabilities.checkpoints
    )
    topology_changing = any(
        _DTENSOR in support.topology_changing
        for support in contract.capabilities.checkpoints
    )
    assert same_topology is same_topology_dtensor
    assert topology_changing is reshard_dtensor
    if same_topology_dtensor:
        support = next(
            item
            for item in contract.capabilities.checkpoints
            if _DTENSOR in item.same_topology
        )
        assert support.mesh_dimensions == (1,)
        assert support.requires_collective
        assert support.atomic_load
    if reshard_dtensor:
        support = next(
            item
            for item in contract.capabilities.checkpoints
            if _DTENSOR in item.topology_changing
        )
        assert support.topology_change_kinds == frozenset(
            {TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION}
        )
        assert TopologyChange.PLACEMENT_RESHARD not in support.topology_change_kinds

    common_names = {
        field.name
        for field in contract.state_layout.fields_for_scope(
            StateScope.OPTIMIZER_COMMON
        )
    }
    assert "gefen_muon_distributed" not in common_names
    if sharded_mode == "distributed":
        manifest = contract.state_layout.field("gefen_muon_distributed")
        assert manifest.scope is StateScope.DERIVED
        assert not manifest.authoritative
        owner = next(
            item
            for item in contract.state_layout.parameter_variants
            if item.name == "quantized_normuon_owner"
        )
        assert owner.extent is StateExtent.OWNER_PARAMETER
        assert owner.role is ParameterStateRole.OWNER
        non_owner = next(
            item
            for item in contract.state_layout.parameter_variants
            if item.name == "distributed_non_owner"
        )
        assert non_owner.fields == ("name", "automatic_period")
        assert non_owner.role is ParameterStateRole.NON_OWNER
        owner_lazy = _matching_variants(
            contract,
            {"name": "layer.weight"},
            layout=_DTENSOR,
            sharded_mode="distributed",
            role=ParameterStateRole.OWNER,
        )
        non_owner_lazy = _matching_variants(
            contract,
            {"name": "layer.weight"},
            layout=_DTENSOR,
            sharded_mode="distributed",
            role=ParameterStateRole.NON_OWNER,
        )
        assert [item.name for item in owner_lazy] == ["distributed_owner_name_only"]
        assert [item.name for item in non_owner_lazy] == [
            "distributed_non_owner_name_only"
        ]

    assert optimizer.state[param] == {"name": "layer.weight"}
    _step_with_fixed_grad(optimizer, [param])
    state = optimizer.state[param]
    assert all(_declares_state_key(contract, key) for key in state)
    expected_variant = (
        "quantized_normuon_replicated_" + sharded_mode
        if normuon
        else "quantized_muon_replicated_" + sharded_mode
    )
    variant = next(
        item
        for item in contract.state_layout.parameter_variants
        if item.name == expected_variant
    )
    persistent_keys = _persistent_parameter_keys(contract, state)
    assert persistent_keys == set(variant.fields)


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_hybrid_contract_preserves_child_namespaces(backup_optimizer):
    matrix = torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    bias = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
        backup_optimizer=backup_optimizer,
    )
    contract = optimizer.optimizer_contract()

    assert contract.implementation == "gefen.GefenMuonHybrid"
    assert tuple(child.role for child in contract.children) == ("muon", "backup")
    assert contract.children[0].contract.implementation == "gefen.GefenMuon"
    if backup_optimizer == "gefen":
        assert contract.children[1].contract.implementation == "gefen.Gefen"
    else:
        assert contract.children[1].implementation == "torch.optim.adamw.AdamW"
        assert contract.children[1].contract is None
    assert contract.state_layout.composite_namespaces == ("muon", "backup")
    assert not contract.capabilities.explicit_process_group_codebook_scope
    assert contract.children[0].contract.capabilities.explicit_process_group_codebook_scope
    if backup_optimizer == "gefen":
        assert contract.children[1].contract.capabilities.explicit_process_group_codebook_scope
    assert {field.name for field in contract.state_layout.fields} == {
        "backup_optimizer"
    }
    checkpoint = contract.capabilities.checkpoints
    assert len(checkpoint) == 1
    assert checkpoint[0].transport is CheckpointTransport.COMPOSITE_NATIVE
    assert checkpoint[0].same_topology == frozenset({ParameterLayout.REPLICATED})
    assert not checkpoint[0].topology_changing
    assert not checkpoint[0].atomic_load


def test_muon_contract_keeps_mixed_normuon_variants_in_one_mode():
    plain = torch.nn.Parameter(torch.ones(4, 4))
    normuon = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = GefenMuon(
        [
            {"params": [("plain", plain)], "normuon": False},
            {"params": [("normuon", normuon)], "normuon": True},
        ],
        fused=False,
        sharded_mode="exact",
    )
    variants = {
        item.name for item in optimizer.optimizer_contract().state_layout.parameter_variants
    }
    assert "quantized_muon_replicated_exact" in variants
    assert "quantized_normuon_replicated_exact" in variants


def test_muon_mixed_approx_distributed_checkpoint_is_same_topology_only():
    first = torch.nn.Parameter(torch.ones(4, 4))
    second = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = GefenMuon(
        [
            {"params": [("first", first)], "sharded_mode": "approx"},
            {"params": [("second", second)], "sharded_mode": "distributed"},
        ],
        fused=False,
    )
    support = _checkpoint_support(
        optimizer.optimizer_contract(),
        CheckpointTransport.PYTORCH_RANK_LOCAL,
        _DTENSOR,
    )
    assert len(support) == 1
    assert support[0].required_sharded_modes == frozenset(
        {"approx", "distributed"}
    )
    assert not support[0].topology_changing


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_hybrid_approx_does_not_require_complete_dtensor_matrix(backup_optimizer):
    matrix = torch.nn.Parameter(torch.ones(4, 4))
    bias = torch.nn.Parameter(torch.ones(4))
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
        sharded_mode="approx",
        backup_optimizer=backup_optimizer,
    )
    contract = optimizer.optimizer_contract()
    support = _training_support(contract, _DTENSOR, "approx")
    assert not support.requires_complete_parameter_storage
    assert not support.requires_complete_logical_matrix


def test_bare_parameters_do_not_claim_canonical_fqns():
    param = torch.nn.Parameter(torch.ones(4, 4))
    optimizer = Gefen([param], fused=False)
    contract = optimizer.optimizer_contract()
    assert optimizer.state[param]["name"] == "param_0"
    assert contract.capabilities.accepts_semantic_parameter_names
    assert not contract.capabilities.canonical_parameter_fqns
    assert not contract.capabilities.stable_shard_identity


def test_contract_is_deeply_immutable_and_query_is_behavior_neutral():
    initial = torch.linspace(-1, 1, 16).reshape(4, 4)
    param = torch.nn.Parameter(initial.clone())
    reference_param = torch.nn.Parameter(initial.clone())
    optimizer = Gefen([("layer.weight", param)], fused=False, factored_v_2d=False)
    reference = Gefen(
        [("layer.weight", reference_param)], fused=False, factored_v_2d=False
    )

    before = copy.deepcopy(optimizer.state_dict())
    contract = optimizer.optimizer_contract()
    assert contract == optimizer.optimizer_contract()
    assert _values_equal(before, optimizer.state_dict())
    with pytest.raises(FrozenInstanceError):
        contract.implementation = "changed"

    source_fields = [
        StateField("name", StateScope.PARAMETER, StateGeometry.OPAQUE, True)
    ]
    source_variant_fields = ["name"]
    source_layouts = {ParameterLayout.REPLICATED}
    source_variants = [
        StateVariant(
            "name_only",
            source_variant_fields,
            source_layouts,
            StateExtent.METADATA_ONLY,
            initialized=False,
        )
    ]
    layout = OptimizerStateLayout(source_fields, source_variants)
    source_fields.clear()
    source_variant_fields.clear()
    source_layouts.clear()
    source_variants.clear()
    assert tuple(field.name for field in layout.fields) == ("name",)
    assert layout.parameter_variants[0].fields == ("name",)
    assert layout.parameter_variants[0].layouts == frozenset(
        {ParameterLayout.REPLICATED}
    )

    grad = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 16
    param.grad = grad.clone()
    reference_param.grad = grad.clone()
    optimizer.step()
    reference.step()
    assert torch.equal(param, reference_param)
    assert _values_equal(optimizer.state_dict(), reference.state_dict())


def test_contract_rejects_duplicate_variant_names():
    field = StateField("name", StateScope.PARAMETER, StateGeometry.OPAQUE, True)
    variant = StateVariant(
        "name_only",
        ("name",),
        frozenset({ParameterLayout.REPLICATED}),
        StateExtent.METADATA_ONLY,
        initialized=False,
    )
    with pytest.raises(ValueError, match="names must be unique"):
        OptimizerStateLayout((field,), (variant, variant))


def test_derived_fields_are_explicitly_non_authoritative():
    optimizer = Gefen(
        [("layer.weight", torch.nn.Parameter(torch.ones(4, 4)))],
        fused=False,
    )
    derived = optimizer.optimizer_contract().state_layout.fields_for_scope(
        StateScope.DERIVED
    )
    assert derived
    assert all(not field.authoritative for field in derived)
    assert {field.name for field in derived if field.checkpointed} == {
        "m_codebook_shape",
        "_gefen_rank_local_payload_",
        "_gefen_rank_local_member",
        "_gefen_checkpoint_metadata",
        "gefen_native_local_shards",
    }
    payload_field = optimizer.optimizer_contract().state_layout.field(
        "_gefen_rank_local_payload_3"
    )
    assert payload_field.key_match is StateKeyMatch.PREFIX


def test_all_public_contract_exports_resolve():
    from gefen import contracts

    assert all(getattr(gefen, name) is not None for name in contracts.__all__)
