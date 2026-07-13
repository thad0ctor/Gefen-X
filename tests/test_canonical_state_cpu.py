import copy
import io

import pytest
import torch

from gefen import (
    CANONICAL_STATE_FORMAT_VERSION,
    CanonicalStateProvider,
    CheckpointTransport,
    Gefen,
    GefenMuon,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    PlacementKind,
    PreparedCanonicalStateImport,
    ProcessGroupIdentity,
    ProcessGroupScope,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)


def _replicated(identity, group=None, member=None):
    placements = ()
    if group is not None:
        coordinate = group.ordered_members.index(member)
        placements = (
            ShardPlacement(
                "dp",
                PlacementKind.REPLICATE,
                coordinate,
                len(group.ordered_members),
            ),
        )
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
        process_group=group,
        local_member=member,
        placements=placements,
    )


def _flat(identity, group, member, offset, length):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        identity,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        LogicalSlice(offset, length),
        process_group=group,
        local_member=member,
        placements=(
            ShardPlacement(
                "dp",
                PlacementKind.FLAT_SHARD,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _finalize(optimizer, bindings, manifest):
    optimizer.post_sharding(
        tuple(ParameterRebinding(parameter, parameter, shard) for parameter, shard in bindings),
        manifest=manifest,
    )


def _snapshot(optimizer):
    return {
        "dict": optimizer.__dict__.copy(),
        "groups": optimizer.param_groups,
        "state": optimizer.state,
        "codebook": optimizer._gefen_codebook,
        "global_step": optimizer._gefen_global_step,
    }


def _assert_snapshot_identity(optimizer, snapshot):
    assert optimizer.param_groups is snapshot["groups"]
    assert optimizer.state is snapshot["state"]
    assert optimizer._gefen_codebook is snapshot["codebook"]
    assert optimizer._gefen_global_step is snapshot["global_step"]
    assert optimizer.__dict__.keys() == snapshot["dict"].keys()
    for key, value in snapshot["dict"].items():
        assert optimizer.__dict__[key] is value


def _two_parameter_source():
    first = torch.nn.Parameter(torch.arange(1, 5, dtype=torch.float32))
    second = torch.nn.Parameter(torch.arange(11, 15, dtype=torch.float32))
    optimizer = Gefen(
        [("first", first), ("second", second)],
        fused=False,
        factored_v_2d=False,
    )
    first_identity = ParameterIdentity("Model.First", (4,))
    second_identity = ParameterIdentity("Model.Second", (4,))
    first_shard = _replicated(first_identity)
    second_shard = _replicated(second_identity)
    manifest = ShardingManifest((first_shard, second_shard))
    _finalize(
        optimizer,
        ((first, first_shard), (second, second_shard)),
        manifest,
    )
    optimizer._resolve_automatic_period = lambda *args: 4
    first.grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    second.grad = torch.tensor([-8.0, 2.0, 1.0, 3.0])
    optimizer.step()
    return optimizer, first, second, first_shard, second_shard, manifest


def _reordered_target(first_value, second_value, first_shard, second_shard, manifest):
    second = torch.nn.Parameter(second_value.detach().clone())
    first = torch.nn.Parameter(first_value.detach().clone())
    optimizer = Gefen(
        [
            {"params": [("second_target", second)]},
            {"params": [("first_target", first)]},
        ],
        fused=False,
        factored_v_2d=False,
    )
    _finalize(
        optimizer,
        ((second, second_shard), (first, first_shard)),
        manifest,
    )
    return optimizer, first, second


def test_canonical_local_capability_is_dynamic_and_exact_binding_only():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = Gefen([("p", parameter)], fused=False, factored_v_2d=False)
    assert isinstance(optimizer, CanonicalStateProvider)
    assert not optimizer.optimizer_contract().capabilities.canonical_state_io
    assert all(
        support.transport is not CheckpointTransport.CANONICAL_LOCAL
        for support in optimizer.optimizer_contract().capabilities.checkpoints
    )

    identity = ParameterIdentity("Model.P", (4,))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    contract = optimizer.optimizer_contract()
    support = next(
        item for item in contract.capabilities.checkpoints if item.transport is CheckpointTransport.CANONICAL_LOCAL
    )
    assert contract.capabilities.canonical_state_io
    assert support.same_topology == frozenset({ParameterLayout.REPLICATED})
    assert not support.topology_changing
    assert support.process_group_scope is ProcessGroupScope.NONE
    assert support.atomic_load
    assert not support.requires_collective


def test_pristine_export_is_primitive_device_neutral_and_excludes_derived_state():
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = Gefen([("p", parameter)], fused=False, factored_v_2d=False)
    identity = ParameterIdentity("Model.P", (8,))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    optimizer.state[parameter]["stepsize"] = torch.ones(1)

    exported = optimizer.export_canonical_state()

    assert exported["format"] == "gefen.bound_state"
    assert exported["format_version"] == CANONICAL_STATE_FORMAT_VERSION
    assert exported["coverage"] == "local_optimizer_fragment"
    assert exported["manifest"][0]["process_group"] is None
    assert set(exported["parameters"]) == {"Model.P"}
    record = exported["parameters"]["Model.P"]
    assert record["state"] == {}
    assert "stepsize" not in record["state"]
    assert "name" not in record["state"]
    assert record["compatibility_name"] == "p"

    buffer = io.BytesIO()
    torch.save(exported, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    assert loaded["format_version"] == CANONICAL_STATE_FORMAT_VERSION
    assert loaded["parameters"]["Model.P"]["state"] == {}


def test_pristine_continuation_routes_period_policy_by_canonical_fqn():
    source_parameter = torch.nn.Parameter(torch.arange(1, 9, dtype=torch.float32))
    source = Gefen(
        [("special_weight", source_parameter)],
        fused=False,
        factored_v_2d=False,
        period_one_substrings=("special",),
    )
    identity = ParameterIdentity("Model.SpecialWeight", (8,))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(source, ((source_parameter, shard),), manifest)

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target = Gefen(
        [("unrelated_target_name", target_parameter)],
        fused=False,
        factored_v_2d=False,
        period_one_substrings=("special",),
    )
    _finalize(target, ((target_parameter, shard),), manifest)
    target.import_canonical_state(source.export_canonical_state())

    gradient = torch.arange(1, 9, dtype=torch.float32)
    source_parameter.grad = gradient.clone()
    target_parameter.grad = gradient.clone()
    source.step()
    target.step()
    assert source.state[source_parameter]["automatic_period"] == 1
    assert target.state[target_parameter]["automatic_period"] == 1
    assert torch.equal(target_parameter, source_parameter)
    assert torch.equal(target._gefen_codebook, source._gefen_codebook)


def test_informational_empty_compatibility_name_round_trips_but_stochastic_policy_is_unclaimed():
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = Gefen([("", parameter)], fused=False, factored_v_2d=False)
    identity = ParameterIdentity("Model.P", (4,))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(optimizer, ((parameter, shard),), manifest)
    exported = optimizer.export_canonical_state()
    assert exported["parameters"]["Model.P"]["compatibility_name"] == ""
    optimizer.import_canonical_state(exported)
    assert optimizer.state[parameter]["name"] == ""

    stochastic_parameter = torch.nn.Parameter(torch.ones(4))
    with pytest.warns(RuntimeWarning, match="stochastic_round=True"):
        stochastic = Gefen(
            [("p", stochastic_parameter)],
            fused=False,
            factored_v_2d=False,
            stochastic_round=True,
        )
    _finalize(stochastic, ((stochastic_parameter, shard),), manifest)
    assert not stochastic.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="supported finalized"):
        stochastic.export_canonical_state()


def test_initialized_import_maps_by_fqn_across_order_and_group_boundaries():
    source, source_first, source_second, first_shard, second_shard, manifest = _two_parameter_source()
    exported = source.export_canonical_state()
    exported_magnitude = exported["parameters"]["Model.First"]["state"][
        "m_magnitude"
    ]
    assert exported_magnitude is not source.state[source_first]["m_magnitude"]
    assert (
        exported_magnitude.untyped_storage().nbytes()
        == exported_magnitude.numel() * exported_magnitude.element_size()
    )
    target, target_first, target_second = _reordered_target(
        source_first,
        source_second,
        first_shard,
        second_shard,
        manifest,
    )
    load_hooks = []
    target.register_load_state_dict_pre_hook(lambda *args: load_hooks.append("pre"))
    target.register_load_state_dict_post_hook(lambda *args: load_hooks.append("post"))
    input_codebook = exported["common"]["gefen_codebook"]
    input_magnitude = exported["parameters"]["Model.First"]["state"]["m_magnitude"]
    input_codebook_value = input_codebook.clone()
    input_magnitude_value = input_magnitude.clone()

    prepared = target.prepare_canonical_state_import(exported)
    assert isinstance(prepared, PreparedCanonicalStateImport)
    assert target._gefen_codebook is None
    assert exported["common"]["gefen_codebook"] is input_codebook
    assert exported["parameters"]["Model.First"]["state"]["m_magnitude"] is input_magnitude
    assert torch.equal(input_codebook, input_codebook_value)
    assert torch.equal(input_magnitude, input_magnitude_value)
    target.commit_canonical_state_import(prepared)

    assert not load_hooks
    assert torch.equal(
        target.state[target_first]["m_magnitude"],
        source.state[source_first]["m_magnitude"],
    )
    assert torch.equal(
        target.state[target_second]["m_magnitude"],
        source.state[source_second]["m_magnitude"],
    )
    assert target.state[target_first]["name"] == "first_target"
    assert target.state[target_second]["name"] == "second_target"

    first_grad = torch.tensor([4.0, -3.0, 2.0, -1.0])
    second_grad = torch.tensor([1.0, 3.0, -5.0, 7.0])
    source_first.grad = first_grad.clone()
    target_first.grad = first_grad.clone()
    source_second.grad = second_grad.clone()
    target_second.grad = second_grad.clone()
    source.step()
    target.step()
    assert torch.equal(target_first, source_first)
    assert torch.equal(target_second, source_second)
    assert torch.equal(target._gefen_codebook, source._gefen_codebook)

    with pytest.raises(RuntimeError, match="already consumed"):
        target.commit_canonical_state_import(prepared)


def test_prepared_import_rejects_wrong_optimizer_and_stale_live_state():
    source, source_first, source_second, first_shard, second_shard, manifest = _two_parameter_source()
    exported = source.export_canonical_state()
    target, _, _ = _reordered_target(
        source_first,
        source_second,
        first_shard,
        second_shard,
        manifest,
    )
    other, _, _ = _reordered_target(
        source_first,
        source_second,
        first_shard,
        second_shard,
        manifest,
    )
    prepared = target.prepare_canonical_state_import(exported)

    other_before = _snapshot(other)
    with pytest.raises(ValueError, match="another optimizer"):
        other.commit_canonical_state_import(prepared)
    _assert_snapshot_identity(other, other_before)

    target._gefen_global_step += 1
    target_before = _snapshot(target)
    with pytest.raises(RuntimeError, match="changed after"):
        target.commit_canonical_state_import(prepared)
    _assert_snapshot_identity(target, target_before)


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_parameter",
        "wrong_manifest",
        "wrong_shard",
        "wrong_policy",
        "wrong_group_options",
        "unknown_state",
        "bad_state_geometry",
        "counter_ahead",
        "secondary_counter_ahead",
        "bool_version",
    ],
)
def test_canonical_corruption_rejects_before_live_mutation(corrupt):
    source, source_first, source_second, first_shard, second_shard, manifest = _two_parameter_source()
    exported = source.export_canonical_state()
    target, _, _ = _reordered_target(
        source_first,
        source_second,
        first_shard,
        second_shard,
        manifest,
    )
    damaged = copy.deepcopy(exported)
    if corrupt == "missing_parameter":
        damaged["parameters"].pop("Model.Second")
    elif corrupt == "wrong_manifest":
        damaged["manifest"][0]["parameter"]["fqn"] = "Wrong.First"
    elif corrupt == "wrong_shard":
        damaged["parameters"]["Model.First"]["shard"] = copy.deepcopy(damaged["parameters"]["Model.Second"]["shard"])
    elif corrupt == "wrong_policy":
        damaged["policy"]["stochastic_round"] = True
    elif corrupt == "wrong_group_options":
        damaged["parameters"]["Model.First"]["group_options"]["lr"] = 9.0
    elif corrupt == "unknown_state":
        damaged["parameters"]["Model.First"]["state"]["exp_avg"] = torch.ones(4)
    elif corrupt == "bad_state_geometry":
        damaged["parameters"]["Model.First"]["state"]["m_magnitude"] = torch.ones(2, 1)
    elif corrupt == "counter_ahead":
        damaged["parameters"]["Model.First"]["state"]["step"] = (
            damaged["common"]["gefen_global_step"] + 1
        )
    elif corrupt == "secondary_counter_ahead":
        damaged["common"]["gefen_global_step"] = 100
        damaged["parameters"]["Model.First"]["state"]["step"] = 1
        damaged["parameters"]["Model.First"]["state"]["vmean_step"] = 2
    else:
        damaged["format_version"] = True
    before = _snapshot(target)

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        target.import_canonical_state(damaged)

    _assert_snapshot_identity(target, before)


def test_import_rejects_initialized_period_that_violates_forced_policy():
    source_parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    source = Gefen(
        [("p", source_parameter)],
        fused=False,
        factored_v_2d=False,
        force_1d_period_one=True,
    )
    identity = ParameterIdentity("Model.P", (8,))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(source, ((source_parameter, shard),), manifest)
    source_parameter.grad = torch.arange(1, 9, dtype=torch.float32)
    source.step()
    damaged = source.export_canonical_state()
    parameter_state = damaged["parameters"]["Model.P"]["state"]
    parameter_state["automatic_period"] = 2
    parameter_state["m_codebook"] = torch.zeros(4, 2, dtype=torch.uint8)
    parameter_state["m_magnitude"] = torch.ones(4, 1, dtype=torch.float32)
    parameter_state["vmean"] = torch.ones(4, 1, dtype=torch.float32)

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target = Gefen(
        [("target", target_parameter)],
        fused=False,
        factored_v_2d=False,
        force_1d_period_one=True,
    )
    _finalize(target, ((target_parameter, shard),), manifest)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="violates period-one policy"):
        target.import_canonical_state(damaged)
    _assert_snapshot_identity(target, before)


def test_codebook_initialized_before_first_step_round_trips_canonically():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    source = Gefen([("p", parameter)], fused=False, factored_v_2d=False)
    identity = ParameterIdentity("Model.P", (8,))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(source, ((parameter, shard),), manifest)
    source._resolve_automatic_period = lambda *args: 4
    parameter.grad = torch.arange(1, 9, dtype=torch.float32)
    assert source.initialize_codebook()
    assert set(source.state[parameter]) == {"name", "automatic_period"}

    target_parameter = torch.nn.Parameter(parameter.detach().clone())
    target = Gefen([("target", target_parameter)], fused=False, factored_v_2d=False)
    _finalize(target, ((target_parameter, shard),), manifest)
    target.import_canonical_state(source.export_canonical_state())

    assert target._gefen_global_step == 0
    assert torch.equal(target._gefen_codebook, source._gefen_codebook)
    assert target.state[target_parameter] == {
        "name": "target",
        "automatic_period": 4,
    }


def test_flat_fragment_is_fqn_keyed_but_rejects_another_member_slice():
    group = ProcessGroupIdentity("flat", ("rank:0", "rank:1"))
    identity = ParameterIdentity("Model.Flat", (8,))
    shards = (
        _flat(identity, group, "rank:0", 0, 4),
        _flat(identity, group, "rank:1", 4, 4),
    )
    manifest = ShardingManifest(shards)
    source_parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    source = Gefen([("flat", source_parameter)], fused=False, factored_v_2d=False)
    _finalize(source, ((source_parameter, shards[0]),), manifest)
    source._resolve_automatic_period = lambda *args: 4
    source_parameter.grad = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source.step()
    exported = source.export_canonical_state()
    support = next(
        item
        for item in source.optimizer_contract().capabilities.checkpoints
        if item.transport is CheckpointTransport.CANONICAL_LOCAL
    )
    assert support.same_topology == frozenset({ParameterLayout.FLATTENED_ELEMENT_SHARD})

    target_parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    target = Gefen([("flat", target_parameter)], fused=False, factored_v_2d=False)
    _finalize(target, ((target_parameter, shards[1]),), manifest)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="shard"):
        target.import_canonical_state(exported)
    _assert_snapshot_identity(target, before)


def test_muon_replicated_supports_canonical_state_but_whole_owner_does_not():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = GefenMuon([("matrix", parameter)], fused=False)
    identity = ParameterIdentity("Model.Matrix", (2, 2))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    assert optimizer.optimizer_contract().capabilities.canonical_state_io

    group = ProcessGroupIdentity("owner", ("rank:0",))
    owner_shard = ShardIdentity(
        identity,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
        LogicalSlice.full(identity),
        process_group=group,
        local_member="rank:0",
        owner="rank:0",
        placements=(
            ShardPlacement(
                "dp",
                PlacementKind.WHOLE_PARAMETER_OWNER,
                0,
                1,
            ),
        ),
    )
    owner_parameter = torch.nn.Parameter(torch.ones(2, 2))
    owner = GefenMuon([("matrix", owner_parameter)], fused=False)
    _finalize(owner, ((owner_parameter, owner_shard),), ShardingManifest((owner_shard,)))
    assert not owner.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="supported finalized"):
        owner.export_canonical_state()


def test_initialized_normuon_requires_its_authoritative_pair_and_continues_exactly():
    source_parameter = torch.nn.Parameter(
        torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    )
    source = GefenMuon(
        [("matrix", source_parameter)],
        fused=False,
        normuon=True,
    )
    identity = ParameterIdentity("Model.Matrix", (2, 2))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(source, ((source_parameter, shard),), manifest)
    source._resolve_automatic_period = lambda *args: 4
    source_parameter.grad = torch.tensor([[0.5, -1.0], [1.5, -2.0]])
    source.step()
    exported = source.export_canonical_state()
    assert {"normuon_v", "normuon_step"}.issubset(
        exported["parameters"]["Model.Matrix"]["state"]
    )

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target = GefenMuon(
        [("target", target_parameter)],
        fused=False,
        normuon=True,
    )
    _finalize(target, ((target_parameter, shard),), manifest)
    damaged = copy.deepcopy(exported)
    damaged_state = damaged["parameters"]["Model.Matrix"]["state"]
    damaged_state.pop("normuon_v")
    damaged_state.pop("normuon_step")
    before = _snapshot(target)
    with pytest.raises(ValueError, match="NorMuon state is incomplete"):
        target.import_canonical_state(damaged)
    _assert_snapshot_identity(target, before)

    target.import_canonical_state(exported)
    next_grad = torch.tensor([[-2.0, 0.25], [0.75, -1.25]])
    source_parameter.grad = next_grad.clone()
    target_parameter.grad = next_grad.clone()
    source.step()
    target.step()
    assert torch.equal(target_parameter, source_parameter)
    assert torch.equal(
        target.state[target_parameter]["normuon_v"],
        source.state[source_parameter]["normuon_v"],
    )


@pytest.mark.parametrize("foreign_state", ["block", "factored"])
def test_muon_rejects_plain_gefen_authoritative_state_variants(foreign_state):
    source_parameter = torch.nn.Parameter(torch.ones(2, 2))
    source = GefenMuon([("matrix", source_parameter)], fused=False)
    identity = ParameterIdentity("Model.Matrix", (2, 2))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(source, ((source_parameter, shard),), manifest)
    source._resolve_automatic_period = lambda *args: 4
    source_parameter.grad = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
    source.step()
    damaged = source.export_canonical_state()
    parameter_state = damaged["parameters"]["Model.Matrix"]["state"]
    if foreign_state == "block":
        parameter_state["vmean"] = torch.zeros_like(
            parameter_state["m_magnitude"]
        )
        parameter_state["vmean_step"] = 1
    else:
        parameter_state["v_row"] = torch.zeros(2, dtype=torch.float32)
        parameter_state["v_col"] = torch.zeros(2, dtype=torch.float32)
        parameter_state["factored_step"] = 1

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target = GefenMuon([("target", target_parameter)], fused=False)
    _finalize(target, ((target_parameter, shard),), manifest)
    before = _snapshot(target)
    with pytest.raises(ValueError, match="declared state variant"):
        target.import_canonical_state(damaged)
    _assert_snapshot_identity(target, before)


def test_unsupported_custom_state_or_callable_policy_disables_canonical_claim():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    optimizer = GefenMuon(
        [("matrix", parameter)],
        fused=False,
    )
    identity = ParameterIdentity("Model.Matrix", (2, 2))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    optimizer.param_groups[0]["custom_callable"] = lambda: None
    assert not optimizer.optimizer_contract().capabilities.canonical_state_io

    plain_parameter = torch.nn.Parameter(torch.ones(4))
    plain = Gefen([("p", plain_parameter)], fused=False)
    plain_identity = ParameterIdentity("Model.P", (4,))
    plain_shard = _replicated(plain_identity)
    _finalize(
        plain,
        ((plain_parameter, plain_shard),),
        ShardingManifest((plain_shard,)),
    )
    plain.state[plain_parameter]["custom_tensor"] = torch.ones(1)
    assert not plain.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        plain.export_canonical_state()

    class TensorSubclass(torch.Tensor):
        pass

    subclass_parameter = torch.nn.Parameter(torch.ones(4))
    subclassed = Gefen([("p", subclass_parameter)], fused=False)
    subclass_identity = ParameterIdentity("Model.Subclass", (4,))
    subclass_shard = _replicated(subclass_identity)
    _finalize(
        subclassed,
        ((subclass_parameter, subclass_shard),),
        ShardingManifest((subclass_shard,)),
    )
    subclassed.param_groups[0]["custom_tensor"] = torch.ones(1).as_subclass(
        TensorSubclass
    )
    assert not subclassed.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        subclassed.export_canonical_state()

    subclassed.param_groups[0].pop("custom_tensor")
    with torch.no_grad():
        nested = torch.nested.nested_tensor(
            [torch.ones(2), torch.ones(3)]
        )
    subclassed.param_groups[0]["custom_nested"] = nested
    assert not subclassed.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        subclassed.export_canonical_state()

    subclassed.param_groups[0].pop("custom_nested")
    subclassed.param_groups[0]["custom_nan"] = torch.tensor([float("nan")])
    assert not subclassed.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        subclassed.export_canonical_state()

    subclassed.param_groups[0].pop("custom_nan")
    if hasattr(torch, "float8_e4m3fn"):
        subclassed.param_groups[0]["custom_float8"] = torch.ones(
            2, dtype=torch.float8_e4m3fn
        )
        assert not subclassed.optimizer_contract().capabilities.canonical_state_io
        with pytest.raises(RuntimeError, match="primitive algorithm policy"):
            subclassed.export_canonical_state()
        subclassed.param_groups[0].pop("custom_float8")

    subclassed.param_groups[0]["custom_conjugate"] = torch.tensor(
        [1.0 + 2.0j]
    ).conj()
    assert subclassed.optimizer_contract().capabilities.canonical_state_io
    prepared = subclassed.prepare_canonical_state_import(
        subclassed.export_canonical_state()
    )
    subclassed.commit_canonical_state_import(prepared)


def test_inference_mode_tensor_lr_has_a_stable_canonical_freshness_token():
    with torch.inference_mode():
        learning_rate = torch.tensor(1e-3)
    parameter = torch.nn.Parameter(torch.ones(4))
    optimizer = Gefen(
        [("p", parameter)],
        lr=learning_rate,
        fused=False,
        factored_v_2d=False,
    )
    identity = ParameterIdentity("Model.P", (4,))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    exported = optimizer.export_canonical_state()
    groups = optimizer.param_groups
    group = optimizer.param_groups[0]

    prepared = optimizer.prepare_canonical_state_import(exported)
    optimizer.commit_canonical_state_import(prepared)
    assert optimizer.param_groups is groups
    assert optimizer.param_groups[0] is group
    assert optimizer.param_groups[0]["lr"] is learning_rate
    assert optimizer.defaults["lr"] is learning_rate


def test_prepared_import_detects_storage_level_live_state_mutation():
    optimizer, _, _, _, _, _ = _two_parameter_source()
    prepared = optimizer.prepare_canonical_state_import(
        optimizer.export_canonical_state()
    )
    magnitude = next(
        state["m_magnitude"]
        for state in optimizer.state.values()
        if "m_magnitude" in state
    )
    version = magnitude._version
    magnitude.numpy()[0, 0] += 7.0
    assert magnitude._version == version
    before = _snapshot(optimizer)

    with pytest.raises(RuntimeError, match="changed after"):
        optimizer.commit_canonical_state_import(prepared)
    _assert_snapshot_identity(optimizer, before)


def test_nonfinite_authoritative_state_disables_the_export_capability():
    optimizer, first, _, _, _, _ = _two_parameter_source()
    optimizer.state[first]["vmean"].fill_(float("inf"))

    assert not optimizer.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        optimizer.export_canonical_state()


def test_invalid_live_counter_relationship_disables_the_export_capability():
    optimizer, first, _, _, _, _ = _two_parameter_source()
    optimizer.state[first]["step"] = optimizer._gefen_global_step + 1

    assert not optimizer.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        optimizer.export_canonical_state()


def test_canonical_policy_is_constructor_validated_and_capability_checked():
    parameter = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(TypeError, match="codebook_refresh_every must be an integer"):
        Gefen(
            [("p", parameter)],
            fused=False,
            codebook_refresh_every=float("nan"),
        )

    optimizer = Gefen([("p", parameter)], fused=False)
    identity = ParameterIdentity("Model.P", (4,))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    optimizer._codebook_refresh_every = float("nan")
    assert not optimizer.optimizer_contract().capabilities.canonical_state_io
    with pytest.raises(RuntimeError, match="primitive algorithm policy"):
        optimizer.export_canonical_state()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_capturable_import_requires_a_fresh_target_before_graph_state_exists():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32, device="cuda"))
    optimizer = Gefen(
        [("p", parameter)],
        fused=False,
        factored_v_2d=False,
        capturable=True,
    )
    identity = ParameterIdentity("Model.P", (8,))
    shard = _replicated(identity)
    _finalize(optimizer, ((parameter, shard),), ShardingManifest((shard,)))
    optimizer._resolve_automatic_period = lambda *args: 4
    parameter.grad = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    optimizer.step()

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        optimizer.step()
    exported = optimizer.export_canonical_state()
    before = _snapshot(optimizer)
    with pytest.raises(RuntimeError, match="requires a fresh target"):
        optimizer.prepare_canonical_state_import(exported)
    _assert_snapshot_identity(optimizer, before)

    fresh_parameter = torch.nn.Parameter(parameter.detach().clone())
    fresh = Gefen(
        [("p", fresh_parameter)],
        fused=False,
        factored_v_2d=False,
        capturable=True,
    )
    _finalize(fresh, ((fresh_parameter, shard),), ShardingManifest((shard,)))
    fresh.import_canonical_state(exported)
    assert fresh._device_gefen_global_step() == exported["common"][
        "gefen_global_step"
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_canonical_cpu_fragment_imports_state_to_cuda_parameter_devices():
    source_parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    source = Gefen([("p", source_parameter)], fused=False, factored_v_2d=False)
    identity = ParameterIdentity("Model.P", (8,))
    shard = _replicated(identity)
    manifest = ShardingManifest((shard,))
    _finalize(source, ((source_parameter, shard),), manifest)
    source._resolve_automatic_period = lambda *args: 4
    source_parameter.grad = torch.arange(1, 9, dtype=torch.float32)
    source.step()
    exported = source.export_canonical_state()
    assert all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for value in exported["parameters"]["Model.P"]["state"].values()
    )

    target_parameter = torch.nn.Parameter(source_parameter.detach().cuda())
    target = Gefen([("p", target_parameter)], fused=False, factored_v_2d=False)
    _finalize(target, ((target_parameter, shard),), manifest)
    target.import_canonical_state(exported)
    assert all(
        not torch.is_tensor(value) or value.device.type == "cuda" for value in target.state[target_parameter].values()
    )
