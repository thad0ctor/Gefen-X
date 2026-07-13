"""Per-step layout/offload guard cost: fast-path tokens, dedupe, boundaries.

The finalized-layout guards run on every ``step()``; these tests pin the
contract that steady-state steps reuse one cached forensic verdict (an
O(local params) identity token check) while every legitimate mutating API and
every boundary operation (checkpoint prepare/commit, rebinding, state
movement/offload, contract readiness) still runs the complete O(params x
world) forensic rebuild. Detection-before-mutation is preserved: anything a
closure or adapter can corrupt through the public optimizer containers still
raises at the step guard itself.
"""

import time

import pytest
import torch

from gefen import (
    Gefen,
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


def _replicated_shard(fqn, shape):
    identity = ParameterIdentity(fqn, shape)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
    )


def _finalized_replicated_optimizer(count=3):
    parameters = [
        torch.nn.Parameter(torch.full((4,), float(index + 1)))
        for index in range(count)
    ]
    optimizer = Gefen(
        [("weight{}".format(index), parameter) for index, parameter in enumerate(parameters)],
        fused=False,
        factored_v_2d=False,
    )
    shards = tuple(
        _replicated_shard("Model.Weight{}".format(index), (4,))
        for index in range(count)
    )
    optimizer.post_sharding(
        tuple(
            ParameterRebinding(parameter, parameter, shard)
            for parameter, shard in zip(parameters, shards)
        ),
        manifest=ShardingManifest(shards),
    )
    return optimizer, parameters


def _flat_sharded_optimizer(members_count, params_count, local_length=4):
    members = tuple("m{:04d}".format(index) for index in range(members_count))
    group = ProcessGroupIdentity("data_parallel", members)
    local_member = members[0]
    manifest_shards = []
    local_shards = []
    for index in range(params_count):
        identity = ParameterIdentity(
            "Model.Block{}.Weight".format(index), (members_count * local_length,)
        )
        offset = 0
        for coordinate, member in enumerate(members):
            shard = ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, local_length),
                placements=(
                    ShardPlacement(
                        "data_parallel",
                        PlacementKind.FLAT_SHARD,
                        coordinate,
                        members_count,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
            manifest_shards.append(shard)
            if member == local_member:
                local_shards.append(shard)
            offset += local_length
    manifest = ShardingManifest(tuple(manifest_shards))
    parameters = [
        torch.nn.Parameter(torch.randn(local_length)) for _ in range(params_count)
    ]
    optimizer = Gefen(
        [
            ("model.block{}.weight".format(index), parameters[index])
            for index in range(params_count)
        ],
        fused=False,
        factored_v_2d=False,
    )
    optimizer.post_sharding(
        tuple(
            ParameterRebinding(parameters[index], parameters[index], local_shards[index])
            for index in range(params_count)
        ),
        manifest=manifest,
    )
    return optimizer, parameters


def _count_full_layout_passes(monkeypatch):
    calls = {"count": 0}
    original = Gefen._finalized_binding_layout_matches_full

    def counted(self):
        calls["count"] += 1
        return original(self)

    monkeypatch.setattr(Gefen, "_finalized_binding_layout_matches_full", counted)
    return calls


def _count_manifest_digest_computes(monkeypatch):
    calls = {"count": 0}
    original = Gefen._compute_codebook_manifest_fingerprint

    def counted(self, manifest):
        calls["count"] += 1
        return original(self, manifest)

    monkeypatch.setattr(
        Gefen, "_compute_codebook_manifest_fingerprint", counted
    )
    return calls


def _step_with_grads(optimizer, parameters):
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.5)
    optimizer.step()


def test_steady_state_step_runs_no_full_layout_forensics(monkeypatch):
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)

    calls = _count_full_layout_passes(monkeypatch)
    for _ in range(3):
        _step_with_grads(optimizer, parameters)
    assert calls["count"] == 0


def test_first_step_after_finalization_validates_fully_once(monkeypatch):
    optimizer, parameters = _finalized_replicated_optimizer()
    calls = _count_full_layout_passes(monkeypatch)
    _step_with_grads(optimizer, parameters)
    assert calls["count"] == 1


def test_manifest_digest_is_computed_once_at_finalization(monkeypatch):
    calls = _count_manifest_digest_computes(monkeypatch)
    optimizer, parameters = _finalized_replicated_optimizer()
    finalize_computes = calls["count"]
    assert finalize_computes >= 1

    first = optimizer._codebook_manifest_fingerprint()
    second = optimizer._codebook_manifest_fingerprint()
    assert calls["count"] == finalize_computes
    assert first == second == optimizer._compute_codebook_manifest_fingerprint(
        optimizer._gefen_sharding_manifest
    )

    _step_with_grads(optimizer, parameters)
    _step_with_grads(optimizer, parameters)
    assert calls["count"] > finalize_computes  # the deliberate compare above
    steady = calls["count"]
    optimizer._codebook_manifest_fingerprint()
    assert calls["count"] == steady


def test_closure_layout_mutation_is_detected_with_a_warm_verdict():
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)  # warm the cached verdict

    rogue = torch.nn.Parameter(torch.ones(4) * 7)
    rogue_before = rogue.detach().clone()

    def mutate_layout():
        optimizer.param_groups[0]["params"][0] = rogue
        rogue.grad = torch.ones_like(rogue)
        return torch.tensor(1.0)

    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.step(mutate_layout)
    assert torch.equal(rogue, rogue_before)


@pytest.mark.parametrize(
    "corruption",
    ["local_bindings", "logical_slots", "group_params", "name_cache", "state_name"],
)
def test_step_detects_public_and_replaced_layout_tampering_after_warmup(corruption):
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)  # warm the cached verdict

    if corruption == "local_bindings":
        optimizer._gefen_local_shard_bindings = tuple(
            reversed(optimizer._gefen_local_shard_bindings)
        )
    elif corruption == "logical_slots":
        optimizer._gefen_logical_slots = tuple(
            reversed(optimizer._gefen_logical_slots)
        )
    elif corruption == "group_params":
        optimizer.param_groups[0]["params"].reverse()
    elif corruption == "name_cache":
        optimizer._param_names[parameters[0]] = "changed"
    else:
        optimizer.state[parameters[0]]["name"] = "changed"

    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        _step_with_grads(optimizer, parameters)


def test_private_registry_inplace_tamper_is_detected_at_the_next_boundary():
    # In-place value swaps inside the private finalized registries preserve
    # every container identity the per-step tokens can see; per the documented
    # contract they are detected by the next full forensic boundary
    # (checkpoint prepare here, and contract readiness below) instead of the
    # next step.
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)

    first, second = parameters[0], parameters[1]
    bindings = optimizer._gefen_shard_bindings
    bindings[first], bindings[second] = bindings[second], bindings[first]

    _step_with_grads(optimizer, parameters)  # fast tokens cannot see this
    assert not optimizer.optimizer_contract().capabilities.stable_shard_identity
    with pytest.raises(RuntimeError, match="changed outside post_sharding"):
        optimizer.state_dict()


def test_mutating_apis_bump_the_layout_version_and_force_revalidation(monkeypatch):
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)
    checkpoint = optimizer.state_dict()

    calls = _count_full_layout_passes(monkeypatch)

    version = optimizer._gefen_layout_version
    optimizer.move_state_()
    assert optimizer._gefen_layout_version == version + 1
    assert optimizer._gefen_layout_forensics_verdict is None

    _step_with_grads(optimizer, parameters)
    after_move = calls["count"]
    assert after_move >= 1
    _step_with_grads(optimizer, parameters)
    assert calls["count"] == after_move  # steady again

    version = optimizer._gefen_layout_version
    optimizer.load_state_dict(checkpoint)
    assert optimizer._gefen_layout_version > version
    assert optimizer._gefen_layout_forensics_verdict is None


def test_state_offload_step_scan_runs_on_every_step(monkeypatch):
    # The offload readiness scan is intentionally NOT cached: the per-parameter
    # offloaded state tensors are legitimately replaced on every step, so a
    # cached verdict cannot represent them, and a token-preserving in-place
    # corruption of a later parameter would otherwise slip past step entry and
    # only be caught mid-step, after earlier parameters were already mutated.
    # The scan is O(local params) and must run before any parameter is staged.
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)

    calls = {"count": 0}

    def counted(self, *, require_cpu_state, allow_poisoned=False):
        calls["count"] += 1
        return None

    monkeypatch.setattr(Gefen, "_state_offload_rejection_reason", counted)
    monkeypatch.setattr(
        Gefen,
        "state_offload_active",
        property(lambda self: True),
    )

    optimizer._assert_state_offload_step_ready()
    optimizer._assert_state_offload_step_ready()
    optimizer._assert_state_offload_step_ready()
    assert calls["count"] == 3

    optimizer._gefen_state_offload_poisoned = True
    with pytest.raises(RuntimeError, match="poisoned"):
        optimizer._assert_state_offload_step_ready()


def test_offload_scan_re_rejects_on_every_step(monkeypatch):
    optimizer, parameters = _finalized_replicated_optimizer()

    calls = {"count": 0}

    def counted(self, *, require_cpu_state, allow_poisoned=False):
        calls["count"] += 1
        return "authoritative offloaded tensors must be tight CPU tensors"

    monkeypatch.setattr(Gefen, "_state_offload_rejection_reason", counted)
    monkeypatch.setattr(
        Gefen,
        "state_offload_active",
        property(lambda self: True),
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="cannot step"):
            optimizer._assert_state_offload_step_ready()
    assert calls["count"] == 2


def test_forensics_caches_stay_out_of_the_public_attribute_namespace():
    optimizer, parameters = _finalized_replicated_optimizer()
    _step_with_grads(optimizer, parameters)
    assert optimizer._gefen_layout_forensics_verdict is not None
    for name in Gefen.__slots__:
        assert name not in optimizer.__dict__


def test_warm_step_guards_are_far_cheaper_than_one_forensic_pass():
    # Relative timing with a deliberately generous margin: the old behavior
    # ran (at least) two full forensic passes inside every step, so the warm
    # per-step guard sequence must be much cheaper than even one full pass on
    # a moderately sized manifest. Identity-token checks are microseconds
    # while the full pass rehashes 96-member identities for every shard, so
    # the 10x assertion holds with orders of magnitude to spare.
    optimizer, parameters = _flat_sharded_optimizer(96, 60)
    _step_with_grads(optimizer, parameters)  # warm caches

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer._assert_state_offload_step_ready()
        optimizer._assert_finalized_binding_layout()
        optimizer._assert_runtime_codebook_process_group()
        optimizer._assert_state_offload_step_ready()
        optimizer._assert_finalized_binding_layout()
        optimizer._assert_runtime_codebook_process_group()
    warm_guard = (time.perf_counter() - start) / iterations

    start = time.perf_counter()
    for _ in range(iterations):
        assert optimizer._finalized_binding_layout_matches(full=True)
    full_pass = (time.perf_counter() - start) / iterations

    assert warm_guard * 10 < full_pass
