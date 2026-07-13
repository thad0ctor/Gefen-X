"""Per-step composite layout-guard cost for a finalized GefenMuonHybrid.

GefenMuonHybrid.step() calls its finalized-layout guard twice per step, and
each call used to run the full O(params) composite forensic rebuild (routing,
ownership, per-child manifest partitioning, local-binding sort). These tests
pin the contract that a steady-state finalized composite reuses one cached
verdict (an O(local params) identity-token check that folds in each child's
own cached fast-path verdict) instead of rebuilding, while every real
finalized-layout change -- an in-place child ``group['params']`` slot swap, a
replaced composite registry, a closure that tampers between the pre- and
post-closure guards -- is still detected before any state mutation.
"""

import pytest
import torch

from gefen import GefenMuonHybrid
from gefen.contracts import (
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ShardIdentity,
    ShardingManifest,
)
from gefen.rebinding import ParameterRebinding


def _replicated_shard(fqn, shape):
    identity = ParameterIdentity(fqn, shape)
    return ShardIdentity(
        identity,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(identity),
    )


def _finalized_hybrid():
    matrix = torch.nn.Parameter(torch.randn(2, 2))
    bias = torch.nn.Parameter(torch.randn(4))
    optimizer = GefenMuonHybrid(
        [("layer.weight", matrix)],
        [("layer.bias", bias)],
        lr=1e-3,
        fused=False,
        sharded_mode="distributed",
    )
    matrix_shard = _replicated_shard("Model.Layer.Weight", (2, 2))
    bias_shard = _replicated_shard("Model.Layer.Bias", (4,))
    optimizer.post_sharding(
        (
            ParameterRebinding(matrix, matrix, matrix_shard),
            ParameterRebinding(bias, bias, bias_shard),
        ),
        manifest=ShardingManifest((matrix_shard, bias_shard)),
    )
    return optimizer, matrix, bias


def _count_full_composite_rebuilds(monkeypatch):
    # The full composite rebuild is the only step-time caller of
    # _gefen_rebinding_children; the cached fast path never touches it. This
    # distinguishes a real rebuild in both the pre- and post-cache code.
    calls = {"count": 0}
    original = GefenMuonHybrid._gefen_rebinding_children

    def counted(self):
        calls["count"] += 1
        return original(self)

    monkeypatch.setattr(GefenMuonHybrid, "_gefen_rebinding_children", counted)
    return calls


def _step_with_grads(optimizer, params):
    for parameter in params:
        parameter.grad = torch.full_like(parameter, 0.5)
    optimizer.step()


def test_steady_state_finalized_hybrid_step_does_not_rebuild(monkeypatch):
    optimizer, matrix, bias = _finalized_hybrid()
    _step_with_grads(optimizer, (matrix, bias))  # warm the cached verdict

    calls = _count_full_composite_rebuilds(monkeypatch)
    for _ in range(3):
        _step_with_grads(optimizer, (matrix, bias))
    # Pre-cache each step's two guard calls each ran a full rebuild (6 total);
    # with the verdict cache a steady-state step rebuilds zero times.
    assert calls["count"] == 0


def test_first_finalized_hybrid_step_rebuilds_then_caches(monkeypatch):
    optimizer, matrix, bias = _finalized_hybrid()
    calls = _count_full_composite_rebuilds(monkeypatch)
    _step_with_grads(optimizer, (matrix, bias))
    # The first guard rebuilds once and caches; the second guard in the same
    # step reuses the warm verdict.
    assert calls["count"] == 1
    assert optimizer._hybrid_layout_forensics_verdict is not None


def test_finalized_hybrid_detects_child_param_slot_swap_with_warm_verdict():
    optimizer, matrix, bias = _finalized_hybrid()
    _step_with_grads(optimizer, (matrix, bias))  # warm the cached verdict
    assert optimizer._hybrid_layout_forensics_verdict is not None

    rogue = torch.nn.Parameter(torch.full((2, 2), 7.0))
    rogue_before = rogue.detach().clone()
    optimizer.muon.param_groups[0]["params"][0] = rogue

    with pytest.raises(RuntimeError, match="finalized parameter layout changed"):
        _step_with_grads(optimizer, (bias,))
    assert torch.equal(rogue, rogue_before)


def test_finalized_hybrid_detects_swapped_child_binding_with_warm_verdict():
    optimizer, matrix, bias = _finalized_hybrid()
    _step_with_grads(optimizer, (matrix, bias))  # warm the cached verdict

    # Swap a child's private finalized manifest for a foreign one after
    # finalization: the composite fast token captures the child manifest by
    # identity, so the swap forces a full rebuild that rejects the layout.
    optimizer.muon._gefen_sharding_manifest = optimizer.backup._gefen_sharding_manifest

    with pytest.raises(RuntimeError, match="finalized parameter layout changed"):
        _step_with_grads(optimizer, (matrix, bias))


def test_finalized_hybrid_rechecks_layout_after_closure():
    optimizer, matrix, bias = _finalized_hybrid()
    _step_with_grads(optimizer, (matrix, bias))  # warm the cached verdict

    rogue = torch.nn.Parameter(torch.full((2, 2), 3.0))
    rogue_before = rogue.detach().clone()
    matrix_before = matrix.detach().clone()

    def mutate_layout():
        # Tamper with a public child container between the pre- and
        # post-closure guards; the post-closure guard must catch it before any
        # child steps.
        optimizer.muon.param_groups[0]["params"][0] = rogue
        rogue.grad = torch.ones_like(rogue)
        bias.grad = torch.ones_like(bias)
        return torch.tensor(1.0)

    with pytest.raises(RuntimeError, match="finalized parameter layout changed"):
        optimizer.step(mutate_layout)
    assert torch.equal(rogue, rogue_before)
    assert torch.equal(matrix, matrix_before)


def test_composite_registry_replacement_is_detected_with_warm_verdict():
    optimizer, matrix, bias = _finalized_hybrid()
    _step_with_grads(optimizer, (matrix, bias))  # warm the cached verdict

    # Replacing the composite ownership registry preserves no container
    # identity the fast token recorded, so the next step guard rebuilds and
    # rejects.
    optimizer._state_param_owner = {}

    with pytest.raises(RuntimeError, match="finalized parameter layout changed"):
        _step_with_grads(optimizer, (matrix, bias))
