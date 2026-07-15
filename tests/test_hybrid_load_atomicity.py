"""Fail-before-mutation tests for GefenMuonHybrid.load_state_dict.

The composite load restores two children (the GefenMuon half and the backup
half). A rejection on *either* half must leave *both* live children
byte-for-byte untouched -- the same two-phase (stage-then-publish) contract
every other mutating entry point on this branch honours.

These tests pin the failure mode of the previous implementation, which committed
the muon child first and only then validated/loaded the backup, recovering via a
second full ``muon.load_state_dict(snapshot)`` reload. That reload could itself
raise (e.g. CUDA OOM re-staging every muon tensor), masking the real backup error
and leaving a half-loaded hybrid. They run on CPU (fused=False).
"""
import copy
import warnings

import pytest
import torch
import torch.nn as nn

from gefen import GefenMuonHybrid


class TinyLM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.hidden = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        return self.lm_head(self.norm(self.hidden(self.embed(idx))))


def _build(backup_optimizer="gefen"):
    torch.manual_seed(0)
    model = TinyLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid(
            model, lr=1e-3, fused=False, backup_optimizer=backup_optimizer
        )
    return opt, model


def _step(opt, model):
    idx = torch.randint(0, 32, (4, 6))
    tgt = torch.randint(0, 32, (4, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt.zero_grad()
        nn.functional.cross_entropy(
            model(idx).reshape(-1, 32), tgt.reshape(-1)
        ).backward()
        opt.step()


def _muon_param_and_state(opt):
    param = opt.muon.param_groups[0]["params"][0]
    state = opt.muon.state[param]
    tensors = {k: v for k, v in state.items() if torch.is_tensor(v)}
    assert tensors, "expected the stepped muon child to hold momentum tensors"
    return param, tensors


def _backup_with_mismatched_group_count(state_dict):
    """A backup half the live backup's loader rejects (extra param group)."""
    bad = copy.deepcopy(state_dict)
    bad["backup"]["param_groups"].append(
        copy.deepcopy(bad["backup"]["param_groups"][0])
    )
    return bad


def test_backup_rejection_leaves_muon_child_byte_for_byte_untouched():
    opt, model = _build()
    _step(opt, model)
    bad = _backup_with_mismatched_group_count(opt.state_dict())

    param, before = _muon_param_and_state(opt)
    before_values = {k: v.detach().clone() for k, v in before.items()}
    before_ids = {k: id(v) for k, v in before.items()}

    # Detect any reload of the muon child. The two-phase load validates the
    # backup half *before* publishing muon, so muon must never be loaded (nor
    # loaded-then-rolled-back, as the previous snapshot/reload path did).
    calls = {"n": 0}
    original_load = opt.muon.load_state_dict

    def counting_load(sd):
        calls["n"] += 1
        return original_load(sd)

    opt.muon.load_state_dict = counting_load

    with pytest.raises(ValueError, match="different number of parameter groups"):
        opt.load_state_dict(bad)

    assert calls["n"] == 0, "backup rejection must not reload the muon child"

    param, after = _muon_param_and_state(opt)
    for key, value in before_values.items():
        assert torch.equal(value, after[key]), f"muon state {key!r} changed value"
    # Deep value comparison is the untouched contract; object identity is an
    # extra witness that no restage happened (a rollback reload swaps objects).
    assert {k: id(v) for k, v in after.items()} == before_ids


def test_backup_rejection_surfaces_over_a_failing_rollback_reload():
    opt, model = _build()
    _step(opt, model)
    bad = _backup_with_mismatched_group_count(opt.state_dict())

    param, before = _muon_param_and_state(opt)
    before_values = {k: v.detach().clone() for k, v in before.items()}

    # Simulate the previous implementation's rollback masking: the muon child's
    # first load (the real one) succeeds, but the second (the rollback reload)
    # blows up -- as re-staging every muon tensor on CUDA can. The *original*
    # backup error must still be what surfaces, and muon must stay untouched.
    calls = {"n": 0}
    original_load = opt.muon.load_state_dict

    def boom_on_rollback(sd):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("rollback reload OOM")
        return original_load(sd)

    opt.muon.load_state_dict = boom_on_rollback

    with pytest.raises(ValueError, match="different number of parameter groups"):
        opt.load_state_dict(bad)

    param, after = _muon_param_and_state(opt)
    for key, value in before_values.items():
        assert torch.equal(value, after[key]), f"muon state {key!r} changed value"


def test_adamw_backup_rejection_leaves_muon_child_untouched():
    opt, model = _build(backup_optimizer="adamw")
    _step(opt, model)
    bad = _backup_with_mismatched_group_count(opt.state_dict())

    param, before = _muon_param_and_state(opt)
    before_values = {k: v.detach().clone() for k, v in before.items()}
    before_ids = {k: id(v) for k, v in before.items()}

    calls = {"n": 0}
    original_load = opt.muon.load_state_dict

    def counting_load(sd):
        calls["n"] += 1
        return original_load(sd)

    opt.muon.load_state_dict = counting_load

    with pytest.raises(ValueError, match="different number of parameter groups"):
        opt.load_state_dict(bad)

    assert calls["n"] == 0, "backup rejection must not reload the muon child"

    param, after = _muon_param_and_state(opt)
    for key, value in before_values.items():
        assert torch.equal(value, after[key]), f"muon state {key!r} changed value"
    assert {k: id(v) for k, v in after.items()} == before_ids


def _child_snapshot(child):
    """Byte-for-byte + object-identity snapshot of a child optimizer's state."""
    snap = {}
    for p, state in child.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                snap[(id(p), key)] = (value.detach().clone(), id(value))
    return snap


def _assert_child_unchanged(child, before):
    after = _child_snapshot(child)
    assert set(after) == set(before), "child gained/lost state entries"
    for key, (value, obj_id) in before.items():
        assert torch.equal(after[key][0], value), f"child state {key!r} changed value"
        assert after[key][1] == obj_id, f"child state {key!r} was re-staged"


def test_adamw_backup_missing_step_leaves_both_children_untouched():
    # torch AdamW.load_state_dict is NOT fail-before-mutation: a checkpoint whose
    # per-parameter state omits "step" installs the new state via
    # super().__setstate__ and only then raises KeyError("step"). The previous
    # code committed the live backup directly in that case, stranding a
    # half-loaded hybrid (untouched muon, corrupted backup). The foreign backup
    # must be staged on an isolated shadow too, so this rejection leaves BOTH
    # children byte-for-byte untouched. (The existing group-count test only
    # exercises torch's early, pre-mutation rejection.)
    opt, model = _build(backup_optimizer="adamw")
    _step(opt, model)
    _step(opt, model)

    bad = copy.deepcopy(opt.state_dict())
    backup_states = bad["backup"]["state"]
    first_key = next(iter(backup_states))
    assert "step" in backup_states[first_key]
    del backup_states[first_key]["step"]

    muon_before = _child_snapshot(opt.muon)
    backup_before = _child_snapshot(opt.backup)

    with pytest.raises(KeyError, match="step"):
        opt.load_state_dict(bad)

    _assert_child_unchanged(opt.muon, muon_before)
    _assert_child_unchanged(opt.backup, backup_before)


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_child_post_hook_failure_commits_both_children_first(backup_optimizer):
    # A load post-hook can raise. It must fire only after EVERY child's raw
    # state is committed -- otherwise a throwing muon post-hook would leave muon
    # advanced while the backup stays at the old step (half-loaded). Advance the
    # source further than the target so the committed step is distinguishable.
    source, source_model = _build(backup_optimizer=backup_optimizer)
    for _ in range(3):
        _step(source, source_model)
    checkpoint = source.state_dict()
    src_muon_step = next(
        float(v["step"]) for v in source.muon.state.values() if "step" in v
    )
    src_backup_step = next(
        float(v["step"]) for v in source.backup.state.values() if "step" in v
    )

    opt, model = _build(backup_optimizer=backup_optimizer)
    _step(opt, model)

    def boom(_optimizer):
        raise RuntimeError("post-hook boom")

    opt.muon.register_load_state_dict_post_hook(boom)

    with pytest.raises(RuntimeError, match="post-hook boom"):
        opt.load_state_dict(checkpoint)

    muon_step = next(float(v["step"]) for v in opt.muon.state.values() if "step" in v)
    backup_step = next(
        float(v["step"]) for v in opt.backup.state.values() if "step" in v
    )
    assert muon_step == src_muon_step, "muon child was not committed"
    assert backup_step == src_backup_step, (
        "backup child was left uncommitted while the muon post-hook ran -- "
        "half-loaded hybrid"
    )
