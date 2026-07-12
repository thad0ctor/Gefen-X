"""CPU tests for the Muon-family DeepSpeed ZeRO fail-fast guards.

DeepSpeed ZeRO (stage 1/2/3) wraps the client optimizer, replaces every
``param_groups[*]["params"]`` with flattened 1-D fp32 partitions (bypassing
``add_param_group``), and relies on torch's defaultdict-like
``optimizer.state`` auto-create when it touches ``optimizer.state[<flat
partition>]`` during the first ``backward()``. Muon's 2D Newton-Schulz
orthogonalization fundamentally cannot step a flat partition, so the Muon
family must fail loudly and self-explanatorily instead of:

* GefenMuon: a cryptic "gradient must be a 2D matrix for <stale name>" at the
  first step (the stale name comes from ``param_names`` that ZeRO left behind).
* GefenMuonHybrid: a bare tensor-keyed ``KeyError`` from deep inside DeepSpeed
  ``stage_1_and_2.get_flat_partition`` (the hybrid's merged ``.state`` was a
  plain dict rebuilt per access, so DeepSpeed's auto-create write was lost).

These tests simulate the exact ZeRO access patterns on CPU; the real
single-process ZeRO-2 runs were validated on GPU separately.
"""
import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon, GefenMuonHybrid


def _muon_named_params():
    torch.manual_seed(0)
    return [
        ("blocks.0.attn.w", nn.Parameter(torch.randn(8, 16))),
        ("blocks.0.mlp.w", nn.Parameter(torch.randn(16, 8))),
    ]


def _backup_named_params():
    torch.manual_seed(1)
    return [
        ("embed.weight", nn.Parameter(torch.randn(12, 8))),
        ("norm.weight", nn.Parameter(torch.randn(8))),
    ]


def _zero_style_flat_swap(groups):
    """Mirror DeepSpeed ZeRO-1/2: replace each group's params with one fresh
    flat 1-D fp32 partition tensor carrying a gradient (param_names, state,
    and everything else are left untouched, exactly as DeepSpeed leaves them).
    """
    flats = []
    for group in groups:
        numel = sum(p.numel() for p in group["params"])
        flat = nn.Parameter(torch.zeros(numel, dtype=torch.float32))
        flat.grad = torch.full_like(flat, 1e-3)
        group["params"] = [flat]
        flats.append(flat)
    return flats


# --------------------------- GefenMuon guard --------------------------------


def test_gefen_muon_zero_flat_swap_step_raises_zero_error():
    opt = GefenMuon(
        _muon_named_params(), lr=1e-3, fused=False, adjust_lr_fn="match_rms_adamw"
    )
    _zero_style_flat_swap(opt.param_groups)
    with pytest.raises(ValueError, match="DeepSpeed ZeRO") as excinfo:
        opt.step()
    msg = str(excinfo.value)
    # The message must carry the full diagnosis and the way out.
    assert "1-D fp32 partitions" in msg
    assert "cannot be applied to a flat partition" in msg
    assert "plain Gefen" in msg
    # And must NOT be the old cryptic stale-name gradient error.
    assert "gradient must be a 2D matrix" not in msg


def test_gefen_muon_guard_fires_even_without_grads():
    # DeepSpeed variants that call an empty warm-up step must hit the guard
    # too: it checks the parameter itself, not the gradient.
    opt = GefenMuon(_muon_named_params(), lr=1e-3, fused=False)
    flats = _zero_style_flat_swap(opt.param_groups)
    for flat in flats:
        flat.grad = None
    with pytest.raises(ValueError, match="DeepSpeed ZeRO"):
        opt.step()


def test_gefen_muon_normal_step_unaffected_by_guard():
    named = _muon_named_params()
    opt = GefenMuon(named, lr=1e-3, fused=False, adjust_lr_fn="match_rms_adamw")
    gen = torch.Generator().manual_seed(7)
    for _, p in named:
        p.grad = torch.randn(p.shape, generator=gen) * 1e-2
    before = [p.detach().clone() for _, p in named]
    opt.step()
    assert all(
        not torch.equal(b, p.detach()) for b, (_, p) in zip(before, named)
    )


# ------------------------- GefenMuonHybrid guards ---------------------------


def _hybrid(**kwargs):
    kwargs.setdefault("lr", 1e-3)
    kwargs.setdefault("fused", False)
    return GefenMuonHybrid(_muon_named_params(), _backup_named_params(), **kwargs)


def test_hybrid_state_unknown_key_read_raises_zero_error():
    opt = _hybrid()
    # Exact DeepSpeed stage_1_and_2.get_flat_partition pattern: a truthiness
    # read of optimizer.state[<fresh flat fp32 partition>], expecting torch's
    # defaultdict auto-create.
    flatten_copy = nn.Parameter(torch.zeros(123, dtype=torch.float32))
    with pytest.raises(KeyError, match="DeepSpeed ZeRO") as excinfo:
        bool(opt.state[flatten_copy])
    msg = str(excinfo.value)
    assert "GefenMuonHybrid" in msg
    assert "plain Gefen" in msg


def test_hybrid_state_unknown_key_write_raises_zero_error():
    opt = _hybrid()
    flatten_copy = nn.Parameter(torch.zeros(64, dtype=torch.float32))
    with pytest.raises(KeyError, match="DeepSpeed ZeRO"):
        opt.state[flatten_copy] = {}


def test_hybrid_state_rejects_flat_partition_even_after_group_swap():
    # Real ZeRO-2 order of operations: deepspeed.initialize FIRST swaps the
    # (shared) group dicts to the flat partitions, and only then, inside the
    # first backward, reads optimizer.state[<flat partition>]. Ownership is
    # frozen at construction, so the swapped-in flat must still be rejected --
    # a live group scan would treat it as a known param, auto-create state,
    # and push the failure to a later bare AttributeError inside engine.step.
    opt = _hybrid()
    flats = _zero_style_flat_swap(opt.param_groups)
    with pytest.raises(KeyError, match="DeepSpeed ZeRO"):
        bool(opt.state[flats[0]])


def test_hybrid_param_groups_assignment_raises_zero_error():
    # DeepSpeed ZeRO's _optimizer_step does `optimizer.param_groups = [...]`;
    # without an explaining setter that dies as a bare "property has no
    # setter" AttributeError.
    opt = _hybrid()
    with pytest.raises(TypeError, match="DeepSpeed ZeRO"):
        opt.param_groups = [{"params": []}]


def test_hybrid_zero_param_swap_step_raises_zero_error():
    # Even if a wrapper writes its state elsewhere and reaches step(), the
    # muon child must refuse the swapped flat partitions loudly. ZeRO mutates
    # the group dicts it got from optimizer.param_groups; the hybrid property
    # returns the children's live group dicts, so mutate through it.
    opt = _hybrid()
    _zero_style_flat_swap(opt.param_groups)
    with pytest.raises(ValueError, match="DeepSpeed ZeRO"):
        opt.step()


# ------------------- GefenMuonHybrid ZeRO-3 ctor guard ----------------------
# ZeRO stage 3 (zero.Init) partitions params to 0-size 1-D placeholders BEFORE
# optimizer construction, so the hybrid's 2D split finds zero Muon candidates
# and would silently build a backup-only hybrid (plain Gefen/AdamW at
# backup_lr, no Muon half, no warning) that trains mechanically but is not the
# configured optimizer.


def test_hybrid_zero3_placeholder_params_ctor_raises():
    placeholders = [
        ("model.embed.weight", nn.Parameter(torch.empty(0))),
        ("model.layers.0.mlp.w", nn.Parameter(torch.empty(0))),
        ("model.norm.weight", nn.Parameter(torch.empty(0))),
    ]
    with pytest.raises(ValueError, match="ZeRO stage 3") as excinfo:
        GefenMuonHybrid(placeholders, lr=1e-3, fused=False)
    msg = str(excinfo.value)
    assert "zero-numel" in msg
    assert "backup-only" in msg
    assert "plain Gefen" in msg


def test_hybrid_zero3_placeholder_two_list_form_raises():
    with pytest.raises(ValueError, match="ZeRO stage 3"):
        GefenMuonHybrid(
            [],
            [("model.norm.weight", nn.Parameter(torch.empty(0)))],
            lr=1e-3,
            fused=False,
        )


def test_hybrid_backup_only_real_1d_model_still_constructs():
    # A genuinely 1D-only model (real storage, no zero-numel placeholders)
    # must keep today's behavior: a backup-only hybrid with muon=None.
    named = [
        ("norm.weight", nn.Parameter(torch.randn(8))),
        ("norm.bias", nn.Parameter(torch.randn(8))),
    ]
    opt = GefenMuonHybrid([], named, lr=1e-3, fused=False)
    assert opt.muon is None
    assert opt.backup is not None


# ----------------- GefenMuonHybrid .state live-view semantics ---------------


def test_hybrid_state_top_level_write_routes_to_owning_child():
    opt = _hybrid()
    p = opt.backup.param_groups[0]["params"][0]
    replacement = dict(opt.state[p], custom_marker=7)
    opt.state[p] = replacement
    # Persists in the owning child and is visible through a FRESH view.
    assert opt.backup.state[p] is replacement
    assert opt.state[p]["custom_marker"] == 7


def test_hybrid_state_autocreates_for_known_param_and_persists():
    # An AdamW backup has NO state entries before the first step, so a read
    # must auto-create in the owning child (torch defaultdict semantics)
    # rather than KeyError or vanish with the throwaway view.
    opt = _hybrid(backup_optimizer="adamw")
    p = opt.backup.param_groups[0]["params"][0]
    assert p not in dict(opt.backup.state)
    entry = opt.state[p]
    entry["warmup_flag"] = True
    assert opt.backup.state[p] is entry
    assert opt.state[p]["warmup_flag"] is True


def test_hybrid_state_merged_read_view_preserved():
    opt = _hybrid()
    merged = opt.state
    assert isinstance(merged, dict)
    child_keys = set(opt.muon.state.keys()) | set(opt.backup.state.keys())
    assert set(merged.keys()) == child_keys
    # Values are the children's LIVE per-param dicts (inner mutation has
    # always persisted; identity must hold so it keeps persisting).
    for p, st in merged.items():
        child = opt.muon if any(
            p is q for g in opt.muon.param_groups for q in g["params"]
        ) else opt.backup
        assert child.state[p] is st
    # Membership and .get never auto-create or raise.
    ghost = nn.Parameter(torch.zeros(3))
    assert ghost not in merged
    assert merged.get(ghost) is None


def test_plain_gefen_state_untouched_by_hybrid_guard():
    # Regression guard for the supported DeepSpeed path: plain Gefen keeps
    # torch's real defaultdict state, so the ZeRO auto-create write pattern
    # must keep working there.
    named = _backup_named_params()
    opt = Gefen(named, lr=1e-3, fused=False)
    flatten_copy = nn.Parameter(torch.zeros(31, dtype=torch.float32))
    assert not opt.state[flatten_copy]  # defaultdict auto-create, no raise
    opt.state[flatten_copy]["momentum_buffer"] = torch.zeros(31)
    assert "momentum_buffer" in opt.state[flatten_copy]
