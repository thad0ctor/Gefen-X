"""CPU-only coverage for constructor validation and the pure math helpers.

These paths guard against silent misconfiguration (bad hyperparams, bad param
splits, double-stepped params) and are device-independent, so they run on
CPU-only CI.

Run: pytest tests/test_validation_cpu.py
"""
import math

import pytest
import torch
import torch.nn as nn

import gefen
from gefen import Gefen, GefenMuon, GefenMuonHybrid, split_params_for_muon, validate_split
from gefen.gefen_muon import (
    NS_SCHEDULES,
    _adjust_lr_ratio,
    _cautious_mask_,
    _normalize_ns_schedule,
    _zeropower_via_newtonschulz,
)
from gefen.params import is_muon_param


def _params_1d():
    return [nn.Parameter(torch.randn(8))]


def _params_2d(name="w"):
    return [(name, nn.Parameter(torch.randn(16, 16)))]


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------

def test_lazy_package_exports():
    # Every name in __all__ must resolve via the lazy __getattr__ (including
    # the "kernels" submodule, which needs its own import branch there).
    for name in gefen.__all__:
        assert getattr(gefen, name) is not None
    with pytest.raises(AttributeError):
        getattr(gefen, "NotAThing")


# ---------------------------------------------------------------------------
# Gefen constructor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1.0},
        {"betas": (1.0, 0.999)},
        {"betas": (0.9, 1.0)},
        {"weight_decay": -0.1},
        {"codebook_refresh_every": -1},
    ],
)
def test_gefen_rejects_bad_hyperparams(kwargs):
    with pytest.raises(ValueError):
        Gefen(_params_1d(), fused=False, **kwargs)


def test_gefen_rejects_capturable_with_codebook_refresh():
    with pytest.raises(ValueError):
        Gefen(_params_1d(), fused=False, capturable=True, codebook_refresh_every=10)


def test_gefen_rejects_non_tensor_param():
    with pytest.raises(TypeError):
        Gefen([("bad", "not a tensor")], fused=False)


def test_gefen_add_param_group_atomic_on_partial_failure():
    # A group whose SECOND param is invalid (complex) must raise and register
    # NONE of the group's params -- add_param_group is all-or-nothing.
    opt = Gefen(_params_1d(), fused=False)
    before = len(opt.param_groups)
    good = nn.Parameter(torch.randn(4))
    bad = nn.Parameter(torch.randn(4, dtype=torch.complex64))
    with pytest.raises(ValueError):
        opt.add_param_group({"params": [("good", good), ("bad", bad)]})
    assert len(opt.param_groups) == before
    # The valid first param must NOT have been left registered.
    assert not any("good" in g.get("param_names", ()) for g in opt.param_groups)


def test_gefen_add_param_group_atomic_on_duplicate():
    # A within-batch duplicate param must raise before ANY group is registered.
    opt = Gefen(_params_1d(), fused=False)
    before = len(opt.param_groups)
    dup = nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError):
        opt.add_param_group({"params": [("a", dup), ("b", dup)]})
    assert len(opt.param_groups) == before


# ---------------------------------------------------------------------------
# GefenMuon constructor validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kwargs",
    [
        {"lr": -1.0},
        {"momentum": -0.5},
        {"weight_decay": -0.1},
        {"adjust_lr_fn": "bogus"},
        {"sharded_mode": "bogus"},
        {"normuon": True, "normuon_beta2": 1.0},
        {"ns_schedule": "bogus"},
    ],
)
def test_gefen_muon_rejects_bad_hyperparams(kwargs):
    with pytest.raises(ValueError):
        GefenMuon(_params_2d(), fused=False, **kwargs)


def test_gefen_muon_rejects_non_2d_param():
    with pytest.raises(ValueError):
        GefenMuon([("b", nn.Parameter(torch.randn(8)))], fused=False)


@pytest.mark.parametrize("ns_steps", [0, -1, 100])
def test_gefen_muon_rejects_bad_ns_steps_on_fixed_path(ns_steps):
    # ns_steps IS consumed on the classic fixed-coefficient path (no schedule,
    # or the "standard" named schedule that defers to ns_coefficients/ns_steps).
    with pytest.raises(ValueError):
        GefenMuon(_params_2d(), fused=False, ns_steps=ns_steps)
    with pytest.raises(ValueError):
        GefenMuon(
            _params_2d(), fused=False, ns_schedule="standard", ns_steps=ns_steps
        )


@pytest.mark.parametrize("ns_schedule", ["tuned3", [(3.0, -4.0, 2.0), (2.5, -3.5, 1.5)]])
def test_gefen_muon_explicit_schedule_ignores_bad_ns_steps(ns_schedule):
    # An explicit per-iteration schedule (named tuned schedule or an explicit
    # list) carries its own length, so an out-of-range ns_steps must NOT raise.
    opt = GefenMuon(
        _params_2d(), fused=False, ns_schedule=ns_schedule, ns_steps=0
    )
    # The schedule length becomes the effective ns_steps, not the ignored arg.
    assert opt.param_groups[0]["ns_steps"] == len(opt.param_groups[0]["ns_coefficients"])


# ---------------------------------------------------------------------------
# Newton-Schulz schedule + helpers
# ---------------------------------------------------------------------------

def test_normalize_ns_schedule_single_tuple_repeats():
    sched = _normalize_ns_schedule((3.0, -4.0, 2.0), ns_steps=5)
    assert sched == [(3.0, -4.0, 2.0)] * 5


def test_normalize_ns_schedule_explicit_list_ignores_ns_steps():
    entries = [(3.0, -4.0, 2.0), (2.5, -3.5, 1.5)]
    assert _normalize_ns_schedule(entries, ns_steps=50) == entries


@pytest.mark.parametrize(
    "coeffs,steps",
    [
        ((), 5),                         # empty
        ((3.0, -4.0), 5),                # single tuple with wrong arity
        ((3.0, -4.0, 2.0), 100),         # too many steps
        ([(3.0, -4.0, 2.0, 1.0)], 5),    # schedule entry with wrong arity
    ],
)
def test_normalize_ns_schedule_rejects(coeffs, steps):
    with pytest.raises(ValueError):
        _normalize_ns_schedule(coeffs, ns_steps=steps)


def test_named_ns_schedules_resolve():
    assert NS_SCHEDULES["standard"] is None
    for name in ("tuned3", "tuned4"):
        sched = _normalize_ns_schedule(NS_SCHEDULES[name], ns_steps=5)
        assert all(len(entry) == 3 for entry in sched)


def test_adjust_lr_ratio():
    shape = torch.Size((1024, 256))
    assert _adjust_lr_ratio(None, shape) == pytest.approx(math.sqrt(1024 / 256))
    assert _adjust_lr_ratio("original", shape) == pytest.approx(math.sqrt(1024 / 256))
    # rows < cols clamps at 1 for the original scaling
    assert _adjust_lr_ratio("original", torch.Size((256, 1024))) == 1.0
    assert _adjust_lr_ratio("match_rms_adamw", shape) == pytest.approx(
        0.2 * math.sqrt(1024)
    )


def test_newton_schulz_orthogonalizes_on_cpu():
    torch.manual_seed(0)
    for shape in ((64, 96), (96, 64)):  # wide and tall (transpose handling)
        grad = torch.randn(shape)
        out = _zeropower_via_newtonschulz(
            grad, (3.4445, -4.7750, 2.0315), ns_steps=10, eps=1e-7
        )
        assert out.shape == grad.shape
        assert torch.isfinite(out).all()
        # The classic quintic drives every singular value into a band around 1
        # (~[0.68, 1.14] at these shapes/steps), not to exact orthogonality.
        singular_values = torch.linalg.svdvals(out.float())
        assert singular_values.min() > 0.5
        assert singular_values.max() < 1.3


def test_newton_schulz_rejects_non_2d():
    with pytest.raises(ValueError):
        _zeropower_via_newtonschulz(
            torch.randn(8), (3.0, -4.0, 2.0), ns_steps=5, eps=1e-7
        )


def test_cautious_mask_zeroes_disagreement_and_preserves_scale():
    torch.manual_seed(1)
    update = torch.randn(64, 64)
    grad = torch.randn(64, 64)
    agree = ((update * grad) > 0)
    expected_scale = agree.numel() / agree.sum().float()
    original = update.clone()
    out = _cautious_mask_(update, grad)
    assert out is update  # in-place contract
    assert torch.all(out[~agree] == 0)
    assert torch.allclose(
        out[agree], original[agree] * expected_scale.to(out.dtype)
    )


# ---------------------------------------------------------------------------
# Param split + validation
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.proj = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 32, bias=False)
        self.frozen = nn.Linear(16, 16)
        self.frozen.weight.requires_grad_(False)
        self.frozen.bias.requires_grad_(False)


def test_split_params_for_muon_routing():
    model = _TinyLM()
    muon, backup = split_params_for_muon(model)
    muon_names = {n for n, _ in muon}
    backup_names = {n for n, _ in backup}
    assert muon_names == {"proj.weight"}
    # embeddings/lm_head (2D but excluded by substring) + 1D go to backup
    assert "embed_tokens.weight" in backup_names
    assert "lm_head.weight" in backup_names
    assert "norm.weight" in backup_names and "proj.bias" in backup_names
    # frozen params are skipped entirely
    assert not any("frozen" in n for n in muon_names | backup_names)


def test_is_muon_param_respects_custom_substrings():
    p2d = nn.Parameter(torch.randn(4, 4))
    assert is_muon_param("layer.attn.weight", p2d)
    assert not is_muon_param("model.wte.weight", p2d)
    assert not is_muon_param("custom.skipme.weight", p2d, ("skipme",))


def test_validate_split_errors():
    a = nn.Parameter(torch.randn(4, 4))
    b = nn.Parameter(torch.randn(4))
    with pytest.raises(ValueError, match="muon list"):
        validate_split([("a", a), ("a2", a)], [("b", b)])
    with pytest.raises(ValueError, match="backup list"):
        validate_split([("a", a)], [("b", b), ("b2", b)])
    with pytest.raises(ValueError, match="BOTH"):
        validate_split([("a", a)], [("a_again", a)])
    with pytest.raises(ValueError, match=r"[Dd]uplicate parameter name"):
        validate_split([("shared", a)], [("SHARED", b)])


def test_validate_split_flags_uncovered_trainable_param():
    model = _TinyLM()
    muon, backup = split_params_for_muon(model)
    with pytest.raises(ValueError, match="neither"):
        validate_split(muon, backup[:-1], model=model)
    # and passes (returning the lists) when the split is complete
    m2, b2 = validate_split(muon, backup, model=model)
    assert m2 == muon and b2 == backup


# ---------------------------------------------------------------------------
# GefenMuonHybrid construction
# ---------------------------------------------------------------------------

def test_hybrid_rejects_empty_params():
    with pytest.raises(ValueError):
        GefenMuonHybrid([], [], lr=1e-3, fused=False)


def test_hybrid_add_param_group_not_implemented():
    opt = GefenMuonHybrid.from_model(_TinyLM(), lr=1e-3, fused=False)
    with pytest.raises(NotImplementedError):
        opt.add_param_group({"params": [nn.Parameter(torch.randn(4, 4))]})


def test_hybrid_warns_on_legacy_adjust_lr_fn():
    model = _TinyLM()
    with pytest.warns(UserWarning, match="adjust_lr_fn"):
        GefenMuonHybrid.from_model(model, lr=1e-3, fused=False, adjust_lr_fn=None)


def test_hybrid_from_model_covers_all_trainable_params():
    model = _TinyLM()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    opt_param_ids = {
        id(p) for group in opt.param_groups for p in group["params"]
    }
    trainable_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert opt_param_ids == trainable_ids


def test_hybrid_per_half_lr_overrides():
    model = _TinyLM()
    opt = GefenMuonHybrid.from_model(
        model, lr=1e-3, backup_lr=5e-4, fused=False
    )
    # The merged param_groups view must expose both halves' lrs.
    lrs = {g["lr"] for g in opt.param_groups}
    assert 5e-4 in lrs and any(abs(lr - 5e-4) > 1e-12 for lr in lrs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
