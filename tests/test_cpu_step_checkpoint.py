"""CPU-only coverage for the optimizer step and checkpoint round-trip paths.

Everything here runs without a GPU (fused=False decomposed paths), so it
executes on CPU-only CI runners:

  * regression: _should_use_v2 must never touch the CUDA runtime for a CPU
    param (it used to call torch.cuda.current_device() via _sm_count, crashing
    CPU-only hosts whenever a param's period exceeded _V2_PERIOD_MAX);
  * step smoke for Gefen / GefenMuon / GefenMuonHybrid on CPU params;
  * bit-exact state_dict/load_state_dict resume on CPU;
  * the factored_v_2d <-> legacy-vmean lazy checkpoint migration branches;
  * step-counter normalization across the capturable toggle.

Run: pytest tests/test_cpu_step_checkpoint.py
"""
import copy

import pytest
import torch
import torch.nn as nn

import gefen.kernels.automatic_gefen_fused as fused_mod
from gefen import Gefen, GefenMuon, GefenMuonHybrid


def _small_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(32, 48),
        nn.Tanh(),
        nn.Linear(48, 16),
    )


def _synthetic_grads(model, num_steps, seed=123):
    """One fixed grad tensor per (step, param): drives A/B runs identically."""
    gen = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(p.shape, generator=gen) * 1e-2 for p in model.parameters()]
        for _ in range(num_steps)
    ]


def _apply_grads(model, grads):
    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone()


def test_should_use_v2_cpu_never_touches_cuda(monkeypatch):
    """CPU device + period above _V2_PERIOD_MAX must resolve without CUDA."""

    def _boom(device):
        raise AssertionError("_sm_count must not be consulted for a CPU param")

    monkeypatch.setattr(fused_mod, "_sm_count", _boom)
    dev = torch.device("cpu")
    period = fused_mod._V2_PERIOD_MAX + 1
    assert fused_mod._should_use_v2(10_000_000, period, dev) is False
    # The tiny-period early-out stays device-independent.
    assert fused_mod._should_use_v2(10_000_000, fused_mod._V2_PERIOD_MAX, dev) is True


@pytest.mark.parametrize("factored", [True, False])
def test_gefen_cpu_step_smoke(factored):
    model = _small_model()
    opt = Gefen(
        list(model.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    grads = _synthetic_grads(model, 3)
    before = [p.detach().clone() for p in model.parameters()]
    for step_grads in grads:
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    for b, p in zip(before, model.parameters()):
        assert torch.isfinite(p).all()
        assert not torch.equal(b, p), "step() left a parameter untouched"


def test_gefen_muon_cpu_step_smoke():
    torch.manual_seed(0)
    params = [
        ("blocks.0.w", nn.Parameter(torch.randn(32, 48))),
        ("blocks.1.w", nn.Parameter(torch.randn(48, 16))),
    ]
    opt = GefenMuon(params, lr=1e-3, fused=False, adjust_lr_fn="match_rms_adamw")
    gen = torch.Generator().manual_seed(7)
    for _ in range(3):
        for _, p in params:
            p.grad = torch.randn(p.shape, generator=gen) * 1e-2
        opt.step()
        opt.zero_grad()
    for _, p in params:
        assert torch.isfinite(p).all()


def test_hybrid_cpu_step_smoke():
    model = _small_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    grads = _synthetic_grads(model, 3)
    for step_grads in grads:
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    for p in model.parameters():
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("factored", [True, False])
def test_state_dict_roundtrip_bit_exact(factored):
    """Save/load mid-run, then step both runs on identical grads: bit-exact."""
    grads_all = _synthetic_grads(_small_model(), 5)

    model_a = _small_model()
    opt_a = Gefen(
        list(model_a.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    for step_grads in grads_all[:2]:
        _apply_grads(model_a, step_grads)
        opt_a.step()
        opt_a.zero_grad()

    saved_opt = copy.deepcopy(opt_a.state_dict())
    saved_params = [p.detach().clone() for p in model_a.parameters()]

    model_b = _small_model()
    with torch.no_grad():
        for p, saved in zip(model_b.parameters(), saved_params):
            p.copy_(saved)
    opt_b = Gefen(
        list(model_b.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    opt_b.load_state_dict(saved_opt)

    for step_grads in grads_all[2:]:
        _apply_grads(model_a, step_grads)
        _apply_grads(model_b, step_grads)
        opt_a.step()
        opt_b.step()
        opt_a.zero_grad()
        opt_b.zero_grad()

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb), "resumed run diverged from continuous run"


def test_load_state_dict_does_not_mutate_caller_dict():
    model = _small_model()
    opt = Gefen(list(model.named_parameters()), lr=1e-3, fused=False)
    _apply_grads(model, _synthetic_grads(model, 1)[0])
    opt.step()
    sd = opt.state_dict()
    opt.load_state_dict(sd)
    # A second load must still see the gefen_* top-level keys (shallow-copy
    # contract in load_state_dict).
    assert "gefen_global_step" in sd and "gefen_codebook" in sd
    opt.load_state_dict(sd)


def _run_and_save(factored):
    model = _small_model()
    opt = Gefen(
        list(model.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    for step_grads in _synthetic_grads(model, 2):
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    return model, opt


def _state_of_2d(opt):
    for group in opt.param_groups:
        (p,) = group["params"]
        if p.ndim == 2:
            return p, opt.state[p]
    raise AssertionError("no 2D param found")


def test_checkpoint_migration_legacy_to_factored():
    """vmean-era checkpoint resumed with factored_v_2d=True grows v_row/v_col."""
    model_src, opt_src = _run_and_save(factored=False)
    _, state_src = _state_of_2d(opt_src)
    assert "vmean" in state_src and "v_row" not in state_src

    model_dst = _small_model()
    with torch.no_grad():
        for pd, ps in zip(model_dst.parameters(), model_src.parameters()):
            pd.copy_(ps)
    opt_dst = Gefen(
        list(model_dst.named_parameters()), lr=1e-3, fused=False, factored_v_2d=True
    )
    opt_dst.load_state_dict(opt_src.state_dict())
    _apply_grads(model_dst, _synthetic_grads(model_dst, 1)[0])
    opt_dst.step()

    p_dst, state_dst = _state_of_2d(opt_dst)
    assert "v_row" in state_dst and "v_col" in state_dst and "factored_step" in state_dst
    assert state_dst["v_row"].shape == (p_dst.shape[0],)
    assert state_dst["v_col"].shape == (p_dst.shape[1],)
    assert torch.isfinite(p_dst).all()


def test_checkpoint_migration_factored_to_legacy():
    """factored_v checkpoint resumed with the flag off recreates vmean, with
    vmean_step backfilled so bias correction treats it as a fresh EMA."""
    model_src, opt_src = _run_and_save(factored=True)
    _, state_src = _state_of_2d(opt_src)
    assert "v_row" in state_src and "vmean" not in state_src

    model_dst = _small_model()
    with torch.no_grad():
        for pd, ps in zip(model_dst.parameters(), model_src.parameters()):
            pd.copy_(ps)
    opt_dst = Gefen(
        list(model_dst.named_parameters()), lr=1e-3, fused=False, factored_v_2d=False
    )
    opt_dst.load_state_dict(opt_src.state_dict())
    _apply_grads(model_dst, _synthetic_grads(model_dst, 1)[0])
    opt_dst.step()

    p_dst, state_dst = _state_of_2d(opt_dst)
    assert "vmean" in state_dst and "vmean_step" in state_dst
    assert torch.isfinite(p_dst).all()


def test_counter_normalization_across_capturable_toggle():
    """A capturable=False checkpoint loads into capturable=True as 0-dim
    tensors (and back to python ints in the other direction)."""
    _, opt_src = _run_and_save(factored=True)
    sd = opt_src.state_dict()

    model_dst = _small_model()
    opt_capt = Gefen(
        list(model_dst.named_parameters()), lr=1e-3, fused=False, capturable=True
    )
    opt_capt.load_state_dict(sd)
    for group in opt_capt.param_groups:
        for p in group["params"]:
            step = opt_capt.state[p]["step"]
            assert torch.is_tensor(step) and step.ndim == 0
            assert step.dtype == torch.float32

    sd_capt = opt_capt.state_dict()
    model_back = _small_model()
    opt_plain = Gefen(
        list(model_back.named_parameters()), lr=1e-3, fused=False, capturable=False
    )
    opt_plain.load_state_dict(sd_capt)
    for group in opt_plain.param_groups:
        for p in group["params"]:
            assert isinstance(opt_plain.state[p]["step"], int)


def test_hybrid_state_dict_roundtrip():
    model = _small_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    for step_grads in _synthetic_grads(model, 2):
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    sd = opt.state_dict()
    assert set(sd.keys()) == {"muon", "backup"}

    model_b = _small_model()
    opt_b = GefenMuonHybrid.from_model(model_b, lr=1e-3, fused=False)
    opt_b.load_state_dict(sd)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
