"""
CPU smoke test for tools/lr_calibration.py.

Runs entirely on the non-fused CPU path (fused=True needs CUDA), so it can run
in the CPU CI smoke job. Exercises:
  A. measure_update_rms returns one finite stat per param group, tagged muon/gefen
  B. calibrate_relative produces a multiplier that lands the muon group's applied
     update RMS on the gefen reference (multiplier * muon_rms ~= gefen_rms), and
     reports the analytic match_rms_adamw comparison
  C. lr_range_test ramps the LR and returns a finite suggested LR
  D. the probes are non-invasive: weight_decay and per-group LRs are restored

Run: python tests/test_lr_calibration.py
"""
import math

import torch

from gefen import GefenMuonHybrid
from gefen.tools.lr_calibration import (
    measure_update_rms,
    calibrate_relative,
    lr_range_test,
)

LR = 1e-3


def _build():
    torch.manual_seed(0)
    # 2D hidden weights -> muon; bias (1D) + embedding -> gefen backup.
    w1 = torch.nn.Parameter(torch.randn(64, 96) * 0.1)
    w2 = torch.nn.Parameter(torch.randn(128, 64) * 0.1)
    bias = torch.nn.Parameter(torch.randn(64) * 0.1)
    emb = torch.nn.Parameter(torch.randn(100, 32) * 0.1)
    muon = [("layer1.weight", w1), ("layer2.weight", w2)]
    backup = [("layer1.bias", bias), ("embed.weight", emb)]
    opt = GefenMuonHybrid(
        muon, backup, lr=LR, weight_decay=0.1, fused=False, adjust_lr_fn=None
    )
    return opt, [w1, w2, bias, emb]


def _make_closure(params):
    # Deterministic pseudo-grad closure: no real model, just populate .grad so
    # the optimizer does a representative step. A fixed seed per call keeps the
    # dry run reproducible.
    gen = torch.Generator().manual_seed(1234)

    def closure():
        for p in params:
            p.grad = torch.randn(p.shape, generator=gen) * 0.01
        # Return a dummy finite "loss"; the calibration probe only needs grads.
        return torch.tensor(1.0)

    return closure


def test_measure_update_rms():
    opt, params = _build()
    closure = _make_closure(params)
    stats = measure_update_rms(opt, closure, num_steps=12, warmup=2)

    assert len(stats) == len(opt.param_groups)
    kinds = {s.label.split(":")[1] for s in stats}
    assert "muon" in kinds and "gefen" in kinds
    for s in stats:
        assert math.isfinite(s.applied_rms), f"{s.label} non-finite RMS"
        assert s.applied_rms > 0
    print("A. measure_update_rms:")
    for s in stats:
        print(f"   {s.label:<28} applied_rms={s.applied_rms:.4e} (muon={s.is_muon})")


def test_calibrate_relative():
    opt, params = _build()
    closure = _make_closure(params)
    cal = calibrate_relative(opt, closure, reference="gefen", num_steps=14, warmup=2)
    print("\nB. " + cal.summary())

    # Reference (a gefen group) must have multiplier ~1.
    ref_mult = cal.multipliers[cal.reference_label]
    assert abs(ref_mult - 1.0) < 1e-6

    # Applying each multiplier should equalize applied RMS to the reference.
    ref_rms = next(s.applied_rms for s in cal.stats if s.label == cal.reference_label)
    for s in cal.stats:
        scaled = s.applied_rms * cal.multipliers[s.label]
        assert math.isclose(scaled, ref_rms, rel_tol=1e-6), s.label

    # The muon comparison block must be populated and finite.
    assert cal.analytic_comparison, "expected a muon analytic comparison"
    for label, c in cal.analytic_comparison.items():
        assert math.isfinite(c["empirical_vs_ref"])
        assert math.isfinite(c["match_rms_adamw"])
        assert c["empirical_vs_ref"] > 0


def test_lr_range_test():
    opt, params = _build()
    closure = _make_closure(params)
    res = lr_range_test(
        opt, closure, num_iter=40, start_lr=1e-6, end_lr=1.0, diverge_factor=1e9
    )
    print("\nC. " + res.summary())
    assert len(res.lrs) > 0
    assert math.isfinite(res.suggested_lr) and res.suggested_lr > 0
    # The suggestion must lie inside the scanned ramp, not just be positive.
    assert res.lrs[0] <= res.suggested_lr <= res.lrs[-1]


def test_non_invasive():
    opt, params = _build()
    closure = _make_closure(params)
    wd_before = [g.get("weight_decay") for g in opt.param_groups]
    lr_before = [g["lr"] for g in opt.param_groups]

    measure_update_rms(opt, closure, num_steps=8, warmup=2)
    lr_range_test(opt, closure, num_iter=20, start_lr=1e-6, end_lr=1.0,
                  diverge_factor=1e9)

    wd_after = [g.get("weight_decay") for g in opt.param_groups]
    lr_after = [g["lr"] for g in opt.param_groups]
    assert wd_before == wd_after, "weight_decay not restored"
    assert lr_before == lr_after, "per-group lr not restored"
    print("\nD. non-invasive: weight_decay and lr restored")


def run():
    test_measure_update_rms()
    test_calibrate_relative()
    test_lr_range_test()
    test_non_invasive()
    print("\nAll lr_calibration smoke checks passed.")


if __name__ == "__main__":
    run()
