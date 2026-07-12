"""
Smoke test for GefenMuonHybrid (hybrid.py).

Exercises the four things the composite is responsible for:
  A. routing + a real optimizer step (2D weights -> GefenMuon, rest -> Gefen),
  B. LR-scheduler propagation through the property-based param_groups (the
     reason GefenMuonHybrid deliberately skips super().__init__()),
  C. state_dict / load_state_dict round-trip (namespaced muon/backup),
  D. add_param_group is rejected.

GPU-only (the fused Gefen/GefenMuon path needs CUDA); skipped without it.

Run: python tests/test_gefen_muon_hybrid.py
"""
import copy

import torch

try:
    import pytest
except ImportError:  # pragma: no cover - allows the script path without pytest
    pytest = None

from gefen import GefenMuonHybrid

DEV = "cuda"
LR = 1e-3


def _build():
    torch.manual_seed(0)
    # 2D hidden weights -> muon (GefenMuon only accepts 2D matrices)
    w1 = torch.nn.Parameter(torch.randn(64, 96, device=DEV, dtype=torch.bfloat16) * 0.1)
    w2 = torch.nn.Parameter(torch.randn(128, 64, device=DEV, dtype=torch.bfloat16) * 0.1)
    # bias (1D) + embedding (2D, but routed to the Gefen backup) -> backup
    bias = torch.nn.Parameter(torch.randn(64, device=DEV, dtype=torch.bfloat16) * 0.1)
    emb = torch.nn.Parameter(torch.randn(100, 32, device=DEV, dtype=torch.bfloat16) * 0.1)
    muon = [("layer1.weight", w1), ("layer2.weight", w2)]
    backup = [("layer1.bias", bias), ("embed.weight", emb)]
    opt = GefenMuonHybrid(muon, backup, lr=LR, fused=True)
    return opt, [w1, w2, bias, emb]


def _set_grads(params):
    for p in params:
        p.grad = torch.randn_like(p) * 0.01


def _step(opt, params):
    opt.zero_grad(set_to_none=True)
    _set_grads(params)
    opt.step()


def run():
    # --- A. routing + step ---
    opt, params = _build()
    # muon got the 2 weights, backup got bias + embedding
    assert len(opt.muon.param_groups) == 1
    assert len(opt.backup.param_groups) == 1
    # param_groups property exposes the child optimizers' preserved groups.
    assert len(opt.param_groups) == 2
    before = [p.detach().clone() for p in params]
    _step(opt, params)
    changed = [not torch.equal(p.detach(), b) for p, b in zip(params, before)]
    finite = [torch.isfinite(p).all().item() for p in params]
    a_ok = all(changed) and all(finite)
    print(f"[A] routing+step: changed={changed} finite={finite} -> {'PASS' if a_ok else 'FAIL'}")

    # --- B. LR scheduler propagation through the property param_groups ---
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    lr_before = opt.muon.param_groups[0]["lr"]
    _step(opt, params)  # optimizer.step() before scheduler.step() (correct order)
    sched.step()
    lr_after_muon = opt.muon.param_groups[0]["lr"]
    lr_after_backup = opt.backup.param_groups[0]["lr"]
    # the scheduler mutated group["lr"] in place; because param_groups returns
    # the children's real group dicts, both sub-optimizers must see the new lr.
    b_ok = (
        abs(lr_after_muon - lr_before * 0.5) < 1e-12
        and abs(lr_after_backup - lr_before * 0.5) < 1e-12
    )
    print(f"[B] scheduler lr {lr_before}->{lr_after_muon} (muon) / {lr_after_backup} "
          f"(backup) -> {'PASS' if b_ok else 'FAIL'}")

    # --- C. state_dict round-trip ---
    _step(opt, params)  # build some optimizer state
    sd = copy.deepcopy(opt.state_dict())
    assert set(sd.keys()) == {"muon", "backup", "backup_optimizer"}
    assert sd["backup_optimizer"] == "gefen"
    opt2, params2 = _build()
    _step(opt2, params2)  # populate live state so load has matching structure
    crashed = None
    try:
        opt2.load_state_dict(sd)
        _step(opt2, params2)  # resumed step must not crash
    except Exception as e:  # noqa: BLE001
        crashed = repr(e)
    c_ok = crashed is None and all(torch.isfinite(p).all().item() for p in params2)
    print(f"[C] state_dict round-trip: crashed={crashed} -> {'PASS' if c_ok else 'FAIL'}")

    # --- D. add_param_group rejected ---
    d_ok = False
    try:
        opt.add_param_group({"params": [torch.nn.Parameter(torch.zeros(2, 2, device=DEV))]})
    except NotImplementedError:
        d_ok = True
    print(f"[D] add_param_group rejected -> {'PASS' if d_ok else 'FAIL'}")

    return a_ok and b_ok and c_ok and d_ok


if pytest is not None:
    requires_cuda = pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GefenMuonHybrid fused path requires a CUDA device",
    )

    @requires_cuda
    def test_gefen_muon_hybrid_smoke():
        assert run()


if __name__ == "__main__":
    ok = run()
    print("\nHYBRID SMOKE OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
