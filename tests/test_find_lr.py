"""
CPU smoke test for the productized LR finder (tools/find_lr.py).

Exercises the range_test path (with a custom closure) for every optimizer family
the finder supports, and checks that the finder is non-destructive (snapshots and
restores the model weights). Sweep mode is HF-model-specific (needs
model(input_ids=, labels=) + token blocks) and is covered by the GPU validation
scripts, not here.

Runs on the non-fused path; a CUDA device may need to be visible for the SM-count
query on larger-period params, but tensors stay on CPU.

Run: python tests/test_find_lr.py
"""
import math

import torch
import torch.nn as nn

from gefen.tools.find_lr import find_lr, build_optimizer, OPTIMIZERS


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        # names chosen so split_params_for_muon / classify_param_type route them:
        self.q_proj = nn.Linear(64, 64, bias=False)          # -> muon
        self.down_proj = nn.Linear(64, 64, bias=False)       # -> muon
        self.input_layernorm = nn.LayerNorm(64)              # -> backup (norm)
        self.embed_tokens = nn.Embedding(100, 32)            # -> backup


def _make_closure(model):
    gen = torch.Generator().manual_seed(0)

    def closure():
        x = torch.randn(8, 64, generator=gen)
        h = model.down_proj(model.q_proj(x))
        h = model.input_layernorm(h)
        idx = torch.randint(0, 100, (8,), generator=gen)
        e = model.embed_tokens(idx)
        loss = (h ** 2).mean() + (e ** 2).mean()
        loss.backward()
        return loss.detach()

    return closure


def _state_fingerprint(model):
    return torch.cat([v.detach().reshape(-1) for v in model.state_dict().values()])


def test_build_optimizer_all_families():
    model = Tiny()
    for opt_name in OPTIMIZERS:
        opt = build_optimizer(model, opt_name, 1e-4, fused=False)
        assert opt is not None
    print("A. build_optimizer covers", OPTIMIZERS, "-> PASS")


def test_range_test_each_family_and_restore():
    for opt_name in ("adamw", "gefen", "hybrid"):
        torch.manual_seed(0)
        model = Tiny()
        before = _state_fingerprint(model).clone()
        res = find_lr(
            model, _make_closure(model), optimizer=opt_name, method="range_test",
            device="cpu", fused=False, num_iter=30, start_lr=1e-6, end_lr=1e-1,
            restore=True, verbose=False,
        )
        assert res.optimizer == opt_name and res.method == "range_test"
        assert math.isfinite(res.lr) and res.lr > 0, (opt_name, res.lr)
        # Non-destructive: weights restored exactly.
        after = _state_fingerprint(model)
        assert torch.equal(before, after), f"{opt_name}: model not restored"
        print(f"B. range_test[{opt_name}] -> lr={res.lr:.2e} (restored) PASS")


def test_restore_false_mutates():
    torch.manual_seed(0)
    model = Tiny()
    before = _state_fingerprint(model).clone()
    find_lr(model, _make_closure(model), optimizer="gefen", method="range_test",
            device="cpu", fused=False, num_iter=20, restore=False, verbose=False)
    after = _state_fingerprint(model)
    assert not torch.equal(before, after), "restore=False should leave the model trained"
    print("C. restore=False mutates the model (as documented) PASS")


def run():
    test_build_optimizer_all_families()
    test_range_test_each_family_and_restore()
    test_restore_false_mutates()
    print("\nAll find_lr smoke checks passed.")


if __name__ == "__main__":
    run()
