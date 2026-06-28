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


class _NS:
    def __init__(self, **kw):
        self.python = kw.get("python")
        self.venv = kw.get("venv")
        self.cuda_home = kw.get("cuda_home")


def test_reexec_planner():
    import sys
    from gefen.tools.find_lr import _reexec_command, _strip_flags
    real = sys.executable                 # a real, existing interpreter
    exe = "/opt/envs/cu130/bin/python"    # a (fake) "current" interpreter, != real
    base_argv = ["--model", "/m", "--optimizer", "gefen", "--method", "sweep"]

    # No --python/--venv -> run as-is (None).
    assert _reexec_command(_NS(), exe, base_argv, {}) is None
    # Target == current interpreter -> no re-exec (checked before existence).
    assert _reexec_command(_NS(python=exe), exe, base_argv, {}) is None
    # Already the re-exec'd child -> no re-exec.
    assert _reexec_command(_NS(python=real), exe, base_argv,
                           {"GEFEN_FINDLR_REEXEC": "1"}) is None
    # Different existing interpreter -> plan to re-exec, with CUDA_HOME + stripped argv.
    plan = _reexec_command(_NS(python=real, cuda_home="/cuda"),
                           exe, ["--venv", "/x"] + base_argv, {"PATH": "/bin"})
    assert plan is not None
    target, new_argv, env = plan
    assert target == real
    assert new_argv[:2] == ["-m", "gefen.tools.find_lr"]
    assert "--venv" not in new_argv and "/x" not in new_argv          # stripped
    assert env["CUDA_HOME"] == "/cuda" and env["PATH"].startswith("/cuda/bin")
    assert env["GEFEN_FINDLR_REEXEC"] == "1"
    # _strip_flags handles both spellings.
    assert _strip_flags(["--venv=/x", "--model", "/m"], ("--venv",)) == ["--model", "/m"]
    print("D. re-exec planner (venv/python/cuda-home, strip, child-guard) PASS")


def test_distribute_lrs():
    from gefen.tools.find_lr import distribute_lrs
    grid = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    # Round-robin across 3 devices: even spread, every LR placed exactly once.
    a = distribute_lrs(grid, ["cuda:0", "cuda:1", "cuda:2"])
    assert a == {"cuda:0": [1e-5, 3e-4], "cuda:1": [3e-5, 1e-3], "cuda:2": [1e-4]}
    placed = [lr for ls in a.values() for lr in ls]
    assert sorted(placed) == sorted(grid)
    # More GPUs than LRs -> empty devices dropped (no idle worker spawned).
    a2 = distribute_lrs([1e-5, 3e-5], ["cuda:0", "cuda:1", "cuda:2"])
    assert a2 == {"cuda:0": [1e-5], "cuda:1": [3e-5]}
    # Single device gets the whole grid (== sequential path coverage).
    assert distribute_lrs(grid, ["cuda:0"]) == {"cuda:0": grid}
    # Balanced sizes: counts differ by at most 1.
    sizes = [len(v) for v in distribute_lrs(grid, ["a", "b"]).values()]
    assert max(sizes) - min(sizes) <= 1
    print("E. distribute_lrs round-robin (even, complete, drops empty) PASS")


def test_decoder_layers():
    from gefen.tools.find_lr import _decoder_layers

    class Inner(nn.Module):
        def __init__(s):
            super().__init__()
            s.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    class CausalLM(nn.Module):  # Llama/Qwen-like: model.model.layers
        def __init__(s):
            super().__init__()
            s.model = Inner()
            s.lm_head = nn.Linear(4, 8)

    class GPT2(nn.Module):  # GPT2-like: model.h
        def __init__(s):
            super().__init__()
            s.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

    assert len(_decoder_layers(CausalLM())) == 3
    assert len(_decoder_layers(GPT2())) == 2
    assert _decoder_layers(nn.Linear(4, 4)) is None  # -> shard whole model only
    print("F. _decoder_layers discovery (model.layers / .h / none) PASS")


def run():
    test_build_optimizer_all_families()
    test_range_test_each_family_and_restore()
    test_restore_false_mutates()
    test_reexec_planner()
    test_distribute_lrs()
    test_decoder_layers()
    print("\nAll find_lr smoke checks passed.")


if __name__ == "__main__":
    run()
