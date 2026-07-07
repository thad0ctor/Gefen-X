"""CPU-only regression coverage for the core-optimizer correctness batch.

Covers: the fp32 grad-aliasing corruption in the factored decomposed fallback,
closure-before-codebook step ordering, add_param_group support, frozen-param
registration, sparse/complex clean rejection, tensor-lr scalarization
(Gefen._lr_scalar), eps / tensor-lr validation symmetry, the print ->
logging/warnings conversion, and the fp16 second-moment overflow fix.

Run: pytest tests/test_core_correctness_cpu.py
"""
import pytest
import torch
import torch.nn as nn

import gefen.gefen as gefen_mod
import gefen.quantization as quantization
from gefen import Gefen
from gefen.gefen import automatic_vmean_update


# ---------------------------------------------------------------------------
# 1. fp32 grad aliasing in the factored (decomposed) fallback
# ---------------------------------------------------------------------------

def test_factored_fp32_step_preserves_grad_and_direction():
    """.float() on an fp32 grad aliases it; the in-place square_ corrupted
    p.grad AND fed g^2 (not g) into the momentum update."""
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(16, 16))
    opt = Gefen([("w", p)], lr=1e-2, fused=False)  # factored_v_2d default True
    grad = -(torch.rand(16, 16) + 0.1)  # strictly negative everywhere
    p.grad = grad.clone()
    before = p.detach().clone()
    opt.step()
    # The critical assert: the step must not mutate the gradient.
    assert torch.equal(p.grad, grad), "step() corrupted p.grad in place"
    # Direction: first-step momentum ~ (1-beta1)*grad < 0, stepsize > 0, so
    # p must move UP. The g^2-fed bug produced an all-positive momentum and
    # moved p down instead.
    delta = p.detach() - before
    assert (delta >= 0).all()
    assert (delta > 0).any()


def test_factored_bf16_step_still_works():
    torch.manual_seed(1)
    p = nn.Parameter(torch.randn(16, 16, dtype=torch.bfloat16))
    opt = Gefen([("w", p)], lr=1e-2, fused=False)
    p.grad = torch.randn(16, 16, dtype=torch.bfloat16) * 0.1
    g0 = p.grad.clone()
    before = p.detach().clone()
    opt.step()
    assert torch.equal(p.grad, g0)
    assert torch.isfinite(p.detach().float()).all()
    assert not torch.equal(before, p.detach())


# ---------------------------------------------------------------------------
# 2. closure ordering: closure must run BEFORE codebook learning
# ---------------------------------------------------------------------------

def test_first_step_with_closure_learns_codebook_and_trains():
    torch.manual_seed(0)
    model = nn.Linear(8, 8)
    opt = Gefen(list(model.named_parameters()), lr=1e-2, fused=False)
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        return loss

    before = [p.detach().clone() for p in model.parameters()]
    # Used to crash: the codebook was learned from (nonexistent) grads before
    # the closure ever produced them.
    losses = [opt.step(closure) for _ in range(3)]
    assert all(loss is not None and torch.isfinite(loss) for loss in losses)
    assert opt._gefen_codebook is not None
    for b, p in zip(before, model.parameters()):
        assert not torch.equal(b, p), "closure-driven step left a param untouched"
    # A closure-driven run must actually train.
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# 3. add_param_group
# ---------------------------------------------------------------------------

def test_add_param_group_steps_and_honors_hypers():
    torch.manual_seed(0)
    p1 = nn.Parameter(torch.randn(8, 8))
    opt = Gefen([("w", p1)], lr=1e-3, fused=False)
    n_before = len(opt.param_groups)
    p2 = nn.Parameter(torch.randn(8, 8))
    opt.add_param_group({"params": [p2], "betas": (0.5, 0.9), "lr": 1e-4})
    added = opt.param_groups[n_before:]
    assert len(added) == 1
    group = added[0]
    # betas translate to the per-group beta1/beta2 keys step() actually reads.
    assert group["beta1"] == 0.5 and group["beta2"] == 0.9
    assert group["lr"] == 1e-4
    assert group["weight_decay"] == opt.defaults["weight_decay"]
    names = [g["name"] for g in opt.param_groups]
    assert all(names) and len(names) == len(set(names))
    p1.grad = torch.randn_like(p1) * 0.1
    p2.grad = torch.randn_like(p2) * 0.1
    before2 = p2.detach().clone()
    opt.step()  # KeyError: 'name' before the fix
    assert not torch.equal(before2, p2), "added param did not update"


def test_add_param_group_accepts_named_params_and_single_tensor():
    opt = Gefen([nn.Parameter(torch.randn(4, 4))], fused=False)
    named = nn.Parameter(torch.randn(4, 4))
    opt.add_param_group({"params": [("extra.weight", named)]})
    assert opt.param_groups[-1]["name"] == "extra.weight"
    bare = nn.Parameter(torch.randn(4))
    opt.add_param_group({"params": bare})
    assert any(p is bare for p in opt.param_groups[-1]["params"])


@pytest.mark.parametrize(
    "bad",
    [
        {"lr": -1.0},
        {"betas": (1.0, 0.9)},
        {"betas": (0.9, 1.0)},
        {"eps": -1.0},
        {"weight_decay": -0.1},
    ],
)
def test_add_param_group_rejects_bad_hyperparams(bad):
    opt = Gefen([nn.Parameter(torch.randn(4, 4))], fused=False)
    group = {"params": [nn.Parameter(torch.randn(4))]}
    group.update(bad)
    with pytest.raises(ValueError):
        opt.add_param_group(group)


def test_add_param_group_rejects_duplicate_param():
    p = nn.Parameter(torch.randn(4, 4))
    opt = Gefen([p], fused=False)
    with pytest.raises(ValueError):
        opt.add_param_group({"params": [p]})


# ---------------------------------------------------------------------------
# 4. frozen params are registered (like torch.optim) and step once unfrozen
# ---------------------------------------------------------------------------

def test_frozen_param_registered_and_updates_after_unfreeze():
    torch.manual_seed(0)
    lin = nn.Linear(8, 8)
    frozen = nn.Linear(8, 8)
    frozen.weight.requires_grad_(False)
    frozen.bias.requires_grad_(False)
    params = list(lin.named_parameters()) + list(frozen.named_parameters())
    opt = Gefen(params, lr=1e-2, fused=False)
    registered = {id(p) for g in opt.param_groups for p in g["params"]}
    assert id(frozen.weight) in registered, "frozen param was silently dropped"
    assert id(frozen.bias) in registered
    # First step: only the trainable half has grads; frozen params no-op.
    lin.weight.grad = torch.randn_like(lin.weight) * 0.1
    lin.bias.grad = torch.randn_like(lin.bias) * 0.1
    opt.step()
    # Unfreeze mid-run and give it a grad: it must now train.
    frozen.weight.requires_grad_(True)
    frozen.weight.grad = torch.randn_like(frozen.weight) * 0.1
    before = frozen.weight.detach().clone()
    opt.step()
    assert not torch.equal(before, frozen.weight)


# ---------------------------------------------------------------------------
# 5. sparse / complex clean rejection
# ---------------------------------------------------------------------------

def test_sparse_grad_raises_cleanly():
    p = nn.Parameter(torch.randn(4, 4))
    opt = Gefen([p], fused=False)
    indices = torch.tensor([[0, 1], [0, 1]])
    values = torch.tensor([1.0, 2.0])
    p.grad = torch.sparse_coo_tensor(indices, values, (4, 4))
    with pytest.raises(RuntimeError, match="sparse gradient"):
        opt.step()


def test_complex_param_rejected_at_construction_and_add():
    complex_param = nn.Parameter(torch.randn(4, 4, dtype=torch.complex64))
    with pytest.raises(ValueError, match="complex"):
        Gefen([complex_param], fused=False)
    opt = Gefen([nn.Parameter(torch.randn(4, 4))], fused=False)
    with pytest.raises(ValueError, match="complex"):
        opt.add_param_group({"params": [complex_param]})


# ---------------------------------------------------------------------------
# 6. tensor-lr scalarization helper (_lr_scalar)
# ---------------------------------------------------------------------------

def test_lr_scalar_caches_and_tracks_tensor_lr():
    lr = torch.tensor(1e-3)
    opt = Gefen([nn.Parameter(torch.randn(4, 4))], lr=lr, fused=False)
    group = opt.param_groups[0]
    v1 = opt._lr_scalar(group)
    assert isinstance(v1, float) and v1 == pytest.approx(1e-3)
    # Cache hit: poke a sentinel into the single-slot cache; an unchanged
    # (identity, _version) pair must be served from the cache, not re-read.
    opt._lr_scalar_cache = (lr, lr._version, 123.0)
    assert opt._lr_scalar(group) == 123.0
    # An in-place scheduler update bumps _version -> fresh .item().
    lr.mul_(2.0)
    assert opt._lr_scalar(group) == pytest.approx(2e-3)
    # A replacement tensor swaps identity -> fresh .item().
    group["lr"] = torch.tensor(5e-4)
    assert opt._lr_scalar(group) == pytest.approx(5e-4)
    # A float lr passes straight through.
    group["lr"] = 7e-4
    assert opt._lr_scalar(group) == 7e-4
    # The cache must not leak into serialized state.
    assert "_lr_scalar_cache" not in opt.state_dict()


def test_tensor_lr_cpu_step_trains():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(8, 8))
    opt = Gefen([("w", p)], lr=torch.tensor(1e-2), fused=False)
    p.grad = torch.randn_like(p) * 0.1
    before = p.detach().clone()
    opt.step()
    assert not torch.equal(before, p)
    assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# 7. validation symmetry (plain param lists vs dict groups)
# ---------------------------------------------------------------------------

def test_eps_validated_for_plain_param_lists():
    with pytest.raises(ValueError, match="epsilon"):
        Gefen([nn.Parameter(torch.randn(4))], eps=-1.0, fused=False)


def test_eps_validated_for_dict_groups():
    with pytest.raises(ValueError, match="epsilon"):
        Gefen(
            [{"params": [nn.Parameter(torch.randn(4))], "eps": -1.0}],
            fused=False,
        )


def test_multi_element_tensor_lr_rejected():
    with pytest.raises(ValueError, match="1-element"):
        Gefen(
            [nn.Parameter(torch.randn(4))],
            lr=torch.tensor([1e-3, 1e-4]),
            fused=False,
        )


# ---------------------------------------------------------------------------
# 8. prints -> logging / warnings
# ---------------------------------------------------------------------------

def test_fused_downgrade_warns_and_init_is_silent(capsys):
    if torch.cuda.is_available():
        pytest.skip("exercises the CUDA-unavailable downgrade")
    with pytest.warns(UserWarning, match="Changing fused to False"):
        Gefen([nn.Parameter(torch.randn(4))], fused=True)
    assert capsys.readouterr().out == ""


def test_construction_does_not_print(capsys):
    Gefen([nn.Parameter(torch.randn(4))], fused=False)
    assert capsys.readouterr().out == ""


def test_pure_python_dp_fallback_warns(monkeypatch):
    def _no_numba():
        raise ModuleNotFoundError("No module named 'numba'")

    monkeypatch.setattr(quantization, "_get_exact_dp_numba_kernel", _no_numba)
    stats = {
        "active_centers": torch.tensor([-1.0, -0.5, 0.5, 1.0]),
        "active_counts": torch.tensor([4.0, 2.0, 2.0, 4.0]),
    }
    with pytest.warns(UserWarning, match="numba"):
        codebook = quantization.exact_dp(stats, num_codebooks=2)
    assert codebook.numel() == 2


# ---------------------------------------------------------------------------
# 9. fp16 second-moment overflow
# ---------------------------------------------------------------------------

def test_fp16_vmean_does_not_overflow():
    vmean = torch.zeros(2, 1, dtype=torch.float32)
    grad_view = torch.full((2, 8), 300.0, dtype=torch.float16)  # 300^2 > fp16 max
    automatic_vmean_update(vmean, grad_view, 0.999, use_fused=False)
    assert torch.isfinite(vmean).all()
    expected = torch.full((2, 1), 300.0**2 * (1 - 0.999))
    assert torch.allclose(vmean, expected, rtol=1e-3)


def test_fp16_vmean_chunked_path_does_not_overflow(monkeypatch):
    monkeypatch.setattr(gefen_mod, "GEFEN_MOMENTUM_UPDATE_CHUNK", 8)
    vmean = torch.zeros(4, 1, dtype=torch.float32)
    grad_view = torch.full((4, 8), 300.0, dtype=torch.float16)
    automatic_vmean_update(vmean, grad_view, 0.999, use_fused=False)
    assert torch.isfinite(vmean).all()


def test_bf16_vmean_bit_identical_to_model_dtype_square():
    """bf16 must keep the exact model-dtype square (fused-kernel parity)."""
    torch.manual_seed(0)
    g = (torch.randn(4, 16) * 3).to(torch.bfloat16)
    vmean = torch.rand(4, 1, dtype=torch.float32)
    expected = vmean.clone()
    beta2 = 0.999
    block = torch.mean(g * g, dim=1, keepdim=True)
    expected.mul_(beta2).add_(block, alpha=1 - beta2)
    automatic_vmean_update(vmean, g, beta2, use_fused=False)
    assert torch.equal(vmean, expected)


# ---------------------------------------------------------------------------
# 10. legacy checkpoints with the removed m_codebook_shape key still load
# ---------------------------------------------------------------------------

def test_load_state_dict_tolerates_legacy_m_codebook_shape_key():
    def make():
        torch.manual_seed(0)
        model = nn.Linear(8, 8)
        opt = Gefen(list(model.named_parameters()), lr=1e-2, fused=False)
        return model, opt

    model, opt = make()
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 0.1
    opt.step()
    sd = opt.state_dict()
    for pstate in sd["state"].values():
        pstate["m_codebook_shape"] = tuple(pstate["m_codebook"].shape)
    model2, opt2 = make()
    opt2.load_state_dict(sd)
    for p in model2.parameters():
        p.grad = torch.randn_like(p) * 0.1
    opt2.step()  # the stale key must be inert
    assert all(torch.isfinite(p).all() for p in model2.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
