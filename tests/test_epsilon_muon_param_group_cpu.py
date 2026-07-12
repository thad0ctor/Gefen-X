"""CPU regressions for epsilon safety and GefenMuon group registration."""

import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon


NON_POSITIVE_OR_NONFINITE = [
    pytest.param(0.0, id="zero"),
    pytest.param(-1e-8, id="negative"),
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="positive-infinity"),
    pytest.param(float("-inf"), id="negative-infinity"),
]


@pytest.mark.parametrize("eps", NON_POSITIVE_OR_NONFINITE)
def test_gefen_requires_finite_positive_eps(eps):
    with pytest.raises(ValueError, match="eps|epsilon"):
        Gefen([nn.Parameter(torch.ones(8))], eps=eps, fused=False)


@pytest.mark.parametrize("argument", ["eps", "normuon_eps"])
@pytest.mark.parametrize("value", NON_POSITIVE_OR_NONFINITE)
def test_gefen_muon_requires_finite_positive_epsilon_values(argument, value):
    with pytest.raises(ValueError, match=argument):
        GefenMuon(
            [nn.Parameter(torch.ones(8, 8))],
            fused=False,
            **{argument: value},
        )


def test_gefen_zero_gradient_step_stays_finite():
    param = nn.Parameter(torch.randn(8))
    before = param.detach().clone()
    opt = Gefen([("weight", param)], eps=1e-8, weight_decay=0.0, fused=False)
    param.grad = torch.zeros_like(param)

    opt.step()

    assert torch.equal(param, before)
    assert torch.isfinite(param).all()


def test_gefen_muon_normuon_zero_gradient_step_stays_finite():
    param = nn.Parameter(torch.randn(8, 8))
    before = param.detach().clone()
    opt = GefenMuon(
        [("weight", param)],
        eps=1e-7,
        normuon=True,
        normuon_eps=1e-8,
        weight_decay=0.0,
        fused=False,
    )
    param.grad = torch.zeros_like(param)

    opt.step()

    assert torch.equal(param, before)
    assert torch.isfinite(param).all()


def test_gefen_muon_add_param_group_populates_all_options_and_steps():
    torch.manual_seed(0)
    first = nn.Parameter(torch.randn(8, 8))
    opt = GefenMuon([("first", first)], weight_decay=0.0, fused=False)

    # Establish the frozen codebook before adding the new group. The new matrix
    # must initialize against that existing codebook without missing group keys.
    first.grad = torch.randn_like(first)
    opt.step()
    opt.zero_grad()

    added = nn.Parameter(torch.randn(8, 8))
    opt.add_param_group(
        {
            "params": [("added", added)],
            "lr": 2e-4,
            "eps": 2e-7,
            "weight_decay": 0.0,
            "momentum": 0.8,
            "nesterov": False,
            "ns_schedule": "tuned3",
            "adjust_lr_fn": "match_rms_adamw",
            "sharded_mode": "approx",
            "fp8_ns": False,
            "fp8_ns_compile": False,
            "batched_ns": True,
            "batched_ns_workspace_bytes": 1 << 20,
            "normuon": True,
            "normuon_beta2": 0.9,
            "normuon_eps": 1e-6,
            "cautious": True,
        }
    )

    group = opt.param_groups[-1]
    assert group["param_names"] == ["added"]
    assert group["lr"] == 2e-4
    assert group["eps"] == 2e-7
    assert group["weight_decay"] == 0.0
    assert group["beta1"] == group["momentum"] == 0.8
    assert group["beta2"] == 0.0
    assert group["nesterov"] is False
    assert group["ns_steps"] == 3
    assert len(group["ns_coefficients"]) == 3
    assert group["adjust_lr_fn"] == "match_rms_adamw"
    assert group["sharded_mode"] == "approx"
    assert group["fp8_ns"] is False
    assert group["fp8_ns_compile"] is False
    assert group["batched_ns"] is True
    assert group["batched_ns_workspace_bytes"] == 1 << 20
    assert group["normuon"] is True
    assert group["normuon_beta2"] == 0.9
    assert group["normuon_eps"] == 1e-6
    assert group["cautious"] is True

    first.grad = torch.randn_like(first)
    added.grad = torch.randn_like(added)
    before = added.detach().clone()
    opt.step()

    assert "step" in opt.state[added]
    assert torch.isfinite(added).all()
    assert not torch.equal(added, before)


def test_gefen_muon_add_param_group_rejects_non_2d_atomically():
    opt = GefenMuon([nn.Parameter(torch.randn(8, 8))], fused=False)
    valid = nn.Parameter(torch.randn(8, 8))
    invalid = nn.Parameter(torch.randn(8))
    group_count = len(opt.param_groups)

    with pytest.raises(ValueError, match="only supports 2D"):
        opt.add_param_group({"params": [("valid", valid), ("invalid", invalid)]})

    assert len(opt.param_groups) == group_count
    assert valid not in opt.state


@pytest.mark.parametrize("normuon_eps", NON_POSITIVE_OR_NONFINITE)
def test_gefen_muon_add_param_group_rejects_bad_normuon_eps_atomically(
    normuon_eps,
):
    opt = GefenMuon([nn.Parameter(torch.randn(8, 8))], fused=False)
    added = nn.Parameter(torch.randn(8, 8))
    group_count = len(opt.param_groups)

    with pytest.raises(ValueError, match="normuon_eps"):
        opt.add_param_group({"params": [added], "normuon_eps": normuon_eps})

    assert len(opt.param_groups) == group_count
    assert added not in opt.state
