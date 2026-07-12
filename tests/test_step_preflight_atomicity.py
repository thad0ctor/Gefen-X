"""Step preflight and periodic-codebook refresh transaction atomicity."""

import copy
import warnings

import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon, GefenMuonHybrid
import gefen.gefen as gefen_module


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _assert_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype
        assert actual.layout == expected.layout
        if actual.layout == torch.strided:
            assert torch.equal(actual, expected)
        else:
            assert torch.equal(actual.to_dense(), expected.to_dense())
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equal(actual_item, expected_item)
        return
    assert actual == expected


def _gefen_children(optimizer):
    if isinstance(optimizer, GefenMuonHybrid):
        return [
            child
            for child in (optimizer.muon, optimizer.backup)
            if isinstance(child, Gefen)
        ]
    return [optimizer] if isinstance(optimizer, Gefen) else []


def _snapshot(optimizer, params):
    runtime = []
    for child in _gefen_children(optimizer):
        runtime.append(
            {
                "global_step": child._gefen_global_step,
                "codebook": _clone(child._gefen_codebook),
                "codebook_by_device": _clone(child._gefen_codebook_by_device),
                "codebook_lut_by_device": _clone(
                    child._gefen_codebook_lut_by_device
                ),
                "sr_seed_by_device": _clone(child._sr_seed_by_device),
            }
        )
    return {
        "params": [
            {
                "layout": param.layout,
                "dense": param.detach().to_dense().clone()
                if param.layout != torch.strided
                else param.detach().clone(),
            }
            for param in params
        ],
        "state_dict": _clone(optimizer.state_dict()),
        "runtime": runtime,
    }


def _assert_snapshot(optimizer, params, expected):
    for param, saved in zip(params, expected["params"]):
        assert param.layout == saved["layout"]
        actual = param.detach().to_dense() if param.layout != torch.strided else param
        assert torch.equal(actual, saved["dense"])
    _assert_equal(optimizer.state_dict(), expected["state_dict"])
    runtime = []
    for child in _gefen_children(optimizer):
        runtime.append(
            {
                "global_step": child._gefen_global_step,
                "codebook": _clone(child._gefen_codebook),
                "codebook_by_device": _clone(child._gefen_codebook_by_device),
                "codebook_lut_by_device": _clone(
                    child._gefen_codebook_lut_by_device
                ),
                "sr_seed_by_device": _clone(child._sr_seed_by_device),
            }
        )
    _assert_equal(runtime, expected["runtime"])


def _invalid_tensor(layout, shape=(8, 8), device="cpu"):
    dense = torch.eye(*shape, device=device)
    if layout == "coo":
        return dense.to_sparse()
    if layout == "csr":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return dense.to_sparse_csr()
    if layout == "mkldnn":
        return dense.to_mkldnn()
    raise AssertionError(layout)


def _make_case(kind, layout, *, device="cpu", fused=False):
    generator = torch.Generator().manual_seed(991)
    first = nn.Parameter(
        (torch.randn(8, 8, generator=generator) * 0.02).to(device)
    )
    invalid = nn.Parameter(_invalid_tensor(layout, device=device))
    if kind == "gefen":
        optimizer = Gefen(
            [("first", first), ("invalid", invalid)],
            lr=2e-3,
            fused=fused,
            factored_v_2d=False,
        )
    elif kind == "muon":
        optimizer = GefenMuon(
            [("first", first), ("invalid", invalid)],
            lr=2e-3,
            fused=fused,
            ns_steps=1,
        )
    elif kind in ("hybrid_gefen", "hybrid_adamw"):
        optimizer = GefenMuonHybrid(
            [("first", first)],
            [("invalid", invalid)],
            lr=2e-3,
            fused=fused,
            ns_steps=1,
            ns_schedule="standard",
            normuon=False,
            backup_optimizer=kind.removeprefix("hybrid_"),
        )
    else:
        raise AssertionError(kind)
    return optimizer, [first, invalid]


@pytest.mark.parametrize("phase", ["first_step", "initialized"])
@pytest.mark.parametrize("layout", ["coo", "csr", "mkldnn"])
@pytest.mark.parametrize(
    "kind", ["gefen", "muon", "hybrid_gefen", "hybrid_adamw"]
)
def test_later_invalid_gradient_is_rejected_before_any_mutation(
    kind, layout, phase
):
    optimizer, params = _make_case(kind, layout)
    first, invalid = params
    if phase == "initialized":
        first.grad = torch.full_like(first, 0.002)
        invalid.grad = None
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    first.grad = torch.full_like(first, -0.003)
    invalid.grad = _invalid_tensor(layout)
    before = _snapshot(optimizer, params)
    with pytest.raises(RuntimeError, match="sparse|strided|layout"):
        optimizer.step()
    _assert_snapshot(optimizer, params, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize(
    "kind", ["gefen", "muon", "hybrid_gefen", "hybrid_adamw"]
)
def test_cuda_initialized_preflight_is_atomic_for_fused_and_unfused(kind, fused):
    optimizer, params = _make_case(
        kind, "coo", device="cuda", fused=fused
    )
    first, invalid = params
    first.grad = torch.full_like(first, 0.002)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    first.grad = torch.full_like(first, -0.003)
    invalid.grad = _invalid_tensor("coo", device="cuda")
    before = _snapshot(optimizer, params)
    with pytest.raises(RuntimeError, match="sparse|strided|layout"):
        optimizer.step()
    _assert_snapshot(optimizer, params, before)


def test_periodic_codebook_refresh_stages_every_index_before_commit(monkeypatch):
    generator = torch.Generator().manual_seed(997)
    params = [
        nn.Parameter(torch.randn(8, 8, generator=generator) * 0.02)
        for _ in range(2)
    ]
    optimizer = Gefen(
        [("first", params[0]), ("second", params[1])],
        lr=2e-3,
        fused=False,
        factored_v_2d=False,
        codebook_refresh_every=1,
    )
    for index, param in enumerate(params):
        param.grad = torch.full_like(param, 0.001 * (index + 1))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    first_indices = optimizer.state[params[0]]["m_codebook"].detach().clone()
    calls = 0

    def fail_on_second(codebook, coefficients):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (255 - first_indices.to(torch.int16)).to(torch.uint8)
        raise RuntimeError("injected second-parameter requantization failure")

    monkeypatch.setattr(
        gefen_module, "gefen_nearest_codebook_indices", fail_on_second
    )
    for index, param in enumerate(params):
        param.grad = torch.full_like(param, -0.003 * (index + 1))
    before = _snapshot(optimizer, params)

    with pytest.raises(RuntimeError, match="second-parameter requantization"):
        optimizer.step()

    assert calls == 2
    _assert_snapshot(optimizer, params, before)
