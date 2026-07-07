"""CPU-only coverage for the kernel-dispatch gating / toolchain-fallback scrub
(Batch 3).

These exercise the device-independent halves of the graceful-fallback machinery:

* M8  -- the lazy fused-toolchain probe catches an extension build/load failure,
         warns exactly once, and permanently downgrades the instance to the
         pure-torch path (the mixed-placement / fp64-fallback GPU behaviors are
         verified separately on a CUDA device).
* m14 -- an empty-string env override (GEFEN_UPDATE_V2="") means UNSET, not a
         truthy force.
* items 2/3 -- an fp64 (and plain fp32) parameter trains through the pure-torch
         path on CPU without hitting a CUDA binding.

Run: pytest tests/test_dispatch_gating_cpu.py
"""
import importlib
import os
import warnings

import pytest
import torch
import torch.nn as nn

import gefen
from gefen import Gefen
import gefen.gefen as gefen_mod


def _cpu_opt(**kwargs):
    # On a CPU-only construction fused auto-flips to False; we drive the probe
    # helper directly for the M8 tests, so the constructor value does not matter.
    p = nn.Parameter(torch.randn(4, 4))
    return Gefen([("w", p)], lr=1e-3, **kwargs)


# ---------------------------------------------------------------------------
# M8 -- fused-toolchain probe: catch, warn once, downgrade
# ---------------------------------------------------------------------------

def test_toolchain_probe_downgrades_on_build_failure(monkeypatch):
    opt = _cpu_opt()
    # Simulate a CUDA box with fused configured on but a broken toolchain.
    opt.fused = True
    opt._fused_build_ok = None

    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("nvcc not found")

    monkeypatch.setattr(gefen_mod, "_ensure_gefen_fused_extension_loaded", boom)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = opt._gefen_fused_toolchain_ok()

    assert ok is False
    assert opt.fused is False  # permanently downgraded
    assert opt._fused_build_ok is False
    assert calls["n"] == 1
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) == 1
    assert "fused=False" in str(user_warnings[0].message)

    # Cached: a second probe must not re-load, re-warn, or re-raise.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        ok2 = opt._gefen_fused_toolchain_ok()
    assert ok2 is False
    assert calls["n"] == 1
    assert not [w for w in caught2 if issubclass(w.category, UserWarning)]

    # The public availability gate reflects the downgrade too.
    assert opt._fused_kernels_available() is False


def test_toolchain_probe_happy_path(monkeypatch):
    opt = _cpu_opt()
    opt.fused = True
    opt._fused_build_ok = None

    calls = {"n": 0}

    def ok_load():
        calls["n"] += 1  # a healthy box: build/load succeeds

    monkeypatch.setattr(gefen_mod, "_ensure_gefen_fused_extension_loaded", ok_load)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert opt._gefen_fused_toolchain_ok() is True
        assert opt._gefen_fused_toolchain_ok() is True  # cached, no re-load

    assert opt.fused is True
    assert opt._fused_build_ok is True
    assert calls["n"] == 1  # probed exactly once
    assert not [w for w in caught if issubclass(w.category, UserWarning)]


def test_toolchain_probe_noop_when_not_configured(monkeypatch):
    # fused already off -> the probe must never touch the loader.
    opt = _cpu_opt()
    opt.fused = False

    def boom():
        raise AssertionError("loader must not be called when fused is off")

    monkeypatch.setattr(gefen_mod, "_ensure_gefen_fused_extension_loaded", boom)
    assert opt._gefen_fused_toolchain_ok() is False
    assert opt._fused_kernels_available() is False


def test_construction_never_probes(monkeypatch):
    # Even stochastic_round=True (which queries the fused intent in __init__)
    # must not trigger the toolchain probe / JIT build at construction time.
    def boom():
        raise AssertionError("construction must not probe the toolchain")

    monkeypatch.setattr(gefen_mod, "_ensure_gefen_fused_extension_loaded", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = _cpu_opt(stochastic_round=True)
    assert opt._fused_build_ok is None  # unprobed


# ---------------------------------------------------------------------------
# m14 -- empty-string env override means UNSET
# ---------------------------------------------------------------------------

def _reload_fused_with_env(value):
    if value is None:
        os.environ.pop("GEFEN_UPDATE_V2", None)
    else:
        os.environ["GEFEN_UPDATE_V2"] = value
    import gefen.kernels.automatic_gefen_fused as m
    return importlib.reload(m)


def test_empty_env_is_unset():
    try:
        m = _reload_fused_with_env("")
        assert m._GEFEN_UPDATE_V2_ENV is None
        # whitespace-only is also unset
        m = _reload_fused_with_env("   ")
        assert m._GEFEN_UPDATE_V2_ENV is None
        # a real value survives (and strips surrounding whitespace)
        m = _reload_fused_with_env(" 1 ")
        assert m._GEFEN_UPDATE_V2_ENV == "1"
        m = _reload_fused_with_env("0")
        assert m._GEFEN_UPDATE_V2_ENV == "0"
    finally:
        _reload_fused_with_env(None)


def test_env_override_helper():
    m = _reload_fused_with_env(None)
    try:
        os.environ["GEFEN_TEST_ENV_OVERRIDE"] = ""
        assert m._env_override("GEFEN_TEST_ENV_OVERRIDE") is None
        os.environ["GEFEN_TEST_ENV_OVERRIDE"] = "  "
        assert m._env_override("GEFEN_TEST_ENV_OVERRIDE") is None
        os.environ["GEFEN_TEST_ENV_OVERRIDE"] = "on"
        assert m._env_override("GEFEN_TEST_ENV_OVERRIDE") == "on"
        os.environ.pop("GEFEN_TEST_ENV_OVERRIDE", None)
        assert m._env_override("GEFEN_TEST_ENV_OVERRIDE") is None
    finally:
        os.environ.pop("GEFEN_TEST_ENV_OVERRIDE", None)


# ---------------------------------------------------------------------------
# items 2/3 -- fp64 / fp32 params train through the pure-torch path on CPU
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_step_trains_dtype(dtype):
    torch.manual_seed(0)
    w = nn.Parameter(torch.randn(16, 16, dtype=dtype))
    # Force the pure-torch path explicitly so the assertion holds whether or not
    # a CUDA device is visible (a CPU param routes per-param to pure-torch even
    # under fused=True, but fused stays True on a GPU box).
    opt = Gefen([("w", w)], lr=1e-2, fused=False)
    assert opt.fused is False
    before = w.detach().clone()
    for _ in range(3):
        opt.zero_grad()
        loss = (w.float() ** 2).sum()
        loss.backward()
        opt.step()
    assert w.dtype == dtype
    assert torch.isfinite(w).all()
    assert not torch.equal(w.detach(), before)  # params actually moved
