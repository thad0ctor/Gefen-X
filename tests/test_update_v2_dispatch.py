"""Behavior of the legacy and fully-fused v1/v2 dispatchers.

With ``GEFEN_UPDATE_V2`` unset, ``_should_use_v2`` retains the legacy
SM-normalized occupancy rule while ``_should_use_v2_full`` uses an independent
shape rule measured against the current full kernels. The environment variable
hard-overrides both dispatchers (1=v2, 0=v1).

The legacy pair is bit-identical. The full pair reduces block-vmean sums in
different orders, so route changes permit documented sub-ULP vmean/p differences;
``test_fused_update_v2_full_parity.py`` pins that tolerance directly.

Run: pytest tests/test_update_v2_dispatch.py  (or python tests/test_update_v2_dispatch.py)
"""
import importlib
import os

import torch

try:
    import pytest
except ImportError:  # script mode
    pytest = None


def reload_with_env(value):
    if value is None:
        os.environ.pop("GEFEN_UPDATE_V2", None)
    else:
        os.environ["GEFEN_UPDATE_V2"] = value
    import gefen.kernels.automatic_gefen_fused as m
    return importlib.reload(m)


def check_env_override():
    """env override forces both dispatchers regardless of shape/device."""
    ok = True
    try:
        for forced, expect in (("1", True), ("0", False), ("true", True), ("off", False)):
            m = reload_with_env(forced)
            dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            # both a "v1-shaped" and a "v2-shaped" call must follow the override.
            got_mid = m._should_use_v2(100_000, 512, dev)   # well-occupied -> v1 by heuristic
            got_few = m._should_use_v2(4, 1_048_576, dev)   # few-block -> v2 by heuristic
            got_full_mid = m._should_use_v2_full(100_000, 512, dev)
            got_full_few = m._should_use_v2_full(4, 1_048_576, dev)
            good = (
                got_mid == expect
                and got_few == expect
                and got_full_mid == expect
                and got_full_few == expect
            )
            ok = ok and good
            print(f"[env={forced}] legacy=({got_mid},{got_few}) "
                  f"full=({got_full_mid},{got_full_few}) expect={expect} "
                  f"{'PASS' if good else 'FAIL'}")
    finally:
        # restore unset so other tests / sessions see the default heuristic.
        reload_with_env(None)
    return ok


def check_heuristic_cuda():
    """env unset: heuristic decides from num_blocks / period / SM count."""
    ok = True
    m = reload_with_env(None)
    dev = torch.device("cuda", 0)
    sm = m._sm_count(dev)
    plo = m._V2_PERIOD_MAX
    bps = m._V2_BLOCKS_PER_SM
    # well-occupied middle: large period, many blocks -> v1
    c_mid = m._should_use_v2(bps * sm * 8, plo * 4, dev) is False
    # tiny period (warp underfill) -> v2 regardless of block count
    c_small_p = m._should_use_v2(bps * sm * 8, plo, dev) is True
    c_p1 = m._should_use_v2(10_000_000, 1, dev) is True
    # few blocks (large period) -> v2 even with full warps
    c_few = m._should_use_v2(bps * sm - 1, 100_000, dev) is True
    # right at the occupancy threshold -> v1 (>= bps*sm with full warp)
    c_thresh = m._should_use_v2(bps * sm, plo + 1, dev) is False
    for label, c in [("mid->v1", c_mid), ("small_period->v2", c_small_p),
                     ("period1->v2", c_p1), ("few_blocks->v2", c_few),
                     ("at_threshold->v1", c_thresh)]:
        ok = ok and c
        print(f"[env unset, sm={sm}] {label}: {'PASS' if c else 'FAIL'}")
    return ok


def check_full_heuristic_cuda():
    """The current full pair has a separately measured crossover."""
    m = reload_with_env(None)
    dev = torch.device("cuda", 0)
    cases = [
        ("tiny-period1->v1", 4096, 1, False),
        ("large-period1->v2", 1_000_000, 1, True),
        ("period12-small->v1", 8192, 12, False),
        ("period16->v1", 262_144, 16, False),
        ("period16-huge->v2", 1_048_576, 16, True),
        ("period512-few->v1", 4, 512, False),
        ("large-period->v2", 512, 8192, True),
    ]
    ok = True
    for label, num_blocks, period, expected in cases:
        got = m._should_use_v2_full(num_blocks, period, dev)
        good = got is expected
        ok = ok and good
        print(f"[full dispatch] {label}: got={'v2' if got else 'v1'} "
              f"{'PASS' if good else 'FAIL'}")
    return ok


def check_capturable_tiny_v1_override_cuda():
    """Capturable tiny params take the ONE-kernel v1-full path where v2 routes.

    ``_CAPT_V1_FULL_TINY_NUMEL`` only applies under ``capturable=True``: tiny
    params are launch-bound inside a captured/replayed graph, so the optimizer
    overrides an affirmative ``_should_use_v2_full`` verdict back to v1-full.
    Spy on the optimizer's two full-update entry points to assert the actual
    route CHOICE (not just numeric parity).
    """
    from gefen.gefen import Gefen
    import gefen.kernels.automatic_gefen_fused as m

    tiny = m._CAPT_V1_FULL_TINY_NUMEL  # threshold is inclusive (<=)
    big = tiny * 2
    dev = torch.device("cuda", 0)
    # Shape sanity: with force_1d_period_one the grad view is [numel, 1], and
    # BOTH sizes route v2 by the shape heuristic alone -- so any v1 call below
    # can only come from the capturable tiny-numel override.
    assert m._should_use_v2_full(tiny, 1, dev) is True
    assert m._should_use_v2_full(big, 1, dev) is True

    def routed(numel, capturable):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.randn(numel, device="cuda") * 0.02)
        opt = Gefen(
            [p],
            lr=1e-3,
            fused=True,
            capturable=capturable,
            force_1d_period_one=True,
        )
        calls = []
        orig_v1 = opt._automatic_gefen_fused_full_update
        orig_v2 = opt._automatic_gefen_fused_update_v2_full

        def spy_v1(*args, **kwargs):
            calls.append("v1")
            return orig_v1(*args, **kwargs)

        def spy_v2(*args, **kwargs):
            calls.append("v2")
            return orig_v2(*args, **kwargs)

        opt._automatic_gefen_fused_full_update = spy_v1
        opt._automatic_gefen_fused_update_v2_full = spy_v2
        for step in range(1, 4):
            torch.manual_seed(step)
            p.grad = torch.randn_like(p) * 0.01
            opt.step()
        assert calls, "fused full-update path never engaged"
        return set(calls)

    ok = True
    cases = [
        ("capturable tiny -> v1", tiny, True, {"v1"}),
        ("capturable above-threshold -> v2", big, True, {"v2"}),
        ("eager tiny -> v2 (override is capturable-only)", tiny, False, {"v2"}),
    ]
    for label, numel, capturable, expect in cases:
        got = routed(numel, capturable)
        good = got == expect
        ok = ok and good
        print(f"[capt tiny override] {label}: got={sorted(got)} "
              f"expect={sorted(expect)} {'PASS' if good else 'FAIL'}")
    return ok


if pytest is not None:

    def test_env_override_forces_dispatch():
        assert check_env_override()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_heuristic_dispatch_cuda():
        assert check_heuristic_cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_full_heuristic_dispatch_cuda():
        assert check_full_heuristic_cuda()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_capturable_tiny_v1_override_cuda():
        assert check_capturable_tiny_v1_override_cuda()


def main():
    ok = check_env_override()
    if torch.cuda.is_available():
        ok = ok and check_heuristic_cuda()
        ok = ok and check_full_heuristic_cuda()
        ok = ok and check_capturable_tiny_v1_override_cuda()
    else:
        print("[env unset] no CUDA, skipping heuristic branch")
    print("\nDISPATCH OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
