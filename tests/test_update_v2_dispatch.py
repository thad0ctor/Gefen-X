"""
Behavior of the occupancy-driven v1/v2 dispatch
(gefen.kernels.automatic_gefen_fused._should_use_v2).

Rule (env GEFEN_UPDATE_V2 unset): use v2 iff period <= _V2_PERIOD_MAX OR
num_blocks < _V2_BLOCKS_PER_SM * SM_count. Env GEFEN_UPDATE_V2 hard-overrides
(1=v2, 0=v1) regardless of shape. Both kernels are bit-identical, so this only
affects speed; the dispatch is calibrated in fused_bench/PERF.md.

Run: python tests/test_update_v2_dispatch.py
"""
import importlib
import os

import torch


def reload_with_env(value):
    if value is None:
        os.environ.pop("GEFEN_UPDATE_V2", None)
    else:
        os.environ["GEFEN_UPDATE_V2"] = value
    import gefen.kernels.automatic_gefen_fused as m
    return importlib.reload(m)


def main():
    ok = True

    # 1. env override forces the decision regardless of shape/device.
    for forced, expect in (("1", True), ("0", False), ("true", True), ("off", False)):
        m = reload_with_env(forced)
        dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        # both a "v1-shaped" and a "v2-shaped" call must follow the override.
        got_mid = m._should_use_v2(100_000, 512, dev)   # well-occupied -> v1 by heuristic
        got_few = m._should_use_v2(4, 1_048_576, dev)   # few-block -> v2 by heuristic
        good = got_mid == expect and got_few == expect
        ok = ok and good
        print(f"[env={forced}] mid={got_mid} few={got_few} expect={expect} "
              f"{'PASS' if good else 'FAIL'}")

    # 2. env unset: heuristic decides from num_blocks / period / SM count.
    if torch.cuda.is_available():
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
    else:
        print("[env unset] no CUDA, skipping heuristic branch")

    # restore unset so other tests / sessions see the default heuristic.
    reload_with_env(None)
    print("\nDISPATCH OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
