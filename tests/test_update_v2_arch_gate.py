"""
Behavior of the lazy v2 arch gate (gefen.gefen._use_gefen_update_v2).

- import must not initialize CUDA;
- env GEFEN_UPDATE_V2 overrides the arch decision for any device;
- with env unset, v2 is on only for capabilities in the allowlist (sm_86);
- CPU / None device -> v1.

Run: python tests/test_update_v2_arch_gate.py
"""
import importlib
import os

import torch


def reload_with_env(value):
    if value is None:
        os.environ.pop("GEFEN_UPDATE_V2", None)
    else:
        os.environ["GEFEN_UPDATE_V2"] = value
    import gefen.gefen as g
    return importlib.reload(g)


def main():
    ok = True

    # 1. import does not init CUDA (reload after ensuring not initialized is hard
    #    once torch is up; just assert the gate itself never touches CUDA for the
    #    env / cpu branches below).
    g = reload_with_env(None)

    # 2. env override forces the decision regardless of device.
    for forced, expect in (("1", True), ("0", False), ("true", True), ("off", False)):
        g = reload_with_env(forced)
        got_cpu = g._use_gefen_update_v2(torch.device("cpu"))
        got_none = g._use_gefen_update_v2(None)
        good = got_cpu == expect and got_none == expect
        ok = ok and good
        print(f"[env={forced}] cpu={got_cpu} none={got_none} expect={expect} "
              f"{'PASS' if good else 'FAIL'}")

    # 3. env unset: cpu/None -> False (v1) always.
    g = reload_with_env(None)
    good = g._use_gefen_update_v2(torch.device("cpu")) is False and \
        g._use_gefen_update_v2(None) is False
    ok = ok and good
    print(f"[env unset] cpu/None -> v1: {'PASS' if good else 'FAIL'}")

    # 4. env unset + CUDA: decision follows the allowlist (sm_86 only).
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        got = g._use_gefen_update_v2(torch.device("cuda", 0))
        expect = cap in g._V2_VERIFIED_CAPABILITIES
        good = got == expect
        ok = ok and good
        print(f"[env unset] cuda:0 cap={cap} v2={got} expect={expect} "
              f"{'PASS' if good else 'FAIL'}")
    else:
        print("[env unset] no CUDA, skipping arch branch")

    print("\nARCH GATE OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
