"""Regression: the period-search backend must be resolved per call from the
tensor's own device, never cached in the module global.

Caching a device-derived value let the first tensor's device leak into every
later call: a CUDA optimizer step set FIND_PERIOD_BACKEND="cuda_kernel", which
then poisoned a subsequent CPU parameter's period search ("cuda_kernel" backend
on a CPU tensor -> ValueError). Manifested as test_lr_calibration failures when a
CUDA test ran first in the same process.
"""
import pytest
import torch

import gefen.gefen as g


def test_resolve_does_not_mutate_global():
    saved = g.FIND_PERIOD_BACKEND
    try:
        g.FIND_PERIOD_BACKEND = None
        assert g._resolve_find_period_backend(torch.zeros(8)) == "cpu"
        # The bug: resolving used to write the global, leaking into later calls.
        assert g.FIND_PERIOD_BACKEND is None
    finally:
        g.FIND_PERIOD_BACKEND = saved


def test_explicit_override_respected():
    saved = g.FIND_PERIOD_BACKEND
    try:
        g.FIND_PERIOD_BACKEND = "cpu"  # explicit override wins regardless of device
        assert g._resolve_find_period_backend(torch.zeros(8)) == "cpu"
    finally:
        g.FIND_PERIOD_BACKEND = saved


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_then_cpu_no_leak():
    saved = g.FIND_PERIOD_BACKEND
    try:
        g.FIND_PERIOD_BACKEND = None
        assert g._resolve_find_period_backend(torch.zeros(8, device="cuda")) == "cuda_kernel"
        # Under the bug this returned "cuda_kernel" (leaked) and later crashed.
        assert g._resolve_find_period_backend(torch.zeros(8)) == "cpu"
    finally:
        g.FIND_PERIOD_BACKEND = saved
