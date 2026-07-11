import copy

import pytest
import torch

import gefen.gefen_muon as gm
from gefen.gefen_muon import GefenMuon
from gefen.hybrid import GefenMuonHybrid


def test_workspace_estimate_matches_documented_peak_model():
    assert gm._batched_ns_workspace_bytes(32, 128, 1024) == 26 << 20
    assert gm._batched_ns_workspace_bytes(32, 256, 2048) == 104 << 20
    assert gm._batched_ns_workspace_bytes(32, 512, 2048) == 224 << 20


@pytest.mark.parametrize(
    "shape,eligible",
    [
        ((512, 2048), True),
        ((2048, 512), True),
        ((256, 4096), True),
        ((513, 2044), False),  # four elements below 2**20, min dim too large
        ((512, 2049), False),  # min dim allowed, numel just over the gate
        ((1024, 1024), False),
        ((0, 1024), False),
    ],
)
def test_shape_gate_boundaries(shape, eligible):
    assert gm._batched_ns_shape_eligible(*shape) is eligible


@pytest.mark.parametrize(
    "count,max_batch,expected",
    [
        (7, 64, ([], 7)),
        (8, 8, ([8], 0)),
        (17, 10, ([9, 8], 0)),
        (17, 8, ([8, 8], 1)),
        (18, 10, ([10, 8], 0)),  # remainder itself is a legal batch, no rebalance
        (15, 8, ([8], 7)),
        (64, 7, ([], 64)),
    ],
)
def test_chunk_sizes_balance_or_leave_serial_tail(count, max_batch, expected):
    per_item = gm._batched_ns_workspace_bytes(1, 128, 1024)
    got = gm._batched_ns_chunk_sizes(
        count, 128, 1024, per_item * max_batch
    )
    assert got == expected
    sizes, tail = got
    assert sum(sizes) + tail == count
    assert all(8 <= size <= max_batch for size in sizes)


@pytest.mark.parametrize("shape", [(17, 31), (31, 17)])
def test_batched_helper_is_close_to_serial_for_both_orientations(shape):
    torch.manual_seed(4)
    inputs = torch.randn(8, *shape, dtype=torch.bfloat16) * 1e-3
    schedule = gm.NS_SCHEDULE_3STEP
    got = gm._zeropower_via_newtonschulz_batched(inputs.clone(), schedule, 3, 1e-7)
    ref = torch.stack(
        [
            gm._zeropower_via_newtonschulz(x, schedule, 3, 1e-7)
            for x in inputs
        ]
    )
    delta = got.float() - ref.float()
    rel_l2 = delta.norm() / ref.float().norm().clamp(min=1e-12)
    assert rel_l2.item() <= 2e-2
    assert delta.abs().max().item() <= 2e-2


def test_batched_helper_zero_input_is_finite_zero():
    inputs = torch.zeros(8, 13, 29, dtype=torch.bfloat16)
    got = gm._zeropower_via_newtonschulz_batched(
        inputs, (gm.DEFAULT_A, gm.DEFAULT_B, gm.DEFAULT_C), 5, 1e-7
    )
    assert torch.isfinite(got).all()
    assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("bad_shape", [(13, 29), (2, 8, 13, 29)])
def test_batched_helper_rejects_non_stacked_input(bad_shape):
    inputs = torch.zeros(*bad_shape, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"\[batch, rows, cols\]"):
        gm._zeropower_via_newtonschulz_batched(
            inputs, gm.NS_SCHEDULE_3STEP, 3, 1e-7
        )


def test_plain_constructor_defaults_batched_ns_off():
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = GefenMuon([("w", p)], fused=False)
    assert opt.param_groups[0]["batched_ns"] is False
    assert (
        opt.param_groups[0]["batched_ns_workspace_bytes"]
        == gm.BATCHED_NS_DEFAULT_WORKSPACE_BYTES
    )

    q = torch.nn.Parameter(torch.randn(8, 8))
    h = GefenMuonHybrid([("hidden.weight", q)], [], lr=1e-3, fused=False)
    assert h.muon.param_groups[0]["batched_ns"] is False


def test_constructor_and_hybrid_forward_opt_in():
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = GefenMuon(
        [("w", p)], fused=False, batched_ns=True,
        batched_ns_workspace_bytes=123456,
    )
    assert opt.param_groups[0]["batched_ns"] is True
    assert opt.param_groups[0]["batched_ns_workspace_bytes"] == 123456

    q = torch.nn.Parameter(torch.randn(8, 8))
    h = GefenMuonHybrid(
        [("hidden.weight", q)], [], lr=1e-3, fused=False,
        batched_ns=True, batched_ns_workspace_bytes=654321,
    )
    assert h.muon.param_groups[0]["batched_ns"] is True
    assert h.muon.param_groups[0]["batched_ns_workspace_bytes"] == 654321


def test_checkpoint_roundtrip_and_old_checkpoint_default_off():
    p = torch.nn.Parameter(torch.randn(8, 8))
    enabled = GefenMuon(
        [("w", p)], fused=False, batched_ns=True,
        batched_ns_workspace_bytes=123456,
    )
    current = enabled.state_dict()

    q = torch.nn.Parameter(torch.randn(8, 8))
    restored = GefenMuon([("w", q)], fused=False, batched_ns=False)
    restored.load_state_dict(copy.deepcopy(current))
    assert restored.param_groups[0]["batched_ns"] is True
    assert restored.param_groups[0]["batched_ns_workspace_bytes"] == 123456

    old = copy.deepcopy(current)
    old["param_groups"][0].pop("batched_ns")
    old["param_groups"][0].pop("batched_ns_workspace_bytes")
    r = torch.nn.Parameter(torch.randn(8, 8))
    migrated = GefenMuon(
        [("w", r)], fused=False, batched_ns=True,
        batched_ns_workspace_bytes=654321,
    )
    migrated.load_state_dict(old)
    assert migrated.param_groups[0]["batched_ns"] is False
    assert (
        migrated.param_groups[0]["batched_ns_workspace_bytes"]
        == gm.BATCHED_NS_DEFAULT_WORKSPACE_BYTES
    )


def test_sharded_and_compiling_keys_are_forced_serial(monkeypatch):
    p = torch.nn.Parameter(torch.randn(8, 8))
    opt = GefenMuon([("w", p)], fused=False, batched_ns=True)
    group = opt.param_groups[0]
    grad = torch.randn_like(p)

    class FakeSharded:
        to_local = object()
        placements = object()
        device_mesh = object()

    assert opt._batched_ns_key(
        (group, "w", FakeSharded(), grad), fused_available=True
    ) is None
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert opt._batched_ns_key(
        (group, "w", p, grad), fused_available=True
    ) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("gate", ["fp8_ns", "ndim", "cpu"])
def test_batched_ns_key_individual_ineligibility_gates(gate):
    p = torch.nn.Parameter(
        torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    )
    opt = GefenMuon([("w", p)], fused=False, batched_ns=True)
    group = opt.param_groups[0]
    grad = torch.randn_like(p)
    # Baseline: an eligible bf16 2D CUDA item batches, so each mutation below
    # flips exactly one gate.
    assert opt._batched_ns_key(
        (group, "w", p, grad), fused_available=True
    ) is not None

    if gate == "fp8_ns":
        group["fp8_ns"] = True
        item = (group, "w", p, grad)
    elif gate == "ndim":
        vec = torch.nn.Parameter(
            torch.randn(128, device="cuda", dtype=torch.bfloat16)
        )
        item = (group, "w", vec, torch.randn_like(vec))
    else:  # non-CUDA device
        cpu_p = torch.nn.Parameter(p.detach().cpu())
        item = (group, "w", cpu_p, grad.cpu())
    assert opt._batched_ns_key(item, fused_available=True) is None


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_workspace_cap_validation(bad):
    p = torch.nn.Parameter(torch.randn(8, 8))
    exc = TypeError if isinstance(bad, (float, bool)) else ValueError
    with pytest.raises(exc):
        GefenMuon(
            [("w", p)], fused=False, batched_ns=True,
            batched_ns_workspace_bytes=bad,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count,expected_batches", [(7, []), (8, [8])])
def test_optimizer_gate_uses_batch_at_eight_only(monkeypatch, count, expected_batches):
    calls = []
    original = gm._zeropower_via_newtonschulz_batched

    def wrapped(inputs, *args, **kwargs):
        calls.append(inputs.shape[0])
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(gm, "_zeropower_via_newtonschulz_batched", wrapped)
    torch.manual_seed(10)
    params = [
        (
            "layer{}.weight".format(i),
            torch.nn.Parameter(
                (torch.randn(128, 256, device="cuda") * 0.02).bfloat16()
            ),
        )
        for i in range(count)
    ]
    opt = GefenMuon(
        params,
        lr=1e-3,
        fused=True,
        ns_schedule="tuned3",
        batched_ns=True,
    )
    if count == 8:
        # A hand-built/newly-added group or partially migrated checkpoint may
        # carry the opt-in without its companion budget. Routing must use the
        # documented default rather than raising a fresh KeyError.
        opt.param_groups[0].pop("batched_ns_workspace_bytes")
    for _, p in params:
        p.grad = (torch.randn_like(p) * 1e-3).bfloat16()
    opt.step()
    torch.cuda.synchronize()
    assert calls == expected_batches
    for _, p in params:
        assert torch.isfinite(p).all()
        step = opt.state[p]["step"]
        assert int(step.item() if torch.is_tensor(step) else step) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "count,max_batch,expected_batches,expected_serial",
    [
        (17, 10, [9, 8], 0),
        (17, 8, [8, 8], 1),
    ],
)
def test_optimizer_workspace_chunk_routing(
    monkeypatch, count, max_batch, expected_batches, expected_serial
):
    batch_calls = []
    serial_calls = []
    original_batch = gm._zeropower_via_newtonschulz_batched
    original_serial = gm._zeropower_via_newtonschulz

    def batch_wrapped(inputs, *args, **kwargs):
        batch_calls.append(inputs.shape[0])
        return original_batch(inputs, *args, **kwargs)

    def serial_wrapped(inputs, *args, **kwargs):
        serial_calls.append(tuple(inputs.shape))
        return original_serial(inputs, *args, **kwargs)

    monkeypatch.setattr(gm, "_zeropower_via_newtonschulz_batched", batch_wrapped)
    monkeypatch.setattr(gm, "_zeropower_via_newtonschulz", serial_wrapped)
    torch.manual_seed(18)
    params = [
        (
            f"layer{i}.weight",
            torch.nn.Parameter(
                (torch.randn(64, 128, device="cuda") * 0.02).bfloat16()
            ),
        )
        for i in range(count)
    ]
    per_item = gm._batched_ns_workspace_bytes(1, 64, 128)
    opt = GefenMuon(
        params, lr=1e-3, fused=True, ns_schedule="tuned3",
        batched_ns=True,
        batched_ns_workspace_bytes=per_item * max_batch,
    )
    for _, p in params:
        p.grad = (torch.randn_like(p) * 1e-3).bfloat16()
    opt.step()
    torch.cuda.synchronize()
    assert batch_calls == expected_batches
    assert len(serial_calls) == expected_serial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "case",
    ("short_batch", "small_workspace", "fp16", "capturable", "nonfused"),
)
def test_ineligible_routes_are_bit_identical_to_serial(case):
    count = 7 if case == "short_batch" else 8
    dtype = torch.float16 if case == "fp16" else torch.bfloat16
    fused = case != "nonfused"
    capturable = case == "capturable"
    torch.manual_seed(20)
    initial = [
        (torch.randn(32, 64, device="cuda") * 0.02).to(dtype)
        for _ in range(count)
    ]
    grads = [
        (torch.randn(32, 64, device="cuda") * 1e-3).to(dtype)
        for _ in range(count)
    ]
    workspace = gm.BATCHED_NS_DEFAULT_WORKSPACE_BYTES
    if case == "small_workspace":
        workspace = gm._batched_ns_workspace_bytes(1, 32, 64) * 7

    def run(batched):
        params = [
            (f"p{i}", torch.nn.Parameter(value.clone()))
            for i, value in enumerate(initial)
        ]
        opt = GefenMuon(
            params, lr=1e-3, fused=fused, ns_schedule="tuned3",
            batched_ns=batched,
            batched_ns_workspace_bytes=workspace,
            capturable=capturable,
        )
        for (_, p), grad in zip(params, grads):
            p.grad = grad.clone()
        opt.step()
        torch.cuda.synchronize()
        return params, opt

    reference, reference_opt = run(False)
    candidate, candidate_opt = run(True)
    for (_, want), (_, got) in zip(reference, candidate):
        assert torch.equal(got, want)
        want_state = reference_opt.state[want]
        got_state = candidate_opt.state[got]
        assert torch.equal(got_state["m_codebook"], want_state["m_codebook"])
        assert torch.equal(got_state["m_magnitude"], want_state["m_magnitude"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(128, 256), (256, 128)])
def test_optimizer_batched_ns_preserves_momentum_state_and_stays_close(shape):
    torch.manual_seed(22)
    initial = [
        (torch.randn(*shape, device="cuda") * 0.02).bfloat16()
        for _ in range(8)
    ]
    grad_steps = [
        [
            (torch.randn(*shape, device="cuda") * 1e-3).bfloat16()
            for _ in range(8)
        ]
        for _ in range(2)
    ]

    def run(batched):
        params = [
            ("layer{}.weight".format(i), torch.nn.Parameter(value.clone()))
            for i, value in enumerate(initial)
        ]
        opt = GefenMuon(
            params, lr=0.05, fused=True, ns_schedule="tuned3",
            adjust_lr_fn="match_rms_adamw", batched_ns=batched,
            normuon=True, stochastic_round=True,
        )
        for grads in grad_steps:
            for (_, p), grad in zip(params, grads):
                p.grad = grad
            opt.step()
        torch.cuda.synchronize()
        return params, opt

    serial_params, serial_opt = run(False)
    batch_params, batch_opt = run(True)
    for (_, serial), (_, batched), start in zip(
        serial_params, batch_params, initial
    ):
        serial_state = serial_opt.state[serial]
        batch_state = batch_opt.state[batched]
        assert torch.equal(
            serial_state["m_codebook"], batch_state["m_codebook"]
        )
        assert torch.equal(
            serial_state["m_magnitude"], batch_state["m_magnitude"]
        )
        assert int(serial_state["normuon_step"]) == 2
        assert int(batch_state["normuon_step"]) == 2
        assert torch.isfinite(batch_state["normuon_v"]).all()
        delta = batched.float() - serial.float()
        serial_update = serial.float() - start.float()
        rel_l2 = delta.norm() / serial_update.norm().clamp(min=1e-12)
        assert rel_l2.item() <= 5e-2
        assert delta.abs().max().item() <= 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_ns_consumes_exact_fused_nesterov_output(monkeypatch):
    """Batch input is the exact old host Nesterov result, state stays EMA."""
    momentum = 0.9
    torch.manual_seed(33)
    initial = [
        (torch.randn(32, 64, device="cuda") * 0.02).bfloat16()
        for _ in range(8)
    ]
    grads = [
        (torch.randn(32, 64, device="cuda") * 1e-3).bfloat16()
        for _ in range(8)
    ]
    captured = []
    original = gm._zeropower_via_newtonschulz_batched

    def capture_input(inputs, *args, **kwargs):
        # The helper consumes its bf16 stack in place; clone before forwarding.
        captured.append(inputs.clone())
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(
        gm, "_zeropower_via_newtonschulz_batched", capture_input
    )

    def run(nesterov):
        params = [
            (f"p{i}", torch.nn.Parameter(value.clone()))
            for i, value in enumerate(initial)
        ]
        opt = GefenMuon(
            params,
            lr=0.05,
            momentum=momentum,
            nesterov=nesterov,
            fused=True,
            ns_schedule="tuned3",
            batched_ns=True,
        )
        assert opt._fused_kernels_available()
        for (_, p), grad in zip(params, grads):
            p.grad = grad.clone()
        opt.step()
        torch.cuda.synchronize()
        return params, opt

    plain_params, plain_opt = run(False)
    nested_params, nested_opt = run(True)
    assert len(captured) == 2
    expected = captured[0].clone()
    expected.mul_(momentum).add_(torch.stack(grads), alpha=1 - momentum)
    assert torch.equal(captured[1], expected)

    effect = 0.0
    for (_, plain), (_, nested) in zip(plain_params, nested_params):
        plain_state = plain_opt.state[plain]
        nested_state = nested_opt.state[nested]
        assert torch.equal(
            nested_state["m_codebook"], plain_state["m_codebook"]
        )
        assert torch.equal(
            nested_state["m_magnitude"], plain_state["m_magnitude"]
        )
        effect = max(effect, (nested - plain).abs().max().item())
    assert effect > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_helper_peak_stays_under_conservative_workspace_model():
    inputs = torch.randn(
        8, 128, 256, device="cuda", dtype=torch.bfloat16
    )
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = gm._zeropower_via_newtonschulz_batched(
        inputs.clone(), gm.NS_SCHEDULE_3STEP, 3, 1e-7
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    budget = gm._batched_ns_workspace_bytes(8, 128, 256)
    assert peak <= budget
    assert torch.isfinite(result).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_capturable_opt_in_never_enters_batched_helper(monkeypatch):
    calls = []
    original = gm._zeropower_via_newtonschulz_batched

    def wrapped(inputs, *args, **kwargs):
        calls.append(inputs.shape[0])
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(gm, "_zeropower_via_newtonschulz_batched", wrapped)
    params = [
        (
            f"p{i}",
            torch.nn.Parameter(
                torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
            ),
        )
        for i in range(8)
    ]
    static_grads = [torch.randn_like(p) * 1e-3 for _, p in params]
    opt = GefenMuon(
        params, lr=1e-3, fused=True, capturable=True,
        ns_schedule="tuned3", batched_ns=True,
    )
    assert opt.param_groups[0]["nesterov"] is True
    for (_, p), grad in zip(params, static_grads):
        p.grad = grad
    for _ in range(3):
        opt.step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()
    graph.replay()
    torch.cuda.synchronize()
    assert calls == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(64, 128), (128, 64)])
def test_cautious_masking_composes_with_batched_ns(monkeypatch, shape):
    calls = []
    original = gm._zeropower_via_newtonschulz_batched

    def wrapped(inputs, *args, **kwargs):
        calls.append(inputs.shape[0])
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(gm, "_zeropower_via_newtonschulz_batched", wrapped)
    torch.manual_seed(9)
    initial = [
        (torch.randn(*shape, device="cuda") * 0.02).bfloat16()
        for _ in range(8)
    ]
    grads = [
        (torch.randn(*shape, device="cuda") * 1e-3).bfloat16()
        for _ in range(8)
    ]

    def run(batched):
        params = [
            (f"p{i}", torch.nn.Parameter(value.clone()))
            for i, value in enumerate(initial)
        ]
        opt = GefenMuon(
            params, lr=0.02, fused=True, ns_schedule="tuned3",
            cautious=True, batched_ns=batched,
        )
        for (_, p), grad in zip(params, grads):
            p.grad = grad.clone()
        opt.step()
        torch.cuda.synchronize()
        return params, opt

    serial_params, serial_opt = run(False)
    assert calls == []
    batch_params, batch_opt = run(True)
    assert calls == [8]
    for (_, serial), (_, batched) in zip(serial_params, batch_params):
        serial_state = serial_opt.state[serial]
        batch_state = batch_opt.state[batched]
        assert torch.equal(
            serial_state["m_codebook"], batch_state["m_codebook"]
        )
        assert torch.equal(
            serial_state["m_magnitude"], batch_state["m_magnitude"]
        )
        assert torch.isfinite(batched).all()
        delta = batched.float() - serial.float()
        assert delta.abs().max().item() <= 1e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_step_batches_per_shape_and_keeps_serial_params_bit_equal(
    monkeypatch,
):
    batch_calls = []
    original = gm._zeropower_via_newtonschulz_batched

    def wrapped(inputs, *args, **kwargs):
        batch_calls.append(tuple(inputs.shape))
        return original(inputs, *args, **kwargs)

    monkeypatch.setattr(gm, "_zeropower_via_newtonschulz_batched", wrapped)
    torch.manual_seed(27)
    specs = (
        [(f"wide{i}.weight", (64, 128), torch.bfloat16) for i in range(8)]
        + [(f"tall{i}.weight", (128, 64), torch.bfloat16) for i in range(9)]
        + [
            ("mindim.weight", (513, 2044), torch.bfloat16),  # min dim over gate
            ("numel.weight", (512, 2049), torch.bfloat16),  # numel over gate
            ("fp32.weight", (64, 128), torch.float32),  # dtype-ineligible
        ]
    )
    initial = [
        (name, (torch.randn(*shape, device="cuda") * 0.02).to(dtype))
        for name, shape, dtype in specs
    ]
    grads = [
        (torch.randn(*shape, device="cuda") * 1e-3).to(dtype)
        for _, shape, dtype in specs
    ]

    def run(batched):
        params = [
            (name, torch.nn.Parameter(value.clone()))
            for name, value in initial
        ]
        opt = GefenMuon(
            params, lr=0.02, fused=True, ns_schedule="tuned3",
            batched_ns=batched,
        )
        for (_, p), grad in zip(params, grads):
            p.grad = grad.clone()
        opt.step()
        torch.cuda.synchronize()
        return params, opt

    serial_params, serial_opt = run(False)
    assert batch_calls == []
    batch_params, batch_opt = run(True)
    # One homogeneous bucket per shape; the ineligible stragglers never batch.
    # The tall (128, 64) bucket is oriented wide before stacking, so both
    # helper calls see [batch, 64, 128].
    assert batch_calls == [(8, 64, 128), (9, 64, 128)]
    for (name, serial), (_, batched) in zip(serial_params, batch_params):
        serial_state = serial_opt.state[serial]
        batch_state = batch_opt.state[batched]
        assert torch.equal(
            serial_state["m_codebook"], batch_state["m_codebook"]
        )
        assert torch.equal(
            serial_state["m_magnitude"], batch_state["m_magnitude"]
        )
        if name.split(".")[0].rstrip("0123456789") in ("mindim", "numel", "fp32"):
            # Serially-routed items must stay on the bit-identical path even
            # when batching engages for their step-mates.
            assert torch.equal(batched, serial)
        else:
            delta = batched.float() - serial.float()
            assert delta.abs().max().item() <= 1e-2
