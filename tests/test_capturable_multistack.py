"""Regression coverage for Gefen's heterogeneous capturable scalar stacks."""

import math

import pytest
import torch

from gefen.gefen import Gefen


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="capturable stacks require CUDA"
)


def _params(seed, shapes, dtype=torch.float32):
    torch.manual_seed(seed)
    return [
        torch.nn.Parameter(
            (torch.randn(*shape, device="cuda", dtype=torch.float32) * 0.02).to(dtype)
        )
        for shape in shapes
    ]


def _set_grads(params, seed):
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p) * 0.01


def _stacks(opt):
    assert opt._capt_stacks is not None
    assert len(opt._capt_stacks) == 1
    return next(iter(opt._capt_stacks.values()))


def _warm_stacks(opt, params):
    # Step one initializes per-param state; step two creates and uses stacks.
    _set_grads(params, 1)
    opt.step()
    _set_grads(params, 2)
    opt.step()


def _scalar_expected(lr, wd, beta1, beta2, step, sqrt_bc2):
    bc2 = 1.0 - beta2**step
    if sqrt_bc2:
        bc2 = math.sqrt(bc2)
    return [
        torch.tensor(lr, dtype=torch.float32).item(),
        torch.tensor(1.0 / bc2, dtype=torch.float32).item(),
        torch.tensor(1.0 / (1.0 - beta1**step), dtype=torch.float32).item(),
        torch.tensor(1.0 - lr * wd, dtype=torch.float32).item(),
    ]


def test_weight_decay_groups_build_deterministic_stacks():
    params = _params(0, [(32, 32)] * 4)
    lr = torch.tensor(1e-3, device="cuda")
    groups = [
        {"params": params[:2], "lr": lr, "weight_decay": 0.1},
        {"params": params[2:], "lr": lr, "weight_decay": 0.0},
    ]
    opt = Gefen(groups, capturable=True, fused=False)
    _warm_stacks(opt, params)

    stacks = _stacks(opt)
    assert len(stacks) == 2
    assert [s["weight_decay0"] for s in stacks] == [0.1, 0.0]
    assert [s["rows"] for s in stacks] == [params[:2], params[2:]]
    assert all(s["lr0"] is lr and s["tensor_lr"] for s in stacks)
    for stack_index, stack in enumerate(stacks):
        assert stack["steps2d"].shape == (2, 2)
        assert stack["scal2d"].shape == (2, 4)
        for row, p in enumerate(stack["rows"]):
            state = opt.state[p]
            assert state["_capt_stack"] == stack_index
            assert state["_capt_row"] == row
            assert state["step"] is stack["step_views"][row]
            assert state["factored_step"] is stack["bc2_views"][row]
            assert state["_capt_scalars"] is stack["scalar_views"][row]
    registry = opt._capt_stacks
    _set_grads(params, 3)
    opt.step()
    assert opt._capt_stacks is registry  # no heterogeneous rebuild-thrash


def test_distinct_tensor_lrs_build_stacks_and_mutate_independently():
    params = _params(0, [(32, 32)] * 4)
    lr_a = torch.tensor(1e-3, device="cuda")
    lr_b = torch.tensor(2e-3, device="cuda")
    groups = [
        {"params": params[:2], "lr": lr_a, "weight_decay": 0.1},
        {"params": params[2:], "lr": lr_b, "weight_decay": 0.1},
    ]
    opt = Gefen(groups, capturable=True, fused=False)
    _warm_stacks(opt, params)

    stacks = _stacks(opt)
    assert len(stacks) == 2
    assert stacks[0]["lr0"] is lr_a
    assert stacks[1]["lr0"] is lr_b

    lr_a.fill_(3e-3)
    lr_b.fill_(4e-3)
    _set_grads(params, 3)
    opt.step()
    for stack, expected_lr in zip(stacks, (lr_a, lr_b)):
        want_lr = expected_lr.float().item()
        want_wd = (1.0 - expected_lr.double() * 0.1).float().item()
        assert torch.all(stack["scal2d"][:, 0] == want_lr)
        assert torch.all(stack["scal2d"][:, 3] == want_wd)


def test_rowwise_betas_and_routing_match_eager_reference():
    shapes = [(32, 32), (32,), (16, 32), (16,)]
    initial = _params(0, shapes)
    grads = []
    torch.manual_seed(1)
    for _ in range(5):
        grads.append([torch.randn_like(p) * 0.01 for p in initial])

    def run(capturable):
        params = [torch.nn.Parameter(p.detach().clone()) for p in initial]
        groups = [
            {
                "params": [params[0]],
                "lr": 1e-3,
                "betas": (0.8, 0.98),
                "weight_decay": 0.1,
            },
            {
                "params": [params[1]],
                "lr": 1e-3,
                "betas": (0.9, 0.999),
                "weight_decay": 0.1,
            },
            {
                "params": [params[2]],
                "lr": 5e-4,
                "betas": (0.85, 0.95),
                "weight_decay": 0.0,
            },
            {
                "params": [params[3]],
                "lr": 5e-4,
                "betas": (0.7, 0.9),
                "weight_decay": 0.0,
            },
        ]
        opt = Gefen(groups, capturable=capturable, fused=False)
        for step_grads in grads:
            for p, grad in zip(params, step_grads):
                p.grad = grad.clone()
            opt.step()
        return params, opt

    ref, _ = run(False)
    got, opt = run(True)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-7)

    stacks = _stacks(opt)
    assert len(stacks) == 2
    assert all(stack["sqrt_mode"] == "mixed" for stack in stacks)
    assert stacks[0]["betas"] == [(0.8, 0.98), (0.9, 0.999)]
    assert stacks[1]["betas"] == [(0.85, 0.95), (0.7, 0.9)]
    for stack in stacks:
        for row, p in enumerate(stack["rows"]):
            group = stack["row_groups"][row]
            step = int(opt.state[p]["step"].item())
            want = _scalar_expected(
                float(group["lr"]),
                group["weight_decay"],
                group["beta1"],
                group["beta2"],
                step,
                p.ndim != 2,
            )
            got_scalars = opt.state[p]["_capt_scalars"].cpu().tolist()
            assert got_scalars == want


def test_float_lr_cohort_moves_together_then_rebuilds_on_divergence():
    params = _params(0, [(32, 32)] * 4)
    opt = Gefen(
        [
            {"params": params[:2], "lr": 1e-3, "weight_decay": 0.1},
            {"params": params[2:], "lr": 1e-3, "weight_decay": 0.1},
        ],
        capturable=True,
        fused=False,
    )
    _warm_stacks(opt, params)
    assert len(_stacks(opt)) == 1
    assert all("_capt_stack" not in opt.state[p] for p in params)

    builds = 0
    original_build = opt._capt_build_stacks

    def counted_build():
        nonlocal builds
        builds += 1
        return original_build()

    opt._capt_build_stacks = counted_build
    for group in opt.param_groups:
        group["lr"] = 2e-3
    _set_grads(params, 3)
    opt.step()
    assert builds == 0
    assert len(_stacks(opt)) == 1

    opt.param_groups[1]["lr"] = 3e-3
    _set_grads(params, 4)
    opt.step()
    assert builds == 1
    assert len(_stacks(opt)) == 2
    assert [opt.state[p]["_capt_stack"] for p in params] == [0, 0, 1, 1]
    _set_grads(params, 5)
    opt.step()
    assert builds == 1  # the heterogeneous registry is retained, not retried


def test_grad_set_changes_rebuild_without_double_increment():
    params = _params(0, [(32, 32)] * 4)
    lr = torch.tensor(1e-3, device="cuda")
    opt = Gefen(params, lr=lr, capturable=True, fused=False)
    _warm_stacks(opt, params)
    assert len(_stacks(opt)[0]["rows"]) == 4

    _set_grads(params, 3)
    params[-1].grad = None
    opt.step()
    assert len(_stacks(opt)[0]["rows"]) == 3
    assert "_capt_row" not in opt.state[params[-1]]
    assert opt.state[params[-1]]["step"].item() == 2

    _set_grads(params, 4)
    opt.step()
    assert len(_stacks(opt)[0]["rows"]) == 4
    assert [opt.state[p]["step"].item() for p in params] == [4, 4, 4, 3]


def test_beta_and_scalar_state_surgery_force_safe_rebuilds():
    params = _params(0, [(32, 32)] * 2)
    opt = Gefen(params, lr=1e-3, capturable=True, fused=False)
    _warm_stacks(opt, params)
    first = _stacks(opt)[0]

    opt.param_groups[0]["beta1"] = 0.7
    _set_grads(params, 3)
    opt.step()
    second = _stacks(opt)[0]
    assert second is not first
    assert second["betas"] == [(0.7, 0.999), (0.7, 0.999)]
    assert torch.all(second["consts2d"][1] == 0.7)

    opt.state[params[0]]["_capt_scalars"] = (
        opt.state[params[0]]["_capt_scalars"].clone()
    )
    _set_grads(params, 4)
    opt.step()
    third = _stacks(opt)[0]
    assert third is not second
    assert opt.state[params[0]]["_capt_scalars"] is third["scalar_views"][0]


def test_permanently_ineligible_set_keeps_empty_sentinel_without_thrash():
    # CPU params under capturable=True are permanently ineligible for the
    # batched stacks (_capt_row_info requires CUDA). The first prologue then
    # builds an EMPTY registry, which _capt_build_stacks keeps as a stable
    # "currently no eligible rows" sentinel: stepping continues on the
    # per-param path and the builder is NOT re-entered every step.
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(32, 32) * 0.02) for _ in range(3)
    ]
    opt = Gefen(params, lr=1e-3, capturable=True, fused=False)

    builds = 0
    original_build = opt._capt_build_stacks

    def counted_build():
        nonlocal builds
        builds += 1
        return original_build()

    opt._capt_build_stacks = counted_build

    before = [p.detach().clone() for p in params]
    for step in range(1, 6):
        _set_grads(params, step)
        opt.step()  # (a) no crash
        # empty-dict sentinel, distinct from the None "never built" state
        assert opt._capt_stacks == {} and opt._capt_stacks is not None
    # (c) built once (first prologue), then trusted -- no rebuild storm
    assert builds == 1
    sentinel = opt._capt_stacks
    # (b) stepping still works on the per-param path
    assert all(
        not torch.equal(p.detach(), b) for p, b in zip(params, before)
    )
    assert all(int(opt.state[p]["step"].item()) == 5 for p in params)
    assert all("_capt_row" not in opt.state[p] for p in params)

    # The sentinel must not be sticky: once a row becomes eligible (a CUDA
    # param joins the stepping set), validation sees the missing device stack
    # and rebuilds exactly once (lazy state init on its first step, rebuild
    # and registration on its second), then settles again.
    pc = torch.nn.Parameter(torch.randn(16, 16, device="cuda") * 0.02)
    opt.add_param_group({"params": [pc]})
    for step in range(6, 9):
        _set_grads(params, step)
        pc.grad = torch.randn_like(pc) * 0.01
        opt.step()
    assert builds == 2
    assert opt._capt_stacks is not sentinel
    assert "_capt_row" in opt.state[pc]


def test_multistack_state_dict_is_compact_and_rebuilds_after_load():
    params = _params(0, [(32, 32)] * 4)
    lr = torch.tensor(1e-3, device="cuda")
    groups = [
        {"params": params[:2], "lr": lr, "weight_decay": 0.1},
        {"params": params[2:], "lr": lr, "weight_decay": 0.0},
    ]
    opt = Gefen(groups, capturable=True, fused=False)
    _warm_stacks(opt, params)
    state_dict = opt.state_dict()

    for state in state_dict["state"].values():
        assert not any(key.startswith("_capt") for key in state)
        for key in ("step", "factored_step"):
            counter = state[key]
            assert counter.storage_offset() == 0
            assert counter.untyped_storage().nbytes() == counter.element_size()

    restored_params = _params(5, [(32, 32)] * 4)
    restored_lr = torch.tensor(1e-3, device="cuda")
    restored = Gefen(
        [
            {
                "params": restored_params[:2],
                "lr": restored_lr,
                "weight_decay": 0.1,
            },
            {
                "params": restored_params[2:],
                "lr": restored_lr,
                "weight_decay": 0.0,
            },
        ],
        capturable=True,
        fused=False,
    )
    restored.load_state_dict(state_dict)
    assert restored._capt_stacks is None
    assert all(
        "_capt_stack" not in state and "_capt_row" not in state
        for state in restored.state.values()
    )
    _set_grads(restored_params, 6)
    restored.step()
    assert len(_stacks(restored)) == 2
    assert all(state["step"].item() == 3 for state in restored.state.values())


def _graph_multistack_run(captured):
    shapes = [(32, 32)] * 4
    params = _params(0, shapes)
    lr_a = torch.tensor(1e-3, device="cuda")
    lr_b = torch.tensor(2e-3, device="cuda")
    opt = Gefen(
        [
            {"params": params[:2], "lr": lr_a, "weight_decay": 0.1},
            {"params": params[2:], "lr": lr_b, "weight_decay": 0.0},
        ],
        capturable=True,
    )
    static_grads = [torch.zeros_like(p) for p in params]
    for p, grad in zip(params, static_grads):
        p.grad = grad
    torch.manual_seed(10)
    grads = [
        [torch.randn_like(p) * 0.01 for p in params]
        for _ in range(6)
    ]
    schedules = [(1e-3 * 0.9**i, 2e-3 * 0.8**i) for i in range(6)]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(2):
            for static, grad in zip(static_grads, grads[i]):
                static.copy_(grad)
            lr_a.fill_(schedules[i][0])
            lr_b.fill_(schedules[i][1])
            opt.step()
    torch.cuda.current_stream().wait_stream(side)
    assert len(_stacks(opt)) == 2

    if captured:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            opt.step()
        for i in range(2, 6):
            for static, grad in zip(static_grads, grads[i]):
                static.copy_(grad)
            lr_a.fill_(schedules[i][0])
            lr_b.fill_(schedules[i][1])
            graph.replay()
    else:
        for i in range(2, 6):
            for static, grad in zip(static_grads, grads[i]):
                static.copy_(grad)
            lr_a.fill_(schedules[i][0])
            lr_b.fill_(schedules[i][1])
            opt.step()
    torch.cuda.synchronize()
    return params, opt, schedules[-1]


def test_multistack_cuda_graph_replay_tracks_independent_tensor_lrs():
    ref, _, _ = _graph_multistack_run(False)
    got, opt, final_lrs = _graph_multistack_run(True)
    for a, b in zip(got, ref):
        assert torch.allclose(a, b, rtol=2e-2, atol=2e-3)
    assert all(state["step"].item() == 6 for state in opt.state.values())

    for stack, final_lr in zip(_stacks(opt), final_lrs):
        want_lr = torch.tensor(final_lr, dtype=torch.float32).item()
        want_wd = torch.tensor(
            1.0 - final_lr * stack["weight_decay0"], dtype=torch.float32
        ).item()
        assert torch.all(stack["scal2d"][:, 0] == want_lr)
        assert torch.all(stack["scal2d"][:, 3] == want_wd)


def test_multistack_compile_reduce_overhead():
    shapes = [(48, 64), (64,), (32, 48), (48,)]
    steps, warmup = 6, 2
    initial = _params(0, shapes)
    torch.manual_seed(1)
    grads = [
        [torch.randn_like(p) * 0.01 for p in initial]
        for _ in range(steps)
    ]

    def run(compiled):
        params = [torch.nn.Parameter(p.detach().clone()) for p in initial]
        lr = torch.tensor(1e-3, device="cuda")
        opt = Gefen(
            [
                {"params": params[:2], "lr": lr, "weight_decay": 0.1},
                {"params": params[2:], "lr": lr, "weight_decay": 0.0},
            ],
            capturable=True,
        )
        static = [torch.zeros_like(p) for p in params]
        for p, grad in zip(params, static):
            p.grad = grad
        compiled_step = (
            torch.compile(opt.step, mode="reduce-overhead", dynamic=False)
            if compiled
            else None
        )
        for i in range(steps):
            for static_grad, grad in zip(static, grads[i]):
                static_grad.copy_(grad)
            lr.fill_(1e-3 * 0.9**i)
            if compiled and i >= warmup:
                torch.compiler.cudagraph_mark_step_begin()
                compiled_step()
            else:
                opt.step()
        torch.cuda.synchronize()
        assert len(_stacks(opt)) == 2
        return params

    try:
        ref = run(False)
        got = run(True)
        for a, b in zip(got, ref):
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-6)
    finally:
        torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Invalidate-under-capture fallback and live-beta consistency
#
# When the per-step validation fails INSIDE a torch.cuda.graph capture (a
# hyperparameter or grad-set change between warmup and capture), the step must
# tear the stacks down and fall back to the per-param path capture-safely --
# rebuilding the orphaned scalar/consts buffers instead of raising
# KeyError("_capt_consts") -- and the per-param path must honor live group
# betas exactly like the batched path does.
# ---------------------------------------------------------------------------


def _warm_fused_for_capture(grads, n_params=2, warmup=3):
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(
            (torch.randn(32, 32, device="cuda") * 0.02).bfloat16()
        )
        for _ in range(n_params)
    ]
    opt = Gefen(params, lr=1e-3, capturable=True, fused=True)
    static_grads = [torch.zeros_like(p) for p in params]
    for p, grad in zip(params, static_grads):
        p.grad = grad
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            for static, grad in zip(static_grads, grads[i]):
                static.copy_(grad)
            opt.step()
    torch.cuda.current_stream().wait_stream(side)
    assert opt._capt_stacks
    return params, opt, static_grads


def _capture_grads(steps, n_params=2, seed=11):
    torch.manual_seed(seed)
    return [
        [
            torch.randn(32, 32, device="cuda", dtype=torch.bfloat16) * 0.01
            for _ in range(n_params)
        ]
        for _ in range(steps)
    ]


def test_beta_change_under_capture_falls_back_and_uses_live_beta():
    grads = _capture_grads(5)

    # Eager live-beta reference: same grads, beta1 changed before step 4.
    torch.manual_seed(0)
    ref_params = [
        torch.nn.Parameter(
            (torch.randn(32, 32, device="cuda") * 0.02).bfloat16()
        )
        for _ in range(2)
    ]
    ref_opt = Gefen(ref_params, lr=1e-3, capturable=True, fused=True)
    for i in range(5):
        if i == 3:
            ref_opt.param_groups[0]["beta1"] = 0.7
        for p, grad in zip(ref_params, grads[i]):
            p.grad = grad.clone()
        ref_opt.step()
    torch.cuda.synchronize()

    params, opt, static_grads = _warm_fused_for_capture(grads)
    opt.param_groups[0]["beta1"] = 0.7

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()  # validation fails mid-capture -> per-param fallback
    assert opt._capt_stacks is None  # torn down during the capture

    for i in range(3, 5):  # replays run steps 4 and 5
        for static, grad in zip(static_grads, grads[i]):
            static.copy_(grad)
        graph.replay()
    torch.cuda.synchronize()

    for p, ref in zip(params, ref_params):
        state = opt.state[p]
        assert int(state["step"].item()) == 5
        want = _scalar_expected(1e-3, 0.0, 0.7, 0.999, 5, p.ndim != 2)
        assert state["_capt_scalars"].cpu().tolist() == want
        assert torch.equal(p.detach(), ref.detach())


def test_grad_drop_under_capture_falls_back_and_steps_remaining_rows():
    grads = _capture_grads(5)
    params, opt, static_grads = _warm_fused_for_capture(grads)
    params[1].grad = None
    frozen = params[1].detach().clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()  # validation fails mid-capture -> per-param fallback
    assert opt._capt_stacks is None

    for i in range(3, 5):  # replays run steps 4 and 5 for params[0]
        static_grads[0].copy_(grads[i][0])
        graph.replay()
    torch.cuda.synchronize()

    assert int(opt.state[params[0]]["step"].item()) == 5
    assert int(opt.state[params[1]]["step"].item()) == 3
    assert torch.equal(params[1].detach(), frozen)
    want = _scalar_expected(1e-3, 0.0, 0.9, 0.999, 5, params[0].ndim != 2)
    assert opt.state[params[0]]["_capt_scalars"].cpu().tolist() == want


def test_per_param_path_honors_live_betas_after_midrun_change():
    def run(force_per_param):
        torch.manual_seed(0)
        params = [
            torch.nn.Parameter(
                (torch.randn(32, 32, device="cuda") * 0.02).bfloat16()
            )
            for _ in range(3)
        ]
        opt = Gefen(
            params, lr=1e-3, betas=(0.9, 0.999), capturable=True, fused=True
        )
        if force_per_param:
            # Disabling the batched prologue keeps every step on the
            # per-param refresh path, mirroring the capture-time fallback
            # without needing a CUDA graph.
            opt._capt_batched_prologue = lambda: None
        torch.manual_seed(1)
        grads = [
            [torch.randn_like(p) * 0.01 for p in params] for _ in range(6)
        ]
        for i in range(6):
            if i + 1 == 3:
                opt.param_groups[0]["beta1"] = 0.5
            for p, grad in zip(params, grads[i]):
                p.grad = grad.clone()
            opt.step()
        torch.cuda.synchronize()
        inv_bc1 = opt.state[params[0]]["_capt_scalars"][2].item()
        return [p.detach().clone() for p in params], inv_bc1

    batched_finals, batched_inv = run(False)
    per_param_finals, per_param_inv = run(True)
    live = _scalar_expected(1e-3, 0.0, 0.5, 0.999, 6, False)[2]
    stale = _scalar_expected(1e-3, 0.0, 0.9, 0.999, 6, False)[2]
    assert batched_inv == live
    assert per_param_inv == live
    assert per_param_inv != stale
    for a, b in zip(batched_finals, per_param_finals):
        assert torch.equal(a, b)
