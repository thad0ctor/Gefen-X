"""
Conformance tests for ``capturable`` (CUDA-graph capturability).

Contract (mirrors torch.optim's ``capturable`` argument, and the requirement in
NVIDIA-NeMo/Emerging-Optimizers#109): with ``capturable=True`` everything that
varies across steps -- the per-param step counters and the bias corrections
derived from them -- lives in device tensors, a tensor ``lr`` is consumed on
device without ``.item()`` syncs, and ``optimizer.step()`` can be captured into
a ``torch.cuda.CUDAGraph`` after the standard warmup steps and replayed with
correct step-dependent math. With ``capturable=False`` (the default), behavior
is bit-identical to the historical host-scalar path, and attempting to capture
``step()`` raises instead of silently freezing the step counters.

Layers covered:
  1. eager parity: capturable=True matches capturable=False stepping (Gefen in
     both factored-v and legacy-vmean modes, GefenMuon, GefenMuonHybrid);
  2. real capture/replay: step() captured in a CUDAGraph, replayed with fresh
     gradients AND an in-place-updated tensor lr each replay, must match the
     same optimizer stepping eagerly (Gefen, GefenMuon, GefenMuonHybrid);
  3. guards: capturing with capturable=False raises; capturable=True rejects
     the host-driven option codebook_refresh_every (stochastic_round is now
     SUPPORTED under capturable: the rounding seed lives in a per-device
     int64 tensor advanced on device once per step);
  4. checkpoint portability: step counters convert across the capturable
     toggle in both directions on load_state_dict;
  5. stochastic rounding under capturable: eager determinism, seed-semantics
     parity across the capturable toggle and across graph replay vs eager,
     and anti-freeze (the seed advances once per replay and the dither
     varies between replays).

The suite needs a CUDA device and is skipped cleanly without one.

Run on the current GPU:  python tests/test_capturable.py
"""
import torch

try:
    import pytest
except ImportError:  # pragma: no cover - allows the script path without pytest
    pytest = None

from gefen.gefen import Gefen
from gefen.gefen_muon import GefenMuon
from gefen.hybrid import GefenMuonHybrid

CUDA_OK = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_OK else "cpu"

if pytest is not None:
    pytestmark = pytest.mark.skipif(not CUDA_OK, reason="capturable needs CUDA")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _named_params(seed, specs):
    torch.manual_seed(seed)
    return [
        (name, torch.nn.Parameter(torch.randn(*shape, device=DEVICE) * 0.02))
        for name, shape in specs
    ]


def _grad_sequence(seed, specs, steps):
    torch.manual_seed(seed)
    return [
        [torch.randn(*shape, device=DEVICE) * 0.1 for _, shape in specs]
        for _ in range(steps)
    ]


# Every step-like counter the capturable contract covers (matches the
# counter_keys normalization in Gefen.load_state_dict).
COUNTER_KEYS = ("step", "vmean_step", "factored_step", "normuon_step")

GEFEN_SPECS = [("w2d", (48, 64)), ("norm", (64,)), ("bias", (48,))]
MUON_SPECS = [("h1", (64, 48)), ("h2", (32, 64))]
HYBRID_MUON_SPECS = [("h1.weight", (64, 48)), ("h2.weight", (32, 64))]
HYBRID_BACKUP_SPECS = [("embed.weight", (96, 32)), ("norm.weight", (64,))]


def _run_eager(make_opt, specs, steps, seed=0, lr_values=None):
    params = _named_params(seed, specs)
    opt = make_opt(params)
    grads = _grad_sequence(seed + 1, specs, steps)
    for i in range(steps):
        for (_name, p), g in zip(params, grads[i]):
            p.grad = g.clone()
        if lr_values is not None:
            for group in opt.param_groups:
                lr = group["lr"]
                if torch.is_tensor(lr):
                    lr.fill_(lr_values[i])
                else:
                    group["lr"] = lr_values[i]
        opt.step()
    return [p.detach().clone() for _, p in params], opt


def _assert_params_close(got, want, rtol=1e-5, atol=1e-7, what=""):
    for i, (a, b) in enumerate(zip(got, want)):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), (
            "{} param {} diverged: max abs diff {:.3e}".format(
                what, i, (a - b).abs().max().item()
            )
        )


# ---------------------------------------------------------------------------
# 1. eager parity: capturable=True vs capturable=False
# ---------------------------------------------------------------------------

def _parity_case(make_base, specs, steps=6, rtol=1e-5, atol=1e-7, what=""):
    ref, _ = _run_eager(lambda ps: make_base(ps, capturable=False), specs, steps)
    got, opt = _run_eager(lambda ps: make_base(ps, capturable=True), specs, steps)
    _assert_params_close(got, ref, rtol=rtol, atol=atol, what=what)
    # EVERY capturable step-like counter lives on the device as a 0-dim fp32
    # tensor (not just "step": the vmean/factored/normuon counters drive their
    # own bias corrections and must obey the same contract).
    for pstate in opt.state.values():
        for key in COUNTER_KEYS:
            step = pstate.get(key)
            if step is None:
                continue
            assert torch.is_tensor(step) and step.is_cuda and step.dim() == 0, (
                key, step,
            )
            assert step.item() == steps, (key, step.item(), steps)


def test_parity_gefen_factored():
    # Both sides on the decomposed (fused=False) arithmetic: the only delta is
    # capturable's device-tensor step/bias-correction handling, so the match
    # is tight.
    _parity_case(
        lambda ps, capturable: Gefen(ps, lr=1e-3, weight_decay=0.01,
                                     fused=False, capturable=capturable),
        GEFEN_SPECS, what="gefen-factored",
    )


def test_parity_gefen_factored_fused():
    # Both sides on the FUSED factored kernel (capturable now stays fused; its
    # per-step scalars flow through the device buffer). The two sides run the
    # identical kernel arithmetic: the fp32 scalars the kernel consumes are
    # asserted BIT-IDENTICAL (device float64 tensor ops vs host python doubles
    # cast to the same fp32 values), so the only residual divergence is the
    # factored kernel's own documented run-to-run atomicAdd nondeterminism in
    # the row/col grad^2 sums (a re-run of the SAME config differs by a couple
    # of 1-ulp bf16 flips) -- hold the params to that envelope.
    lr, wd, beta1, beta2 = 1e-3, 0.1, 0.9, 0.999

    def run(capturable, steps=6, shape=(768, 512), seed=3):
        torch.manual_seed(seed)
        p = torch.nn.Parameter(
            (torch.randn(*shape, device=DEVICE) * 0.02).bfloat16()
        )
        opt = Gefen([("w", p)], lr=lr, weight_decay=wd, capturable=capturable)
        torch.manual_seed(seed + 1)
        for _ in range(steps):
            p.grad = torch.randn(*shape, device=DEVICE).bfloat16() * 1e-3
            opt.step()
        state = opt.state[p]
        return p.detach().float(), state

    ref, _ = run(capturable=False)
    got, state = run(capturable=True)

    # The device-computed kernel scalars match the host-scalar path's fp32
    # casts exactly ([lr, 1/bc2, 1/bc1, 1 - lr*wd] at the final step count).
    def f32(x):
        return torch.tensor(x, dtype=torch.float32).item()

    k = int(state["step"].item())
    expected = [
        f32(lr),
        f32(1.0 / (1.0 - beta2 ** int(state["factored_step"].item()))),
        f32(1.0 / (1.0 - beta1 ** k)),
        f32(1.0 - lr * wd),
    ]
    buf = state["_capt_scalars"].cpu()
    assert [buf[i].item() for i in range(4)] == expected, (buf, expected)

    # Params: within the fused factored path's own run-to-run noise. The abs
    # gate is calibrated to the documented atomicAdd nondeterminism above: the
    # params sit in the top binade [0.0625, 0.125) where 1 bf16 ULP = 4.88e-4,
    # and a 2-ULP event (9.77e-4) was observed on a 188-SM Blackwell -- so gate
    # at 1.5e-3 (>2 ULP with headroom, still far below real numeric drift).
    diff = (got - ref).abs()
    frac_diff = (diff > 0).float().mean().item()
    assert diff.max().item() <= 1.5e-3, diff.max().item()
    assert frac_diff <= 1e-3, frac_diff


def test_parity_gefen_legacy_vmean():
    _parity_case(
        lambda ps, capturable: Gefen(ps, lr=1e-3, weight_decay=0.01,
                                     factored_v_2d=False, capturable=capturable),
        GEFEN_SPECS, what="gefen-legacy",
    )


def test_parity_gefen_nonfused():
    _parity_case(
        lambda ps, capturable: Gefen(ps, lr=1e-3, weight_decay=0.01, fused=False,
                                     factored_v_2d=False, capturable=capturable),
        GEFEN_SPECS, what="gefen-nonfused",
    )


def test_parity_muon():
    _parity_case(
        lambda ps, capturable: GefenMuon(
            ps, lr=1e-3, weight_decay=0.01, adjust_lr_fn="match_rms_adamw",
            ns_schedule="tuned3", normuon=True, capturable=capturable,
        ),
        MUON_SPECS, what="muon",
    )


def test_parity_hybrid():
    def build(capturable):
        muon_named = _named_params(0, HYBRID_MUON_SPECS)
        backup_named = _named_params(1, HYBRID_BACKUP_SPECS)
        opt = GefenMuonHybrid(
            muon_named, backup_named, lr=1e-3, backup_lr=5e-4,
            weight_decay=0.01, capturable=capturable,
        )
        return muon_named + backup_named, opt

    specs = HYBRID_MUON_SPECS + HYBRID_BACKUP_SPECS
    results = []
    for capturable in (False, True):
        params, opt = build(capturable)
        grads = _grad_sequence(2, specs, 5)
        for i in range(5):
            for (_name, p), g in zip(params, grads[i]):
                p.grad = g.clone()
            opt.step()
        results.append([p.detach().clone() for _, p in params])
    _assert_params_close(results[1], results[0], what="hybrid")


# ---------------------------------------------------------------------------
# 2. real CUDA-graph capture / replay
# ---------------------------------------------------------------------------

def _run_captured(make_opt, specs, grads, lr_values, warmup, seed):
    """Warmup eagerly, capture one step() into a CUDAGraph, then drive every
    post-warmup step by replay() with fresh grads + an in-place tensor lr."""
    steps = len(lr_values)
    params = _named_params(seed, specs)
    lr = torch.tensor(lr_values[0], device=DEVICE)
    opt = make_opt(params, lr)
    # Static grad buffers: assigned once, refilled in place before every step.
    static_grads = [torch.zeros_like(p) for _, p in params]
    for (_, p), sg in zip(params, static_grads):
        p.grad = sg

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for i in range(warmup):
            for sg, g in zip(static_grads, grads[i]):
                sg.copy_(g)
            lr.fill_(lr_values[i])
            opt.step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    # Capture RECORDS the kernels without executing them (that is why warmup
    # ran real steps first), so every post-warmup step -- including the first
    # -- is performed by replay().
    with torch.cuda.graph(graph):
        opt.step()
    for i in range(warmup, steps):
        for sg, g in zip(static_grads, grads[i]):
            sg.copy_(g)
        lr.fill_(lr_values[i])
        graph.replay()
    torch.cuda.synchronize()
    return params, opt


def _capture_replay_case(make_opt, specs, warmup=3, replays=4, seed=0,
                         lr_start=1e-3, lr_decay=0.9, rtol=2e-2, atol=2e-3,
                         what=""):
    """Three-way conformance check for a captured step():

    1. the captured trajectory matches the same optimizer run eagerly with the
       same grad + lr schedule (moderate tolerance: graph replay can take
       address/algorithm paths whose reduction order differs by ulps, which
       the quantized/NS pipelines amplify -- see the anti-freeze checks below
       for the exact assertions);
    2. the in-place lr schedule BITES across replays: a second captured run
       with a constant lr must land visibly elsewhere (this is the check that
       would fail if the lr were baked into the graph);
    3. every step counter advanced once per replay.
    """
    steps = warmup + replays
    lr_sched = [lr_start * (lr_decay ** i) for i in range(steps)]
    grads = _grad_sequence(seed + 1, specs, steps)

    # --- eager reference (capturable=True, no graph) ---
    ref_params = _named_params(seed, specs)
    ref_lr = torch.tensor(lr_start, device=DEVICE)
    ref_opt = make_opt(ref_params, ref_lr)
    for i in range(steps):
        for (_, p), g in zip(ref_params, grads[i]):
            p.grad = g.clone()
        ref_lr.fill_(lr_sched[i])
        ref_opt.step()

    params, opt = _run_captured(make_opt, specs, grads, lr_sched, warmup, seed)
    _assert_params_close(
        [p.detach().clone() for _, p in params],
        [p.detach().clone() for _, p in ref_params],
        rtol=rtol, atol=atol, what=what,
    )

    const_params, _ = _run_captured(
        make_opt, specs, grads, [lr_start] * steps, warmup, seed
    )
    sched_effect = max(
        (a.detach() - b.detach()).abs().max().item()
        for (_, a), (_, b) in zip(params, const_params)
    )
    assert sched_effect > 5e-5, (
        "{}: constant-lr and scheduled-lr captured runs are indistinguishable "
        "({:.3e}) -- the lr looks baked into the graph".format(
            what, sched_effect
        )
    )

    # Every counter kind present must advance once per replay -- a frozen
    # secondary counter (e.g. normuon_step) shifts its bias correction subtly
    # enough to hide inside the param tolerance above, so assert it exactly.
    for pstate in opt.state.values():
        for key in COUNTER_KEYS:
            counter = pstate.get(key)
            if counter is None:
                continue
            assert counter.item() == steps, (
                "{} counter did not advance across replays: {} != {}".format(
                    key, counter.item(), steps
                )
            )
    return params, opt


def test_graph_capture_gefen():
    _capture_replay_case(
        lambda ps, lr: Gefen(ps, lr=lr, weight_decay=0.01, capturable=True),
        GEFEN_SPECS, what="gefen-graph",
    )


def test_graph_capture_gefen_legacy_vmean():
    import math

    warmup, replays = 3, 4
    steps = warmup + replays
    lr_start, lr_decay, wd = 1e-3, 0.9, 0.01
    _, opt = _capture_replay_case(
        lambda ps, lr: Gefen(ps, lr=lr, weight_decay=wd, factored_v_2d=False,
                             capturable=True),
        GEFEN_SPECS, warmup=warmup, replays=replays, lr_start=lr_start,
        lr_decay=lr_decay, what="gefen-legacy-graph",
    )
    # Anti-freeze, exactly. The fused kernels compute the per-block stepsize
    # in-kernel, reading the per-step scalars from the device buffer
    # state["_capt_scalars"] = [lr, 1/sqrt(bc2), 1/bc1, 1 - lr*wd]. After the
    # final replay that buffer must hold EXACTLY (fp32-cast of the double
    # math) the values for the FINAL step count / final scheduled lr -- and
    # must NOT hold the values frozen at their capture-time step. A graph that
    # baked the host-scalar corrections/lr passes the loose param comparison
    # above but fails this.
    beta1, beta2 = 0.9, 0.999

    def f32(x):
        return torch.tensor(x, dtype=torch.float32).item()

    lr_final = lr_start * lr_decay ** (steps - 1)
    checked = 0
    for pstate in opt.state.values():
        if "_capt_scalars" not in pstate:
            continue
        buf = pstate["_capt_scalars"].cpu()

        def expected(k, lr):
            return [
                f32(lr),
                f32(1.0 / math.sqrt(1.0 - beta2 ** k)),
                f32(1.0 / (1.0 - beta1 ** k)),
                f32(1.0 - float(torch.tensor(lr).float().double()) * wd),
            ]

        got = [buf[i].item() for i in range(4)]
        # Exact equality with the final-step values (both sides are the same
        # double op chain cast to fp32)...
        assert got == expected(steps, lr_final), (got, expected(steps, lr_final))
        # ...and exact inequality with every slot's capture-time value: the
        # capture records step warmup+1 with the warmup-final lr.
        frozen = expected(warmup + 1, lr_start * lr_decay ** (warmup - 1))
        for slot in range(4):
            assert got[slot] != frozen[slot], (
                "slot {} of _capt_scalars matches its capture-time value -- "
                "the per-step scalars froze in the graph".format(slot)
            )
        checked += 1
    assert checked == len(GEFEN_SPECS), checked


def test_graph_capture_muon():
    # bf16 Newton-Schulz amplifies the ulp-level eager-vs-replay divergence
    # (cuBLAS algorithm/workspace selection differs inside a capture), so the
    # muon tolerance is looser; the schedule-bites and step-counter checks in
    # _capture_replay_case carry the exactness.
    _capture_replay_case(
        lambda ps, lr: GefenMuon(
            ps, lr=lr, weight_decay=0.01, adjust_lr_fn="match_rms_adamw",
            ns_schedule="tuned3", normuon=True, nesterov=True, capturable=True,
        ),
        MUON_SPECS, rtol=5e-2, atol=5e-3, what="muon-graph",
    )


def test_graph_capture_muon_fused_nesterov_preserves_momentum_state():
    # Integration contract for fused Nesterov + optimizer-level capturability:
    # graph replay must consume the folded momentum output (so the parameter
    # trajectory differs from classical momentum) without folding Nesterov back
    # into the persistent quantized EMA state. The ordinary test above pins the
    # nesterov=True captured trajectory to eager; this cross-run check pins the
    # state/effect split that the fused emitter promises.
    warmup, replays = 3, 4
    steps = warmup + replays
    lr_values = [1e-3 * 0.9**i for i in range(steps)]
    grads = _grad_sequence(91, MUON_SPECS, steps)

    def make(nesterov):
        return lambda ps, lr: GefenMuon(
            ps,
            lr=lr,
            weight_decay=0.01,
            momentum=0.9,
            nesterov=nesterov,
            adjust_lr_fn="match_rms_adamw",
            ns_schedule="tuned3",
            normuon=False,
            capturable=True,
        )

    nested_params, nested_opt = _run_captured(
        make(True), MUON_SPECS, grads, lr_values, warmup, seed=90
    )
    plain_params, plain_opt = _run_captured(
        make(False), MUON_SPECS, grads, lr_values, warmup, seed=90
    )
    assert nested_opt._fused_kernels_available()
    assert plain_opt._fused_kernels_available()

    # Identical externally supplied grads mean the underlying EMA state must be
    # bit-identical even though Nesterov changes the dense tensor sent through
    # Newton-Schulz and therefore the final parameters.
    for (_name_n, p_nested), (_name_p, p_plain) in zip(
        nested_params, plain_params
    ):
        nested_state = nested_opt.state[p_nested]
        plain_state = plain_opt.state[p_plain]
        assert nested_state["step"].item() == plain_state["step"].item() == steps
        assert torch.equal(
            nested_state["m_codebook"], plain_state["m_codebook"]
        )
        assert torch.equal(
            nested_state["m_magnitude"], plain_state["m_magnitude"]
        )

    nesterov_effect = max(
        (p_nested.detach() - p_plain.detach()).abs().max().item()
        for (_name_n, p_nested), (_name_p, p_plain) in zip(
            nested_params, plain_params
        )
    )
    assert nesterov_effect > 1e-5, nesterov_effect


def test_graph_capture_hybrid():
    specs = HYBRID_MUON_SPECS + HYBRID_BACKUP_SPECS

    def make(ps, lr):
        muon_named = ps[: len(HYBRID_MUON_SPECS)]
        backup_named = ps[len(HYBRID_MUON_SPECS):]
        return GefenMuonHybrid(
            muon_named, backup_named, lr=lr, weight_decay=0.01, capturable=True,
        )

    _capture_replay_case(make, specs, rtol=5e-2, atol=5e-3, what="hybrid-graph")


def test_compile_reduce_overhead_gefen():
    # torch.compile(mode="reduce-overhead") wraps step() in CUDA graphs via
    # cudagraph trees. Supported pattern: run the first steps EAGERLY (codebook
    # learning is host-driven, and the eager capturable step marks the params /
    # state tensors as static addresses for cudagraphs), then compile.
    # dynamic=False sidesteps a dynamo symbolic-shapes bug (torch 2.12,
    # sourceless symbols from per-param state ints). The fused extension
    # kernels stay IN the traced graph as torch.ops.gefen custom ops; the only
    # partition points are the deliberate @torch._dynamo.disable host helpers
    # (the batched scalar-refresh prologue and the global-step tail), so a
    # zero-graph-break assertion would be wrong by design. This test pins
    # numerical parity with the eager capturable step regardless of how dynamo
    # partitions it; the step-latency contract is pinned by the benches.
    steps, warmup = 8, 2

    def run(compiled):
        params = _named_params(0, GEFEN_SPECS)
        opt = Gefen(params, lr=1e-3, weight_decay=0.01, capturable=True)
        grads = _grad_sequence(1, GEFEN_SPECS, steps)
        static = [torch.zeros_like(p) for _, p in params]
        for (_, p), s in zip(params, static):
            p.grad = s
        compiled_fn = (
            torch.compile(opt.step, mode="reduce-overhead", dynamic=False)
            if compiled else None
        )
        for i in range(steps):
            for s, g in zip(static, grads[i]):
                s.copy_(g)
            if compiled and i >= warmup:
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn()
            else:
                opt.step()
        torch.cuda.synchronize()
        return [p.detach().clone() for _, p in params]

    ref = run(compiled=False)
    got = run(compiled=True)
    _assert_params_close(got, ref, rtol=1e-4, atol=1e-6, what="gefen-compile")
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# 3. guards
# ---------------------------------------------------------------------------

def test_capture_without_capturable_raises():
    params = _named_params(0, GEFEN_SPECS)
    opt = Gefen(params, lr=1e-3)
    grads = _grad_sequence(1, GEFEN_SPECS, 2)
    for (_, p), g in zip(params, grads[0]):
        p.grad = g
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        opt.step()  # warmup so capture would not hit host-side init
    torch.cuda.current_stream().wait_stream(side)
    for (_, p), g in zip(params, grads[1]):
        p.grad = g
    graph = torch.cuda.CUDAGraph()
    if pytest is not None:
        with pytest.raises(RuntimeError, match="capturable=False"):
            with torch.cuda.graph(graph):
                opt.step()
    else:
        try:
            with torch.cuda.graph(graph):
                opt.step()
        except RuntimeError as e:
            assert "capturable=False" in str(e)
        else:
            raise AssertionError("capture with capturable=False did not raise")


def test_capturable_rejects_host_driven_options():
    # codebook_refresh_every stays host-driven (a periodic re-learn cannot run
    # inside a replayed graph) and is still rejected...
    try:
        Gefen(
            _named_params(0, GEFEN_SPECS), lr=1e-3, capturable=True,
            codebook_refresh_every=10,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "capturable=True with codebook_refresh_every did not raise"
        )
    # ...but stochastic_round now constructs fine: its per-step seed lives in
    # a device tensor advanced once per step, so a captured graph never
    # freezes it.
    Gefen(
        _named_params(0, GEFEN_SPECS), lr=1e-3, capturable=True,
        stochastic_round=True,
    )


# ---------------------------------------------------------------------------
# 4. checkpoint portability across the capturable toggle
# ---------------------------------------------------------------------------

def _assert_roundtrip_counters(opt, steps, want_tensor, what):
    seen = set()
    for pstate in opt.state.values():
        for key in COUNTER_KEYS:
            if key not in pstate:
                continue
            value = pstate[key]
            if want_tensor:
                assert torch.is_tensor(value) and value.is_cuda and (
                    value.dim() == 0
                ), (what, key, value)
                assert value.item() == steps, (what, key, value)
            else:
                assert isinstance(value, int), (what, key, type(value))
                assert value == steps, (what, key, value)
            seen.add(key)
    return seen


def _roundtrip_case(build, specs, steps=3, what=""):
    """capturable=True run -> load into plain (counters become python ints),
    plain run -> load into capturable=True (counters become 0-dim CUDA fp32
    tensors); both loaded optimizers must keep stepping."""
    def run(capturable):
        params, opt = build(capturable)
        grads = _grad_sequence(2, specs, steps)
        for i in range(steps):
            for (_name, p), g in zip(params, grads[i]):
                p.grad = g.clone()
            opt.step()
        return opt, grads

    opt_cap, grads = run(capturable=True)
    params_plain, opt_plain = build(False)
    opt_plain.load_state_dict(opt_cap.state_dict())
    seen = _assert_roundtrip_counters(
        opt_plain, steps, want_tensor=False, what=what + "-to-plain"
    )
    # ...and it keeps stepping (host math path).
    for (_name, p), g in zip(params_plain, grads[0]):
        p.grad = g.clone()
    opt_plain.step()

    opt_plain2, grads2 = run(capturable=False)
    params_cap, opt_cap2 = build(True)
    opt_cap2.load_state_dict(opt_plain2.state_dict())
    seen |= _assert_roundtrip_counters(
        opt_cap2, steps, want_tensor=True, what=what + "-to-capturable"
    )
    for (_name, p), g in zip(params_cap, grads2[0]):
        p.grad = g.clone()
    opt_cap2.step()
    return seen


def test_state_dict_roundtrip_across_toggle():
    def build(capturable):
        params = _named_params(0, GEFEN_SPECS)
        return params, Gefen(params, lr=1e-3, capturable=capturable)

    seen = _roundtrip_case(build, GEFEN_SPECS, what="gefen")
    assert "step" in seen, seen


def test_state_dict_roundtrip_across_toggle_muon():
    def build(capturable):
        params = _named_params(0, MUON_SPECS)
        return params, GefenMuon(
            params, lr=1e-3, normuon=True, capturable=capturable
        )

    seen = _roundtrip_case(build, MUON_SPECS, what="muon")
    assert "normuon_step" in seen, seen


def test_state_dict_roundtrip_across_toggle_hybrid():
    specs = HYBRID_MUON_SPECS + HYBRID_BACKUP_SPECS

    def build(capturable):
        muon_named = _named_params(0, HYBRID_MUON_SPECS)
        backup_named = _named_params(1, HYBRID_BACKUP_SPECS)
        opt = GefenMuonHybrid(
            muon_named, backup_named, lr=1e-3, weight_decay=0.01,
            capturable=capturable,
        )
        return muon_named + backup_named, opt

    seen = _roundtrip_case(build, specs, what="hybrid")
    assert "step" in seen, seen


# ---------------------------------------------------------------------------
# 5. stochastic rounding under capturable
# ---------------------------------------------------------------------------

def _sr_gefen(ps, capturable, lr=1e-3, stochastic_round=True):
    # factored_v_2d=False so every parameter steps through the vmean-family
    # kernels: with the GEFEN_SPECS sizes the capturable run routes v1-full
    # (deterministic tree reductions -- no atomicAdd feeds the params), so
    # identical runs are bit-reproducible and graph-vs-eager can be asserted
    # exactly.
    return Gefen(ps, lr=lr, weight_decay=0.01, factored_v_2d=False,
                 stochastic_round=stochastic_round, capturable=capturable)


def _run_sr_eager(capturable, steps=6, seed=0, stochastic_round=True):
    params = _named_params(seed, GEFEN_SPECS)
    opt = _sr_gefen(params, capturable, stochastic_round=stochastic_round)
    grads = _grad_sequence(seed + 1, GEFEN_SPECS, steps)
    for i in range(steps):
        for (_, p), g in zip(params, grads[i]):
            p.grad = g.clone()
        opt.step()
    torch.cuda.synchronize()
    return params, opt


def _m_codebooks(opt):
    # Stored quantized-momentum indices in param-group order -- SR acts
    # exactly here, so these are the tensors seed-parity assertions bite on.
    out = []
    for group in opt.param_groups:
        for p in group["params"]:
            pstate = opt.state.get(p)
            if pstate and "m_codebook" in pstate:
                out.append(pstate["m_codebook"].detach().clone())
    return out


def test_sr_capturable_eager_deterministic():
    # (a) two identical capturable+SR eager runs bit-match: the device seed is
    # initialized and advanced identically, so every rounding decision repeats.
    params1, opt1 = _run_sr_eager(capturable=True)
    params2, opt2 = _run_sr_eager(capturable=True)
    for (_, a), (_, b) in zip(params1, params2):
        assert torch.equal(a.detach(), b.detach())
    cbs1, cbs2 = _m_codebooks(opt1), _m_codebooks(opt2)
    assert len(cbs1) == len(GEFEN_SPECS)
    for a, b in zip(cbs1, cbs2):
        assert torch.equal(a, b)
    # ONE per-device int64 seed, advanced once per step, mirroring the host
    # global step.
    seeds = list(opt1._sr_seed_by_device.values())
    assert len(seeds) == 1
    assert seeds[0].dtype == torch.int64 and seeds[0].dim() == 0
    assert seeds[0].item() == 6 == opt1._gefen_global_step


def test_sr_noncapturable_unchanged_and_cross_toggle_seed_parity():
    # (d) the capturable=False host-seed path is untouched: every rounding
    # decision (the stored m_codebook indices) repeats bitwise run-to-run,
    # and no device seed tensor is ever created. (Params are held to a tight
    # allclose, not bitwise: these shapes route v2-full here, whose vmean
    # atomicAdd is documented run-to-run nondeterministic at ulp level --
    # pre-existing and independent of SR.)
    params_a, opt_a = _run_sr_eager(capturable=False)
    params_b, opt_b = _run_sr_eager(capturable=False)
    _assert_params_close(
        [p.detach().clone() for _, p in params_a],
        [p.detach().clone() for _, p in params_b],
        rtol=1e-6, atol=1e-8, what="sr-noncapturable-rerun",
    )
    for a, b in zip(_m_codebooks(opt_a), _m_codebooks(opt_b)):
        assert torch.equal(a, b)
    assert not opt_a._sr_seed_by_device

    # Seed-SEMANTICS parity across the toggle: the device seed holds exactly
    # the host global step at every step, so the SR decisions -- the stored
    # m_codebook indices -- bit-match capturable=True against the legacy
    # host-seeded run. (Params are not compared bitwise across the toggle:
    # capturable may route tiny params v1-full where the host path routes
    # v2-full; m_sign is bit-identical across that routing by kernel contract
    # while vmean/p differ at ulp level.)
    _, opt_c = _run_sr_eager(capturable=True)
    for a, c in zip(_m_codebooks(opt_a), _m_codebooks(opt_c)):
        assert torch.equal(a, c)

    # ...and SR actually bites: nearest rounding lands on different indices.
    _, opt_nr = _run_sr_eager(capturable=True, stochastic_round=False)
    assert any(
        not torch.equal(a, n)
        for a, n in zip(_m_codebooks(opt_a), _m_codebooks(opt_nr))
    )


def test_graph_capture_sr_parity():
    # (b) warmup + capture + K replays vs the same optimizer eager, same
    # grads/lr schedule: the device seed advances once per replay exactly as
    # the eager capturable step advances it, so step k bit-matches step k.
    steps, warmup = 7, 3
    lr_sched = [1e-3 * (0.9 ** i) for i in range(steps)]
    grads = _grad_sequence(1, GEFEN_SPECS, steps)

    ref_params = _named_params(0, GEFEN_SPECS)
    ref_lr = torch.tensor(lr_sched[0], device=DEVICE)
    ref_opt = _sr_gefen(ref_params, True, lr=ref_lr)
    for i in range(steps):
        for (_, p), g in zip(ref_params, grads[i]):
            p.grad = g.clone()
        ref_lr.fill_(lr_sched[i])
        ref_opt.step()
    torch.cuda.synchronize()

    params, opt = _run_captured(
        lambda ps, lr: _sr_gefen(ps, True, lr=lr),
        GEFEN_SPECS, grads, lr_sched, warmup, 0,
    )
    # SR acts on the stored indices: exact.
    for a, b in zip(_m_codebooks(opt), _m_codebooks(ref_opt)):
        assert torch.equal(a, b)
    # v1-full arithmetic is deterministic, so the params match bitwise too.
    for (_, a), (_, b) in zip(params, ref_params):
        assert torch.equal(a.detach(), b.detach())
    # The seed advanced once per eager step and once per replay on both sides
    # (capture itself records the advance without executing it).
    for o in (opt, ref_opt):
        (seed,) = o._sr_seed_by_device.values()
        assert seed.item() == steps


def test_sr_seed_advances_and_dither_varies_across_replays():
    # (c) anti-freeze: across replays with IDENTICAL grads the dither must
    # VARY -- a frozen (capture-time) seed would repeat the exact rounding
    # pattern every replay.
    warmup, replays = 2, 12
    params = _named_params(0, GEFEN_SPECS)
    lr = torch.tensor(1e-3, device=DEVICE)
    opt = _sr_gefen(params, True, lr=lr)
    fixed_grads = _grad_sequence(1, GEFEN_SPECS, 1)[0]
    static = [torch.zeros_like(p) for _, p in params]
    for (_, p), s in zip(params, static):
        p.grad = s
    for s, g in zip(static, fixed_grads):
        s.copy_(g)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            opt.step()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()

    snapshots = []
    for _ in range(replays):
        graph.replay()
        torch.cuda.synchronize()
        snapshots.append(_m_codebooks(opt))

    # The seed advanced by exactly the replay count (capture only RECORDS the
    # on-device add_, it does not execute it).
    (seed,) = opt._sr_seed_by_device.values()
    assert seed.item() == warmup + replays

    # And the rounding pattern varies between consecutive replays: with the
    # momentum EMA settled on constant grads, index changes are (almost) pure
    # dither near codeword boundaries.
    varied = sum(
        1
        for prev, cur in zip(snapshots, snapshots[1:])
        if any(not torch.equal(a, b) for a, b in zip(prev, cur))
    )
    assert varied > 0, (
        "m_codebook indices identical across ALL consecutive replays -- the "
        "stochastic-rounding seed looks frozen in the captured graph"
    )


def test_compile_reduce_overhead_sr():
    # torch.compile(reduce-overhead) with SR: the seed tensor is a
    # static-marked input the gefen custom ops read inside the compiled
    # graph, advanced eagerly at the dynamo-disabled step tail. The stored
    # indices (where SR acts) must match the eager capturable+SR run exactly
    # and the params tightly; the seed advances once per compiled call.
    steps, warmup = 8, 2

    def run(compiled):
        params = _named_params(0, GEFEN_SPECS)
        opt = _sr_gefen(params, True)
        grads = _grad_sequence(1, GEFEN_SPECS, steps)
        static = [torch.zeros_like(p) for _, p in params]
        for (_, p), s in zip(params, static):
            p.grad = s
        compiled_fn = (
            torch.compile(opt.step, mode="reduce-overhead", dynamic=False)
            if compiled else None
        )
        for i in range(steps):
            for s, g in zip(static, grads[i]):
                s.copy_(g)
            if compiled and i >= warmup:
                torch.compiler.cudagraph_mark_step_begin()
                compiled_fn()
            else:
                opt.step()
        torch.cuda.synchronize()
        return [p.detach().clone() for _, p in params], opt

    ref, ref_opt = run(compiled=False)
    got, got_opt = run(compiled=True)
    for a, b in zip(_m_codebooks(got_opt), _m_codebooks(ref_opt)):
        assert torch.equal(a, b)
    _assert_params_close(got, ref, rtol=1e-4, atol=1e-6, what="gefen-sr-compile")
    for o in (ref_opt, got_opt):
        (seed,) = o._sr_seed_by_device.values()
        assert seed.item() == steps
    torch._dynamo.reset()


def test_graph_capture_hybrid_stochastic_round():
    # Hybrid smoke under capture with SR on: both halves (Muon momentum kernel
    # + backup Gefen update kernels) run device-seeded SR inside the graph;
    # the three-way conformance checks of _capture_replay_case must hold and
    # each sub-optimizer's seed must advance once per step/replay.
    specs = HYBRID_MUON_SPECS + HYBRID_BACKUP_SPECS

    def make(ps, lr):
        muon_named = ps[: len(HYBRID_MUON_SPECS)]
        backup_named = ps[len(HYBRID_MUON_SPECS):]
        return GefenMuonHybrid(
            muon_named, backup_named, lr=lr, weight_decay=0.01,
            capturable=True, stochastic_round=True,
        )

    _, opt = _capture_replay_case(
        make, specs, rtol=5e-2, atol=5e-3, what="hybrid-sr-graph"
    )
    for sub in (opt.muon, opt.backup):
        assert sub is not None and sub._sr_seed_by_device
        for seed in sub._sr_seed_by_device.values():
            assert seed.item() == 7  # warmup 3 + replays 4


if __name__ == "__main__":
    if not CUDA_OK:
        print("CUDA unavailable; skipping capturable suite")
    else:
        for fn_name, fn in sorted(
            {k: v for k, v in globals().items() if k.startswith("test_")}.items()
        ):
            print("::", fn_name)
            fn()
        print("all capturable tests passed")
