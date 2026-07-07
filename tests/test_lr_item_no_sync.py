"""Regression test for the tensor-LR `.item()` hoist (GefenMuon._lr_scalar).

GefenMuon assigns one param-group per param. The pre-change `_adjust_lr` did a
fresh `lr.item()` (a D2H sync) on every per-param call, so a tensor LR meant
~one sync per param per step, serializing the Newton-Schulz pipeline. The fix
caches the scalar read keyed on `(tensor identity, tensor._version)`.

This test pins the three contractual properties of that cache:
  1. STEADY STATE: a *constant* tensor LR adds ZERO `Tensor.item()` D2H syncs
     per param after warmup (vs a python-float LR baseline). FAILS pre-fix
     (the per-param `lr.item()` shows up as `n_params` extra syncs per step).
  2. VERSION-BUMP RE-READ: an in-place scheduler step (`lr.mul_`, which bumps
     `_version`) is observed by the cache -- the scheduled-LR run is bit-exact
     to an equivalent python-float schedule and differs from a constant LR.
  3. BIT-EXACT VALUE: a constant tensor LR is bit-exact to the equivalent
     python-float LR (the cached value equals `lr.item()`).

Run: python tests/test_lr_item_no_sync.py   (or via pytest)
"""
import torch

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from gefen import GefenMuon  # noqa: E402

_CUDA = torch.cuda.is_available()
_SKIP = "GefenMuon fused path requires CUDA"

# A handful of 2D matrices of mixed shape -> several one-param groups.
_SPEC = [(256, 256), (384, 256), (256, 384), (512, 256), (256, 512)]


def _build(lr, seed=1234, dev="cuda"):
    """Build a GefenMuon over the `_SPEC` 2D matrices (one param-group per param)."""
    torch.manual_seed(seed)
    params = []
    for i, (r, c) in enumerate(_SPEC):
        p = (torch.randn(r, c, device=dev, dtype=torch.bfloat16) * 0.02).requires_grad_(True)
        params.append((f"l{i}", p))
    opt = GefenMuon(params, lr=lr, ns_steps=5, fused=True, adjust_lr_fn="match_rms_adamw")
    return opt, params


def _assign_grads(params, gen, dev="cuda"):
    """Assign deterministic bf16 grads to every param (fixed generator)."""
    for _, p in params:
        p.grad = torch.randn(p.shape, generator=gen, device=dev, dtype=torch.bfloat16) * 1e-3


# ---- Tensor.item() counter ------------------------------------------------
_orig_item = torch.Tensor.item


class _ItemCounter:
    """Context manager that counts `Tensor.item()` calls (D2H syncs) in its body."""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        cnt = self

        def counting_item(self):
            cnt.n += 1
            return _orig_item(self)

        torch.Tensor.item = counting_item
        return self

    def __exit__(self, *a):
        torch.Tensor.item = _orig_item


def _count_items(opt, params, n_steps, dev="cuda"):
    """Run `n_steps` constant-LR steps and return the `Tensor.item()` count."""
    gen = torch.Generator(device=dev).manual_seed(99)
    torch.cuda.synchronize()
    with _ItemCounter() as c:
        for _ in range(n_steps):
            _assign_grads(params, gen, dev)
            opt.step()
        torch.cuda.synchronize()
    return c.n


def test_constant_tensor_lr_adds_zero_item_syncs():
    """Property 1 -- FAILS pre-fix: a constant tensor LR must add no per-param
    D2H syncs over a python-float LR after warmup."""
    if not _CUDA:
        if pytest:
            pytest.skip(_SKIP)
        return
    dev = "cuda"
    n_warm, n_steps = 3, 5

    opt_f, pf = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    for _ in range(n_warm):
        _assign_grads(pf, gen, dev)
        opt_f.step()
    float_items = _count_items(opt_f, pf, n_steps, dev)

    opt_t, pt = _build(torch.tensor(1e-3, device=dev), dev=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    for _ in range(n_warm):
        _assign_grads(pt, gen, dev)
        opt_t.step()
    const_items = _count_items(opt_t, pt, n_steps, dev)

    # The constant tensor LR must not introduce any extra item() syncs. Pre-fix
    # this is float_items + n_params * n_steps.
    assert const_items == float_items, (
        f"constant tensor LR added {const_items - float_items} item() syncs over "
        f"{n_steps} steps / {len(_SPEC)} params (expected 0); the per-param "
        f"lr.item() was not hoisted/cached"
    )


def test_inplace_schedule_reread_on_version_bump():
    """Property 2 -- the cache must re-read when an in-place LR update bumps
    _version: a scheduled tensor LR == equivalent python-float schedule, and
    differs from a constant LR."""
    if not _CUDA:
        if pytest:
            pytest.skip(_SKIP)
        return
    dev = "cuda"
    n_steps = 6
    factor = 0.9

    # (a) in-place-scheduled tensor LR (mul_ each step -> _version bump)
    lr_t = torch.tensor(1e-3, device=dev)
    opt_s, ps = _build(lr_t, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    for _ in range(n_steps):
        lr_t.mul_(factor)
        _assign_grads(ps, gen, dev)
        opt_s.step()
    torch.cuda.synchronize()
    sched = [p.detach().clone() for _, p in ps]

    # (b) equivalent python-float schedule (set group["lr"] directly each step)
    opt_r, pr = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    val = 1e-3
    for _ in range(n_steps):
        val *= factor
        for g in opt_r.param_groups:
            g["lr"] = val
        _assign_grads(pr, gen, dev)
        opt_r.step()
    torch.cuda.synchronize()
    ref = [p.detach().clone() for _, p in pr]

    for (n, _), a, b in zip(ps, sched, ref):
        assert torch.equal(a, b), f"scheduled tensor LR != float schedule at {n}"

    # And the schedule must actually have taken effect (not stale-cached): a
    # constant-LR run must differ.
    opt_c, pc = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    for _ in range(n_steps):
        _assign_grads(pc, gen, dev)
        opt_c.step()
    torch.cuda.synchronize()
    const = [p.detach().clone() for _, p in pc]
    assert any(not torch.equal(a, b) for a, b in zip(sched, const)), (
        "scheduled tensor LR produced identical params to a constant LR -- the "
        "cache served a stale value and ignored the _version bump"
    )


def test_constant_tensor_lr_bit_exact_to_float():
    """Property 3 -- a constant tensor LR is bit-exact to the float LR (the
    cached scalar equals lr.item())."""
    if not _CUDA:
        if pytest:
            pytest.skip(_SKIP)
        return
    dev = "cuda"
    n_steps = 10

    opt_f, pf = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    for _ in range(n_steps):
        _assign_grads(pf, gen, dev)
        opt_f.step()

    opt_t, pt = _build(torch.tensor(1e-3, device=dev), dev=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    for _ in range(n_steps):
        _assign_grads(pt, gen, dev)
        opt_t.step()
    torch.cuda.synchronize()

    for (n, _), (_, b) in zip(pf, pt):
        a = dict(pf)[n]
        assert torch.equal(a.detach(), b.detach()), f"float vs const-tensor LR diverged at {n}"


def test_inplace_schedule_item_syncs_are_o_steps_not_o_params():
    """Property 4 -- FAILS with a per-GROUP cache: under an in-place scheduler
    (`lr.mul_` each step bumps `_version`), the cache must be shared across param
    groups so the extra `.item()` syncs are O(steps), independent of param count.
    A per-group cache re-reads once per param per scheduled step -> O(params*steps).

    Measured as (tensor in-place schedule item count) - (equivalent python-float
    schedule item count): the float schedule does no `lr.item()`, so the diff is
    exactly the lr-driven D2H syncs. With the shared cache that diff == n_steps
    (one read per version bump); with a per-group cache it == n_params*n_steps."""
    if not _CUDA:
        if pytest:
            pytest.skip(_SKIP)
        return
    dev = "cuda"
    n_warm, n_steps, factor = 2, 5, 0.9
    n_params = len(_SPEC)

    # (a) python-float schedule baseline -- set group["lr"] to a float each step.
    opt_f, pf = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    for _ in range(n_warm):
        _assign_grads(pf, gen, dev)
        opt_f.step()
    torch.cuda.synchronize()
    with _ItemCounter() as cf:
        val = 1e-3
        for _ in range(n_steps):
            val *= factor
            for g in opt_f.param_groups:
                g["lr"] = val
            _assign_grads(pf, gen, dev)
            opt_f.step()
        torch.cuda.synchronize()
    float_items = cf.n

    # (b) in-place tensor schedule -- one tensor lr shared by every group,
    # mul_'d each step (bumps _version).
    lr_t = torch.tensor(1e-3, device=dev)
    opt_t, pt = _build(lr_t, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(99)
    for _ in range(n_warm):
        _assign_grads(pt, gen, dev)
        opt_t.step()
    torch.cuda.synchronize()
    with _ItemCounter() as ct:
        for _ in range(n_steps):
            lr_t.mul_(factor)
            _assign_grads(pt, gen, dev)
            opt_t.step()
        torch.cuda.synchronize()
    sched_items = ct.n

    extra = sched_items - float_items
    assert extra <= n_steps, (
        f"in-place scheduled tensor LR added {extra} item() syncs over {n_steps} "
        f"steps / {n_params} params (expected <= {n_steps}; a per-group cache "
        f"gives {n_params * n_steps}). The lr-scalar cache is not shared across "
        f"param groups."
    )


def test_lr_scalar_cache_not_in_param_groups_or_state_dict():
    """Property 5 -- the lr-scalar cache must stay an internal optimizer
    attribute: it must NOT live on any param group or in state_dict() (else it
    bloats / breaks checkpoints, and the per-group variant this PR removed did
    exactly that). Checked for both float and tensor LRs after stepping."""
    if not _CUDA:
        if pytest:
            pytest.skip(_SKIP)
        return
    dev = "cuda"

    for lr in (1e-3, torch.tensor(1e-3, device=dev)):
        opt, params = _build(lr, dev=dev)
        gen = torch.Generator(device=dev).manual_seed(99)
        for _ in range(3):
            _assign_grads(params, gen, dev)
            opt.step()
        torch.cuda.synchronize()

        # Not stored on any param group (live object or serialized form).
        for g in opt.param_groups:
            assert "_lr_scalar_cache" not in g, "cache leaked into a live param group"
        sd = opt.state_dict()
        for g in sd["param_groups"]:
            assert "_lr_scalar_cache" not in g, "cache leaked into state_dict param_groups"
        # And nowhere else in the (shallow) state_dict structure.
        assert "_lr_scalar_cache" not in sd, "cache leaked into state_dict top level"

        # Sanity: for a tensor LR the cache *does* exist as an instance attribute
        # (so we know the absence above is real isolation, not a no-op).
        if isinstance(lr, torch.Tensor):
            assert getattr(opt, "_lr_scalar_cache", None), (
                "tensor LR should have populated the internal instance cache"
            )


def test_fresh_tensor_lr_each_step_identity_invalidation():
    """Property 6 -- _lr_scalar is keyed on (identity, _version); this pins the
    IDENTITY path: replacing group["lr"] with a fresh tensor each step (no
    _version bump on the new object) must still be observed -- bit-exact to the
    equivalent python-float schedule and different from a constant LR."""
    if not _CUDA:
        if pytest:
            pytest.skip(_SKIP)
        return
    dev = "cuda"
    n_steps = 6
    factor = 0.9
    schedule = [1e-3 * (factor ** (i + 1)) for i in range(n_steps)]

    # (a) fresh-tensor-per-step: a brand new lr tensor object each step.
    opt_i, pi = _build(torch.tensor(1e-3, device=dev), dev=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    for v in schedule:
        new_lr = torch.tensor(v, device=dev)  # distinct identity each step
        for g in opt_i.param_groups:
            g["lr"] = new_lr
        _assign_grads(pi, gen, dev)
        opt_i.step()
    torch.cuda.synchronize()
    ident = [p.detach().clone() for _, p in pi]

    # (b) equivalent python-float schedule.
    opt_r, pr = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    for v in schedule:
        for g in opt_r.param_groups:
            g["lr"] = v
        _assign_grads(pr, gen, dev)
        opt_r.step()
    torch.cuda.synchronize()
    ref = [p.detach().clone() for _, p in pr]

    for (n, _), a, b in zip(pi, ident, ref):
        assert torch.equal(a, b), f"fresh-tensor LR != float schedule at {n} (identity not invalidated)"

    # And it must differ from a constant LR (the schedule actually took effect).
    opt_c, pc = _build(1e-3, dev=dev)
    gen = torch.Generator(device=dev).manual_seed(7)
    for _ in range(n_steps):
        _assign_grads(pc, gen, dev)
        opt_c.step()
    torch.cuda.synchronize()
    const = [p.detach().clone() for _, p in pc]
    assert any(not torch.equal(a, b) for a, b in zip(ident, const)), (
        "fresh-tensor schedule produced identical params to a constant LR -- "
        "identity-keyed invalidation served a stale value"
    )


if __name__ == "__main__":
    test_constant_tensor_lr_adds_zero_item_syncs()
    test_inplace_schedule_reread_on_version_bump()
    test_constant_tensor_lr_bit_exact_to_float()
    test_inplace_schedule_item_syncs_are_o_steps_not_o_params()
    test_lr_scalar_cache_not_in_param_groups_or_state_dict()
    test_fresh_tensor_lr_each_step_identity_invalidation()
    print("test_lr_item_no_sync: all passed")
