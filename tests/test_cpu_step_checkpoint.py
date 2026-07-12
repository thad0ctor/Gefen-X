"""CPU-only coverage for the optimizer step and checkpoint round-trip paths.

Everything here runs without a GPU (fused=False decomposed paths), so it
executes on CPU-only CI runners:

  * regression: _should_use_v2 must never touch the CUDA runtime for a CPU
    param (it used to call torch.cuda.current_device() via _sm_count, crashing
    CPU-only hosts whenever a param's period exceeded _V2_PERIOD_MAX);
  * step smoke for Gefen / GefenMuon / GefenMuonHybrid on CPU params;
  * bit-exact state_dict/load_state_dict resume on CPU;
  * the factored_v_2d <-> legacy-vmean lazy checkpoint migration branches;
  * step-counter normalization across the capturable toggle.

Run: pytest tests/test_cpu_step_checkpoint.py
"""
import copy

import pytest
import torch
import torch.nn as nn

import gefen.kernels.automatic_gefen_fused as fused_mod
from gefen import Gefen, GefenMuon, GefenMuonHybrid


def _small_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(32, 48),
        nn.Tanh(),
        nn.Linear(48, 16),
    )


def _synthetic_grads(model, num_steps, seed=123):
    """One fixed grad tensor per (step, param): drives A/B runs identically."""
    gen = torch.Generator().manual_seed(seed)
    return [
        [torch.randn(p.shape, generator=gen) * 1e-2 for p in model.parameters()]
        for _ in range(num_steps)
    ]


def _apply_grads(model, grads):
    for p, g in zip(model.parameters(), grads):
        p.grad = g.clone()


def test_should_use_v2_cpu_never_touches_cuda(monkeypatch):
    """CPU device + period above _V2_PERIOD_MAX must resolve without CUDA."""

    def _boom(device):
        raise AssertionError("_sm_count must not be consulted for a CPU param")

    monkeypatch.setattr(fused_mod, "_sm_count", _boom)
    dev = torch.device("cpu")
    period = fused_mod._V2_PERIOD_MAX + 1
    assert fused_mod._should_use_v2(10_000_000, period, dev) is False
    # The tiny-period early-out stays device-independent.
    assert fused_mod._should_use_v2(10_000_000, fused_mod._V2_PERIOD_MAX, dev) is True


@pytest.mark.parametrize("factored", [True, False])
def test_gefen_cpu_step_smoke(factored):
    model = _small_model()
    opt = Gefen(
        list(model.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    grads = _synthetic_grads(model, 3)
    before = [p.detach().clone() for p in model.parameters()]
    for step_grads in grads:
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    for b, p in zip(before, model.parameters()):
        assert torch.isfinite(p).all()
        assert not torch.equal(b, p), "step() left a parameter untouched"


def test_gefen_muon_cpu_step_smoke():
    torch.manual_seed(0)
    params = [
        ("blocks.0.w", nn.Parameter(torch.randn(32, 48))),
        ("blocks.1.w", nn.Parameter(torch.randn(48, 16))),
    ]
    opt = GefenMuon(params, lr=1e-3, fused=False, adjust_lr_fn="match_rms_adamw")
    gen = torch.Generator().manual_seed(7)
    for _ in range(3):
        for _, p in params:
            p.grad = torch.randn(p.shape, generator=gen) * 1e-2
        opt.step()
        opt.zero_grad()
    for _, p in params:
        assert torch.isfinite(p).all()


def test_hybrid_cpu_step_smoke():
    model = _small_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    grads = _synthetic_grads(model, 3)
    for step_grads in grads:
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    for p in model.parameters():
        assert torch.isfinite(p).all()


@pytest.mark.parametrize("factored", [True, False])
def test_state_dict_roundtrip_bit_exact(factored):
    """Save/load mid-run, then step both runs on identical grads: bit-exact."""
    grads_all = _synthetic_grads(_small_model(), 5)

    model_a = _small_model()
    opt_a = Gefen(
        list(model_a.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    for step_grads in grads_all[:2]:
        _apply_grads(model_a, step_grads)
        opt_a.step()
        opt_a.zero_grad()

    saved_opt = copy.deepcopy(opt_a.state_dict())
    saved_params = [p.detach().clone() for p in model_a.parameters()]

    model_b = _small_model()
    with torch.no_grad():
        for p, saved in zip(model_b.parameters(), saved_params):
            p.copy_(saved)
    opt_b = Gefen(
        list(model_b.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    opt_b.load_state_dict(saved_opt)

    for step_grads in grads_all[2:]:
        _apply_grads(model_a, step_grads)
        _apply_grads(model_b, step_grads)
        opt_a.step()
        opt_b.step()
        opt_a.zero_grad()
        opt_b.zero_grad()

    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb), "resumed run diverged from continuous run"


def _zero_style_flat_swap(opt):
    """Simulate DeepSpeed ZeRO-1/2 wrapping: replace every group's ``params``
    with one fresh flat fp32 partition tensor, leaving ``opt.state`` untouched
    (exactly what DeepSpeed does -- it never edits the inner optimizer's state,
    so the ``name`` entries Gefen pre-registered for the original model params
    become unreachable from ``param_groups``)."""
    flat_params = []
    for group in opt.param_groups:
        numel = sum(p.numel() for p in group["params"])
        flat = nn.Parameter(torch.zeros(numel, dtype=torch.float32))
        group["params"] = [flat]
        flat_params.append(flat)
    return flat_params


def test_state_dict_survives_zero_style_param_swap():
    """ZeRO swaps group params for flat partitions without touching state;
    state_dict() must skip the orphaned entries instead of KeyError-ing."""
    model = _small_model()
    opt = Gefen(list(model.named_parameters()), lr=1e-3, fused=False)
    _apply_grads(model, _synthetic_grads(model, 1)[0])
    opt.step()
    opt.zero_grad()

    flat_params = _zero_style_flat_swap(opt)
    # DeepSpeed then drives steps on the flat fp32 partitions.
    for flat in flat_params:
        flat.grad = torch.full_like(flat, 1e-3)
    opt.step()
    opt.zero_grad()

    live_keys_before = list(opt.state.keys())
    sd = opt.state_dict()  # KeyError on unfixed code (torch param_mappings)

    # Only the reachable flat partitions are serialized, indexed 0..n-1 in
    # param_groups order; the orphaned original-param entries are withheld.
    n_flat = len(flat_params)
    assert set(sd["state"].keys()) == set(range(n_flat))
    serialized_ids = [
        pid for group in sd["param_groups"] for pid in group["params"]
    ]
    assert serialized_ids == list(range(n_flat))
    for pstate in sd["state"].values():
        assert "vmean" in pstate or "exp_avg" in pstate or len(pstate) > 1

    # Live self.state is untouched: same keys, same order, orphaned entries
    # still carry their pre-registered names.
    assert list(opt.state.keys()) == live_keys_before
    for (name, param) in model.named_parameters():
        assert opt.state[param]["name"] == name.lower()

    # Saving is idempotent (the withhold/restore leaves nothing behind).
    sd2 = opt.state_dict()
    _assert_state_dict_deep_equal(sd2, sd)


def test_zero_style_swap_save_does_not_perturb_normal_save():
    """After undoing the swap, a normal save must match the pristine pre-swap
    save exactly (state + param_groups): the orphan filter fires only while
    orphans exist and never leaks into the normal path."""
    model = _small_model()
    opt = Gefen(list(model.named_parameters()), lr=1e-3, fused=False)
    _apply_grads(model, _synthetic_grads(model, 1)[0])
    opt.step()
    opt.zero_grad()

    pristine = copy.deepcopy(opt.state_dict())
    original_group_params = [list(g["params"]) for g in opt.param_groups]

    _zero_style_flat_swap(opt)
    opt.state_dict()  # swap-path save

    for group, params in zip(opt.param_groups, original_group_params):
        group["params"] = params
    after = opt.state_dict()
    _assert_state_dict_deep_equal(after["state"], pristine["state"], "state")
    _assert_state_dict_deep_equal(
        after["param_groups"], pristine["param_groups"], "param_groups"
    )


def test_gefen_muon_state_dict_survives_zero_style_param_swap():
    """GefenMuon inherits Gefen.state_dict; the orphan filter must cover it
    too (its constructor pre-registers names the same way)."""
    torch.manual_seed(0)
    params = [
        ("blocks.0.w", nn.Parameter(torch.randn(32, 48))),
        ("blocks.1.w", nn.Parameter(torch.randn(48, 16))),
    ]
    opt = GefenMuon(params, lr=1e-3, fused=False, adjust_lr_fn="match_rms_adamw")
    gen = torch.Generator().manual_seed(7)
    for _, p in params:
        p.grad = torch.randn(p.shape, generator=gen) * 1e-2
    opt.step()
    opt.zero_grad()

    _zero_style_flat_swap(opt)
    sd = opt.state_dict()  # KeyError on unfixed code
    assert all(isinstance(pid, int) for pid in sd["state"])
    # Orphaned entries survive live for _param_name lookups.
    for name, p in params:
        assert opt.state[p]["name"] == name


def test_load_state_dict_does_not_mutate_caller_dict():
    model = _small_model()
    opt = Gefen(list(model.named_parameters()), lr=1e-3, fused=False)
    _apply_grads(model, _synthetic_grads(model, 1)[0])
    opt.step()
    sd = opt.state_dict()
    opt.load_state_dict(sd)
    # A second load must still see the gefen_* top-level keys (shallow-copy
    # contract in load_state_dict).
    assert "gefen_global_step" in sd and "gefen_codebook" in sd
    opt.load_state_dict(sd)


def test_load_legacy_flattened_param_group_checkpoint():
    model_src, opt_src = _run_and_save(factored=False)
    legacy_sd = _legacy_flattened_param_groups(opt_src.state_dict())
    assert len(legacy_sd["param_groups"]) == sum(1 for _ in model_src.parameters())

    model_dst = _small_model()
    with torch.no_grad():
        for pd, ps in zip(model_dst.parameters(), model_src.parameters()):
            pd.copy_(ps)
    opt_dst = Gefen(list(model_dst.named_parameters()), lr=1e-3, fused=False)
    opt_dst.load_state_dict(legacy_sd)

    grouped_sd = opt_dst.state_dict()
    assert len(grouped_sd["param_groups"]) == 1
    assert grouped_sd["param_groups"][0]["param_names"] == [
        name for name, _ in model_dst.named_parameters()
    ]
    _apply_grads(model_dst, _synthetic_grads(model_dst, 1)[0])
    opt_dst.step()
    for p in model_dst.parameters():
        assert torch.isfinite(p).all()


def test_load_legacy_flattened_checkpoint_rejects_heterogeneous_group_hypers():
    _, opt_src = _run_and_save(factored=False)
    legacy_sd = _legacy_flattened_param_groups(opt_src.state_dict())
    legacy_sd["param_groups"][1]["lr"] = legacy_sd["param_groups"][0]["lr"] * 2

    model_dst = _small_model()
    opt_dst = Gefen(list(model_dst.named_parameters()), lr=1e-3, fused=False)

    with pytest.raises(
        ValueError,
        match="legacy Gefen checkpoint with per-parameter hyperparameters",
    ):
        opt_dst.load_state_dict(legacy_sd)


def test_load_legacy_flattened_checkpoint_rejects_changed_trainable_set():
    _, opt_src = _run_and_save(factored=False)
    legacy_sd = _legacy_flattened_param_groups(opt_src.state_dict())

    smaller_model = nn.Sequential(nn.Linear(32, 48))
    opt_dst = Gefen(list(smaller_model.named_parameters()), lr=1e-3, fused=False)

    with pytest.raises(ValueError, match="parameter group"):
        opt_dst.load_state_dict(legacy_sd)


def _multi_group_run_and_save():
    """A 2-group [2, 1] live layout, stepped twice, returned with its state.

    Names differ per param so a legacy repack can be shown to restore them in
    destination order.
    """
    torch.manual_seed(0)
    pa = nn.Parameter(torch.randn(6, 4))
    pb = nn.Parameter(torch.randn(6))
    pc = nn.Parameter(torch.randn(5, 3))
    opt = Gefen(
        [
            {"params": [("block.weight", pa), ("block.bias", pb)], "lr": 1e-3},
            {"params": [("head.weight", pc)], "lr": 1e-3},
        ],
        fused=False,
    )
    gen = torch.Generator().manual_seed(3)
    for _ in range(2):
        for p in (pa, pb, pc):
            p.grad = torch.randn(p.shape, generator=gen) * 1e-2
        opt.step()
        opt.zero_grad()
    return [pa, pb, pc], opt


def _fresh_multi_group_opt():
    """A destination optimizer with the same [2, 1] layout but DIFFERENT names,
    so a successful legacy load must overwrite them with the checkpoint's."""
    qa = nn.Parameter(torch.zeros(6, 4))
    qb = nn.Parameter(torch.zeros(6))
    qc = nn.Parameter(torch.zeros(5, 3))
    opt = Gefen(
        [
            {"params": [("x.weight", qa), ("x.bias", qb)], "lr": 1e-3},
            {"params": [("y.weight", qc)], "lr": 1e-3},
        ],
        fused=False,
    )
    return [qa, qb, qc], opt


def test_load_legacy_flattened_checkpoint_multi_group_repack():
    # Live layout is two groups of sizes [2, 1]; the legacy checkpoint is one
    # group per param (3 flat groups). The repacker must fold the 3 flat groups
    # back into the 2 live groups, restoring each group's names in order.
    _, opt_src = _multi_group_run_and_save()
    legacy_sd = _legacy_flattened_param_groups(opt_src.state_dict())
    assert len(legacy_sd["param_groups"]) == 3

    params_dst, opt_dst = _fresh_multi_group_opt()
    opt_dst.load_state_dict(legacy_sd)

    grouped = opt_dst.state_dict()["param_groups"]
    assert [len(g["params"]) for g in grouped] == [2, 1]
    # Names come from the checkpoint (overwriting the destination's x/y names),
    # restored in per-group destination order.
    assert grouped[0]["param_names"] == ["block.weight", "block.bias"]
    assert grouped[1]["param_names"] == ["head.weight"]

    for p in params_dst:
        p.grad = torch.zeros_like(p)
    opt_dst.step()
    for p in params_dst:
        assert torch.isfinite(p).all()


def test_load_legacy_flattened_checkpoint_tensor_valued_hyper():
    # A legacy checkpoint whose per-param groups carry a TENSOR-valued lr routes
    # the repacker's homogeneity check through _same_serialized_group_value's
    # torch.equal branch. All params of live group 0 share the same tensor lr,
    # so the compare returns True and the load succeeds.
    _, opt_src = _multi_group_run_and_save()
    legacy_sd = _legacy_flattened_param_groups(opt_src.state_dict())
    # The first two flat groups map onto live group 0 (size 2): same tensor lr.
    legacy_sd["param_groups"][0]["lr"] = torch.tensor(1e-3)
    legacy_sd["param_groups"][1]["lr"] = torch.tensor(1e-3)

    params_dst, opt_dst = _fresh_multi_group_opt()
    opt_dst.load_state_dict(legacy_sd)
    assert torch.is_tensor(opt_dst.param_groups[0]["lr"])
    for p in params_dst:
        p.grad = torch.zeros_like(p)
    opt_dst.step()
    for p in params_dst:
        assert torch.isfinite(p).all()

    # Heterogeneous tensor lrs across the SAME live group take the same
    # torch.equal branch (returning False) and must raise -- mirroring the
    # scalar heterogeneous-hyper rejection.
    legacy_het = _legacy_flattened_param_groups(opt_src.state_dict())
    legacy_het["param_groups"][0]["lr"] = torch.tensor(1e-3)
    legacy_het["param_groups"][1]["lr"] = torch.tensor(2e-3)
    _, opt_het = _fresh_multi_group_opt()
    with pytest.raises(
        ValueError,
        match="legacy Gefen checkpoint with per-parameter hyperparameters",
    ):
        opt_het.load_state_dict(legacy_het)


def _run_and_save(factored):
    model = _small_model()
    opt = Gefen(
        list(model.named_parameters()), lr=1e-3, fused=False, factored_v_2d=factored
    )
    for step_grads in _synthetic_grads(model, 2):
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    return model, opt


def _state_of_2d(opt):
    for group in opt.param_groups:
        for p in group["params"]:
            if p.ndim == 2:
                return p, opt.state[p]
    raise AssertionError("no 2D param found")


def _legacy_flattened_param_groups(state_dict):
    state_dict = copy.deepcopy(state_dict)
    legacy_groups = []
    for group in state_dict["param_groups"]:
        names = group.get("param_names") or [
            "{}_{}".format(group.get("name", "param"), i)
            for i in range(len(group["params"]))
        ]
        for param_id, name in zip(group["params"], names):
            legacy_group = {
                k: v
                for k, v in group.items()
                if k not in {"params", "param_names"}
            }
            legacy_group["params"] = [param_id]
            legacy_group["name"] = name
            legacy_groups.append(legacy_group)
    state_dict["param_groups"] = legacy_groups
    return state_dict


def test_checkpoint_migration_legacy_to_factored():
    """vmean-era checkpoint resumed with factored_v_2d=True grows v_row/v_col."""
    model_src, opt_src = _run_and_save(factored=False)
    _, state_src = _state_of_2d(opt_src)
    assert "vmean" in state_src and "v_row" not in state_src

    model_dst = _small_model()
    with torch.no_grad():
        for pd, ps in zip(model_dst.parameters(), model_src.parameters()):
            pd.copy_(ps)
    opt_dst = Gefen(
        list(model_dst.named_parameters()), lr=1e-3, fused=False, factored_v_2d=True
    )
    opt_dst.load_state_dict(opt_src.state_dict())
    _apply_grads(model_dst, _synthetic_grads(model_dst, 1)[0])
    opt_dst.step()

    p_dst, state_dst = _state_of_2d(opt_dst)
    assert "v_row" in state_dst and "v_col" in state_dst and "factored_step" in state_dst
    assert state_dst["v_row"].shape == (p_dst.shape[0],)
    assert state_dst["v_col"].shape == (p_dst.shape[1],)
    assert torch.isfinite(p_dst).all()


def test_checkpoint_migration_factored_to_legacy():
    """factored_v checkpoint resumed with the flag off recreates vmean, with
    vmean_step backfilled so bias correction treats it as a fresh EMA."""
    model_src, opt_src = _run_and_save(factored=True)
    _, state_src = _state_of_2d(opt_src)
    assert "v_row" in state_src and "vmean" not in state_src

    model_dst = _small_model()
    with torch.no_grad():
        for pd, ps in zip(model_dst.parameters(), model_src.parameters()):
            pd.copy_(ps)
    opt_dst = Gefen(
        list(model_dst.named_parameters()), lr=1e-3, fused=False, factored_v_2d=False
    )
    opt_dst.load_state_dict(opt_src.state_dict())
    _apply_grads(model_dst, _synthetic_grads(model_dst, 1)[0])
    opt_dst.step()

    p_dst, state_dst = _state_of_2d(opt_dst)
    assert "vmean" in state_dst and "vmean_step" in state_dst
    assert torch.isfinite(p_dst).all()


def test_counter_normalization_across_capturable_toggle():
    """A capturable=False checkpoint loads into capturable=True as 0-dim
    tensors (and back to python ints in the other direction)."""
    _, opt_src = _run_and_save(factored=True)
    sd = opt_src.state_dict()

    model_dst = _small_model()
    opt_capt = Gefen(
        list(model_dst.named_parameters()), lr=1e-3, fused=False, capturable=True
    )
    opt_capt.load_state_dict(sd)
    for group in opt_capt.param_groups:
        for p in group["params"]:
            step = opt_capt.state[p]["step"]
            assert torch.is_tensor(step) and step.ndim == 0
            assert step.dtype == torch.float32

    sd_capt = opt_capt.state_dict()
    model_back = _small_model()
    opt_plain = Gefen(
        list(model_back.named_parameters()), lr=1e-3, fused=False, capturable=False
    )
    opt_plain.load_state_dict(sd_capt)
    for group in opt_plain.param_groups:
        for p in group["params"]:
            assert isinstance(opt_plain.state[p]["step"], int)


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_hybrid_state_dict_roundtrip(backup_optimizer):
    model = _small_model()
    opt = GefenMuonHybrid.from_model(
        model,
        lr=1e-3,
        backup_optimizer=backup_optimizer,
        fused=False,
    )
    for step_grads in _synthetic_grads(model, 2):
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()
    sd = copy.deepcopy(opt.state_dict())
    assert set(sd.keys()) == {"muon", "backup", "backup_optimizer"}
    assert sd["backup_optimizer"] == backup_optimizer

    model_b = _small_model()
    with torch.no_grad():
        for param_b, param in zip(model_b.parameters(), model.parameters()):
            param_b.copy_(param)
    opt_b = GefenMuonHybrid.from_model(
        model_b,
        lr=1e-3,
        backup_optimizer=backup_optimizer,
        fused=False,
    )
    opt_b.load_state_dict(sd)

    next_grads = _synthetic_grads(model, 1, seed=987)[0]
    _apply_grads(model, next_grads)
    _apply_grads(model_b, next_grads)
    opt.step()
    opt_b.step()
    for param, param_b in zip(model.parameters(), model_b.parameters()):
        assert torch.equal(param, param_b)


def test_hybrid_loads_untagged_legacy_gefen_checkpoint_bit_exact():
    model = _small_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    grads = _synthetic_grads(model, 4, seed=456)
    for step_grads in grads[:2]:
        _apply_grads(model, step_grads)
        opt.step()
        opt.zero_grad()

    legacy_state = copy.deepcopy(opt.state_dict())
    legacy_state.pop("backup_optimizer")
    assert "backup_optimizer" not in legacy_state

    restored_model = _small_model(seed=1)
    with torch.no_grad():
        for restored_param, param in zip(
            restored_model.parameters(), model.parameters()
        ):
            restored_param.copy_(param)
    restored = GefenMuonHybrid.from_model(restored_model, lr=1e-3, fused=False)
    restored.load_state_dict(legacy_state)

    for step_grads in grads[2:]:
        _apply_grads(model, step_grads)
        _apply_grads(restored_model, step_grads)
        opt.step()
        restored.step()
        opt.zero_grad()
        restored.zero_grad()

    for param, restored_param in zip(
        model.parameters(), restored_model.parameters()
    ):
        assert torch.equal(param, restored_param), (
            "legacy untagged checkpoint diverged after resume"
        )


def _assert_state_dict_deep_equal(actual, expected, path="state_dict"):
    """Recursive bit-exact compare (torch.equal on tensors, == elsewhere)."""
    assert type(actual) is type(expected), path
    if isinstance(actual, dict):
        assert set(actual.keys()) == set(expected.keys()), path
        for key in actual:
            _assert_state_dict_deep_equal(
                actual[key], expected[key], "{}[{!r}]".format(path, key)
            )
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected), path
        for i, (item_a, item_e) in enumerate(zip(actual, expected)):
            _assert_state_dict_deep_equal(
                item_a, item_e, "{}[{}]".format(path, i)
            )
    elif torch.is_tensor(actual):
        assert actual.dtype == expected.dtype, path
        assert torch.equal(actual, expected), path
    else:
        assert actual == expected, path


def _assert_rejected_load_is_true_no_op(
    foreign_model, checkpoint_backend, target_backend, raises_match
):
    """Prove a rejected load is a true no-op: save a foreign checkpoint from
    ``foreign_model``, attempt to load it into a target hybrid (expecting
    ``raises_match``), then assert the target's state_dict is deep-equal to
    the pre-attempt snapshot and its subsequent steps stay bit-exact vs a
    control optimizer that never attempted the load."""
    grads_all = _synthetic_grads(_small_model(), 5)

    foreign = GefenMuonHybrid.from_model(
        foreign_model,
        lr=1e-3,
        backup_optimizer=checkpoint_backend,
        fused=False,
    )
    for step_grads in _synthetic_grads(foreign_model, 2, seed=321):
        _apply_grads(foreign_model, step_grads)
        foreign.step()
        foreign.zero_grad()
    foreign_sd = copy.deepcopy(foreign.state_dict())

    # Target and control: identical init, stepped on identical grads.
    model_t = _small_model()
    model_c = _small_model()
    opt_t = GefenMuonHybrid.from_model(
        model_t, lr=1e-3, backup_optimizer=target_backend, fused=False
    )
    opt_c = GefenMuonHybrid.from_model(
        model_c, lr=1e-3, backup_optimizer=target_backend, fused=False
    )
    for step_grads in grads_all[:2]:
        for model, opt in ((model_t, opt_t), (model_c, opt_c)):
            _apply_grads(model, step_grads)
            opt.step()
            opt.zero_grad()

    snapshot = copy.deepcopy(opt_t.state_dict())
    with pytest.raises(ValueError, match=raises_match):
        opt_t.load_state_dict(foreign_sd)

    # (a) Optimizer state is fully intact after the rejected load.
    _assert_state_dict_deep_equal(opt_t.state_dict(), snapshot)

    # (b) Subsequent steps stay bit-exact vs the control run.
    for step_grads in grads_all[2:]:
        for model, opt in ((model_t, opt_t), (model_c, opt_c)):
            _apply_grads(model, step_grads)
            opt.step()
            opt.zero_grad()
    for param_t, param_c in zip(model_t.parameters(), model_c.parameters()):
        assert torch.equal(param_t, param_c), (
            "rejected load perturbed subsequent training"
        )


@pytest.mark.parametrize(
    ("target_backend", "checkpoint_backend"),
    [("gefen", "adamw"), ("adamw", "gefen")],
)
def test_hybrid_rejected_cross_backend_load_is_a_true_no_op(
    target_backend, checkpoint_backend
):
    """A rejected cross-backend load must be a true no-op: state_dict deep-equal
    to the pre-attempt snapshot, and subsequent steps bit-exact vs a control
    optimizer that never attempted the load."""
    # Foreign checkpoint saved by a hybrid on the OTHER backup backend.
    _assert_rejected_load_is_true_no_op(
        _small_model(seed=1),
        checkpoint_backend,
        target_backend,
        raises_match="backup_optimizer",
    )


@pytest.mark.parametrize("backup_optimizer", ["gefen", "adamw"])
def test_hybrid_rejected_backup_layout_load_is_a_true_no_op(backup_optimizer):
    """A load whose backup half has a mismatched internal layout must roll the
    already-loaded muon half back: state_dict deep-equal to the pre-attempt
    snapshot, and subsequent steps bit-exact vs a control that never attempted
    the load. (The muon child loads first, so without rollback the hybrid was
    left half-loaded: new muon, old backup.)"""
    # Foreign checkpoint: identical muon half (same 2D linear weights), but an
    # extra LayerNorm adds two 1D params to the backup half, so the muon child
    # loads cleanly while the backup child's layout is rejected.
    torch.manual_seed(1)
    foreign_model = nn.Sequential(
        nn.Linear(32, 48),
        nn.Tanh(),
        nn.Linear(48, 16),
        nn.LayerNorm(16),
    )
    _assert_rejected_load_is_true_no_op(
        foreign_model,
        backup_optimizer,
        backup_optimizer,
        raises_match="parameter group",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
