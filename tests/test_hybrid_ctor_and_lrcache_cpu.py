"""CPU tests for the D4 single-argument hybrid constructor and the REVIEW-1
unified single-slot tensor-lr cache. Run without a GPU."""
import warnings

import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon, GefenMuonHybrid
from gefen.gefen_muon import BATCHED_NS_DEFAULT_WORKSPACE_BYTES


def _param_names(groups):
    return {name for group in groups for name in group.get("param_names", ())}


class TinyLM(nn.Module):
    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.hidden = nn.Linear(dim, dim, bias=True)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        return self.lm_head(self.norm(self.hidden(self.embed(idx))))


def _step(opt, model):
    idx = torch.randint(0, 32, (4, 6))
    tgt = torch.randint(0, 32, (4, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        def closure():
            opt.zero_grad()
            loss = nn.functional.cross_entropy(
                model(idx).reshape(-1, 32), tgt.reshape(-1)
            )
            loss.backward()
            return loss

        return opt.step(closure).item()


# ---- D4: single-argument constructor forms -------------------------------

def test_hybrid_from_module_first_arg_constructs_and_steps():
    model = TinyLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid(model, lr=1e-3, fused=False)
    # hidden weight -> muon; embed + lm_head + norm/bias -> backup.
    muon_names = _param_names(opt.muon.param_groups)
    backup_names = _param_names(opt.backup.param_groups)
    assert any("hidden.weight" in n for n in muon_names)
    assert any("embed" in n for n in backup_names)
    assert any("lm_head" in n for n in backup_names)
    assert not any("embed" in n or "lm_head" in n for n in muon_names)
    assert _step(opt, model) > 0.0


def test_hybrid_named_params_iterable_first_arg():
    model = TinyLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid(list(model.named_parameters()), lr=1e-3, fused=False)
    backup_names = _param_names(opt.backup.param_groups)
    assert any("embed" in n for n in backup_names)
    assert _step(opt, model) > 0.0


def test_hybrid_bare_tensors_rejected():
    model = TinyLM()
    with pytest.raises(TypeError, match=r"can't be routed|Bare tensors"):
        GefenMuonHybrid(list(model.parameters()), lr=1e-3, fused=False)


def test_hybrid_two_list_form_still_works():
    model = TinyLM()
    muon = [("hidden.weight", model.hidden.weight)]
    backup = [
        (n, p)
        for n, p in model.named_parameters()
        if p is not model.hidden.weight
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid(muon, backup, lr=1e-3, fused=False)
    assert opt.muon is not None and opt.backup is not None
    assert _step(opt, model) > 0.0


def test_backup_substrings_with_two_lists_raises():
    model = TinyLM()
    muon = [("hidden.weight", model.hidden.weight)]
    backup = [("norm.weight", model.norm.weight)]
    with pytest.raises(TypeError, match="backup_substrings is only valid"):
        GefenMuonHybrid(muon, backup, lr=1e-3, backup_substrings=("embed",))


def test_backup_substrings_override_on_module_form():
    model = TinyLM()
    # Force the hidden weight to the backup by name.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid(
            model, lr=1e-3, fused=False, backup_substrings=("embed", "lm_head", "hidden")
        )
    muon_names = _param_names(opt.muon.param_groups) if opt.muon else set()
    assert not any("hidden.weight" in n for n in muon_names)


# ---- Configurable Gefen / AdamW backup -----------------------------------

def test_hybrid_backup_optimizer_selector_preserves_names_and_repr():
    default_model = TinyLM()
    default = GefenMuonHybrid.from_model(default_model, lr=1e-3, fused=False)
    assert isinstance(default.backup, Gefen)
    assert default.backup_optimizer == "gefen"

    adamw_model = TinyLM()
    adamw = GefenMuonHybrid.from_model(
        adamw_model, lr=1e-3, backup_optimizer="adamw", fused=False
    )
    assert isinstance(adamw.backup, torch.optim.AdamW)
    assert adamw.backup_optimizer == "adamw"
    assert _param_names(adamw.backup.param_groups) == {
        name
        for name, param in adamw_model.named_parameters()
        if param is not adamw_model.hidden.weight
    }
    rendered = repr(adamw)
    assert "muon_params=1" in rendered
    assert "backup_params=5" in rendered
    assert "backup_optimizer='adamw'" in rendered


def test_hybrid_rejects_unknown_backup_optimizer():
    with pytest.raises(ValueError, match="backup_optimizer.*gefen.*adamw"):
        GefenMuonHybrid.from_model(
            TinyLM(), lr=1e-3, backup_optimizer="sgd", fused=False
        )


@pytest.mark.parametrize(("fused", "capturable"), [(False, False), (True, True)])
def test_hybrid_adamw_backup_forwards_execution_flags(fused, capturable):
    param = nn.Parameter(torch.ones(4))
    opt = GefenMuonHybrid(
        [],
        [("bias", param)],
        lr=1e-3,
        backup_optimizer="adamw",
        fused=fused,
        capturable=capturable,
    )
    group = opt.backup.param_groups[0]
    assert group["fused"] is fused
    assert group["capturable"] is capturable


def test_hybrid_adamw_backup_matches_standalone_with_named_no_decay_groups():
    torch.manual_seed(13)
    initial_decay = torch.randn(5, 3)
    initial_no_decay = torch.randn(5)
    hybrid_decay = nn.Parameter(initial_decay.clone())
    hybrid_no_decay = nn.Parameter(initial_no_decay.clone())
    reference_decay = nn.Parameter(initial_decay.clone())
    reference_no_decay = nn.Parameter(initial_no_decay.clone())
    lr = 4e-4
    betas = (0.8, 0.95)
    eps = 2e-7
    weight_decay = 0.2

    opt = GefenMuonHybrid(
        [],
        [("proj.weight", hybrid_decay), ("norm.bias", hybrid_no_decay)],
        lr=1e-3,
        backup_lr=lr,
        backup_weight_decay=weight_decay,
        backup_optimizer="adamw",
        no_decay_substrings=("bias",),
        backup_1d_period_one=True,
        backup_2d_period_one=True,
        betas=betas,
        eps=eps,
        fused=False,
    )
    reference = torch.optim.AdamW(
        [
            {
                "params": [reference_decay],
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
            {
                "params": [reference_no_decay],
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            },
        ],
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        fused=False,
    )

    assert [group["param_names"] for group in opt.backup.param_groups] == [
        ["proj.weight"],
        ["norm.bias"],
    ]
    assert [group["weight_decay"] for group in opt.backup.param_groups] == [
        weight_decay,
        0.0,
    ]
    assert all(group["lr"] == lr for group in opt.backup.param_groups)
    assert all(group["betas"] == betas for group in opt.backup.param_groups)
    assert all(group["eps"] == eps for group in opt.backup.param_groups)

    generator = torch.Generator().manual_seed(19)
    for _ in range(4):
        grad_decay = torch.randn(hybrid_decay.shape, generator=generator)
        grad_no_decay = torch.randn(hybrid_no_decay.shape, generator=generator)
        hybrid_decay.grad = grad_decay.clone()
        reference_decay.grad = grad_decay.clone()
        hybrid_no_decay.grad = grad_no_decay.clone()
        reference_no_decay.grad = grad_no_decay.clone()
        opt.step()
        reference.step()
        opt.zero_grad()
        reference.zero_grad()

    torch.testing.assert_close(hybrid_decay, reference_decay, rtol=0, atol=0)
    torch.testing.assert_close(hybrid_no_decay, reference_no_decay, rtol=0, atol=0)
    assert opt.backup.state[hybrid_decay]["exp_avg_sq"].shape == hybrid_decay.shape
    assert opt.backup.state[hybrid_no_decay]["exp_avg_sq"].shape == hybrid_no_decay.shape


def test_hybrid_adamw_backup_scheduler_preserves_split_lr_ratio():
    model = TinyLM()
    opt = GefenMuonHybrid.from_model(
        model,
        lr=1e-3,
        backup_lr=2.5e-4,
        backup_optimizer="adamw",
        fused=False,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    _step(opt, model)
    scheduler.step()

    assert [group["lr"] for group in opt.muon.param_groups] == pytest.approx([1e-4])
    assert [group["lr"] for group in opt.backup.param_groups] == pytest.approx(
        [2.5e-5]
    )


# ---- Batched-NS ctor kwargs forward to the Muon child ---------------------

def test_hybrid_forwards_batched_ns_opt_in_to_muon_child():
    model = TinyLM()
    workspace = 64 << 20  # distinctive non-default cap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid.from_model(
            model,
            lr=1e-3,
            fused=False,
            batched_ns=True,
            batched_ns_workspace_bytes=workspace,
        )
    # Instance config on the GefenMuon child...
    assert opt.muon._batched_ns is True
    assert opt.muon._batched_ns_workspace_bytes == workspace
    # ... and the per-group entries the step path actually consults.
    for group in opt.muon.param_groups:
        assert group["batched_ns"] is True
        assert group["batched_ns_workspace_bytes"] == workspace
    # The opt-in must not disturb stepping (CPU params are batch-ineligible,
    # so _batched_ns_key routes them serial).
    assert _step(opt, model) > 0.0


def test_hybrid_default_leaves_batched_ns_off_in_muon_child():
    model = TinyLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    assert opt.muon._batched_ns is False
    for group in opt.muon.param_groups:
        assert group["batched_ns"] is False
        assert (
            group["batched_ns_workspace_bytes"]
            == BATCHED_NS_DEFAULT_WORKSPACE_BYTES
        )


# ---- Issue #44: public param_groups preserve caller grouping --------------

def test_gefen_preserves_user_param_groups_for_integrations():
    p0 = nn.Parameter(torch.randn(4, 4))
    p1 = nn.Parameter(torch.randn(4))
    p2 = nn.Parameter(torch.randn(4, 4))
    opt = Gefen(
        [
            {"params": [("layer0.weight", p0), ("layer0.bias", p1)], "lr": 1e-3},
            {"params": [("layer1.weight", p2)], "lr": 2e-3},
        ],
        fused=False,
    )

    assert len(opt.param_groups) == 2
    assert [len(group["params"]) for group in opt.param_groups] == [2, 1]
    assert opt.param_groups[0]["param_names"] == ["layer0.weight", "layer0.bias"]
    assert opt.param_groups[1]["param_names"] == ["layer1.weight"]
    assert len(opt.state_dict()["param_groups"]) == 2

    for group, lr in zip(opt.param_groups, [1e-4, 2e-4]):
        group["lr"] = lr
    assert [group["lr"] for group in opt.param_groups] == [1e-4, 2e-4]


def test_gefen_same_name_params_keep_isolated_state_within_one_group():
    p0 = nn.Parameter(torch.ones(3, 3))
    p1 = nn.Parameter(torch.ones(3, 3) * 2.0)
    opt = Gefen(
        [{"params": [("shared.weight", p0), ("shared.weight", p1)], "lr": 1e-3}],
        fused=False,
    )

    p0.grad = torch.full_like(p0, 0.01)
    p1.grad = torch.full_like(p1, -0.02)
    opt.step()

    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["param_names"] == ["shared.weight", "shared.weight"]
    assert p0 in opt.state and p1 in opt.state
    assert opt.state[p0] is not opt.state[p1]
    assert opt.state[p0]["name"] == "shared.weight"
    assert opt.state[p1]["name"] == "shared.weight"
    assert opt.state[p0]["m_codebook"].data_ptr() != opt.state[p1][
        "m_codebook"
    ].data_ptr()


def test_gefen_auto_names_do_not_collide_when_group_names_repeat():
    p0 = nn.Parameter(torch.ones(2, 2))
    p1 = nn.Parameter(torch.ones(2, 2))
    p2 = nn.Parameter(torch.ones(2, 2))
    opt = Gefen([{"params": [p0], "name": "dup"}], lr=1e-3, fused=False)

    opt.add_param_group({"params": [p1], "name": "dup"})
    opt.add_param_group({"params": [p2], "name": "dup"})

    assert [group["param_names"] for group in opt.param_groups] == [
        ["dup"],
        ["dup_1"],
        ["dup_2"],
    ]
    assert opt.state[p0]["name"] == "dup"
    assert opt.state[p1]["name"] == "dup_1"
    assert opt.state[p2]["name"] == "dup_2"


def test_gefen_per_group_lr_and_weight_decay_route_numerically():
    p_no_decay = nn.Parameter(torch.ones(4))
    p_low_lr = nn.Parameter(torch.ones(4))
    p_high_lr = nn.Parameter(torch.ones(4))
    opt = Gefen(
        [
            {
                "params": [("no_decay", p_no_decay)],
                "lr": 3e-2,
                "weight_decay": 0.0,
            },
            {"params": [("low_lr", p_low_lr)], "lr": 1e-2, "weight_decay": 0.1},
            {"params": [("high_lr", p_high_lr)], "lr": 3e-2, "weight_decay": 0.1},
        ],
        lr=1e-2,
        weight_decay=0.0,
        fused=False,
    )
    for p in (p_no_decay, p_low_lr, p_high_lr):
        p.grad = torch.zeros_like(p)

    opt.step()

    torch.testing.assert_close(p_no_decay, torch.ones_like(p_no_decay))
    torch.testing.assert_close(p_low_lr, torch.full_like(p_low_lr, 1.0 - 1e-2 * 0.1))
    torch.testing.assert_close(
        p_high_lr,
        torch.full_like(p_high_lr, 1.0 - 3e-2 * 0.1),
    )
    low_delta = (1.0 - p_low_lr).abs().mean().item()
    high_delta = (1.0 - p_high_lr).abs().mean().item()
    assert high_delta == pytest.approx(3.0 * low_delta, rel=1e-4)


def test_hybrid_param_groups_are_per_half_not_per_parameter():
    model = TinyLM()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = GefenMuonHybrid(model, lr=1e-3, fused=False)

    assert len(opt.muon.param_groups) == 1
    assert len(opt.backup.param_groups) == 1
    assert len(opt.param_groups) == 2
    assert sum(len(group["params"]) for group in opt.param_groups) == sum(
        1 for _ in model.parameters()
    )


# ---- REVIEW-1: unified single-slot tensor-lr cache -----------------------

def test_gefen_lr_scalar_is_single_slot_not_dict():
    lin = nn.Linear(8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = Gefen(lin.parameters(), lr=torch.tensor(1e-3), fused=False)
    g = opt.param_groups[0]
    assert opt._lr_scalar(g) == pytest.approx(1e-3)
    cache = opt._lr_scalar_cache
    # Single slot is a 3-tuple (tensor, version, value), NOT an id-keyed dict.
    assert isinstance(cache, tuple) and len(cache) == 3
    assert not isinstance(cache, dict)


def test_gefen_lr_scalar_resyncs_on_version_bump():
    lin = nn.Linear(8, 8)
    lr = torch.tensor(1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = Gefen(lin.parameters(), lr=lr, fused=False)
    g = opt.param_groups[0]
    assert opt._lr_scalar(g) == pytest.approx(1e-3)
    lr.mul_(2.0)  # in-place -> bumps _version
    assert opt._lr_scalar(g) == pytest.approx(2e-3)


def test_float_lr_never_touches_cache():
    lin = nn.Linear(8, 8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        opt = Gefen(lin.parameters(), lr=1e-3, fused=False)
    g = opt.param_groups[0]
    assert opt._lr_scalar(g) == 1e-3
    assert getattr(opt, "_lr_scalar_cache", None) is None


def test_gefenmuon_inherits_gefen_lr_scalar():
    # No own override -> the unified single-slot method is used.
    assert GefenMuon._lr_scalar is Gefen._lr_scalar
