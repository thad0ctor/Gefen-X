"""CPU tests for the D4 single-argument hybrid constructor and the REVIEW-1
unified single-slot tensor-lr cache. Run without a GPU."""
import warnings

import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon, GefenMuonHybrid


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
