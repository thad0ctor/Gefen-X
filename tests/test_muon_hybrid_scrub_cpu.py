"""CPU-only regression coverage for the muon/hybrid correctness batch.

Covers, in order:
  * GefenMuonHybrid.load_state_dict schema guard (standard {"state",
    "param_groups"} checkpoints, half-presence mismatches, and backup-optimizer
    mismatches raise before either child is loaded);
  * module-type-aware embedding/head routing in split_params_for_muon
    (GPT-2-like wte/wpe + tied lm_head, T5-like `shared`, classifier heads,
    tied heads under non-suggestive names) + the split summary log;
  * GefenMuon constructor validation (ns_steps in [1, 100), momentum in [0, 1),
    eps > 0) and the _normalize_ns_schedule ns_steps floor;
  * hybrid instance-level step / state_dict / load_state_dict hook dispatch
    (they registered fine but never fired);
  * GefenMuon.step advancing the global step through the dynamo-disabled
    helper (compile recompile-storm regression);
  * validate_split rejecting bare tensors with a helpful TypeError;
  * the single-slot _lr_scalar cache (no unbounded growth on fresh lr tensors).

Run: pytest tests/test_muon_hybrid_scrub_cpu.py
"""
import logging
from unittest import mock

import pytest
import torch
import torch.nn as nn

from gefen import (
    GefenMuon,
    GefenMuonHybrid,
    split_params_for_muon,
    validate_split,
)
from gefen.gefen_muon import _normalize_ns_schedule


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(16, 24), nn.Tanh(), nn.Linear(24, 8))


def _params_2d(name="w", shape=(8, 12), seed=0):
    torch.manual_seed(seed)
    return [(name, nn.Parameter(torch.randn(*shape)))]


def _step_once(opt, model):
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-2
    opt.step()
    opt.zero_grad()


# ---------------------------------------------------------------------------
# 1. hybrid load_state_dict schema guard
# ---------------------------------------------------------------------------

def test_hybrid_load_rejects_standard_schema_checkpoint():
    opt = GefenMuonHybrid.from_model(_tiny_model(), lr=1e-3, fused=False)
    # A standard-format checkpoint (e.g. FSDP/DeepSpeed-consolidated) used to be
    # silently skipped -- training resumed with zeroed momentum.
    with pytest.raises(ValueError, match="not supported"):
        opt.load_state_dict({"state": {}, "param_groups": []})


def test_hybrid_load_rejects_missing_half():
    model = _tiny_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    _step_once(opt, model)
    sd = opt.state_dict()
    with pytest.raises(ValueError, match="backup"):
        opt.load_state_dict({"muon": sd["muon"], "backup": None})
    with pytest.raises(ValueError, match="muon"):
        opt.load_state_dict({"backup": sd["backup"]})


def test_hybrid_load_rejects_extra_half():
    model = _tiny_model()
    full = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    _step_once(full, model)
    sd = full.state_dict()
    # A muon-only hybrid must refuse a checkpoint that carries a backup half.
    muon_only = GefenMuonHybrid(
        [(n, p) for n, p in model.named_parameters() if p.ndim == 2], [],
        lr=1e-3, fused=False,
    )
    with pytest.raises(ValueError, match="backup"):
        muon_only.load_state_dict(sd)


def test_hybrid_single_half_roundtrip_still_works():
    model = _tiny_model()
    muon_named = [(n, p) for n, p in model.named_parameters() if p.ndim == 2]
    backup_named = [(n, p) for n, p in model.named_parameters() if p.ndim != 2]

    muon_only = GefenMuonHybrid(muon_named, [], lr=1e-3, fused=False)
    _step_once(muon_only, model)
    muon_only.load_state_dict(muon_only.state_dict())

    backup_only = GefenMuonHybrid([], backup_named, lr=1e-3, fused=False)
    _step_once(backup_only, model)
    backup_only.load_state_dict(backup_only.state_dict())


@pytest.mark.parametrize(
    ("source_backend", "target_backend"),
    [("gefen", "adamw"), ("adamw", "gefen")],
)
def test_hybrid_load_rejects_backup_optimizer_mismatch_before_child_load(
    source_backend, target_backend
):
    source_model = _tiny_model()
    source = GefenMuonHybrid.from_model(
        source_model,
        lr=1e-3,
        backup_optimizer=source_backend,
        fused=False,
    )
    _step_once(source, source_model)

    target = GefenMuonHybrid.from_model(
        _tiny_model(1),
        lr=1e-3,
        backup_optimizer=target_backend,
        fused=False,
    )
    with (
        mock.patch.object(target.muon, "load_state_dict") as muon_load,
        mock.patch.object(target.backup, "load_state_dict") as backup_load,
        pytest.raises(ValueError, match="backup_optimizer"),
    ):
        target.load_state_dict(source.state_dict())
    muon_load.assert_not_called()
    backup_load.assert_not_called()


def test_hybrid_legacy_untagged_checkpoint_is_gefen_not_adamw():
    source_model = _tiny_model()
    source = GefenMuonHybrid.from_model(source_model, lr=1e-3, fused=False)
    _step_once(source, source_model)
    legacy_state = source.state_dict()
    legacy_state.pop("backup_optimizer")

    target = GefenMuonHybrid.from_model(
        _tiny_model(1),
        lr=1e-3,
        backup_optimizer="adamw",
        fused=False,
    )
    with pytest.raises(ValueError, match="backup_optimizer"):
        target.load_state_dict(legacy_state)


def test_hybrid_backup_optimizer_tag_is_ignored_without_backup_half():
    source_params = _params_2d(seed=3)
    target_params = _params_2d(seed=4)
    source = GefenMuonHybrid(
        source_params, [], lr=1e-3, backup_optimizer="gefen", fused=False
    )
    source_params[0][1].grad = torch.randn_like(source_params[0][1])
    source.step()

    target = GefenMuonHybrid(
        target_params, [], lr=1e-3, backup_optimizer="adamw", fused=False
    )
    target.load_state_dict(source.state_dict())


def test_hybrid_load_rejects_unknown_backup_optimizer_tag():
    model = _tiny_model()
    source = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    state = source.state_dict()
    state["backup_optimizer"] = "sgd"
    with pytest.raises(ValueError, match="backup_optimizer"):
        source.load_state_dict(state)


# ---------------------------------------------------------------------------
# 2. module-type-aware embedding/head routing
# ---------------------------------------------------------------------------

class _TinyGPT2(nn.Module):
    """GPT-2-ish: token + POSITION embeddings and a TIED lm_head."""

    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50, 16)
        self.wpe = nn.Embedding(32, 16)
        self.attn = nn.Linear(16, 16)
        self.ln_f = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 50, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying


class _TinyT5(nn.Module):
    """T5-ish: the shared embedding's name matches NO backup substring."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(64, 16)
        self.q = nn.Linear(16, 16, bias=False)


class _TinySeqCls(nn.Module):
    """Classifier heads: HF names them `score` / `classifier`."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(16, 16)
        self.score = nn.Linear(16, 4, bias=False)
        self.classifier = nn.Linear(16, 3, bias=False)


class _TiedHeadFirst(nn.Module):
    """Tied head registered BEFORE its embedding, under a non-suggestive name,
    so named_parameters surfaces the shared tensor as `decoder.weight`: only
    the tensor-identity (module-type) check can route it to the backup."""

    def __init__(self):
        super().__init__()
        self.decoder = nn.Linear(16, 64, bias=False)
        self.tok = nn.Embedding(64, 16)
        self.decoder.weight = self.tok.weight


def _split_names(model, **kwargs):
    muon, backup = split_params_for_muon(model, **kwargs)
    return {n for n, _ in muon}, {n for n, _ in backup}


def test_split_routes_gpt2_embeddings_and_tied_head_to_backup():
    muon_names, backup_names = _split_names(_TinyGPT2())
    assert muon_names == {"attn.weight"}
    # tied lm_head dedups onto wte.weight in named_parameters
    assert {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias", "attn.bias"} <= backup_names
    assert not any("wte" in n or "wpe" in n or "lm_head" in n for n in muon_names)


def test_split_routes_t5_shared_embedding_to_backup():
    muon_names, backup_names = _split_names(_TinyT5())
    # "shared" matches no name substring; module-type routing must catch it.
    assert "shared.weight" in backup_names
    assert muon_names == {"q.weight"}


def test_split_routes_classifier_heads_to_backup():
    muon_names, backup_names = _split_names(_TinySeqCls())
    assert {"score.weight", "classifier.weight"} <= backup_names
    assert muon_names == {"backbone.weight"}


def test_split_routes_tied_head_by_tensor_identity():
    muon_names, backup_names = _split_names(_TiedHeadFirst())
    # The shared tensor surfaces as decoder.weight; it IS the embedding weight.
    assert "decoder.weight" in backup_names
    assert "decoder.weight" not in muon_names


def test_split_logs_summary(caplog):
    with caplog.at_level(logging.INFO, logger="gefen.params"):
        split_params_for_muon(_TinyGPT2())
    records = [r for r in caplog.records if "split_params_for_muon" in r.getMessage()]
    assert len(records) == 1
    assert "wte.weight" in records[0].getMessage()


def test_hybrid_from_model_with_tied_embeddings():
    model = _TinyGPT2()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    _step_once(opt, model)
    for p in model.parameters():
        assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# 3. + 7. constructor validation: ns_steps, momentum, eps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ns_steps", [0, -1])
def test_gefen_muon_rejects_non_positive_ns_steps(ns_steps):
    with pytest.raises(ValueError, match="ns_steps"):
        GefenMuon(_params_2d(), fused=False, ns_steps=ns_steps)


@pytest.mark.parametrize("ns_steps", [0, -3])
def test_normalize_ns_schedule_rejects_non_positive_steps(ns_steps):
    with pytest.raises(ValueError, match="ns_steps"):
        _normalize_ns_schedule((3.0, -4.0, 2.0), ns_steps=ns_steps)


def test_gefen_muon_rejects_momentum_one():
    # momentum=1.0 used to fail deep inside Gefen with an AdamW-beta message.
    with pytest.raises(ValueError, match="momentum"):
        GefenMuon(_params_2d(), fused=False, momentum=1.0)


@pytest.mark.parametrize("eps", [0.0, -1e-7])
def test_gefen_muon_rejects_non_positive_eps(eps):
    # eps=0 + an all-zero momentum matrix -> 0/0 = NaN in the NS normalization.
    with pytest.raises(ValueError, match="eps"):
        GefenMuon(_params_2d(), fused=False, eps=eps)


# ---------------------------------------------------------------------------
# 4. hybrid instance hook dispatch
# ---------------------------------------------------------------------------

def test_hybrid_instance_hooks_fire_exactly_once():
    model = _tiny_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    calls = {k: [] for k in ("pre", "post", "sd_pre", "sd_post", "lsd_pre", "lsd_post")}

    opt.register_step_pre_hook(lambda o, a, k: calls["pre"].append(o))
    opt.register_step_post_hook(lambda o, a, k: calls["post"].append(o))
    opt.register_state_dict_pre_hook(lambda o: calls["sd_pre"].append(o))
    opt.register_state_dict_post_hook(lambda o, sd: calls["sd_post"].append((o, sd)))
    opt.register_load_state_dict_pre_hook(lambda o, sd: calls["lsd_pre"].append((o, sd)))
    opt.register_load_state_dict_post_hook(lambda o: calls["lsd_post"].append(o))

    _step_once(opt, model)
    sd = opt.state_dict()
    opt.load_state_dict(sd)

    assert calls["pre"] == [opt]
    assert calls["post"] == [opt]
    assert calls["sd_pre"] == [opt]
    assert len(calls["sd_post"]) == 1
    assert calls["sd_post"][0][0] is opt
    assert set(calls["sd_post"][0][1]) == {
        "muon",
        "backup",
        "backup_optimizer",
    }
    assert len(calls["lsd_pre"]) == 1
    assert calls["lsd_pre"][0][0] is opt
    assert calls["lsd_post"] == [opt]


def test_hybrid_step_pre_hook_can_rewrite_args():
    # Mirror torch.optim.Optimizer semantics: a pre-hook may return
    # (new_args, new_kwargs), and a bogus return raises RuntimeError.
    model = _tiny_model()
    opt = GefenMuonHybrid.from_model(model, lr=1e-3, fused=False)
    seen = []

    def closure():
        seen.append("closure")
        return 42.0

    opt.register_step_pre_hook(lambda o, a, k: ((o, closure), {}))
    for p in model.parameters():
        p.grad = torch.randn_like(p) * 1e-2
    loss = opt.step()
    assert seen == ["closure"] and loss == 42.0

    opt2 = GefenMuonHybrid.from_model(_tiny_model(1), lr=1e-3, fused=False)
    opt2.register_step_pre_hook(lambda o, a, k: "bogus")
    with pytest.raises(RuntimeError, match="new_args"):
        opt2.step()


# ---------------------------------------------------------------------------
# 5. global step advances through the dynamo-disabled helper
# ---------------------------------------------------------------------------

def test_muon_step_uses_advance_gefen_global_step_helper():
    params = _params_2d()
    opt = GefenMuon(params, lr=1e-3, fused=False)
    with mock.patch.object(
        GefenMuon, "_advance_gefen_global_step", autospec=True
    ) as helper:
        params[0][1].grad = torch.randn_like(params[0][1]) * 1e-2
        opt.step()
    helper.assert_called_once()

    # ... and unpatched, the counter still increments once per step.
    start = opt._gefen_global_step
    for _ in range(3):
        params[0][1].grad = torch.randn_like(params[0][1]) * 1e-2
        opt.step()
    assert opt._gefen_global_step == start + 3


# ---------------------------------------------------------------------------
# 8. validate_split bare-tensor robustness
# ---------------------------------------------------------------------------

def test_validate_split_rejects_bare_tensors():
    w = nn.Parameter(torch.randn(4, 4))
    with pytest.raises(TypeError, match="named_parameters"):
        validate_split([w], [])
    # A 2-row tensor used to tuple-unpack into two rows and "validate".
    with pytest.raises(TypeError, match="from_model"):
        validate_split([], [torch.randn(2, 5)])
    with pytest.raises(TypeError, match=r"\(name, param\)"):
        validate_split([("w", "not a tensor")], [])
    with pytest.raises(TypeError, match=r"\(name, param\)"):
        validate_split([("w", w, "extra")], [])


# ---------------------------------------------------------------------------
# 9. single-slot _lr_scalar cache
# ---------------------------------------------------------------------------

def test_lr_scalar_cache_is_single_slot_and_correct():
    opt = GefenMuon(_params_2d(), lr=1e-3, fused=False)

    # A loop assigning a FRESH tensor lr each step must not grow the cache.
    for i in range(5):
        t = torch.tensor(1e-3 * (i + 1))
        assert opt._lr_scalar({"lr": t}) == pytest.approx(t.item())
    cache = opt._lr_scalar_cache
    assert isinstance(cache, tuple) and len(cache) == 3

    # In-place scheduler updates (a _version bump) are observed...
    t = torch.tensor(2e-3)
    assert opt._lr_scalar({"lr": t}) == pytest.approx(2e-3)
    t.mul_(0.5)
    assert opt._lr_scalar({"lr": t}) == pytest.approx(1e-3)
    # ... unchanged tensors hit the cache (same value back), and float lrs
    # pass through untouched.
    assert opt._lr_scalar({"lr": t}) == pytest.approx(1e-3)
    assert opt._lr_scalar({"lr": 3e-4}) == 3e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
