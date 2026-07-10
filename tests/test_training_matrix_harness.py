"""CPU coverage for the reproducible optimizer/pretraining matrix."""

from __future__ import annotations

import json
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.training_matrix.cells import (
    ALL_CELLS,
    CELL_RECIPES,
    CORE_CELLS,
    ISOLATION_CELLS,
    CellBuildConfig,
    OptimizerPair,
    build_optimizer,
    resolve_cell,
)
import benchmarks.training_matrix.comparison as comparison_module
import benchmarks.training_matrix.consistency_2x2 as consistency_driver
import benchmarks.training_matrix.run_matrix as matrix_driver
from benchmarks.training_matrix.consistency_2x2 import build_plan
from benchmarks.training_matrix.comparison import (
    SOURCE_FINGERPRINT_ENV,
    attach_comparison,
    canonical_json,
    comparison_key,
    source_fingerprint,
)
from benchmarks.training_matrix.data import build_dataset
from benchmarks.training_matrix.hf_sft import (
    _local_model_manifest_sha256,
    accumulate_hf_microbatches,
)
from benchmarks.training_matrix.run_matrix import commands as matrix_commands
from benchmarks.training_matrix.run_matrix import parse_args as parse_matrix_args
from benchmarks.training_matrix.schedule import GroupLRSchedule
from benchmarks.training_matrix.summarize import render_markdown
from benchmarks.training_matrix.tiny_qwen import (
    TinyQwenConfig,
    TinyQwenForCausalLM,
    expected_parameter_count,
    preset_config,
)
from benchmarks.training_matrix.train import (
    accumulate_tiny_microbatches,
    materialize_finite_update_loss,
    normalize_gradients,
    parse_args,
    run,
    snapshot_peak_then_measure_serialized_state,
    training_batch_metadata,
    throughput_measurement_metadata,
    validate_and_clip_gradients,
)
from gefen import Gefen, GefenMuon, GefenMuonHybrid, split_params_for_muon


EXPECTED_CORE_CELLS = (
    "adamw",
    "torch_muon_adamw",
    "gefen_muon_classic_adamw",
    "gefen_hybrid_period1_all",
    "gefen_hybrid_recommended",
    "gefen_hybrid_literal",
    "gefen_hybrid_recommended_2d",
)
EXPECTED_ISOLATION_CELLS = (
    "gefen_muon_tuned3_adamw",
    "gefen_muon_normuon_adamw",
    "gefen_muon_recommended_adamw",
    "gefen_hybrid_split_lr_only",
    "gefen_hybrid_period1_only",
)


def _model(sequence_length: int = 8) -> TinyQwenForCausalLM:
    torch.manual_seed(7)
    return TinyQwenForCausalLM(
        TinyQwenConfig(
            max_seq_len=sequence_length,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )


def test_registry_has_the_seven_explicit_cells_and_unambiguous_labels():
    assert CORE_CELLS == EXPECTED_CORE_CELLS
    assert ISOLATION_CELLS == EXPECTED_ISOLATION_CELLS
    assert set(ALL_CELLS) == set(EXPECTED_CORE_CELLS + EXPECTED_ISOLATION_CELLS)
    assert CELL_RECIPES["gefen_hybrid_literal"].recipe_class == "literal_constructor_defaults"
    assert CELL_RECIPES["gefen_hybrid_recommended"].recipe_class == "recommended"
    assert CELL_RECIPES["gefen_hybrid_recommended_2d"].recipe_class == "recommended_plus_2d_period_one"
    assert CELL_RECIPES["gefen_muon_tuned3_adamw"].recipe_class == "isolation_ns_schedule"
    assert CELL_RECIPES["gefen_muon_normuon_adamw"].recipe_class == "isolation_normuon"


def test_resolved_lr_and_weight_decay_are_explicit_and_tunable():
    config = CellBuildConfig(lr=2e-3, weight_decay=0.03)
    literal = resolve_cell("gefen_hybrid_literal", config)
    recommended = resolve_cell("gefen_hybrid_recommended", config)
    recommended_2d = resolve_cell("gefen_hybrid_recommended_2d", config)
    classic = resolve_cell("gefen_muon_classic_adamw", config)
    tuned3 = resolve_cell("gefen_muon_tuned3_adamw", config)
    normuon = resolve_cell("gefen_muon_normuon_adamw", config)

    assert literal["backup_lr"] == pytest.approx(2e-3)
    assert recommended["backup_lr"] == pytest.approx(1e-3)
    assert recommended_2d["backup_lr"] == pytest.approx(1e-3)
    assert recommended["muon_weight_decay"] == pytest.approx(0.03)
    assert recommended["backup_weight_decay"] == pytest.approx(0.03)
    assert classic["ns_schedule"] == "standard"
    assert classic["ns_steps"] == 5
    assert classic["normuon"] is False
    assert classic["adjust_lr_fn"] == "match_rms_adamw"
    assert (classic["ns_schedule"], classic["normuon"], classic["backup_lr"]) == (
        "standard",
        False,
        pytest.approx(2e-3),
    )
    assert (tuned3["ns_schedule"], tuned3["normuon"], tuned3["backup_lr"]) == (
        "tuned3",
        False,
        pytest.approx(2e-3),
    )
    assert (normuon["ns_schedule"], normuon["normuon"], normuon["backup_lr"]) == (
        "tuned3",
        True,
        pytest.approx(2e-3),
    )
    assert resolve_cell("torch_muon_adamw", config)["adjust_lr_fn"] == "match_rms_adamw"

    overridden = resolve_cell(
        "gefen_hybrid_recommended",
        CellBuildConfig(
            lr=2e-3,
            muon_lr=3e-3,
            backup_lr=7e-4,
            weight_decay=0.03,
            muon_weight_decay=0.04,
            backup_weight_decay=0.01,
            adjust_lr_fn="original",
        ),
    )
    assert overridden["primary_lr"] == pytest.approx(3e-3)
    assert overridden["backup_lr"] == pytest.approx(7e-4)
    assert overridden["muon_weight_decay"] == pytest.approx(0.04)
    assert overridden["backup_weight_decay"] == pytest.approx(0.01)
    assert overridden["adjust_lr_fn"] == "original"


def test_auxiliary_only_isolation_edges_hold_hidden_recipe_fixed():
    config = CellBuildConfig(lr=2e-3, weight_decay=0.01)
    normuon_adamw = resolve_cell("gefen_muon_normuon_adamw", config)
    literal_hybrid = resolve_cell("gefen_hybrid_literal", config)
    recommended_adamw = resolve_cell("gefen_muon_recommended_adamw", config)
    split_lr_hybrid = resolve_cell("gefen_hybrid_split_lr_only", config)
    invariant_keys = (
        "primary_lr",
        "backup_lr",
        "muon_weight_decay",
        "backup_weight_decay",
        "ns_schedule",
        "adjust_lr_fn",
        "normuon",
        "backup_1d_period_one",
        "backup_2d_period_one",
    )
    assert {key: normuon_adamw[key] for key in invariant_keys} == {
        key: literal_hybrid[key] for key in invariant_keys
    }
    assert {key: recommended_adamw[key] for key in invariant_keys} == {
        key: split_lr_hybrid[key] for key in invariant_keys
    }
    assert normuon_adamw["auxiliary"] == "torch.optim.AdamW"
    assert literal_hybrid["auxiliary"] == "gefen.Gefen"


def test_documented_screening_and_pretraining_preset_counts():
    screen = preset_config("screen_33m", max_seq_len=1024)
    pretrain = preset_config("pretrain_134m", max_seq_len=1024)
    assert expected_parameter_count(screen) == 33_169_408
    assert expected_parameter_count(pretrain) == 134_216_576
    with torch.device("meta"):
        screen_model = TinyQwenForCausalLM(screen)
        pretrain_model = TinyQwenForCausalLM(pretrain)
    assert sum(parameter.numel() for parameter in screen_model.parameters()) == 33_169_408
    assert sum(parameter.numel() for parameter in pretrain_model.parameters()) == 134_216_576


def test_training_batch_metadata_counts_optimizer_updates_not_microsteps():
    metadata = training_batch_metadata(
        micro_batch_size=2,
        gradient_accumulation_steps=64,
        sequence_length=1024,
        optimizer_updates=500,
    )
    assert metadata == {
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 64,
        "effective_batch_size": 128,
        "tokens_per_optimizer_update": 131_072,
        "optimizer_updates": 500,
    }


def test_throughput_metadata_is_shared_and_counts_post_warmup_updates():
    assert throughput_measurement_metadata(
        measured_tokens=12_288,
        measured_seconds=3.0,
        optimizer_updates=20,
        warmup_updates=5,
    ) == {
        "training_step_tokens_per_second": 4096.0,
        "throughput_measured_updates": 15,
    }
    assert throughput_measurement_metadata(
        measured_tokens=0,
        measured_seconds=0.0,
        optimizer_updates=5,
        warmup_updates=5,
    ) == {
        "training_step_tokens_per_second": None,
        "throughput_measured_updates": 0,
    }


def test_save_optimizer_help_is_explicitly_archival_only(capsys):
    with pytest.raises(SystemExit, match="0"):
        parse_args(["--help"])
    rendered = " ".join(capsys.readouterr().out.split())
    assert "archive optimizer state for external inspection only" in rendered
    assert "does not restore optimizer/scheduler/update state for training resume" in rendered


def test_token_weighted_accumulation_matches_full_batch_with_uneven_masks():
    full_batch_model = _model()
    accumulated_model = copy.deepcopy(full_batch_model)
    ids = torch.stack((torch.arange(8), torch.arange(8, 16))).remainder(256)
    labels = ids.clone()
    labels[0, :2] = -100
    labels[1, :6] = -100

    full_batch_model(ids, labels=labels).loss.backward()
    total_supervised = 0
    for row in range(2):
        output = accumulated_model(ids[row : row + 1], labels=labels[row : row + 1])
        output.loss_sum.backward()
        total_supervised += int((labels[row : row + 1, 1:] != -100).sum().item())
    normalize_gradients(accumulated_model.parameters(), total_supervised)

    for reference, accumulated in zip(full_batch_model.parameters(), accumulated_model.parameters()):
        assert torch.allclose(reference.grad, accumulated.grad, atol=2e-6, rtol=2e-5)


def test_tiny_qwen_has_qwen_style_split_and_tied_embedding_backup():
    model = _model()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    muon, backup = split_params_for_muon(model)
    muon_names = {name for name, _ in muon}
    backup_names = {name for name, _ in backup}
    assert "model.layers.0.self_attn.q_proj.weight" in muon_names
    assert "model.layers.0.mlp.down_proj.weight" in muon_names
    assert "model.embed_tokens.weight" in backup_names
    assert "model.norm.weight" in backup_names


@pytest.mark.parametrize("cell", ALL_CELLS)
def test_every_available_cell_takes_a_real_cpu_step(cell):
    if cell == "torch_muon_adamw" and not hasattr(torch.optim, "Muon"):
        pytest.skip("torch.optim.Muon requires PyTorch 2.9+")
    if cell.startswith("gefen") and importlib.util.find_spec("numba") is None:
        pytest.skip("CPU Gefen step needs numba; the package declares it as a runtime dependency")
    model = _model()
    optimizer, resolved = build_optimizer(
        model,
        cell,
        CellBuildConfig(lr=1e-3, weight_decay=0.01, fused=False),
    )
    ids = torch.arange(8).remainder(256)[None]
    loss = model(ids, labels=ids).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert resolved["name"] == cell

    if cell == "torch_muon_adamw":
        assert isinstance(optimizer, OptimizerPair)
        assert isinstance(optimizer.primary, torch.optim.Muon)
        assert isinstance(optimizer.auxiliary, torch.optim.AdamW)
        assert optimizer.primary.param_groups[0]["eps"] == resolved["muon_eps"]
        assert optimizer.auxiliary.param_groups[0]["eps"] == resolved["backup_eps"]
        assert resolved["primary_fused"] is None
        assert optimizer.auxiliary.param_groups[0]["fused"] is resolved["auxiliary_fused"]
    elif cell.startswith("gefen_muon_") and cell.endswith("_adamw"):
        assert isinstance(optimizer, OptimizerPair)
        assert isinstance(optimizer.primary, GefenMuon)
        assert isinstance(optimizer.auxiliary, torch.optim.AdamW)
        assert optimizer.primary.param_groups[0]["eps"] == resolved["muon_eps"]
        assert optimizer.auxiliary.param_groups[0]["eps"] == resolved["backup_eps"]
        assert optimizer.primary.fused is resolved["primary_fused"]
        assert optimizer.auxiliary.param_groups[0]["fused"] is resolved["auxiliary_fused"]
    elif cell.startswith("gefen_hybrid"):
        assert isinstance(optimizer, GefenMuonHybrid)
        assert isinstance(optimizer.backup, Gefen)
        assert optimizer.muon.param_groups[0]["eps"] == resolved["muon_eps"]
        assert optimizer.backup.param_groups[0]["eps"] == resolved["backup_eps"]
        assert optimizer.muon.fused is resolved["primary_fused"]
        assert optimizer.backup.fused is resolved["auxiliary_fused"]
    else:
        assert optimizer.param_groups[0]["eps"] == resolved["backup_eps"]
        assert optimizer.param_groups[0]["fused"] is resolved["primary_fused"]
    if resolved["ns_schedule"] == "tuned3":
        primary = optimizer.muon if isinstance(optimizer, GefenMuonHybrid) else optimizer.primary
        assert resolved["ns_steps"] == 3
        assert primary.param_groups[0]["ns_steps"] == 3


def test_warmup_cosine_preserves_recommended_split_lr_ratio():
    optimizer, _ = build_optimizer(
        _model(),
        "gefen_hybrid_recommended",
        CellBuildConfig(lr=2e-3, weight_decay=0.0, fused=False),
    )
    schedule = GroupLRSchedule(
        optimizer,
        schedule="warmup_cosine",
        total_steps=4,
        warmup_steps=1,
        min_lr_ratio=0.1,
    )
    assert schedule.base_lrs == pytest.approx([2e-3, 1e-3])
    for step in range(4):
        lrs = schedule.apply(step)
        assert lrs[0] / lrs[1] == pytest.approx(2.0)
    assert lrs == pytest.approx([2e-4, 1e-4])


def test_dataset_seed_fixes_validation_and_training_order():
    kwargs = dict(
        phase="pretrain",
        source="synthetic",
        sequence_length=16,
        validation_blocks=8,
        updates=10,
        batch_size=2,
    )
    first = build_dataset(seed=3, **kwargs)
    repeat = build_dataset(seed=3, **kwargs)
    changed = build_dataset(seed=4, **kwargs)
    assert first.data_sha256 == repeat.data_sha256
    assert first.order_sha256 == repeat.order_sha256
    assert first.order == repeat.order
    assert (first.data_sha256, first.order_sha256) != (
        changed.data_sha256,
        changed.order_sha256,
    )
    assert set(first.training_source_ids).isdisjoint(first.validation_source_ids)
    assert set(first.training_source_hashes).isdisjoint(first.validation_source_hashes)


TINY_SHAKESPEARE_CACHE = (
    Path.home() / ".cache/huggingface/datasets/winglian___tiny-shakespeare"
)


@pytest.mark.skipif(
    not TINY_SHAKESPEARE_CACHE.exists(),
    reason="cached Tiny Shakespeare is an optional local integration asset",
)
def test_cached_tiny_shakespeare_hash_groups_are_disjoint():
    bundle = build_dataset(
        phase="pretrain",
        source="hf",
        hf_dataset="winglian/tiny-shakespeare",
        hf_split="train",
        text_column="text",
        cache_only=True,
        sequence_length=1024,
        validation_blocks=256,
        updates=1,
        batch_size=1,
        seed=0,
    )
    assert len(bundle.validation_blocks) == 256
    assert set(bundle.training_source_ids).isdisjoint(bundle.validation_source_ids)
    assert set(bundle.training_source_hashes).isdisjoint(bundle.validation_source_hashes)
    assert bundle.source_metadata["original_row_count"] == 472
    assert bundle.source_metadata["unitization_method"] == "whole_row_content_hash_groups"


def test_single_text_document_uses_exact_contiguous_byte_ranges(tmp_path):
    text = "".join(chr(65 + (index % 26)) for index in range(4096))
    path = tmp_path / "single.txt"
    path.write_text(text, encoding="utf-8")
    bundle = build_dataset(
        phase="pretrain",
        source="text",
        text_file=str(path),
        sequence_length=64,
        validation_blocks=4,
        updates=2,
        batch_size=1,
        seed=9,
    )
    metadata = bundle.source_metadata
    assert metadata["unitization_method"] == "single_document_contiguous_byte_ranges"
    assert metadata["validation_byte_range"] == [0, 256]
    assert metadata["training_byte_range"] == [256, 4096]
    assert len(bundle.validation_blocks) == 4
    assert len(bundle.train_blocks) == 60
    validation_bytes = bytes(bundle.validation_blocks.rows.flatten().tolist())
    training_bytes = bytes(bundle.train_blocks.rows.flatten().tolist())
    assert validation_bytes == text.encode()[:256]
    assert training_bytes == text.encode()[256:4096]


WIKITEXT_DOCUMENT_CACHE = (
    Path.home() / ".cache/huggingface/datasets/EleutherAI___wikitext_document_level"
)


@pytest.mark.skipif(
    not WIKITEXT_DOCUMENT_CACHE.exists(),
    reason="pinned document-level WikiText is an optional local integration asset",
)
def test_pinned_wikitext_official_split_provenance():
    bundle = build_dataset(
        phase="pretrain",
        source="hf",
        hf_dataset="EleutherAI/wikitext_document_level",
        hf_config="wikitext-103-raw-v1",
        hf_revision="647234772b9554e208af6c826f23b99e3cac88c8",
        hf_split="train",
        hf_validation_split="validation",
        text_column="page",
        cache_only=True,
        max_train_blocks=8,
        sequence_length=1024,
        validation_blocks=4,
        updates=1,
        batch_size=1,
        seed=0,
    )
    metadata = bundle.source_metadata
    assert metadata["hf_train_fingerprint"] == "57fbbccacc214a80"
    assert metadata["hf_validation_fingerprint"] == "db990d651d8c2e83"
    assert metadata["train_rows_total"] == 29_444
    assert metadata["validation_rows_total"] == 60
    assert metadata["training_source_ids_sha256"]
    assert metadata["training_source_hashes_sha256"]
    assert bundle.training_source_ids[0] == "train:row0"
    assert bundle.validation_source_ids[0] == "validation:row0"


def test_sft_splits_whole_examples_before_packing():
    bundle = build_dataset(
        phase="sft",
        source="synthetic",
        sequence_length=32,
        validation_blocks=16,
        updates=4,
        batch_size=1,
        seed=11,
    )
    assert set(bundle.training_source_ids).isdisjoint(bundle.validation_source_ids)
    assert set(bundle.training_source_hashes).isdisjoint(bundle.validation_source_hashes)
    assert bundle.source_metadata["validation_source_records"] < bundle.source_metadata["source_records"]


def test_accumulation_helpers_do_not_materialize_device_scalars_per_microstep(monkeypatch):
    ids = torch.stack((torch.arange(8), torch.arange(8, 16))).remainder(256)
    labels = ids.clone()
    labels[0, :2] = -100
    labels[1, :6] = -100
    microbatches = [
        (
            ids[row : row + 1],
            labels[row : row + 1],
            int((labels[row : row + 1, 1:] != -100).sum().item()),
        )
        for row in range(2)
    ]

    tiny = _model()

    class DummyHF(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(256, 8)
            self.head = torch.nn.Linear(8, 256)

        def forward(self, input_ids, labels):
            logits = self.head(self.embedding(input_ids))[:, :-1]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 256),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            return SimpleNamespace(loss=loss)

    hf = DummyHF()

    def forbidden_item(*_args, **_kwargs):
        raise AssertionError("Tensor.item() inside accumulation helper")

    monkeypatch.setattr(torch.Tensor, "item", forbidden_item)
    accumulate_tiny_microbatches(tiny, microbatches, torch.device("cpu"))
    accumulate_hf_microbatches(hf, microbatches, torch.device("cpu"))


def test_local_model_manifest_streams_full_same_size_and_nested_contents(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "local-model"
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weight-A")
    tokenizer = tokenizer_dir / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer-A")

    def forbidden_read_bytes(*_args, **_kwargs):
        raise AssertionError("local manifests must stream files instead of read_bytes()")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    original = _local_model_manifest_sha256(str(model_dir))

    shard.write_bytes(b"weight-B")  # Same filename and byte count, different weights.
    changed_weights = _local_model_manifest_sha256(str(model_dir))
    assert changed_weights != original

    tokenizer.write_bytes(b"tokenizer-B")  # Nested tokenizer content is covered too.
    changed_tokenizer = _local_model_manifest_sha256(str(model_dir))
    assert changed_tokenizer not in {original, changed_weights}
    assert _local_model_manifest_sha256(str(tmp_path / "missing")) is None


def test_nonfinite_loss_and_gradients_are_rejected_before_optimizer_mutation():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match="non-finite accumulated training loss"):
        materialize_finite_update_loss(torch.tensor(float("inf")), 1, torch.device("cpu"))
    assert torch.equal(parameter, before)
    assert not optimizer.state

    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_and_clip_gradients([parameter], 0.0)
    assert torch.equal(parameter, before)
    assert not optimizer.state


def test_peak_memory_is_snapshotted_before_state_serialization(monkeypatch):
    events = []

    class FakeOptimizer:
        def state_dict(self):
            events.append("state_dict")
            return {"state": {0: {"moment": torch.ones(3)}}}

    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda _device: events.append("allocated") or 123,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda _device: events.append("reserved") or 456,
    )
    allocated, reserved, state_bytes = snapshot_peak_then_measure_serialized_state(
        FakeOptimizer(), torch.device("cuda")
    )
    assert (allocated, reserved, state_bytes) == (123, 456, 12)
    assert events == ["allocated", "reserved", "state_dict"]


def test_pretrain_checkpoint_hands_weights_and_lineage_to_sft(tmp_path):
    checkpoint = tmp_path / "pretrain.pt"
    shared = [
        "--cell",
        "adamw",
        "--source",
        "synthetic",
        "--steps",
        "1",
        "--batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--seq-len",
        "16",
        "--validation-blocks",
        "1",
        "--eval-every",
        "1",
        "--tail-evals",
        "1",
        "--throughput-warmup",
        "0",
        "--schedule",
        "constant",
        "--device",
        "cpu",
        "--dtype",
        "float32",
        "--no-fused",
        "--hidden-size",
        "16",
        "--intermediate-size",
        "32",
        "--layers",
        "1",
        "--heads",
        "2",
        "--kv-heads",
        "1",
    ]
    pretrain = run(parse_args([*shared, "--phase", "pretrain", "--checkpoint-out", str(checkpoint)]))
    sft = run(
        parse_args(
            [
                *shared,
                "--phase",
                "sft",
                "--init-checkpoint",
                str(checkpoint),
            ]
        )
    )
    assert pretrain["checkpoint_out"] == str(checkpoint.resolve())
    assert sft["initialization"]["metadata"]["phase"] == "pretrain"
    assert sft["initialization"]["metadata"]["cell"] == "adamw"
    assert sft["model"]["parameter_count"] == pretrain["model"]["parameter_count"]
    assert sft["training_batch"]["gradient_accumulation_steps"] == 1
    assert sft["tail_eval_count"] == 1
    assert sft["tail_eval_mean"] == sft["final_eval_loss"]
    assert sft["evaluation"][0]["step"] == 0
    assert sft["comparison_id"] == comparison_key(sft)[1]
    json.dumps(pretrain)
    json.dumps(sft)


def test_mixed_tiny_and_hf_result_schemas_summarize_without_crashing():
    def row(run_name, cell, data_hash, parameters, throughput, model_extra):
        return {
            "run_name": run_name,
            "cell": cell,
            "phase": "sft",
            "seed": 0,
            "data": {"data_sha256": data_hash},
            "model": {"parameter_count": parameters, **model_extra},
            "schedule": {"name": "constant", "total_steps": 10},
            "initialization": None,
            "tail_eval_mean": 1.25,
            "final_eval_loss": 1.24,
            "throughput_tokens_per_second": throughput,
            "optimizer_state_bytes_per_parameter": 1.5,
            "runtime": {"device_name": "NVIDIA GeForce RTX 3090"},
        }

    rows = [
        row("tiny-adamw", "adamw", "tiny", 1000, 100.0, {"preset": "custom"}),
        row("tiny-gefen", "gefen_hybrid_recommended", "tiny", 1000, 125.0, {"preset": "custom"}),
        row("hf-adamw", "adamw", "hf", 600_000_000, 20.0, {"id": "Qwen/Qwen3-0.6B"}),
        row("hf-gefen", "gefen_hybrid_recommended", "hf", 600_000_000, 22.0, {"id": "Qwen/Qwen3-0.6B"}),
    ]
    rendered = render_markdown(rows)
    assert "tiny-gefen" in rendered and "1.250x" in rendered
    assert "hf-gefen" in rendered and "1.100x" in rendered


def test_strict_fallback_comparison_rejects_every_changed_invariant():
    baseline = {
        "format": "legacy",
        "run_name": "baseline",
        "cell": "adamw",
        "phase": "sft",
        "seed": 0,
        "model": {"id": "model-a", "parameter_count": 10},
        "data": {
            "data_sha256": "data-a",
            "order_sha256": "order-a",
            "source_hash": "source-a",
        },
        "schedule": {
            "name": "constant",
            "total_steps": 10,
            "warmup_steps": 0,
            "min_lr_ratio": 0.1,
            "base_lrs": [1e-3],
        },
        "training_batch": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "effective_batch_size": 2,
            "tokens_per_optimizer_update": 32,
            "optimizer_updates": 10,
        },
        "optimizer": {
            "lr": 1e-3,
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "momentum": 0.95,
            "nesterov": True,
            "fused": True,
        },
        "initialization": {
            "checkpoint_sha256": "checkpoint-a",
            "metadata": {"cell": "adamw"},
        },
        "tail_eval_mean": 1.0,
        "final_eval_loss": 1.0,
        "throughput_tokens_per_second": 100.0,
        "optimizer_state_bytes_per_parameter": 8.0,
        "runtime": {
            "device_name": "NVIDIA GeForce RTX 3090",
            "cuda_visible_devices": "GPU-a",
            "torch_version": "2.12.0",
            "torch_cuda_version": "13.3",
            "deterministic_kernels": False,
            "git": {"commit": "commit-a", "dirty": False},
        },
    }
    matched = copy.deepcopy(baseline)
    matched.update(
        run_name="matched",
        cell="gefen_hybrid_recommended",
        throughput_tokens_per_second=125.0,
        optimizer_state_bytes_per_parameter=1.2,
    )
    mutations = (
        ("seed", lambda row: row.__setitem__("seed", 1)),
        ("model", lambda row: row["model"].__setitem__("id", "model-b")),
        ("data", lambda row: row["data"].__setitem__("data_sha256", "data-b")),
        ("order", lambda row: row["data"].__setitem__("order_sha256", "order-b")),
        ("lr", lambda row: row["optimizer"].__setitem__("lr", 2e-3)),
        ("wd", lambda row: row["optimizer"].__setitem__("weight_decay", 0.1)),
        ("schedule", lambda row: row["schedule"]["base_lrs"].__setitem__(0, 2e-3)),
        (
            "accum",
            lambda row: row["training_batch"].__setitem__("gradient_accumulation_steps", 4),
        ),
        ("uuid", lambda row: row["runtime"].__setitem__("cuda_visible_devices", "GPU-b")),
        ("git", lambda row: row["runtime"]["git"].__setitem__("commit", "commit-b")),
        (
            "checkpoint",
            lambda row: row["initialization"].__setitem__("checkpoint_sha256", "checkpoint-b"),
        ),
    )
    unmatched = []
    for name, mutate in mutations:
        row = copy.deepcopy(matched)
        row["run_name"] = f"unmatched-{name}"
        mutate(row)
        unmatched.append(row)
    rendered = render_markdown([baseline, matched, *unmatched])
    lines = {line.split("|")[1].strip(): line for line in rendered.splitlines()[2:]}
    assert "1.250x" in lines["matched"]
    for name, _ in mutations:
        assert "—" in lines[f"unmatched-{name}"]

    duplicate = copy.deepcopy(baseline)
    duplicate["run_name"] = "duplicate"
    with pytest.raises(ValueError, match="duplicate adamw baselines"):
        render_markdown([baseline, duplicate, matched])


def test_explicit_comparison_id_is_content_validated():
    context = {"backend": "tiny", "seed": 0, "device": "GPU-a"}
    baseline = {"cell": "adamw"}
    attach_comparison(baseline, context)
    assert comparison_key(baseline) == ("explicit", baseline["comparison_id"])
    baseline["comparison_context"]["seed"] = 1
    with pytest.raises(ValueError, match="stale/tampered"):
        comparison_key(baseline)


def test_matrix_default_remains_core_cells_and_all_is_explicit():
    default_commands = matrix_commands(parse_matrix_args([]))
    default_names = [command[command.index("--cell") + 1] for command in default_commands]
    assert tuple(default_names) == CORE_CELLS
    all_commands = matrix_commands(parse_matrix_args(["--cells", "all"]))
    all_names = [command[command.index("--cell") + 1] for command in all_commands]
    assert tuple(all_names) == ALL_CELLS


def test_generated_outputs_do_not_change_source_fingerprint():
    output_dir = ROOT / "benchmarks" / "training_matrix" / "out" / "fingerprint-test"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "results.jsonl"
    before = source_fingerprint(ROOT)
    artifact.write_text('{"generated": true}\n', encoding="utf-8")
    after = source_fingerprint(ROOT)
    artifact.unlink()
    output_dir.rmdir()
    assert after == before


def test_source_change_guard_rejects_a_new_revision(monkeypatch):
    captured = {"commit": "abc", "diff_sha256": "clean", "dirty": False}
    state = {"current": captured.copy()}
    monkeypatch.setattr(
        comparison_module,
        "source_fingerprint",
        lambda _root: state["current"].copy(),
    )
    comparison_module.require_unchanged_source(ROOT, captured)

    state["current"] = {"commit": "abc", "diff_sha256": "edited", "dirty": True}
    with pytest.raises(RuntimeError, match="refusing to launch the next cell"):
        comparison_module.require_unchanged_source(ROOT, captured)


def test_matrix_launcher_preserves_capture_and_guards_before_second_cell(
    tmp_path, monkeypatch
):
    captured = {"commit": "abc", "diff_sha256": "diff", "dirty": True}
    launches = []
    checks = []
    monkeypatch.setattr(matrix_driver, "source_fingerprint", lambda _root: captured.copy())

    def reject_next(_root, expected):
        checks.append(expected)
        raise RuntimeError("source changed")

    def fake_run(command, *, cwd, env, check):
        launches.append((command, cwd, env.copy(), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix_driver, "require_unchanged_source", reject_next)
    monkeypatch.setattr(matrix_driver.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="source changed"):
        matrix_driver.main(
            [
                "--cells",
                "adamw,torch_muon_adamw",
                "--output-dir",
                str(tmp_path),
                "--execute",
            ]
        )
    assert len(launches) == 1
    assert checks == [captured]
    assert launches[0][2][SOURCE_FINGERPRINT_ENV] == canonical_json(captured)


def test_consistency_launcher_preserves_capture_and_guards_before_second_job(
    tmp_path, monkeypatch
):
    config = {
        "pretrain_cells": ["adamw", "gefen_hybrid_recommended"],
        "sft_cells": ["adamw", "gefen_hybrid_recommended"],
        "pretrain": {"source": "synthetic", "steps": 1},
        "sft": {"source": "synthetic", "steps": 1},
    }
    config_path = tmp_path / "consistency.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    captured = {"commit": "def", "diff_sha256": "diff", "dirty": True}
    launches = []
    checks = []
    monkeypatch.setattr(
        consistency_driver,
        "source_fingerprint",
        lambda _root: captured.copy(),
    )

    def reject_next(_root, expected):
        checks.append(expected)
        raise RuntimeError("source changed")

    def fake_run(command, *, cwd, env, check):
        launches.append((command, cwd, env.copy(), check))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(consistency_driver, "require_unchanged_source", reject_next)
    monkeypatch.setattr(consistency_driver.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="source changed"):
        consistency_driver.main(
            [
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--execute",
            ]
        )
    assert len(launches) == 1
    assert checks == [captured]
    assert launches[0][2][SOURCE_FINGERPRINT_ENV] == canonical_json(captured)


def test_consistency_plan_is_two_pretrains_crossed_with_four_sft_runs(tmp_path):
    config = {
        "pretrain_cells": ["adamw", "gefen_hybrid_recommended"],
        "sft_cells": ["adamw", "gefen_hybrid_recommended"],
        "shared": {"seed": 0, "seq_len": 16},
        "pretrain": {"source": "synthetic", "steps": 2},
        "sft": {"source": "synthetic", "steps": 1},
    }
    plan = build_plan(config, "python", tmp_path)
    assert len(plan) == 6
    rendered = [" ".join(command) for command in plan]
    assert sum(" --phase pretrain " in f" {command} " for command in rendered) == 2
    assert sum(" --phase sft " in f" {command} " for command in rendered) == 4
    for pretrain_cell in ("adamw", "gefen_hybrid_recommended"):
        checkpoint = str(tmp_path / "checkpoints" / f"pretrain__{pretrain_cell}.pt")
        assert sum(checkpoint in command for command in rendered) == 3
