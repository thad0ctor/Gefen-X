"""Strict composite-envelope coverage for Gefen-backed Hybrid portable state."""

import copy
import io

import pytest
import torch

from gefen.portable_hybrid import (
    HYBRID_PORTABLE_STATE_COVERAGE,
    HYBRID_PORTABLE_STATE_FORMAT,
    HYBRID_PORTABLE_STATE_FORMAT_VERSION,
    HYBRID_PORTABLE_STATE_IMPLEMENTATION,
    build_hybrid_portable_state_document,
    normalize_hybrid_portable_state_document,
)
from gefen.portable_schema import build_portable_state_document


def _parameter_record(fqn, tensor):
    return {
        "identity": {
            "schema_version": 1,
            "fqn": fqn,
            "global_shape": list(tensor.shape),
        },
        "algorithm_options": {"period": 1},
        "state_variant": "period_selected",
        "state": {"step": 3, "momentum": tensor},
        "projection_hints": {},
    }


def _child(role, *, step=7, deterministic=True, fqn=None, tensor=None):
    if fqn is None:
        fqn = "Model.Weight" if role == "muon" else "Model.Bias"
    if tensor is None:
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    implementation = "gefen.GefenMuon" if role == "muon" else "gefen.Gefen"
    return build_portable_state_document(
        implementation=implementation,
        policy={"role": role},
        common={
            "gefen_global_step": step,
            "gefen_deterministic": deterministic,
        },
        parameters={fqn: _parameter_record(fqn, tensor)},
        provenance={"source_layouts": ["replicated"]},
    )


def _document(*, muon=None, backup=None):
    if muon is None:
        muon = _child("muon")
    if backup is None:
        backup = _child("backup")
    return build_hybrid_portable_state_document(
        backup_optimizer="gefen",
        routing={
            "Model.Weight": "muon",
            "Model.Bias": "backup",
        },
        children={"muon": muon, "backup": backup},
    )


def test_hybrid_portable_document_is_complete_owned_and_weights_only_safe():
    source = torch.arange(30, dtype=torch.float32)[3:15:2].reshape(2, 3)
    muon = _child("muon", tensor=source)
    document = _document(muon=muon)

    assert document["format"] == HYBRID_PORTABLE_STATE_FORMAT
    assert document["format_version"] == HYBRID_PORTABLE_STATE_FORMAT_VERSION
    assert document["coverage"] == HYBRID_PORTABLE_STATE_COVERAGE
    assert document["implementation"] == HYBRID_PORTABLE_STATE_IMPLEMENTATION
    assert document["completion"]["status"] == "complete"
    nested = document["children"]["muon"]["parameters"]["Model.Weight"]["state"]["momentum"]
    assert nested.device.type == "cpu"
    assert nested.is_contiguous()
    assert nested.storage_offset() == 0
    assert nested.untyped_storage().nbytes() == nested.numel() * nested.element_size()
    assert torch.equal(nested, source)
    assert nested is not source

    buffer = io.BytesIO()
    torch.save(document, buffer)
    buffer.seek(0)
    normalized = normalize_hybrid_portable_state_document(torch.load(buffer, weights_only=True))
    assert normalized["completion"] == document["completion"]


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("format", "other", "format"),
        ("format_version", 2, "format_version"),
        ("coverage", "partial", "coverage"),
        ("implementation", "gefen.Gefen", "implementation"),
        ("backup_optimizer", "adamw", "Gefen backup"),
    ],
)
def test_hybrid_portable_document_rejects_wrong_envelope_identity(key, value, match):
    document = _document()
    document[key] = value

    with pytest.raises(ValueError, match=match):
        normalize_hybrid_portable_state_document(document)


def test_hybrid_portable_document_rejects_outer_and_nested_digest_corruption():
    document = _document()
    outer = copy.deepcopy(document)
    outer["routing"]["Model.Bias"] = "muon"
    with pytest.raises(ValueError, match="routing"):
        normalize_hybrid_portable_state_document(outer)

    nested = copy.deepcopy(document)
    nested["children"]["muon"]["parameters"]["Model.Weight"]["state"]["step"] += 1
    with pytest.raises(ValueError, match="digest"):
        normalize_hybrid_portable_state_document(nested)

    completion = copy.deepcopy(document)
    completion["completion"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        normalize_hybrid_portable_state_document(completion)


def test_hybrid_portable_document_rejects_schema_and_routing_mismatches():
    document = _document()
    extra = copy.deepcopy(document)
    extra["extra"] = None
    with pytest.raises(ValueError, match="top-level"):
        normalize_hybrid_portable_state_document(extra)

    missing_role = copy.deepcopy(document)
    del missing_role["children"]["backup"]
    with pytest.raises(ValueError, match="children"):
        normalize_hybrid_portable_state_document(missing_role)

    missing_fqn = copy.deepcopy(document)
    del missing_fqn["routing"]["Model.Bias"]
    with pytest.raises(ValueError, match="routing"):
        normalize_hybrid_portable_state_document(missing_fqn)

    wrong_child = copy.deepcopy(document)
    wrong_child["children"]["muon"] = _child("backup", fqn="Model.Weight")
    with pytest.raises(ValueError, match="implementation"):
        normalize_hybrid_portable_state_document(wrong_child)


@pytest.mark.parametrize(
    ("backup", "match"),
    [
        (_child("backup", step=8), "global steps"),
        (_child("backup", deterministic=False), "deterministic"),
        (_child("backup", fqn="Model.Weight"), "disjoint"),
    ],
)
def test_hybrid_portable_document_rejects_incompatible_children(backup, match):
    with pytest.raises(ValueError, match=match):
        build_hybrid_portable_state_document(
            backup_optimizer="gefen",
            routing={"Model.Weight": "muon", next(iter(backup["parameters"])): "backup"},
            children={"muon": _child("muon"), "backup": backup},
        )


@pytest.mark.parametrize("role", ["muon", "backup"])
def test_hybrid_portable_document_supports_one_present_child(role):
    child = _child(role)
    fqn = next(iter(child["parameters"]))
    children = {"muon": None, "backup": None}
    children[role] = child

    document = build_hybrid_portable_state_document(
        backup_optimizer="gefen",
        routing={fqn: role},
        children=children,
    )

    assert document["children"][role] is not None
    assert document["children"]["backup" if role == "muon" else "muon"] is None


def test_hybrid_portable_document_rejects_no_children_and_noncanonical_routing():
    with pytest.raises(ValueError, match="at least one child"):
        build_hybrid_portable_state_document(
            backup_optimizer="gefen",
            routing={},
            children={"muon": None, "backup": None},
        )
    with pytest.raises(ValueError, match="trimmed FQNs"):
        build_hybrid_portable_state_document(
            backup_optimizer="gefen",
            routing={" Model.Weight": "muon"},
            children={"muon": _child("muon"), "backup": None},
        )
