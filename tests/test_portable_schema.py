"""Strict portable-state v3 envelope coverage."""

import copy
import io

import pytest
import torch

import gefen
import gefen.portable_schema as portable_schema_module
from gefen.portable_schema import (
    PORTABLE_STATE_COVERAGE,
    PORTABLE_STATE_DIGEST_ALGORITHM,
    PORTABLE_STATE_FORMAT,
    PORTABLE_STATE_FORMAT_VERSION,
    build_portable_state_document,
    normalize_portable_state_document,
    portable_state_digest,
)


def _parameter_record(*, tensor=None):
    if tensor is None:
        tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return {
        "identity": {
            "schema_version": 1,
            "fqn": "Model.Weight",
            "global_shape": [2, 3],
        },
        "algorithm_options": {"lr": 1.0e-3, "betas": (0.9, 0.999)},
        "state_variant": "quantized_momentum",
        "state": {"step": 3, "momentum": tensor},
        "projection_hints": {"source_periods": [3]},
    }


def _document(*, tensor=None):
    return build_portable_state_document(
        implementation="gefen.Gefen",
        policy={"factored_v_2d": False},
        common={
            "gefen_global_step": 3,
            "gefen_deterministic": False,
        },
        parameters={"Model.Weight": _parameter_record(tensor=tensor)},
        provenance={"source_layouts": ["replicated"]},
    )


def test_portable_document_is_complete_device_neutral_and_weights_only_safe():
    backing = torch.arange(30, dtype=torch.float32)
    view = backing[5:17:2].reshape(2, 3)
    document = _document(tensor=view)

    assert document["format"] == PORTABLE_STATE_FORMAT
    assert document["format_version"] == PORTABLE_STATE_FORMAT_VERSION
    assert document["coverage"] == PORTABLE_STATE_COVERAGE
    assert document["completion"]["status"] == "complete"
    assert (
        document["completion"]["digest_algorithm"]
        == PORTABLE_STATE_DIGEST_ALGORITHM
    )
    momentum = document["parameters"]["Model.Weight"]["state"]["momentum"]
    assert momentum.device.type == "cpu"
    assert momentum.is_contiguous()
    assert momentum.storage_offset() == 0
    assert momentum.untyped_storage().nbytes() == momentum.numel() * momentum.element_size()
    assert torch.equal(momentum, view)
    assert momentum is not view

    buffer = io.BytesIO()
    torch.save(document, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=True)
    normalized = normalize_portable_state_document(loaded)
    assert normalized["completion"] == document["completion"]


def test_portable_digest_is_deterministic_and_type_shape_dtype_sensitive():
    left = {"b": [1, 2], "a": torch.tensor([1.0, 2.0])}
    right = {"a": torch.tensor([1.0, 2.0]), "b": [1, 2]}
    baseline = portable_state_digest(left)

    assert baseline == portable_state_digest(right)
    assert baseline != portable_state_digest({"a": torch.tensor([[1.0, 2.0]]), "b": [1, 2]})
    assert baseline != portable_state_digest({"a": torch.tensor([1.0, 2.0], dtype=torch.float64), "b": [1, 2]})
    assert baseline != portable_state_digest({"a": torch.tensor([1.0, 3.0]), "b": [1, 2]})
    assert baseline != portable_state_digest({"a": torch.tensor([1.0, 2.0]), "b": (1, 2)})


def test_portable_digest_streams_tensor_bytes_without_changing_the_digest(monkeypatch):
    payload = {"tensor": torch.arange(257, dtype=torch.float32)}
    baseline = portable_state_digest(payload)

    monkeypatch.setattr(portable_schema_module, "_PORTABLE_DIGEST_CHUNK_BYTES", 17)

    assert portable_state_digest(payload) == baseline


def test_portable_digest_v3_grammar_has_a_cross_version_golden_vector():
    payload = {
        "a": None,
        "b": True,
        "c": -12345678901234567890,
        "d": -0.0,
        "e": "x\u2603",
        "f": [1, (2, 3)],
        "g": torch.tensor([[1, -2], [3, -4]], dtype=torch.int16),
        "h": torch.tensor([1.5, -2.25], dtype=torch.bfloat16),
        "i": torch.tensor(3 + 4j, dtype=torch.complex64),
        "j": torch.empty((0, 2), dtype=torch.float64),
    }

    assert portable_state_digest(payload) == (
        "348e09a77f1b3eae286adda1573722df9"
        "38be6d8100631b272665f8c22c6e23a"
    )


def test_portable_clone_streams_noncontiguous_values_and_finite_checks(monkeypatch):
    source = torch.arange(514, dtype=torch.float32)[1::2]
    assert not source.is_contiguous()
    calls = []
    original = portable_schema_module._read_portable_tensor_chunk

    def tracked(value, start, stop):
        calls.append((start, stop))
        return original(value, start, stop)

    monkeypatch.setattr(portable_schema_module, "_PORTABLE_CLONE_CHUNK_BYTES", 17)
    monkeypatch.setattr(portable_schema_module, "_read_portable_tensor_chunk", tracked)

    cloned = portable_schema_module._clone_portable_value(source, path="tensor")

    assert torch.equal(cloned, source)
    assert cloned.is_contiguous()
    assert cloned.storage_offset() == 0
    assert cloned.untyped_storage().nbytes() == cloned.numel() * cloned.element_size()
    assert len(calls) > 1
    assert max(stop - start for start, stop in calls) <= 4


def test_builder_and_normalizer_each_hash_the_tensor_tree_once(monkeypatch):
    calls = []
    original = portable_schema_module._canonical_portable_state_digest

    def tracked(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(portable_schema_module, "_canonical_portable_state_digest", tracked)

    document = _document()
    assert len(calls) == 1
    normalize_portable_state_document(document)
    assert len(calls) == 2


def test_multibyte_tensor_digest_is_canonical_little_endian(monkeypatch):
    little_endian_values = torch.tensor([0x0102, 0x0304], dtype=torch.int16)
    simulated_big_endian_storage = torch.tensor([0x0201, 0x0403], dtype=torch.int16)
    baseline = portable_state_digest(little_endian_values)

    monkeypatch.setattr(portable_schema_module, "_PORTABLE_NATIVE_BYTEORDER", "big")

    assert portable_state_digest(simulated_big_endian_storage) == baseline


@pytest.mark.parametrize(
    "corruption",
    (
        "payload",
        "digest",
        "status",
        "algorithm",
        "missing_top",
        "bool_version",
        "coverage",
        "implementation",
        "fqn",
        "shape_type",
        "state_variant",
        "state_type",
        "provenance",
    ),
)
def test_portable_schema_and_completion_corruption_are_rejected(corruption):
    document = _document()
    damaged = copy.deepcopy(document)
    if corruption == "payload":
        damaged["common"]["gefen_global_step"] = 4
    elif corruption == "digest":
        damaged["completion"]["digest"] = "0" * 64
    elif corruption == "status":
        damaged["completion"]["status"] = "preparing"
    elif corruption == "algorithm":
        damaged["completion"]["digest_algorithm"] = "md5"
    elif corruption == "missing_top":
        damaged.pop("policy")
    elif corruption == "bool_version":
        damaged["format_version"] = True
    elif corruption == "coverage":
        damaged["coverage"] = "local_optimizer_fragment"
    elif corruption == "implementation":
        damaged["implementation"] = ""
    elif corruption == "fqn":
        damaged["parameters"]["Model.Weight"]["identity"]["fqn"] = "Other.Weight"
    elif corruption == "shape_type":
        damaged["parameters"]["Model.Weight"]["identity"]["global_shape"] = (2, 3)
    elif corruption == "state_variant":
        damaged["parameters"]["Model.Weight"]["state_variant"] = ""
    elif corruption == "state_type":
        damaged["parameters"]["Model.Weight"]["state"] = []
    else:
        damaged["provenance"] = []

    with pytest.raises((TypeError, ValueError)):
        normalize_portable_state_document(damaged)


def test_expected_implementation_is_checked_after_digest_validation():
    document = _document()

    with pytest.raises(ValueError, match="does not match the target"):
        normalize_portable_state_document(
            document, expected_implementation="gefen.GefenMuon"
        )

    document["completion"]["digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest does not match"):
        normalize_portable_state_document(
            document, expected_implementation="gefen.GefenMuon"
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        torch.tensor([float("nan")]),
        torch.tensor([float("inf")]),
        lambda: None,
    ),
)
def test_builder_rejects_nonportable_values(bad_value):
    record = _parameter_record()
    record["state"]["bad"] = bad_value

    with pytest.raises((TypeError, ValueError)):
        build_portable_state_document(
            implementation="gefen.Gefen",
            policy={},
            common={},
            parameters={"Model.Weight": record},
        )


def test_builder_and_normalizer_do_not_alias_or_mutate_inputs():
    record = _parameter_record()
    parameters = {"Model.Weight": record}
    source_momentum = record["state"]["momentum"]

    document = build_portable_state_document(
        implementation="gefen.Gefen",
        policy={},
        common={},
        parameters=parameters,
    )
    normalized = normalize_portable_state_document(document)

    assert record["state"]["momentum"] is source_momentum
    assert document is not normalized
    assert document["parameters"] is not parameters
    assert document["parameters"]["Model.Weight"] is not record
    assert document["parameters"]["Model.Weight"]["state"]["momentum"] is not source_momentum
    assert normalized["parameters"]["Model.Weight"]["state"]["momentum"] is not document["parameters"]["Model.Weight"]["state"]["momentum"]


def test_portable_schema_and_checkpoint_binding_exports_are_public():
    assert gefen.PORTABLE_STATE_FORMAT_VERSION == PORTABLE_STATE_FORMAT_VERSION
    assert gefen.build_portable_state_document is build_portable_state_document
    assert gefen.normalize_portable_state_document is normalize_portable_state_document
    assert gefen.portable_state_digest is portable_state_digest
    assert gefen.CheckpointProcessGroupBinding.__module__ == "gefen.checkpoint"
    assert gefen.PortableStateLimits.__module__ == "gefen.portable_state"
    assert gefen.PortableStateProvider.__module__ == "gefen.contracts"
    assert gefen.load_portable_dcp.__module__ == "gefen.portable_dcp"
    assert gefen.save_portable_dcp.__module__ == "gefen.portable_dcp"
    assert gefen.LogicalRegion.__module__ == "gefen.contracts"
