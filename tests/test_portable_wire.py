import dataclasses
import hashlib
import math
import struct

import pytest
import torch

import gefen.portable_wire as wire
from gefen.portable_wire import (
    _CanonicalWireLimits,
    _CanonicalWirePlan,
    _CanonicalWireTensorSpec,
    _PreparedCanonicalWireValue,
    _parse_canonical_wire_metadata,
    _prepare_canonical_wire_value,
    _reconstruct_canonical_wire_value,
)


def _limits(**changes):
    limits = _CanonicalWireLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=8 << 20,
        max_collective_metadata_bytes=1 << 20,
        max_metadata_bytes=1 << 20,
    )
    return dataclasses.replace(limits, **changes)


def _assert_value_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert type(left) is torch.Tensor
        assert type(right) is torch.Tensor
        assert left.dtype == right.dtype
        assert tuple(left.shape) == tuple(right.shape)
        assert torch.equal(left, right)
        return
    assert type(left) is type(right)
    if type(left) is dict:
        assert list(left) == list(right)
        for key in left:
            _assert_value_equal(left[key], right[key])
    elif type(left) in {list, tuple}:
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_value_equal(left_item, right_item)
    elif type(left) is float and left == 0.0:
        assert math.copysign(1.0, left) == math.copysign(1.0, right)
    else:
        assert left == right


def _round_trip(value, limits=None):
    limits = _limits() if limits is None else limits
    plan = _prepare_canonical_wire_value(value, limits)
    prepared = _parse_canonical_wire_metadata(plan.metadata, limits=limits)
    assert prepared.metadata_digest == plan.metadata_digest
    assert prepared.fragment_digest == plan.fragment_digest
    assert prepared.tensor_specs == plan.tensor_specs
    result = _reconstruct_canonical_wire_value(
        prepared,
        plan.tensors,
        expected_fragment_digest=plan.fragment_digest,
    )
    _assert_value_equal(value, result)
    return plan, prepared, result


def _metadata(root, descriptors=()):
    return (
        b"GFNCV1\0\0"
        + struct.pack(">HHI", 1, 0, 0)
        + root
        + struct.pack(">Q", len(descriptors))
        + b"".join(descriptors)
    )


def _string(value):
    encoded = value.encode("utf-8")
    return b"\x05" + struct.pack(">Q", len(encoded)) + encoded


def _descriptor(dtype_code=1, shape=(), numel=1, nbytes=1, digest=b"\0" * 32, flags=0):
    return (
        struct.pack(">HHI", dtype_code, len(shape), flags)
        + b"".join(struct.pack(">Q", dimension) for dimension in shape)
        + struct.pack(">QQ", numel, nbytes)
        + digest
    )


def test_golden_metadata_tensor_and_fragment_digests():
    value = {"a": -1, "b": torch.tensor([1, -2], dtype=torch.int16)}
    plan = _prepare_canonical_wire_value(value, _limits())
    assert plan.metadata.hex() == (
        "47464e435631000000010000000000000800000000000000020500000000000000016103010000000000000001010500000000000000016209000000000000000000000000000000010004000100000000000000000000000200000000000000020000000000000004"
        "356f76987e72498bcbc4866de355fdcbad68f9a1eb8807f1073728ee463f1e27"
    )
    assert plan.metadata_digest.hex() == "c5894b6b9cc4f0f8591ed4fbb66d943b0bb69362fb460df413c42b36def889d8"
    assert plan.fragment_digest.hex() == "155e9c7c022131d7ea971fd7bc1d69e13c7f9dcbaf89301bb29c135a24de4846"
    _round_trip(value)


def test_round_trip_primitives_nested_containers_and_edge_values():
    value = {
        "bools": [False, True],
        "floats": (0.0, -0.0, float.fromhex("0x0.0000000000001p-1022"), 1.25),
        "integers": [0, 1, -1, (1 << 4095) - 1],
        "none": None,
        "strings": ["", "ASCII", "λ🙂"],
    }
    _round_trip(value)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (torch.bool, [False, True, True]),
        (torch.uint8, [0, 1, 255]),
        (torch.int8, [-128, 0, 127]),
        (torch.int16, [-32768, 0, 32767]),
        (torch.int32, [-(1 << 31), 0, (1 << 31) - 1]),
        (torch.int64, [-(1 << 63), 0, (1 << 63) - 1]),
        (torch.float16, [-1.5, 0.0, 2.25]),
        (torch.bfloat16, [-1.5, 0.0, 2.25]),
        (torch.float32, [-1.5, 0.0, 2.25]),
        (torch.float64, [-1.5, 0.0, 2.25]),
        (torch.complex64, [complex(-1.5, 2.0), 0j, complex(2.25, -3.5)]),
        (torch.complex128, [complex(-1.5, 2.0), 0j, complex(2.25, -3.5)]),
    ],
)
def test_round_trip_fixed_dtype_table(dtype, values):
    plan, _, result = _round_trip(torch.tensor(values, dtype=dtype))
    assert plan.tensor_specs[0].dtype_code == 1 + [record[1] for record in wire._DTYPE_RECORDS].index(dtype)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()


def test_round_trip_noncontiguous_conjugate_scalar_and_empty_tensors():
    noncontiguous = torch.arange(30, dtype=torch.float64).reshape(5, 6).t()[1:5:2]
    conjugate = torch.tensor([1 + 2j, 3 - 4j], dtype=torch.complex128).conj()
    scalar = torch.tensor(-7, dtype=torch.int64)
    empty = torch.empty((3, 0, 2), dtype=torch.float32)
    plan, _, result = _round_trip([noncontiguous, conjugate, scalar, empty])
    assert not noncontiguous.is_contiguous()
    assert conjugate.is_conj()
    assert all(tensor.is_contiguous() for tensor in plan.tensors)
    assert not result[1].is_conj()


def test_tensor_digest_uses_canonical_little_endian_component_bytes():
    value = torch.tensor([0x01020304], dtype=torch.int32)
    plan = _prepare_canonical_wire_value(value, _limits())
    expected = hashlib.sha256()
    expected.update(b"gefen.canonical_wire.tensor.v1\0")
    expected.update(struct.pack(">HHQ", 5, 1, 1))
    expected.update(struct.pack(">Q", 4))
    expected.update(b"\x04\x03\x02\x01")
    assert plan.tensor_specs[0].digest == expected.digest()


def test_fragment_digest_frames_metadata_and_ordered_tensor_digests():
    plan = _prepare_canonical_wire_value(
        [torch.tensor([1], dtype=torch.int16), torch.tensor([2.0], dtype=torch.float64)],
        _limits(),
    )
    expected = hashlib.sha256()
    expected.update(b"gefen.canonical_wire.fragment.v1\0")
    expected.update(hashlib.sha256(plan.metadata).digest())
    expected.update(struct.pack(">Q", 2))
    for spec in plan.tensor_specs:
        expected.update(struct.pack(">Q", spec.nbytes))
        expected.update(spec.digest)
    assert plan.fragment_digest == expected.digest()


def test_dictionary_encoding_is_deterministic_and_sorted():
    left = _prepare_canonical_wire_value({"z": 1, "a": 2}, _limits())
    right = _prepare_canonical_wire_value({"a": 2, "z": 1}, _limits())
    assert left.metadata == right.metadata
    assert left.fragment_digest == right.fragment_digest


def test_prepare_and_reconstruct_own_tensor_storage_and_containers():
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    value = {"tensor": [source], "stable": 3}
    plan = _prepare_canonical_wire_value(value, _limits(chunk_bytes=3))
    source.fill_(99)
    value["tensor"].append("late")
    prepared = _parse_canonical_wire_metadata(plan.metadata, limits=_limits(chunk_bytes=3))
    result = _reconstruct_canonical_wire_value(prepared, plan.tensors)
    assert torch.equal(result["tensor"][0], torch.arange(6, dtype=torch.float32).reshape(2, 3).t())
    assert result["tensor"][0].data_ptr() != plan.tensors[0].data_ptr()
    plan.tensors[0].fill_(-1)
    assert not torch.equal(result["tensor"][0], plan.tensors[0])
    result["tensor"].append("owned")
    assert len(value["tensor"]) == 2


def test_tiny_chunk_budget_is_wire_identical_for_real_and_complex_tensors():
    value = {
        "complex": torch.tensor([1 + 2j, -3 + 4j, 5 - 6j], dtype=torch.complex128),
        "view": torch.arange(60, dtype=torch.float64).reshape(6, 10)[:, ::3],
    }
    normal = _prepare_canonical_wire_value(value, _limits())
    tiny_limits = _limits(chunk_bytes=1)
    tiny = _prepare_canonical_wire_value(value, tiny_limits)
    assert tiny.metadata == normal.metadata
    assert tiny.metadata_digest == normal.metadata_digest
    assert tiny.fragment_digest == normal.fragment_digest
    for tiny_tensor, normal_tensor in zip(tiny.tensors, normal.tensors):
        assert torch.equal(tiny_tensor, normal_tensor)
    prepared = _parse_canonical_wire_metadata(tiny.metadata, limits=tiny_limits)
    result = _reconstruct_canonical_wire_value(prepared, tiny.tensors)
    _assert_value_equal(value, result)


def test_noncontiguous_clone_scales_coordinate_scratch_to_chunk_budget(monkeypatch):
    value = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    value = value.permute(3, 2, 1, 0)
    counts = []
    original = wire.torch.arange

    def tracked(start, stop, **kwargs):
        counts.append(stop - start)
        return original(start, stop, **kwargs)

    monkeypatch.setattr(wire.torch, "arange", tracked)
    limits = _limits(chunk_bytes=100)
    plan = _prepare_canonical_wire_value(value, limits)

    assert counts
    assert max(counts) <= 100 // (value.element_size() + 8 * (value.ndim + 2))
    assert torch.equal(plan.tensors[0], value)


@pytest.mark.parametrize(
    "metadata",
    [
        b"",
        b"BADMAGIC" + struct.pack(">HHI", 1, 0, 0) + b"\0" + struct.pack(">Q", 0),
        b"GFNCV1\0\0" + struct.pack(">HHI", 2, 0, 0) + b"\0" + struct.pack(">Q", 0),
        b"GFNCV1\0\0" + struct.pack(">HHI", 1, 1, 0) + b"\0" + struct.pack(">Q", 0),
        b"GFNCV1\0\0" + struct.pack(">HHI", 1, 0, 1) + b"\0" + struct.pack(">Q", 0),
        _metadata(b"\xff"),
        _metadata(b"\0") + b"trailing",
    ],
)
def test_parser_rejects_headers_unknown_tags_truncation_and_trailing_bytes(metadata):
    with pytest.raises(ValueError):
        _parse_canonical_wire_metadata(metadata, limits=_limits())


@pytest.mark.parametrize(
    "root",
    [
        b"\x03\x02" + struct.pack(">Q", 0),
        b"\x03\x00" + struct.pack(">Q", 1) + b"\0",
        b"\x03\x01" + struct.pack(">Q", 0),
        b"\x04" + struct.pack(">d", math.inf),
        b"\x05" + struct.pack(">Q", 1) + b"\xff",
    ],
)
def test_parser_rejects_noncanonical_scalars(root):
    with pytest.raises(ValueError):
        _parse_canonical_wire_metadata(_metadata(root), limits=_limits())


def test_parser_rejects_non_string_unsorted_and_duplicate_dictionary_keys():
    roots = [
        b"\x08" + struct.pack(">Q", 1) + b"\0\0",
        b"\x08" + struct.pack(">Q", 2) + _string("b") + b"\0" + _string("a") + b"\0",
        b"\x08" + struct.pack(">Q", 2) + _string("a") + b"\0" + _string("a") + b"\0",
    ]
    for root in roots:
        with pytest.raises(ValueError):
            _parse_canonical_wire_metadata(_metadata(root), limits=_limits())


def test_parser_rejects_tensor_reference_and_descriptor_count_mismatches():
    cases = [
        _metadata(b"\x09" + struct.pack(">Q", 1), [_descriptor()]),
        _metadata(b"\x09" + struct.pack(">Q", 0)),
        _metadata(b"\0", [_descriptor()]),
    ]
    for metadata in cases:
        with pytest.raises(ValueError):
            _parse_canonical_wire_metadata(metadata, limits=_limits())


@pytest.mark.parametrize(
    "descriptor",
    [
        _descriptor(dtype_code=99),
        _descriptor(flags=1),
        _descriptor(dtype_code=5, shape=(2,), numel=3, nbytes=12),
        _descriptor(dtype_code=5, shape=(2,), numel=2, nbytes=7),
    ],
)
def test_parser_rejects_invalid_tensor_descriptors(descriptor):
    metadata = _metadata(b"\x09" + struct.pack(">Q", 0), [descriptor])
    with pytest.raises(ValueError):
        _parse_canonical_wire_metadata(metadata, limits=_limits())


def test_parser_rejects_tensor_shape_product_overflow():
    descriptor = _descriptor(dtype_code=1, shape=((1 << 62), 4), numel=0, nbytes=0)
    metadata = _metadata(b"\x09" + struct.pack(">Q", 0), [descriptor])
    with pytest.raises(ValueError, match="overflows"):
        _parse_canonical_wire_metadata(metadata, limits=_limits())


def test_parser_rejects_unrepresentable_dimension_even_when_product_is_zero():
    descriptor = _descriptor(dtype_code=1, shape=((1 << 63), 0), numel=0, nbytes=0)
    metadata = _metadata(b"\x09" + struct.pack(">Q", 0), [descriptor])
    with pytest.raises(ValueError, match="signed int64"):
        _parse_canonical_wire_metadata(metadata, limits=_limits())


def test_parser_validates_all_metadata_before_tensor_allocation(monkeypatch):
    plan = _prepare_canonical_wire_value(torch.arange(4), _limits())

    def fail_allocation(*args, **kwargs):
        raise AssertionError("parser allocated a tensor")

    monkeypatch.setattr(wire.torch, "empty", fail_allocation)
    prepared = _parse_canonical_wire_metadata(plan.metadata, limits=_limits())
    assert prepared.tensor_specs == plan.tensor_specs
    with pytest.raises(ValueError, match="trailing"):
        _parse_canonical_wire_metadata(plan.metadata + b"x", limits=_limits())


@pytest.mark.parametrize(
    "field",
    [field.name for field in dataclasses.fields(_CanonicalWireLimits)],
)
def test_limits_require_strict_positive_ints(field):
    limits = _limits()
    with pytest.raises(ValueError):
        dataclasses.replace(limits, **{field: 0})
    with pytest.raises(TypeError):
        dataclasses.replace(limits, **{field: True})


def test_limits_reject_unrepresentable_or_inconsistent_bounds():
    with pytest.raises(TypeError):
        _CanonicalWireLimits()
    with pytest.raises(ValueError):
        _limits(max_tensor_rank=(1 << 16))
    with pytest.raises(ValueError, match="recursive parsing"):
        _limits(max_tree_depth=257)
    with pytest.raises(ValueError):
        _CanonicalWireLimits(
            max_fragment_tensor_bytes=(1 << 40) + 1,
            max_collective_tensor_bytes=1 << 40,
            max_collective_metadata_bytes=1 << 30,
        )
    with pytest.raises(ValueError):
        _CanonicalWireLimits(
            max_fragment_tensor_bytes=1 << 20,
            max_collective_tensor_bytes=8 << 20,
            max_collective_metadata_bytes=1 << 30,
            max_metadata_bytes=(1 << 30) + 1,
        )


def test_prepare_and_parse_enforce_tree_and_scalar_limits():
    cases = [
        ([None, None], _limits(max_container_items=1), "max_container_items"),
        ([[None]], _limits(max_tree_depth=1), "max_tree_depth"),
        ([None, None], _limits(max_tree_nodes=2), "max_tree_nodes"),
        ("ab", _limits(max_string_bytes=1), "max_string_bytes"),
        (256, _limits(max_integer_bytes=1), "max_integer_bytes"),
    ]
    for value, limits, match in cases:
        with pytest.raises(ValueError, match=match):
            _prepare_canonical_wire_value(value, limits)
    encoded = [
        (_metadata(b"\x06" + struct.pack(">Q", 2) + b"\0\0"), _limits(max_container_items=1)),
        (_metadata(b"\x06" + struct.pack(">Q", 1) + b"\x06" + struct.pack(">Q", 1) + b"\0"), _limits(max_tree_depth=1)),
        (_metadata(b"\x06" + struct.pack(">Q", 2) + b"\0\0"), _limits(max_tree_nodes=2)),
        (_metadata(_string("ab")), _limits(max_string_bytes=1)),
        (_metadata(b"\x03\0" + struct.pack(">Q", 2) + b"\x01\x00"), _limits(max_integer_bytes=1)),
    ]
    for metadata, limits in encoded:
        with pytest.raises(ValueError):
            _parse_canonical_wire_metadata(metadata, limits=limits)


def test_prepare_and_parse_enforce_metadata_tensor_count_rank_and_byte_limits():
    tensor = torch.arange(4, dtype=torch.int32)
    normal = _prepare_canonical_wire_value(tensor, _limits())
    two_tensors = _prepare_canonical_wire_value([tensor, tensor], _limits())
    rank_two = _prepare_canonical_wire_value(tensor.reshape(2, 2), _limits())
    with pytest.raises(ValueError, match="max_metadata_bytes"):
        _prepare_canonical_wire_value(None, _limits(max_metadata_bytes=24))
    with pytest.raises(ValueError, match="max_metadata_bytes"):
        _parse_canonical_wire_metadata(normal.metadata, limits=_limits(max_metadata_bytes=len(normal.metadata) - 1))
    with pytest.raises(ValueError, match="max_tensors"):
        _prepare_canonical_wire_value([tensor, tensor], _limits(max_tensors=1))
    with pytest.raises(ValueError, match="max_tensors"):
        _parse_canonical_wire_metadata(two_tensors.metadata, limits=_limits(max_tensors=1))
    with pytest.raises(ValueError, match="max_tensor_rank"):
        _prepare_canonical_wire_value(tensor.reshape(2, 2), _limits(max_tensor_rank=1))
    with pytest.raises(ValueError, match="max_tensor_rank"):
        _parse_canonical_wire_metadata(rank_two.metadata, limits=_limits(max_tensor_rank=1))
    with pytest.raises(ValueError, match="max_fragment_tensor_bytes"):
        _prepare_canonical_wire_value(tensor, _limits(max_fragment_tensor_bytes=15))
    with pytest.raises(ValueError, match="max_fragment_tensor_bytes"):
        _parse_canonical_wire_metadata(normal.metadata, limits=_limits(max_fragment_tensor_bytes=15))


def test_reconstruction_rejects_fragment_payload_and_geometry_corruption():
    value = torch.tensor([1.0, 2.0], dtype=torch.float32)
    plan = _prepare_canonical_wire_value(value, _limits())
    prepared = _parse_canonical_wire_metadata(plan.metadata, limits=_limits())
    with pytest.raises(ValueError, match="fragment digest"):
        _reconstruct_canonical_wire_value(prepared, plan.tensors, expected_fragment_digest=b"x" * 32)
    corrupted = (plan.tensors[0].clone(),)
    corrupted[0][0] = 9
    with pytest.raises(ValueError, match="invalid digest"):
        _reconstruct_canonical_wire_value(prepared, corrupted)
    with pytest.raises(ValueError, match="payload count"):
        _reconstruct_canonical_wire_value(prepared, ())
    with pytest.raises(ValueError, match="descriptor"):
        _reconstruct_canonical_wire_value(prepared, (plan.tensors[0].to(torch.float64),))
    with pytest.raises(ValueError, match="descriptor"):
        _reconstruct_canonical_wire_value(prepared, (plan.tensors[0].reshape(1, 2),))


def test_reconstruction_rejects_nonfinite_nontight_and_noncontiguous_payloads():
    plan = _prepare_canonical_wire_value(torch.tensor([1.0, 2.0]), _limits())
    prepared = _parse_canonical_wire_metadata(plan.metadata, limits=_limits())
    nonfinite = plan.tensors[0].clone()
    nonfinite[0] = math.inf
    with pytest.raises(ValueError, match="finite"):
        _reconstruct_canonical_wire_value(prepared, (nonfinite,))
    backing = torch.empty(3, dtype=torch.float32)
    backing[:2].copy_(plan.tensors[0])
    with pytest.raises(ValueError, match="descriptor"):
        _reconstruct_canonical_wire_value(prepared, (backing[:2],))
    noncontiguous = torch.stack((plan.tensors[0], plan.tensors[0]))[:, 0]
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="descriptor"):
        _reconstruct_canonical_wire_value(prepared, (noncontiguous,))


def test_prepare_rejects_unsupported_values_nonfinite_and_tensor_subclasses():
    class TensorSubclass(torch.Tensor):
        pass

    invalid = [object(), {1: "bad"}, float("nan"), torch.tensor([float("inf")]), torch.tensor([1], dtype=torch.uint16)]
    invalid.append(torch.Tensor._make_subclass(TensorSubclass, torch.tensor([1.0]), False))
    for value in invalid:
        with pytest.raises((TypeError, ValueError)):
            _prepare_canonical_wire_value(value, _limits())


def test_strict_internal_type_construction_and_api_arguments():
    limits = _limits()
    with pytest.raises(TypeError):
        _prepare_canonical_wire_value(None, object())
    with pytest.raises(TypeError):
        _parse_canonical_wire_metadata(bytearray(_metadata(b"\0")), limits=limits)
    with pytest.raises(TypeError):
        _reconstruct_canonical_wire_value(object(), ())
    with pytest.raises(TypeError):
        _reconstruct_canonical_wire_value(
            _parse_canonical_wire_metadata(_metadata(b"\0"), limits=limits),
            iter(()),
        )
    with pytest.raises(ValueError):
        _CanonicalWireTensorSpec(0, 1, torch.int8, (), 1, 1, b"x" * 32)
    assert dataclasses.is_dataclass(_CanonicalWirePlan)
    assert dataclasses.is_dataclass(_PreparedCanonicalWireValue)
    with pytest.raises(dataclasses.FrozenInstanceError):
        limits.chunk_bytes = 1
