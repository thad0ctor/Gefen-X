import warnings

import pytest
import torch

import gefen.portable as portable_math
from gefen.portable import (
    _decode_quantized_momentum,
    _expand_block_second_moment,
    _expand_factored_second_moment,
    _project_factored_second_moment,
    _recompress_dense_momentum,
    _reduce_block_second_moment,
    _validate_state_counter,
)


def _assert_tight(tensor, *, shape, dtype=torch.float32):
    assert type(tensor) is torch.Tensor
    assert tuple(tensor.shape) == tuple(shape)
    assert tensor.dtype == dtype
    assert tensor.layout is torch.strided
    assert not tensor.requires_grad
    assert tensor.grad_fn is None
    assert tensor.is_contiguous()
    assert tensor.storage_offset() == 0
    assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert bool(torch.isfinite(tensor).all())


def _codebook():
    return torch.tensor([-1.0, -0.5, 0.5, 1.0], dtype=torch.float32)


def _sparse_tensor():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            (4,),
        )


def _reference_nearest_indices(codebook, values):
    codebook = codebook.contiguous()
    flat = values.reshape(-1).float()
    insertion = torch.searchsorted(codebook, flat)
    left = (insertion - 1).clamp(0, codebook.numel() - 1)
    right = insertion.clamp(0, codebook.numel() - 1)
    return torch.where(
        (flat - codebook[left]).abs() <= (flat - codebook[right]).abs(),
        left,
        right,
    ).to(torch.uint8).reshape(values.shape)


def _reference_recompress(momentum, codebook, period):
    blocks = momentum.reshape(-1, period)
    magnitudes = blocks.abs().amax(dim=1, keepdim=True)
    normalized = blocks.clone()
    nonzero = magnitudes > 0
    normalized.div_(magnitudes)
    normalized.masked_fill_(~nonzero, 0.0)
    return _reference_nearest_indices(codebook, normalized), magnitudes


def test_decode_quantized_momentum_validates_geometry_and_returns_dense_tight_fp32():
    codebook = torch.tensor([-1.0, -0.25, 0.25, 1.0], dtype=torch.float32)
    indices = torch.tensor([[0, 2], [3, 1]], dtype=torch.uint8)
    magnitudes = torch.tensor([[2.0], [4.0]], dtype=torch.float32)
    codebook_before = codebook.clone()
    indices_before = indices.clone()
    magnitudes_before = magnitudes.clone()

    dense = _decode_quantized_momentum(
        codebook,
        indices,
        magnitudes,
        logical_shape=(2, 2),
        period=2,
        step=7,
    )

    _assert_tight(dense, shape=(2, 2))
    assert torch.equal(dense, torch.tensor([[-2.0, 0.5], [4.0, -1.0]]))
    assert torch.equal(codebook, codebook_before)
    assert torch.equal(indices, indices_before)
    assert torch.equal(magnitudes, magnitudes_before)
    assert dense.untyped_storage().data_ptr() != magnitudes.untyped_storage().data_ptr()


def test_recompression_uses_exact_lower_index_tie_semantics():
    momentum = torch.tensor([[-2.0, -1.5], [2.0, 0.0]], requires_grad=True)
    indices, magnitudes = _recompress_dense_momentum(
        momentum,
        _codebook(),
        period=2,
        step=1,
    )

    _assert_tight(indices, shape=(2, 2), dtype=torch.uint8)
    _assert_tight(magnitudes, shape=(2, 1))
    assert torch.equal(magnitudes, torch.tensor([[2.0], [2.0]]))
    # -0.75 ties indices 0/1 and 0.0 ties indices 1/2. Both choose the lower index.
    assert torch.equal(indices, torch.tensor([[0, 0], [3, 1]], dtype=torch.uint8))
    assert torch.equal(momentum.detach(), torch.tensor([[-2.0, -1.5], [2.0, 0.0]]))


def test_period_one_recompression_is_exact_for_every_finite_fp32_scale():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    maximum = torch.tensor(torch.finfo(torch.float32).max)
    momentum = torch.stack(
        (
            -maximum,
            torch.tensor(-123.75),
            -tiny,
            torch.tensor(-0.0),
            torch.tensor(0.0),
            tiny,
            torch.tensor(19.125),
            maximum,
        )
    )
    codebook = torch.tensor(
        [-1.0, -1.0, -0.2, 0.3, 1.0, 1.0],
        dtype=torch.float32,
    )

    indices, magnitudes = _recompress_dense_momentum(
        momentum,
        codebook,
        period=1,
        step=99,
    )
    reconstructed = _decode_quantized_momentum(
        codebook,
        indices,
        magnitudes,
        logical_shape=momentum.shape,
        period=1,
        step=99,
    )

    assert torch.equal(reconstructed.view(torch.int32), momentum.view(torch.int32))
    assert torch.equal(magnitudes.reshape(-1), momentum.abs())
    assert torch.equal(indices[momentum < 0], torch.zeros_like(indices[momentum < 0]))
    assert torch.equal(indices[momentum > 0], torch.full_like(indices[momentum > 0], 4))
    assert indices[3].item() == 0
    assert indices[4].item() == codebook.numel() - 1


def test_period_one_block_reduction_preserves_every_fp32_bit():
    tiny = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))
    dense = torch.stack(
        (
            torch.tensor(-0.0),
            torch.tensor(0.0),
            tiny,
            torch.tensor(torch.finfo(torch.float32).tiny),
            torch.tensor(19.125),
            torch.tensor(torch.finfo(torch.float32).max),
        )
    )

    reduced = _reduce_block_second_moment(dense, period=1, step=99)

    assert torch.equal(
        reduced.reshape(-1).view(torch.int32),
        dense.view(torch.int32),
    )


def test_scalar_momentum_roundtrips_as_zero_dimensional_logical_state():
    momentum = torch.tensor(-123.75, requires_grad=True)
    indices, magnitudes = _recompress_dense_momentum(
        momentum,
        _codebook(),
        period=1,
        step=2,
    )
    decoded = _decode_quantized_momentum(
        _codebook(),
        indices,
        magnitudes,
        logical_shape=(),
        period=1,
        step=2,
    )

    _assert_tight(indices, shape=(1, 1), dtype=torch.uint8)
    _assert_tight(magnitudes, shape=(1, 1))
    _assert_tight(decoded, shape=())
    assert decoded.ndim == 0
    assert torch.equal(decoded, momentum.detach())


def test_expanded_zero_stride_inputs_preserve_logical_row_major_values():
    momentum = torch.tensor([-2.0, 0.0, 3.0]).reshape(3, 1).expand(3, 4)
    assert momentum.stride() == (1, 0)
    codebook = torch.tensor([-1.0, 0.0, 1.0])
    indices, magnitudes = _recompress_dense_momentum(
        momentum,
        codebook,
        period=4,
        step=1,
    )
    decoded = _decode_quantized_momentum(
        codebook,
        indices,
        magnitudes,
        logical_shape=momentum.shape,
        period=4,
        step=1,
    )
    assert torch.equal(decoded, momentum)

    dense_second_moment = torch.tensor([1.0, 3.0, 5.0]).reshape(3, 1).expand(3, 4)
    reduced = _reduce_block_second_moment(
        dense_second_moment,
        period=4,
        step=1,
    )
    assert torch.equal(reduced, torch.tensor([[1.0], [3.0], [5.0]]))
    expanded_blocks = _expand_block_second_moment(
        torch.tensor([[2.0]]).expand(3, 1),
        logical_shape=(3, 4),
        period=4,
        step=1,
    )
    assert torch.equal(expanded_blocks, torch.full((3, 4), 2.0))

    factored = _expand_factored_second_moment(
        torch.tensor([2.0]).expand(3),
        torch.tensor([4.0]).expand(5),
        logical_shape=(3, 5),
        step=1,
    )
    assert torch.equal(factored, torch.full((3, 5), 4.0))
    projected_row, projected_column = _project_factored_second_moment(
        torch.tensor([7.0]).expand(4, 6),
        step=1,
    )
    assert torch.equal(projected_row, torch.full((4,), 7.0))
    assert torch.equal(projected_column, torch.full((6,), 7.0))


def test_noncontiguous_codebook_is_normalized_without_searchsorted_warning():
    codebook = torch.linspace(-1.0, 1.0, 17)[::2]
    assert not codebook.is_contiguous()
    values = torch.tensor([-1.0, -0.875, -0.75, 0.0, 0.875, 1.0])
    assert torch.equal(
        portable_math._nearest_codebook_indices(codebook, values),
        _reference_nearest_indices(codebook, values),
    )
    indices, magnitudes = _recompress_dense_momentum(
        values,
        codebook,
        period=2,
        step=1,
    )
    assert torch.equal(
        _decode_quantized_momentum(
            codebook,
            indices,
            magnitudes,
            logical_shape=values.shape,
            period=2,
            step=1,
        ),
        (codebook[indices.long()] * magnitudes).reshape(values.shape),
    )


@pytest.mark.parametrize("period", [1, 2, 4, 8])
def test_recompression_supports_every_divisor_as_a_target_period(period):
    momentum = torch.tensor(
        [[-8.0, -5.0, -3.0, -1.0], [0.0, 1.0, 2.0, 8.0]],
        dtype=torch.float32,
    ).t()
    assert not momentum.is_contiguous()
    codebook = torch.linspace(-1.0, 1.0, 17)

    indices, magnitudes = _recompress_dense_momentum(
        momentum,
        codebook,
        period=period,
        step=2,
    )
    blocks = momentum.numel() // period
    expected_magnitudes = momentum.reshape(blocks, period).abs().amax(dim=1, keepdim=True)
    assert torch.equal(magnitudes, expected_magnitudes)
    _assert_tight(indices, shape=(blocks, period), dtype=torch.uint8)
    _assert_tight(magnitudes, shape=(blocks, 1))


def test_tiny_chunk_budget_matches_whole_tensor_references_across_partial_tails(monkeypatch):
    monkeypatch.setattr(portable_math, "_PORTABLE_STATE_CHUNK_ELEMENTS", 7)
    codebook = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    normalized = torch.tensor(
        [
            [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25],
            [0.5, 0.75, 1.0, -0.75, 0.25, -0.25],
            [0.1, -0.1, 0.9, -0.9, 0.6, -0.6],
            [0.2, -0.2, 0.8, -0.8, 0.4, -0.4],
            [0.3, -0.3, 0.7, -0.7, 1.0, -1.0],
        ]
    ).t()
    assert not normalized.is_contiguous()
    expected_nearest = _reference_nearest_indices(codebook, normalized)
    actual_nearest = portable_math._nearest_codebook_indices(codebook, normalized)
    assert torch.equal(actual_nearest, expected_nearest)

    momentum = torch.linspace(-9.0, 7.0, 30).reshape(6, 5).t()
    assert not momentum.is_contiguous()
    expected_indices, expected_magnitudes = _reference_recompress(
        momentum,
        codebook,
        6,
    )
    actual_indices, actual_magnitudes = _recompress_dense_momentum(
        momentum,
        codebook,
        period=6,
        step=3,
    )
    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_magnitudes, expected_magnitudes)
    decoded = _decode_quantized_momentum(
        codebook,
        actual_indices,
        actual_magnitudes,
        logical_shape=momentum.shape,
        period=6,
        step=3,
    )
    expected_decoded = (
        codebook[expected_indices.long()] * expected_magnitudes
    ).reshape(momentum.shape)
    assert torch.equal(decoded, expected_decoded)

    dense_second_moment = torch.arange(1, 31, dtype=torch.float32).reshape(6, 5).t()
    expected_blocks = dense_second_moment.reshape(-1, 6).to(torch.float64).mean(
        dim=1,
        keepdim=True,
    ).to(torch.float32)
    actual_blocks = _reduce_block_second_moment(
        dense_second_moment,
        period=6,
        step=3,
    )
    assert torch.equal(actual_blocks, expected_blocks)
    expanded_blocks = _expand_block_second_moment(
        actual_blocks,
        logical_shape=dense_second_moment.shape,
        period=6,
        step=3,
    )
    expected_expanded_blocks = torch.repeat_interleave(
        expected_blocks.reshape(-1),
        6,
    ).reshape(dense_second_moment.shape)
    assert torch.equal(expanded_blocks, expected_expanded_blocks)

    row = torch.tensor([1.0, 2.0, 4.0, 5.0, 8.0])
    column = torch.tensor([0.5, 1.0, 2.0, 3.0, 4.0, 5.5])
    expected_factored = (
        torch.outer(row.to(torch.float64), column.to(torch.float64))
        / row.to(torch.float64).mean()
    ).to(torch.float32)
    actual_factored = _expand_factored_second_moment(
        row,
        column,
        logical_shape=(5, 6),
        step=3,
    )
    assert torch.equal(actual_factored, expected_factored)
    actual_row, actual_column = _project_factored_second_moment(
        actual_factored.t(),
        step=3,
    )
    expected64 = actual_factored.t().to(torch.float64)
    assert torch.equal(actual_row, expected64.mean(dim=1).to(torch.float32))
    assert torch.equal(actual_column, expected64.mean(dim=0).to(torch.float32))


def test_period_larger_than_chunk_budget_keeps_whole_blocks_and_matches_reference(monkeypatch):
    monkeypatch.setattr(portable_math, "_PORTABLE_STATE_CHUNK_ELEMENTS", 5)
    period = 11
    codebook = torch.linspace(-1.0, 1.0, 9)
    momentum = torch.linspace(-17.0, 13.0, 33).reshape(11, 3).t()
    assert not momentum.is_contiguous()

    expected_indices, expected_magnitudes = _reference_recompress(
        momentum,
        codebook,
        period,
    )
    actual_indices, actual_magnitudes = _recompress_dense_momentum(
        momentum,
        codebook,
        period=period,
        step=2,
    )
    decoded = _decode_quantized_momentum(
        codebook,
        actual_indices,
        actual_magnitudes,
        logical_shape=momentum.shape,
        period=period,
        step=2,
    )

    assert torch.equal(actual_indices, expected_indices)
    assert torch.equal(actual_magnitudes, expected_magnitudes)
    assert torch.equal(
        decoded,
        (codebook[expected_indices.long()] * expected_magnitudes).reshape(momentum.shape),
    )
    second_moment = momentum.square()
    actual_blocks = _reduce_block_second_moment(
        second_moment,
        period=period,
        step=2,
    )
    expected_blocks = second_moment.reshape(-1, period).to(torch.float64).mean(
        dim=1,
        keepdim=True,
    ).to(torch.float32)
    assert torch.equal(actual_blocks, expected_blocks)


def test_block_second_moment_expands_then_projects_to_a_new_period():
    blocks = torch.tensor([[1.0], [3.0], [5.0], [7.0]], requires_grad=True)
    dense = _expand_block_second_moment(
        blocks,
        logical_shape=(2, 4),
        period=2,
        step=3,
    )
    projected = _reduce_block_second_moment(dense, period=4, step=3)

    _assert_tight(dense, shape=(2, 4))
    _assert_tight(projected, shape=(2, 1))
    assert torch.equal(
        dense,
        torch.tensor([[1.0, 1.0, 3.0, 3.0], [5.0, 5.0, 7.0, 7.0]]),
    )
    assert torch.equal(projected, torch.tensor([[2.0], [6.0]]))
    assert torch.equal(blocks.detach(), torch.tensor([[1.0], [3.0], [5.0], [7.0]]))


def test_block_reduction_uses_overflow_safe_means():
    maximum = torch.finfo(torch.float32).max
    dense = torch.full((2, 4), maximum)

    projected = _reduce_block_second_moment(dense, period=4, step=1)

    assert torch.equal(projected, torch.full((2, 1), maximum))
    _assert_tight(projected, shape=(2, 1))


def test_factored_expansion_and_projection_follow_adafactor_geometry():
    row = torch.tensor([1.0, 3.0])
    column = torch.tensor([2.0, 4.0, 6.0])
    dense = _expand_factored_second_moment(
        row,
        column,
        logical_shape=(2, 3),
        step=4,
    )

    _assert_tight(dense, shape=(2, 3))
    assert torch.equal(dense, torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]]))

    source = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]).t()
    assert not source.is_contiguous()
    projected_row, projected_column = _project_factored_second_moment(source, step=4)
    assert torch.equal(projected_row, torch.tensor([1.5, 3.5, 5.5]))
    assert torch.equal(projected_column, torch.tensor([3.0, 4.0]))
    _assert_tight(projected_row, shape=(3,))
    _assert_tight(projected_column, shape=(2,))


def test_factored_projection_is_overflow_safe_and_roundtrips_consistent_rank_one_state():
    maximum = torch.finfo(torch.float32).max
    projected_row, projected_column = _project_factored_second_moment(
        torch.full((2, 2), maximum),
        step=1,
    )
    assert torch.equal(projected_row, torch.full((2,), maximum))
    assert torch.equal(projected_column, torch.full((2,), maximum))

    row = torch.tensor([1.0, 3.0])
    column = torch.tensor([1.0, 2.0, 3.0])
    dense = _expand_factored_second_moment(
        row,
        column,
        logical_shape=(2, 3),
        step=2,
    )
    roundtrip_row, roundtrip_column = _project_factored_second_moment(dense, step=2)
    torch.testing.assert_close(roundtrip_row, row, rtol=0, atol=0)
    torch.testing.assert_close(roundtrip_column, column, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)])
def test_zero_element_momentum_and_block_state_are_well_defined(shape):
    momentum = torch.empty(shape, dtype=torch.float32, requires_grad=True)
    codebook = _codebook()
    indices, magnitudes = _recompress_dense_momentum(
        momentum,
        codebook,
        period=7,
        step=1,
    )
    decoded = _decode_quantized_momentum(
        codebook,
        indices,
        magnitudes,
        logical_shape=shape,
        period=7,
        step=1,
    )
    reduced = _reduce_block_second_moment(momentum.detach(), period=7, step=1)
    expanded = _expand_block_second_moment(
        reduced,
        logical_shape=shape,
        period=7,
        step=1,
    )

    _assert_tight(indices, shape=(0, 7), dtype=torch.uint8)
    _assert_tight(magnitudes, shape=(0, 1))
    _assert_tight(decoded, shape=shape)
    _assert_tight(reduced, shape=(0, 1))
    _assert_tight(expanded, shape=shape)


def test_empty_index_decode_does_not_invoke_max(monkeypatch):
    original_max = torch.Tensor.max

    def reject_empty_max(tensor, *args, **kwargs):
        if tensor.numel() == 0:
            raise AssertionError("max() was called on empty momentum indices")
        return original_max(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "max", reject_empty_max)
    decoded = _decode_quantized_momentum(
        _codebook(),
        torch.empty((0, 13), dtype=torch.uint8),
        torch.empty((0, 1), dtype=torch.float32),
        logical_shape=(0, 9),
        period=13,
        step=1,
    )
    _assert_tight(decoded, shape=(0, 9))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (0, 0)])
def test_zero_element_factored_state_uses_finite_empty_geometry(shape):
    dense = torch.empty(shape, dtype=torch.float32)
    row, column = _project_factored_second_moment(dense, step=1)
    expanded = _expand_factored_second_moment(
        row,
        column,
        logical_shape=shape,
        step=1,
    )

    _assert_tight(row, shape=(shape[0],))
    _assert_tight(column, shape=(shape[1],))
    _assert_tight(expanded, shape=shape)
    assert torch.count_nonzero(row) == 0
    assert torch.count_nonzero(column) == 0


@pytest.mark.parametrize("value", [True, 1.0, torch.tensor(1), None])
def test_counter_validation_rejects_non_host_ints(value):
    with pytest.raises(TypeError, match="host int"):
        _validate_state_counter(value, name="step", minimum=1)


def test_counter_validation_is_strict_about_bounds_and_its_own_arguments():
    assert _validate_state_counter(0, name="global_step") == 0
    assert _validate_state_counter(3, name="step", minimum=1) == 3
    with pytest.raises(ValueError, match="at least 1"):
        _validate_state_counter(0, name="step", minimum=1)
    with pytest.raises(TypeError, match="name"):
        _validate_state_counter(1, name=object())
    with pytest.raises(ValueError, match="must not be empty"):
        _validate_state_counter(1, name="")
    with pytest.raises(TypeError, match="minimum"):
        _validate_state_counter(1, name="step", minimum=True)
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_state_counter(1, name="step", minimum=-1)


@pytest.mark.parametrize("period", [True, 2.0, 0, -1, 3])
def test_invalid_periods_fail_before_projection(period):
    exception = TypeError if type(period) is not int else ValueError
    with pytest.raises(exception, match="period"):
        _recompress_dense_momentum(
            torch.ones(4),
            _codebook(),
            period=period,
            step=1,
        )


@pytest.mark.parametrize(
    "shape,exception",
    [([4], TypeError), ((True,), TypeError), ((-1, 4), ValueError), ((3,), ValueError)],
)
def test_invalid_logical_shapes_and_geometry_are_rejected(shape, exception):
    with pytest.raises(exception):
        _decode_quantized_momentum(
            _codebook(),
            torch.zeros((2, 2), dtype=torch.uint8),
            torch.ones((2, 1)),
            logical_shape=shape,
            period=2,
            step=1,
        )


@pytest.mark.parametrize(
    "codebook,match",
    [
        (torch.tensor([], dtype=torch.float32), "between 1 and 256"),
        (torch.zeros(257, dtype=torch.float32), "between 1 and 256"),
        (torch.tensor([-1.0, 0.5, 0.25, 1.0]), "sorted"),
        (torch.tensor([-1.1, 0.0, 1.0]), r"\[-1, 1\]"),
        (torch.tensor([-1.0, float("nan"), 1.0]), "finite"),
    ],
)
def test_invalid_codebook_values_are_rejected(codebook, match):
    with pytest.raises(ValueError, match=match):
        _recompress_dense_momentum(
            torch.ones(4),
            codebook,
            period=2,
            step=1,
        )


@pytest.mark.parametrize(
    "codebook",
    [torch.ones(4, dtype=torch.float64), torch.ones((2, 2), dtype=torch.float32)],
)
def test_invalid_codebook_dtype_or_rank_is_rejected(codebook):
    with pytest.raises((TypeError, ValueError), match="codebook"):
        _recompress_dense_momentum(
            torch.ones(4),
            codebook,
            period=2,
            step=1,
        )


@pytest.mark.parametrize(
    "momentum",
    [
        torch.ones(4, dtype=torch.float64),
        torch.tensor([1.0, float("inf"), 2.0, 3.0]),
        torch.nn.Parameter(torch.ones(4)),
        _sparse_tensor(),
        torch.empty(4, device="meta"),
    ],
)
def test_momentum_input_must_be_plain_strided_finite_fp32(momentum):
    with pytest.raises((TypeError, ValueError), match="momentum"):
        _recompress_dense_momentum(
            momentum,
            _codebook(),
            period=2,
            step=1,
        )


@pytest.mark.parametrize(
    "indices,magnitudes,match",
    [
        (torch.zeros((2, 2), dtype=torch.int64), torch.ones((2, 1)), "indices"),
        (torch.zeros((4,), dtype=torch.uint8), torch.ones((2, 1)), "indices"),
        (torch.full((2, 2), 4, dtype=torch.uint8), torch.ones((2, 1)), "out-of-range"),
        (torch.zeros((2, 2), dtype=torch.uint8), torch.ones((2,), dtype=torch.float32), "magnitudes"),
        (torch.zeros((2, 2), dtype=torch.uint8), -torch.ones((2, 1)), "nonnegative"),
        (
            torch.zeros((2, 2), dtype=torch.uint8),
            torch.tensor([[1.0], [float("nan")]]),
            "finite",
        ),
    ],
)
def test_decode_rejects_malformed_quantized_state(indices, magnitudes, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _decode_quantized_momentum(
            _codebook(),
            indices,
            magnitudes,
            logical_shape=(4,),
            period=2,
            step=1,
        )


def test_second_moment_projection_rejects_negative_or_nonfinite_state():
    with pytest.raises(ValueError, match="nonnegative"):
        _reduce_block_second_moment(torch.tensor([1.0, -1.0]), period=1, step=1)
    with pytest.raises(ValueError, match="finite"):
        _project_factored_second_moment(
            torch.tensor([[1.0, float("nan")]]),
            step=1,
        )


def test_factored_expansion_rejects_bad_geometry_and_degenerate_inconsistent_state():
    with pytest.raises(ValueError, match="2-D"):
        _expand_factored_second_moment(
            torch.ones(4),
            torch.ones(1),
            logical_shape=(4,),
            step=1,
        )
    with pytest.raises(ValueError, match="do not match"):
        _expand_factored_second_moment(
            torch.ones(3),
            torch.ones(2),
            logical_shape=(2, 2),
            step=1,
        )
    with pytest.raises(ValueError, match="zero row mean"):
        _expand_factored_second_moment(
            torch.zeros(2),
            torch.ones(2),
            logical_shape=(2, 2),
            step=1,
        )


def test_factored_expansion_rejects_a_finite_input_whose_result_overflows_fp32():
    maximum = torch.finfo(torch.float32).max
    tiny = torch.finfo(torch.float32).tiny
    with pytest.raises(ValueError, match="cannot be represented"):
        _expand_factored_second_moment(
            torch.tensor([tiny, maximum]),
            torch.tensor([maximum, maximum]),
            logical_shape=(2, 2),
            step=1,
        )
