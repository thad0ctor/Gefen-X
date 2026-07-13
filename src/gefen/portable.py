"""Strict pure math for topology-independent Gefen optimizer state."""

import math

import torch


_PORTABLE_STATE_CHUNK_ELEMENTS = 1 << 20


def _chunk_element_budget() -> int:
    budget = _PORTABLE_STATE_CHUNK_ELEMENTS
    if type(budget) is not int or budget <= 0:
        raise RuntimeError("portable state chunk budget must be a positive int")
    return budget


def _element_chunks(numel: int):
    budget = _chunk_element_budget()
    for start in range(0, numel, budget):
        yield start, min(start + budget, numel)


def _whole_row_chunks(rows: int, width: int):
    if rows == 0:
        return
    rows_per_chunk = max(1, _chunk_element_budget() // max(1, width))
    for start in range(0, rows, rows_per_chunk):
        yield start, min(start + rows_per_chunk, rows)


def _read_flat_chunk(value: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Read one logical row-major flat slice without flattening the whole input."""

    detached = value.detach()
    if detached.is_contiguous():
        return detached.reshape(-1)[start:stop]
    linear = torch.arange(start, stop, dtype=torch.int64, device=detached.device)
    remainder = linear
    reversed_coordinates = []
    for dimension in reversed(detached.shape):
        reversed_coordinates.append(torch.remainder(remainder, dimension))
        remainder = torch.div(remainder, dimension, rounding_mode="floor")
    coordinates = tuple(reversed(reversed_coordinates))
    return detached[coordinates]


def _tensor_values_valid(value: torch.Tensor, *, nonnegative: bool) -> tuple[bool, bool]:
    finite = True
    nonnegative_values = True
    for start, stop in _element_chunks(value.numel()):
        chunk = _read_flat_chunk(value, start, stop)
        try:
            if not bool(torch.isfinite(chunk).all()):
                finite = False
                break
        except (NotImplementedError, RuntimeError, TypeError) as exc:
            raise TypeError("tensor dtype does not support finite-state validation") from exc
        if nonnegative and not bool((chunk >= 0).all()):
            nonnegative_values = False
            break
    return finite, nonnegative_values


def _validate_state_counter(value, *, name: str, minimum: int = 0) -> int:
    """Return one strict host counter after validating its lower bound."""

    if type(name) is not str:
        raise TypeError("counter name must be a string")
    if not name:
        raise ValueError("counter name must not be empty")
    if type(minimum) is not int:
        raise TypeError("counter minimum must be an int")
    if minimum < 0:
        raise ValueError("counter minimum must be nonnegative")
    if type(value) is not int:
        raise TypeError("{} must be a host int".format(name))
    if value < minimum:
        raise ValueError("{} must be at least {}".format(name, minimum))
    return value


def _validate_plain_tensor(
    value,
    *,
    name: str,
    dtype: torch.dtype,
    ndim: int | None = None,
    nonnegative: bool = False,
) -> torch.Tensor:
    if type(value) is not torch.Tensor:
        raise TypeError("{} must be a plain torch.Tensor".format(name))
    if (
        value.layout is not torch.strided
        or value.is_meta
        or value.is_nested
        or value.is_quantized
    ):
        raise TypeError("{} must be a materialized strided tensor".format(name))
    if value.dtype != dtype:
        raise TypeError("{} must have dtype {}".format(name, dtype))
    if ndim is not None and value.ndim != ndim:
        raise ValueError("{} must be {}-D".format(name, ndim))
    try:
        finite, nonnegative_values = _tensor_values_valid(
            value,
            nonnegative=nonnegative,
        )
    except TypeError as exc:
        raise TypeError("{} dtype does not support finite-state validation".format(name)) from exc
    if not finite:
        raise ValueError("{} must be finite".format(name))
    if nonnegative and not nonnegative_values:
        raise ValueError("{} must be nonnegative".format(name))
    return value


def _validate_logical_shape(logical_shape) -> tuple[int, ...]:
    if type(logical_shape) not in {tuple, torch.Size}:
        raise TypeError("logical_shape must be a tuple or torch.Size")
    shape = tuple(logical_shape)
    for dimension in shape:
        if type(dimension) is not int:
            raise TypeError("logical_shape dimensions must be ints")
        if dimension < 0:
            raise ValueError("logical_shape dimensions must be nonnegative")
    return shape


def _validate_period(period, *, numel: int) -> int:
    if type(period) is not int:
        raise TypeError("period must be an int")
    if period <= 0:
        raise ValueError("period must be positive")
    if numel % period != 0:
        raise ValueError(
            "logical element count {} is not divisible by period {}".format(
                numel,
                period,
            )
        )
    return period


def _validate_codebook(codebook) -> torch.Tensor:
    codebook = _validate_plain_tensor(
        codebook,
        name="codebook",
        dtype=torch.float32,
        ndim=1,
    )
    if not 1 <= codebook.numel() <= 256:
        raise ValueError("codebook must contain between 1 and 256 entries")
    if codebook.numel() > 1 and not bool(
        torch.all(codebook.detach()[1:] >= codebook.detach()[:-1])
    ):
        raise ValueError("codebook must be sorted in nondecreasing order")
    if not bool(((codebook.detach() >= -1.0) & (codebook.detach() <= 1.0)).all()):
        raise ValueError("codebook entries must lie in [-1, 1]")
    # ``searchsorted`` warns and performs its own hidden copy for a strided
    # boundary tensor. A Gefen codebook has at most 256 elements, so normalize
    # this bounded input once and reuse it across every logical-state chunk.
    return codebook.detach().contiguous()


def _new_output(shape, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.empty(shape, dtype=dtype, device=device)


def _finish_output(result: torch.Tensor, *, name: str) -> torch.Tensor:
    if result.is_floating_point() and not _tensor_values_valid(
        result,
        nonnegative=False,
    )[0]:
        raise ValueError("{} cannot be represented as finite {} state".format(name, result.dtype))
    if (
        result.requires_grad
        or not result.is_contiguous()
        or result.storage_offset() != 0
        or result.untyped_storage().nbytes() != result.numel() * result.element_size()
    ):
        raise RuntimeError("{} projection did not produce tight detached state".format(name))
    return result


def _nearest_codebook_indices(
    codebook: torch.Tensor,
    normalized_values: torch.Tensor,
) -> torch.Tensor:
    """Return nearest codewords, resolving equal distances to the lower index."""

    if codebook.device != normalized_values.device:
        raise ValueError("codebook and normalized values must be on the same device")
    codebook = codebook.detach().contiguous()
    output = _new_output(
        normalized_values.shape,
        dtype=torch.uint8,
        device=normalized_values.device,
    )
    if codebook.numel() == 1:
        output.zero_()
        return _finish_output(output, name="momentum indices")

    output_flat = output.reshape(-1)
    for start, stop in _element_chunks(normalized_values.numel()):
        segment = _read_flat_chunk(normalized_values, start, stop).float()
        insertion = torch.searchsorted(codebook, segment)
        left = (insertion - 1).clamp_(0, codebook.numel() - 1)
        right = insertion.clamp_(0, codebook.numel() - 1)
        indices = torch.where(
            (segment - codebook[left]).abs()
            <= (segment - codebook[right]).abs(),
            left,
            right,
        )
        output_flat[start:stop].copy_(indices.to(torch.uint8))
    return _finish_output(output, name="momentum indices")


def _decode_quantized_momentum(
    codebook,
    indices,
    magnitudes,
    *,
    logical_shape,
    period,
    step,
) -> torch.Tensor:
    """Validate and decode block-quantized momentum to dense logical fp32 state."""

    _validate_state_counter(step, name="step", minimum=1)
    shape = _validate_logical_shape(logical_shape)
    numel = math.prod(shape)
    period = _validate_period(period, numel=numel)
    codebook = _validate_codebook(codebook)
    indices = _validate_plain_tensor(
        indices,
        name="momentum indices",
        dtype=torch.uint8,
        ndim=2,
    )
    magnitudes = _validate_plain_tensor(
        magnitudes,
        name="momentum magnitudes",
        dtype=torch.float32,
        ndim=2,
        nonnegative=True,
    )
    if indices.device != codebook.device or magnitudes.device != codebook.device:
        raise ValueError("codebook, momentum indices, and magnitudes must share a device")

    blocks = numel // period
    if tuple(indices.shape) != (blocks, period):
        raise ValueError(
            "momentum indices must have shape ({}, {})".format(blocks, period)
        )
    if tuple(magnitudes.shape) != (blocks, 1):
        raise ValueError("momentum magnitudes must have shape ({}, 1)".format(blocks))
    for start, stop in _element_chunks(indices.numel()):
        if int(_read_flat_chunk(indices, start, stop).max().item()) >= codebook.numel():
            raise ValueError("momentum indices contain an out-of-range codebook entry")

    dense = _new_output(shape, dtype=torch.float32, device=codebook.device)
    dense_flat = dense.reshape(-1)
    for row_start, row_stop in _whole_row_chunks(blocks, period):
        flat_start = row_start * period
        flat_stop = row_stop * period
        index_chunk = _read_flat_chunk(indices, flat_start, flat_stop).long()
        coefficients = codebook.index_select(0, index_chunk)
        magnitude_chunk = _read_flat_chunk(
            magnitudes,
            row_start,
            row_stop,
        ).reshape(-1, 1)
        decoded = coefficients.reshape(-1, period) * magnitude_chunk
        dense_flat[flat_start:flat_stop].copy_(decoded.reshape(-1))
    return _finish_output(dense, name="dense momentum")


def _recompress_dense_momentum(
    momentum,
    codebook,
    *,
    period,
    step,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compress dense fp32 momentum into one magnitude and byte indices per block."""

    _validate_state_counter(step, name="step", minimum=1)
    momentum = _validate_plain_tensor(
        momentum,
        name="dense momentum",
        dtype=torch.float32,
    )
    codebook = _validate_codebook(codebook)
    if momentum.device != codebook.device:
        raise ValueError("dense momentum and codebook must share a device")
    period = _validate_period(period, numel=momentum.numel())
    preserve_zero_sign = (
        period == 1
        and bool(codebook[0] == -1.0)
        and bool(codebook[-1] == 1.0)
    )

    blocks = momentum.numel() // period
    indices = _new_output(
        (blocks, period),
        dtype=torch.uint8,
        device=momentum.device,
    )
    magnitudes = _new_output(
        (blocks, 1),
        dtype=torch.float32,
        device=momentum.device,
    )
    for row_start, row_stop in _whole_row_chunks(blocks, period):
        flat_start = row_start * period
        flat_stop = row_stop * period
        block_values = _read_flat_chunk(
            momentum,
            flat_start,
            flat_stop,
        ).reshape(-1, period)
        magnitude_chunk = block_values.abs().amax(dim=1, keepdim=True)
        normalized = block_values.clone()
        nonzero = magnitude_chunk > 0
        normalized.div_(magnitude_chunk)
        normalized.masked_fill_(~nonzero, 0.0)
        index_chunk = _nearest_codebook_indices(codebook, normalized)
        if preserve_zero_sign:
            zero_values = block_values == 0
            zero_indices = torch.where(
                torch.signbit(block_values),
                torch.zeros_like(index_chunk),
                torch.full_like(index_chunk, codebook.numel() - 1),
            )
            index_chunk = torch.where(zero_values, zero_indices, index_chunk)
        indices[row_start:row_stop].copy_(index_chunk)
        magnitudes[row_start:row_stop].copy_(magnitude_chunk)
    return (
        _finish_output(indices, name="momentum indices"),
        _finish_output(magnitudes, name="momentum magnitudes"),
    )


def _expand_block_second_moment(
    block_values,
    *,
    logical_shape,
    period,
    step,
) -> torch.Tensor:
    """Expand one nonnegative fp32 scalar per block to dense logical state."""

    _validate_state_counter(step, name="vmean_step", minimum=1)
    shape = _validate_logical_shape(logical_shape)
    numel = math.prod(shape)
    period = _validate_period(period, numel=numel)
    block_values = _validate_plain_tensor(
        block_values,
        name="block second moment",
        dtype=torch.float32,
        ndim=2,
        nonnegative=True,
    )
    blocks = numel // period
    if tuple(block_values.shape) != (blocks, 1):
        raise ValueError("block second moment must have shape ({}, 1)".format(blocks))

    dense = _new_output(shape, dtype=torch.float32, device=block_values.device)
    dense_flat = dense.reshape(-1)
    for row_start, row_stop in _whole_row_chunks(blocks, period):
        flat_start = row_start * period
        flat_stop = row_stop * period
        block_chunk = _read_flat_chunk(
            block_values,
            row_start,
            row_stop,
        ).reshape(-1, 1)
        dense_flat[flat_start:flat_stop].reshape(-1, period).copy_(
            block_chunk.expand(-1, period)
        )
    return _finish_output(dense, name="dense second moment")


def _reduce_block_second_moment(
    dense,
    *,
    period,
    step,
) -> torch.Tensor:
    """Reduce dense nonnegative fp32 state to target-block arithmetic means."""

    _validate_state_counter(step, name="vmean_step", minimum=1)
    dense = _validate_plain_tensor(
        dense,
        name="dense second moment",
        dtype=torch.float32,
        nonnegative=True,
    )
    period = _validate_period(period, numel=dense.numel())
    blocks = dense.numel() // period
    reduced = _new_output(
        (blocks, 1),
        dtype=torch.float32,
        device=dense.device,
    )
    for row_start, row_stop in _whole_row_chunks(blocks, period):
        flat_start = row_start * period
        flat_stop = row_stop * period
        values64 = _read_flat_chunk(dense, flat_start, flat_stop).reshape(
            -1,
            period,
        ).to(torch.float64)
        reduced[row_start:row_stop].copy_(
            values64.mean(dim=1, keepdim=True).to(torch.float32)
        )
    return _finish_output(reduced, name="block second moment")


def _expand_factored_second_moment(
    row,
    column,
    *,
    logical_shape,
    step,
) -> torch.Tensor:
    """Expand Adafactor row/column state as ``outer(row, column) / mean(row)``."""

    _validate_state_counter(step, name="factored_step", minimum=1)
    shape = _validate_logical_shape(logical_shape)
    if len(shape) != 2:
        raise ValueError("factored second moment requires a 2-D logical shape")
    row = _validate_plain_tensor(
        row,
        name="row second moment",
        dtype=torch.float32,
        ndim=1,
        nonnegative=True,
    )
    column = _validate_plain_tensor(
        column,
        name="column second moment",
        dtype=torch.float32,
        ndim=1,
        nonnegative=True,
    )
    if row.device != column.device:
        raise ValueError("row and column second moments must share a device")
    if tuple(row.shape) != (shape[0],) or tuple(column.shape) != (shape[1],):
        raise ValueError("factored second-moment vectors do not match logical_shape")
    dense = _new_output(shape, dtype=torch.float32, device=row.device)
    if math.prod(shape) == 0:
        return _finish_output(dense, name="dense second moment")

    row_mean = row.detach().mean(dtype=torch.float64)
    if bool(row_mean == 0):
        has_nonzero_column = any(
            bool((_read_flat_chunk(column, start, stop) != 0).any())
            for start, stop in _element_chunks(column.numel())
        )
        if has_nonzero_column:
            raise ValueError(
                "zero row mean is inconsistent with a nonzero column second moment"
            )
        dense.zero_()
    else:
        column64 = column.detach().to(torch.float64)
        for row_start, row_stop in _whole_row_chunks(shape[0], shape[1]):
            row64 = _read_flat_chunk(row, row_start, row_stop).to(torch.float64)
            expanded = torch.outer(row64, column64).div_(row_mean).to(torch.float32)
            dense[row_start:row_stop].copy_(expanded)
    return _finish_output(dense, name="dense second moment")


def _project_factored_second_moment(
    dense,
    *,
    step,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project dense 2-D second moment to its row and column arithmetic means."""

    _validate_state_counter(step, name="factored_step", minimum=1)
    dense = _validate_plain_tensor(
        dense,
        name="dense second moment",
        dtype=torch.float32,
        ndim=2,
        nonnegative=True,
    )
    rows, columns = dense.shape
    row = _new_output((rows,), dtype=torch.float32, device=dense.device)
    column = _new_output((columns,), dtype=torch.float32, device=dense.device)
    if dense.numel() == 0:
        row.zero_()
        column.zero_()
    else:
        for matrix, output in ((dense.detach(), row), (dense.detach().t(), column)):
            matrix_rows, width = matrix.shape
            for row_start, row_stop in _whole_row_chunks(matrix_rows, width):
                values64 = matrix[row_start:row_stop].to(torch.float64)
                output[row_start:row_stop].copy_(
                    values64.mean(dim=1).to(torch.float32)
                )
    return (
        _finish_output(row, name="row second moment"),
        _finish_output(column, name="column second moment"),
    )


__all__ = []
