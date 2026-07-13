"""Primitive, device-neutral helpers for canonical optimizer state."""

import math

import torch


CANONICAL_STATE_FORMAT_VERSION = 1
_IMPORT_PLAN_TOKEN = object()


class PreparedCanonicalStateImport:
    """Opaque, single-use result of canonical import preparation."""

    __slots__ = ("_optimizer", "_live_token", "_staged", "_consumed")

    def __init__(self, optimizer, live_token, staged, *, _token=None):
        if _token is not _IMPORT_PLAN_TOKEN:
            raise TypeError("PreparedCanonicalStateImport values are created by an optimizer")
        self._optimizer = optimizer
        self._live_token = live_token
        self._staged = staged
        self._consumed = False


def make_prepared_canonical_state_import(optimizer, live_token, staged):
    return PreparedCanonicalStateImport(
        optimizer,
        live_token,
        staged,
        _token=_IMPORT_PLAN_TOKEN,
    )


def canonical_value_supported(value, *, finite_tensors=False) -> bool:
    """Return whether ``value`` has a deterministic weights-only wire form."""

    if type(value) is torch.Tensor:
        supported = (
            value.layout is torch.strided
            and not value.is_meta
            and not value.is_nested
            and not value.is_quantized
        )
        if (
            supported
            and finite_tensors
            and (value.is_floating_point() or value.is_complex())
        ):
            try:
                supported = bool(torch.isfinite(value.detach()).all())
            except (NotImplementedError, RuntimeError, TypeError):
                supported = False
        return supported
    if torch.is_tensor(value):
        return False
    if type(value) is float:
        return math.isfinite(value)
    if value is None or type(value) in {bool, int, str}:
        return True
    if type(value) in {list, tuple}:
        return all(
            canonical_value_supported(
                item, finite_tensors=finite_tensors
            )
            for item in value
        )
    if type(value) is dict:
        return all(
            type(key) is str
            and canonical_value_supported(
                item, finite_tensors=finite_tensors
            )
            for key, item in value.items()
        )
    return False


def clone_canonical_value(value, *, path="value"):
    """Clone one supported value into a CPU, weights-only-safe wire value."""

    if torch.is_tensor(value):
        if (
            type(value) is not torch.Tensor
            or value.layout is not torch.strided
            or value.is_meta
            or value.is_nested
            or value.is_quantized
        ):
            raise TypeError(
                "{} must be a plain materialized strided tensor for canonical state".format(
                    path
                )
            )
        cloned = (
            value.detach()
            .to(device="cpu")
            .resolve_conj()
            .resolve_neg()
            .contiguous()
            .clone()
        )
        if cloned.is_floating_point() or cloned.is_complex():
            try:
                finite = bool(torch.isfinite(cloned).all())
            except (NotImplementedError, RuntimeError, TypeError) as exc:
                raise ValueError(
                    "{} tensor dtype does not support finite canonical state".format(
                        path
                    )
                ) from exc
            if not finite:
                raise ValueError("{} tensor must be finite".format(path))
        return cloned
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("{} must be finite".format(path))
        return value
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is list:
        return [clone_canonical_value(item, path="{}[{}]".format(path, index)) for index, item in enumerate(value)]
    if type(value) is tuple:
        return tuple(clone_canonical_value(item, path="{}[{}]".format(path, index)) for index, item in enumerate(value))
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("{} dictionary keys must be strings".format(path))
        return {key: clone_canonical_value(value[key], path="{}.{}".format(path, key)) for key in sorted(value)}
    raise TypeError("{} has unsupported canonical-state type {}".format(path, type(value).__name__))


def canonical_values_equal(left, right) -> bool:
    """Compare canonical values exactly while ignoring tensor device."""

    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(
                left.detach().cpu().contiguous(),
                right.detach().cpu().contiguous(),
            )
        )
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(canonical_values_equal(left[key], right[key]) for key in left)
    if type(left) in {list, tuple}:
        return len(left) == len(right) and all(canonical_values_equal(a, b) for a, b in zip(left, right))
    return left == right


__all__ = [
    "CANONICAL_STATE_FORMAT_VERSION",
    "PreparedCanonicalStateImport",
]
