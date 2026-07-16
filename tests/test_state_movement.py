"""Atomic optimizer-state movement coverage for Gefen and GefenMuon."""

import copy
from collections import defaultdict, OrderedDict
import warnings

import pytest
import torch

from gefen import Gefen, GefenMuon, ParameterIdentity


_KINDS = ("block", "factored", "muon", "normuon")
_MOVABLE_STATE_KEYS = frozenset(
    {
        "step",
        "m_codebook",
        "m_magnitude",
        "vmean",
        "vmean_step",
        "v_row",
        "v_col",
        "factored_step",
        "normuon_v",
        "normuon_step",
    }
)
_SCRATCH_KEYS = frozenset({"stepsize", "_h_buf"})
_CACHE_ATTRS = (
    "_gefen_codebook_by_device",
    "_gefen_codebook_lut_by_device",
    "_sr_seed_by_device",
    "_gefen_global_step_by_device",
)


def _build_initialized(kind):
    shape = (8,) if kind == "block" else (2, 4)
    values = torch.linspace(-0.4, 0.7, 8, dtype=torch.float32).reshape(shape)
    parameter = torch.nn.Parameter(values.clone())
    tensor_lr = torch.tensor(2.0e-3, dtype=torch.float32)
    group_metadata = {"labels": ["preserve", kind]}
    group = {
        "params": [("layer.weight", parameter)],
        "lr": tensor_lr,
        "movement_metadata": group_metadata,
    }
    if kind == "block":
        optimizer = Gefen(
            [group],
            lr=tensor_lr,
            fused=False,
            factored_v_2d=False,
        )
    elif kind == "factored":
        optimizer = Gefen(
            [group],
            lr=tensor_lr,
            fused=False,
            factored_v_2d=True,
        )
    elif kind in ("muon", "normuon"):
        optimizer = GefenMuon(
            [group],
            lr=tensor_lr,
            weight_decay=0.0,
            fused=False,
            ns_steps=1,
            normuon=kind == "normuon",
        )
    else:
        raise AssertionError("unknown optimizer kind: {}".format(kind))

    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    optimizer._predict_period_from_grad_sq = lambda *args, **kwargs: 4
    parameter.grad = torch.linspace(-1.25, 0.75, 8).reshape_as(parameter)
    optimizer.step()
    return optimizer, parameter, tensor_lr, group_metadata


def _oversized_copy(tensor):
    flat_size = tensor.numel()
    backing = torch.empty(
        flat_size + 11,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result = backing.narrow(0, 5, flat_size).view(tensor.shape)
    result.copy_(tensor)
    assert result.untyped_storage().nbytes() > flat_size * tensor.element_size()
    return result


def _make_persistent_state_oversized(optimizer, parameter):
    state = optimizer.state[parameter]
    for key, value in tuple(state.items()):
        if key in _MOVABLE_STATE_KEYS and torch.is_tensor(value):
            state[key] = _oversized_copy(value)
    optimizer._gefen_codebook = _oversized_copy(optimizer._gefen_codebook)


def _persistent_tensor_snapshot(optimizer, parameter):
    state = optimizer.state[parameter]
    result = {
        key: (value, value.detach().clone())
        for key, value in state.items()
        if key in _MOVABLE_STATE_KEYS and torch.is_tensor(value)
    }
    result["_gefen_codebook"] = (
        optimizer._gefen_codebook,
        optimizer._gefen_codebook.detach().clone(),
    )
    return result


def _assert_fresh_tight_copy(actual, old, expected, device):
    assert actual is not old
    assert actual.device == device
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.layout == torch.strided
    assert actual.is_contiguous()
    assert actual.storage_offset() == 0
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _seed_discardable_runtime_state(optimizer, parameter, tensor_lr):
    state = optimizer.state[parameter]
    state["stepsize"] = _oversized_copy(torch.tensor([91.0]))
    state["_h_buf"] = _oversized_copy(torch.tensor([92.0]))
    carrier = _oversized_copy(torch.tensor([7, 11, 13], dtype=torch.int64))
    state["_gefen_rank_local_payload_0"] = carrier
    state_metadata = {
        "labels": ["persistent", "metadata"],
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "layout": torch.strided,
        "memory_format": torch.contiguous_format,
        "shape": torch.Size((2, 3)),
        "complex": 1.0 + 2.0j,
        "bytes": b"metadata",
        "optional": None,
    }
    state["movement_metadata"] = state_metadata

    cpu = torch.device("cpu")
    optimizer._gefen_codebook_by_device[cpu] = optimizer._gefen_codebook
    optimizer._gefen_codebook_lut_by_device[cpu] = torch.tensor([19.0])
    optimizer._gefen_codebook_scope_validated = True
    optimizer._static_mark_sig = ("stale",)
    optimizer._lr_scalar_cache = (
        tensor_lr,
        tensor_lr._version,
        float(tensor_lr.item()),
    )
    return carrier, state_metadata


def _tensor_payload(tensor):
    if tensor.device.type == "meta":
        return None
    if tensor.is_nested:
        return tuple(item.detach().clone() for item in tensor.unbind())
    return tensor.detach().clone()


def _snapshot_tree(value):
    if torch.is_tensor(value):
        return (
            "tensor",
            value,
            value._version,
            value.dtype,
            value.device,
            value.layout,
            _tensor_payload(value),
        )
    if isinstance(value, dict):
        return (
            "dict",
            value,
            tuple((key, _snapshot_tree(item)) for key, item in value.items()),
        )
    if isinstance(value, list):
        return ("list", value, tuple(_snapshot_tree(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", value, tuple(_snapshot_tree(item) for item in value))
    return ("leaf", value, copy.deepcopy(value))


def _assert_tensor_payload(actual, payload):
    if payload is None:
        return
    if actual.is_nested:
        actual_items = actual.unbind()
        assert len(actual_items) == len(payload)
        for item, expected in zip(actual_items, payload):
            torch.testing.assert_close(item, expected, rtol=0, atol=0, equal_nan=True)
        return
    torch.testing.assert_close(actual, payload, rtol=0, atol=0, equal_nan=True)


def _assert_tree_exact(actual, snapshot):
    kind = snapshot[0]
    if kind == "tensor":
        _, reference, version, dtype, device, layout, payload = snapshot
        assert actual is reference
        assert actual._version == version
        assert actual.dtype == dtype
        assert actual.device == device
        assert actual.layout == layout
        _assert_tensor_payload(actual, payload)
        return
    if kind == "dict":
        _, reference, entries = snapshot
        assert actual is reference
        assert tuple(actual) == tuple(key for key, _ in entries)
        for key, child in entries:
            _assert_tree_exact(actual[key], child)
        return
    if kind in ("list", "tuple"):
        _, reference, entries = snapshot
        assert actual is reference
        assert len(actual) == len(entries)
        for item, child in zip(actual, entries):
            _assert_tree_exact(item, child)
        return
    _, reference, expected = snapshot
    assert actual is reference
    assert actual == expected


def _snapshot_exact_optimizer(optimizer):
    parameters = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    return {
        "top": optimizer.__dict__.copy(),
        "state": _snapshot_tree(optimizer.state),
        "groups": _snapshot_tree(optimizer.param_groups),
        "defaults": _snapshot_tree(optimizer.defaults),
        "param_names": _snapshot_tree(optimizer._param_names),
        "codebook": _snapshot_tree(optimizer._gefen_codebook),
        "caches": {
            name: _snapshot_tree(getattr(optimizer, name)) for name in _CACHE_ATTRS
        },
        "grads": tuple(
            (parameter, _snapshot_tree(parameter.grad)) for parameter in parameters
        ),
    }


def _assert_exact_optimizer_snapshot(optimizer, snapshot):
    assert optimizer.__dict__.keys() == snapshot["top"].keys()
    for key, value in snapshot["top"].items():
        assert optimizer.__dict__[key] is value
    _assert_tree_exact(optimizer.state, snapshot["state"])
    _assert_tree_exact(optimizer.param_groups, snapshot["groups"])
    _assert_tree_exact(optimizer.defaults, snapshot["defaults"])
    _assert_tree_exact(optimizer._param_names, snapshot["param_names"])
    _assert_tree_exact(optimizer._gefen_codebook, snapshot["codebook"])
    for name, expected in snapshot["caches"].items():
        _assert_tree_exact(getattr(optimizer, name), expected)
    for parameter, expected in snapshot["grads"]:
        _assert_tree_exact(parameter.grad, expected)


def _movement_candidates(optimizer, parameter):
    candidates = {_tensor_storage_token(optimizer._gefen_codebook)}
    candidates.update(
        _tensor_storage_token(value)
        for key, value in optimizer.state[parameter].items()
        if key in _MOVABLE_STATE_KEYS and torch.is_tensor(value)
    )
    return candidates


def _tensor_storage_token(tensor):
    return (
        tensor.device,
        tensor.untyped_storage().data_ptr(),
        tensor.storage_offset(),
        tensor.numel(),
    )


def _install_late_to_failure(monkeypatch, candidates, *, destination_type):
    original_to = torch.Tensor.to
    completed_copies = []

    def flaky_to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        if (
            _tensor_storage_token(tensor) in candidates
            and result is not tensor
            and result.device.type == destination_type
        ):
            completed_copies.append(result)
            if len(completed_copies) == 3:
                raise RuntimeError("injected late state-copy failure")
        return result

    monkeypatch.setattr(torch.Tensor, "to", flaky_to)
    return completed_copies


def _move_parameter_module(parameter, device):
    module = torch.nn.Module()
    module.register_parameter("weight", parameter)
    module.to(device)
    assert module.weight is parameter
    return module


def _persistent_values(optimizer, parameter):
    values = {
        key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in optimizer.state[parameter].items()
        if key in _MOVABLE_STATE_KEYS
    }
    values["_gefen_codebook"] = optimizer._gefen_codebook.detach().cpu().clone()
    return values


def _assert_persistent_values_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        if torch.is_tensor(left[key]):
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0, equal_nan=True)
        else:
            assert left[key] == right[key]


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("explicit_destination", [False, True])
def test_cpu_state_movement_is_fresh_tight_and_preserves_live_training_objects(
    kind, explicit_destination
):
    optimizer, parameter, tensor_lr, group_metadata = _build_initialized(kind)
    assert optimizer.optimizer_contract().capabilities.atomic_state_movement
    assert not optimizer.optimizer_contract().capabilities.state_offload

    _make_persistent_state_oversized(optimizer, parameter)
    carrier, state_metadata = _seed_discardable_runtime_state(
        optimizer, parameter, tensor_lr
    )
    tensors_before = _persistent_tensor_snapshot(optimizer, parameter)
    state_before = optimizer.state[parameter]
    non_tensor_before = {
        key: value
        for key, value in state_before.items()
        if not torch.is_tensor(value) and key not in _SCRATCH_KEYS
    }
    groups_before = optimizer.param_groups
    group_before = optimizer.param_groups[0]
    group_params_before = group_before["params"]
    defaults_before = optimizer.defaults
    param_names_before = optimizer._param_names
    grad_before = parameter.grad
    grad_value_before = grad_before.detach().clone()
    parameter_value_before = parameter.detach().clone()
    global_step_before = optimizer._gefen_global_step

    destination = torch.device("cpu") if explicit_destination else None
    optimizer.move_state_(destination)

    assert optimizer.param_groups is groups_before
    assert optimizer.param_groups[0] is group_before
    assert optimizer.param_groups[0]["params"] is group_params_before
    assert optimizer.param_groups[0]["params"][0] is parameter
    assert optimizer.param_groups[0]["lr"] is tensor_lr
    assert optimizer.param_groups[0]["movement_metadata"] is group_metadata
    assert optimizer.defaults is defaults_before
    assert optimizer.defaults["lr"] is tensor_lr
    assert optimizer._param_names is param_names_before
    assert parameter.grad is grad_before
    torch.testing.assert_close(parameter.grad, grad_value_before, rtol=0, atol=0)
    torch.testing.assert_close(parameter, parameter_value_before, rtol=0, atol=0)
    assert optimizer._gefen_global_step == global_step_before

    state = optimizer.state[parameter]
    for key, value in non_tensor_before.items():
        assert state[key] is value
    assert state["movement_metadata"] is state_metadata
    assert state["_gefen_rank_local_payload_0"] is carrier
    assert "stepsize" not in state
    assert "_h_buf" not in state

    for key, (old, expected) in tensors_before.items():
        actual = optimizer._gefen_codebook if key == "_gefen_codebook" else state[key]
        _assert_fresh_tight_copy(actual, old, expected, torch.device("cpu"))

    assert optimizer._gefen_codebook_by_device == {}
    assert optimizer._gefen_codebook_lut_by_device == {}
    assert optimizer._sr_seed_by_device == {}
    assert optimizer._gefen_global_step_by_device == {}
    assert optimizer._gefen_codebook_scope_validated is False
    assert optimizer._capt_stacks is None
    assert optimizer._static_mark_sig is None
    assert optimizer._lr_scalar_cache is None
    assert optimizer.optimizer_contract().capabilities.atomic_state_movement
    assert not optimizer.optimizer_contract().capabilities.state_offload


def test_pristine_state_movement_is_repeatable_and_returns_none():
    parameter = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer = Gefen(
        [("layer.weight", parameter)],
        fused=False,
        factored_v_2d=False,
    )
    original_state = optimizer.state
    original_parameter_state = optimizer.state[parameter]
    parameter_value = parameter.detach().clone()

    assert optimizer.move_state_() is None
    first_state = optimizer.state
    first_parameter_state = optimizer.state[parameter]
    assert first_state is not original_state
    assert first_parameter_state is not original_parameter_state
    assert first_parameter_state == {"name": "layer.weight"}
    assert optimizer._gefen_codebook is None

    assert optimizer.move_state_(torch.device("cpu")) is None
    assert optimizer.state is not first_state
    assert optimizer.state[parameter] is not first_parameter_state
    assert optimizer.state[parameter] == {"name": "layer.weight"}
    torch.testing.assert_close(parameter, parameter_value, rtol=0, atol=0)
    assert optimizer.optimizer_contract().capabilities.atomic_state_movement


def test_codebook_preinitialized_state_moves_without_initializing_parameter_tensors():
    parameter = torch.nn.Parameter(torch.arange(1, 9, dtype=torch.float32))
    tensor_lr = torch.tensor(3.0e-3)
    optimizer = Gefen(
        [("layer.weight", parameter)],
        lr=tensor_lr,
        fused=False,
        factored_v_2d=False,
    )
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    parameter.grad = torch.linspace(-2.0, 1.0, parameter.numel())
    optimizer._ensure_gefen_codebook(reuse_existing_periods=False)
    assert set(optimizer.state[parameter]) == {"name", "automatic_period"}
    optimizer._gefen_codebook = _oversized_copy(optimizer._gefen_codebook)
    old_codebook = optimizer._gefen_codebook
    expected_codebook = old_codebook.detach().clone()
    grad_before = parameter.grad
    state_metadata = {"preinitialized": True}
    optimizer.state[parameter]["movement_metadata"] = state_metadata

    optimizer.move_state_()

    assert set(optimizer.state[parameter]) == {
        "name",
        "automatic_period",
        "movement_metadata",
    }
    assert optimizer.state[parameter]["movement_metadata"] is state_metadata
    assert parameter.grad is grad_before
    assert optimizer._gefen_global_step == 0
    _assert_fresh_tight_copy(
        optimizer._gefen_codebook,
        old_codebook,
        expected_codebook,
        torch.device("cpu"),
    )


def test_noncontiguous_declared_state_is_normalized_to_a_tight_copy():
    optimizer, parameter, _, _ = _build_initialized("block")
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    assert not source.is_contiguous()
    optimizer.state[parameter]["m_magnitude"] = source
    expected = source.detach().clone()

    optimizer.move_state_()

    _assert_fresh_tight_copy(
        optimizer.state[parameter]["m_magnitude"],
        source,
        expected,
        torch.device("cpu"),
    )


def test_wrapper_orphan_state_is_preserved_and_co_located_with_its_key():
    optimizer, _, _, _ = _build_initialized("block")
    orphan = torch.nn.Parameter(torch.arange(6, dtype=torch.float32))
    orphan_tensor = _oversized_copy(torch.linspace(1.0, 2.0, 3))
    orphan_carrier = _oversized_copy(torch.tensor([3, 5, 8], dtype=torch.uint8))
    orphan_metadata = {"owner": "wrapper"}
    old_orphan_state = {
        "name": "orphan.weight",
        "automatic_period": 2,
        "step": 7,
        "m_magnitude": orphan_tensor,
        "stepsize": torch.tensor([99.0]),
        "_gefen_rank_local_payload_0": orphan_carrier,
        "movement_metadata": orphan_metadata,
    }
    optimizer.state[orphan] = old_orphan_state
    expected_tensor = orphan_tensor.detach().clone()

    optimizer.move_state_(torch.device("cpu"))

    assert orphan in optimizer.state
    assert optimizer.state[orphan] is not old_orphan_state
    moved = optimizer.state[orphan]
    assert moved["name"] == "orphan.weight"
    assert moved["automatic_period"] == 2
    assert moved["step"] == 7
    assert moved["movement_metadata"] is orphan_metadata
    assert moved["_gefen_rank_local_payload_0"] is orphan_carrier
    assert "stepsize" not in moved
    _assert_fresh_tight_copy(
        moved["m_magnitude"],
        orphan_tensor,
        expected_tensor,
        torch.device("cpu"),
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_native_checkpoint_continuation_remains_exact_after_movement(kind):
    optimizer, parameter, _, _ = _build_initialized(kind)
    resumed, resumed_parameter, _, _ = _build_initialized(kind)

    optimizer.move_state_()
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    _assert_persistent_values_equal(
        _persistent_values(optimizer, parameter),
        _persistent_values(resumed, resumed_parameter),
    )

    continuation_grad = torch.linspace(0.45, -0.85, parameter.numel()).reshape_as(
        parameter
    )
    parameter.grad = continuation_grad.clone()
    resumed_parameter.grad = continuation_grad.clone()
    optimizer.step()
    resumed.step()

    torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
    _assert_persistent_values_equal(
        _persistent_values(optimizer, parameter),
        _persistent_values(resumed, resumed_parameter),
    )


def test_movement_invalidates_prepared_canonical_import_but_keeps_io_available():
    def build_finalized():
        parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.7, 8))
        optimizer = Gefen(
            [("layer.weight", parameter)],
            fused=False,
            factored_v_2d=False,
        )
        optimizer.rebind_parameter(
            parameter,
            parameter,
            identity=ParameterIdentity("Layer.Weight", (8,)),
        )
        optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
        return optimizer, parameter

    source, source_parameter = build_finalized()
    source_parameter.grad = torch.linspace(-1.0, 0.8, 8)
    source.step()
    source.move_state_()
    exported = source.export_canonical_state()

    target, target_parameter = build_finalized()
    prepared = target.prepare_canonical_state_import(exported)
    target.move_state_()
    with pytest.raises(RuntimeError, match="changed after canonical import preparation"):
        target.commit_canonical_state_import(prepared)

    assert target.optimizer_contract().capabilities.canonical_state_io
    assert target.optimizer_contract().capabilities.atomic_state_movement
    target.import_canonical_state(exported)
    _assert_persistent_values_equal(
        _persistent_values(source, source_parameter),
        _persistent_values(target, target_parameter),
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_late_cpu_copy_failure_is_exactly_atomic(kind, monkeypatch):
    optimizer, parameter, _, _ = _build_initialized(kind)
    _make_persistent_state_oversized(optimizer, parameter)
    candidates = _movement_candidates(optimizer, parameter)
    assert len(candidates) >= 3
    snapshot = _snapshot_exact_optimizer(optimizer)
    completed = _install_late_to_failure(
        monkeypatch,
        candidates,
        destination_type="cpu",
    )

    with pytest.raises(RuntimeError, match="injected late state-copy failure"):
        optimizer.move_state_()

    assert len(completed) == 3
    _assert_exact_optimizer_snapshot(optimizer, snapshot)


def _build_rejection_case(case):
    if case == "capturable":
        parameter = torch.nn.Parameter(torch.ones(8))
        optimizer = Gefen(
            [("layer.weight", parameter)],
            fused=False,
            factored_v_2d=False,
            capturable=True,
        )
        return optimizer, parameter, None
    if case == "meta_parameter":
        parameter = torch.nn.Parameter(torch.empty(8, device="meta"))
        optimizer = Gefen(
            [("layer.weight", parameter)],
            fused=False,
            factored_v_2d=False,
        )
        return optimizer, parameter, None

    optimizer, parameter, _, _ = _build_initialized("block")
    if case == "meta_destination":
        return optimizer, parameter, torch.device("meta")
    if case == "mismatched_destination":
        return optimizer, parameter, torch.device("cuda:0")
    if case == "meta_state":
        optimizer.state[parameter]["m_magnitude"] = optimizer.state[parameter][
            "m_magnitude"
        ].to("meta")
    elif case == "undeclared_tensor":
        optimizer.state[parameter]["extension"] = torch.ones(2)
    elif case == "tensor_in_extension_container":
        optimizer.state[parameter]["extension"] = {
            "nested": [torch.ones(2)]
        }
    elif case == "tensor_in_ordered_extension":
        optimizer.state[parameter]["extension"] = OrderedDict(
            (("nested", torch.ones(2)),)
        )
    elif case == "tensor_in_rank_local_carrier":
        optimizer.state[parameter]["_gefen_rank_local_payload_0"] = {
            "nested": torch.ones(2)
        }
    elif case == "nested_tensor_layout":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            optimizer.state[parameter]["m_magnitude"] = torch.nested.nested_tensor(
                [torch.ones(1), torch.ones(2)]
            )
    elif case == "tensor_subclass_state":
        class StateTensor(torch.Tensor):
            pass

        optimizer.state[parameter]["m_magnitude"] = optimizer.state[parameter][
            "m_magnitude"
        ].as_subclass(StateTensor)
    elif case in {"opaque_tensor_extension", "opaque_scalar_extension"}:
        class OpaqueExtension:
            def __init__(self, payload):
                self.payload = payload

            def __eq__(self, other):
                if not isinstance(other, OpaqueExtension):
                    return False
                if torch.is_tensor(self.payload):
                    return torch.equal(self.payload, other.payload)
                return self.payload == other.payload

        payload = torch.ones(1) if case == "opaque_tensor_extension" else "metadata"
        optimizer.state[parameter]["extension"] = OpaqueExtension(payload)
    elif case == "defaultdict_factory_extension":
        hidden = torch.ones(1)
        optimizer.state[parameter]["extension"] = defaultdict(
            lambda: hidden,
            {"metadata": "value"},
        )
    elif case == "ordered_hidden_extension":
        extension = OrderedDict((("metadata", "value"),))
        extension.hidden_tensor = torch.ones(1)
        optimizer.state[parameter]["extension"] = extension
    elif case == "custom_parameter_state_mapping":
        custom_state = OrderedDict(optimizer.state[parameter])
        custom_state.hidden_tensor = torch.ones(1)
        optimizer.state[parameter] = custom_state
    elif case == "custom_top_level_state_mapping":
        custom_state = OrderedDict(optimizer.state)
        custom_state.hidden_tensor = torch.ones(1)
        optimizer.state = custom_state
    else:
        raise AssertionError("unknown rejection case: {}".format(case))
    return optimizer, parameter, None


@pytest.mark.parametrize(
    "case",
    (
        "capturable",
        "meta_parameter",
        "meta_destination",
        "mismatched_destination",
        "meta_state",
        "undeclared_tensor",
        "tensor_in_extension_container",
        "tensor_in_ordered_extension",
        "tensor_in_rank_local_carrier",
        "nested_tensor_layout",
        "tensor_subclass_state",
        "opaque_tensor_extension",
        "opaque_scalar_extension",
        "defaultdict_factory_extension",
        "ordered_hidden_extension",
        "custom_parameter_state_mapping",
        "custom_top_level_state_mapping",
    ),
)
def test_invalid_state_movement_is_rejected_before_any_live_mutation(case):
    optimizer, _, destination = _build_rejection_case(case)
    snapshot = _snapshot_exact_optimizer(optimizer)

    if case not in {"meta_destination", "mismatched_destination"}:
        assert not optimizer.optimizer_contract().capabilities.atomic_state_movement

    with pytest.raises((TypeError, ValueError, RuntimeError)):
        optimizer.move_state_(destination)

    _assert_exact_optimizer_snapshot(optimizer, snapshot)


@pytest.mark.parametrize("graph_type", ("cycle", "shared_container"))
def test_non_tree_extension_metadata_is_rejected_without_mutation(graph_type):
    optimizer, parameter, _, _ = _build_initialized("block")
    if graph_type == "cycle":
        extension = []
        extension.append(extension)
    else:
        shared = ["metadata"]
        extension = [shared, shared]
    optimizer.state[parameter]["extension"] = extension
    state_before = optimizer.state
    parameter_state_before = optimizer.state[parameter]
    items_before = tuple(parameter_state_before.items())
    codebook_before = optimizer._gefen_codebook

    assert not optimizer.optimizer_contract().capabilities.atomic_state_movement
    with pytest.raises(RuntimeError, match="not provably tensor-free metadata"):
        optimizer.move_state_()

    assert optimizer.state is state_before
    assert optimizer.state[parameter] is parameter_state_before
    assert tuple(optimizer.state[parameter]) == tuple(key for key, _ in items_before)
    for key, value in items_before:
        assert optimizer.state[parameter][key] is value
    assert optimizer._gefen_codebook is codebook_before


@pytest.mark.parametrize(
    "runtime_state",
    ("compile", "capture", "capturable_stacks", "device_counter", "sr_seed"),
)
def test_active_runtime_state_disables_movement_without_mutation(
    runtime_state, monkeypatch
):
    optimizer, _, _, _ = _build_initialized("block")
    if runtime_state == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif runtime_state == "capture":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    elif runtime_state == "capturable_stacks":
        optimizer._capt_stacks = {}
    elif runtime_state == "device_counter":
        optimizer._gefen_global_step_by_device[torch.device("cpu")] = torch.tensor(1)
    elif runtime_state == "sr_seed":
        optimizer._sr_seed_by_device[torch.device("cpu")] = torch.tensor(1)
    else:
        raise AssertionError("unknown runtime state: {}".format(runtime_state))

    assert not optimizer.optimizer_contract().capabilities.atomic_state_movement
    snapshot = _snapshot_exact_optimizer(optimizer)
    with pytest.raises(RuntimeError):
        optimizer.move_state_()
    _assert_exact_optimizer_snapshot(optimizer, snapshot)


def test_stale_finalized_binding_disables_movement_without_mutation():
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = Gefen(
        [("layer.weight", parameter)],
        fused=False,
        factored_v_2d=False,
    )
    optimizer.rebind_parameter(
        parameter,
        parameter,
        identity=ParameterIdentity("Layer.Weight", (8,)),
    )
    replacement = torch.nn.Parameter(torch.full((8,), 2.0))
    optimizer.param_groups[0]["params"][0] = replacement

    assert not optimizer.optimizer_contract().capabilities.atomic_state_movement
    snapshot = _snapshot_exact_optimizer(optimizer)
    with pytest.raises(RuntimeError, match="finalized parameter layout changed"):
        optimizer.move_state_()
    _assert_exact_optimizer_snapshot(optimizer, snapshot)


def test_movement_does_not_narrow_muons_generic_local_state_device_helper():
    meta_parameter = torch.empty(4, device="meta")

    assert GefenMuon._state_tensor_device(meta_parameter) == torch.device("meta")
    with pytest.raises(RuntimeError, match="only CPU and CUDA"):
        GefenMuon._state_move_parameter_device(meta_parameter)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("kind", _KINDS)
def test_cpu_cuda_cpu_state_round_trip_preserves_cpu_continuation(kind):
    optimizer, parameter, tensor_lr, _ = _build_initialized(kind)
    reference, reference_parameter, reference_lr, _ = _build_initialized(kind)
    initial_reference = _persistent_values(reference, reference_parameter)
    _assert_persistent_values_equal(
        _persistent_values(optimizer, parameter), initial_reference
    )
    module = _move_parameter_module(parameter, "cuda")
    cuda_grad = parameter.grad
    cuda_lr = optimizer.param_groups[0]["lr"]

    optimizer.move_state_()

    assert parameter.grad is cuda_grad
    assert optimizer.param_groups[0]["lr"] is cuda_lr is tensor_lr
    assert optimizer._gefen_codebook.device.type == "cuda"
    for key, value in optimizer.state[parameter].items():
        if key in _MOVABLE_STATE_KEYS and torch.is_tensor(value):
            assert value.device.type == "cuda"

    module.to("cpu")
    assert module.weight is parameter
    cpu_grad = parameter.grad
    optimizer.move_state_(torch.device("cpu"))
    assert parameter.grad is cpu_grad
    assert optimizer.param_groups[0]["lr"] is tensor_lr
    assert optimizer._gefen_codebook.device.type == "cpu"

    continuation_grad = torch.linspace(0.6, -0.9, parameter.numel()).reshape_as(parameter)
    parameter.grad = continuation_grad.clone()
    reference_parameter.grad = continuation_grad.clone()
    optimizer.step()
    reference.step()

    assert optimizer.param_groups[0]["lr"] is tensor_lr
    assert reference.param_groups[0]["lr"] is reference_lr
    torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
    _assert_persistent_values_equal(
        _persistent_values(optimizer, parameter),
        _persistent_values(reference, reference_parameter),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_real_cpu_to_cuda_copy_failure_is_exactly_atomic(monkeypatch):
    optimizer, parameter, _, _ = _build_initialized("block")
    module = _move_parameter_module(parameter, "cuda")
    assert optimizer._gefen_codebook.device.type == "cpu"
    assert optimizer.state[parameter]["m_codebook"].device.type == "cpu"
    candidates = _movement_candidates(optimizer, parameter)
    snapshot = _snapshot_exact_optimizer(optimizer)
    completed = _install_late_to_failure(
        monkeypatch,
        candidates,
        destination_type="cuda",
    )

    with pytest.raises(RuntimeError, match="injected late state-copy failure"):
        optimizer.move_state_()

    assert len(completed) == 3
    assert all(tensor.device.type == "cuda" for tensor in completed)
    _assert_exact_optimizer_snapshot(optimizer, snapshot)
    module.to("cpu")
