"""Fail-before-mutation coverage for native Gefen checkpoint restores."""

import copy

import pytest
import torch

from gefen import Gefen, GefenMuon


def _build_optimizer(kind, seed):
    generator = torch.Generator().manual_seed(seed)
    param = torch.nn.Parameter(torch.randn(8, 8, generator=generator))
    named = [("layer.weight", param)]
    if kind == "block":
        optimizer = Gefen(named, lr=1e-3, fused=False, factored_v_2d=False)
    elif kind == "factored":
        optimizer = Gefen(named, lr=1e-3, fused=False, factored_v_2d=True)
    elif kind == "muon":
        optimizer = GefenMuon(named, lr=1e-3, fused=False, normuon=False)
    elif kind == "normuon":
        optimizer = GefenMuon(named, lr=1e-3, fused=False, normuon=True)
    else:
        raise AssertionError("unknown optimizer kind: {}".format(kind))
    param.grad = torch.randn(param.shape, generator=generator)
    optimizer.step()
    optimizer.zero_grad()
    return optimizer, param


def _seed_live_runtime_state(optimizer, param):
    device = torch.device("cpu")
    pstate = optimizer.state[param]
    pstate["_capt_stack"] = torch.tensor([11.0])
    pstate["_capt_row"] = 3
    pstate["_capt_scalars"] = torch.tensor([12.0])
    optimizer._capt_stacks = {device: ({"rows": [param]},)}
    optimizer._gefen_codebook_by_device[device] = torch.tensor([13.0])
    optimizer._gefen_codebook_lut_by_device[device] = torch.tensor([14.0])
    optimizer._sr_seed_by_device[device] = torch.tensor(15, dtype=torch.int64)
    optimizer._gefen_global_step_by_device[device] = torch.tensor(16, dtype=torch.int64)
    optimizer._static_mark_sig = ("sentinel", 17)


def _snapshot_live_optimizer(optimizer):
    state_refs = dict(optimizer.state)
    state_values = {}
    state_tensor_refs = {}
    for param, pstate in state_refs.items():
        state_values[param] = {}
        state_tensor_refs[param] = {}
        for key, value in pstate.items():
            if torch.is_tensor(value):
                state_values[param][key] = value.detach().clone()
                state_tensor_refs[param][key] = (value, value._version)
            else:
                state_values[param][key] = copy.deepcopy(value)

    cache_attrs = (
        "_gefen_codebook_by_device",
        "_gefen_codebook_lut_by_device",
        "_sr_seed_by_device",
        "_gefen_global_step_by_device",
    )
    caches = {}
    for attr in cache_attrs:
        cache = getattr(optimizer, attr)
        caches[attr] = (
            cache,
            {
                key: (value, value.detach().clone(), value._version)
                for key, value in cache.items()
            },
        )
    return {
        "state": optimizer.state,
        "state_refs": state_refs,
        "state_values": state_values,
        "state_tensor_refs": state_tensor_refs,
        "param_groups": optimizer.param_groups,
        "group_refs": tuple(optimizer.param_groups),
        "group_param_refs": tuple(group["params"] for group in optimizer.param_groups),
        "group_values": tuple(
            {
                key: copy.deepcopy(value)
                for key, value in group.items()
                if key != "params"
            }
            for group in optimizer.param_groups
        ),
        "defaults": optimizer.defaults,
        "defaults_value": copy.deepcopy(optimizer.defaults),
        "param_names": optimizer._param_names,
        "param_names_value": dict(optimizer._param_names),
        "global_step": optimizer._gefen_global_step,
        "codebook": optimizer._gefen_codebook,
        "capt_stacks": optimizer._capt_stacks,
        "static_mark_sig": optimizer._static_mark_sig,
        "caches": caches,
    }


def _assert_live_optimizer_unchanged(optimizer, snapshot):
    assert optimizer.state is snapshot["state"]
    assert optimizer.param_groups is snapshot["param_groups"]
    assert optimizer.defaults is snapshot["defaults"]
    assert optimizer.defaults == snapshot["defaults_value"]
    assert optimizer._param_names is snapshot["param_names"]
    assert optimizer._param_names == snapshot["param_names_value"]
    assert optimizer._gefen_global_step is snapshot["global_step"]
    assert optimizer._gefen_codebook is snapshot["codebook"]
    assert optimizer._capt_stacks is snapshot["capt_stacks"]
    assert optimizer._static_mark_sig is snapshot["static_mark_sig"]

    assert tuple(optimizer.param_groups) == snapshot["group_refs"]
    for group, group_ref, params_ref, values in zip(
        optimizer.param_groups,
        snapshot["group_refs"],
        snapshot["group_param_refs"],
        snapshot["group_values"],
    ):
        assert group is group_ref
        assert group["params"] is params_ref
        assert {key: value for key, value in group.items() if key != "params"} == values

    assert set(optimizer.state) == set(snapshot["state_refs"])
    for param, pstate_ref in snapshot["state_refs"].items():
        pstate = optimizer.state[param]
        assert pstate is pstate_ref
        assert set(pstate) == set(snapshot["state_values"][param])
        for key, expected in snapshot["state_values"][param].items():
            value = pstate[key]
            if torch.is_tensor(value):
                tensor_ref, version = snapshot["state_tensor_refs"][param][key]
                assert value is tensor_ref
                assert value._version == version
                assert torch.equal(value, expected)
            else:
                assert value == expected

    for attr, (cache_ref, entries) in snapshot["caches"].items():
        cache = getattr(optimizer, attr)
        assert cache is cache_ref
        assert set(cache) == set(entries)
        for key, (tensor_ref, expected, version) in entries.items():
            assert cache[key] is tensor_ref
            assert cache[key]._version == version
            assert torch.equal(cache[key], expected)


def _assert_nested_equal(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, dict):
        assert set(actual) == set(expected)
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_nested_equal(left, right)
    elif torch.is_tensor(actual):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    else:
        assert actual == expected


def _set_all_codebook_copies(checkpoint, codebook):
    checkpoint["gefen_codebook"] = codebook
    for group in checkpoint["param_groups"]:
        metadata = group.get("_gefen_checkpoint_metadata")
        if metadata is not None:
            metadata["codebook"] = codebook


def _corrupt_automatic_period(checkpoint):
    next(iter(checkpoint["state"].values()))["automatic_period"] = 0


def _corrupt_momentum_indices(checkpoint):
    pstate = next(iter(checkpoint["state"].values()))
    pstate["m_codebook"] = pstate["m_codebook"].to(torch.float32)


def _corrupt_momentum_magnitude(checkpoint):
    pstate = next(iter(checkpoint["state"].values()))
    pstate["m_magnitude"] = torch.full_like(pstate["m_magnitude"], -1.0)


def _remove_quantized_momentum(checkpoint):
    pstate = next(iter(checkpoint["state"].values()))
    pstate.pop("m_codebook")
    pstate.pop("m_magnitude")


def _corrupt_block_second_moment(checkpoint):
    pstate = next(iter(checkpoint["state"].values()))
    pstate["vmean"] = torch.full_like(pstate["vmean"], float("nan"))


def _corrupt_factored_second_moment(checkpoint):
    next(iter(checkpoint["state"].values())).pop("v_col")


def _corrupt_normuon_second_moment(checkpoint):
    pstate = next(iter(checkpoint["state"].values()))
    pstate["normuon_v"] = torch.full_like(pstate["normuon_v"], -1.0)


def _corrupt_codebook(checkpoint):
    codebook = torch.linspace(-1.0, 1.0, 255, dtype=torch.float32)
    _set_all_codebook_copies(checkpoint, codebook)


@pytest.mark.parametrize(
    ("kind", "counter_key", "invalid_value"),
    [
        ("block", "step", torch.tensor([1.0, 2.0])),
        ("block", "step", torch.tensor(1.5)),
        ("block", "step", torch.tensor(True)),
        ("block", "step", torch.tensor(-1.0)),
        ("block", "step", torch.tensor(float("nan"))),
        ("block", "vmean_step", torch.tensor([1.0, 2.0])),
        ("factored", "factored_step", torch.tensor([1.0, 2.0])),
        ("muon", "step", torch.tensor([1.0, 2.0])),
        ("normuon", "normuon_step", torch.tensor([1.0, 2.0])),
    ],
)
def test_native_counter_normalization_failure_is_fail_before_mutation(
    kind, counter_key, invalid_value
):
    source, _ = _build_optimizer(kind, seed=1)
    target, target_param = _build_optimizer(kind, seed=2)
    checkpoint = copy.deepcopy(source.state_dict())
    next(iter(checkpoint["state"].values()))[counter_key] = invalid_value
    checkpoint_before = copy.deepcopy(checkpoint)
    _seed_live_runtime_state(target, target_param)
    snapshot = _snapshot_live_optimizer(target)
    post_calls = []
    target.register_load_state_dict_post_hook(lambda optimizer: post_calls.append(optimizer))

    with pytest.raises((RuntimeError, ValueError)):
        target.load_state_dict(checkpoint)

    assert post_calls == []
    _assert_live_optimizer_unchanged(target, snapshot)
    _assert_nested_equal(checkpoint, checkpoint_before)


@pytest.mark.parametrize("kind", ["block", "muon"])
def test_native_base_layout_failure_preserves_live_caches_and_objects(kind):
    source, _ = _build_optimizer(kind, seed=3)
    target, target_param = _build_optimizer(kind, seed=4)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["param_groups"].append(copy.deepcopy(checkpoint["param_groups"][0]))
    checkpoint_before = copy.deepcopy(checkpoint)
    _seed_live_runtime_state(target, target_param)
    snapshot = _snapshot_live_optimizer(target)

    with pytest.raises(ValueError, match="different number of parameter groups"):
        target.load_state_dict(checkpoint)

    _assert_live_optimizer_unchanged(target, snapshot)
    _assert_nested_equal(checkpoint, checkpoint_before)


@pytest.mark.parametrize(
    ("kind", "corrupt"),
    [
        ("block", _corrupt_automatic_period),
        ("block", _corrupt_momentum_indices),
        ("block", _corrupt_momentum_magnitude),
        ("block", _remove_quantized_momentum),
        ("block", _corrupt_block_second_moment),
        ("block", _corrupt_codebook),
        ("factored", _corrupt_factored_second_moment),
        ("normuon", _corrupt_normuon_second_moment),
    ],
    ids=(
        "automatic-period",
        "momentum-indices",
        "momentum-magnitude",
        "missing-momentum",
        "block-second-moment",
        "codebook",
        "factored-second-moment",
        "normuon-second-moment",
    ),
)
def test_invalid_native_state_is_rejected_before_live_mutation(kind, corrupt):
    source, _ = _build_optimizer(kind, seed=9)
    target, target_param = _build_optimizer(kind, seed=10)
    checkpoint = copy.deepcopy(source.state_dict())
    corrupt(checkpoint)
    checkpoint_before = copy.deepcopy(checkpoint)
    _seed_live_runtime_state(target, target_param)
    snapshot = _snapshot_live_optimizer(target)

    with pytest.raises(ValueError):
        target.load_state_dict(checkpoint)

    _assert_live_optimizer_unchanged(target, snapshot)
    _assert_nested_equal(checkpoint, checkpoint_before)


def test_native_load_hooks_run_once_around_transaction():
    source, _ = _build_optimizer("block", seed=5)
    target, _ = _build_optimizer("block", seed=6)
    checkpoint = copy.deepcopy(source.state_dict())
    calls = []

    def replace_with_invalid_counter(optimizer, state_dict):
        calls.append(("pre", optimizer))
        replacement = copy.deepcopy(state_dict)
        next(iter(replacement["state"].values()))["step"] = torch.tensor([1.0, 2.0])
        return replacement

    target.register_load_state_dict_pre_hook(replace_with_invalid_counter)
    target.register_load_state_dict_post_hook(
        lambda optimizer: calls.append(("post", optimizer))
    )
    snapshot = _snapshot_live_optimizer(target)

    with pytest.raises((RuntimeError, ValueError)):
        target.load_state_dict(checkpoint)

    assert calls == [("pre", target)]
    _assert_live_optimizer_unchanged(target, snapshot)


def test_successful_native_load_hooks_run_once_and_commit():
    source, source_param = _build_optimizer("factored", seed=7)
    source_param.grad = torch.full_like(source_param, 0.25)
    source.step()
    source.zero_grad()
    target, target_param = _build_optimizer("factored", seed=8)
    checkpoint = copy.deepcopy(source.state_dict())
    calls = []
    target.register_load_state_dict_pre_hook(
        lambda optimizer, state_dict: calls.append(("pre", optimizer))
    )
    target.register_load_state_dict_post_hook(
        lambda optimizer: calls.append(("post", optimizer))
    )
    old_state = target.state
    old_defaults = target.defaults

    target.load_state_dict(checkpoint)

    assert calls == [("pre", target), ("post", target)]
    assert target.state is not old_state
    assert target.defaults is old_defaults
    assert target._gefen_global_step == source._gefen_global_step == 2
    assert target.state[target_param]["step"] == source.state[source_param]["step"] == 2
    assert torch.equal(target._gefen_codebook, source._gefen_codebook)


def test_rejected_late_parameter_corruption_preserves_bit_exact_continuation():
    def build(seed):
        generator = torch.Generator().manual_seed(seed)
        params = [
            torch.nn.Parameter(torch.randn(4, 8, generator=generator)),
            torch.nn.Parameter(torch.randn(8, 4, generator=generator)),
        ]
        optimizer = Gefen(
            [("first.weight", params[0]), ("second.weight", params[1])],
            lr=1e-3,
            fused=False,
            factored_v_2d=False,
        )
        for param in params:
            param.grad = torch.randn(param.shape, generator=generator)
        optimizer.step()
        optimizer.zero_grad()
        return optimizer, params

    source, _ = build(11)
    target, target_params = build(12)
    control, control_params = build(12)
    checkpoint = copy.deepcopy(source.state_dict())
    list(checkpoint["state"].values())[-1]["step"] = torch.tensor([1.0, 2.0])
    snapshot = _snapshot_live_optimizer(target)

    with pytest.raises(ValueError):
        target.load_state_dict(checkpoint)

    _assert_live_optimizer_unchanged(target, snapshot)
    generator = torch.Generator().manual_seed(13)
    next_grads = [
        torch.randn(param.shape, generator=generator) for param in target_params
    ]
    for params, optimizer in (
        (target_params, target),
        (control_params, control),
    ):
        for param, grad in zip(params, next_grads):
            param.grad = grad.clone()
        optimizer.step()
        optimizer.zero_grad()

    for target_param, control_param in zip(target_params, control_params):
        assert torch.equal(target_param, control_param)
    _assert_nested_equal(target.state_dict(), control.state_dict())
