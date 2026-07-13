"""Shared deep fail-before-mutation snapshot helpers for optimizer tests.

``deep_state_snapshot`` captures both the container identities and bitwise
clones of every tensor reachable from an optimizer (per-parameter state,
codebook tensors and per-device caches, counters, param-group contents). A
regression that publishes staged data by mutating live containers in place —
``state[param].update(...)``, ``copy_()`` into an existing state tensor, an
in-place ``group['params']`` element swap, or an in-place option edit —
preserves every top-level object identity and is only caught by comparing the
live values against these clones. ``assert_deep_state_snapshot`` therefore
checks identity AND bitwise value equality together.
"""

import copy

import torch


def _cloned(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if type(value) is dict:
        return {key: _cloned(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_cloned(item) for item in value)
    return copy.deepcopy(value)


def _nested_equal(live, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(live)
        assert torch.equal(live, expected)
        return
    assert type(live) is type(expected)
    if isinstance(expected, dict):
        assert set(live) == set(expected)
        for key in expected:
            _nested_equal(live[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(live) == len(expected)
        for live_item, expected_item in zip(live, expected):
            _nested_equal(live_item, expected_item)
    else:
        assert live == expected


def _tensor_value_pairs(optimizer):
    """Every tensor reachable from ``optimizer.__dict__`` with a clone of it."""

    pairs = []
    visited = set()

    def visit(value):
        if torch.is_tensor(value):
            if id(value) not in visited:
                visited.add(id(value))
                pairs.append((value, value.detach().clone()))
        elif isinstance(value, dict):
            if id(value) in visited:
                return
            visited.add(id(value))
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in visited:
                return
            visited.add(id(value))
            for item in value:
                visit(item)

    for value in optimizer.__dict__.values():
        visit(value)
    return tuple(pairs)


def deep_state_snapshot(optimizer):
    return {
        "attributes": optimizer.__dict__.copy(),
        "state": optimizer.state,
        "state_items": tuple(
            (parameter, state, _cloned(dict(state)))
            for parameter, state in optimizer.state.items()
        ),
        "param_groups": optimizer.param_groups,
        "groups": tuple(
            (
                group,
                group["params"],
                tuple(group["params"]),
                _cloned({key: value for key, value in group.items() if key != "params"}),
            )
            for group in optimizer.param_groups
        ),
        "tensors": _tensor_value_pairs(optimizer),
    }


def assert_deep_state_snapshot(optimizer, snapshot):
    assert optimizer.__dict__.keys() == snapshot["attributes"].keys()
    for name, value in snapshot["attributes"].items():
        assert optimizer.__dict__[name] is value
    assert optimizer.state is snapshot["state"]
    assert optimizer.param_groups is snapshot["param_groups"]
    assert len(optimizer.state) == len(snapshot["state_items"])
    for live_parameter, (parameter, state, expected) in zip(
        optimizer.state, snapshot["state_items"]
    ):
        assert live_parameter is parameter
        assert optimizer.state[parameter] is state
        _nested_equal(dict(state), expected)
    assert len(optimizer.param_groups) == len(snapshot["groups"])
    for live_group, (group, params, expected_params, expected_options) in zip(
        optimizer.param_groups, snapshot["groups"]
    ):
        assert live_group is group
        assert group["params"] is params
        assert len(params) == len(expected_params)
        for live_param, expected_param in zip(params, expected_params):
            assert live_param is expected_param
        _nested_equal(
            {key: value for key, value in group.items() if key != "params"},
            expected_options,
        )
    for tensor, expected in snapshot["tensors"]:
        assert torch.equal(tensor, expected)
