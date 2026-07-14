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

The staged post_sharding transaction also publishes two per-parameter registry
dicts on each child — ``_param_names`` (parameter -> compatibility FQN string)
and ``_gefen_shard_bindings`` (parameter -> ``ShardIdentity``) — by rebuilding
them on a staged shadow and swapping the shadow in wholesale. Those dicts keep
their live object identity across a transaction, so a regression that wrote
``self._param_names[target] = ...`` on the LIVE child before a later child
raised would leave a half-populated registry that top-level identity checks
miss. ``deep_state_snapshot`` therefore also records the key set (by parameter
identity) and cloned values of these registries, and
``assert_deep_state_snapshot`` value-compares them after the transaction.
"""

import copy

import torch


# Per-parameter registry dicts each Gefen/GefenMuon child rebuilds on a staged
# shadow during a post_sharding transaction. Keys are live parameters (compared
# by identity); values are FQN strings / ``ShardIdentity`` objects (compared by
# value). Absent on optimizer types that do not stage these (handled below).
_CHILD_REGISTRY_ATTRS = ("_param_names", "_gefen_shard_bindings")


def _cloned(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if type(value) is dict:
        return {key: _cloned(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return type(value)(_cloned(item) for item in value)
    return copy.deepcopy(value)


def _tensors_bitwise_equal(live, expected):
    # ``torch.equal`` reports +0.0/-0.0 as equal and every NaN as unequal, so it
    # can miss a failed transaction that flips a sign bit and can spuriously fail
    # on unchanged NaN payloads. A fail-before-mutation snapshot needs exact byte
    # preservation, so compare dtype/layout/shape and then the raw bytes.
    if (
        live.dtype != expected.dtype
        or live.layout != expected.layout
        or tuple(live.shape) != tuple(expected.shape)
    ):
        return False
    live_bytes = live.detach().contiguous().view(torch.uint8)
    expected_bytes = expected.detach().contiguous().view(torch.uint8)
    return bool(torch.equal(live_bytes, expected_bytes))


def _nested_equal(live, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(live)
        assert _tensors_bitwise_equal(live, expected)
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


def _registry_snapshot(optimizer):
    """Cloned contents of each per-parameter registry dict that staging swaps.

    Returns ``name -> ((param, cloned_value), ...)`` for every registry that is
    present as a ``dict`` on ``optimizer``; registries absent on this optimizer
    type are skipped so the helper stays usable across optimizer kinds. Keys are
    kept by reference for identity comparison; values are cloned so a later
    in-place value mutation cannot alias the snapshot.
    """

    registries = {}
    for name in _CHILD_REGISTRY_ATTRS:
        registry = getattr(optimizer, name, None)
        if type(registry) is not dict:
            continue
        registries[name] = tuple(
            (key, _cloned(value)) for key, value in registry.items()
        )
    return registries


def deep_state_snapshot(optimizer):
    return {
        "attributes": optimizer.__dict__.copy(),
        "registries": _registry_snapshot(optimizer),
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
        assert _tensors_bitwise_equal(tensor, expected)
    for name, expected_entries in snapshot["registries"].items():
        live = getattr(optimizer, name, None)
        assert type(live) is dict
        live_entries = tuple(live.items())
        assert len(live_entries) == len(expected_entries)
        for (live_key, live_value), (expected_key, expected_value) in zip(
            live_entries, expected_entries
        ):
            assert live_key is expected_key
            _nested_equal(live_value, expected_value)
