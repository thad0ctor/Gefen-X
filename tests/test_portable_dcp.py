"""Warning-strict CPU coverage for tensor-only portable DCP persistence."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.metadata import (
    BytesStorageMetadata,
    ChunkStorageMetadata,
    Metadata,
    TensorProperties,
    TensorStorageMetadata,
)

import gefen
import gefen.portable_dcp as portable_dcp
from gefen import load_portable_dcp, save_portable_dcp
from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.gefen import Gefen
from gefen.gefen_muon import GefenMuon
from gefen.portable import _decode_quantized_momentum
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_MEMBER = "rank:0"
_SINGLE_PROCESS_DCP_WARNING = (
    r"^torch\.distributed is (?:disabled, )?unavailable or uninitialized, "
    r"assuming the intent is to (?:save|load) in a single process\.$"
)
_TYPED_STORAGE_DCP_WARNING = (
    r"^TypedStorage is deprecated\. It will be removed in the future and UntypedStorage will be the only storage class\. "
    r"This should only matter to you if you are using storages directly\.  To access UntypedStorage directly, use "
    r"tensor\.untyped_storage\(\) instead of tensor\.storage\(\)$"
)


class _TensorPropertiesSubclass(TensorProperties):
    pass


@pytest.fixture(autouse=True)
def _warning_strict():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings(
            "ignore",
            message=_SINGLE_PROCESS_DCP_WARNING,
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=_TYPED_STORAGE_DCP_WARNING,
            category=UserWarning,
        )
        yield


def _limits(
    *,
    max_fragment_tensor_bytes=4 << 20,
    max_metadata_bytes=64 << 20,
):
    return PortableStateLimits(
        max_fragment_tensor_bytes=max_fragment_tensor_bytes,
        max_collective_tensor_bytes=16 << 20,
        max_collective_metadata_bytes=64 << 20,
        max_metadata_bytes=max_metadata_bytes,
    )


def _bindings(group):
    return (
        CodebookProcessGroupBinding(
            group,
            _MEMBER,
            None,
            torch.device("cpu"),
        ),
        CheckpointProcessGroupBinding(
            group,
            _MEMBER,
            None,
            torch.device("cpu"),
        ),
    )


def _finalize(optimizer, parameter, *, layout):
    group = ProcessGroupIdentity("checkpoint", (_MEMBER,))
    identity = ParameterIdentity("layer.weight", tuple(parameter.shape))
    if layout is ParameterLayout.REPLICATED:
        kind = PlacementKind.REPLICATE
        owner = None
    else:
        kind = PlacementKind.WHOLE_PARAMETER_OWNER
        owner = _MEMBER
    shard = ShardIdentity(
        identity,
        layout,
        LogicalSlice.full(identity),
        placements=(ShardPlacement("checkpoint", kind, 0, 1),),
        process_group=group,
        local_member=_MEMBER,
        owner=owner,
    )
    codebook_binding, checkpoint_binding = _bindings(group)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shard),),
        manifest=ShardingManifest((shard,)),
        codebook_process_group=codebook_binding,
    )
    return checkpoint_binding


def _plain(*, factored, deterministic):
    parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
    optimizer = Gefen(
        [("weight", parameter)],
        fused=False,
        factored_v_2d=factored,
        force_2d_period_one=True,
        deterministic=deterministic,
    )
    binding = _finalize(
        optimizer,
        parameter,
        layout=ParameterLayout.REPLICATED,
    )
    return optimizer, parameter, binding


def _muon(*, deterministic):
    parameter = torch.nn.Parameter(torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
    optimizer = GefenMuon(
        [("weight", parameter)],
        fused=False,
        sharded_mode="distributed",
        normuon=True,
        deterministic=deterministic,
    )
    binding = _finalize(
        optimizer,
        parameter,
        layout=ParameterLayout.WHOLE_PARAMETER_OWNER,
    )
    return optimizer, parameter, binding


def _optimizer_pair(variant):
    if variant == "muon-normuon":
        source, source_parameter, source_binding = _muon(deterministic=True)
        target, target_parameter, target_binding = _muon(deterministic=False)
    else:
        factored = variant == "plain-factored"
        source, source_parameter, source_binding = _plain(
            factored=factored,
            deterministic=True,
        )
        target, target_parameter, target_binding = _plain(
            factored=factored,
            deterministic=False,
        )
    return (
        source,
        source_parameter,
        source_binding,
        target,
        target_parameter,
        target_binding,
    )


def _initialize(optimizer, parameter, *, variant):
    optimizer._gefen_global_step = 4
    optimizer._gefen_codebook = torch.linspace(
        -1.0,
        1.0,
        256,
        dtype=torch.float32,
    )
    state = optimizer.state[parameter]
    state.update(
        {
            "automatic_period": 1,
            "step": 4,
            "m_codebook": torch.tensor(
                [[0], [255], [128], [64], [192], [0]],
                dtype=torch.uint8,
            ),
            "m_magnitude": torch.tensor(
                [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]],
                dtype=torch.float32,
            ),
        }
    )
    if variant == "plain-block":
        state.update(
            {
                "vmean": torch.arange(1, 7, dtype=torch.float32).reshape(6, 1),
                "vmean_step": 3,
            }
        )
    elif variant == "plain-factored":
        state.update(
            {
                "v_row": torch.tensor([2.0, 3.0], dtype=torch.float32),
                "v_col": torch.tensor([5.0, 7.0, 11.0], dtype=torch.float32),
                "factored_step": 3,
            }
        )
    elif variant == "muon-normuon":
        state.update(
            {
                "normuon_v": torch.tensor([[2.0], [3.0]], dtype=torch.float32),
                "normuon_step": 2,
            }
        )
    else:
        raise AssertionError("unknown test variant")


def _assert_loaded_state(target, target_parameter, expected, *, variant):
    assert target._gefen_global_step == 4
    assert target._deterministic is True
    assert torch.equal(
        target._gefen_codebook,
        expected["common"]["gefen_codebook"],
    )
    target_state = target.state[target_parameter]
    target_momentum = _decode_quantized_momentum(
        target._gefen_codebook,
        target_state["m_codebook"],
        target_state["m_magnitude"],
        logical_shape=tuple(target_parameter.shape),
        period=1,
        step=target_state["step"],
    )
    record = expected["parameters"]["layer.weight"]
    assert torch.equal(
        target_momentum.view(torch.int32),
        record["state"]["momentum"].view(torch.int32),
    )
    if variant == "plain-block":
        assert torch.equal(
            target_state["vmean"].reshape(2, 3),
            record["state"]["second_moment"],
        )
        assert target_state["vmean_step"] == 3
    elif variant == "plain-factored":
        assert torch.equal(target_state["v_row"], record["state"]["v_row"])
        assert torch.equal(target_state["v_col"], record["state"]["v_col"])
        assert target_state["factored_step"] == 3
    else:
        assert torch.equal(
            target_state["normuon_v"],
            record["state"]["normuon_v"],
        )
        assert target_state["normuon_step"] == 2


@pytest.mark.parametrize(
    "variant",
    ["plain-block", "plain-factored", "muon-normuon"],
)
def test_filesystem_dcp_round_trip_is_tensor_only(tmp_path, variant):
    (
        source,
        source_parameter,
        source_binding,
        target,
        target_parameter,
        target_binding,
    ) = _optimizer_pair(variant)
    _initialize(source, source_parameter, variant=variant)
    limits = _limits()
    expected = source.export_portable_state(
        checkpoint_process_group=source_binding,
        transaction_id="dcp-expected-{}".format(variant),
        limits=limits,
    )
    checkpoint = tmp_path / variant

    metadata = save_portable_dcp(
        source,
        checkpoint_process_group=source_binding,
        storage_writer=dcp.FileSystemWriter(checkpoint),
        transaction_id="dcp-save-{}".format(variant),
        limits=limits,
    )

    assert type(metadata) is Metadata
    assert metadata.state_dict_metadata
    assert all(type(value) is TensorStorageMetadata for value in metadata.state_dict_metadata.values())
    assert not any(type(value) is BytesStorageMetadata for value in metadata.state_dict_metadata.values())
    on_disk_metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    assert set(on_disk_metadata.state_dict_metadata) == set(metadata.state_dict_metadata)
    assert all(type(value) is TensorStorageMetadata for value in on_disk_metadata.state_dict_metadata.values())

    load_portable_dcp(
        target,
        checkpoint_process_group=target_binding,
        storage_reader=dcp.FileSystemReader(checkpoint),
        transaction_id="dcp-load-{}".format(variant),
        limits=limits,
    )

    _assert_loaded_state(
        target,
        target_parameter,
        expected,
        variant=variant,
    )


@pytest.mark.parametrize(
    "namespace",
    ["", " optimizer", "optimizer.", "optim\x00izer", "optim/izer", "optimé"],
)
def test_namespace_validation_precedes_dcp_io(tmp_path, namespace):
    optimizer, parameter, binding = _plain(
        factored=False,
        deterministic=True,
    )
    _initialize(optimizer, parameter, variant="plain-block")
    checkpoint = tmp_path / "invalid-namespace"

    with pytest.raises(RuntimeError, match="namespace"):
        save_portable_dcp(
            optimizer,
            checkpoint_process_group=binding,
            storage_writer=dcp.FileSystemWriter(checkpoint),
            transaction_id="dcp-invalid-namespace",
            limits=_limits(),
            namespace=namespace,
        )

    assert not (checkpoint / ".metadata").exists()


@pytest.mark.parametrize(
    ("members", "world_size", "group_zero", "match"),
    [
        (("rank:0",), 2, 0, "singleton"),
        (("rank:1", "rank:2"), 3, 1, "global rank zero"),
    ],
)
def test_dcp_runtime_rejects_unsafe_default_world_or_subgroup(
    monkeypatch,
    members,
    world_size,
    group_zero,
    match,
):
    binding = SimpleNamespace(
        identity=SimpleNamespace(ordered_members=members),
        process_group=object(),
        collective_device=torch.device("cpu"),
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    monkeypatch.setattr(
        torch.distributed,
        "get_global_rank",
        lambda _group, _rank: group_zero,
    )

    with pytest.raises(RuntimeError, match=match):
        portable_dcp._validate_dcp_runtime(binding)


def test_load_rejects_namespace_mismatch_without_target_mutation(tmp_path):
    (
        source,
        source_parameter,
        source_binding,
        target,
        _,
        target_binding,
    ) = _optimizer_pair("plain-block")
    _initialize(source, source_parameter, variant="plain-block")
    checkpoint = tmp_path / "namespace-mismatch"
    save_portable_dcp(
        source,
        checkpoint_process_group=source_binding,
        storage_writer=dcp.FileSystemWriter(checkpoint),
        transaction_id="dcp-save-namespace-mismatch",
        limits=_limits(),
        namespace="optimizer",
    )
    before = target._canonical_import_live_token()

    with pytest.raises(CheckpointException):
        load_portable_dcp(
            target,
            checkpoint_process_group=target_binding,
            storage_reader=dcp.FileSystemReader(checkpoint),
            transaction_id="dcp-load-namespace-mismatch",
            limits=_limits(),
            namespace="other",
        )

    assert target._canonical_import_live_token() == before


def _tensor_metadata(dtype, shape, *, offsets=None, sizes=None):
    shape = torch.Size(shape)
    if offsets is None:
        offsets = (0,) * len(shape)
    if sizes is None:
        sizes = shape
    return TensorStorageMetadata(
        properties=TensorProperties(dtype=dtype),
        size=shape,
        chunks=[
            ChunkStorageMetadata(
                offsets=torch.Size(offsets),
                sizes=torch.Size(sizes),
            )
        ],
    )


def _planner_metadata(case, *, namespace, limits):
    metadata_key = portable_dcp._flat_key(
        namespace,
        portable_dcp._DCP_METADATA_KEY,
    )
    payload_key = portable_dcp._flat_key(
        namespace,
        portable_dcp._payload_key(0),
    )
    entries = {metadata_key: _tensor_metadata(torch.uint8, (1,))}
    if case == "oversized-wire-metadata":
        entries[metadata_key] = _tensor_metadata(
            torch.uint8,
            (limits.max_metadata_bytes + 1,),
        )
    elif case == "oversized-payload":
        entries[payload_key] = _tensor_metadata(
            torch.uint8,
            (limits.max_fragment_tensor_bytes + 1,),
        )
    elif case == "extra-key":
        entries[portable_dcp._flat_key(namespace, "extra")] = _tensor_metadata(
            torch.uint8,
            (1,),
        )
    elif case == "byte-io":
        entries[metadata_key] = BytesStorageMetadata()
    elif case == "partial-tensor":
        entries[metadata_key] = _tensor_metadata(
            torch.uint8,
            (8,),
            offsets=(0,),
            sizes=(4,),
        )
    elif case == "partial-planner-data":
        entries[payload_key] = _tensor_metadata(torch.float32, (1,))
    elif case == "properties-subclass":
        entry = _tensor_metadata(torch.uint8, (1,))
        entry.properties = _TensorPropertiesSubclass(dtype=torch.uint8)
        entries[metadata_key] = entry
    elif case == "list-size":
        entry = _tensor_metadata(torch.uint8, (1,))
        entry.size = [1]
        entries[metadata_key] = entry
    elif case == "list-chunk-offsets":
        entry = _tensor_metadata(torch.uint8, (1,))
        entry.chunks[0].offsets = [0]
        entries[metadata_key] = entry
    else:
        raise AssertionError("unknown planner metadata case")
    planner_data = {key: tuple(key.split(".", 1)) for key in entries}
    if case == "partial-planner-data":
        planner_data.pop(payload_key)
    return Metadata(
        state_dict_metadata=entries,
        planner_data=planner_data,
    )


@pytest.mark.parametrize(
    ("case", "error_type", "match"),
    [
        ("oversized-wire-metadata", ValueError, "metadata exceeds"),
        ("oversized-payload", ValueError, "payload tensors exceed"),
        ("extra-key", ValueError, "unexpected tensor keys"),
        ("byte-io", TypeError, "full-tensor DCP metadata"),
        ("partial-tensor", ValueError, "complete unsharded tensor"),
        ("partial-planner-data", ValueError, "incompatible planner paths"),
        ("properties-subclass", TypeError, "unsupported tensor properties"),
        ("list-size", TypeError, "unsupported tensor properties"),
        ("list-chunk-offsets", ValueError, "complete unsharded tensor"),
    ],
)
def test_load_planner_rejects_untrusted_metadata_before_allocation(
    monkeypatch,
    case,
    error_type,
    match,
):
    namespace = "optimizer"
    limits = _limits(
        max_fragment_tensor_bytes=64,
        max_metadata_bytes=64,
    )._wire_limits(collective=True)
    metadata = _planner_metadata(
        case,
        namespace=namespace,
        limits=limits,
    )
    planner = portable_dcp._load_planner(namespace, limits)
    allocations = []

    def fail_if_allocated(*args, **kwargs):
        allocations.append((args, kwargs))
        raise AssertionError("untrusted DCP metadata reached tensor allocation")

    monkeypatch.setattr(torch, "empty", fail_if_allocated)

    with pytest.raises(error_type, match=match):
        planner.set_up_planner(
            {namespace: {}},
            metadata,
            is_coordinator=True,
        )

    assert allocations == []


def test_dcp_envelope_container_limit_is_symmetric_before_save():
    limits = PortableStateLimits(
        max_fragment_tensor_bytes=64,
        max_collective_tensor_bytes=64,
        max_collective_metadata_bytes=4096,
        max_metadata_bytes=4096,
        max_container_items=3,
        max_tensors=8,
    )._wire_limits(collective=True)
    plan = portable_dcp._prepare_canonical_wire_value(
        (
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
        ),
        limits,
    )

    with pytest.raises(ValueError, match="max_container_items"):
        portable_dcp._state_from_plan("optimizer", plan, limits)


def test_corrupted_tensor_payload_leaves_target_unchanged(tmp_path):
    (
        source,
        source_parameter,
        source_binding,
        target,
        target_parameter,
        target_binding,
    ) = _optimizer_pair("plain-block")
    _initialize(source, source_parameter, variant="plain-block")
    _initialize(target, target_parameter, variant="plain-block")
    target._deterministic = False
    target._gefen_global_step = 9
    limits = _limits()
    original = tmp_path / "original"
    corrupted = tmp_path / "corrupted"
    save_portable_dcp(
        source,
        checkpoint_process_group=source_binding,
        storage_writer=dcp.FileSystemWriter(original),
        transaction_id="dcp-save-corruption-source",
        limits=limits,
    )

    wire_limits = limits._wire_limits(collective=True)
    state = {"optimizer": {}}
    dcp.load(
        state,
        storage_reader=dcp.FileSystemReader(original),
        planner=portable_dcp._load_planner("optimizer", wire_limits),
        process_group=None,
    )
    payload_keys = sorted(key for key in state["optimizer"] if key.startswith(portable_dcp._DCP_PAYLOAD_PREFIX))
    assert payload_keys
    payload = next(state["optimizer"][key] for key in payload_keys if state["optimizer"][key].numel())
    raw = payload.view(torch.uint8).reshape(-1)
    raw[0] = int(raw[0]) ^ 1
    dcp.save(
        state,
        storage_writer=dcp.FileSystemWriter(corrupted),
        process_group=None,
    )
    before = target._canonical_import_live_token()

    with pytest.raises(RuntimeError, match="invalid digest"):
        load_portable_dcp(
            target,
            checkpoint_process_group=target_binding,
            storage_reader=dcp.FileSystemReader(corrupted),
            transaction_id="dcp-load-corrupted-payload",
            limits=limits,
        )

    assert target._canonical_import_live_token() == before


def test_portable_dcp_helpers_are_public_exports():
    assert gefen.save_portable_dcp is portable_dcp.save_portable_dcp
    assert gefen.load_portable_dcp is portable_dcp.load_portable_dcp
    assert save_portable_dcp is portable_dcp.save_portable_dcp
    assert load_portable_dcp is portable_dcp.load_portable_dcp
    assert "save_portable_dcp" in gefen.__all__
    assert "load_portable_dcp" in gefen.__all__
    assert portable_dcp.__all__ == [
        "load_portable_dcp",
        "save_portable_dcp",
    ]
