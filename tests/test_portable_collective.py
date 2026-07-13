from datetime import timedelta
import hashlib
import multiprocessing as mp
import os
import tempfile

import pytest
import torch
import torch.distributed as dist

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.contracts import ProcessGroupIdentity
import gefen.portable_collective as portable_collective
from gefen.portable_collective import (
    _CanonicalCollectiveError,
    _collective_unanimous_status,
    _collective_visit_canonical_fragments,
)
from gefen.portable_wire import _CanonicalWireLimits, _PreparedCanonicalWireValue


def _limits(**overrides):
    values = {
        "max_fragment_tensor_bytes": 1 << 20,
        "max_collective_tensor_bytes": 4 << 20,
        "max_collective_metadata_bytes": 4 << 20,
        "chunk_bytes": 5,
        "max_members": 8,
        "max_metadata_bytes": 1 << 20,
        "max_tree_nodes": 1000,
        "max_tree_depth": 16,
        "max_container_items": 1000,
        "max_string_bytes": 4096,
        "max_integer_bytes": 128,
        "max_tensors": 128,
        "max_tensor_rank": 8,
        "diagnostic_bytes": 512,
    }
    values.update(overrides)
    return _CanonicalWireLimits(**values)


def _singleton_binding():
    identity = ProcessGroupIdentity("checkpoint", ("worker:solo",))
    return CheckpointProcessGroupBinding(
        identity,
        "worker:solo",
        None,
        torch.device("cpu"),
    )


def _context(value=b"context"):
    return hashlib.sha256(value).digest()


def test_singleton_visits_owned_digest_verified_value_without_distributed_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("object collectives and distributed calls are forbidden")

    monkeypatch.setattr(dist, "all_gather_object", forbidden)
    monkeypatch.setattr(dist, "broadcast_object_list", forbidden)
    monkeypatch.setattr(dist, "all_gather", forbidden)
    monkeypatch.setattr(dist, "broadcast", forbidden)
    original = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    plans = []
    consumed = []

    def validate(member, prepared):
        assert member == "worker:solo"
        assert type(prepared) is _PreparedCanonicalWireValue
        plans.append((member, prepared.total_tensor_bytes, len(prepared.tensor_specs)))

    def consume(member, value):
        consumed.append((member, value))

    _collective_visit_canonical_fragments(
        _singleton_binding(),
        {"flag": True, "payload": [original]},
        operation="export",
        transaction_id="transaction-1",
        context_digest=_context(),
        limits=_limits(),
        validate_plan=validate,
        consume=consume,
    )

    assert plans == [("worker:solo", original.numel() * original.element_size(), 1)]
    assert [member for member, _ in consumed] == ["worker:solo"]
    result = consumed[0][1]
    torch.testing.assert_close(result["payload"][0], original, rtol=0, atol=0)
    assert result["payload"][0] is not original
    assert result["payload"][0].device.type == "cpu"
    assert result["payload"][0].is_contiguous()
    original.add_(100)
    torch.testing.assert_close(
        result["payload"][0],
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        rtol=0,
        atol=0,
    )


def test_singleton_transfers_scalar_empty_complex_and_small_chunk_payloads():
    value = {
        "bool_scalar": torch.tensor(True),
        "complex": torch.tensor([1 + 2j, -3 + 4j], dtype=torch.complex64),
        "empty": torch.empty((2, 0, 3), dtype=torch.uint8),
        "integer_scalar": torch.tensor(-7, dtype=torch.int64),
    }
    consumed = []
    _collective_visit_canonical_fragments(
        _singleton_binding(),
        value,
        operation="export",
        transaction_id="transaction-dtypes",
        context_digest=_context(),
        limits=_limits(chunk_bytes=3),
        consume=lambda member, result: consumed.append(result),
    )

    assert len(consumed) == 1
    for name, expected in value.items():
        torch.testing.assert_close(consumed[0][name], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("fragment", "validate_plan", "consume", "message"),
    [
        (object(), None, None, "unsupported canonical wire type"),
        (None, lambda member, plan: (_ for _ in ()).throw(ValueError("invalid plan")), None, "invalid plan"),
        (None, None, lambda member, value: (_ for _ in ()).throw(RuntimeError("cannot stage")), "cannot stage"),
    ],
)
def test_singleton_converts_local_preparation_and_callback_failures_to_collective_rejections(
    fragment,
    validate_plan,
    consume,
    message,
):
    with pytest.raises(_CanonicalCollectiveError, match=message) as caught:
        _collective_visit_canonical_fragments(
            _singleton_binding(),
            fragment,
            operation="import",
            transaction_id="transaction-2",
            context_digest=_context(),
            limits=_limits(),
            validate_plan=validate_plan,
            consume=consume,
        )
    assert "semantic-coordinate[0]" in str(caught.value)


def test_singleton_does_not_invoke_consumer_after_metadata_corruption(monkeypatch):
    original = portable_collective._metadata_from_tensor
    consumed = []

    def corrupt(value):
        metadata = bytearray(original(value))
        metadata[0] ^= 1
        return bytes(metadata)

    monkeypatch.setattr(portable_collective, "_metadata_from_tensor", corrupt)
    with pytest.raises(_CanonicalCollectiveError, match="metadata digest"):
        _collective_visit_canonical_fragments(
            _singleton_binding(),
            torch.arange(4, dtype=torch.float32),
            operation="export",
            transaction_id="transaction-corrupt",
            context_digest=_context(),
            limits=_limits(),
            consume=lambda member, value: consumed.append(value),
        )
    assert consumed == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"operation": ""}, "operation"),
        ({"transaction_id": " bad"}, "transaction_id"),
        ({"context_digest": b"short"}, "context_digest"),
        ({"validate_plan": object()}, "validate_plan"),
        ({"consume": object()}, "consume"),
    ],
)
def test_singleton_strict_invocation_fields_fail_through_fixed_status_exchange(changes, message):
    arguments = {
        "operation": "export",
        "transaction_id": "transaction-strict",
        "context_digest": _context(),
        "limits": _limits(),
    }
    arguments.update(changes)
    with pytest.raises(_CanonicalCollectiveError, match=message):
        _collective_visit_canonical_fragments(
            _singleton_binding(),
            None,
            **arguments,
        )


def test_singleton_rejects_limits_that_do_not_fit_signed_header_fields():
    too_large = (1 << 63)
    limits = _limits(
        max_fragment_tensor_bytes=too_large,
        max_collective_tensor_bytes=too_large,
    )
    with pytest.raises(_CanonicalCollectiveError, match="signed int64"):
        _collective_visit_canonical_fragments(
            _singleton_binding(),
            None,
            operation="export",
            transaction_id="transaction-overflow",
            context_digest=_context(),
            limits=limits,
        )


def test_unanimous_status_uses_only_fixed_exchanges_and_reports_local_error(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no distributed collective is allowed for a singleton")

    monkeypatch.setattr(dist, "all_gather", forbidden)
    monkeypatch.setattr(dist, "broadcast", forbidden)
    monkeypatch.setattr(portable_collective, "_prepare_canonical_wire_value", forbidden)
    _collective_unanimous_status(
        _singleton_binding(),
        None,
        operation="import-ready",
        transaction_id="transaction-ready",
        context_digest=_context(),
        limits=_limits(),
    )
    with pytest.raises(_CanonicalCollectiveError, match="stale target"):
        _collective_unanimous_status(
            _singleton_binding(),
            ValueError("stale target"),
            operation="import-fresh",
            transaction_id="transaction-ready",
            context_digest=_context(),
            limits=_limits(),
        )


def _capture_collective_error(callable_object):
    try:
        callable_object()
    except _CanonicalCollectiveError as exc:
        return str(exc)
    return None


def _distributed_worker(rank, init_file, queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=60),
        )
        members = ("worker:left", "worker:right")
        binding = CheckpointProcessGroupBinding(
            ProcessGroupIdentity("checkpoint", members),
            members[rank],
            dist.group.WORLD,
            torch.device("cpu"),
        )
        original_all_gather_object = dist.all_gather_object
        original_broadcast_object_list = dist.broadcast_object_list

        def forbidden(*args, **kwargs):
            raise AssertionError("object collectives are forbidden")

        dist.all_gather_object = forbidden
        dist.broadcast_object_list = forbidden
        successes = []
        validation_order = []

        _collective_visit_canonical_fragments(
            binding,
            {
                "coordinate": rank,
                "payload": torch.arange(4, dtype=torch.float32) + rank * 10,
            },
            operation="export",
            transaction_id="distributed-success",
            context_digest=_context(),
            limits=_limits(chunk_bytes=7),
            validate_plan=lambda member, plan: validation_order.append((member, len(plan.tensor_specs))),
            consume=lambda member, value: successes.append(
                (member, value["coordinate"], value["payload"].tolist())
            ),
        )
        dist.barrier()

        context_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                None,
                operation="export",
                transaction_id="divergent-context",
                context_digest=_context(b"left" if rank == 0 else b"right"),
                limits=_limits(),
            )
        )
        dist.barrier()

        operation_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                None,
                operation="export" if rank == 0 else "import",
                transaction_id="divergent-operation",
                context_digest=_context(),
                limits=_limits(),
            )
        )
        dist.barrier()

        transaction_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                None,
                operation="export",
                transaction_id="transaction-left" if rank == 0 else "transaction-right",
                context_digest=_context(),
                limits=_limits(),
            )
        )
        dist.barrier()

        limits_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                None,
                operation="export",
                transaction_id="divergent-limits",
                context_digest=_context(),
                limits=_limits(chunk_bytes=5 + rank),
            )
        )
        dist.barrier()

        wrong_binding = CheckpointProcessGroupBinding(
            ProcessGroupIdentity("checkpoint", members),
            members[1] if rank == 0 else members[rank],
            dist.group.WORLD,
            torch.device("cpu"),
        )
        coordinate_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                wrong_binding,
                None,
                operation="export",
                transaction_id="coordinate-mismatch",
                context_digest=_context(),
                limits=_limits(),
            )
        )
        dist.barrier()

        callback_presence_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                None,
                operation="import",
                transaction_id="callback-presence",
                context_digest=_context(),
                limits=_limits(),
                consume=None if rank == 0 else (lambda member, value: None),
            )
        )
        dist.barrier()

        prepare_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                object() if rank == 0 else None,
                operation="export",
                transaction_id="prepare-failure",
                context_digest=_context(),
                limits=_limits(),
            )
        )
        dist.barrier()

        def validate(member, plan):
            if rank == 1 and member == members[0]:
                raise ValueError("receiver rejected plan")

        validate_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                torch.arange(2, dtype=torch.float32),
                operation="import",
                transaction_id="validate-failure",
                context_digest=_context(),
                limits=_limits(),
                validate_plan=validate,
            )
        )
        dist.barrier()

        def consume(member, value):
            if rank == 0 and member == members[1]:
                raise RuntimeError("receiver could not stage value")

        consume_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                torch.arange(2, dtype=torch.float32),
                operation="import",
                transaction_id="consume-failure",
                context_digest=_context(),
                limits=_limits(),
                consume=consume,
            )
        )
        dist.barrier()

        budget_error = _capture_collective_error(
            lambda: _collective_visit_canonical_fragments(
                binding,
                torch.arange(4, dtype=torch.float32),
                operation="export",
                transaction_id="aggregate-budget",
                context_digest=_context(),
                limits=_limits(
                    max_fragment_tensor_bytes=16,
                    max_collective_tensor_bytes=24,
                ),
            )
        )
        dist.barrier()

        original_broadcast = portable_collective._broadcast_tensor
        broadcast_count = 0

        def corrupt_broadcast(bound_binding, value, *, source, member_count):
            nonlocal broadcast_count
            broadcast_count += 1
            if rank == 0 and source == 0 and broadcast_count == 2:
                value[0] ^= 1
            return original_broadcast(
                bound_binding,
                value,
                source=source,
                member_count=member_count,
            )

        portable_collective._broadcast_tensor = corrupt_broadcast
        try:
            corruption_error = _capture_collective_error(
                lambda: _collective_visit_canonical_fragments(
                    binding,
                    torch.arange(2, dtype=torch.float32),
                    operation="export",
                    transaction_id="payload-corruption",
                    context_digest=_context(),
                    limits=_limits(),
                )
            )
        finally:
            portable_collective._broadcast_tensor = original_broadcast
        dist.barrier()

        status_error = _capture_collective_error(
            lambda: _collective_unanimous_status(
                binding,
                ValueError("target changed") if rank == 1 else None,
                operation="import-fresh",
                transaction_id="status-helper",
                context_digest=_context(),
                limits=_limits(),
            )
        )
        dist.barrier()

        dist.all_gather_object = original_all_gather_object
        dist.broadcast_object_list = original_broadcast_object_list
        queue.put(
            {
                "rank": rank,
                "successes": successes,
                "validation_order": validation_order,
                "context_error": context_error,
                "operation_error": operation_error,
                "transaction_error": transaction_error,
                "limits_error": limits_error,
                "coordinate_error": coordinate_error,
                "callback_presence_error": callback_presence_error,
                "prepare_error": prepare_error,
                "validate_error": validate_error,
                "consume_error": consume_error,
                "budget_error": budget_error,
                "corruption_error": corruption_error,
                "status_error": status_error,
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "fatal_error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_workers():
    context = mp.get_context("spawn")
    queue = context.Queue()
    fd, init_file = tempfile.mkstemp(prefix="gefen-portable-collective-")
    os.close(fd)
    os.unlink(init_file)
    processes = [
        context.Process(target=_distributed_worker, args=(rank, init_file, queue))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=120) for _ in processes]
        for process in processes:
            process.join(timeout=15)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                pytest.fail("portable-collective worker hung")
            assert process.exitcode == 0
        return sorted(results, key=lambda item: item["rank"])
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is required for the canonical collective transport test",
)
def test_two_process_gloo_transport_is_symmetric_bounded_and_tensor_only():
    results = _run_distributed_workers()

    assert all("fatal_error" not in result for result in results), results
    expected_successes = [
        ("worker:left", 0, [0.0, 1.0, 2.0, 3.0]),
        ("worker:right", 1, [10.0, 11.0, 12.0, 13.0]),
    ]
    assert all(result["successes"] == expected_successes for result in results)
    assert all(
        result["validation_order"] == [("worker:left", 1), ("worker:right", 1)]
        for result in results
    )
    for key, expected in (
        ("context_error", "context"),
        ("operation_error", "operation"),
        ("transaction_error", "transaction"),
        ("limits_error", "limits"),
        ("coordinate_error", "semantic coordinate/order"),
        ("callback_presence_error", "callback configuration"),
        ("prepare_error", "semantic-coordinate[0]"),
        ("validate_error", "receiver rejected plan"),
        ("consume_error", "receiver could not stage value"),
        ("budget_error", "max_collective_tensor_bytes"),
        ("corruption_error", "invalid digest"),
        ("status_error", "target changed"),
    ):
        messages = [result[key] for result in results]
        assert messages[0] == messages[1]
        assert expected in messages[0]
