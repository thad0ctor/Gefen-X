"""Warning-strict Gloo regressions for portable runtime consensus gates."""

from datetime import timedelta
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import traceback
import warnings

import pytest
import torch
import torch.distributed as dist

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
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_WORLD = 2
_SINGLE_FQN = "model.weight"


class _CheckpointBindingSubclass(CheckpointProcessGroupBinding):
    pass


def _members():
    return tuple("rank:{}".format(rank) for rank in range(_WORLD))


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=4 << 20,
        max_collective_metadata_bytes=4 << 20,
        chunk_bytes=17,
        max_members=4,
        max_metadata_bytes=1 << 20,
        max_tree_nodes=10_000,
        max_tree_depth=32,
        max_container_items=10_000,
        max_string_bytes=16 << 10,
        max_integer_bytes=128,
        max_tensors=256,
        max_tensor_rank=8,
        diagnostic_bytes=1024,
    )


def _bindings(group, rank):
    member = _members()[rank]
    codebook = CodebookProcessGroupBinding(
        group,
        member,
        dist.group.WORLD,
        torch.device("cpu"),
    )
    checkpoint = CheckpointProcessGroupBinding(
        group,
        member,
        dist.group.WORLD,
        torch.device("cpu"),
    )
    return codebook, checkpoint


def _replicated_shards(identity, group):
    return tuple(
        ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
            placements=(
                ShardPlacement(
                    "checkpoint",
                    PlacementKind.REPLICATE,
                    coordinate,
                    len(group.ordered_members),
                ),
            ),
            process_group=group,
            local_member=member,
        )
        for coordinate, member in enumerate(group.ordered_members)
    )


def _new_optimizer(entries, *, deterministic):
    return Gefen(
        entries,
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        force_2d_period_one=True,
        factored_v_2d=False,
        deterministic=deterministic,
    )


def _make_single(rank, group, *, compatibility_name, deterministic):
    identity = ParameterIdentity(_SINGLE_FQN, (2, 3))
    parameter = torch.nn.Parameter(torch.zeros(identity.global_shape, dtype=torch.float32))
    optimizer = _new_optimizer(
        [(compatibility_name, parameter)],
        deterministic=deterministic,
    )
    shards = _replicated_shards(identity, group)
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shards[rank]),),
        manifest=ShardingManifest(shards),
        codebook_process_group=codebook_binding,
    )
    return optimizer, checkpoint_binding


def _make_two(rank, group, *, reverse, deterministic):
    identities = (
        ParameterIdentity("model.alpha", (2, 2)),
        ParameterIdentity("model.beta", (2, 2)),
    )
    alpha = torch.nn.Parameter(torch.zeros(identities[0].global_shape, dtype=torch.float32))
    beta = torch.nn.Parameter(torch.zeros(identities[1].global_shape, dtype=torch.float32))
    entries = [("alpha", alpha), ("beta", beta)]
    if reverse:
        entries.reverse()
    optimizer = _new_optimizer(entries, deterministic=deterministic)
    shards_by_identity = tuple(_replicated_shards(identity, group) for identity in identities)
    manifest = ShardingManifest(tuple(shard for shards in shards_by_identity for shard in shards))
    codebook_binding, checkpoint_binding = _bindings(group, rank)
    optimizer.post_sharding(
        (
            ParameterRebinding(alpha, alpha, shards_by_identity[0][rank]),
            ParameterRebinding(beta, beta, shards_by_identity[1][rank]),
        ),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, checkpoint_binding


def _subclass_binding(binding):
    return _CheckpointBindingSubclass(
        binding.identity,
        binding.local_member,
        binding.process_group,
        binding.collective_device,
    )


def _binding_with_process_group(binding, process_group):
    return CheckpointProcessGroupBinding(
        binding.identity,
        binding.local_member,
        process_group,
        binding.collective_device,
    )


def _attempt_failure(optimizer, operation):
    state_mapping = optimizer.state
    param_groups = optimizer.param_groups
    defaults = optimizer.defaults
    parameters = tuple(parameter for group in optimizer.param_groups for parameter in group["params"])
    state_objects = tuple(optimizer.state[parameter] for parameter in parameters)
    token = optimizer._canonical_import_live_token()
    global_step = optimizer._gefen_global_step
    codebook = optimizer._gefen_codebook
    deterministic = optimizer._deterministic
    try:
        operation()
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = None
    return {
        "message": message,
        "token_unchanged": optimizer._canonical_import_live_token() == token,
        "containers_unchanged": (
            optimizer.state is state_mapping
            and optimizer.param_groups is param_groups
            and optimizer.defaults is defaults
        ),
        "state_objects_unchanged": all(
            optimizer.state[parameter] is state for parameter, state in zip(parameters, state_objects)
        ),
        "states_pristine": all(
            optimizer.state[parameter] == {"name": optimizer._param_names[parameter]} for parameter in parameters
        ),
        "common_unchanged": (
            optimizer._gefen_global_step == global_step
            and optimizer._gefen_codebook is codebook
            and optimizer._deterministic is deterministic
        ),
    }


def _worker(rank, init_file, result_queue):
    warnings.simplefilter("error")
    alternate_process_group = None
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=20),
        )
        group = ProcessGroupIdentity("portable_consensus", _members())
        alternate_process_group = dist.new_group(
            ranks=list(range(_WORLD)),
            backend="gloo",
            timeout=timedelta(seconds=20),
        )

        source, exact_binding = _make_single(
            rank,
            group,
            compatibility_name="weight",
            deterministic=True,
        )
        document = source.export_portable_state(
            checkpoint_process_group=exact_binding,
            transaction_id="consensus-healthy-single-export",
            limits=_limits(),
        )
        asymmetric_binding = _subclass_binding(exact_binding) if rank == 0 else exact_binding
        binding_export = _attempt_failure(
            source,
            lambda: source.export_portable_state(
                checkpoint_process_group=asymmetric_binding,
                transaction_id="consensus-subclass-export",
                limits=_limits(),
            ),
        )
        dist.barrier()
        object_binding = object() if rank == 0 else exact_binding
        object_export = _attempt_failure(
            source,
            lambda: source.export_portable_state(
                checkpoint_process_group=object_binding,
                transaction_id="consensus-object-export",
                limits=_limits(),
            ),
        )
        dist.barrier()
        alternate_binding = (
            _binding_with_process_group(exact_binding, alternate_process_group) if rank == 0 else exact_binding
        )
        alternate_export = _attempt_failure(
            source,
            lambda: source.export_portable_state(
                checkpoint_process_group=alternate_binding,
                transaction_id="consensus-alternate-handle-export",
                limits=_limits(),
            ),
        )
        dist.barrier()

        binding_target, target_binding = _make_single(
            rank,
            group,
            compatibility_name="weight",
            deterministic=False,
        )
        asymmetric_target_binding = _subclass_binding(target_binding) if rank == 0 else target_binding
        binding_import = _attempt_failure(
            binding_target,
            lambda: binding_target.import_portable_state(
                document,
                checkpoint_process_group=asymmetric_target_binding,
                transaction_id="consensus-subclass-import",
                limits=_limits(),
            ),
        )
        dist.barrier()
        object_target, object_target_binding = _make_single(
            rank,
            group,
            compatibility_name="weight",
            deterministic=False,
        )
        asymmetric_object_binding = object() if rank == 0 else object_target_binding
        object_import = _attempt_failure(
            object_target,
            lambda: object_target.import_portable_state(
                document,
                checkpoint_process_group=asymmetric_object_binding,
                transaction_id="consensus-object-import",
                limits=_limits(),
            ),
        )
        dist.barrier()
        alternate_target, alternate_target_binding = _make_single(
            rank,
            group,
            compatibility_name="weight",
            deterministic=False,
        )
        asymmetric_alternate_binding = (
            _binding_with_process_group(
                alternate_target_binding,
                alternate_process_group,
            )
            if rank == 0
            else alternate_target_binding
        )
        alternate_import = _attempt_failure(
            alternate_target,
            lambda: alternate_target.import_portable_state(
                document,
                checkpoint_process_group=asymmetric_alternate_binding,
                transaction_id="consensus-alternate-handle-import",
                limits=_limits(),
            ),
        )
        dist.barrier()

        name_target, name_binding = _make_single(
            rank,
            group,
            compatibility_name="weight" if rank == 0 else "other_weight",
            deterministic=False,
        )
        compatibility_import = _attempt_failure(
            name_target,
            lambda: name_target.import_portable_state(
                document,
                checkpoint_process_group=name_binding,
                transaction_id="consensus-compatibility-import",
                limits=_limits(),
            ),
        )
        dist.barrier()

        two_source, two_source_binding = _make_two(
            rank,
            group,
            reverse=False,
            deterministic=True,
        )
        two_document = two_source.export_portable_state(
            checkpoint_process_group=two_source_binding,
            transaction_id="consensus-healthy-two-export",
            limits=_limits(),
        )
        position_target, position_binding = _make_two(
            rank,
            group,
            reverse=rank == 1,
            deterministic=False,
        )
        position_import = _attempt_failure(
            position_target,
            lambda: position_target.import_portable_state(
                two_document,
                checkpoint_process_group=position_binding,
                transaction_id="consensus-position-import",
                limits=_limits(),
            ),
        )
        dist.barrier()

        result_queue.put(
            {
                "rank": rank,
                "binding_export": binding_export,
                "binding_import": binding_import,
                "object_export": object_export,
                "object_import": object_import,
                "alternate_export": alternate_export,
                "alternate_import": alternate_import,
                "compatibility_import": compatibility_import,
                "position_import": position_import,
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            if alternate_process_group is not None:
                dist.destroy_process_group(alternate_process_group)
            dist.destroy_process_group()


def _run_workers():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-consensus-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [context.Process(target=_worker, args=(rank, init_file, result_queue)) for rank in range(_WORLD)]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=60))
        except queue_module.Empty:
            pass
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
        if os.path.exists(init_file):
            os.unlink(init_file)
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="portable runtime consensus coverage requires Gloo",
)
def test_two_process_portable_runtime_rejects_asymmetric_consensus_inputs():
    results = _run_workers()
    assert all("fatal_error" not in result for result in results), results

    expected_messages = {
        "binding_export": "CheckpointProcessGroupBinding",
        "binding_import": "CheckpointProcessGroupBinding",
        "object_export": "CheckpointProcessGroupBinding",
        "object_import": "CheckpointProcessGroupBinding",
        "alternate_export": "optimizer-owned runtime transport",
        "alternate_import": "optimizer-owned runtime transport",
        "compatibility_import": "context",
        "position_import": "context",
    }
    failures = []
    for case, expected_message in expected_messages.items():
        messages = [result[case]["message"] for result in results]
        if messages[0] != messages[1]:
            failures.append((case, "messages diverged", messages))
        if messages[0] is None or expected_message not in messages[0]:
            failures.append((case, "missing expected failure", messages))
        mutation_free = all(
            all(
                result[case][key]
                for key in (
                    "token_unchanged",
                    "containers_unchanged",
                    "state_objects_unchanged",
                    "states_pristine",
                    "common_unchanged",
                )
            )
            for result in results
        )
        if not mutation_free:
            failures.append((case, "live state changed", [result[case] for result in results]))
    assert not failures, (failures, results)
