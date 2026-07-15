from dataclasses import FrozenInstanceError, fields
from datetime import timedelta
import multiprocessing as mp
import os
import tempfile

import pytest
import torch
import torch.distributed as dist

from gefen.checkpoint import CheckpointProcessGroupBinding
from gefen.contracts import ProcessGroupIdentity


def test_checkpoint_binding_is_frozen_rank_neutral_and_deterministically_ordered():
    identity = ProcessGroupIdentity("pipeline:1/data", ("worker:b",))
    binding = CheckpointProcessGroupBinding(identity, "worker:b", None, torch.device("cpu"))

    assert binding.identity is identity
    assert binding.local_member == "worker:b"
    assert binding.process_group is None
    assert binding.collective_device == torch.device("cpu")
    assert binding.sort_key == ("pipeline:1/data", ("worker:b",))
    assert tuple(field.name for field in fields(binding)) == (
        "identity",
        "local_member",
        "process_group",
        "collective_device",
    )
    with pytest.raises(FrozenInstanceError):
        binding.local_member = "worker:a"


def test_checkpoint_binding_validates_descriptor_types_membership_and_device():
    identity = ProcessGroupIdentity("local", ("worker:0",))

    with pytest.raises(TypeError, match="ProcessGroupIdentity"):
        CheckpointProcessGroupBinding(object(), "worker:0", None, torch.device("cpu"))
    with pytest.raises(TypeError, match="local_member"):
        CheckpointProcessGroupBinding(identity, 0, None, torch.device("cpu"))
    with pytest.raises(ValueError, match="belong"):
        CheckpointProcessGroupBinding(identity, "worker:1", None, torch.device("cpu"))
    with pytest.raises(TypeError, match="torch.device"):
        CheckpointProcessGroupBinding(identity, "worker:0", None, "cpu")
    with pytest.raises(ValueError, match="CPU or CUDA"):
        CheckpointProcessGroupBinding(identity, "worker:0", None, torch.device("meta"))


def test_validate_collective_device_allows_gloo_cuda_but_requires_available_device():
    identity = ProcessGroupIdentity("local", ("worker:0",))
    unavailable_index = (
        torch.cuda.device_count() if torch.cuda.is_available() else 0
    )
    cuda_binding = CheckpointProcessGroupBinding(
        identity, "worker:0", None, torch.device("cuda", unavailable_index)
    )
    # Gloo supports CUDA tensors, so a CUDA collective device is not rejected on
    # backend grounds; it must instead fail the later availability check when the
    # configured device does not exist.
    with pytest.raises(ValueError, match="CUDA device is unavailable"):
        cuda_binding._validate_collective_device("gloo")

    cpu_binding = CheckpointProcessGroupBinding(
        identity, "worker:0", None, torch.device("cpu")
    )
    # NCCL still requires a CUDA collective device.
    with pytest.raises(ValueError, match="runtime backend"):
        cpu_binding._validate_collective_device("nccl")
    # Gloo with a CPU device remains valid.
    cpu_binding._validate_collective_device("gloo")


def test_checkpoint_binding_requires_explicit_multi_member_handle_and_local_singleton():
    singleton = ProcessGroupIdentity("local", ("worker:0",))
    multiple = ProcessGroupIdentity("data", ("worker:0", "worker:1"))

    with pytest.raises(ValueError, match="one-member"):
        CheckpointProcessGroupBinding(singleton, "worker:0", object(), torch.device("cpu"))
    with pytest.raises(ValueError, match="explicit"):
        CheckpointProcessGroupBinding(multiple, "worker:0", None, torch.device("cpu"))
    with pytest.raises(TypeError, match="ProcessGroup"):
        CheckpointProcessGroupBinding(multiple, "worker:0", object(), torch.device("cpu"))


def _expect_rejection(callable_object, exception_type, message):
    try:
        callable_object()
    except exception_type as exc:
        return message in str(exc)
    return False


def _distributed_binding_worker(rank, world_size, init_file, queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=45),
        )
        world_members = tuple("world:{}".format(index) for index in range(world_size))
        world_identity = ProcessGroupIdentity("world", world_members)
        world_binding = CheckpointProcessGroupBinding(
            world_identity,
            world_members[rank],
            dist.group.WORLD,
            torch.device("cpu"),
        )
        world_binding.validate_runtime()
        world_validated = world_binding.process_group is dist.group.WORLD

        wrong_member_binding = CheckpointProcessGroupBinding(
            world_identity,
            world_members[(rank + 1) % world_size],
            dist.group.WORLD,
            torch.device("cpu"),
        )
        order_mismatch_rejected = _expect_rejection(
            wrong_member_binding.validate_runtime,
            ValueError,
            "ordered semantic members",
        )

        short_identity = ProcessGroupIdentity("short", ("short:0", "short:1"))
        short_binding = CheckpointProcessGroupBinding(
            short_identity,
            short_identity.ordered_members[rank % 2],
            dist.group.WORLD,
            torch.device("cpu"),
        )
        size_mismatch_rejected = _expect_rejection(
            short_binding.validate_runtime,
            ValueError,
            "world size",
        )

        # Gloo accepts CUDA tensors, so a CUDA collective device is not rejected
        # on backend grounds; it must instead fail the later availability check
        # when the configured device does not exist. An index at or beyond the
        # visible device count (0 when no CUDA is present) is always unavailable.
        unavailable_index = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        unavailable_device_binding = CheckpointProcessGroupBinding(
            world_identity,
            world_members[rank],
            dist.group.WORLD,
            torch.device("cuda", unavailable_index),
        )
        cuda_unavailable_rejected = _expect_rejection(
            unavailable_device_binding.validate_runtime,
            ValueError,
            "CUDA device is unavailable",
        )

        subgroup_global_ranks = (0, 2)
        subgroup = dist.new_group(ranks=list(subgroup_global_ranks), backend="gloo")
        subgroup_members = ("subgroup:left", "subgroup:right")
        subgroup_validated = False
        nonmember_rejected = False
        if rank in subgroup_global_ranks:
            coordinate = subgroup_global_ranks.index(rank)
            subgroup_binding = CheckpointProcessGroupBinding(
                ProcessGroupIdentity("subgroup", subgroup_members),
                subgroup_members[coordinate],
                subgroup,
                torch.device("cpu"),
            )
            subgroup_binding.validate_runtime()
            subgroup_validated = True
        else:
            nonmember_rejected = _expect_rejection(
                lambda: CheckpointProcessGroupBinding(
                    ProcessGroupIdentity("subgroup", subgroup_members),
                    subgroup_members[0],
                    subgroup,
                    torch.device("cpu"),
                ),
                TypeError,
                "ProcessGroup",
            )

        dist.destroy_process_group()
        uninitialized_rejected = _expect_rejection(
            world_binding.validate_runtime,
            RuntimeError,
            "initialized torch.distributed",
        )
        queue.put(
            {
                "rank": rank,
                "world_validated": world_validated,
                "order_mismatch_rejected": order_mismatch_rejected,
                "size_mismatch_rejected": size_mismatch_rejected,
                "cuda_unavailable_rejected": cuda_unavailable_rejected,
                "subgroup_validated": subgroup_validated,
                "nonmember_rejected": nonmember_rejected,
                "uninitialized_rejected": uninitialized_rejected,
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_binding_workers():
    context = mp.get_context("spawn")
    queue = context.Queue()
    fd, init_file = tempfile.mkstemp(prefix="gefen-checkpoint-binding-")
    os.close(fd)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_distributed_binding_worker,
            args=(rank, 3, init_file, queue),
        )
        for rank in range(3)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                pytest.fail("checkpoint-binding worker hung")
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
    reason="Gloo is required for process-group binding validation",
)
def test_checkpoint_binding_validates_real_world_and_subgroup_membership():
    results = _run_distributed_binding_workers()

    assert all("error" not in item for item in results), results
    assert all(item["world_validated"] for item in results)
    assert all(item["order_mismatch_rejected"] for item in results)
    assert all(item["size_mismatch_rejected"] for item in results)
    assert all(item["cuda_unavailable_rejected"] for item in results)
    assert all(item["subgroup_validated"] for item in results if item["rank"] in (0, 2))
    assert results[1]["nonmember_rejected"]
    assert all(item["uninitialized_rejected"] for item in results)
