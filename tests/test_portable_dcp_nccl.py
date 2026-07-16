"""Warning-strict CUDA/NCCL coverage for portable DCP persistence."""

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
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter

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
from gefen.portable_state import PortableStateLimits
from gefen.rebinding import ParameterRebinding


_WORLD = 2
_FQN = "model.weight"


def _limits():
    return PortableStateLimits(
        max_fragment_tensor_bytes=1 << 20,
        max_collective_tensor_bytes=4 << 20,
        max_collective_metadata_bytes=4 << 20,
        chunk_bytes=13,
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


def _members(world_size, prefix):
    return tuple("{}:{}".format(prefix, rank) for rank in range(world_size))


def _replicated_manifest(identity, group):
    shards = tuple(
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
    return ShardingManifest(shards), shards


def _optimizer(rank, group, *, deterministic):
    device = torch.device("cuda", rank)
    parameter = torch.nn.Parameter(torch.linspace(-0.4, 0.3, 8, dtype=torch.float32, device=device).reshape(2, 4))
    optimizer = Gefen(
        [("weight", parameter)],
        lr=2.5e-3,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        weight_decay=0.03,
        fused=False,
        force_2d_period_one=True,
        factored_v_2d=False,
        deterministic=deterministic,
    )
    manifest, shards = _replicated_manifest(ParameterIdentity(_FQN, (2, 4)), group)
    member = group.ordered_members[rank]
    codebook_binding = CodebookProcessGroupBinding(
        group,
        member,
        dist.group.WORLD,
        device,
    )
    checkpoint_binding = CheckpointProcessGroupBinding(
        group,
        member,
        dist.group.WORLD,
        device,
    )
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, shards[rank]),),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    return optimizer, parameter, checkpoint_binding


def _codebook(device):
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32, device=device)


def _momentum(device):
    return (
        torch.tensor(
            [
                -2147483648,
                0,
                1,
                -2147483647,
                1056964608,
                -1090519040,
                1078984704,
                -1058013184,
            ],
            dtype=torch.int32,
            device=device,
        )
        .view(torch.float32)
        .reshape(2, 4)
        .clone()
    )


def _second_moment(device):
    return (
        torch.tensor(
            [
                -2147483648,
                0,
                1,
                8388608,
                1048576000,
                1065353216,
                1073741824,
                2139095039,
            ],
            dtype=torch.int32,
            device=device,
        )
        .view(torch.float32)
        .reshape(2, 4)
        .clone()
    )


def _seed_state(optimizer, parameter):
    momentum = _momentum(parameter.device)
    flat = momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8, device=parameter.device),
        torch.full((flat.numel(),), 255, dtype=torch.uint8, device=parameter.device),
    )
    optimizer._gefen_global_step = 9
    optimizer._gefen_codebook = _codebook(parameter.device)
    optimizer.state[parameter].update(
        {
            "automatic_period": 1,
            "step": 7,
            "m_codebook": indices.reshape(-1, 1),
            "m_magnitude": flat.abs().reshape(-1, 1).clone(),
            "vmean": _second_moment(parameter.device).reshape(-1, 1),
            "vmean_step": 6,
        }
    )


def _tensor_bits_equal(left, right):
    return (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().cpu().contiguous().view(torch.uint8),
            right.detach().cpu().contiguous().view(torch.uint8),
        )
    )


def _state_equal(left, right):
    if set(left) != set(right):
        return False
    for key, expected in right.items():
        actual = left[key]
        if torch.is_tensor(expected):
            if not _tensor_bits_equal(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def _optimizer_state_equal(left, left_parameter, right, right_parameter):
    return (
        left._gefen_global_step == right._gefen_global_step
        and left._deterministic is right._deterministic
        and _tensor_bits_equal(left._gefen_codebook, right._gefen_codebook)
        and _state_equal(left.state[left_parameter], right.state[right_parameter])
    )


def _strict_worker_warnings():
    warnings.simplefilter("error")
    warnings.filterwarnings(
        "ignore",
        message="TypedStorage is deprecated.*",
        category=UserWarning,
    )


def _worker(rank, world_size, init_file, checkpoint_dir, result_queue):
    _strict_worker_warnings()
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        group = ProcessGroupIdentity(
            "portable_dcp_nccl",
            _members(world_size, "rank"),
        )
        source, source_parameter, source_binding = _optimizer(
            rank,
            group,
            deterministic=True,
        )
        _seed_state(source, source_parameter)

        save_portable_dcp(
            source,
            checkpoint_process_group=source_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="portable-dcp-nccl-save-v1",
            limits=_limits(),
        )
        dist.barrier(device_ids=[rank])

        target, target_parameter, target_binding = _optimizer(
            rank,
            group,
            deterministic=False,
        )
        load_portable_dcp(
            target,
            checkpoint_process_group=target_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="portable-dcp-nccl-load-v1",
            limits=_limits(),
        )
        restored_exact = _tensor_bits_equal(target_parameter, source_parameter) and _optimizer_state_equal(
            target,
            target_parameter,
            source,
            source_parameter,
        )

        gradient = torch.tensor(
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            dtype=torch.float32,
            device=source_parameter.device,
        ).reshape(2, 4)
        source_parameter.grad = gradient.clone()
        target_parameter.grad = gradient.clone()
        source.step()
        target.step()
        continuation = {
            "parameter": _tensor_bits_equal(target_parameter, source_parameter),
            "global_step": target._gefen_global_step == source._gefen_global_step,
            "deterministic": target._deterministic is source._deterministic,
            "codebook": _tensor_bits_equal(target._gefen_codebook, source._gefen_codebook),
            "state": _state_equal(target.state[target_parameter], source.state[source_parameter]),
        }
        next_step_exact = all(continuation.values())
        result_queue.put(
            {
                "rank": rank,
                "restored_exact": restored_exact,
                "next_step_exact": next_step_exact,
                "continuation": continuation,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _cross_world_save_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        group = ProcessGroupIdentity(
            "portable_dcp_nccl_save",
            _members(world_size, "save"),
        )
        source, source_parameter, source_binding = _optimizer(
            rank,
            group,
            deterministic=True,
        )
        _seed_state(source, source_parameter)
        save_portable_dcp(
            source,
            checkpoint_process_group=source_binding,
            storage_writer=FileSystemWriter(checkpoint_dir),
            transaction_id="portable-dcp-nccl-two-to-four-save",
            limits=_limits(),
        )
        entries = FileSystemReader(checkpoint_dir).read_metadata().state_dict_metadata
        field_entries = [
            entry
            for key, entry in entries.items()
            if key.startswith("optimizer.__gefen_portable_field_")
        ]
        result_queue.put(
            {
                "rank": rank,
                "saved": True,
                "v2_sharded": (
                    "optimizer.__gefen_portable_metadata_v2__" in entries
                    and bool(field_entries)
                    and all(
                        len(entry.chunks) == world_size
                        for entry in field_entries
                    )
                ),
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _cross_world_load_worker(
    rank,
    world_size,
    init_file,
    checkpoint_dir,
    result_queue,
):
    _strict_worker_warnings()
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=90),
        )
        group = ProcessGroupIdentity(
            "portable_dcp_nccl_load",
            _members(world_size, "load"),
        )
        target, target_parameter, target_binding = _optimizer(
            rank,
            group,
            deterministic=False,
        )
        reference, reference_parameter, _reference_binding = _optimizer(
            rank,
            group,
            deterministic=True,
        )
        _seed_state(reference, reference_parameter)
        load_portable_dcp(
            target,
            checkpoint_process_group=target_binding,
            storage_reader=FileSystemReader(checkpoint_dir),
            transaction_id="portable-dcp-nccl-two-to-four-load",
            limits=_limits(),
        )
        restored_exact = (
            _tensor_bits_equal(target_parameter, reference_parameter)
            and _optimizer_state_equal(
                target,
                target_parameter,
                reference,
                reference_parameter,
            )
        )
        gradient = torch.tensor(
            [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            dtype=torch.float32,
            device=target_parameter.device,
        ).reshape(2, 4)
        target_parameter.grad = gradient.clone()
        reference_parameter.grad = gradient.clone()
        target.step()
        reference.step()
        continuation = {
            "parameter": _tensor_bits_equal(
                target_parameter,
                reference_parameter,
            ),
            "global_step": (
                target._gefen_global_step == reference._gefen_global_step
            ),
            "deterministic": (
                target._deterministic is reference._deterministic
            ),
            "codebook": _tensor_bits_equal(
                target._gefen_codebook,
                reference._gefen_codebook,
            ),
            "state": _state_equal(
                target.state[target_parameter],
                reference.state[reference_parameter],
            ),
        }
        result_queue.put(
            {
                "rank": rank,
                "restored_exact": restored_exact,
                "next_step_exact": all(continuation.values()),
                "continuation": continuation,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_phase(worker, world_size, checkpoint_dir):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-portable-dcp-nccl-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=worker,
            args=(rank, world_size, init_file, checkpoint_dir, result_queue),
        )
        for rank in range(world_size)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=180))
        except queue_module.Empty:
            pass
        for process in processes:
            process.join(timeout=10)
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
    assert len(results) == world_size, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), [process.exitcode for process in processes]
    return sorted(results, key=lambda item: item["rank"])


def _run_workers(checkpoint_dir):
    return _run_phase(_worker, _WORLD, checkpoint_dir)


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < _WORLD,
    reason="portable DCP NCCL coverage requires NCCL and two CUDA devices",
)
def test_portable_dcp_round_trip_and_next_step_are_exact_on_nccl(tmp_path):
    results = _run_workers(str(tmp_path / "checkpoint"))

    assert all("fatal_error" not in result for result in results), results
    assert [result["rank"] for result in results] == [0, 1]
    assert all(result["restored_exact"] and result["next_step_exact"] for result in results), results
    assert all(all(result["continuation"].values()) for result in results), results


@pytest.mark.skipif(
    not dist.is_available()
    or not dist.is_nccl_available()
    or torch.cuda.device_count() < 4,
    reason="portable DCP 2-to-4 coverage requires NCCL and four CUDA devices",
)
def test_portable_dcp_checkpoint_saved_on_two_gpus_resumes_on_four(tmp_path):
    checkpoint_dir = str(tmp_path / "checkpoint-two-to-four")
    save_results = _run_phase(
        _cross_world_save_worker,
        2,
        checkpoint_dir,
    )
    assert all("fatal_error" not in result for result in save_results), save_results
    assert all(result["saved"] and result["v2_sharded"] for result in save_results)

    load_results = _run_phase(
        _cross_world_load_worker,
        4,
        checkpoint_dir,
    )
    assert all("fatal_error" not in result for result in load_results), load_results
    assert [result["rank"] for result in load_results] == [0, 1, 2, 3]
    assert all(result["restored_exact"] and result["next_step_exact"] for result in load_results), load_results
    assert all(all(result["continuation"].values()) for result in load_results), load_results
