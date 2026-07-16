"""Two-rank mixed-child Hybrid DTensor rebinding and native-load coverage."""

from datetime import timedelta
import copy
import multiprocessing as mp
import os
import queue as queue_module
import shutil
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist


_WORLD = 2
_MEMBERS = ("rank:0", "rank:1")
_MUON_FQN = "Model.Muon.Weight"
_BACKUP_FQN = "Model.Backup.Weight"


def _region(shape, coordinate, dimension):
    from gefen import LogicalRegion

    global_length = shape[dimension]
    chunk = (global_length + _WORLD - 1) // _WORLD
    offset = min(coordinate * chunk, global_length)
    length = max(0, min(global_length, offset + chunk) - offset)
    offsets = [0] * len(shape)
    lengths = list(shape)
    offsets[dimension] = offset
    lengths[dimension] = length
    return LogicalRegion(tuple(offsets), tuple(lengths))


def _manifest():
    from gefen import (
        ParameterIdentity,
        ParameterLayout,
        PlacementKind,
        ProcessGroupIdentity,
        ShardIdentity,
        ShardPlacement,
        ShardingManifest,
    )

    group = ProcessGroupIdentity("data_parallel", _MEMBERS)
    shards = []
    by_fqn = {}
    for fqn, shape, dimension in (
        (_MUON_FQN, (4, 4), 0),
        (_BACKUP_FQN, (3, 4), 1),
    ):
        identity = ParameterIdentity(fqn, shape)
        parameter_shards = []
        for coordinate, member in enumerate(_MEMBERS):
            shard = ShardIdentity(
                identity,
                ParameterLayout.DTENSOR_1D_DEFAULT_WORLD,
                _region(shape, coordinate, dimension),
                placements=(
                    ShardPlacement(
                        "dp",
                        PlacementKind.DIMENSION_SHARD,
                        coordinate,
                        _WORLD,
                        parameter_dimension=dimension,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
            shards.append(shard)
            parameter_shards.append(shard)
        by_fqn[fqn] = tuple(parameter_shards)
    return ShardingManifest(tuple(shards)), by_fqn


def _optimizer(old_muon, old_backup, backup_optimizer, sharded_mode):
    from gefen import GefenMuonHybrid

    return GefenMuonHybrid(
        [("muon.weight", old_muon)],
        [("backup.weight", old_backup)],
        lr=1e-3,
        fused=False,
        deterministic=True,
        sharded_mode=sharded_mode,
        backup_optimizer=backup_optimizer,
        backup_2d_period_one=True,
    )


def _local(value):
    value = value.to_local()
    return value.wait() if hasattr(value, "wait") else value


def _tree_exact(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not torch.is_tensor(left) or not torch.is_tensor(right):
            return False
        left_dtensor = hasattr(left, "to_local")
        right_dtensor = hasattr(right, "to_local")
        if left_dtensor != right_dtensor:
            return False
        if left_dtensor:
            if (
                tuple(left.shape) != tuple(right.shape)
                or tuple(map(str, left.placements)) != tuple(map(str, right.placements))
            ):
                return False
            left = _local(left)
            right = _local(right)
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _tree_exact(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _tree_exact(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _first_mismatch(left, right, path="root"):
    if _tree_exact(left, right):
        return None
    if torch.is_tensor(left) or torch.is_tensor(right):
        return "{}: tensor mismatch {} {}".format(
            path,
            tuple(left.shape) if torch.is_tensor(left) else type(left).__name__,
            tuple(right.shape) if torch.is_tensor(right) else type(right).__name__,
        )
    if type(left) is not type(right):
        return "{}: type {} != {}".format(path, type(left).__name__, type(right).__name__)
    if isinstance(left, dict):
        if set(left) != set(right):
            return "{}: keys {} != {}".format(path, sorted(map(str, left)), sorted(map(str, right)))
        for key in left:
            mismatch = _first_mismatch(left[key], right[key], "{}[{!r}]".format(path, key))
            if mismatch is not None:
                return mismatch
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return "{}: length {} != {}".format(path, len(left), len(right))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            mismatch = _first_mismatch(left_item, right_item, "{}[{}]".format(path, index))
            if mismatch is not None:
                return mismatch
    return "{}: {!r} != {!r}".format(path, left, right)


def _semantic_checkpoint(value):
    if isinstance(value, dict):
        return {
            key: _semantic_checkpoint(item)
            for key, item in value.items()
            if key
            not in {
                "_gefen_rank_local_member",
                "_gefen_rank_local_payload_0",
                "_gefen_rank_local_payload_1",
            }
        }
    if isinstance(value, list):
        return [_semantic_checkpoint(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_semantic_checkpoint(item) for item in value)
    return value


def _build(rank, mesh, full_muon, full_backup, backup_optimizer, sharded_mode):
    from torch import nn
    from torch.distributed.tensor import Shard, distribute_tensor

    from gefen import (
        CheckpointTransport,
        CodebookProcessGroupBinding,
        ParameterLayout,
        ParameterRebinding,
        ProcessGroupScope,
    )

    manifest, shards = _manifest()
    old_muon = nn.Parameter(full_muon.clone())
    old_backup = nn.Parameter(full_backup.clone())
    muon = nn.Parameter(distribute_tensor(full_muon.clone(), mesh, [Shard(0)]))
    backup = nn.Parameter(distribute_tensor(full_backup.clone(), mesh, [Shard(1)]))
    optimizer = _optimizer(
        old_muon,
        old_backup,
        backup_optimizer,
        sharded_mode,
    )
    codebook_binding = CodebookProcessGroupBinding(
        shards[_MUON_FQN][rank].process_group,
        _MEMBERS[rank],
        dist.group.WORLD,
        torch.device("cpu"),
    )
    optimizer.post_sharding(
        (
            ParameterRebinding(old_muon, muon, shards[_MUON_FQN][rank]),
            ParameterRebinding(old_backup, backup, shards[_BACKUP_FQN][rank]),
        ),
        manifest=manifest,
        codebook_process_group=codebook_binding,
    )
    assert optimizer.shard_identity(muon) == shards[_MUON_FQN][rank]
    assert optimizer.shard_identity(backup) == shards[_BACKUP_FQN][rank]
    contract = optimizer.optimizer_contract()
    dtensor_native = [
        item
        for item in contract.capabilities.checkpoints
        if item.transport is CheckpointTransport.COMPOSITE_NATIVE
        and item.same_topology
        == frozenset({ParameterLayout.DTENSOR_1D_DEFAULT_WORLD})
    ]
    assert len(dtensor_native) == 1
    assert dtensor_native[0].process_group_scope is ProcessGroupScope.DEFAULT_WORLD
    assert dtensor_native[0].mesh_dimensions == (1,)
    assert dtensor_native[0].requires_collective
    assert dtensor_native[0].atomic_load
    if backup_optimizer == "adamw":
        assert all(
            item.transport is not CheckpointTransport.CANONICAL_GLOBAL
            for item in contract.capabilities.checkpoints
        )
    return optimizer, muon, backup


def _set_gradients(muon, backup, mesh, muon_gradient, backup_gradient):
    from torch.distributed.tensor import Shard, distribute_tensor

    muon.grad = distribute_tensor(muon_gradient.clone(), mesh, [Shard(0)])
    backup.grad = distribute_tensor(backup_gradient.clone(), mesh, [Shard(1)])


def _worker(rank, init_file, backup_optimizer, sharded_mode, result_queue):
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=120),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        full_muon = torch.linspace(-0.75, 0.75, 16, dtype=torch.float32).reshape(4, 4)
        full_backup = torch.linspace(-0.5, 0.625, 12, dtype=torch.float32).reshape(3, 4)
        first_muon_gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4).sin()
        first_backup_gradient = torch.arange(12, dtype=torch.float32).reshape(3, 4).cos()

        asymmetric, asymmetric_muon, asymmetric_backup = _build(
            rank,
            mesh,
            full_muon,
            full_backup,
            backup_optimizer,
            sharded_mode,
        )
        asymmetric_before = copy.deepcopy(asymmetric.state_dict())
        asymmetric_muon_before = _local(asymmetric_muon.detach()).clone()
        asymmetric_backup_before = _local(asymmetric_backup.detach()).clone()
        asymmetric_gradient = distribute_tensor(
            first_muon_gradient.clone(),
            mesh,
            [Shard(0)],
        )
        if rank == 0:
            asymmetric_muon.grad = asymmetric_gradient
        try:
            asymmetric.step()
        except RuntimeError as exc:
            asymmetric_error = "gradient presence" in str(exc)
        else:
            asymmetric_error = False
        asymmetric_rejected = (
            asymmetric_error
            and _tree_exact(asymmetric_before, asymmetric.state_dict())
            and torch.equal(
                _local(asymmetric_muon.detach()),
                asymmetric_muon_before,
            )
            and torch.equal(
                _local(asymmetric_backup.detach()),
                asymmetric_backup_before,
            )
        )

        optimizer, muon, backup = _build(
            rank,
            mesh,
            full_muon,
            full_backup,
            backup_optimizer,
            sharded_mode,
        )
        _set_gradients(
            muon,
            backup,
            mesh,
            first_muon_gradient,
            first_backup_gradient,
        )
        optimizer.step()
        optimizer.zero_grad()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        current_muon = muon.detach().full_tensor().clone()
        current_backup = backup.detach().full_tensor().clone()

        rejected, rejected_muon, rejected_backup = _build(
            rank,
            mesh,
            current_muon,
            current_backup,
            backup_optimizer,
            sharded_mode,
        )
        rejected_before = copy.deepcopy(rejected.state_dict())
        rejected_errors = []
        for rejected_role in ("muon", "backup"):
            invalid = copy.deepcopy(checkpoint)
            invalid[rejected_role]["param_groups"].append(
                copy.deepcopy(invalid[rejected_role]["param_groups"][0])
            )
            try:
                rejected.load_state_dict(invalid)
            except (RuntimeError, ValueError):
                rejected_errors.append(True)
            else:
                rejected_errors.append(False)
        rejected_unchanged = (
            all(rejected_errors)
            and _tree_exact(rejected_before, rejected.state_dict())
            and torch.equal(_local(rejected_muon.detach()), _local(muon.detach()))
            and torch.equal(_local(rejected_backup.detach()), _local(backup.detach()))
        )

        resumed, resumed_muon, resumed_backup = _build(
            rank,
            mesh,
            current_muon,
            current_backup,
            backup_optimizer,
            sharded_mode,
        )
        resumed.load_state_dict(checkpoint)
        restored_source_state = _semantic_checkpoint(optimizer.state_dict())
        restored_resumed_state = _semantic_checkpoint(resumed.state_dict())
        restored = _tree_exact(restored_source_state, restored_resumed_state)
        restored_mismatch = _first_mismatch(
            restored_source_state,
            restored_resumed_state,
        )

        second_muon_gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4).cos()
        second_backup_gradient = torch.arange(12, dtype=torch.float32).reshape(3, 4).sin()
        _set_gradients(
            muon,
            backup,
            mesh,
            second_muon_gradient,
            second_backup_gradient,
        )
        _set_gradients(
            resumed_muon,
            resumed_backup,
            mesh,
            second_muon_gradient,
            second_backup_gradient,
        )
        optimizer.step()
        resumed.step()
        muon_parameter_exact = torch.equal(
            _local(muon.detach()), _local(resumed_muon.detach())
        )
        backup_parameter_exact = torch.equal(
            _local(backup.detach()), _local(resumed_backup.detach())
        )
        continued_source_state = _semantic_checkpoint(optimizer.state_dict())
        continued_resumed_state = _semantic_checkpoint(resumed.state_dict())
        continued_state_exact = _tree_exact(
            continued_source_state,
            continued_resumed_state,
        )
        continued = (
            muon_parameter_exact
            and backup_parameter_exact
            and continued_state_exact
        )
        result_queue.put(
            {
                "rank": rank,
                "restored": restored,
                "continued": continued,
                "rejected_unchanged": rejected_unchanged,
                "asymmetric_rejected": asymmetric_rejected,
                "restored_mismatch": restored_mismatch,
                "continued_mismatch": _first_mismatch(
                    continued_source_state,
                    continued_resumed_state,
                ),
                "muon_parameter_exact": muon_parameter_exact,
                "backup_parameter_exact": backup_parameter_exact,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _checkpoint_failure_worker(rank, init_file, result_queue):
    from torch.distributed.tensor import init_device_mesh

    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=60),
        )
        mesh = init_device_mesh("cpu", (_WORLD,), mesh_dim_names=("dp",))
        full_muon = torch.linspace(-0.75, 0.75, 16, dtype=torch.float32).reshape(4, 4)
        full_backup = torch.linspace(-0.5, 0.625, 12, dtype=torch.float32).reshape(3, 4)
        muon_gradient = torch.arange(16, dtype=torch.float32).reshape(4, 4).sin()
        backup_gradient = torch.arange(12, dtype=torch.float32).reshape(3, 4).cos()

        optimizer, muon, backup = _build(
            rank,
            mesh,
            full_muon,
            full_backup,
            "adamw",
            "approx",
        )
        _set_gradients(muon, backup, mesh, muon_gradient, backup_gradient)
        optimizer.step()
        optimizer.zero_grad()
        checkpoint = copy.deepcopy(optimizer.state_dict())
        save_before = _semantic_checkpoint(checkpoint)
        save_muon_before = _local(muon.detach()).clone()
        save_backup_before = _local(backup.detach()).clone()

        hook_handle = None
        if rank == 0:
            def reject_save(_optimizer):
                raise ValueError("injected rank-local Hybrid save rejection")

            hook_handle = optimizer.register_state_dict_pre_hook(reject_save)
        try:
            optimizer.state_dict()
        except RuntimeError as exc:
            save_error = "state-dict pre-hook" in str(exc)
        else:
            save_error = False
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        save_after = _semantic_checkpoint(optimizer.state_dict())
        save_rejected = (
            save_error
            and _tree_exact(save_before, save_after)
            and torch.equal(_local(muon.detach()), save_muon_before)
            and torch.equal(_local(backup.detach()), save_backup_before)
        )

        current_muon = muon.detach().full_tensor().clone()
        current_backup = backup.detach().full_tensor().clone()
        rejected, rejected_muon, rejected_backup = _build(
            rank,
            mesh,
            current_muon,
            current_backup,
            "adamw",
            "approx",
        )
        load_before = copy.deepcopy(rejected.state_dict())
        load_muon_before = _local(rejected_muon.detach()).clone()
        load_backup_before = _local(rejected_backup.detach()).clone()
        invalid = copy.deepcopy(checkpoint)
        if rank == 0:
            invalid["backup"]["param_groups"].append(
                copy.deepcopy(invalid["backup"]["param_groups"][0])
            )
        try:
            rejected.load_state_dict(invalid)
        except RuntimeError as exc:
            load_error = "backup load staging" in str(exc)
        else:
            load_error = False
        load_after = rejected.state_dict()
        load_rejected = (
            load_error
            and _tree_exact(load_before, load_after)
            and torch.equal(_local(rejected_muon.detach()), load_muon_before)
            and torch.equal(_local(rejected_backup.detach()), load_backup_before)
        )
        result_queue.put(
            {
                "rank": rank,
                "save_rejected": save_rejected,
                "load_rejected": load_rejected,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run(backup_optimizer, sharded_mode):
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-hybrid-mixed-dtensor-")
    init_file = os.path.join(directory, "store")
    processes = [
        context.Process(
            target=_worker,
            args=(rank, init_file, backup_optimizer, sharded_mode, result_queue),
        )
        for rank in range(_WORLD)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=300))
        except queue_module.Empty:
            pass
        for process in processes:
            process.join(timeout=15)
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
        shutil.rmtree(directory, ignore_errors=True)
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), results
    return results


def _run_checkpoint_failures():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    directory = tempfile.mkdtemp(prefix="gefen-hybrid-mixed-dtensor-failure-")
    init_file = os.path.join(directory, "store")
    processes = [
        context.Process(
            target=_checkpoint_failure_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(_WORLD)
    ]
    results = []
    try:
        for process in processes:
            process.start()
        try:
            for _ in processes:
                results.append(result_queue.get(timeout=120))
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
        shutil.rmtree(directory, ignore_errors=True)
    assert len(results) == _WORLD, (results, [process.exitcode for process in processes])
    assert all(process.exitcode == 0 for process in processes), results
    return results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="mixed Hybrid DTensor rebinding requires Gloo",
)
@pytest.mark.parametrize(
    ("backup_optimizer", "sharded_mode"),
    [
        ("gefen", "approx"),
        ("gefen", "exact"),
        ("gefen", "distributed"),
        ("adamw", "approx"),
        ("adamw", "exact"),
        ("adamw", "distributed"),
    ],
)
def test_mixed_hybrid_dtensor_native_load_is_atomic_and_continues_exactly(
    backup_optimizer,
    sharded_mode,
):
    results = _run(backup_optimizer, sharded_mode)
    assert all("traceback" not in result for result in results), results
    assert all(
        result["restored"]
        and result["continued"]
        and result["rejected_unchanged"]
        and result["asymmetric_rejected"]
        for result in results
    ), results


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="mixed Hybrid DTensor checkpoint failure agreement requires Gloo",
)
def test_mixed_hybrid_dtensor_checkpoint_failures_are_rank_symmetric_and_atomic():
    results = _run_checkpoint_failures()
    assert all("traceback" not in result for result in results), results
    assert all(
        result["save_rejected"] and result["load_rejected"]
        for result in results
    ), results
