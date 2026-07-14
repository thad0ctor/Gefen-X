"""Scoped failure-protocol coverage for GefenMuonHybrid.step.

GradScaler attaches found_inf/grad_scale to the composite (never to the
children), and the hybrid's structural gradient preflight runs before either
child's scoped step collectives. Under the one multi-member codebook binding
shared by both children, both decisions must therefore be group decisions:

* a rank-local AMP overflow must not skip the children on one member while the
  peers enter the children's scoped step collectives -- found_inf/grad_scale
  agreement is validated collectively and an overflow skip is entered/exited
  symmetrically on every member;
* a rank-local structural preflight failure must raise on every scope member
  together via the children's synchronized failure protocol instead of
  stranding peers inside a child's scope header all_gather.

Single-process behavior (no binding, and the collective-free one-member
binding) keeps the local semantics.
"""

from datetime import timedelta
import multiprocessing as mp
import os
import queue as queue_module
import tempfile
import traceback

import pytest
import torch
import torch.distributed as dist

from gefen import GefenMuonHybrid
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
from gefen.rebinding import ParameterRebinding


_WORLD = 2
_MUON_FQN = "model.matrix"
_BACKUP_FQN = "model.vector"
_MUON_OWNER = "rank:1"
_BACKUP_LENGTHS = (3, 5)


def _members():
    return tuple("rank:{}".format(rank) for rank in range(_WORLD))


def _owner_shards(identity, group, owner):
    return tuple(
        ShardIdentity(
            identity,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            (LogicalSlice.full(identity) if member == owner else LogicalSlice(0, 0)),
            placements=(
                ShardPlacement(
                    "dp",
                    PlacementKind.WHOLE_PARAMETER_OWNER,
                    coordinate,
                    len(group.ordered_members),
                ),
            ),
            process_group=group,
            local_member=member,
            owner=owner,
        )
        for coordinate, member in enumerate(group.ordered_members)
    )


def _flat_shards(identity, group, lengths):
    offset = 0
    shards = []
    for coordinate, (member, length) in enumerate(zip(group.ordered_members, lengths)):
        shards.append(
            ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, length),
                placements=(
                    ShardPlacement(
                        "dp",
                        PlacementKind.FLAT_SHARD,
                        coordinate,
                        len(group.ordered_members),
                    ),
                ),
                process_group=group,
                local_member=member,
            )
        )
        offset += length
    return tuple(shards)


def _muon_initial():
    return torch.linspace(-0.3, 0.2, 6, dtype=torch.float32).reshape(3, 2)


def _backup_initial():
    return torch.linspace(-0.4, 0.3, 8, dtype=torch.float32)


def _muon_gradient():
    return torch.tensor(
        [[0.4, -0.7], [1.1, -1.3], [1.7, -1.9]],
        dtype=torch.float32,
    )


def _backup_gradient():
    return torch.tensor(
        [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
        dtype=torch.float32,
    )


def _codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)


def _quantized_period_one(momentum):
    flat = momentum.reshape(-1)
    indices = torch.where(
        torch.signbit(flat),
        torch.zeros(flat.numel(), dtype=torch.uint8),
        torch.full((flat.numel(),), 255, dtype=torch.uint8),
    )
    return indices.reshape(-1, 1), flat.abs().reshape(-1, 1).clone()


def _hybrid_kwargs():
    return dict(
        lr=2.5e-3,
        muon_lr=3.0e-3,
        backup_lr=2.5e-3,
        weight_decay=0.03,
        muon_weight_decay=0.02,
        backup_weight_decay=0.03,
        backup_optimizer="gefen",
        backup_1d_period_one=True,
        betas=(0.8, 0.97),
        eps=2.0e-8,
        fused=False,
        momentum=0.85,
        nesterov=False,
        ns_steps=2,
        deterministic=True,
        normuon=True,
        normuon_beta2=0.9,
        normuon_eps=3.0e-8,
    )


def _make_scoped_hybrid(rank, group):
    muon_identity = ParameterIdentity(_MUON_FQN, (3, 2))
    backup_identity = ParameterIdentity(_BACKUP_FQN, (8,))
    old_muon = torch.nn.Parameter(_muon_initial().clone())
    old_backup = torch.nn.Parameter(_backup_initial().clone())
    optimizer = GefenMuonHybrid(
        [("matrix", old_muon)],
        [("vector", old_backup)],
        sharded_mode="distributed",
        **_hybrid_kwargs(),
    )
    muon_shards = _owner_shards(muon_identity, group, _MUON_OWNER)
    backup_shards = _flat_shards(backup_identity, group, _BACKUP_LENGTHS)
    muon_local = old_muon if _members()[rank] == _MUON_OWNER else None
    backup_shard = backup_shards[rank]
    start = backup_shard.logical_slice.flat_offset
    stop = start + backup_shard.logical_slice.length
    backup_local = torch.nn.Parameter(_backup_initial()[start:stop].clone())
    binding = CodebookProcessGroupBinding(
        group,
        _members()[rank],
        dist.group.WORLD,
        torch.device("cpu"),
    )
    optimizer.post_sharding(
        (
            ParameterRebinding(old_muon, muon_local, muon_shards[rank]),
            ParameterRebinding(old_backup, backup_local, backup_shard),
        ),
        manifest=ShardingManifest(muon_shards + backup_shards),
        codebook_process_group=binding,
    )
    return optimizer, muon_local, backup_local, backup_shard


def _seed_hybrid_state(optimizer, muon_parameter, backup_parameter, backup_shard):
    for child in (optimizer.muon, optimizer.backup):
        child._gefen_global_step = 13
        child._gefen_codebook = _codebook()
    if muon_parameter is not None:
        momentum = torch.tensor(
            [[-0.75, 0.5], [1.25, -1.5], [2.0, -2.5]],
            dtype=torch.float32,
        )
        indices, magnitudes = _quantized_period_one(momentum)
        optimizer.muon.state[muon_parameter].update(
            {
                "automatic_period": 1,
                "step": 11,
                "m_codebook": indices,
                "m_magnitude": magnitudes,
                "normuon_v": torch.tensor([[0.5], [1.5], [2.5]], dtype=torch.float32),
                "normuon_step": 10,
            }
        )
    start = backup_shard.logical_slice.flat_offset
    stop = start + backup_shard.logical_slice.length
    momentum = torch.tensor(
        [-0.25, 0.5, -0.75, 1.0, -1.25, 1.5, -1.75, 2.0],
        dtype=torch.float32,
    )[start:stop]
    indices, magnitudes = _quantized_period_one(momentum)
    optimizer.backup.state[backup_parameter].update(
        {
            "automatic_period": 1,
            "step": 11,
            "m_codebook": indices,
            "m_magnitude": magnitudes,
            "vmean": torch.linspace(0.2, 0.9, 8, dtype=torch.float32)[start:stop].reshape(-1, 1).clone(),
            "vmean_step": 10,
        }
    )


def _bits_equal(left, right):
    return (
        torch.is_tensor(left)
        and torch.is_tensor(right)
        and left.dtype == right.dtype
        and tuple(left.shape) == tuple(right.shape)
        and torch.equal(
            left.detach().contiguous().view(torch.uint8),
            right.detach().contiguous().view(torch.uint8),
        )
    )


def _set_local_gradients(muon_parameter, backup_parameter, backup_shard, scale=1.0):
    if muon_parameter is not None:
        muon_parameter.grad = _muon_gradient() * scale
    start = backup_shard.logical_slice.flat_offset
    stop = start + backup_shard.logical_slice.length
    backup_parameter.grad = _backup_gradient()[start:stop] * scale


def _untouched(optimizer, muon_parameter, backup_parameter):
    return (
        optimizer.muon._gefen_global_step == 0
        and optimizer.backup._gefen_global_step == 0
        and optimizer.muon._gefen_codebook is None
        and optimizer.backup._gefen_codebook is None
        and (muon_parameter is None or _bits_equal(muon_parameter, _muon_initial()))
        and (
            muon_parameter is None
            or optimizer.muon.state[muon_parameter] == {"name": "matrix"}
        )
        and optimizer.backup.state[backup_parameter] == {"name": "vector"}
    )


def _amp_divergent_overflow_result(rank, group):
    optimizer, muon_parameter, backup_parameter, backup_shard = _make_scoped_hybrid(rank, group)
    _set_local_gradients(muon_parameter, backup_parameter, backup_shard)
    grad_before = backup_parameter.grad.detach().clone()
    optimizer.found_inf = torch.tensor(float(rank == 0))
    optimizer.grad_scale = torch.tensor(8.0)
    try:
        optimizer.step()
        message = None
    except RuntimeError as exc:
        message = str(exc)
    return {
        "message": message,
        "untouched": _untouched(optimizer, muon_parameter, backup_parameter),
        "grads_untouched": _bits_equal(backup_parameter.grad, grad_before),
    }


def _amp_group_wide_overflow_result(rank, group):
    optimizer, muon_parameter, backup_parameter, backup_shard = _make_scoped_hybrid(rank, group)
    _set_local_gradients(muon_parameter, backup_parameter, backup_shard)
    grad_before = backup_parameter.grad.detach().clone()
    optimizer.found_inf = torch.tensor(1.0)
    optimizer.grad_scale = torch.tensor(8.0)
    post_hook_calls = []
    optimizer.register_step_post_hook(
        lambda _optimizer, _args, _kwargs: post_hook_calls.append(True)
    )
    try:
        optimizer.step()
        skipped = True
    except RuntimeError:
        skipped = False
    return {
        "skipped": skipped,
        "post_hook_fired": len(post_hook_calls) == 1,
        "untouched": _untouched(optimizer, muon_parameter, backup_parameter),
        "grads_untouched": _bits_equal(backup_parameter.grad, grad_before),
    }


def _amp_agreeing_finite_result(rank, group):
    target, target_muon, target_backup, target_shard = _make_scoped_hybrid(rank, group)
    reference, reference_muon, reference_backup, reference_shard = _make_scoped_hybrid(rank, group)
    _seed_hybrid_state(target, target_muon, target_backup, target_shard)
    _seed_hybrid_state(reference, reference_muon, reference_backup, reference_shard)
    _set_local_gradients(target_muon, target_backup, target_shard, scale=8.0)
    _set_local_gradients(reference_muon, reference_backup, reference_shard)
    target.found_inf = torch.tensor(0.0)
    target.grad_scale = torch.tensor(8.0)
    target.step()
    reference.step()
    return {
        "stepped": (
            target.muon._gefen_global_step == 14
            and target.backup._gefen_global_step == 14
        ),
        "unscaled_once_exactly": (
            (target_muon is None or _bits_equal(target_muon, reference_muon))
            and _bits_equal(target_backup, reference_backup)
        ),
    }


def _preflight_divergent_result(rank, group):
    optimizer, muon_parameter, backup_parameter, backup_shard = _make_scoped_hybrid(rank, group)
    _set_local_gradients(muon_parameter, backup_parameter, backup_shard)
    if rank == 0:
        backup_parameter.grad = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            size=(backup_shard.logical_slice.length,),
        )
    try:
        optimizer.step()
        message = None
    except RuntimeError as exc:
        message = str(exc)
    return {
        "message": message,
        "untouched": _untouched(optimizer, muon_parameter, backup_parameter),
    }


def _closure_divergent_result(rank, group):
    # The closure runs BEFORE any scope synchronization. A rank-local closure
    # failure must raise on every scope member together instead of leaving the
    # failing rank to exit step() while the peer enters the composite preflight
    # synchronize / a child's scoped step collectives and hangs.
    optimizer, muon_parameter, backup_parameter, backup_shard = _make_scoped_hybrid(rank, group)
    _set_local_gradients(muon_parameter, backup_parameter, backup_shard)

    def closure():
        if rank == 0:
            raise RuntimeError("closure boom on rank:0")
        return torch.tensor(1.0)

    try:
        optimizer.step(closure)
        message = None
    except RuntimeError as exc:
        message = str(exc)
    return {
        "message": message,
        "untouched": _untouched(optimizer, muon_parameter, backup_parameter),
    }


def _checkpoint_parent_failure_result(rank, group):
    optimizer, muon_parameter, backup_parameter, backup_shard = _make_scoped_hybrid(
        rank, group
    )
    _seed_hybrid_state(
        optimizer,
        muon_parameter,
        backup_parameter,
        backup_shard,
    )
    hook_handle = None
    if rank == 0:
        def reject_save(_optimizer):
            raise RuntimeError("checkpoint pre-hook boom on rank:0")

        hook_handle = optimizer.register_state_dict_pre_hook(reject_save)
    try:
        optimizer.state_dict()
        message = None
    except RuntimeError as exc:
        message = str(exc)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return {"message": message}


def _distributed_worker(rank, init_file, result_queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=_WORLD,
            timeout=timedelta(seconds=45),
        )
        group = ProcessGroupIdentity("hybrid_scoped_failure", _members())
        amp_divergent = _amp_divergent_overflow_result(rank, group)
        dist.barrier()
        amp_group_wide = _amp_group_wide_overflow_result(rank, group)
        dist.barrier()
        amp_finite = _amp_agreeing_finite_result(rank, group)
        dist.barrier()
        preflight = _preflight_divergent_result(rank, group)
        dist.barrier()
        closure_divergent = _closure_divergent_result(rank, group)
        dist.barrier()
        checkpoint_parent = _checkpoint_parent_failure_result(rank, group)
        dist.barrier()
        result_queue.put(
            {
                "rank": rank,
                "amp_divergent": amp_divergent,
                "amp_group_wide": amp_group_wide,
                "amp_finite": amp_finite,
                "preflight": preflight,
                "closure_divergent": closure_divergent,
                "checkpoint_parent": checkpoint_parent,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_workers():
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    descriptor, init_file = tempfile.mkstemp(prefix="gefen-hybrid-scoped-failure-")
    os.close(descriptor)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_distributed_worker,
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
                results.append(result_queue.get(timeout=240))
        except queue_module.Empty:
            pass
        for process in processes:
            process.join(timeout=10)
        # A worker that queued its result but then hangs in
        # destroy_process_group() (or exits nonzero) means the release-critical
        # synchronization did not shut down cleanly; fail rather than silently
        # terminate it below.
        hung = [index for index, process in enumerate(processes) if process.is_alive()]
        exit_codes = [process.exitcode for process in processes]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
        if os.path.exists(init_file):
            os.unlink(init_file)
    assert not hung, ("hybrid scoped-failure workers hung", hung, exit_codes)
    assert all(code == 0 for code in exit_codes), exit_codes
    assert len(results) == _WORLD, (results, exit_codes)
    return sorted(results, key=lambda item: item["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="hybrid scoped failure-protocol coverage requires Gloo",
)
def test_hybrid_step_synchronizes_amp_and_preflight_across_the_scope():
    results = _run_distributed_workers()
    assert all("fatal_error" not in result for result in results), results

    # Rank-divergent found_inf must raise the children's collective AMP
    # agreement error on EVERY member (never a one-sided silent skip that
    # strands the peer inside a child's scope header collective), leaving
    # both children and the gradients untouched on both ranks.
    divergent = [result["amp_divergent"] for result in results]
    assert all(
        item["message"] is not None and "group-aware gradient scaler" in item["message"]
        for item in divergent
    ), divergent
    assert all(item["untouched"] and item["grads_untouched"] for item in divergent), divergent

    # A group-wide overflow skips BOTH children symmetrically on every
    # member: step returns through the post-hook path with no mutation and
    # no gradient unscale, and every member exits together (the barriers
    # after this scenario would otherwise desynchronize).
    group_wide = [result["amp_group_wide"] for result in results]
    assert all(
        item["skipped"] and item["post_hook_fired"] and item["untouched"] and item["grads_untouched"]
        for item in group_wide
    ), group_wide

    # An agreeing finite scale steps both children on every member and
    # unscales the union of both children's gradients exactly once: the AMP
    # step over 8x-scaled gradients is bit-identical to the plain step.
    finite = [result["amp_finite"] for result in results]
    assert all(item["stepped"] and item["unscaled_once_exactly"] for item in finite), finite

    # A rank-local structural preflight failure raises on every scope member
    # together through the children's synchronized failure protocol, keeping
    # the atomic both-children-skip semantics on both ranks.
    preflight = [result["preflight"] for result in results]
    assert preflight[0]["message"] is not None and preflight[1]["message"] is not None, preflight
    assert "gradient preflight failed on local member rank:0" in preflight[0]["message"]
    assert "sparse gradients" in preflight[0]["message"]
    assert "gradient preflight failed on another process-group member" in preflight[1]["message"]
    assert all(item["untouched"] for item in preflight), preflight

    # A rank-local closure failure (before any scope synchronization) raises on
    # every scope member together through the composite preamble synchronization
    # instead of stranding the peer inside a later scoped collective. Both ranks
    # exit with an error and leave both children untouched.
    closure_divergent = [result["closure_divergent"] for result in results]
    assert closure_divergent[0]["message"] is not None and closure_divergent[1]["message"] is not None, closure_divergent
    assert "closure boom on rank:0" in closure_divergent[0]["message"]
    assert "step preamble failed on another process-group member" in closure_divergent[1]["message"]
    assert all(item["untouched"] for item in closure_divergent), closure_divergent

    # Hybrid parent checkpoint phases vote through the installed scope before
    # the distributed-Muon child enters owner-state consolidation. A one-rank
    # pre-hook failure therefore raises on both members instead of hanging the
    # peer in the child's first broadcast.
    checkpoint_parent = [result["checkpoint_parent"] for result in results]
    assert all(item["message"] is not None for item in checkpoint_parent), checkpoint_parent
    assert "checkpoint pre-hook boom on rank:0" in checkpoint_parent[0]["message"]
    assert "state-dict pre-hook failed on another process-group member" in checkpoint_parent[1]["message"]


def _make_plain_hybrid():
    matrix = torch.nn.Parameter(_muon_initial().clone())
    vector = torch.nn.Parameter(_backup_initial().clone())
    optimizer = GefenMuonHybrid(
        [("matrix", matrix)],
        [("vector", vector)],
        **_hybrid_kwargs(),
    )
    return optimizer, matrix, vector


def test_unscoped_amp_overflow_and_finite_step_behavior_is_preserved():
    optimizer, matrix, vector = _make_plain_hybrid()
    matrix.grad = _muon_gradient().clone()
    vector.grad = _backup_gradient().clone()
    optimizer.found_inf = torch.tensor(1.0)
    optimizer.grad_scale = torch.tensor(4.0)
    assert optimizer.step() is None
    assert optimizer.muon._gefen_global_step == 0
    assert optimizer.backup._gefen_global_step == 0
    assert _bits_equal(matrix, _muon_initial())
    assert _bits_equal(vector, _backup_initial())
    assert _bits_equal(vector.grad, _backup_gradient())

    matrix.grad = _muon_gradient() * 4.0
    vector.grad = _backup_gradient() * 4.0
    optimizer.found_inf = torch.tensor(0.0)
    optimizer.grad_scale = torch.tensor(4.0)
    optimizer.step()
    assert optimizer.muon._gefen_global_step == 1
    assert optimizer.backup._gefen_global_step == 1
    assert not _bits_equal(matrix, _muon_initial())
    assert not _bits_equal(vector, _backup_initial())
    assert _bits_equal(vector.grad, _backup_gradient())


def test_unscoped_preflight_failure_raises_locally_and_skips_both_children():
    optimizer, matrix, vector = _make_plain_hybrid()
    matrix.grad = _muon_gradient().clone()
    vector.grad = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([1.0]),
        size=(8,),
    )
    with pytest.raises(RuntimeError) as excinfo:
        optimizer.step()
    assert "sparse gradients" in str(excinfo.value)
    assert "scoped Gefen codebook" not in str(excinfo.value)
    assert optimizer.muon._gefen_global_step == 0
    assert optimizer.backup._gefen_global_step == 0
    assert optimizer.muon._gefen_codebook is None
    assert optimizer.backup._gefen_codebook is None
    assert _bits_equal(matrix, _muon_initial())
    assert _bits_equal(vector, _backup_initial())


def _make_one_member_scoped_hybrid():
    group = ProcessGroupIdentity("solo", ("rank:0",))
    muon_identity = ParameterIdentity(_MUON_FQN, (3, 2))
    backup_identity = ParameterIdentity(_BACKUP_FQN, (8,))
    matrix = torch.nn.Parameter(_muon_initial().clone())
    vector = torch.nn.Parameter(_backup_initial().clone())
    optimizer = GefenMuonHybrid(
        [("matrix", matrix)],
        [("vector", vector)],
        **_hybrid_kwargs(),
    )

    def _replicated(identity):
        return ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
            placements=(ShardPlacement("dp", PlacementKind.REPLICATE, 0, 1),),
            process_group=group,
            local_member="rank:0",
        )

    muon_shard = _replicated(muon_identity)
    backup_shard = _replicated(backup_identity)
    binding = CodebookProcessGroupBinding(group, "rank:0", None, torch.device("cpu"))
    optimizer.post_sharding(
        (
            ParameterRebinding(matrix, matrix, muon_shard),
            ParameterRebinding(vector, vector, backup_shard),
        ),
        manifest=ShardingManifest((muon_shard, backup_shard)),
        codebook_process_group=binding,
    )
    return optimizer, matrix, vector


def test_one_member_scope_amp_skip_and_preflight_raise_stay_collective_free():
    optimizer, matrix, vector = _make_one_member_scoped_hybrid()
    matrix.grad = _muon_gradient().clone()
    vector.grad = _backup_gradient().clone()
    optimizer.found_inf = torch.tensor(1.0)
    optimizer.grad_scale = torch.tensor(2.0)
    assert optimizer.step() is None
    assert optimizer.muon._gefen_global_step == 0
    assert optimizer.backup._gefen_global_step == 0
    assert _bits_equal(matrix, _muon_initial())
    assert _bits_equal(vector, _backup_initial())

    vector.grad = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([1.0]),
        size=(8,),
    )
    del optimizer.found_inf
    del optimizer.grad_scale
    # A one-member binding synchronizes without collectives and re-raises the
    # local structural error unchanged, exactly like the children.
    with pytest.raises(RuntimeError) as excinfo:
        optimizer.step()
    assert "sparse gradients" in str(excinfo.value)
    assert optimizer.muon._gefen_global_step == 0
    assert optimizer.backup._gefen_global_step == 0

    matrix.grad = _muon_gradient() * 2.0
    vector.grad = _backup_gradient() * 2.0
    optimizer.found_inf = torch.tensor(0.0)
    optimizer.grad_scale = torch.tensor(2.0)
    optimizer.step()
    assert optimizer.muon._gefen_global_step == 1
    assert optimizer.backup._gefen_global_step == 1
    assert not _bits_equal(matrix, _muon_initial())
    assert not _bits_equal(vector, _backup_initial())
    assert _bits_equal(vector.grad, _backup_gradient())
