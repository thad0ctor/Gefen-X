from datetime import timedelta
import copy
import multiprocessing as mp
import os
import tempfile

import pytest
import torch
import torch.distributed as dist

from gefen import (
    CodebookProcessGroupBinding,
    Gefen,
    GefenMuon,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)
from gefen.gefen import learn_gefen_exact_codebook_from_grad_periods
import gefen.gefen as gefen_module


def _replicated(parameter, group, member):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.REPLICATED,
        LogicalSlice.full(parameter),
        process_group=group,
        local_member=member,
        placements=(
            ShardPlacement(
                "dp",
                PlacementKind.REPLICATE,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _flat(parameter, group, member, offset, length):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.FLATTENED_ELEMENT_SHARD,
        LogicalSlice(offset, length),
        process_group=group,
        local_member=member,
        placements=(
            ShardPlacement(
                "dp",
                PlacementKind.FLAT_SHARD,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _whole_owner(parameter, group, member, owner):
    coordinate = group.ordered_members.index(member)
    return ShardIdentity(
        parameter,
        ParameterLayout.WHOLE_PARAMETER_OWNER,
        LogicalSlice.full(parameter) if member == owner else LogicalSlice(0, 0),
        process_group=group,
        local_member=member,
        owner=owner,
        placements=(
            ShardPlacement(
                "dp",
                PlacementKind.WHOLE_PARAMETER_OWNER,
                coordinate,
                len(group.ordered_members),
            ),
        ),
    )


def _binding(group, rank, process_group):
    return CodebookProcessGroupBinding(group, "rank:{}".format(rank), process_group, torch.device("cpu"))


def _finalize(optimizer, parameter, local_shard, manifest, binding):
    optimizer.post_sharding(
        (ParameterRebinding(parameter, parameter, local_shard),),
        manifest=manifest,
        codebook_process_group=binding,
    )


def _distributed_worker(rank, world, init_file, queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=45),
        )
        members = tuple("rank:{}".format(index) for index in range(world))
        group = ProcessGroupIdentity("data_parallel", members)
        runtime_group = dist.group.WORLD

        # Runtime coordinate/member validation must fail before post_sharding
        # publication, then leave the same optimizer retryable with a valid map.
        invalid_param = torch.nn.Parameter(torch.ones(4))
        invalid_optimizer = Gefen([("invalid", invalid_param)], fused=False, factored_v_2d=False)
        invalid_identity = ParameterIdentity("Invalid", (4,))
        reversed_group = ProcessGroupIdentity("data_parallel", tuple(reversed(members)))
        invalid_records = tuple(
            _replicated(invalid_identity, reversed_group, member) for member in reversed_group.ordered_members
        )
        invalid_local = next(item for item in invalid_records if item.local_member == "rank:{}".format(rank))
        bad_binding = CodebookProcessGroupBinding(
            reversed_group,
            "rank:{}".format(rank),
            runtime_group,
            torch.device("cpu"),
        )
        try:
            invalid_optimizer.post_sharding(
                (ParameterRebinding(invalid_param, invalid_param, invalid_local),),
                manifest=ShardingManifest(invalid_records),
                codebook_process_group=bad_binding,
            )
            invalid_rejected = False
        except ValueError:
            invalid_rejected = True
        invalid_atomic = (
            not invalid_optimizer._gefen_post_sharding_finalized
            and invalid_optimizer._gefen_codebook_process_group is None
            and invalid_optimizer.param_groups[0]["params"] == [invalid_param]
        )

        # Replicated logical state contributes exactly once from ordered member
        # zero. Rank 1 intentionally has different values but the same period;
        # including both replicas would produce a different histogram oracle.
        replicated_param = torch.nn.Parameter(torch.zeros(4))
        replicated_optimizer = Gefen(
            [("replicated", replicated_param)],
            fused=False,
            factored_v_2d=False,
        )
        replicated_identity = ParameterIdentity("Replicated", (4,))
        replicated_records = tuple(_replicated(replicated_identity, group, member) for member in members)
        _finalize(
            replicated_optimizer,
            replicated_param,
            replicated_records[rank],
            ShardingManifest(replicated_records),
            _binding(group, rank, runtime_group),
        )
        replicated_optimizer._resolve_automatic_period = lambda *args: 4
        canonical_grad = torch.tensor([-4.0, -1.0, 2.0, 8.0])
        other_grad = torch.tensor([-8.0, 3.0, 4.0, 4.0])
        replicated_param.grad = canonical_grad.clone() if rank == 0 else other_grad
        replicated_initialized = replicated_optimizer.initialize_codebook()
        replicated_oracle = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=(("Replicated", canonical_grad, 4, canonical_grad),),
            codebook_device=torch.device("cpu"),
            num_codebooks=256,
            force_endpoints=True,
            verbose=False,
            compute_mse_logging=False,
            use_fused_histogram=False,
        )
        replicated_logical_once = torch.equal(replicated_optimizer._gefen_codebook, replicated_oracle)

        mismatch_param = torch.nn.Parameter(torch.zeros(4))
        mismatch_optimizer = Gefen([("mismatch", mismatch_param)], fused=False, factored_v_2d=False)
        mismatch_identity = ParameterIdentity("Mismatch", (4,))
        mismatch_records = tuple(_replicated(mismatch_identity, group, member) for member in members)
        _finalize(
            mismatch_optimizer,
            mismatch_param,
            mismatch_records[rank],
            ShardingManifest(mismatch_records),
            _binding(group, rank, runtime_group),
        )
        mismatch_optimizer._resolve_automatic_period = lambda *args: 4
        mismatch_param.grad = canonical_grad.clone() if rank == 0 else None
        try:
            mismatch_optimizer.initialize_codebook()
            mismatch_rejected = False
        except RuntimeError as exc:
            mismatch_rejected = "identical gradient presence" in str(exc)
        mismatch_atomic = (
            mismatch_optimizer._gefen_codebook is None
            and mismatch_optimizer._gefen_global_step == 0
            and mismatch_optimizer.state[mismatch_param] == {"name": "mismatch"}
        )

        amp_param = torch.nn.Parameter(torch.zeros(4))
        amp_optimizer = Gefen([("amp", amp_param)], fused=False, factored_v_2d=False)
        amp_identity = ParameterIdentity("Amp", (8,))
        amp_records = tuple(_flat(amp_identity, group, member, index * 4, 4) for index, member in enumerate(members))
        _finalize(
            amp_optimizer,
            amp_param,
            amp_records[rank],
            ShardingManifest(amp_records),
            _binding(group, rank, runtime_group),
        )
        amp_optimizer._resolve_automatic_period = lambda *args: 4
        amp_param.grad = canonical_grad.clone() if rank == 0 else other_grad.clone()
        amp_grad_before = amp_param.grad.detach().clone()
        amp_optimizer.found_inf = torch.tensor(float(rank == 0))
        amp_optimizer.grad_scale = torch.tensor(8.0)
        try:
            amp_optimizer.step()
            amp_mismatch_rejected = False
        except RuntimeError as exc:
            amp_mismatch_rejected = "group-aware gradient scaler" in str(exc)
        amp_optimizer.found_inf = torch.tensor(1.0)
        amp_optimizer.step()
        amp_overflow_atomic = (
            amp_optimizer._gefen_global_step == 0
            and amp_optimizer._gefen_codebook is None
            and amp_optimizer.state[amp_param] == {"name": "amp"}
            and torch.equal(amp_param.grad, amp_grad_before)
            and torch.equal(amp_param, torch.zeros_like(amp_param))
        )
        if rank == 0:
            amp_optimizer.found_inf = torch.tensor(0.0)
        else:
            del amp_optimizer.found_inf
            del amp_optimizer.grad_scale
        try:
            amp_optimizer.step()
            amp_protocol_rejected = False
        except RuntimeError as exc:
            amp_protocol_rejected = "policy" in str(exc)

        # Flattened shards each contribute once. The global oracle uses two
        # period-4 blocks, exactly matching the two physical local shards.
        flat_param = torch.nn.Parameter(torch.zeros(4))
        flat_optimizer = Gefen([("flat", flat_param)], fused=False, factored_v_2d=False)
        flat_identity = ParameterIdentity("Flat", (8,))
        flat_records = tuple(_flat(flat_identity, group, member, index * 4, 4) for index, member in enumerate(members))
        _finalize(
            flat_optimizer,
            flat_param,
            flat_records[rank],
            ShardingManifest(flat_records),
            _binding(group, rank, runtime_group),
        )
        flat_optimizer._resolve_automatic_period = lambda *args: 4
        flat_grads = (
            torch.tensor([-3.0, -2.0, 1.0, 7.0]),
            torch.tensor([-9.0, 2.0, 5.0, 6.0]),
        )
        flat_param.grad = flat_grads[rank].clone()
        flat_initialized = flat_optimizer.initialize_codebook()
        flat_oracle = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=tuple(("Flat", gradient, 4, gradient) for gradient in flat_grads),
            codebook_device=torch.device("cpu"),
            num_codebooks=256,
            force_endpoints=True,
            verbose=False,
            compute_mse_logging=False,
            use_fused_histogram=False,
        )
        flat_global = torch.equal(flat_optimizer._gefen_codebook, flat_oracle)
        flat_native_claim = any(
            ParameterLayout.FLATTENED_ELEMENT_SHARD in item.same_topology
            for item in flat_optimizer.optimizer_contract().capabilities.checkpoints
        )

        flat_mismatch_param = torch.nn.Parameter(torch.zeros(4))
        flat_mismatch_optimizer = Gefen(
            [("flat_mismatch", flat_mismatch_param)],
            fused=False,
            factored_v_2d=False,
        )
        flat_mismatch_identity = ParameterIdentity("FlatMismatch", (8,))
        flat_mismatch_records = tuple(
            _flat(flat_mismatch_identity, group, member, index * 4, 4) for index, member in enumerate(members)
        )
        _finalize(
            flat_mismatch_optimizer,
            flat_mismatch_param,
            flat_mismatch_records[rank],
            ShardingManifest(flat_mismatch_records),
            _binding(group, rank, runtime_group),
        )
        flat_mismatch_optimizer._resolve_automatic_period = lambda *args: 4
        flat_mismatch_param.grad = flat_grads[rank].clone() if rank == 0 else None
        try:
            flat_mismatch_optimizer.initialize_codebook()
            flat_mismatch_rejected = False
        except RuntimeError as exc:
            flat_mismatch_rejected = "every nonempty shard" in str(exc)
        flat_mismatch_atomic = flat_mismatch_optimizer._gefen_codebook is None and flat_mismatch_optimizer.state[
            flat_mismatch_param
        ] == {"name": "flat_mismatch"}

        gathered_codebooks = [torch.empty_like(flat_optimizer._gefen_codebook) for _ in range(world)]
        dist.all_gather(gathered_codebooks, flat_optimizer._gefen_codebook)
        flat_agreement = all(torch.equal(item, gathered_codebooks[0]) for item in gathered_codebooks[1:])
        flat_optimizer.step()
        flat_checkpoint = copy.deepcopy(flat_optimizer.state_dict())
        flat_canonical = flat_optimizer.export_canonical_state()
        checkpoint_scopes = [None] * world
        dist.all_gather_object(
            checkpoint_scopes,
            flat_checkpoint["gefen_codebook_scope"],
            group=runtime_group,
        )
        rank_neutral_checkpoint_scope = all(item == checkpoint_scopes[0] for item in checkpoint_scopes[1:])
        local_shard_records = [None] * world
        dist.all_gather_object(
            local_shard_records,
            flat_checkpoint["gefen_native_local_shards"],
            group=runtime_group,
        )
        rank_local_checkpoint_identity = local_shard_records[0] != local_shard_records[1]
        flat_v2_guard_valid = (
            flat_checkpoint["gefen_native_local_shards"]["format_version"] == 2
            and len(flat_checkpoint["gefen_native_local_shards"]["logical_slots"])
            == 1
            and flat_checkpoint["gefen_native_local_shards"]["logical_slots"][0][
                "group_index"
            ]
            == 0
            and flat_checkpoint["gefen_native_local_shards"]["logical_slots"][0][
                "original_slot_index"
            ]
            == 0
            and flat_checkpoint["gefen_native_local_shards"]["logical_slots"][0][
                "compatibility_name"
            ]
            == "flat"
            and flat_checkpoint["gefen_native_local_shards"]["logical_slots"][0][
                "shard"
            ]["local_member"]
            == "rank:{}".format(rank)
        )
        rank_zero_checkpoint = [flat_checkpoint if rank == 0 else None]
        dist.broadcast_object_list(rank_zero_checkpoint, src=0, group=runtime_group)
        cross_param = torch.nn.Parameter(flat_param.detach().clone())
        cross_target = Gefen([("flat", cross_param)], fused=False, factored_v_2d=False)
        _finalize(
            cross_target,
            cross_param,
            flat_records[rank],
            ShardingManifest(flat_records),
            _binding(group, rank, runtime_group),
        )
        try:
            cross_target.load_state_dict(rank_zero_checkpoint[0])
            cross_member_guard = rank == 0
        except ValueError as exc:
            cross_member_guard = rank == 1 and "local-shard identity" in str(exc)
        canonical_shards = [None] * world
        dist.all_gather_object(
            canonical_shards,
            flat_canonical["parameters"]["Flat"]["shard"],
            group=runtime_group,
        )
        canonical_rank_local_identity = canonical_shards[0] != canonical_shards[1]
        rank_zero_canonical = [flat_canonical if rank == 0 else None]
        dist.broadcast_object_list(rank_zero_canonical, src=0, group=runtime_group)
        cross_canonical_param = torch.nn.Parameter(flat_param.detach().clone())
        cross_canonical = Gefen(
            [("flat", cross_canonical_param)],
            fused=False,
            factored_v_2d=False,
        )
        _finalize(
            cross_canonical,
            cross_canonical_param,
            flat_records[rank],
            ShardingManifest(flat_records),
            _binding(group, rank, runtime_group),
        )
        try:
            cross_canonical.import_canonical_state(rank_zero_canonical[0])
            canonical_cross_member_guard = rank == 0
        except ValueError as exc:
            canonical_cross_member_guard = rank == 1 and "shard" in str(exc)
        resumed_param = torch.nn.Parameter(flat_param.detach().clone())
        resumed = Gefen([("flat", resumed_param)], fused=False, factored_v_2d=False)
        resumed_binding = _binding(group, rank, runtime_group)
        _finalize(
            resumed,
            resumed_param,
            flat_records[rank],
            ShardingManifest(flat_records),
            resumed_binding,
        )
        resumed.load_state_dict(flat_checkpoint)
        legacy_checkpoint = copy.deepcopy(flat_checkpoint)
        legacy_guard = flat_optimizer._serialized_native_local_shards_v1()
        legacy_checkpoint["gefen_native_local_shards"] = legacy_guard
        for legacy_group in legacy_checkpoint["param_groups"]:
            legacy_group["_gefen_checkpoint_metadata"][
                "native_local_shards"
            ] = copy.deepcopy(legacy_guard)
        legacy_param = torch.nn.Parameter(flat_param.detach().clone())
        legacy_resumed = Gefen(
            [("flat", legacy_param)],
            fused=False,
            factored_v_2d=False,
        )
        legacy_binding = _binding(group, rank, runtime_group)
        _finalize(
            legacy_resumed,
            legacy_param,
            flat_records[rank],
            ShardingManifest(flat_records),
            legacy_binding,
        )
        legacy_resumed.load_state_dict(legacy_checkpoint)
        legacy_v1_guard_accepted = (
            legacy_resumed.codebook_process_group_binding() is legacy_binding
            and torch.equal(
                legacy_resumed._gefen_codebook,
                flat_optimizer._gefen_codebook,
            )
        )
        canonical_param = torch.nn.Parameter(flat_param.detach().clone())
        canonical_resumed = Gefen(
            [("flat", canonical_param)],
            fused=False,
            factored_v_2d=False,
        )
        canonical_binding = _binding(group, rank, runtime_group)
        _finalize(
            canonical_resumed,
            canonical_param,
            flat_records[rank],
            ShardingManifest(flat_records),
            canonical_binding,
        )
        canonical_resumed.import_canonical_state(flat_canonical)
        continuation_grad = flat_grads[rank].flip(0).clone()
        resumed_param.grad = continuation_grad.clone()
        canonical_param.grad = continuation_grad.clone()
        flat_param.grad = continuation_grad.clone()
        canonical_resumed.step()
        resumed.step()
        flat_optimizer.step()
        flat_checkpoint_continuation = (
            torch.equal(resumed_param, flat_param)
            and resumed.codebook_process_group_binding() is resumed_binding
            and torch.equal(resumed._gefen_codebook, flat_optimizer._gefen_codebook)
        )
        flat_canonical_continuation = (
            torch.equal(canonical_param, flat_param)
            and canonical_resumed.codebook_process_group_binding() is canonical_binding
            and torch.equal(
                canonical_resumed._gefen_codebook,
                flat_optimizer._gefen_codebook,
            )
        )
        refresh_succeeded = flat_optimizer.refresh_codebook()
        continuation_grads = tuple(gradient.flip(0) for gradient in flat_grads)
        refresh_oracle = learn_gefen_exact_codebook_from_grad_periods(
            grad_periods=tuple(("Flat", gradient, 4, gradient) for gradient in continuation_grads),
            codebook_device=torch.device("cpu"),
            num_codebooks=256,
            force_endpoints=True,
            verbose=False,
            compute_mse_logging=False,
            use_fused_histogram=False,
        )
        refresh_matches_oracle = torch.equal(flat_optimizer._gefen_codebook, refresh_oracle)
        refreshed_codebooks = [torch.empty_like(flat_optimizer._gefen_codebook) for _ in range(world)]
        dist.all_gather(refreshed_codebooks, flat_optimizer._gefen_codebook)
        refresh_agreement = all(torch.equal(item, refreshed_codebooks[0]) for item in refreshed_codebooks[1:])
        refresh_failure_grad = flat_grads[rank].roll(1).clone()
        flat_param.grad = refresh_failure_grad
        old_refresh_codebook_object = flat_optimizer._gefen_codebook
        old_refresh_codebook = flat_optimizer._gefen_codebook.detach().clone()
        old_refresh_indices = flat_optimizer.state[flat_param]["m_codebook"].detach().clone()
        old_refresh_magnitude = flat_optimizer.state[flat_param]["m_magnitude"].detach().clone()
        old_refresh_period = flat_optimizer.state[flat_param]["automatic_period"]
        old_refresh_step = flat_optimizer._gefen_global_step
        codebook_cache = flat_optimizer._gefen_codebook_by_device
        lut_cache = flat_optimizer._gefen_codebook_lut_by_device
        cache_marker = torch.tensor([17.0])
        lut_marker = torch.tensor([19.0])
        codebook_cache[torch.device("cpu")] = cache_marker
        lut_cache[torch.device("cpu")] = lut_marker
        original_nearest = gefen_module.gefen_nearest_codebook_indices
        if rank == 1:

            def fail_nearest(*args, **kwargs):
                raise RuntimeError("rank-local requantization failure")

            gefen_module.gefen_nearest_codebook_indices = fail_nearest
        try:
            flat_optimizer.refresh_codebook()
            refresh_failure_seen = False
        except RuntimeError as exc:
            refresh_failure_seen = "requantization" in str(exc)
        finally:
            gefen_module.gefen_nearest_codebook_indices = original_nearest
        refresh_failure_atomic = (
            flat_optimizer._gefen_codebook is old_refresh_codebook_object
            and torch.equal(flat_optimizer._gefen_codebook, old_refresh_codebook)
            and torch.equal(
                flat_optimizer.state[flat_param]["m_codebook"],
                old_refresh_indices,
            )
            and torch.equal(
                flat_optimizer.state[flat_param]["m_magnitude"],
                old_refresh_magnitude,
            )
            and flat_optimizer.state[flat_param]["automatic_period"] == old_refresh_period
            and flat_optimizer._gefen_global_step == old_refresh_step
            and flat_optimizer._gefen_codebook_by_device is codebook_cache
            and flat_optimizer._gefen_codebook_lut_by_device is lut_cache
            and codebook_cache[torch.device("cpu")] is cache_marker
            and lut_cache[torch.device("cpu")] is lut_marker
        )
        refresh_retry_succeeded = flat_optimizer.refresh_codebook()

        periodic_param = torch.nn.Parameter(torch.zeros(4))
        periodic_optimizer = Gefen(
            [("periodic", periodic_param)],
            fused=False,
            factored_v_2d=False,
            codebook_refresh_every=2,
        )
        periodic_identity = ParameterIdentity("Periodic", (8,))
        periodic_records = tuple(
            _flat(periodic_identity, group, member, index * 4, 4) for index, member in enumerate(members)
        )
        _finalize(
            periodic_optimizer,
            periodic_param,
            periodic_records[rank],
            ShardingManifest(periodic_records),
            _binding(group, rank, runtime_group),
        )
        periodic_optimizer._resolve_automatic_period = lambda *args: 4
        periodic_param.grad = flat_grads[rank].clone()
        periodic_optimizer.step()
        first_periodic_codebook = periodic_optimizer._gefen_codebook

        if rank == 0:
            periodic_param.grad = torch.sparse_coo_tensor(
                torch.tensor([[0]]),
                torch.tensor([1.0]),
                size=(4,),
            )
        else:
            periodic_param.grad = flat_grads[rank].roll(1).clone()
        try:
            periodic_optimizer.step()
            periodic_nondue_failure_rejected = False
        except RuntimeError as exc:
            periodic_nondue_failure_rejected = "gradient preflight" in str(exc)
        periodic_nondue_failure_rejected = (
            periodic_nondue_failure_rejected
            and periodic_optimizer._gefen_global_step == 1
            and periodic_optimizer._gefen_codebook is first_periodic_codebook
        )

        periodic_param.grad = flat_grads[rank].roll(1).clone()
        periodic_optimizer.step()
        periodic_nondue_failure_rejected = (
            periodic_nondue_failure_rejected
            and periodic_optimizer._gefen_global_step == 2
            and periodic_optimizer._gefen_codebook is first_periodic_codebook
        )
        periodic_param.grad = flat_grads[rank].flip(0).clone()
        periodic_optimizer.step()
        periodic_codebooks = [torch.empty_like(periodic_optimizer._gefen_codebook) for _ in range(world)]
        dist.all_gather(periodic_codebooks, periodic_optimizer._gefen_codebook)
        periodic_refresh_valid = (
            periodic_optimizer._gefen_global_step == 3
            and periodic_optimizer._gefen_codebook is not first_periodic_codebook
            and all(torch.equal(item, periodic_codebooks[0]) for item in periodic_codebooks[1:])
            and periodic_optimizer.state_dict()["gefen_codebook_scope"]["refresh_every"] == 2
        )

        # Whole-matrix owner mode keeps no fake tensor/state on nonowners. Every
        # member still joins codebook collectives and receives common state;
        # the adapter performs the separately declared post-step matrix sync.
        owner_source = torch.nn.Parameter(torch.ones(2, 2))
        owner_optimizer = GefenMuon([("matrix", owner_source)], fused=False)
        owner_identity = ParameterIdentity("Matrix", (2, 2))
        owner_records = tuple(_whole_owner(owner_identity, group, member, "rank:0") for member in members)
        owner_optimizer.post_sharding(
            (
                ParameterRebinding(
                    owner_source,
                    owner_source if rank == 0 else None,
                    owner_records[rank],
                ),
            ),
            manifest=ShardingManifest(owner_records),
            codebook_process_group=_binding(group, rank, runtime_group),
        )
        if rank == 0:
            owner_source.grad = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        owner_initialized = owner_optimizer.initialize_codebook()
        owner_codebooks = [torch.empty_like(owner_optimizer._gefen_codebook) for _ in range(world)]
        dist.all_gather(owner_codebooks, owner_optimizer._gefen_codebook)
        owner_agreement = all(torch.equal(item, owner_codebooks[0]) for item in owner_codebooks[1:])
        owner_optimizer.step()
        synchronized_matrix = owner_source.detach().clone() if rank == 0 else torch.empty(2, 2, dtype=torch.float32)
        dist.broadcast(synchronized_matrix, src=0, group=runtime_group)
        owner_step_valid = (
            owner_optimizer._gefen_global_step == 1
            and not torch.equal(synchronized_matrix, torch.ones_like(synchronized_matrix))
            and (rank != 0 or torch.equal(owner_source, synchronized_matrix))
            and (rank == 0 or (not owner_optimizer.param_groups[0]["params"] and not owner_optimizer.state))
        )
        owner_support = next(
            item
            for item in owner_optimizer.optimizer_contract().capabilities.training
            if item.layout is ParameterLayout.WHOLE_PARAMETER_OWNER
        )
        owner_contract_valid = owner_support.requires_post_step_parameter_sync
        owner_amp_protocol = owner_optimizer._step_supports_amp_scaling

        # A failure after the shared histogram reduction must be observed by
        # every member before staged periods or the codebook are committed.
        failure_param = torch.nn.Parameter(torch.zeros(4))
        failure_optimizer = Gefen([("failure", failure_param)], fused=False, factored_v_2d=False)
        failure_identity = ParameterIdentity("Failure", (8,))
        failure_records = tuple(
            _flat(failure_identity, group, member, index * 4, 4) for index, member in enumerate(members)
        )
        _finalize(
            failure_optimizer,
            failure_param,
            failure_records[rank],
            ShardingManifest(failure_records),
            _binding(group, rank, runtime_group),
        )
        failure_optimizer._resolve_automatic_period = lambda *args: 4
        failure_param.grad = flat_grads[rank].clone()
        original_exact_dp = gefen_module.quantization_module.exact_dp
        if rank == 1:

            def fail_exact_dp(*args, **kwargs):
                raise RuntimeError("rank-local exact-DP failure")

            gefen_module.quantization_module.exact_dp = fail_exact_dp
        try:
            failure_optimizer.initialize_codebook()
            failure_seen = False
        except RuntimeError as exc:
            failure_seen = "exact-DP" in str(exc)
        finally:
            gefen_module.quantization_module.exact_dp = original_exact_dp
        failure_atomic = (
            failure_optimizer._gefen_codebook is None
            and failure_optimizer._gefen_global_step == 0
            and failure_optimizer.state[failure_param] == {"name": "failure"}
        )
        retry_succeeded = failure_optimizer.initialize_codebook()
        restored_codebook = failure_optimizer._gefen_codebook
        if rank == 0:
            failure_optimizer._gefen_codebook = None
        try:
            failure_optimizer.refresh_codebook()
            presence_mismatch_rejected = False
        except RuntimeError as exc:
            presence_mismatch_rejected = "old codebook differs" in str(exc)
        if rank == 0:
            failure_optimizer._gefen_codebook = restored_codebook
        original_manifest = failure_optimizer._gefen_sharding_manifest
        failure_optimizer._gefen_codebook_scope_validated = False
        manifest_guard_codebook = failure_optimizer._gefen_codebook
        manifest_guard_step = failure_optimizer._gefen_global_step
        if rank == 0:
            alternate_identity = ParameterIdentity("Alternate", (8,))
            alternate_records = tuple(
                _flat(alternate_identity, group, member, index * 4, 4) for index, member in enumerate(members)
            )
            failure_optimizer._gefen_sharding_manifest = ShardingManifest(alternate_records)
        try:
            failure_optimizer.initialize_codebook()
            manifest_mismatch_rejected = False
        except RuntimeError:
            manifest_mismatch_rejected = True
        manifest_mismatch_rejected = (
            manifest_mismatch_rejected
            and failure_optimizer._gefen_codebook is manifest_guard_codebook
            and failure_optimizer._gefen_global_step == manifest_guard_step
        )
        failure_optimizer._gefen_sharding_manifest = original_manifest

        queue.put(
            {
                "rank": rank,
                "invalid_rejected": invalid_rejected,
                "invalid_atomic": invalid_atomic,
                "replicated_initialized": replicated_initialized,
                "replicated_logical_once": replicated_logical_once,
                "mismatch_rejected": mismatch_rejected,
                "mismatch_atomic": mismatch_atomic,
                "amp_overflow_atomic": amp_overflow_atomic,
                "amp_mismatch_rejected": amp_mismatch_rejected,
                "amp_protocol_rejected": amp_protocol_rejected,
                "flat_initialized": flat_initialized,
                "flat_global": flat_global,
                "flat_native_claim": flat_native_claim,
                "flat_mismatch_rejected": flat_mismatch_rejected,
                "flat_mismatch_atomic": flat_mismatch_atomic,
                "flat_agreement": flat_agreement,
                "flat_checkpoint_continuation": flat_checkpoint_continuation,
                "rank_neutral_checkpoint_scope": rank_neutral_checkpoint_scope,
                "rank_local_checkpoint_identity": rank_local_checkpoint_identity,
                "flat_v2_guard_valid": flat_v2_guard_valid,
                "legacy_v1_guard_accepted": legacy_v1_guard_accepted,
                "cross_member_guard": cross_member_guard,
                "canonical_rank_local_identity": canonical_rank_local_identity,
                "canonical_cross_member_guard": canonical_cross_member_guard,
                "flat_canonical_continuation": flat_canonical_continuation,
                "refresh_succeeded": refresh_succeeded,
                "refresh_matches_oracle": refresh_matches_oracle,
                "refresh_agreement": refresh_agreement,
                "refresh_failure_seen": refresh_failure_seen,
                "refresh_failure_atomic": refresh_failure_atomic,
                "refresh_retry_succeeded": refresh_retry_succeeded,
                "periodic_nondue_failure_rejected": periodic_nondue_failure_rejected,
                "periodic_refresh_valid": periodic_refresh_valid,
                "owner_initialized": owner_initialized,
                "owner_agreement": owner_agreement,
                "owner_step_valid": owner_step_valid,
                "owner_contract_valid": owner_contract_valid,
                "owner_amp_protocol": owner_amp_protocol,
                "failure_seen": failure_seen,
                "failure_atomic": failure_atomic,
                "retry_succeeded": retry_succeeded,
                "presence_mismatch_rejected": presence_mismatch_rejected,
                "manifest_mismatch_rejected": manifest_mismatch_rejected,
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_workers(world=2):
    context = mp.get_context("spawn")
    queue = context.Queue()
    fd, init_file = tempfile.mkstemp(prefix="gefen-codebook-scope-")
    os.close(fd)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, world, init_file, queue),
        )
        for rank in range(world)
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
                pytest.fail("scoped codebook worker hung")
            assert process.exitcode == 0
        return sorted(results, key=lambda item: item["rank"])
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_explicit_gloo_scope_aggregates_logical_state_and_fails_atomically():
    results = _run_workers()

    assert all("error" not in item for item in results), results
    for item in results:
        assert item["invalid_rejected"], item
        assert item["invalid_atomic"], item
        assert item["replicated_initialized"], item
        assert item["replicated_logical_once"], item
        assert item["mismatch_rejected"], item
        assert item["mismatch_atomic"], item
        assert item["amp_overflow_atomic"], item
        assert item["amp_mismatch_rejected"], item
        assert item["amp_protocol_rejected"], item
        assert item["flat_initialized"], item
        assert item["flat_global"], item
        assert item["flat_native_claim"], item
        assert item["flat_mismatch_rejected"], item
        assert item["flat_mismatch_atomic"], item
        assert item["flat_agreement"], item
        assert item["flat_checkpoint_continuation"], item
        assert item["rank_neutral_checkpoint_scope"], item
        assert item["rank_local_checkpoint_identity"], item
        assert item["flat_v2_guard_valid"], item
        assert item["legacy_v1_guard_accepted"], item
        assert item["cross_member_guard"], item
        assert item["canonical_rank_local_identity"], item
        assert item["canonical_cross_member_guard"], item
        assert item["flat_canonical_continuation"], item
        assert item["refresh_succeeded"], item
        assert item["refresh_matches_oracle"], item
        assert item["refresh_agreement"], item
        assert item["refresh_failure_seen"], item
        assert item["refresh_failure_atomic"], item
        assert item["refresh_retry_succeeded"], item
        assert item["periodic_nondue_failure_rejected"], item
        assert item["periodic_refresh_valid"], item
        assert item["owner_initialized"], item
        assert item["owner_agreement"], item
        assert item["owner_step_valid"], item
        assert item["owner_contract_valid"], item
        assert item["owner_amp_protocol"], item
        assert item["failure_seen"], item
        assert item["failure_atomic"], item
        assert item["retry_succeeded"], item
        assert item["presence_mismatch_rejected"], item
        assert item["manifest_mismatch_rejected"], item


def _subgroup_worker(rank, world, init_file, queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=45),
        )
        first_ranks = (0, 2)
        second_ranks = (1, 3)
        first_group = dist.new_group(ranks=list(first_ranks))
        second_group = dist.new_group(ranks=list(second_ranks))
        if rank in first_ranks:
            group_index = 0
            group_ranks = first_ranks
            runtime_group = first_group
            canonical_grad = torch.tensor([-1.0, -0.5, 0.25, 1.0])
        else:
            group_index = 1
            group_ranks = second_ranks
            runtime_group = second_group
            canonical_grad = torch.tensor([-1.0, -0.9, 0.8, 1.0])
        coordinate = group_ranks.index(rank)
        members = ("slot:0", "slot:1")
        identity = ProcessGroupIdentity("replica_subgroup:{}".format(group_index), members)
        parameter_identity = ParameterIdentity("Subgroup{}.Weight".format(group_index), (4,))
        records = tuple(_replicated(parameter_identity, identity, member) for member in members)
        parameter = torch.nn.Parameter(torch.zeros(4))
        optimizer = Gefen([("weight", parameter)], fused=False, factored_v_2d=False)
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, records[coordinate]),),
            manifest=ShardingManifest(records),
            codebook_process_group=CodebookProcessGroupBinding(
                identity,
                members[coordinate],
                runtime_group,
                torch.device("cpu"),
            ),
        )
        optimizer._resolve_automatic_period = lambda *args: 4
        parameter.grad = canonical_grad.clone() if coordinate == 0 else torch.tensor([-1.0, -0.2, 0.1, 1.0])
        optimizer.initialize_codebook()
        queue.put(
            {
                "rank": rank,
                "group": group_index,
                "codebook": optimizer._gefen_codebook.tolist(),
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_subgroup_workers():
    context = mp.get_context("spawn")
    queue = context.Queue()
    fd, init_file = tempfile.mkstemp(prefix="gefen-codebook-subgroups-")
    os.close(fd)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_subgroup_worker,
            args=(rank, 4, init_file, queue),
        )
        for rank in range(4)
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
                pytest.fail("scoped codebook subgroup worker hung")
            assert process.exitcode == 0
        return sorted(results, key=lambda item: item["rank"])
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_explicit_gloo_subgroups_are_isolated_from_default_world():
    results = _run_subgroup_workers()

    assert all("error" not in item for item in results), results
    first = [item["codebook"] for item in results if item["group"] == 0]
    second = [item["codebook"] for item in results if item["group"] == 1]
    assert first[0] == first[1]
    assert second[0] == second[1]
    assert first[0] != second[0]


def _nccl_empty_owner_worker(rank, world, init_file, queue):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            init_method="file://{}".format(init_file),
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=60),
        )
        members = tuple("rank:{}".format(index) for index in range(world))
        group = ProcessGroupIdentity("nccl_owner", members)
        identity = ParameterIdentity("Matrix", (2, 2))
        records = tuple(_whole_owner(identity, group, member, "rank:0") for member in members)
        source = torch.nn.Parameter(torch.ones(2, 2, device=torch.device("cuda", rank)))
        optimizer = GefenMuon([("matrix", source)], fused=False)
        optimizer.post_sharding(
            (
                ParameterRebinding(
                    source,
                    source if rank == 0 else None,
                    records[rank],
                ),
            ),
            manifest=ShardingManifest(records),
            codebook_process_group=CodebookProcessGroupBinding(
                group,
                members[rank],
                dist.group.WORLD,
                torch.device("cuda", rank),
            ),
        )
        if rank == 0:
            source.grad = torch.tensor([[1.0, -2.0], [3.0, -4.0]], device=source.device)
        initialized = optimizer.initialize_codebook()
        local = optimizer._gefen_codebook.to(torch.device("cuda", rank))
        gathered = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        agreement = all(torch.equal(item, gathered[0]) for item in gathered[1:])
        optimizer.step()
        queue.put(
            {
                "rank": rank,
                "initialized": initialized,
                "agreement": agreement,
                "empty_nonowner": rank == 0
                or (
                    not optimizer.param_groups[0]["params"]
                    and not optimizer.state
                    and optimizer._gefen_codebook.device.type == "cpu"
                ),
                "step": optimizer._gefen_global_step,
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2,
    reason="requires NCCL and two CUDA devices",
)
def test_nccl_scope_uses_explicit_collective_device_with_empty_nonowner():
    context = mp.get_context("spawn")
    queue = context.Queue()
    fd, init_file = tempfile.mkstemp(prefix="gefen-codebook-nccl-")
    os.close(fd)
    os.unlink(init_file)
    processes = [
        context.Process(
            target=_nccl_empty_owner_worker,
            args=(rank, 2, init_file, queue),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=90) for _ in processes]
        for process in processes:
            process.join(timeout=15)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                pytest.fail("NCCL scoped codebook worker hung")
            assert process.exitcode == 0
        results.sort(key=lambda item: item["rank"])
        assert all("error" not in item for item in results), results
        assert all(item["initialized"] for item in results)
        assert all(item["agreement"] for item in results)
        assert all(item["empty_nonowner"] for item in results)
        assert all(item["step"] == 1 for item in results)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if os.path.exists(init_file):
            os.unlink(init_file)
