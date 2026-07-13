"""Scope-agreed codebook decisions and canonical import parameter freshness."""

from datetime import timedelta
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

from _state_snapshot import assert_deep_state_snapshot, deep_state_snapshot


_WORLD = 2


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


def _init_worker(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method="file://{}".format(init_file),
        rank=rank,
        world_size=_WORLD,
        timeout=timedelta(seconds=90),
    )
    members = tuple("rank:{}".format(index) for index in range(_WORLD))
    return ProcessGroupIdentity("data_parallel", members), members


def _reuse_decision_worker(rank, init_file, queue):
    """Rank-asymmetric automatic_period presence must not gate collectives.

    Reproduces the documented FSDP optim-state consolidation resume: the
    loader restores per-parameter state but strips the optimizer-common
    codebook on every member. A whole-parameter owner then holds restored
    periods while empty non-owner members hold no state at all, so the
    rank-local _resuming_from_checkpoint() predicate diverges. The group
    must still resolve ONE reuse_existing_periods decision before the
    conditionally issued "initialize" operation header.
    """

    try:
        group, members = _init_worker(rank, init_file)
        runtime_group = dist.group.WORLD

        owner_source = torch.nn.Parameter(torch.ones(2, 2))
        optimizer = GefenMuon([("matrix", owner_source)], fused=False)
        identity = ParameterIdentity("Matrix", (2, 2))
        records = tuple(
            _whole_owner(identity, group, member, "rank:0") for member in members
        )
        optimizer.post_sharding(
            (
                ParameterRebinding(
                    owner_source,
                    owner_source if rank == 0 else None,
                    records[rank],
                ),
            ),
            manifest=ShardingManifest(records),
            codebook_process_group=CodebookProcessGroupBinding(
                group, members[rank], runtime_group, torch.device("cpu")
            ),
        )
        if rank == 0:
            owner_source.grad = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        optimizer.step()
        first_period = (
            int(optimizer.state[owner_source]["automatic_period"])
            if rank == 0
            else None
        )
        state_asymmetric = (
            bool(optimizer.state) if rank == 0 else not optimizer.state
        )

        # Simulate the consolidation loader on every member: per-parameter
        # state survives, the optimizer-common codebook does not.
        optimizer._gefen_codebook = None
        optimizer._gefen_codebook_by_device.clear()
        optimizer._gefen_codebook_lut_by_device.clear()
        optimizer._gefen_codebook_scope_validated = False

        if rank == 0:
            owner_source.grad = torch.tensor([[0.5, -1.5], [2.5, -3.5]])

        # Capture the reuse decision each member would gate collectives on,
        # then abort symmetrically before any divergent collective runs.
        captured = []

        def capture(reuse_existing_periods=False):
            captured.append(bool(reuse_existing_periods))
            raise RuntimeError("captured reuse decision")

        optimizer._ensure_gefen_codebook = capture
        try:
            optimizer.step()
            raise AssertionError("sentinel resume step unexpectedly completed")
        except RuntimeError as exc:
            if "captured reuse decision" not in str(exc):
                raise
        finally:
            del optimizer._ensure_gefen_codebook

        decisions = [None] * _WORLD
        dist.all_gather_object(decisions, captured[0])

        resume_completed = False
        codebook_agreement = False
        period_preserved = False
        if len(set(decisions)) == 1:
            # The schedule is agreed; the real resume step must complete
            # collectively and keep the restored periods on the owner.
            step_before = optimizer._gefen_global_step
            optimizer.step()
            resume_completed = (
                optimizer._gefen_codebook is not None
                and optimizer._gefen_global_step == step_before + 1
            )
            codebooks = [
                torch.empty_like(optimizer._gefen_codebook) for _ in range(_WORLD)
            ]
            dist.all_gather(codebooks, optimizer._gefen_codebook)
            codebook_agreement = all(
                torch.equal(item, codebooks[0]) for item in codebooks[1:]
            )
            if rank == 0:
                period_preserved = (
                    int(optimizer.state[owner_source]["automatic_period"])
                    == first_period
                )
            else:
                period_preserved = not optimizer.state
        queue.put(
            {
                "rank": rank,
                "state_asymmetric": state_asymmetric,
                "decisions": decisions,
                "resume_completed": resume_completed,
                "codebook_agreement": codebook_agreement,
                "period_preserved": period_preserved,
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _revalidation_decision_worker(rank, init_file, queue):
    """A rank-local scope-validated reset must re-validate on every member.

    move_state_ is collective-free and resets _gefen_codebook_scope_validated
    on the member that called it only, while every fingerprint the step
    header exchanges stays identical. The next step must make the SAME
    early-return decision inside _ensure_codebook_scope_agreement on every
    member.
    """

    try:
        group, members = _init_worker(rank, init_file)
        runtime_group = dist.group.WORLD

        parameter = torch.nn.Parameter(torch.zeros(4))
        optimizer = Gefen([("weight", parameter)], fused=False, factored_v_2d=False)
        identity = ParameterIdentity("Weight", (4,))
        records = tuple(_replicated(identity, group, member) for member in members)
        optimizer.post_sharding(
            (ParameterRebinding(parameter, parameter, records[rank]),),
            manifest=ShardingManifest(records),
            codebook_process_group=CodebookProcessGroupBinding(
                group, members[rank], runtime_group, torch.device("cpu")
            ),
        )
        optimizer._resolve_automatic_period = lambda *args: 4
        parameter.grad = torch.tensor([-1.0, -0.5, 0.25, 1.0])
        optimizer.step()
        validated_after_first = bool(optimizer._gefen_codebook_scope_validated)

        # Advertised collective-free state movement on ONE member only; every
        # state value stays bit-identical, so the step header fingerprints
        # still agree across members.
        if rank == 0:
            optimizer.move_state_(torch.device("cpu"))
        reset_asymmetric = (
            not optimizer._gefen_codebook_scope_validated
            if rank == 0
            else bool(optimizer._gefen_codebook_scope_validated)
        )

        parameter.grad = torch.tensor([0.75, -0.25, 0.5, -1.0])
        captured = []

        def capture():
            captured.append(not optimizer._gefen_codebook_scope_validated)
            raise RuntimeError("captured revalidation decision")

        optimizer._ensure_codebook_scope_agreement = capture
        try:
            optimizer.step()
            raise AssertionError("sentinel step unexpectedly completed")
        except RuntimeError as exc:
            if "captured revalidation decision" not in str(exc):
                raise
        finally:
            del optimizer._ensure_codebook_scope_agreement

        decisions = [None] * _WORLD
        dist.all_gather_object(decisions, captured[0])

        second_completed = False
        revalidated = False
        parameters_agree = False
        if len(set(decisions)) == 1:
            optimizer.step()
            second_completed = True
            revalidated = bool(optimizer._gefen_codebook_scope_validated)
            gathered = [torch.empty_like(parameter.detach()) for _ in range(_WORLD)]
            dist.all_gather(gathered, parameter.detach())
            parameters_agree = all(
                torch.equal(item, gathered[0]) for item in gathered[1:]
            )
        queue.put(
            {
                "rank": rank,
                "validated_after_first": validated_after_first,
                "reset_asymmetric": reset_asymmetric,
                "decisions": decisions,
                "second_completed": second_completed,
                "revalidated": revalidated,
                "parameters_agree": parameters_agree,
            }
        )
    except Exception as exc:
        queue.put({"rank": rank, "error": repr(exc)})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_workers(target):
    context = mp.get_context("spawn")
    queue = context.Queue()
    fd, init_file = tempfile.mkstemp(prefix="gefen-scope-agreement-")
    os.close(fd)
    os.unlink(init_file)
    processes = [
        context.Process(target=target, args=(rank, init_file, queue))
        for rank in range(_WORLD)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=120) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                pytest.fail("scoped agreement worker hung")
            assert process.exitcode == 0
        return sorted(results, key=lambda item: item["rank"])
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if os.path.exists(init_file):
            os.unlink(init_file)


def test_resume_reuse_decision_is_scope_agreed_for_empty_members():
    results = _run_workers(_reuse_decision_worker)

    assert all("error" not in item for item in results), results
    for item in results:
        assert item["state_asymmetric"], item
        # Every member must gate the conditional "initialize" header on the
        # same group-wide decision; a resumed scope reuses restored periods.
        assert len(set(item["decisions"])) == 1, item
        assert all(item["decisions"]), item
        assert item["resume_completed"], item
        assert item["codebook_agreement"], item
        assert item["period_preserved"], item


def test_rank_local_scope_validated_reset_revalidates_on_every_member():
    results = _run_workers(_revalidation_decision_worker)

    assert all("error" not in item for item in results), results
    for item in results:
        assert item["validated_after_first"], item
        assert item["reset_asymmetric"], item
        # Every member must make the same re-validation decision after a
        # collective-free rank-local reset.
        assert len(set(item["decisions"])) == 1, item
        assert item["second_completed"], item
        assert item["revalidated"], item
        assert item["parameters_agree"], item


def _build_finalized_canonical():
    parameter = torch.nn.Parameter(torch.linspace(-0.5, 0.7, 8))
    optimizer = Gefen(
        [("layer.weight", parameter)],
        fused=False,
        factored_v_2d=False,
    )
    optimizer.rebind_parameter(
        parameter,
        parameter,
        identity=ParameterIdentity("Layer.Weight", (8,)),
    )
    optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
    return optimizer, parameter


def _exported_canonical_state():
    source, source_parameter = _build_finalized_canonical()
    source_parameter.grad = torch.linspace(-1.0, 0.8, 8)
    source.step()
    return source.export_canonical_state()


def test_parameter_storage_retarget_invalidates_prepared_canonical_import():
    exported = _exported_canonical_state()
    target, target_parameter = _build_finalized_canonical()

    prepared = target.prepare_canonical_state_import(exported)
    with torch.no_grad():
        target_parameter.data = target_parameter.data.clone()

    before_rejection = deep_state_snapshot(target)
    with pytest.raises(
        RuntimeError, match="changed after canonical import preparation"
    ):
        target.commit_canonical_state_import(prepared)
    # The rejected commit must be a no-op before any recovery import masks it.
    assert_deep_state_snapshot(target, before_rejection)

    # The refusal must keep canonical I/O available with a fresh preparation.
    target.import_canonical_state(exported)
    assert "automatic_period" in target.state[target_parameter]


def test_parameter_inplace_mutation_invalidates_prepared_canonical_import():
    exported = _exported_canonical_state()
    target, target_parameter = _build_finalized_canonical()

    prepared = target.prepare_canonical_state_import(exported)
    with torch.no_grad():
        target_parameter.mul_(2.0)

    before_rejection = deep_state_snapshot(target)
    with pytest.raises(
        RuntimeError, match="changed after canonical import preparation"
    ):
        target.commit_canonical_state_import(prepared)
    # The rejected commit must be a no-op before any recovery import masks it.
    assert_deep_state_snapshot(target, before_rejection)

    target.import_canonical_state(exported)
    assert "automatic_period" in target.state[target_parameter]


def test_undisturbed_prepared_canonical_import_still_commits():
    exported = _exported_canonical_state()
    target, target_parameter = _build_finalized_canonical()

    prepared = target.prepare_canonical_state_import(exported)
    target.commit_canonical_state_import(prepared)
    assert "automatic_period" in target.state[target_parameter]
