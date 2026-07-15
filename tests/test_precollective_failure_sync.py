"""CPU/Gloo coverage for failures before sharded Muon step collectives."""

from __future__ import annotations

import copy
from datetime import timedelta
import os
from pathlib import Path
import queue as queue_module
import tempfile
import time
import traceback

import pytest
import torch
import torch.distributed as dist
from torch import nn

from gefen import GefenMuon, GefenMuonHybrid


_WORLD_SIZE = 2
_PROCESS_GROUP_TIMEOUT_SECONDS = 20
_STEP_DEADLINE_SECONDS = 10.0
_WORKER_DEADLINE_SECONDS = 60.0
_CASES = ("muon_exact", "muon_distributed", "hybrid_exact")


def _local_clone(value: torch.Tensor) -> torch.Tensor:
    value = value.to_local() if hasattr(value, "to_local") else value
    value = value.wait() if hasattr(value, "wait") else value
    return value.detach().clone()


def _clone_tree(value):
    if torch.is_tensor(value):
        return _local_clone(value)
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _trees_equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_trees_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_trees_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _attribute_snapshot(owner, name: str):
    if not hasattr(owner, name):
        return (False, None, None)
    value = getattr(owner, name)
    return (True, id(value), _clone_tree(value))


def _optimizer_snapshot(optimizer, parameters):
    children = tuple(getattr(optimizer, "_subopts", (optimizer,)))
    child_states = []
    codebooks = []
    global_steps = []
    for child in children:
        child_states.append(
            (
                id(child.state),
                tuple(
                    (id(parameter), id(state), _clone_tree(state))
                    for parameter, state in child.state.items()
                ),
            )
        )
        codebooks.append(
            tuple(
                _attribute_snapshot(child, name)
                for name in (
                    "_gefen_codebook",
                    "_gefen_codebook_by_device",
                    "_gefen_codebook_lut_by_device",
                )
            )
        )
        global_steps.append(
            tuple(
                _attribute_snapshot(child, name)
                for name in (
                    "_gefen_global_step",
                    "_gefen_global_step_by_device",
                )
            )
        )
    return {
        "parameters": tuple(_local_clone(parameter) for parameter in parameters),
        "gradients": tuple(
            None if parameter.grad is None else _local_clone(parameter.grad)
            for parameter in parameters
        ),
        "state": tuple(child_states),
        "codebook": tuple(codebooks),
        "global_step": tuple(global_steps),
    }


def _snapshot_comparison(before, after):
    return {
        name: _trees_equal(before[name], after[name])
        for name in ("parameters", "gradients", "state", "codebook", "global_step")
    }


def _make_optimizer(mesh, case: str):
    from torch.distributed.tensor import Shard, distribute_tensor

    full_weight = torch.linspace(-0.8, 0.9, 64, dtype=torch.float32).reshape(8, 8)
    full_weight_grad = torch.linspace(0.6, -0.7, 64, dtype=torch.float32).reshape(8, 8)
    weight = nn.Parameter(distribute_tensor(full_weight.clone(), mesh, [Shard(0)]))

    if case == "hybrid_backup_sharded":
        # Invert the ownership of hybrid_exact: the Muon half holds a REPLICATED
        # (non-sharded) 2D weight while the BACKUP half owns the sharded mesh.
        # Deriving the failure-sync scope from the Muon child alone would then
        # miss that backup-only mesh, so a one-rank preflight failure would raise
        # on the failing rank while its mesh peers stepped and mutated their
        # backup shard (cross-rank divergence). The hybrid's union scope must
        # fold the backup mesh in so every rank fails fast instead.
        muon_weight = nn.Parameter(
            torch.linspace(-0.5, 0.6, 64, dtype=torch.float32).reshape(8, 8)
        )
        optimizer = GefenMuonHybrid(
            [("muon.weight", muon_weight)],
            [("backup.weight", weight)],
            lr=1e-3,
            fused=False,
            ns_steps=1,
            ns_schedule="standard",
            sharded_mode="exact",
            backup_optimizer="gefen",
            normuon=False,
        )
        parameters = (muon_weight, weight)
        muon_weight.grad = torch.linspace(
            0.6, -0.7, 64, dtype=torch.float32
        ).reshape(8, 8)
    elif case == "hybrid_exact":
        bias = nn.Parameter(torch.linspace(-0.4, 0.3, 8, dtype=torch.float32))
        optimizer = GefenMuonHybrid(
            [("weight", weight)],
            [("bias", bias)],
            lr=1e-3,
            fused=False,
            ns_steps=1,
            ns_schedule="standard",
            sharded_mode="exact",
            backup_optimizer="gefen",
            normuon=False,
        )
        parameters = (weight, bias)
        bias.grad = torch.linspace(0.2, -0.3, 8, dtype=torch.float32)
    else:
        mode = case.removeprefix("muon_")
        optimizer = GefenMuon(
            [("weight", weight)],
            lr=1e-3,
            fused=False,
            ns_steps=1,
            ns_schedule="standard",
            sharded_mode=mode,
        )
        parameters = (weight,)

    weight.grad = distribute_tensor(full_weight_grad.clone(), mesh, [Shard(0)])
    return optimizer, parameters


def _closure_failure_result(rank: int, mesh, case: str):
    optimizer, parameters = _make_optimizer(mesh, case)
    before = _optimizer_snapshot(optimizer, parameters)

    def closure():
        if rank == 0:
            raise RuntimeError("rank-zero closure failure")
        return torch.tensor(1.0)

    started = time.monotonic()
    try:
        optimizer.step(closure)
        message = None
        error_traceback = None
    except BaseException as exc:
        message = "{}: {}".format(type(exc).__name__, exc)
        error_traceback = traceback.format_exc()
    elapsed = time.monotonic() - started
    after = _optimizer_snapshot(optimizer, parameters)
    return {
        "message": message,
        "traceback": error_traceback,
        "elapsed": elapsed,
        "unchanged": _snapshot_comparison(before, after),
    }


def _amp_control_result(rank: int, mesh, case: str, scenario: str):
    optimizer, parameters = _make_optimizer(mesh, case)
    if scenario == "divergent_found_inf":
        optimizer.found_inf = torch.tensor(float(rank == 0))
        optimizer.grad_scale = torch.tensor(8.0)
    elif scenario == "protocol_presence":
        if rank == 0:
            optimizer.found_inf = torch.tensor(0.0)
            optimizer.grad_scale = torch.tensor(8.0)
    elif scenario == "scale_disagreement":
        optimizer.found_inf = torch.tensor(0.0)
        optimizer.grad_scale = torch.tensor(8.0 if rank == 0 else 4.0)
    elif scenario == "group_wide_overflow":
        optimizer.found_inf = torch.tensor(1.0)
        optimizer.grad_scale = torch.tensor(8.0)
    else:
        raise AssertionError("unknown AMP-control scenario: {}".format(scenario))

    post_hook_calls = []
    hook_handle = None
    if case == "hybrid_exact" and scenario == "group_wide_overflow":
        hook_handle = optimizer.register_step_post_hook(
            lambda _optimizer, _args, _kwargs: post_hook_calls.append(True)
        )
    before = _optimizer_snapshot(optimizer, parameters)

    started = time.monotonic()
    try:
        optimizer.step()
        message = None
        error_traceback = None
    except BaseException as exc:
        message = "{}: {}".format(type(exc).__name__, exc)
        error_traceback = traceback.format_exc()
    elapsed = time.monotonic() - started
    after = _optimizer_snapshot(optimizer, parameters)
    if hook_handle is not None:
        hook_handle.remove()
    return {
        "message": message,
        "traceback": error_traceback,
        "elapsed": elapsed,
        "controls_present": hasattr(optimizer, "found_inf") or hasattr(optimizer, "grad_scale"),
        "local_found_inf": (
            float(optimizer.found_inf.item()) if hasattr(optimizer, "found_inf") else None
        ),
        "local_grad_scale": (
            float(optimizer.grad_scale.item()) if hasattr(optimizer, "grad_scale") else None
        ),
        "post_hook_calls": (
            len(post_hook_calls)
            if case == "hybrid_exact" and scenario == "group_wide_overflow"
            else None
        ),
        "unchanged": _snapshot_comparison(before, after),
    }


def _distributed_worker(rank: int, init_method: str, case: str, result_queue) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        from torch.distributed.tensor import init_device_mesh

        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        closure_failure = _closure_failure_result(rank, mesh, case)
        dist.barrier()
        divergent_found_inf = _amp_control_result(rank, mesh, case, "divergent_found_inf")
        dist.barrier()
        protocol_presence = _amp_control_result(rank, mesh, case, "protocol_presence")
        dist.barrier()
        scale_disagreement = _amp_control_result(rank, mesh, case, "scale_disagreement")
        dist.barrier()
        group_wide_overflow = _amp_control_result(rank, mesh, case, "group_wide_overflow")
        dist.barrier()
        result_queue.put(
            {
                "rank": rank,
                "closure_failure": closure_failure,
                "divergent_found_inf": divergent_found_inf,
                "protocol_presence": protocol_presence,
                "scale_disagreement": scale_disagreement,
                "group_wide_overflow": group_wide_overflow,
            }
        )
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _backup_sharded_worker(rank: int, init_method: str, case: str, result_queue) -> None:
    """Worker for the backup-only-mesh case: only the closure-failure scenario.

    Constructs the hybrid where the BACKUP half owns the sharded mesh and the
    Muon half is non-sharded, induces a rank-0 closure failure, and reports
    whether the failure fanned out to every mesh rank (fail-fast) with no
    parameter/state mutation and no hang.
    """
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=_WORLD_SIZE,
            timeout=timedelta(seconds=_PROCESS_GROUP_TIMEOUT_SECONDS),
        )
        from torch.distributed.tensor import init_device_mesh

        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        closure_failure = _closure_failure_result(rank, mesh, case)
        dist.barrier()
        result_queue.put({"rank": rank, "closure_failure": closure_failure})
    except BaseException:
        result_queue.put({"rank": rank, "fatal_error": traceback.format_exc()})
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _run_distributed_case(case: str, worker_target=_distributed_worker):
    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    descriptor, rendezvous_path = tempfile.mkstemp(prefix="gefen-precollective-sync-")
    os.close(descriptor)
    os.unlink(rendezvous_path)
    init_method = Path(rendezvous_path).resolve().as_uri()
    processes = [
        context.Process(
            target=worker_target,
            args=(rank, init_method, case, result_queue),
        )
        for rank in range(_WORLD_SIZE)
    ]
    results = []
    hung_ranks = []
    exit_codes = []
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + _WORKER_DEADLINE_SECONDS
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        hung_ranks = [rank for rank, process in enumerate(processes) if process.is_alive()]
        exit_codes = [process.exitcode for process in processes]
        while len(results) < _WORLD_SIZE:
            try:
                results.append(result_queue.get(timeout=0.5))
            except queue_module.Empty:
                break
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
        try:
            os.unlink(rendezvous_path)
        except FileNotFoundError:
            pass

    assert not hung_ranks, (case, "workers hung", hung_ranks, exit_codes, results)
    assert all(code == 0 for code in exit_codes), (case, "nonzero worker exit", exit_codes, results)
    assert len(results) == _WORLD_SIZE, (case, "missing worker result", exit_codes, results)
    return sorted(results, key=lambda result: result["rank"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="pre-collective failure synchronization coverage requires Gloo",
)
@pytest.mark.parametrize("case", _CASES)
def test_rank_local_closure_and_amp_controls_are_synchronized(case):
    results = _run_distributed_case(case)
    assert all("fatal_error" not in result for result in results), results

    closure_results = [result["closure_failure"] for result in results]
    assert all(item["message"] is not None for item in closure_results), closure_results
    phase = "hybrid step preflight" if case == "hybrid_exact" else "step preflight"
    assert "GefenMuon {} failed on local rank 0".format(phase) in closure_results[0]["message"]
    assert "rank-zero closure failure" in closure_results[0]["message"]
    assert "GefenMuon {} failed on another process-group member".format(phase) in closure_results[1]["message"]
    assert all(item["elapsed"] < _STEP_DEADLINE_SECONDS for item in closure_results), closure_results
    assert all(all(item["unchanged"].values()) for item in closure_results), closure_results

    overflow_results = [result["divergent_found_inf"] for result in results]
    assert [item["local_found_inf"] for item in overflow_results] == [1.0, 0.0]
    assert all(
        item["message"] is not None
        and "AMP found_inf differs across process-group members" in item["message"]
        and "group-aware gradient scaler" in item["message"]
        for item in overflow_results
    ), overflow_results
    assert all(item["elapsed"] < _STEP_DEADLINE_SECONDS for item in overflow_results), overflow_results
    assert all(all(item["unchanged"].values()) for item in overflow_results), overflow_results

    presence_results = [result["protocol_presence"] for result in results]
    assert [item["controls_present"] for item in presence_results] == [True, False]
    assert all(
        item["message"] is not None
        and "AMP protocol presence differs across process-group members" in item["message"]
        and "group-aware gradient scaler" in item["message"]
        for item in presence_results
    ), presence_results
    assert all(item["elapsed"] < _STEP_DEADLINE_SECONDS for item in presence_results), presence_results
    assert all(all(item["unchanged"].values()) for item in presence_results), presence_results

    scale_results = [result["scale_disagreement"] for result in results]
    assert [item["local_found_inf"] for item in scale_results] == [0.0, 0.0]
    assert [item["local_grad_scale"] for item in scale_results] == [8.0, 4.0]
    assert all(
        item["message"] is not None
        and "AMP grad_scale differs across process-group members" in item["message"]
        and "group-aware gradient scaler" in item["message"]
        for item in scale_results
    ), scale_results
    assert all(item["elapsed"] < _STEP_DEADLINE_SECONDS for item in scale_results), scale_results
    assert all(all(item["unchanged"].values()) for item in scale_results), scale_results

    skip_results = [result["group_wide_overflow"] for result in results]
    assert all(item["message"] is None for item in skip_results), skip_results
    assert all(item["elapsed"] < _STEP_DEADLINE_SECONDS for item in skip_results), skip_results
    assert all(all(item["unchanged"].values()) for item in skip_results), skip_results
    if case == "hybrid_exact":
        assert [item["post_hook_calls"] for item in skip_results] == [1, 1]
    else:
        assert [item["post_hook_calls"] for item in skip_results] == [None, None]


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="pre-collective failure synchronization coverage requires Gloo",
)
def test_backup_only_mesh_preflight_failure_is_synchronized():
    """A backup-owned sharded mesh the Muon half does not touch is in scope.

    Regression for the composite failure-sync deriving its process-group scope
    from the Muon child only: when the BACKUP half owns the sharded mesh (and
    the Muon half is non-sharded), a rank-0 preflight failure must fan out to
    every mesh rank so ALL ranks fail fast with no mutation -- rather than
    rank 0 raising while its mesh peer silently steps and mutates its backup
    shard. Also guards against a hang (bounded worker/step deadlines).
    """
    results = _run_distributed_case(
        "hybrid_backup_sharded", worker_target=_backup_sharded_worker
    )
    assert all("fatal_error" not in result for result in results), results

    closure_results = [result["closure_failure"] for result in results]
    # Both ranks fail fast (the failure was synchronized across the backup mesh).
    assert all(item["message"] is not None for item in closure_results), closure_results
    assert (
        "GefenMuon hybrid step preflight failed on local rank 0"
        in closure_results[0]["message"]
    ), closure_results
    assert "rank-zero closure failure" in closure_results[0]["message"], closure_results
    assert (
        "GefenMuon hybrid step preflight failed on another process-group member"
        in closure_results[1]["message"]
    ), closure_results
    # Liveness: neither rank hung, and no parameter/state/gradient moved on
    # either rank (the failing rank AND its mesh peer both roll back).
    assert all(
        item["elapsed"] < _STEP_DEADLINE_SECONDS for item in closure_results
    ), closure_results
    assert all(
        all(item["unchanged"].values()) for item in closure_results
    ), closure_results
