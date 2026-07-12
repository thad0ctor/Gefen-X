"""Fail-closed checkpoint validation for Parallel-Muon owner state."""

import copy
from datetime import timedelta
import io
import os
import queue
import socket
import traceback

import pytest
import torch
import torch.nn as nn

from gefen import GefenMuon

_DISTRIBUTED_METADATA = "muon_distributed_state"


def _cpu_optimizer(*, mode="distributed"):
    generator = torch.Generator().manual_seed(771)
    params = [
        (
            name,
            nn.Parameter(torch.randn(*shape, generator=generator) * 0.02),
        )
        for name, shape in (("left", (8, 8)), ("right", (6, 10)))
    ]
    optimizer = GefenMuon(
        params,
        lr=2e-3,
        fused=False,
        ns_steps=1,
        sharded_mode=mode,
    )
    return optimizer, params


def _step_cpu(optimizer, params, seed=772):
    generator = torch.Generator().manual_seed(seed)
    for _, param in params:
        param.grad = torch.randn(param.shape, generator=generator) * 0.002
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _marker_v2(optimizer, state_dict, world=2):
    saved_ids = [
        saved_id
        for group in state_dict["param_groups"]
        for saved_id in group["params"]
    ]
    live = [
        (name, param)
        for group in optimizer.param_groups
        for name, param in optimizer._iter_group_params_with_names(group)
    ]
    return {
        "version": 2,
        "ownership": "stable_full_param_index_v1",
        "consolidated": True,
        "groups": [
            {
                "world_size": world,
                "params": [
                    {
                        "saved_id": saved_id,
                        "name": str(name),
                        "shape": tuple(param.shape),
                        "owner": index % world,
                        "state_keys": tuple(
                            sorted(str(key) for key in state_dict["state"][saved_id])
                        ),
                        "initialized": any(
                            str(key) != "name"
                            for key in state_dict["state"][saved_id]
                        ),
                    }
                    for index, ((name, param), saved_id) in enumerate(
                        zip(live, saved_ids)
                    )
                ],
            }
        ],
    }


def _clone(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    return copy.deepcopy(value)


def _assert_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_equal(actual_item, expected_item)
        return
    assert actual == expected


def _simulate_live_dtensor(monkeypatch, optimizer):
    process_group = object()
    monkeypatch.setattr(optimizer, "_dist_available", lambda: True)
    monkeypatch.setattr(
        optimizer, "_distributed_process_group", lambda param: process_group
    )
    monkeypatch.setattr(optimizer, "_drop_non_owned_distributed_state", lambda: None)


def _stateful_checkpoint():
    source, source_params = _cpu_optimizer()
    _step_cpu(source, source_params)
    checkpoint = _clone(source.state_dict())
    checkpoint["gefen_muon_distributed"] = _marker_v2(source, checkpoint)
    return checkpoint


def _corrupt(checkpoint, case):
    corrupted = _clone(checkpoint)
    marker = corrupted["gefen_muon_distributed"]
    entry = marker["groups"][0]["params"][0]
    saved_id = corrupted["param_groups"][0]["params"][0]
    if case == "missing_marker":
        corrupted.pop("gefen_muon_distributed")
    elif case == "markerless_deleted_state":
        corrupted.pop("gefen_muon_distributed")
        for group in corrupted["param_groups"]:
            for index, param_id in enumerate(group["params"]):
                corrupted["state"][param_id] = {
                    "name": group["param_names"][index]
                }
    elif case == "incomplete_marker":
        corrupted["gefen_muon_distributed"] = {"consolidated": True}
    elif case == "wrong_version":
        marker["version"] = 99
    elif case == "wrong_ownership":
        marker["ownership"] = "rank_zero_v0"
    elif case == "not_consolidated":
        marker["consolidated"] = False
    elif case == "missing_groups":
        marker.pop("groups")
    elif case == "bad_world":
        marker["groups"][0]["world_size"] = 0
    elif case == "bad_owner":
        entry["owner"] = 1
    elif case == "bad_saved_id":
        entry["saved_id"] = 999
    elif case == "duplicate_saved_id":
        marker["groups"][0]["params"][1]["saved_id"] = entry["saved_id"]
    elif case == "bad_name":
        entry["name"] = "other"
    elif case == "bad_shape":
        entry["shape"] = (1, 64)
    elif case == "missing_state":
        corrupted["state"].pop(saved_id)
    elif case == "partial_core":
        corrupted["state"][saved_id].pop("m_magnitude")
        entry["state_keys"] = tuple(sorted(corrupted["state"][saved_id]))
    elif case == "fractional_step":
        corrupted["state"][saved_id]["step"] = 1.5
    elif case == "negative_magnitude":
        corrupted["state"][saved_id]["m_magnitude"].fill_(-1)
    elif case == "bad_codebook":
        bad = torch.zeros(12, dtype=torch.float32)
        corrupted["gefen_codebook"] = bad
        for group in corrupted["param_groups"]:
            group["_gefen_checkpoint_metadata"]["codebook"] = bad.clone()
    elif case == "nonstring_manifest_key":
        entry["state_keys"] = (*entry["state_keys"], 7)
    elif case == "bad_initialized":
        entry["initialized"] = False
    else:
        raise AssertionError(case)
    return corrupted


@pytest.mark.parametrize(
    "case",
    [
        "missing_marker",
        "markerless_deleted_state",
        "incomplete_marker",
        "wrong_version",
        "wrong_ownership",
        "not_consolidated",
        "missing_groups",
        "bad_world",
        "bad_owner",
        "bad_saved_id",
        "duplicate_saved_id",
        "bad_name",
        "bad_shape",
        "missing_state",
        "partial_core",
        "fractional_step",
        "negative_magnitude",
        "bad_codebook",
        "nonstring_manifest_key",
        "bad_initialized",
    ],
)
def test_populated_distributed_schema_rejects_before_mutation(monkeypatch, case):
    checkpoint = _corrupt(_stateful_checkpoint(), case)
    target, target_params = _cpu_optimizer()
    before_state = _clone(target.state)
    before_groups = _clone(target.param_groups)
    before_params = [param.detach().clone() for _, param in target_params]
    before_codebook = target._gefen_codebook
    before_step = target._gefen_global_step
    _simulate_live_dtensor(monkeypatch, target)

    with pytest.raises(ValueError, match="refused populated"):
        target.load_state_dict(checkpoint)

    _assert_equal(target.state, before_state)
    _assert_equal(target.param_groups, before_groups)
    for (_, param), expected in zip(target_params, before_params):
        assert torch.equal(param.detach(), expected)
    assert target._gefen_codebook is before_codebook
    assert target._gefen_global_step == before_step


def test_valid_v2_and_released_v1_schemas_load(monkeypatch):
    checkpoint = _stateful_checkpoint()
    target, _ = _cpu_optimizer()
    _simulate_live_dtensor(monkeypatch, target)
    target.load_state_dict(_clone(checkpoint))
    assert target._gefen_global_step == 1

    released_v1 = _clone(checkpoint)
    released_v1["gefen_muon_distributed"] = {
        "version": 1,
        "ownership": "stable_full_param_index_v1",
        "consolidated": True,
    }
    target_v1, _ = _cpu_optimizer()
    _simulate_live_dtensor(monkeypatch, target_v1)
    target_v1.load_state_dict(released_v1)
    assert target_v1._gefen_global_step == 1


def test_markerless_pristine_and_non_distributed_legacy_remain_safe(monkeypatch):
    fresh_source, _ = _cpu_optimizer()
    fresh = _clone(fresh_source.state_dict())
    fresh_target, _ = _cpu_optimizer()
    _simulate_live_dtensor(monkeypatch, fresh_target)
    fresh_target.load_state_dict(fresh)
    assert fresh_target._gefen_global_step == 0

    exact_source, exact_params = _cpu_optimizer(mode="exact")
    _step_cpu(exact_source, exact_params)
    exact_target, _ = _cpu_optimizer(mode="exact")
    exact_target.load_state_dict(_clone(exact_source.state_dict()))
    assert exact_target._gefen_global_step == 1


def test_released_v1_mixed_owner_state_fails_closed(monkeypatch):
    checkpoint = _stateful_checkpoint()
    checkpoint["gefen_muon_distributed"] = {
        "version": 1,
        "ownership": "stable_full_param_index_v1",
        "consolidated": True,
    }
    second_id = checkpoint["param_groups"][0]["params"][1]
    checkpoint["state"][second_id] = {"name": "right"}
    target, _ = _cpu_optimizer()
    _simulate_live_dtensor(monkeypatch, target)
    with pytest.raises(ValueError, match="mixes initialized and empty"):
        target.load_state_dict(checkpoint)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _distributed_grads(step, shapes, device):
    generator = torch.Generator(device="cpu").manual_seed(8800 + step)
    return [
        (torch.randn(*shape, generator=generator) * 0.002).to(
            device, torch.bfloat16
        )
        for shape in shapes
    ]


def _clone_local_state(state):
    return {
        key: (
            value.to_local().detach().clone()
            if torch.is_tensor(value) and hasattr(value, "to_local")
            else value.detach().clone()
            if torch.is_tensor(value)
            else copy.deepcopy(value)
        )
        for key, value in state.items()
    }


def _local_state_equal(actual, expected):
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if torch.is_tensor(actual_value) and hasattr(actual_value, "to_local"):
            actual_value = actual_value.to_local()
        if torch.is_tensor(expected_value):
            if not torch.is_tensor(actual_value) or not torch.equal(
                actual_value, expected_value
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _mixed_cpu_checkpoint_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=30),
        )
        mesh = init_device_mesh("cpu", (world,))

        def full(seed):
            return torch.randn(8, 8, generator=torch.Generator().manual_seed(seed))

        def sharded(value):
            return nn.Parameter(distribute_tensor(value.clone(), mesh, [Shard(0)]))

        def assign(param, value):
            param.grad = distribute_tensor(value.clone(), mesh, [Shard(0)])

        def live_snapshot(optimizer, params):
            return (
                optimizer._gefen_global_step,
                None
                if optimizer._gefen_codebook is None
                else optimizer._gefen_codebook.detach().clone(),
                [_clone_local_state(optimizer.state[param]) for param in params],
                [
                    param.detach().to_local().clone()
                    if hasattr(param, "to_local")
                    else param.detach().clone()
                    for param in params
                ],
            )

        def snapshot_equal(optimizer, params, snapshot):
            step, codebook, states, values = snapshot
            current_codebook = optimizer._gefen_codebook
            return (
                optimizer._gefen_global_step == step
                and (
                    (current_codebook is None and codebook is None)
                    or (
                        torch.is_tensor(current_codebook)
                        and torch.is_tensor(codebook)
                        and torch.equal(current_codebook, codebook)
                    )
                )
                and all(
                    _local_state_equal(optimizer.state[param], expected)
                    for param, expected in zip(params, states)
                )
                and all(
                    torch.equal(
                        param.detach().to_local()
                        if hasattr(param, "to_local")
                        else param.detach(),
                        expected,
                    )
                    for param, expected in zip(params, values)
                )
            )

        # One distributed optimizer group may mix a Parallel-Muon DTensor with
        # a replicated plain-tensor fallback. Only the former belongs in the
        # saved-world owner proof.
        initial_parallel, initial_fallback = full(920), full(921)

        def make_fallback_pair(parallel_value, fallback_value):
            parallel = sharded(parallel_value)
            fallback = nn.Parameter(fallback_value.clone())
            optimizer = GefenMuon(
                [
                    {
                        "params": [
                            ("parallel", parallel),
                            ("fallback", fallback),
                        ],
                        "sharded_mode": "distributed",
                    }
                ],
                lr=2e-3,
                fused=False,
                ns_steps=1,
            )
            return optimizer, [parallel, fallback]

        source, source_params = make_fallback_pair(
            initial_parallel, initial_fallback
        )
        assign(source_params[0], full(922) * 0.01)
        source_params[1].grad = full(923) * 0.01
        source.step()
        checkpoint = _clone(source.state_dict())
        saved_ids = checkpoint["param_groups"][0]["params"]
        marker = checkpoint["gefen_muon_distributed"]
        manifest_ids = [
            item["saved_id"]
            for group in marker["groups"]
            for item in group["params"]
        ]
        fallback_schema_ok = manifest_ids == [saved_ids[0]]

        current_parallel = source_params[0].detach().full_tensor().clone()
        current_fallback = source_params[1].detach().clone()
        resumed, resumed_params = make_fallback_pair(
            current_parallel, current_fallback
        )
        resumed.load_state_dict(_clone(checkpoint))
        assign(source_params[0], full(924) * 0.01)
        source_params[1].grad = full(925) * 0.01
        assign(resumed_params[0], full(924) * 0.01)
        resumed_params[1].grad = full(925) * 0.01
        source.step()
        resumed.step()
        fallback_continuation = torch.equal(
            source_params[0].detach().to_local(),
            resumed_params[0].detach().to_local(),
        ) and torch.equal(source_params[1], resumed_params[1])

        # Adding the replicated fallback to every copy of the owner proof must
        # fail before touching even a pre-existing target transaction.
        corrupted = _clone(checkpoint)
        marker_copies = [corrupted["gefen_muon_distributed"]]
        marker_copies.extend(
            group["_gefen_checkpoint_metadata"][
                _DISTRIBUTED_METADATA
            ]
            for group in corrupted["param_groups"]
        )
        for marker_copy in marker_copies:
            extra = _clone(marker_copy["groups"][0]["params"][0])
            extra.update(
                {
                    "saved_id": saved_ids[1],
                    "name": "fallback",
                    "shape": tuple(initial_fallback.shape),
                    "owner": 1,
                }
            )
            marker_copy["groups"][0]["params"].append(extra)
        rejected, rejected_params = make_fallback_pair(
            initial_parallel, initial_fallback
        )
        before = live_snapshot(rejected, rejected_params)
        rejection_message = None
        try:
            rejected.load_state_dict(corrupted)
        except ValueError as exc:
            rejection_message = str(exc)
        fallback_atomic = (
            rejection_message is not None
            and "ordered eligible" in rejection_message
            and snapshot_equal(rejected, rejected_params, before)
        )

        # Rank-local approx state and stable-owner distributed state compose in
        # one optimizer: consolidate owners first, then wrap all real state into
        # rank-indexed payloads, and reverse that order on load validation.
        initial_approx, initial_distributed = full(930), full(931)

        def make_composed_pair(approx_value, distributed_value):
            approx = sharded(approx_value)
            distributed = sharded(distributed_value)
            optimizer = GefenMuon(
                [
                    {
                        "params": [("approx", approx)],
                        "sharded_mode": "approx",
                    },
                    {
                        "params": [("distributed", distributed)],
                        "sharded_mode": "distributed",
                    },
                ],
                lr=2e-3,
                fused=False,
                ns_steps=1,
            )
            return optimizer, [approx, distributed]

        composed, composed_params = make_composed_pair(
            initial_approx, initial_distributed
        )
        assign(composed_params[0], full(932) * 0.01)
        assign(composed_params[1], full(933) * 0.01)
        composed.step()
        composed_checkpoint = _clone(composed.state_dict())
        current_composed = [
            param.detach().full_tensor().clone() for param in composed_params
        ]
        composed_resumed, composed_resumed_params = make_composed_pair(
            *current_composed
        )
        composed_resumed.load_state_dict(_clone(composed_checkpoint))
        metadata_only_checkpoint = _clone(composed_checkpoint)
        metadata_only_checkpoint.pop("gefen_muon_distributed")
        metadata_resumed, metadata_resumed_params = make_composed_pair(
            *current_composed
        )
        metadata_resumed.load_state_dict(metadata_only_checkpoint)
        metadata_transport_ok = (
            metadata_resumed._gefen_global_step
            == composed_resumed._gefen_global_step
            and all(
                _local_state_equal(
                    metadata_resumed.state[left],
                    composed_resumed.state[right],
                )
                for left, right in zip(
                    metadata_resumed_params, composed_resumed_params
                )
            )
        )
        assign(composed_params[0], full(934) * 0.01)
        assign(composed_params[1], full(935) * 0.01)
        assign(composed_resumed_params[0], full(934) * 0.01)
        assign(composed_resumed_params[1], full(935) * 0.01)
        composed.step()
        composed_resumed.step()
        composed_continuation = all(
            torch.equal(left.detach().to_local(), right.detach().to_local())
            for left, right in zip(composed_params, composed_resumed_params)
        )

        result_queue.put(
            (
                "result",
                rank,
                fallback_schema_ok,
                fallback_continuation,
                fallback_atomic,
                composed_continuation,
                metadata_transport_ok,
                rejection_message,
            )
        )
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_mixed_parallel_fallback_and_rank_local_composition_cpu():
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_mixed_cpu_checkpoint_worker,
            args=(rank, world, port, result_queue),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()

    messages = []
    try:
        for _ in range(world):
            messages.append(result_queue.get(timeout=60))
    except queue.Empty:
        pass
    finally:
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    errors = [item[2] for item in messages if item[0] == "error"]
    assert not errors, "\n".join(errors)
    results = {item[1]: item[2:] for item in messages if item[0] == "result"}
    assert set(results) == {0, 1}, messages
    for rank, result in results.items():
        (
            schema,
            fallback_resume,
            atomic,
            composed_resume,
            metadata_transport,
            message,
        ) = result
        assert schema, rank
        assert fallback_resume, rank
        assert atomic, (rank, message)
        assert composed_resume, rank
        assert metadata_transport, rank


def _checkpoint_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=15),
        )
        mesh = init_device_mesh("cuda", (world,))
        specs = (("left", (16, 16)), ("right", (12, 20)))
        generator = torch.Generator(device="cpu").manual_seed(8799)
        initial = [
            (torch.randn(*shape, generator=generator) * 0.02).to(torch.bfloat16)
            for _, shape in specs
        ]

        def make_params(full_values):
            return [
                (
                    name,
                    nn.Parameter(
                        distribute_tensor(
                            value.to(device).clone(), mesh, [Shard(0)]
                        )
                    ),
                )
                for (name, _), value in zip(specs, full_values)
            ]

        def make_optimizer(params):
            return GefenMuon(
                params,
                lr=2e-3,
                fused=False,
                ns_steps=1,
                sharded_mode="distributed",
            )

        source_params = make_params(initial)
        source = make_optimizer(source_params)
        shapes = [shape for _, shape in specs]
        for step in range(2):
            for (_, param), grad in zip(
                source_params, _distributed_grads(step, shapes, device)
            ):
                param.grad = distribute_tensor(grad, mesh, [Shard(0)])
            source.step()
            source.zero_grad(set_to_none=True)

        checkpoint = _clone(source.state_dict())
        checkpoint_params = [
            param.detach().full_tensor().clone() for _, param in source_params
        ]

        invalid_params = make_params(initial)
        invalid_target = make_optimizer(invalid_params)
        for (_, param), grad in zip(
            invalid_params, _distributed_grads(9, shapes, device)
        ):
            param.grad = distribute_tensor(grad, mesh, [Shard(0)])
        invalid_target.step()
        invalid_target.zero_grad(set_to_none=True)
        params_before = [
            param.detach().to_local().clone() for _, param in invalid_params
        ]
        states_before = [
            _clone_local_state(invalid_target.state[param])
            for _, param in invalid_params
        ]
        codebook_before = invalid_target._gefen_codebook.detach().clone()
        step_before = invalid_target._gefen_global_step
        invalid_checkpoint = _clone(checkpoint)
        invalid_checkpoint.pop("gefen_muon_distributed")
        for group in invalid_checkpoint["param_groups"]:
            group["_gefen_checkpoint_metadata"].pop(
                _DISTRIBUTED_METADATA, None
            )
        error = None
        try:
            invalid_target.load_state_dict(invalid_checkpoint)
        except ValueError as exc:
            error = str(exc)
        rejection_ok = (
            error is not None
            and "refused populated" in error
            and invalid_target._gefen_global_step == step_before
            and torch.equal(invalid_target._gefen_codebook, codebook_before)
            and all(
                torch.equal(param.detach().to_local(), expected)
                for (_, param), expected in zip(invalid_params, params_before)
            )
            and all(
                _local_state_equal(invalid_target.state[param], expected)
                for (_, param), expected in zip(invalid_params, states_before)
            )
        )

        for (_, param), grad in zip(
            source_params, _distributed_grads(2, shapes, device)
        ):
            param.grad = distribute_tensor(grad, mesh, [Shard(0)])
        source.step()

        resumed_params = make_params(checkpoint_params)
        resumed = make_optimizer(resumed_params)
        resumed.load_state_dict(_clone(checkpoint))
        for (_, param), grad in zip(
            resumed_params, _distributed_grads(2, shapes, device)
        ):
            param.grad = distribute_tensor(grad, mesh, [Shard(0)])
        resumed.step()
        continuation_ok = all(
            torch.equal(
                resumed_param.detach().to_local(), source_param.detach().to_local()
            )
            for (_, resumed_param), (_, source_param) in zip(
                resumed_params, source_params
            )
        )
        marker = checkpoint.get("gefen_muon_distributed", {})
        result_queue.put(
            (
                "result",
                rank,
                rejection_ok,
                continuation_ok,
                marker.get("version"),
                error,
            )
        )
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available(),
    reason="Parallel-Muon checkpoint safety requires two CUDA GPUs and NCCL",
)
def test_parallel_muon_rejection_is_atomic_and_valid_v2_continues():
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_checkpoint_worker,
            args=(rank, world, port, result_queue),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()

    messages = []
    try:
        for _ in range(world):
            messages.append(result_queue.get(timeout=45))
    except queue.Empty:
        pass
    finally:
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    errors = [item[2] for item in messages if item[0] == "error"]
    assert not errors, "\n".join(errors)
    results = {
        rank: (rejection, continuation, version, message)
        for kind, rank, rejection, continuation, version, message in messages
        if kind == "result"
    }
    assert set(results) == {0, 1}, messages
    for rank, (rejection, continuation, version, message) in results.items():
        assert rejection, (rank, message)
        assert continuation, rank
        assert version == 2, rank


def _single_process_resume_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=60),
        )
        mesh = init_device_mesh("cpu", (world,))
        specs = (("left", (8, 8)), ("right", (6, 10)))
        generator = torch.Generator().manual_seed(9314)
        params = [
            (
                name,
                nn.Parameter(
                    distribute_tensor(
                        (torch.randn(*shape, generator=generator) * 0.02),
                        mesh,
                        [Shard(0)],
                    )
                ),
            )
            for name, shape in specs
        ]
        optimizer = GefenMuon(
            params,
            lr=2e-3,
            fused=False,
            ns_steps=1,
            sharded_mode="distributed",
        )
        grad_generator = torch.Generator().manual_seed(9315)
        for _, param in params:
            param.grad = distribute_tensor(
                torch.randn(param.shape, generator=grad_generator) * 0.002,
                mesh,
                [Shard(0)],
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        checkpoint = _clone(optimizer.state_dict())
        full_values = [
            param.detach().full_tensor().clone() for _, param in params
        ]
        if rank == 0:
            # Serialize instead of queueing live tensors: tensor transport
            # over mp.Queue shares memory via file descriptors that die with
            # this worker process, racing the parent's read
            # (ConnectionResetError on slow CI hosts). Bytes pickle normally.
            buffer = io.BytesIO()
            torch.save(
                {"checkpoint": checkpoint, "full_values": full_values}, buffer
            )
            result_queue.put(("checkpoint", rank, buffer.getvalue()))
        else:
            result_queue.put(("done", rank, None))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="consolidated single-process resume regression needs Gloo",
)
def test_consolidated_v2_checkpoint_loads_into_single_process_optimizer():
    """A consolidated distributed save must resume without torch.distributed.

    Consolidation exists so every rank's ``state_dict()`` carries the complete
    owner state; the natural consumer is a single-process resume or eval run
    that never initializes a process group. Released-v1 and markerless loads
    already support that, so the version-2 manifest must not be stricter.
    """
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_single_process_resume_worker,
            args=(rank, world, port, result_queue),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()

    messages = []
    try:
        for _ in range(world):
            messages.append(result_queue.get(timeout=90))
    except queue.Empty:
        pass
    finally:
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes), [
        process.exitcode for process in processes
    ]
    errors = [payload for kind, _, payload in messages if kind == "error"]
    assert not errors, "\n".join(errors)
    payloads = [payload for kind, _, payload in messages if kind == "checkpoint"]
    assert len(payloads) == 1, messages
    saved = torch.load(io.BytesIO(payloads[0]), weights_only=False)
    checkpoint, full_values = saved["checkpoint"], saved["full_values"]

    marker = checkpoint.get("gefen_muon_distributed")
    assert isinstance(marker, dict) and marker.get("version") == 2, marker
    assert not torch.distributed.is_initialized()

    specs = (("left", (8, 8)), ("right", (6, 10)))
    params = [
        (name, nn.Parameter(value.clone()))
        for (name, _), value in zip(specs, full_values)
    ]
    optimizer = GefenMuon(
        params,
        lr=2e-3,
        fused=False,
        ns_steps=1,
        sharded_mode="distributed",
    )
    optimizer.load_state_dict(_clone(checkpoint))

    assert optimizer._gefen_global_step == 1
    saved_ids = [
        saved_id
        for group in checkpoint["param_groups"]
        for saved_id in group["params"]
    ]
    for (name, param), saved_id in zip(params, saved_ids):
        live_state = optimizer.state[param]
        saved_state = checkpoint["state"][saved_id]
        for key in ("automatic_period", "step", "m_codebook", "m_magnitude"):
            assert key in live_state, (name, key)
            expected = saved_state[key]
            actual = live_state[key]
            if torch.is_tensor(expected):
                assert torch.is_tensor(actual) and torch.equal(
                    actual, expected
                ), (name, key)
            else:
                assert actual == expected, (name, key)

    grad_generator = torch.Generator().manual_seed(9316)
    before = [param.detach().clone() for _, param in params]
    for _, param in params:
        param.grad = torch.randn(param.shape, generator=grad_generator) * 0.002
    optimizer.step()
    assert all(
        not torch.equal(param.detach(), previous)
        for (_, param), previous in zip(params, before)
    )
