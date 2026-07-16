"""Same-topology FSDP2/DTensor optimizer checkpoint continuation."""

from __future__ import annotations

import copy
import os
import queue
import socket
import traceback

import pytest
import torch


_CARRIER_PREFIX = "_gefen_rank_local_payload_"
_MEMBER = "_gefen_rank_local_member"
_FORMAT = "rank_local_dtensor_v2"


def _carrier_key(rank: int) -> str:
    return f"{_CARRIER_PREFIX}{rank}"


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _local_value(value):
    if hasattr(value, "to_local"):
        value = value.to_local()
    if hasattr(value, "wait"):
        value = value.wait()
    return value


def _clone_value(value):
    if torch.is_tensor(value):
        return _local_value(value).detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def _values_equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _optimizer_snapshot(optimizer):
    params = [param for group in optimizer.param_groups for param in group["params"]]
    return {
        "state": [_clone_value(optimizer.state[param]) for param in params],
        "groups": [
            _clone_value(
                {
                    key: value
                    for key, value in group.items()
                    if key not in ("params", "_gefen_checkpoint_metadata")
                }
            )
            for group in optimizer.param_groups
        ],
        "params": [_clone_value(param) for param in params],
        "global_step": optimizer._gefen_global_step,
        "codebook": _clone_value(optimizer._gefen_codebook),
        "codebook_cache": _clone_value(optimizer._gefen_codebook_by_device),
        "seed_cache": _clone_value(optimizer._sr_seed_by_device),
    }


def _persistent_optimizer_snapshot(optimizer):
    scratch = {
        "stepsize",
        "_h_buf",
        "_capt_scalars",
        "_capt_consts",
        "_capt_consts_key",
        "_capt_stack",
        "_capt_row",
        _MEMBER,
    }
    params = [param for group in optimizer.param_groups for param in group["params"]]
    return {
        "state": [
            _clone_value(
                {
                    key: value
                    for key, value in optimizer.state[param].items()
                    if key not in scratch
                    and not (
                        isinstance(key, str) and key.startswith(_CARRIER_PREFIX)
                    )
                }
            )
            for param in params
        ],
        "groups": [
            _clone_value(
                {
                    key: value
                    for key, value in group.items()
                    if key not in ("params", "_gefen_checkpoint_metadata")
                }
            )
            for group in optimizer.param_groups
        ],
        "params": [_clone_value(param) for param in params],
        "global_step": optimizer._gefen_global_step,
        "codebook": _clone_value(optimizer._gefen_codebook),
    }


def _worker(
    rank: int, world: int, port: str, optimizer_kind: str, result_queue
) -> None:
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict, set_optimizer_state_dict
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    from gefen import Gefen, GefenMuon

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        # Init inside the try: a rendezvous/init failure -- a real possibility on
        # a loaded CI runner -- must surface its traceback, not merely exit
        # nonzero with no explanation.
        dist.init_process_group("gloo", rank=rank, world_size=world)
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))

        def make_model(full_values):
            model = nn.Module()
            model.register_parameter(
                "first",
                nn.Parameter(distribute_tensor(full_values[0].clone(), mesh, [Shard(0)])),
            )
            model.register_parameter(
                "second",
                nn.Parameter(distribute_tensor(full_values[1].clone(), mesh, [Shard(0)])),
            )
            return model

        def make_optimizer(model):
            if optimizer_kind == "muon_normuon":
                return GefenMuon(
                    [
                        {"params": [("first", model.first)]},
                        {"params": [("second", model.second)]},
                    ],
                    lr=1e-3,
                    fused=False,
                    sharded_mode="approx",
                    normuon=True,
                )
            return Gefen(
                [
                    {"params": [("first", model.first)]},
                    {"params": [("second", model.second)]},
                ],
                lr=1e-3,
                fused=False,
                factored_v_2d=False,
            )

        def assign_grads(model, grads):
            model.first.grad = distribute_tensor(grads[0].clone(), mesh, [Shard(0)])
            model.second.grad = distribute_tensor(grads[1].clone(), mesh, [Shard(0)])

        initial = [
            torch.linspace(-1, 1, 64).reshape(8, 8),
            torch.linspace(1, -1, 64).reshape(8, 8),
        ]
        first_grads = [
            torch.cat((torch.ones(4, 8), torch.arange(32).reshape(4, 8).sin())),
            torch.arange(64).reshape(8, 8).cos() * 0.75,
        ]
        second_grads = [
            torch.arange(64).reshape(8, 8).cos(),
            torch.arange(64).reshape(8, 8).sin() * 1.25,
        ]
        model = make_model(initial)
        optimizer = make_optimizer(model)

        pristine_osd = optimizer.state_dict()
        pristine_target_model = make_model(initial)
        pristine_target = make_optimizer(pristine_target_model)
        pristine_target.load_state_dict(pristine_osd)
        pristine_restored = _values_equal(
            _persistent_optimizer_snapshot(optimizer),
            _persistent_optimizer_snapshot(pristine_target),
        )

        assign_grads(model, first_grads)
        optimizer.step()

        current_params = [
            model.first.detach().full_tensor().clone(),
            model.second.detach().full_tensor().clone(),
        ]
        full_osd = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        handoff = [full_osd if rank == 0 else None]
        dist.broadcast_object_list(handoff, src=0)
        full_osd = handoff[0]

        saved_ids = [
            item
            for group in full_osd["param_groups"]
            for item in group["params"]
        ]
        carrier_state = full_osd["state"][saved_ids[0]]
        member_state = full_osd["state"][saved_ids[1]]
        expected_carrier_keys = {_carrier_key(item) for item in range(world)}
        markers = [
            group["_gefen_checkpoint_metadata"]["rank_local_sharded_state"]
            for group in full_osd["param_groups"]
        ]
        marker = markers[0]
        payload = Gefen._deserialize_rank_local_payload(
            carrier_state[_carrier_key(rank)]
        )
        schema_ok = (
            len(saved_ids) == 2
            and set(carrier_state) == expected_carrier_keys | {"name"}
            and carrier_state["name"] == 0
            and member_state == {"name": 0, _MEMBER: True}
            and all(
                torch.is_tensor(carrier_state[key])
                for key in expected_carrier_keys
            )
            and marker["format"] == _FORMAT
            and marker["world_size"] == world
            and marker["world_ranks"] == list(range(world))
            and marker["mesh"]["shape"] == [world]
            and marker["mesh"]["ranks"] == list(range(world))
            and all(item == marker for item in markers[1:])
            and payload is not None
            and payload["format"] == _FORMAT
            and payload["global_rank"] == rank
            and payload["group_rank"] == rank
            and payload["world_ranks"] == list(range(world))
            and payload["signature"] == marker["signatures"][str(rank)]
            and payload["parameter_manifest"] == marker["parameter_manifest"]
            and [item["name"] for item in payload["parameter_manifest"]]
            == ["first", "second"]
            and payload["global_step"] == optimizer._gefen_global_step
        )

        resumed_model = make_model(current_params)
        resumed = make_optimizer(resumed_model)
        resumed.load_state_dict(full_osd)
        restored = _values_equal(
            _persistent_optimizer_snapshot(optimizer),
            _persistent_optimizer_snapshot(resumed),
        )

        flat_osd = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
                flatten_optimizer_state_dict=True,
            ),
        )
        flat_handoff = [flat_osd if rank == 0 else None]
        dist.broadcast_object_list(flat_handoff, src=0)
        flat_osd = flat_handoff[0]
        flat_model = make_model(current_params)
        flat_resumed = make_optimizer(flat_model)
        set_optimizer_state_dict(
            flat_model,
            flat_resumed,
            flat_osd,
            options=StateDictOptions(
                flatten_optimizer_state_dict=True,
            ),
        )
        flatten_restored = _values_equal(
            _persistent_optimizer_snapshot(optimizer),
            _persistent_optimizer_snapshot(flat_resumed),
        )

        assign_grads(model, second_grads)
        assign_grads(resumed_model, second_grads)
        assign_grads(flat_model, second_grads)
        optimizer.step()
        resumed.step()
        flat_resumed.step()
        exact = all(
            torch.equal(source.detach().full_tensor(), target.detach().full_tensor())
            for source, target in zip(model.parameters(), resumed_model.parameters())
        )
        flatten_exact = all(
            torch.equal(source.detach().full_tensor(), target.detach().full_tensor())
            for source, target in zip(model.parameters(), flat_model.parameters())
        )

        original_step = optimizer._gefen_global_step
        optimizer._gefen_global_step = original_step + rank
        before_divergent_save = _optimizer_snapshot(optimizer)
        divergent_rejected = False
        try:
            optimizer.state_dict()
        except RuntimeError as exc:
            divergent_rejected = "global_step differs across ranks" in str(exc)
        divergent_unchanged = _values_equal(
            before_divergent_save, _optimizer_snapshot(optimizer)
        )
        optimizer._gefen_global_step = original_step

        def rewrite_rank_payload(checkpoint, mutate):
            local_ids = [
                item
                for group in checkpoint["param_groups"]
                for item in group["params"]
            ]
            local_carrier = checkpoint["state"][local_ids[0]]
            payload_key = _carrier_key(rank)
            local_payload = Gefen._deserialize_rank_local_payload(
                local_carrier[payload_key]
            )
            mutate(local_payload)
            local_carrier[payload_key] = Gefen._serialize_rank_local_payload(
                local_payload
            )

        corruptions = {}

        bad = copy.deepcopy(full_osd)
        bad["state"][saved_ids[0]][_carrier_key(rank)] = torch.zeros(
            8, dtype=torch.uint8
        )
        corruptions["corrupt_bytes"] = bad

        bad = copy.deepcopy(full_osd)
        for carrier_key in expected_carrier_keys:
            bad["state"][saved_ids[0]].pop(carrier_key)
        corruptions["missing_carrier"] = bad

        bad = copy.deepcopy(full_osd)
        bad["state"][saved_ids[0]].pop(_carrier_key(rank))
        corruptions["missing_rank_payload"] = bad

        bad = copy.deepcopy(full_osd)
        bad["state"][saved_ids[1]] = copy.deepcopy(
            bad["state"][saved_ids[0]]
        )
        corruptions["duplicate_carrier"] = bad

        bad = copy.deepcopy(full_osd)
        rewrite_rank_payload(
            bad,
            lambda item: item.__setitem__(
                "codebook", torch.full((256,), float("nan"), dtype=torch.float32)
            ),
        )
        corruptions["bad_codebook"] = bad

        bad = copy.deepcopy(full_osd)
        bad["param_groups"][-1]["params"] = [saved_ids[0]]
        corruptions["duplicate_outer_parameter_id"] = bad

        bad = copy.deepcopy(full_osd)
        for group in bad["param_groups"]:
            group["_gefen_checkpoint_metadata"]["rank_local_sharded_state"][
                "signatures"
            ][str(rank)][0]["local_shape"] = [999]
        corruptions["bad_topology"] = bad

        bad = copy.deepcopy(full_osd)
        rewrite_rank_payload(
            bad,
            lambda item: item.__setitem__("global_rank", (rank + 1) % world),
        )
        corruptions["bad_rank_binding"] = bad

        bad = copy.deepcopy(full_osd)
        rewrite_rank_payload(
            bad,
            lambda item: item["states"].reverse(),
        )
        corruptions["bad_parameter_order"] = bad

        bad = copy.deepcopy(full_osd)
        rewrite_rank_payload(
            bad,
            lambda item: item["states"][0].pop("step"),
        )
        corruptions["missing_step"] = bad

        bad = copy.deepcopy(full_osd)
        rewrite_rank_payload(
            bad,
            lambda item: item["states"][0]["m_magnitude"].view(-1).__setitem__(
                0, -1.0
            ),
        )
        corruptions["negative_m_magnitude"] = bad

        if optimizer_kind == "gefen":
            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0].pop("vmean"),
            )
            corruptions["missing_vmean"] = bad

            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0]["vmean"].view(-1).__setitem__(
                    0, -1.0
                ),
            )
            corruptions["negative_vmean"] = bad

            def install_bad_factored_state(item, key):
                pstate = item["states"][0]
                rows, cols = item["signature"][0]["shape"]
                pstate["v_row"] = torch.zeros(rows, dtype=torch.float32)
                pstate["v_col"] = torch.zeros(cols, dtype=torch.float32)
                pstate["factored_step"] = _clone_value(pstate["step"])
                pstate[key][0] = -1.0

            for key in ("v_row", "v_col"):
                bad = copy.deepcopy(full_osd)
                rewrite_rank_payload(
                    bad,
                    lambda item, key=key: install_bad_factored_state(item, key),
                )
                corruptions[f"negative_{key}"] = bad
        else:
            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0].pop("normuon_step"),
            )
            corruptions["incomplete_normuon_pair"] = bad

            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0]["normuon_v"].view(-1).__setitem__(
                    0, -1.0
                ),
            )
            corruptions["negative_normuon_v"] = bad

            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0].__setitem__(
                    "normuon_v",
                    torch.zeros(
                        item["states"][0]["normuon_v"].shape[0] + 1,
                        1,
                        dtype=torch.float32,
                    ),
                ),
            )
            corruptions["wrong_normuon_shape"] = bad

            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0].__setitem__(
                    "normuon_v", item["states"][0]["normuon_v"].to(torch.float64)
                ),
            )
            corruptions["wrong_normuon_dtype"] = bad

            bad = copy.deepcopy(full_osd)
            rewrite_rank_payload(
                bad,
                lambda item: item["states"][0].__setitem__("normuon_step", 0),
            )
            corruptions["zero_normuon_step"] = bad

        bad = copy.deepcopy(full_osd)
        second_metadata = copy.deepcopy(
            bad["param_groups"][1]["_gefen_checkpoint_metadata"]
        )
        second_metadata["rank_local_sharded_state"]["global_step"] += 1
        bad["param_groups"][1]["_gefen_checkpoint_metadata"] = second_metadata
        corruptions["inconsistent_group_marker"] = bad

        rejection_checks = {}
        for name, bad_checkpoint in corruptions.items():
            checkpoint_before = _clone_value(bad_checkpoint)
            target_model = make_model(current_params)
            target = make_optimizer(target_model)
            assign_grads(target_model, first_grads)
            target.step()
            before = _optimizer_snapshot(target)
            rejected = False
            try:
                target.load_state_dict(bad_checkpoint)
            except (ValueError, RuntimeError):
                rejected = True
            rejection_checks[name] = (
                rejected
                and _values_equal(before, _optimizer_snapshot(target))
                and _values_equal(checkpoint_before, bad_checkpoint)
            )

        rank_checks = [None] * world
        dist.all_gather_object(
            rank_checks,
            {
                "schema": schema_ok,
                "pristine_restored": pristine_restored,
                "restored": restored,
                "flatten_restored": flatten_restored,
                "exact": exact,
                "flatten_exact": flatten_exact,
                "divergent_rejected": divergent_rejected,
                "divergent_unchanged": divergent_unchanged,
                **rejection_checks,
            },
        )
        # Every rank reports. Previously only rank 0 put a result, so a failure
        # on any other rank was invisible: the parent took rank 0's success off
        # the queue and left the failing rank's traceback unread, surfacing the
        # whole thing as a bare nonzero exit code.
        result_queue.put(
            {"rank": rank, "checks": rank_checks if rank == 0 else None}
        )
    except BaseException:
        result_queue.put({"rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
@pytest.mark.parametrize("optimizer_kind", ["gefen", "muon_normuon"])
def test_full_dcp_handoff_is_exact_on_same_dtensor_topology(optimizer_kind):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_worker,
            args=(rank, 2, port, optimizer_kind, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    # Drain one result per rank: every worker reports, so a failure on a non-zero
    # rank surfaces its traceback here instead of being masked by rank 0's
    # success and reduced to an unexplained nonzero exit code.
    results = []
    terminated = []
    try:
        for _ in processes:
            results.append(result_queue.get(timeout=180))
    except queue.Empty:
        pass
    finally:
        for process in processes:
            process.join(timeout=60)
        for process in processes:
            if process.is_alive():
                terminated.append(process.pid)
                process.terminate()
                process.join(timeout=10)
    tracebacks = [item["traceback"] for item in results if "traceback" in item]
    assert not tracebacks, "distributed checkpoint worker raised:\n" + "\n".join(
        tracebacks
    )
    assert len(results) == len(processes), (
        "distributed checkpoint workers timed out: got {}/{} results, "
        "terminated={}, exitcodes={}".format(
            len(results), len(processes), terminated, [p.exitcode for p in processes]
        )
    )
    assert all(process.exitcode == 0 for process in processes), (
        "worker exited nonzero (terminated={}): exitcodes={}".format(
            terminated, [p.exitcode for p in processes]
        )
    )
    rank_checks = next(
        (item["checks"] for item in results if item.get("checks") is not None), None
    )
    assert isinstance(rank_checks, list), rank_checks
    failures = [
        {name: value for name, value in checks.items() if not value}
        for checks in rank_checks
    ]
    assert all(not rank_failures for rank_failures in failures), failures


def _fully_shard_worker(
    rank: int, world: int, port: str, optimizer_kind: str, result_queue
) -> None:
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor

    from gefen import Gefen, GefenMuon

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
        full = torch.linspace(-1, 1, 64, device="cuda").reshape(8, 8)
        first_grad = torch.cat(
            (
                torch.ones(4, 8, device="cuda"),
                torch.arange(32, device="cuda").reshape(4, 8).sin(),
            ),
            dim=0,
        )
        model = nn.Linear(8, 8, bias=False, device="cuda")
        with torch.no_grad():
            model.weight.copy_(full)
        fully_shard(model, mesh=mesh)
        if optimizer_kind == "gefen":
            optimizer = Gefen(
                model.named_parameters(),
                lr=1e-3,
                fused=False,
                factored_v_2d=False,
            )
        else:
            optimizer = GefenMuon(
                model.named_parameters(),
                lr=1e-3,
                fused=False,
                sharded_mode="approx",
            )
        model.weight.grad = distribute_tensor(first_grad, mesh, [Shard(0)])
        optimizer.step()
        current = model.weight.detach().full_tensor().clone()
        original_codebook = optimizer._gefen_codebook.detach().cpu().clone()
        original_indices = optimizer.state[model.weight]["m_codebook"].detach().cpu().clone()
        full_osd = get_optimizer_state_dict(
            model,
            optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        resumed_model = nn.Linear(8, 8, bias=False, device="cuda")
        with torch.no_grad():
            resumed_model.weight.copy_(current)
        fully_shard(resumed_model, mesh=mesh)
        if optimizer_kind == "gefen":
            resumed = Gefen(
                resumed_model.named_parameters(),
                lr=1e-3,
                fused=False,
                factored_v_2d=False,
            )
        else:
            resumed = GefenMuon(
                resumed_model.named_parameters(),
                lr=1e-3,
                fused=False,
                sharded_mode="approx",
            )
        set_optimizer_state_dict(
            resumed_model,
            resumed,
            full_osd if rank == 0 else {},
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )
        restored = (
            torch.equal(resumed._gefen_codebook.detach().cpu(), original_codebook)
            and torch.equal(
                resumed.state[resumed_model.weight]["m_codebook"].detach().cpu(),
                original_indices,
            )
        )

        second_grad = torch.arange(64, device="cuda").reshape(8, 8).cos()
        model.weight.grad = distribute_tensor(second_grad, mesh, [Shard(0)])
        resumed_model.weight.grad = distribute_tensor(second_grad, mesh, [Shard(0)])
        optimizer.step()
        resumed.step()
        uninterrupted = model.weight.detach().full_tensor()
        after_resume = resumed_model.weight.detach().full_tensor()
        exact = torch.equal(uninterrupted, after_resume)
        checks = [None] * world
        dist.all_gather_object(checks, (restored, exact))
        if rank == 0:
            result_queue.put(checks)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_nccl_available(),
    reason="fully_shard checkpoint continuation requires two CUDA GPUs and NCCL",
)
@pytest.mark.parametrize("optimizer_kind", ["gefen", "muon_approx"])
def test_rank_local_full_dcp_set_optimizer_state_is_exact_under_fully_shard(
    optimizer_kind,
):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_fully_shard_worker,
            args=(rank, 2, port, optimizer_kind, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        checks = result_queue.get(timeout=180)
    except queue.Empty:
        checks = None
    for process in processes:
        process.join(timeout=180)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert checks is not None, "fully_shard checkpoint workers timed out"
    assert all(process.exitcode == 0 for process in processes)
    assert all(all(rank_check) for rank_check in checks), checks


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="rank-local unwrap needs a (gloo) process group and DTensor",
)
def test_rank_local_unwrap_tolerates_and_backfills_legacy_vmean():
    # Exercise the actual load path -- _unwrap_rank_local_sharded_checkpoint --
    # not just the validator: a pre-counter rank-local checkpoint (vmean without
    # the separate vmean_step counter) must load, and the first resumed step must
    # backfill vmean_step from step. A single-rank gloo world drives the same
    # rank_local_dtensor_v2 format the multi-rank path uses, so it stays a
    # CPU-only regression guard for the load-path tolerance (rejected pre-fix).
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = _free_port()
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("gloo", rank=0, world_size=1)
    try:
        from gefen import Gefen

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp",))

        def build():
            model = nn.Module()
            model.register_parameter(
                "w",
                nn.Parameter(
                    distribute_tensor(
                        torch.linspace(-1, 1, 64).reshape(8, 8), mesh, [Shard(0)]
                    )
                ),
            )
            optimizer = Gefen(
                [{"params": [("w", model.w)]}],
                lr=1e-3,
                fused=False,
                factored_v_2d=False,
            )
            return model, optimizer

        def apply_grad(model):
            model.w.grad = distribute_tensor(
                torch.arange(64).reshape(8, 8).float().cos(), mesh, [Shard(0)]
            )

        model, optimizer = build()
        for _ in range(2):
            apply_grad(model)
            optimizer.step()

        # Save a pre-counter rank-local checkpoint: drop the separate vmean_step
        # counter from the live state before serializing.
        param = optimizer.param_groups[0]["params"][0]
        assert "vmean_step" in optimizer.state[param]
        optimizer.state[param].pop("vmean_step")
        checkpoint = copy.deepcopy(optimizer.state_dict())
        assert any(
            "rank_local_sharded_state" in group.get("_gefen_checkpoint_metadata", {})
            for group in checkpoint["param_groups"]
        ), "expected a rank-local sharded checkpoint"

        # Load through the rank-local unwrap path (rejected before the fix). The
        # pre-counter payload must be accepted with vmean_step still absent...
        target_model, target_optimizer = build()
        target_optimizer.load_state_dict(checkpoint)
        target_param = target_optimizer.param_groups[0]["params"][0]
        assert "vmean" in target_optimizer.state[target_param]
        assert "vmean_step" not in target_optimizer.state[target_param]

        # ...and the first resumed step must backfill vmean_step from step.
        apply_grad(target_model)
        target_optimizer.step()
        assert "vmean_step" in target_optimizer.state[target_param]
    finally:
        dist.destroy_process_group()
