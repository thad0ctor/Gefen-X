"""Real-process-group coverage for atomic optimizer-state movement."""

from __future__ import annotations

from datetime import timedelta
import os
import queue
import socket
import traceback

import pytest
import torch


_AUTHORITATIVE_TENSOR_KEYS = frozenset(
    {
        "step",
        "m_codebook",
        "m_magnitude",
        "vmean",
        "vmean_step",
        "v_row",
        "v_col",
        "factored_step",
        "normuon_v",
        "normuon_step",
    }
)


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _oversized_copy(tensor: torch.Tensor) -> torch.Tensor:
    backing = torch.empty(
        tensor.numel() + 13,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result = backing.narrow(0, 7, tensor.numel()).view(tensor.shape)
    result.copy_(tensor)
    assert result.untyped_storage().nbytes() > tensor.numel() * tensor.element_size()
    return result


def _assert_fresh_tight_copy(
    actual: torch.Tensor,
    old: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert type(actual) is torch.Tensor
    assert actual is not old
    assert actual.device == torch.device("cpu")
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.layout is torch.strided
    assert actual.is_contiguous()
    assert actual.storage_offset() == 0
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


def _seed_and_snapshot_movable_state(optimizer, parameters, rank: int, label: str):
    first_state = optimizer.state[parameters[0]]
    carrier = torch.tensor([rank, len(label), 41], dtype=torch.int64)
    metadata = {"label": label, "members": ["preserve", rank]}
    first_state["_gefen_rank_local_payload_{}".format(rank)] = carrier
    first_state["movement_metadata"] = metadata

    records = []
    for parameter_index, parameter in enumerate(parameters):
        for key, value in tuple(optimizer.state[parameter].items()):
            if key not in _AUTHORITATIVE_TENSOR_KEYS or not torch.is_tensor(value):
                continue
            assert type(value) is torch.Tensor
            oversized = _oversized_copy(value)
            optimizer.state[parameter][key] = oversized
            records.append(
                (
                    parameter_index,
                    key,
                    oversized,
                    oversized.detach().clone(),
                )
            )

    assert type(optimizer._gefen_codebook) is torch.Tensor
    optimizer._gefen_codebook = _oversized_copy(optimizer._gefen_codebook)
    codebook_record = (
        optimizer._gefen_codebook,
        optimizer._gefen_codebook.detach().clone(),
    )
    return records, codebook_record, carrier, metadata


def _assert_authoritative_state_equal(optimizer, reference, parameters, reference_parameters):
    for parameter, reference_parameter in zip(parameters, reference_parameters):
        state = optimizer.state[parameter]
        reference_state = reference.state[reference_parameter]
        keys = {
            key
            for key in state
            if key in _AUTHORITATIVE_TENSOR_KEYS
        } | {
            key
            for key in reference_state
            if key in _AUTHORITATIVE_TENSOR_KEYS
        }
        for key in keys:
            assert key in state and key in reference_state
            actual = state[key]
            expected = reference_state[key]
            if torch.is_tensor(expected):
                assert torch.is_tensor(actual)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            else:
                assert actual == expected
    torch.testing.assert_close(
        optimizer._gefen_codebook,
        reference._gefen_codebook,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def _dtensor_case(rank, world, mesh, kind: str) -> None:
    import torch.nn as nn
    from torch.distributed.tensor import Shard, distribute_tensor

    from gefen import Gefen, GefenMuon

    shapes = ((4, 4), (4, 6)) if kind == "gefen" else ((4, 4), (1, 4))
    generator = torch.Generator().manual_seed(7100 + sum(ord(item) for item in kind))
    initial = [torch.randn(shape, generator=generator) * 0.05 for shape in shapes]

    def build():
        parameters = [
            nn.Parameter(distribute_tensor(value.clone(), mesh, [Shard(0)]))
            for value in initial
        ]
        tensor_lr = torch.tensor(2.0e-3)
        group_metadata = {"kind": kind, "ordered_members": list(range(world))}
        group = {
            "params": [
                ("{}.{}".format(kind, index), parameter)
                for index, parameter in enumerate(parameters)
            ],
            "lr": tensor_lr,
            "movement_metadata": group_metadata,
        }
        if kind == "gefen":
            optimizer = Gefen(
                [group],
                lr=tensor_lr,
                fused=False,
                factored_v_2d=False,
            )
        else:
            optimizer = GefenMuon(
                [group],
                lr=tensor_lr,
                fused=False,
                ns_steps=1,
                weight_decay=0.0,
                sharded_mode=kind,
            )
        optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
        optimizer._predict_period_from_grad_sq = lambda *args, **kwargs: 4
        return optimizer, parameters, tensor_lr, group_metadata

    def assign_grads(parameters, seed):
        grad_generator = torch.Generator().manual_seed(seed)
        for parameter, shape in zip(parameters, shapes):
            full_grad = torch.randn(shape, generator=grad_generator) * 0.01
            parameter.grad = distribute_tensor(full_grad, mesh, [Shard(0)])

    optimizer, parameters, tensor_lr, group_metadata = build()
    reference, reference_parameters, _, _ = build()
    assign_grads(parameters, 7200)
    assign_grads(reference_parameters, 7200)
    optimizer.step()
    reference.step()

    if kind == "distributed":
        stateful = {
            index
            for index, parameter in enumerate(parameters)
            if any(
                key in _AUTHORITATIVE_TENSOR_KEYS
                for key in optimizer.state[parameter]
            )
        }
        assert stateful == {rank}
        if rank == 1:
            assert parameters[1].to_local().numel() == 0
            assert optimizer.state[parameters[1]]["m_codebook"].numel() == shapes[1][0] * shapes[1][1]
    elif kind == "approx":
        stateful = {
            index
            for index, parameter in enumerate(parameters)
            if any(
                key in _AUTHORITATIVE_TENSOR_KEYS
                for key in optimizer.state[parameter]
            )
        }
        assert stateful == ({0, 1} if rank == 0 else {0})
    else:
        assert all(
            any(key in _AUTHORITATIVE_TENSOR_KEYS for key in optimizer.state[parameter])
            for parameter in parameters
        )

    records, codebook_record, carrier, state_metadata = _seed_and_snapshot_movable_state(
        optimizer,
        parameters,
        rank,
        kind,
    )
    assert records
    contract_before = optimizer.optimizer_contract()
    assert contract_before.capabilities.atomic_state_movement
    assert not contract_before.capabilities.state_offload

    groups_before = optimizer.param_groups
    group_before = optimizer.param_groups[0]
    group_params_before = group_before["params"]
    defaults_before = optimizer.defaults
    names_before = optimizer._param_names
    grads_before = [parameter.grad for parameter in parameters]
    local_values_before = [parameter.detach().to_local().clone() for parameter in parameters]
    meshes_before = [parameter.device_mesh for parameter in parameters]
    placements_before = [parameter.placements for parameter in parameters]
    mesh_groups_before = [parameter.device_mesh.get_group() for parameter in parameters]
    codebook_binding_before = optimizer._gefen_codebook_process_group
    shard_bindings_before = optimizer._gefen_shard_bindings
    local_bindings_before = optimizer._gefen_local_shard_bindings
    manifest_before = optimizer._gefen_sharding_manifest

    optimizer.move_state_()

    assert optimizer.optimizer_contract() == contract_before
    assert optimizer.param_groups is groups_before
    assert optimizer.param_groups[0] is group_before
    assert optimizer.param_groups[0]["params"] is group_params_before
    assert optimizer.param_groups[0]["lr"] is tensor_lr
    assert optimizer.param_groups[0]["movement_metadata"] is group_metadata
    assert optimizer.defaults is defaults_before
    assert optimizer.defaults["lr"] is tensor_lr
    assert optimizer._param_names is names_before
    assert optimizer._gefen_codebook_process_group is codebook_binding_before
    assert optimizer._gefen_shard_bindings is shard_bindings_before
    assert optimizer._gefen_local_shard_bindings is local_bindings_before
    assert optimizer._gefen_sharding_manifest is manifest_before

    for index, parameter in enumerate(parameters):
        assert optimizer.param_groups[0]["params"][index] is parameter
        assert parameter.grad is grads_before[index]
        assert parameter.device_mesh is meshes_before[index]
        assert parameter.placements == placements_before[index]
        assert parameter.device_mesh.get_group() is mesh_groups_before[index]
        torch.testing.assert_close(
            parameter.detach().to_local(),
            local_values_before[index],
            rtol=0,
            atol=0,
        )

    for parameter_index, key, old, expected in records:
        _assert_fresh_tight_copy(
            optimizer.state[parameters[parameter_index]][key],
            old,
            expected,
        )
    _assert_fresh_tight_copy(
        optimizer._gefen_codebook,
        codebook_record[0],
        codebook_record[1],
    )
    assert optimizer.state[parameters[0]]["_gefen_rank_local_payload_{}".format(rank)] is carrier
    assert optimizer.state[parameters[0]]["movement_metadata"] is state_metadata

    assign_grads(parameters, 7300)
    assign_grads(reference_parameters, 7300)
    optimizer.step()
    reference.step()
    for parameter, reference_parameter in zip(parameters, reference_parameters):
        torch.testing.assert_close(
            parameter.detach().to_local(),
            reference_parameter.detach().to_local(),
            rtol=0,
            atol=0,
        )
    _assert_authoritative_state_equal(
        optimizer,
        reference,
        parameters,
        reference_parameters,
    )


def _whole_owner_case(rank, world) -> None:
    import torch.distributed as dist
    import torch.nn as nn

    from gefen import (
        CodebookProcessGroupBinding,
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

    members = tuple("rank:{}".format(index) for index in range(world))
    group_identity = ProcessGroupIdentity("movement_owner", members)
    identities = (
        ParameterIdentity("Owner.First", (4, 4)),
        ParameterIdentity("Owner.Second", (4, 4)),
    )

    def owner_shard(identity, member, owner):
        return ShardIdentity(
            identity,
            ParameterLayout.WHOLE_PARAMETER_OWNER,
            LogicalSlice.full(identity) if member == owner else LogicalSlice(0, 0),
            process_group=group_identity,
            local_member=member,
            owner=owner,
            placements=(
                ShardPlacement(
                    "dp",
                    PlacementKind.WHOLE_PARAMETER_OWNER,
                    members.index(member),
                    world,
                ),
            ),
        )

    records_by_parameter = tuple(
        tuple(
            owner_shard(identity, member, members[index])
            for member in members
        )
        for index, identity in enumerate(identities)
    )
    manifest = ShardingManifest(
        tuple(
            shard
            for parameter_records in records_by_parameter
            for shard in parameter_records
        )
    )

    def build():
        generator = torch.Generator().manual_seed(7400)
        old_parameters = [
            nn.Parameter(torch.randn(identity.global_shape, generator=generator) * 0.05)
            for identity in identities
        ]
        tensor_lr = torch.tensor(2.0e-3)
        group_metadata = {"layout": "whole-owner", "rank": rank}
        optimizer = GefenMuon(
            [
                {
                    "params": [
                        ("owner.{}".format(index), parameter)
                        for index, parameter in enumerate(old_parameters)
                    ],
                    "lr": tensor_lr,
                    "movement_metadata": group_metadata,
                }
            ],
            lr=tensor_lr,
            fused=False,
            ns_steps=1,
            weight_decay=0.0,
        )
        local_member = members[rank]
        local_records = [
            next(
                shard
                for shard in parameter_records
                if shard.local_member == local_member
            )
            for parameter_records in records_by_parameter
        ]
        binding = CodebookProcessGroupBinding(
            group_identity,
            local_member,
            dist.group.WORLD,
            torch.device("cpu"),
        )
        optimizer.post_sharding(
            tuple(
                ParameterRebinding(
                    parameter,
                    parameter if local_record.owner == local_member else None,
                    local_record,
                )
                for parameter, local_record in zip(old_parameters, local_records)
            ),
            manifest=manifest,
            codebook_process_group=binding,
        )
        optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
        optimizer._predict_period_from_grad_sq = lambda *args, **kwargs: 4
        return (
            optimizer,
            list(optimizer.param_groups[0]["params"]),
            tensor_lr,
            group_metadata,
            binding,
        )

    def assign_grad(parameters, seed):
        generator = torch.Generator().manual_seed(seed + rank)
        assert len(parameters) == 1
        parameters[0].grad = torch.randn(parameters[0].shape, generator=generator) * 0.01

    optimizer, parameters, tensor_lr, group_metadata, binding = build()
    reference, reference_parameters, _, _, _ = build()
    assert len(parameters) == 1
    assert len(optimizer.shard_bindings()) == 2
    assert sum(parameter is None for parameter, _ in optimizer.shard_bindings()) == 1

    assign_grad(parameters, 7500)
    assign_grad(reference_parameters, 7500)
    optimizer.step()
    reference.step()

    records, codebook_record, carrier, state_metadata = _seed_and_snapshot_movable_state(
        optimizer,
        parameters,
        rank,
        "whole-owner",
    )
    assert records
    contract_before = optimizer.optimizer_contract()
    assert contract_before.capabilities.atomic_state_movement
    assert not contract_before.capabilities.state_offload
    bindings_before = optimizer.shard_bindings()
    manifest_before = optimizer.sharding_manifest()
    binding_before = optimizer.codebook_process_group_binding()
    groups_before = optimizer.param_groups
    group_before = optimizer.param_groups[0]
    group_params_before = group_before["params"]
    defaults_before = optimizer.defaults
    names_before = optimizer._param_names
    grad_before = parameters[0].grad
    value_before = parameters[0].detach().clone()

    optimizer.move_state_()

    assert optimizer.optimizer_contract() == contract_before
    assert optimizer.param_groups is groups_before
    assert optimizer.param_groups[0] is group_before
    assert optimizer.param_groups[0]["params"] is group_params_before
    assert optimizer.param_groups[0]["params"][0] is parameters[0]
    assert optimizer.param_groups[0]["lr"] is tensor_lr
    assert optimizer.param_groups[0]["movement_metadata"] is group_metadata
    assert optimizer.defaults is defaults_before
    assert optimizer.defaults["lr"] is tensor_lr
    assert optimizer._param_names is names_before
    assert optimizer.shard_bindings() is bindings_before
    assert optimizer.sharding_manifest() is manifest_before
    assert optimizer.codebook_process_group_binding() is binding_before is binding
    assert parameters[0].grad is grad_before
    torch.testing.assert_close(parameters[0], value_before, rtol=0, atol=0)

    for parameter_index, key, old, expected in records:
        _assert_fresh_tight_copy(
            optimizer.state[parameters[parameter_index]][key],
            old,
            expected,
        )
    _assert_fresh_tight_copy(
        optimizer._gefen_codebook,
        codebook_record[0],
        codebook_record[1],
    )
    assert optimizer.state[parameters[0]]["_gefen_rank_local_payload_{}".format(rank)] is carrier
    assert optimizer.state[parameters[0]]["movement_metadata"] is state_metadata

    assign_grad(parameters, 7600)
    assign_grad(reference_parameters, 7600)
    optimizer.step()
    reference.step()
    torch.testing.assert_close(
        parameters[0],
        reference_parameters[0],
        rtol=0,
        atol=0,
    )
    _assert_authoritative_state_equal(
        optimizer,
        reference,
        parameters,
        reference_parameters,
    )


def _distributed_worker(rank, world, port, result_queue) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor import init_device_mesh

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=90),
        )
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))
        for kind in ("gefen", "exact", "approx", "distributed"):
            _dtensor_case(rank, world, mesh, kind)
        _whole_owner_case(rank, world)
        result_queue.put(("result", rank))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason="distributed state movement coverage needs Gloo",
)
def test_atomic_state_movement_across_distributed_cpu_representations():
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, world, port, result_queue),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()

    messages = []
    try:
        for _ in range(world):
            messages.append(result_queue.get(timeout=180))
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
    errors = [item[2] for item in messages if item[0] == "error"]
    assert not errors, "\n".join(errors)
    assert {item[1] for item in messages if item[0] == "result"} == set(range(world))
