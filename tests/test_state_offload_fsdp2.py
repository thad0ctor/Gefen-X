"""Two-GPU FSDP2 coverage for CPU-authoritative optimizer-state offload."""

import copy
import os
import queue
import socket
import traceback

import pytest
import torch
import torch.nn as nn


def _free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _state_tensors_are_cpu(optimizer) -> bool:
    return all(
        not torch.is_tensor(value) or value.device.type == "cpu"
        for parameter_state in optimizer.state.values()
        for value in parameter_state.values()
    )


def _restored_state_uses_local_cuda(optimizer) -> bool:
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if not torch.is_tensor(value):
                continue
            if str(key).startswith("_gefen_rank_local_payload_"):
                if value.device.type != "cpu":
                    return False
            elif value.device.type != "cuda":
                return False
    return True


def _fixed_codebook():
    return torch.linspace(-1.0, 1.0, 256, dtype=torch.float32, device="cuda")


def _fsdp2_offload_worker(
    rank, world, port, activate_before_first_step, fused, result_queue
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

        class TinyShardModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(8, 7, bias=True, device="cuda")
                self.tiny = nn.Parameter(torch.tensor([0.125], device="cuda"))

            def forward(self, inputs):
                return self.linear(inputs) + self.tiny.reshape(1, 1)

        torch.manual_seed(11)
        reference_model = TinyShardModel()
        offloaded_model = copy.deepcopy(reference_model)
        fully_shard(reference_model, mesh=mesh)
        fully_shard(offloaded_model, mesh=mesh)

        def make_optimizer(model):
            optimizer = Gefen(
                model.named_parameters(),
                lr=3e-3,
                fused=fused,
                factored_v_2d=True,
            )

            def resolve_period(_name, _parameter, grad):
                local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
                if hasattr(local_grad, "wait"):
                    local_grad = local_grad.wait()
                return 4 if local_grad.numel() and local_grad.numel() % 4 == 0 else 1

            optimizer._resolve_automatic_period = resolve_period
            return optimizer

        reference_optimizer = make_optimizer(reference_model)
        offloaded_optimizer = make_optimizer(offloaded_model)
        if activate_before_first_step:
            offloaded_optimizer.offload_state_()

        generator = torch.Generator(device="cuda").manual_seed(500 + rank)
        checks = {
            "exact_losses": True,
            "exact_local_parameters": True,
            "cpu_authoritative": True,
        }
        for step in range(4):
            inputs = torch.randn(12, 8, generator=generator, device="cuda")
            targets = torch.randn(12, 7, generator=generator, device="cuda")
            losses = []
            for model, optimizer in (
                (reference_model, reference_optimizer),
                (offloaded_model, offloaded_optimizer),
            ):
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(inputs) - targets) ** 2).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
            checks["exact_losses"] &= torch.equal(losses[0], losses[1])
            for reference_parameter, offloaded_parameter in zip(
                reference_model.parameters(), offloaded_model.parameters()
            ):
                checks["exact_local_parameters"] &= torch.equal(
                    reference_parameter.to_local(), offloaded_parameter.to_local()
                )
            if step == 0 and not activate_before_first_step:
                offloaded_optimizer.offload_state_()
            if offloaded_optimizer.state_offload_active:
                checks["cpu_authoritative"] &= (
                    _state_tensors_are_cpu(offloaded_optimizer)
                    and torch.is_tensor(offloaded_optimizer._gefen_codebook)
                    and offloaded_optimizer._gefen_codebook.device.type == "cuda"
                )

        checks["native_codebook"] = (
            torch.is_tensor(offloaded_optimizer._gefen_codebook)
            and offloaded_optimizer._gefen_codebook.device.type == "cuda"
        )
        checks["member_schema"] = any(
            state.get("_gefen_rank_local_member") is True
            for state in offloaded_optimizer.state.values()
        )
        local_sizes = [
            parameter.to_local().numel()
            for parameter in offloaded_model.parameters()
        ]
        checks["empty_shard_exercised"] = (rank == 0) or 0 in local_sizes

        offloaded_optimizer.restore_state_()
        checks["restored"] = (
            not offloaded_optimizer.state_offload_active
            and _restored_state_uses_local_cuda(offloaded_optimizer)
        )
        gathered = [None] * world
        dist.all_gather_object(gathered, checks)
        if rank == 0:
            result_queue.put(gathered)
    except BaseException:
        result_queue.put(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _fsdp2_offload_dcp_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

        full_weight = torch.linspace(-0.7, 0.8, 56, device="cuda").reshape(7, 8)
        full_bias = torch.linspace(-0.2, 0.3, 7, device="cuda")
        source_model = nn.Linear(8, 7, bias=True, device="cuda")
        with torch.no_grad():
            source_model.weight.copy_(full_weight)
            source_model.bias.copy_(full_bias)
        fully_shard(source_model, mesh=mesh)
        source_optimizer = Gefen(
            source_model.named_parameters(),
            lr=2e-3,
            fused=False,
            factored_v_2d=True,
        )
        def resolve_period(_name, _parameter, grad):
            local_grad = grad.to_local() if hasattr(grad, "to_local") else grad
            if hasattr(local_grad, "wait"):
                local_grad = local_grad.wait()
            return 4 if local_grad.numel() and local_grad.numel() % 4 == 0 else 1

        source_optimizer._resolve_automatic_period = resolve_period
        source_optimizer._gefen_codebook = _fixed_codebook()
        source_optimizer.offload_state_()

        generator = torch.Generator(device="cuda").manual_seed(900 + rank)
        for _ in range(2):
            inputs = torch.randn(10, 8, generator=generator, device="cuda")
            targets = torch.randn(10, 7, generator=generator, device="cuda")
            source_optimizer.zero_grad(set_to_none=True)
            loss = ((source_model(inputs) - targets) ** 2).mean()
            loss.backward()
            source_optimizer.step()

        full_osd = get_optimizer_state_dict(
            source_model,
            source_optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if full_osd:
            serialized_states = list(full_osd["state"].values())
            format_preserved = (
                len(serialized_states) == 2
                and sum(
                    any(
                        str(key).startswith("_gefen_rank_local_payload_")
                        for key in state
                    )
                    for state in serialized_states
                )
                == 1
                and sum(
                    state.get("_gefen_rank_local_member") is True
                    for state in serialized_states
                )
                == 1
                and all(
                    group.get("_gefen_checkpoint_metadata", {})
                    .get("rank_local_sharded_state", {})
                    .get("format")
                    == "rank_local_dtensor_v2"
                    for group in full_osd["param_groups"]
                )
            )
        else:
            format_preserved = rank != 0

        current_weight = source_model.weight.detach().full_tensor().clone()
        current_bias = source_model.bias.detach().full_tensor().clone()
        resumed_model = nn.Linear(8, 7, bias=True, device="cuda")
        with torch.no_grad():
            resumed_model.weight.copy_(current_weight)
            resumed_model.bias.copy_(current_bias)
        fully_shard(resumed_model, mesh=mesh)
        resumed_optimizer = Gefen(
            resumed_model.named_parameters(),
            lr=2e-3,
            fused=False,
            factored_v_2d=True,
        )
        resumed_optimizer._resolve_automatic_period = resolve_period
        resumed_optimizer.offload_state_()
        set_optimizer_state_dict(
            resumed_model,
            resumed_optimizer,
            full_osd if rank == 0 else {},
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        checks = {
            "format_preserved": format_preserved,
            "active_after_load": resumed_optimizer.state_offload_active
            and _state_tensors_are_cpu(resumed_optimizer)
            and resumed_optimizer._gefen_codebook.device.type == "cuda",
        }
        inputs = torch.randn(10, 8, generator=generator, device="cuda")
        targets = torch.randn(10, 7, generator=generator, device="cuda")
        losses = []
        for model, optimizer in (
            (source_model, source_optimizer),
            (resumed_model, resumed_optimizer),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(inputs) - targets) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
        checks["exact_continuation"] = torch.equal(
            losses[0], losses[1]
        ) and all(
            torch.equal(source_parameter.to_local(), resumed_parameter.to_local())
            for source_parameter, resumed_parameter in zip(
                source_model.parameters(), resumed_model.parameters()
            )
        )
        checks["cpu_after_step"] = _state_tensors_are_cpu(resumed_optimizer)

        gathered = [None] * world
        dist.all_gather_object(gathered, checks)
        if rank == 0:
            result_queue.put(gathered)
    except BaseException:
        result_queue.put(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _fsdp2_offload_failure_sync_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
        model = nn.Linear(8, 8, bias=False, device="cuda")
        fully_shard(model, mesh=mesh)
        optimizer = Gefen(
            model.named_parameters(), lr=2e-3, fused=False, factored_v_2d=False
        )
        optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
        optimizer._gefen_codebook = _fixed_codebook()

        first_grad = torch.linspace(-0.8, 0.9, 64, device="cuda").reshape(8, 8)
        model.weight.grad = distribute_tensor(first_grad, mesh, [Shard(0)])
        optimizer.step()
        state_before = optimizer.state
        parameter_state_before = optimizer.state[model.weight]
        codebook_before = optimizer._gefen_codebook

        if rank == 0:
            def fail_cpu_copy(_tensor):
                raise RuntimeError("injected rank-local activation failure")

            optimizer._copy_state_tensor_to_offload_cpu = fail_cpu_copy
        activation_error = None
        try:
            optimizer.offload_state_()
        except RuntimeError as exc:
            activation_error = str(exc)
        if rank == 0:
            del optimizer._copy_state_tensor_to_offload_cpu
        checks = {
            "activation_atomic": activation_error is not None
            and "activation staging" in activation_error
            and optimizer.state is state_before
            and optimizer.state[model.weight] is parameter_state_before
            and optimizer._gefen_codebook is codebook_before
            and not optimizer.state_offload_active
        }

        optimizer.offload_state_()
        local_parameter_before = model.weight.to_local().detach().clone()
        global_step_before = optimizer._gefen_global_step
        if rank == 0:
            optimizer.state[model.weight]["extension"] = torch.ones(1)
        second_grad = torch.linspace(0.7, -0.6, 64, device="cuda").reshape(8, 8)
        model.weight.grad = distribute_tensor(second_grad, mesh, [Shard(0)])
        step_error = None
        try:
            optimizer.step()
        except RuntimeError as exc:
            step_error = str(exc)
        if rank == 0:
            optimizer.state[model.weight].pop("extension")
        checks["step_atomic"] = (
            step_error is not None
            and "step preflight" in step_error
            and torch.equal(model.weight.to_local(), local_parameter_before)
            and optimizer._gefen_global_step == global_step_before
        )

        if rank == 0:
            optimizer._gefen_state_offload_poisoned = True
        export_error = None
        try:
            optimizer.state_dict()
        except RuntimeError as exc:
            export_error = str(exc)
        checks["export_synchronized"] = (
            export_error is not None and "state export preflight" in export_error
        )

        gathered = [None] * world
        dist.all_gather_object(gathered, checks)
        if rank == 0:
            result_queue.put(gathered)
    except BaseException:
        result_queue.put(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _fsdp2_offload_extended_failure_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard, distribute_tensor

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
        model = nn.Linear(8, 8, bias=False, device="cuda")
        fully_shard(model, mesh=mesh)
        optimizer = Gefen(model.named_parameters(), lr=2e-3, fused=False)
        optimizer._resolve_automatic_period = lambda *args, **kwargs: 4
        optimizer._gefen_codebook = _fixed_codebook()

        first_grad = torch.linspace(-0.8, 0.9, 64, device="cuda").reshape(8, 8)
        model.weight.grad = distribute_tensor(first_grad, mesh, [Shard(0)])
        optimizer.step()
        checks = {}

        state_before = optimizer.state
        invalid_device_error = None
        try:
            optimizer.offload_state_("cuda" if rank == 0 else "cpu")
        except RuntimeError as exc:
            invalid_device_error = str(exc)
        checks["device_validation_synchronized"] = (
            invalid_device_error is not None
            and "activation preflight" in invalid_device_error
            and optimizer.state is state_before
            and not optimizer.state_offload_active
        )

        original_params = optimizer.param_groups[0]["params"]
        if rank == 0:
            optimizer.param_groups[0]["params"] = []
        ownership_error = None
        try:
            optimizer.offload_state_()
        except RuntimeError as exc:
            ownership_error = str(exc)
        if rank == 0:
            optimizer.param_groups[0]["params"] = original_params
        checks["ownership_failure_synchronized"] = (
            ownership_error is not None
            and "activation preflight" in ownership_error
            and optimizer.state is state_before
            and not optimizer.state_offload_active
        )

        optimizer.offload_state_()
        state_before = optimizer.state
        parameter_state_before = optimizer.state[model.weight]
        codebook_before = optimizer._gefen_codebook
        if rank == 0:

            def fail_restore_copy(_tensor, _device):
                raise RuntimeError("injected rank-local restore failure")

            optimizer._copy_state_tensor_for_move = fail_restore_copy
        restore_error = None
        try:
            optimizer.restore_state_()
        except RuntimeError as exc:
            restore_error = str(exc)
        if rank == 0:
            del optimizer._copy_state_tensor_for_move
        checks["restore_failure_atomic"] = (
            restore_error is not None
            and "state movement staging" in restore_error
            and optimizer.state is state_before
            and optimizer.state[model.weight] is parameter_state_before
            and optimizer._gefen_codebook is codebook_before
            and optimizer.state_offload_active
        )

        removed_transport = None
        if rank == 0:
            carrier_state = optimizer.state[model.weight]
            transport_key = next(
                key
                for key in carrier_state
                if str(key).startswith("_gefen_rank_local_payload_")
            )
            removed_transport = (transport_key, carrier_state.pop(transport_key))
        state_before = optimizer.state
        schema_error = None
        try:
            optimizer.move_state_()
        except RuntimeError as exc:
            schema_error = str(exc)
        if removed_transport is not None:
            optimizer.state[model.weight][removed_transport[0]] = removed_transport[1]
        checks["movement_schema_failure_atomic"] = (
            schema_error is not None
            and "state movement preflight" in schema_error
            and optimizer.state is state_before
            and optimizer.state_offload_active
        )

        export_hook = None
        if rank == 0:

            def fail_export_hook(_optimizer):
                raise RuntimeError("injected rank-local export hook failure")

            export_hook = optimizer.register_state_dict_pre_hook(fail_export_hook)
        export_error = None
        try:
            optimizer.state_dict()
        except RuntimeError as exc:
            export_error = str(exc)
        if export_hook is not None:
            export_hook.remove()
        checks["export_hook_synchronized"] = (
            export_error is not None and "state export pre-hooks" in export_error
        )

        checkpoint = optimizer.state_dict()
        state_before = optimizer.state
        parameter_state_before = optimizer.state[model.weight]
        load_hook = None
        if rank == 0:

            def fail_load_hook(_optimizer, _state_dict):
                raise RuntimeError("injected rank-local load hook failure")

            load_hook = optimizer.register_load_state_dict_pre_hook(fail_load_hook)
        load_error = None
        try:
            optimizer.load_state_dict(checkpoint)
        except RuntimeError as exc:
            load_error = str(exc)
        if load_hook is not None:
            load_hook.remove()
        checks["load_hook_synchronized_atomic"] = (
            load_error is not None
            and "checkpoint load pre-hooks" in load_error
            and optimizer.state is state_before
            and optimizer.state[model.weight] is parameter_state_before
        )

        state_before = optimizer.state
        parameter_state_before = optimizer.state[model.weight]
        if rank == 0:

            def fail_load_staging(_staged):
                raise RuntimeError("injected rank-local load staging failure")

            optimizer._stage_loaded_state_for_offload = fail_load_staging
        load_staging_error = None
        try:
            optimizer.load_state_dict(checkpoint)
        except RuntimeError as exc:
            load_staging_error = str(exc)
        if rank == 0:
            del optimizer._stage_loaded_state_for_offload
        checks["load_staging_synchronized_atomic"] = (
            load_staging_error is not None
            and "checkpoint load staging" in load_staging_error
            and optimizer.state is state_before
            and optimizer.state[model.weight] is parameter_state_before
        )

        local_parameter_before = model.weight.to_local().detach().clone()
        global_step_before = optimizer._gefen_global_step

        def closure():
            if rank == 0:
                raise RuntimeError("injected rank-local closure failure")
            return torch.ones((), device="cuda")

        closure_error = None
        try:
            optimizer.step(closure)
        except RuntimeError as exc:
            closure_error = str(exc)
        checks["closure_failure_synchronized_atomic"] = (
            closure_error is not None
            and "step closure" in closure_error
            and torch.equal(model.weight.to_local(), local_parameter_before)
            and optimizer._gefen_global_step == global_step_before
            and not optimizer.state_offload_poisoned
        )

        second_grad = torch.linspace(0.7, -0.6, 64, device="cuda").reshape(8, 8)
        model.weight.grad = distribute_tensor(second_grad, mesh, [Shard(0)])
        if rank == 0:

            def fail_copyback(_tensor):
                raise RuntimeError("injected rank-local post-update copyback failure")

            optimizer._copy_state_tensor_to_offload_cpu = fail_copyback
        outcome_error = None
        try:
            optimizer.step()
        except RuntimeError as exc:
            outcome_error = str(exc)
        if rank == 0:
            del optimizer._copy_state_tensor_to_offload_cpu
        checks["post_update_failure_synchronized_poisoned"] = (
            outcome_error is not None
            and "step outcome" in outcome_error
            and optimizer.state_offload_poisoned
        )

        gathered = [None] * world
        dist.all_gather_object(gathered, checks)
        if rank == 0:
            result_queue.put(gathered)
    except BaseException:
        result_queue.put(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _fsdp2_unsupported_mesh_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from gefen import Gefen

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world)
        mesh = init_device_mesh(
            "cuda", (1, world), mesh_dim_names=("replicate", "shard")
        )
        model = nn.Linear(8, 8, bias=False, device="cuda")
        fully_shard(model, mesh=mesh)
        optimizer = Gefen(model.named_parameters(), fused=False)
        state_before = optimizer.state

        error = None
        try:
            optimizer.offload_state_()
        except RuntimeError as exc:
            error = str(exc)
        checks = {
            "specific_error": error is not None
            and "one-dimensional default-world DTensor topology" in error,
            "atomic": optimizer.state is state_before
            and not optimizer.state_offload_active,
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, checks)
        if rank == 0:
            result_queue.put(gathered)
    except BaseException:
        result_queue.put(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_two_rank_worker(worker, *worker_args):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=worker,
            args=(rank, 2, port, *worker_args, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        result = result_queue.get(timeout=240)
    except queue.Empty:
        result = None
    for process in processes:
        process.join(timeout=240)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
    assert result is not None, "FSDP2 offload workers timed out"
    if isinstance(result, str):
        pytest.fail("FSDP2 offload worker raised:\n" + result)
    assert all(process.exitcode == 0 for process in processes)
    return result


FSDP2_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available(),
    reason="FSDP2 offload requires two CUDA GPUs and NCCL",
)


@FSDP2_REQUIRED
@pytest.mark.parametrize("activate_before_first_step", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_fsdp2_offload_is_bit_exact_and_keeps_rank_local_cpu_authority(
    activate_before_first_step, fused
):
    result = _run_two_rank_worker(
        _fsdp2_offload_worker, activate_before_first_step, fused
    )
    assert all(all(rank_result.values()) for rank_result in result), result


@FSDP2_REQUIRED
def test_fsdp2_active_offload_dcp_roundtrip_is_exact_and_preserves_format():
    result = _run_two_rank_worker(_fsdp2_offload_dcp_worker)
    assert all(all(rank_result.values()) for rank_result in result), result


@FSDP2_REQUIRED
def test_fsdp2_offload_failures_are_synchronized_before_mutation_or_export():
    result = _run_two_rank_worker(_fsdp2_offload_failure_sync_worker)
    assert all(all(rank_result.values()) for rank_result in result), result


@FSDP2_REQUIRED
def test_fsdp2_offload_extended_failures_are_synchronized_and_fail_closed():
    result = _run_two_rank_worker(_fsdp2_offload_extended_failure_worker)
    assert all(all(rank_result.values()) for rank_result in result), result


@FSDP2_REQUIRED
def test_fsdp2_offload_rejects_unsupported_multidimensional_mesh_atomically():
    result = _run_two_rank_worker(_fsdp2_unsupported_mesh_worker)
    assert all(all(rank_result.values()) for rank_result in result), result
