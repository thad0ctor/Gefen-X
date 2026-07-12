"""Rank-consistent gradient-presence guards for sharded GefenMuon."""

import copy
from datetime import timedelta
import os
import queue
import socket
import traceback

import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon, GefenMuonHybrid


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _clone_state(state):
    return {
        key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state.items()
    }


def _state_equal(actual, expected):
    if set(actual) != set(expected):
        return False
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if torch.is_tensor(expected_value):
            if not torch.is_tensor(actual_value) or not torch.equal(
                actual_value, expected_value
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _mismatch_worker(rank, world, port, result_queue):
    import torch.distributed as dist
    from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        torch.cuda.set_device(rank)
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=12),
        )
        mesh = init_device_mesh("cuda", (world,))
        results = []

        for mode in ("exact", "distributed"):
            for phase in ("first_step", "initialized"):
                generator = torch.Generator(device="cpu").manual_seed(
                    1000 + len(results)
                )
                full_init = torch.randn(8, 8, generator=generator).to(rank)
                first_grad = (torch.randn(8, 8, generator=generator) * 0.01).to(
                    rank
                )
                mismatch_grad = (
                    torch.randn(8, 8, generator=generator) * 0.01
                ).to(rank)
                param = nn.Parameter(
                    distribute_tensor(full_init.clone(), mesh, [Shard(0)])
                )
                optimizer = GefenMuon(
                    [("weight", param)],
                    lr=1e-3,
                    fused=False,
                    sharded_mode=mode,
                )

                first_grad_dt = distribute_tensor(first_grad, mesh, [Shard(0)])
                mismatch_grad_dt = distribute_tensor(
                    mismatch_grad, mesh, [Shard(0)]
                )
                if phase == "initialized":
                    param.grad = first_grad_dt
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Build the DTensor on every rank so the test itself contributes
                # no unmatched collective; only assignment differs by rank.
                param.grad = None if rank == 0 else mismatch_grad_dt
                param_before = param.detach().to_local().clone()
                state_before = _clone_state(optimizer.state[param])
                codebook_before = (
                    None
                    if optimizer._gefen_codebook is None
                    else optimizer._gefen_codebook.detach().clone()
                )
                global_step_before = optimizer._gefen_global_step

                message = None
                try:
                    optimizer.step()
                except RuntimeError as exc:
                    message = str(exc)

                codebook_after = optimizer._gefen_codebook
                codebook_unchanged = (
                    codebook_before is None and codebook_after is None
                ) or (
                    codebook_before is not None
                    and codebook_after is not None
                    and torch.equal(codebook_before, codebook_after)
                )
                results.append(
                    {
                        "mode": mode,
                        "phase": phase,
                        "message": message,
                        "clear_error": message is not None
                        and "identical gradient presence" in message
                        and "weight (1/2 mesh ranks have gradients)" in message,
                        "param_unchanged": torch.equal(
                            param.detach().to_local(), param_before
                        ),
                        "state_unchanged": _state_equal(
                            optimizer.state[param], state_before
                        ),
                        "codebook_unchanged": codebook_unchanged,
                        "global_step_unchanged": optimizer._gefen_global_step
                        == global_step_before,
                    }
                )
                param.grad = None
                dist.barrier(device_ids=[rank])

        result_queue.put(("result", rank, results))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _distributed_amp_mismatch_worker(rank, world, port, result_queue):
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
            timeout=timedelta(seconds=12),
        )
        mesh = init_device_mesh("cpu", (world,))
        equivalent_mesh = init_device_mesh("cpu", (world,))
        results = []

        for kind in (
            "gefen",
            "gefen_distinct_mesh",
            "muon_exact",
            "muon_distributed",
            "muon_approx",
            "hybrid",
            "hybrid_mixed",
        ):
            generator = torch.Generator(device="cpu").manual_seed(
                4000 + len(results)
            )
            full_weight = torch.randn(8, 8, generator=generator).to(torch.float16)
            full_weight_grad = (
                torch.randn(8, 8, generator=generator) * 0.01
            ).to(torch.float16)
            weight = nn.Parameter(
                distribute_tensor(full_weight.clone(), mesh, [Shard(0)])
            )
            weight_grad = distribute_tensor(
                (full_weight_grad * 8.0).clone(), mesh, [Shard(0)]
            )
            params = [weight]

            if kind == "gefen":
                optimizer = Gefen(
                    [("weight", weight)],
                    lr=1e-3,
                    fused=False,
                    factored_v_2d=False,
                )
            elif kind == "gefen_distinct_mesh":
                full_other = torch.randn(8, generator=generator)
                full_other_grad = torch.randn(8, generator=generator) * 0.01
                other = nn.Parameter(
                    distribute_tensor(
                        full_other.clone(), equivalent_mesh, [Shard(0)]
                    )
                )
                other_grad = distribute_tensor(
                    (full_other_grad * 8.0).clone(), equivalent_mesh, [Shard(0)]
                )
                named_params = [("weight", weight), ("other", other)]
                if rank == 1:
                    named_params.reverse()
                optimizer = Gefen(
                    named_params,
                    lr=1e-3,
                    fused=False,
                    factored_v_2d=False,
                )
                weight.grad = weight_grad
                other.grad = None if rank == 0 else other_grad
                params.append(other)
            elif kind.startswith("muon_"):
                optimizer = GefenMuon(
                    [("weight", weight)],
                    lr=1e-3,
                    fused=False,
                    sharded_mode=kind.removeprefix("muon_"),
                )
            elif kind == "hybrid":
                full_bias = torch.randn(8, generator=generator)
                full_bias_grad = torch.randn(8, generator=generator) * 0.01
                bias = nn.Parameter(
                    distribute_tensor(full_bias.clone(), mesh, [Shard(0)])
                )
                bias_grad = distribute_tensor(
                    (full_bias_grad * 8.0).clone(), mesh, [Shard(0)]
                )
                optimizer = GefenMuonHybrid(
                    [("weight", weight)],
                    [("bias", bias)],
                    lr=1e-3,
                    fused=False,
                    sharded_mode="exact",
                    backup_optimizer="adamw",
                    normuon=False,
                )
                bias.grad = bias_grad
                params.append(bias)
            else:
                # Native AMP is selected by this ordinary local FP16 matrix,
                # while the only DTensor is a BF16 backup parameter. GradScaler
                # still scans that DTensor, so its rank-divergent presence must
                # be caught by the same union-wide preflight.
                weight = nn.Parameter(full_weight.clone())
                full_bias = torch.randn(8, generator=generator).to(torch.bfloat16)
                full_bias_grad = (
                    torch.randn(8, generator=generator) * 0.01
                ).to(torch.bfloat16)
                bias = nn.Parameter(
                    distribute_tensor(full_bias.clone(), mesh, [Shard(0)])
                )
                bias_grad = distribute_tensor(
                    (full_bias_grad * 8.0).clone(), mesh, [Shard(0)]
                )
                optimizer = GefenMuonHybrid(
                    [("weight", weight)],
                    [("bias", bias)],
                    lr=1e-3,
                    fused=False,
                    sharded_mode="exact",
                    backup_optimizer="adamw",
                    normuon=False,
                )
                weight.grad = (full_weight_grad * 8.0).clone()
                bias.grad = None if rank == 0 else bias_grad
                params = [weight, bias]

            # Construct the same DTensor gradient on both ranks so the test
            # itself is collective-safe, then deliberately assign it on only
            # one rank. Hybrid also keeps a shared FP32 backup gradient active,
            # proving the inactive FP16 rank cannot select the ordinary path.
            if kind not in ("gefen_distinct_mesh", "hybrid_mixed"):
                weight.grad = None if rank == 0 else weight_grad
            scaler = torch.amp.GradScaler("cpu", init_scale=8.0)
            scaler.scale(torch.ones(()))

            def local_clone(tensor):
                tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
                if hasattr(tensor, "wait"):
                    tensor = tensor.wait()
                return tensor.detach().clone()

            param_before = [local_clone(param) for param in params]
            grad_before = [
                None if param.grad is None else local_clone(param.grad)
                for param in params
            ]
            children = getattr(optimizer, "_subopts", [optimizer])
            state_before = {
                id(param): _clone_state(child.state.get(param, {}))
                for child in children
                for group in child.param_groups
                for param in group["params"]
            }
            codebook_before = {
                id(child): (
                    None
                    if getattr(child, "_gefen_codebook", None) is None
                    else child._gefen_codebook.detach().clone()
                )
                for child in children
            }
            step_before = {
                id(child): getattr(child, "_gefen_global_step", None)
                for child in children
            }
            scale_before = scaler.get_scale()

            message = None
            try:
                scaler.step(optimizer)
            except RuntimeError as exc:
                message = str(exc)

            params_unchanged = all(
                torch.equal(local_clone(param), expected)
                for param, expected in zip(params, param_before)
            )
            grads_unchanged = all(
                (param.grad is None and expected is None)
                or (
                    param.grad is not None
                    and expected is not None
                    and torch.equal(local_clone(param.grad), expected)
                )
                for param, expected in zip(params, grad_before)
            )
            states_unchanged = all(
                _state_equal(child.state.get(param, {}), state_before[id(param)])
                for child in children
                for group in child.param_groups
                for param in group["params"]
            )
            codebooks_unchanged = all(
                (
                    codebook_before[id(child)] is None
                    and getattr(child, "_gefen_codebook", None) is None
                )
                or (
                    codebook_before[id(child)] is not None
                    and getattr(child, "_gefen_codebook", None) is not None
                    and torch.equal(
                        child._gefen_codebook, codebook_before[id(child)]
                    )
                )
                for child in children
            )
            steps_unchanged = all(
                getattr(child, "_gefen_global_step", None) == step_before[id(child)]
                for child in children
            )
            results.append(
                {
                    "kind": kind,
                    "message": message,
                    "clear_error": message is not None
                    and "GradScaler integration requires identical DTensor gradient presence"
                    in message
                    and "{} (1/2 mesh ranks have gradients)".format(
                        (
                            "bias"
                            if kind == "hybrid_mixed"
                            else "other"
                            if kind == "gefen_distinct_mesh"
                            else "weight"
                        )
                    )
                    in message,
                    "params_unchanged": params_unchanged,
                    "grads_unchanged": grads_unchanged,
                    "states_unchanged": states_unchanged,
                    "codebooks_unchanged": codebooks_unchanged,
                    "steps_unchanged": steps_unchanged,
                    "scale_unchanged": scaler.get_scale() == scale_before,
                }
            )
            for param in params:
                param.grad = None
            dist.barrier()

        result_queue.put(("result", rank, results))
    except Exception:
        result_queue.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_nonsharded_step_never_collects(monkeypatch):
    param = nn.Parameter(torch.randn(4, 4))
    optimizer = GefenMuon([("weight", param)], fused=False)
    monkeypatch.setattr(optimizer, "_dist_available", lambda: True)

    def unexpected_collective(*args, **kwargs):
        raise AssertionError("plain parameters entered the DTensor preflight")

    monkeypatch.setattr(torch.distributed, "all_reduce", unexpected_collective)
    param.grad = torch.randn_like(param)
    optimizer.step()


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="DTensor GradScaler protocol regression needs Gloo",
)
def test_dtensor_fp16_grad_presence_mismatch_fails_before_grad_scaler_scan():
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_distributed_amp_mismatch_worker,
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
    errors = [payload for kind, _, payload in messages if kind == "error"]
    assert not errors, "\n".join(errors)
    rank_results = {
        rank: payload for kind, rank, payload in messages if kind == "result"
    }
    assert set(rank_results) == {0, 1}, messages

    expected_kinds = {
        "gefen",
        "gefen_distinct_mesh",
        "muon_exact",
        "muon_distributed",
        "muon_approx",
        "hybrid",
        "hybrid_mixed",
    }
    for rank, results in rank_results.items():
        assert {item["kind"] for item in results} == expected_kinds
        for item in results:
            assert item["clear_error"], (rank, item)
            assert item["params_unchanged"], (rank, item)
            assert item["grads_unchanged"], (rank, item)
            assert item["states_unchanged"], (rank, item)
            assert item["codebooks_unchanged"], (rank, item)
            assert item["steps_unchanged"], (rank, item)
            assert item["scale_unchanged"], (rank, item)

    for case_index in range(len(rank_results[0])):
        assert (
            rank_results[0][case_index]["message"]
            == rank_results[1][case_index]["message"]
        )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available(),
    reason="gradient-presence mismatch regression needs two CUDA GPUs and NCCL",
)
def test_sharded_grad_presence_mismatch_fails_on_every_rank():
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_mismatch_worker,
            args=(rank, world, port, result_queue),
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()

    messages = []
    try:
        for _ in range(world):
            messages.append(result_queue.get(timeout=30))
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
    errors = [payload for kind, _, payload in messages if kind == "error"]
    assert not errors, "\n".join(errors)
    rank_results = {
        rank: payload for kind, rank, payload in messages if kind == "result"
    }
    assert set(rank_results) == {0, 1}, messages

    expected_cases = {
        (mode, phase)
        for mode in ("exact", "distributed")
        for phase in ("first_step", "initialized")
    }
    for rank, results in rank_results.items():
        assert {(item["mode"], item["phase"]) for item in results} == expected_cases
        for item in results:
            assert item["clear_error"], (rank, item)
            assert item["param_unchanged"], (rank, item)
            assert item["state_unchanged"], (rank, item)
            assert item["codebook_unchanged"], (rank, item)
            assert item["global_step_unchanged"], (rank, item)

    # The all-reduced counts and stable names make the diagnostic identical on
    # every mesh rank, which is what lets the job fail synchronously.
    for case_index in range(len(rank_results[0])):
        assert (
            rank_results[0][case_index]["message"]
            == rank_results[1][case_index]["message"]
        )
