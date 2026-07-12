"""Native and ordinary GradScaler protocols for Gefen optimizers."""

import copy
from datetime import timedelta
import os
import queue
import socket
import subprocess
import sys
import traceback

import pytest
import torch
import torch.nn as nn

from gefen import Gefen, GefenMuon, GefenMuonHybrid


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GradScaler integration requires CUDA"
)


class _Fp16ManualGradScaler(torch.amp.GradScaler):
    """Test-only equivalent of ShardedGradScaler's FP16-capable unscale."""

    def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
        return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)


def _initial_values(dtype):
    generator = torch.Generator(device="cpu").manual_seed(20260712)
    matrix = (torch.randn(32, 32, generator=generator) * 0.02).to(
        device="cuda", dtype=dtype
    )
    bias = (torch.randn(32, generator=generator) * 0.02).to(
        device="cuda", dtype=dtype
    )
    return matrix, bias


def _grad_values(dtype):
    generator = torch.Generator(device="cpu").manual_seed(20260713)
    matrix = (torch.randn(32, 32, generator=generator) * 0.002).to(
        device="cuda", dtype=dtype
    )
    bias = (torch.randn(32, generator=generator) * 0.002).to(
        device="cuda", dtype=dtype
    )
    return matrix, bias


def _make_optimizer(kind, dtype, fused, *, backup_optimizer="gefen"):
    matrix_init, bias_init = _initial_values(dtype)
    matrix = nn.Parameter(matrix_init.clone())
    if kind == "gefen":
        optimizer = Gefen(
            [("layer.weight", matrix)],
            lr=2e-3,
            fused=fused,
            factored_v_2d=False,
        )
        return optimizer, [matrix]
    if kind == "muon":
        optimizer = GefenMuon(
            [("layer.weight", matrix)],
            lr=2e-3,
            fused=fused,
            ns_steps=2,
            ns_schedule="standard",
            adjust_lr_fn="original",
        )
        return optimizer, [matrix]
    if kind == "hybrid":
        bias = nn.Parameter(bias_init.clone())
        optimizer = GefenMuonHybrid(
            [("layer.weight", matrix)],
            [("layer.bias", bias)],
            lr=2e-3,
            fused=fused,
            ns_steps=2,
            ns_schedule="standard",
            normuon=False,
            backup_optimizer=backup_optimizer,
        )
        return optimizer, [matrix, bias]
    raise AssertionError("unknown optimizer kind: {}".format(kind))


def _new_scaler(scale=128.0):
    scaler = torch.amp.GradScaler("cuda", init_scale=scale)
    # GradScaler initializes its device scale lazily on scale(loss).
    scaler.scale(torch.ones((), device="cuda"))
    return scaler


def _new_fp16_manual_scaler(scale=128.0):
    scaler = _Fp16ManualGradScaler("cuda", init_scale=scale)
    scaler.scale(torch.ones((), device="cuda"))
    return scaler


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _assign_grads(params, grads):
    for param, grad in zip(params, grads):
        param.grad = grad.clone()


def _assign_scaled_grads(params, grads, scale):
    for param, grad in zip(params, grads):
        param.grad = (grad * scale).to(dtype=param.dtype)


def _clone_nested(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    return copy.deepcopy(value)


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(expected):
        assert torch.is_tensor(actual)
        assert actual.dtype == expected.dtype
        assert actual.device == expected.device
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def _snapshot(optimizer, params):
    return {
        "params": [param.detach().clone() for param in params],
        "state_dict": _clone_nested(optimizer.state_dict()),
    }


def _assert_snapshot_equal(optimizer, params, snapshot):
    for param, expected in zip(params, snapshot["params"]):
        assert torch.equal(param.detach(), expected)
    _assert_nested_equal(optimizer.state_dict(), snapshot["state_dict"])


def _clone_local_state(state):
    cloned = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            if hasattr(value, "to_local"):
                value = value.to_local()
            cloned[key] = value.detach().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


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


def _case_grads(kind, dtype):
    matrix_grad, bias_grad = _grad_values(dtype)
    return [matrix_grad, bias_grad] if kind == "hybrid" else [matrix_grad]


def _dtensor_amp_overflow_worker(rank, world, port, result_queue):
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
            timeout=timedelta(seconds=15),
        )
        mesh = init_device_mesh("cuda", (world,))
        results = []

        for mode in ("exact", "distributed"):
            for phase in ("first_step", "initialized"):
                generator = torch.Generator(device="cpu").manual_seed(
                    5000 + len(results)
                )
                full_init = (torch.randn(16, 16, generator=generator) * 0.02).to(
                    rank, torch.float16
                )
                full_grad = (
                    torch.randn(16, 16, generator=generator) * 0.002
                ).to(rank, torch.float16)
                param = nn.Parameter(
                    distribute_tensor(full_init.clone(), mesh, [Shard(0)])
                )
                optimizer = GefenMuon(
                    [("weight", param)],
                    lr=2e-3,
                    fused=False,
                    ns_steps=2,
                    ns_schedule="standard",
                    sharded_mode=mode,
                )
                scaler = _new_scaler(scale=8.0)

                if phase == "initialized":
                    finite_grad = distribute_tensor(
                        (full_grad * scaler.get_scale()).clone(), mesh, [Shard(0)]
                    )
                    param.grad = finite_grad
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                param_before = param.detach().to_local().clone()
                state_before = _clone_local_state(optimizer.state[param])
                codebook_before = (
                    None
                    if optimizer._gefen_codebook is None
                    else optimizer._gefen_codebook.detach().clone()
                )
                global_step_before = optimizer._gefen_global_step
                scale_before = scaler.get_scale()

                overflow_grad = distribute_tensor(
                    (full_grad * scale_before).clone(), mesh, [Shard(0)]
                )
                if rank == 0:
                    overflow_grad.to_local().view(-1)[0] = float("inf")
                param.grad = overflow_grad
                native_protocol = optimizer._step_supports_amp_scaling
                scaler.step(optimizer)
                scaler.update()

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
                        "native_protocol": native_protocol,
                        "param_unchanged": torch.equal(
                            param.detach().to_local(), param_before
                        ),
                        "state_unchanged": _local_state_equal(
                            optimizer.state[param], state_before
                        ),
                        "codebook_unchanged": codebook_unchanged,
                        "global_step_unchanged": optimizer._gefen_global_step
                        == global_step_before,
                        "scale_before": scale_before,
                        "scale_after": scaler.get_scale(),
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


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("kind", ["gefen", "muon", "hybrid"])
def test_finite_true_fp16_scaler_step_matches_explicit_unscaled(kind, fused):
    reference, reference_params = _make_optimizer(kind, torch.float16, fused)
    scaled, scaled_params = _make_optimizer(kind, torch.float16, fused)
    grads = _case_grads(kind, torch.float16)

    _assign_grads(reference_params, grads)
    reference.step()

    scaler = _new_scaler()
    _assign_scaled_grads(scaled_params, grads, scaler.get_scale())
    assert scaled._step_supports_amp_scaling
    scaler.step(scaled)
    scaler.update()

    for scaled_grad, unscaled_grad in zip(
        (param.grad for param in scaled_params), grads
    ):
        assert torch.equal(scaled_grad, unscaled_grad)
    for actual, expected in zip(scaled_params, reference_params):
        assert torch.equal(actual.detach(), expected.detach())
    _assert_nested_equal(scaled.state_dict(), reference.state_dict())


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("kind", ["gefen", "muon", "hybrid"])
@pytest.mark.parametrize("initialized", [False, True], ids=["first_step", "initialized"])
def test_true_fp16_overflow_is_a_complete_noop(kind, fused, initialized):
    optimizer, params = _make_optimizer(kind, torch.float16, fused)
    scaler = _new_scaler()
    grads = _case_grads(kind, torch.float16)

    if initialized:
        _assign_scaled_grads(params, grads, scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    snapshot = _snapshot(optimizer, params)
    _assign_scaled_grads(params, grads, scaler.get_scale())
    params[0].grad.view(-1)[0] = float("inf")
    scale_before = scaler.get_scale()
    result = scaler.step(optimizer)
    scaler.update()

    assert result is None
    assert scaler.get_scale() < scale_before
    _assert_snapshot_equal(optimizer, params, snapshot)


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("kind", ["gefen", "muon", "hybrid"])
def test_manual_unscale_is_not_applied_twice(kind, fused):
    # Explicit GradScaler.unscale_ supports FP32 master gradients and is the
    # path used by gradient clipping in Accelerate/Trainer. True FP16 tensors
    # require native automatic unscale because base GradScaler rejects them.
    reference, reference_params = _make_optimizer(kind, torch.float32, fused)
    scaled, scaled_params = _make_optimizer(kind, torch.float32, fused)
    grads = _case_grads(kind, torch.float32)
    clipped_grads = [grad * 0.25 for grad in grads]

    _assign_grads(reference_params, clipped_grads)
    reference.step()

    scaler = _new_scaler()
    assert not scaled._step_supports_amp_scaling
    _assign_scaled_grads(scaled_params, grads, scaler.get_scale())
    scaler.unscale_(scaled)
    for param, grad in zip(scaled_params, grads):
        assert torch.equal(param.grad, grad)
        param.grad.mul_(0.25)
    scaler.step(scaled)
    scaler.update()

    for actual, expected in zip(scaled_params, reference_params):
        assert torch.equal(actual.detach(), expected.detach())
    _assert_nested_equal(scaled.state_dict(), reference.state_dict())


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("kind", ["gefen", "muon", "hybrid"])
def test_native_fp16_manual_unscale_stage_is_not_divided_twice(kind, fused):
    reference, reference_params = _make_optimizer(kind, torch.float16, fused)
    scaled, scaled_params = _make_optimizer(kind, torch.float16, fused)
    grads = _case_grads(kind, torch.float16)
    clipped_grads = [(grad * 0.5).to(torch.float16) for grad in grads]

    _assign_grads(reference_params, clipped_grads)
    reference.step()

    scaler = _new_fp16_manual_scaler()
    _assign_scaled_grads(scaled_params, grads, scaler.get_scale())
    scaler.unscale_(scaled)
    assert scaled._step_supports_amp_scaling
    for param, grad in zip(scaled_params, grads):
        assert torch.equal(param.grad, grad)
        param.grad.mul_(0.5)
    scaler.step(scaled)
    scaler.update()

    for actual, expected in zip(scaled_params, reference_params):
        assert torch.equal(actual.detach(), expected.detach())
    _assert_nested_equal(scaled.state_dict(), reference.state_dict())


@pytest.mark.parametrize("fused", [False, True], ids=["unfused", "fused"])
@pytest.mark.parametrize("initialized", [False, True], ids=["first_step", "initialized"])
@pytest.mark.parametrize("overflow_half", ["muon", "backup"])
def test_hybrid_overflow_in_one_half_skips_both_halves(
    fused, initialized, overflow_half
):
    optimizer, params = _make_optimizer(
        "hybrid", torch.float16, fused, backup_optimizer="adamw"
    )
    scaler = _new_scaler()
    grads = _case_grads("hybrid", torch.float16)

    if initialized:
        _assign_scaled_grads(params, grads, scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    snapshot = _snapshot(optimizer, params)
    _assign_scaled_grads(params, grads, scaler.get_scale())
    overflow_index = 0 if overflow_half == "muon" else 1
    params[overflow_index].grad.view(-1)[0] = float("inf")
    scaler.step(optimizer)
    scaler.update()

    _assert_snapshot_equal(optimizer, params, snapshot)


@pytest.mark.parametrize("kind", ["gefen", "muon", "hybrid"])
def test_bf16_stays_on_the_ordinary_optimizer_path(kind):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support BF16")
    optimizer, params = _make_optimizer(kind, torch.bfloat16, fused=False)
    grads = _case_grads(kind, torch.bfloat16)
    assert not optimizer._step_supports_amp_scaling
    before = [param.detach().clone() for param in params]
    _assign_grads(params, grads)
    optimizer.step()
    assert any(
        not torch.equal(param.detach(), initial)
        for param, initial in zip(params, before)
    )


def test_accelerate_fp32_master_overflow_reports_step_skipped():
    pytest.importorskip("accelerate")
    # Isolate Accelerator's process-global state so this regression composes
    # with Trainer/Accelerate tests that choose a different mixed-precision mode.
    script = r"""
import torch
from accelerate import Accelerator
from gefen import Gefen

accelerator = Accelerator(mixed_precision="fp16")
param = torch.nn.Parameter(torch.randn(16, 16, device="cuda"))
optimizer = Gefen([("weight", param)], fused=False)
wrapped = accelerator.prepare_optimizer(optimizer)
wrapped.scaler.scale(torch.ones((), device="cuda"))
param.grad = torch.full_like(param, float("inf"))
before = param.detach().clone()
wrapped.step()
assert wrapped.step_was_skipped
assert torch.equal(param.detach(), before)
assert optimizer._gefen_global_step == 0
assert optimizer._gefen_codebook is None
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_accelerate_bf16_keeps_the_unscaled_ordinary_step():
    pytest.importorskip("accelerate")
    script = r"""
import torch
from accelerate import Accelerator
from gefen import Gefen

accelerator = Accelerator(mixed_precision="bf16")
param = torch.nn.Parameter(torch.randn(16, 16, device="cuda"))
optimizer = Gefen([("weight", param)], fused=False)
wrapped = accelerator.prepare_optimizer(optimizer)
assert wrapped.scaler is None
inputs = torch.randn(8, 16, device="cuda")
with accelerator.autocast():
    loss = (inputs @ param).square().mean()
accelerator.backward(loss)
before = param.detach().clone()
wrapped.step()
assert not wrapped.step_was_skipped
assert not torch.equal(param.detach(), before)
assert optimizer._gefen_global_step == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_fsdp1_flat_fp16_disables_native_protocol_and_fails_before_step():
    optimizer, params = _make_optimizer("muon", torch.float16, fused=False)
    param = params[0]
    # FSDP1 exposes FlatParameter as a regular local tensor; unlike DTensor it
    # has no dispatcher that globally reduces scaler-owned found_inf flags.
    param._is_flat_param = True
    grad = _case_grads("muon", torch.float16)[0]
    scaler = _new_scaler()
    _assign_scaled_grads(params, [grad], scaler.get_scale())
    snapshot = _snapshot(optimizer, params)

    with pytest.warns(RuntimeWarning, match="ShardedGradScaler"):
        assert not optimizer._step_supports_amp_scaling
    with pytest.raises(ValueError, match="Attempting to unscale FP16 gradients"):
        scaler.step(optimizer)
    _assert_snapshot_equal(optimizer, params, snapshot)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available(),
    reason="DTensor AMP overflow regression needs two CUDA GPUs and NCCL",
)
def test_dtensor_rank_local_fp16_overflow_skips_every_rank():
    import torch.multiprocessing as mp

    world = 2
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    port = _free_port()
    processes = [
        context.Process(
            target=_dtensor_amp_overflow_worker,
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

    expected_cases = {
        (mode, phase)
        for mode in ("exact", "distributed")
        for phase in ("first_step", "initialized")
    }
    for rank, results in rank_results.items():
        assert {(item["mode"], item["phase"]) for item in results} == expected_cases
        for item in results:
            assert item["native_protocol"], (rank, item)
            assert item["param_unchanged"], (rank, item)
            assert item["state_unchanged"], (rank, item)
            assert item["codebook_unchanged"], (rank, item)
            assert item["global_step_unchanged"], (rank, item)
            assert item["scale_after"] == item["scale_before"] / 2, (rank, item)

    # DTensor's scaler-owned MAX reduction must keep every rank's scale in lockstep.
    for case_index in range(len(rank_results[0])):
        assert (
            rank_results[0][case_index]["scale_after"]
            == rank_results[1][case_index]["scale_after"]
        )
