"""Replica-exact routing and checkpoint coverage for ``deterministic=True``."""

import copy
import hashlib
import multiprocessing as mp
import traceback
from unittest import mock

import pytest
import torch
import torch.nn as nn

import gefen.gefen as gefen_mod
from gefen import Gefen, GefenMuon, GefenMuonHybrid


def _deep_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(
            _deep_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(
            _deep_equal(left, right) for left, right in zip(actual, expected)
        )
    if torch.is_tensor(actual):
        return actual.dtype == expected.dtype and torch.equal(actual, expected)
    return actual == expected


def _stepped_cpu_optimizer(*, deterministic):
    generator = torch.Generator().manual_seed(11)
    param = nn.Parameter(torch.randn(16, 12, generator=generator))
    optimizer = Gefen(
        [("weight", param)],
        lr=1e-3,
        fused=False,
        deterministic=deterministic,
    )
    param.grad = torch.randn(param.shape, generator=generator)
    optimizer.step()
    return param, optimizer


def test_deterministic_configuration_validation():
    param = nn.Parameter(torch.ones(2, 2))
    with pytest.raises(TypeError, match="deterministic must be a bool"):
        Gefen([param], fused=False, deterministic=1)
    with pytest.raises(ValueError, match="cannot honor stochastic_round"):
        Gefen(
            [param],
            fused=False,
            deterministic=True,
            factored_v_2d=True,
            stochastic_round=True,
        )

    # Muon's momentum-only fused kernel is replica-exact and supports its
    # stateless, step-seeded stochastic rounding under the same policy.
    muon = GefenMuon(
        [("weight", nn.Parameter(torch.ones(2, 2)))],
        fused=False,
        deterministic=True,
        stochastic_round=True,
    )
    assert muon._deterministic is True
    assert muon._factored_v_2d is False


def test_deterministic_checkpoint_tags_and_matching_roundtrip():
    source_param, source = _stepped_cpu_optimizer(deterministic=True)
    checkpoint = copy.deepcopy(source.state_dict())
    assert checkpoint["gefen_deterministic"] is True
    assert all(
        group["_gefen_checkpoint_metadata"]["deterministic"] is True
        for group in checkpoint["param_groups"]
    )

    target_param = nn.Parameter(source_param.detach().clone())
    target = Gefen(
        [("weight", target_param)],
        lr=1e-3,
        fused=False,
        deterministic=True,
    )
    target.load_state_dict(checkpoint)
    target_param.grad = torch.full_like(target_param, 0.125)
    source_param.grad = torch.full_like(source_param, 0.125)
    target.step()
    source.step()
    assert torch.equal(target_param, source_param)


def test_deterministic_checkpoint_mismatch_rejected_before_mutation():
    _, source = _stepped_cpu_optimizer(deterministic=True)
    checkpoint = copy.deepcopy(source.state_dict())

    target_param = nn.Parameter(torch.zeros(16, 12))
    target = Gefen(
        [("weight", target_param)],
        lr=1e-3,
        fused=False,
        deterministic=False,
    )
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="intentional state migration"):
        target.load_state_dict(checkpoint)
    assert _deep_equal(target.state_dict(), before)

    # DCP keeps conventional top-level keys only. The mirrored group tag must
    # preserve the same safety check when the top-level Gefen extras are gone.
    dcp_style = {
        "state": checkpoint["state"],
        "param_groups": checkpoint["param_groups"],
    }
    with pytest.raises(ValueError, match="intentional state migration"):
        target.load_state_dict(dcp_style)
    assert _deep_equal(target.state_dict(), before)


@pytest.mark.parametrize("invalid_tag", [None, 0, 1, "true"])
def test_deterministic_checkpoint_top_level_tag_requires_actual_bool(invalid_tag):
    _, source = _stepped_cpu_optimizer(deterministic=True)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["gefen_deterministic"] = invalid_tag

    target = Gefen(
        [("weight", nn.Parameter(torch.zeros(16, 12)))],
        lr=1e-3,
        fused=False,
        deterministic=True,
    )
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="top-level deterministic policy must be a bool"):
        target.load_state_dict(checkpoint)
    assert _deep_equal(target.state_dict(), before)


def test_deterministic_checkpoint_group_tag_requires_actual_bool():
    _, source = _stepped_cpu_optimizer(deterministic=True)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["param_groups"][0]["_gefen_checkpoint_metadata"][
        "deterministic"
    ] = 1

    target = Gefen(
        [("weight", nn.Parameter(torch.zeros(16, 12)))],
        lr=1e-3,
        fused=False,
        deterministic=True,
    )
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(
        ValueError, match="parameter-group deterministic policy must be a bool"
    ):
        target.load_state_dict(checkpoint)
    assert _deep_equal(target.state_dict(), before)


def test_legacy_untagged_checkpoint_loads_under_live_policy():
    _, source = _stepped_cpu_optimizer(deterministic=False)
    legacy = copy.deepcopy(source.state_dict())
    legacy.pop("gefen_deterministic")
    for group in legacy["param_groups"]:
        group["_gefen_checkpoint_metadata"].pop("deterministic")

    target = Gefen(
        [("weight", nn.Parameter(torch.zeros(16, 12)))],
        lr=1e-3,
        fused=False,
        deterministic=True,
    )
    target.load_state_dict(legacy)
    assert target._deterministic is True


def test_hybrid_plumbs_one_deterministic_policy_to_gefen_children():
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
    optimizer = GefenMuonHybrid.from_model(
        model,
        lr=1e-3,
        backup_optimizer="gefen",
        fused=False,
        deterministic=True,
    )
    assert optimizer._deterministic is True
    assert optimizer.muon is not None and optimizer.muon._deterministic is True
    assert optimizer.backup is not None and optimizer.backup._deterministic is True
    # Keep the established nested hybrid schema; each Gefen child carries the
    # policy in its own ordinary state dict and mirrored group metadata.
    assert set(optimizer.state_dict()) == {"muon", "backup", "backup_optimizer"}


def _fixed_codebook(device):
    return torch.linspace(-1.0, 1.0, 256, device=device, dtype=torch.float32)


def _tensor_digest(tensor):
    tensor = tensor.detach().contiguous().cpu()
    payload = tensor.view(torch.uint8).numpy().tobytes()
    header = "{}:{}:".format(tuple(tensor.shape), tensor.dtype).encode()
    return hashlib.sha256(header + payload).hexdigest()


def _result_digest(param, optimizer):
    state = optimizer.state[param]
    result = {"param": _tensor_digest(param)}
    for key in sorted(state):
        value = state[key]
        if torch.is_tensor(value):
            result[key] = _tensor_digest(value)
        elif isinstance(value, (bool, int, float, str)):
            result[key] = repr(value)
    result["codebook"] = _tensor_digest(optimizer._gefen_codebook)
    return result


def _run_fused_deterministic_cases(device_index):
    device = torch.device("cuda", device_index)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    factored_init = torch.randn(384, 128, generator=generator) * 0.02
    factored_grad = torch.randn(384, 128, generator=generator) * 1e-3
    factored_param = nn.Parameter(factored_init.to(device))
    factored = Gefen(
        [("linear_qkv.weight", factored_param)],
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
        factored_v_2d=True,
        force_2d_period_one=True,
        deterministic=True,
    )
    factored._gefen_codebook = _fixed_codebook(device)
    factored_param.grad = factored_grad.to(device)
    factored.step()

    block_init = torch.randn(384, 128, generator=generator) * 0.02
    block_grad = torch.randn(384, 128, generator=generator) * 1e-3
    block_param = nn.Parameter(block_init.to(device))
    block = Gefen(
        [("projection.weight", block_param)],
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
        factored_v_2d=False,
        deterministic=True,
    )
    block._gefen_codebook = _fixed_codebook(device)
    # Two very large blocks are normally routed to atomic v2-full. Pre-seeding
    # the period also removes first-step period-search variability from this
    # update-kernel regression.
    block.state[block_param]["automatic_period"] = 24_576
    block_param.grad = block_grad.to(device)
    block.step()

    torch.cuda.synchronize(device)
    return {
        "factored": _result_digest(factored_param, factored),
        "block": _result_digest(block_param, block),
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_fused_repeated_runs_are_bit_exact():
    first = _run_fused_deterministic_cases(0)
    second = _run_fused_deterministic_cases(0)
    assert first == second


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_routes_fused_block_v1_and_factored_fallback():
    device = torch.device("cuda", 0)
    generator = torch.Generator(device="cpu").manual_seed(55)

    block_param = nn.Parameter(torch.randn(64, 32, generator=generator).to(device))
    block = Gefen(
        [("block", block_param)],
        fused=True,
        factored_v_2d=False,
        deterministic=True,
    )
    block._gefen_codebook = _fixed_codebook(device)
    block.state[block_param]["automatic_period"] = 1_024
    block_param.grad = torch.randn(64, 32, generator=generator).to(device)
    v1_update = gefen_mod._automatic_gefen_fused_full_update_cuda
    with mock.patch.object(
        gefen_mod,
        "_should_use_v2_full",
        side_effect=AssertionError("deterministic mode consulted v2 routing"),
    ), mock.patch.object(
        gefen_mod,
        "_automatic_gefen_fused_full_update_cuda",
        wraps=v1_update,
    ) as v1_spy:
        block.step()
    assert v1_spy.call_count == 1

    factored_param = nn.Parameter(torch.randn(64, 32, generator=generator).to(device))
    factored = Gefen(
        [("factored", factored_param)],
        fused=True,
        factored_v_2d=True,
        force_2d_period_one=True,
        deterministic=True,
    )
    factored._gefen_codebook = _fixed_codebook(device)
    factored_param.grad = torch.randn(64, 32, generator=generator).to(device)
    with mock.patch.object(
        gefen_mod,
        "_gefen_factored_update_cuda",
        side_effect=AssertionError("deterministic mode used atomic factored stats"),
    ):
        factored.step()
    assert "v_row" in factored.state[factored_param]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_deterministic_automatic_period_avoids_atomic_cuda_backend():
    device = torch.device("cuda", 0)
    generator = torch.Generator(device="cpu").manual_seed(88)
    param = nn.Parameter(torch.randn(96, 64, generator=generator).to(device))
    optimizer = Gefen(
        [("weight", param)],
        lr=1e-3,
        fused=True,
        deterministic=True,
    )
    param.grad = torch.randn(param.shape, generator=generator).to(device)

    period_search = gefen_mod.find_period_by_block_variance
    with mock.patch.object(
        gefen_mod, "FIND_PERIOD_BACKEND", "cuda_kernel"
    ), mock.patch.object(
        gefen_mod, "find_period_by_block_variance", wraps=period_search
    ) as period_spy:
        optimizer.step()

    assert period_spy.call_count >= 1
    assert all(call.kwargs["backend"] == "gpu" for call in period_spy.call_args_list)


def _run_fused_automatic_first_step(device_index):
    device = torch.device("cuda", device_index)
    generator = torch.Generator(device="cpu").manual_seed(2027)
    initial = torch.randn(192, 128, generator=generator) * 0.02
    grad = torch.randn(192, 128, generator=generator) * 1e-3
    param = nn.Parameter(initial.to(device))
    optimizer = Gefen(
        [("weight", param)],
        lr=1e-3,
        weight_decay=0.01,
        fused=True,
        deterministic=True,
    )
    param.grad = grad.to(device)
    optimizer.step()
    torch.cuda.synchronize(device)
    return _result_digest(param, optimizer)


def _replica_worker(rank, queue, case):
    try:
        torch.cuda.set_device(rank)
        if case == "preseeded":
            result = _run_fused_deterministic_cases(rank)
        elif case == "automatic_first_step":
            result = _run_fused_automatic_first_step(rank)
        else:
            raise ValueError("unknown deterministic replica case: {}".format(case))
        queue.put(("ok", rank, result))
    except Exception:
        queue.put(("error", rank, traceback.format_exc()))
        raise


def _run_two_process_replica_case(case):
    capabilities = [torch.cuda.get_device_capability(index) for index in range(2)]
    if capabilities[0] != capabilities[1]:
        pytest.skip("replica-exact test requires two GPUs with one compute capability")

    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(target=_replica_worker, args=(rank, queue, case))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    messages = [queue.get(timeout=120) for _ in processes]
    for process in processes:
        process.join(timeout=120)
        assert not process.is_alive()
        assert process.exitcode == 0

    errors = [message for message in messages if message[0] == "error"]
    assert not errors, errors
    return {rank: result for _, rank, result in messages}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_deterministic_fused_two_process_replicas_are_bit_exact():
    results = _run_two_process_replica_case("preseeded")
    assert results[0] == results[1]


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_deterministic_fused_first_step_replicas_are_bit_exact():
    results = _run_two_process_replica_case("automatic_first_step")
    assert results[0] == results[1]
