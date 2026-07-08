"""
2-rank CUDA-graph capture/replay validation for capturable FSDP2/DTensor steps.

This intentionally complements tests/test_muon_fsdp2_parity.py rather than
folding into it: the parity suite proves eager sharded math, while this file
proves that the same sharded optimizer step can be recorded into per-rank CUDA
graphs and replayed in lockstep with fresh static gradients and tensor lr values.

Run on two visible CUDA devices:

    CUDA_VISIBLE_DEVICES=1,2 python tests/test_capturable_fsdp2.py

By default the direct script and pytest cover plain Gefen plus all three Muon
FSDP2 sharded modes (approx, exact, distributed). To narrow a direct run:

    GEFEN_CAPTURABLE_FSDP2_CASES=muon-exact \
      CUDA_VISIBLE_DEVICES=1,2 python tests/test_capturable_fsdp2.py
"""
import os
import queue
import socket
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed  # noqa: F401
import torch.nn as nn

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None


WORLD = 2
WARMUP = 3
# Many replays: capture races and NCCL-graph-replay leaks specific to the sharded
# path stay hidden under a handful of replays but surface over a long replay loop
# (the same class of bug the 386M single-process bench caught at scale).
REPLAYS = 48
LR_START = 5e-3
LR_DECAY = 0.9
WD = 0.01

COUNTER_KEYS = ("step", "vmean_step", "factored_step", "normuon_step")

CASES = {
    "gefen-backup": {
        "kind": "gefen",
        "specs": [
            ("backup_w", (24, 16)),
            ("backup_empty_rank1", (1,)),
            ("backup_norm", (16,)),
        ],
        "dtype": torch.float32,
        "empty_rank1_name": "backup_empty_rank1",
        "rtol": 2e-2,
        "atol": 2e-3,
    },
    "muon-approx": {
        "kind": "muon",
        "sharded_mode": "approx",
        "specs": [
            ("muon_w0", (32, 48)),
            ("muon_empty_rank1", (1, 8)),
            ("muon_w2", (17, 24)),
        ],
        "dtype": torch.bfloat16,
        "empty_rank1_name": "muon_empty_rank1",
        "rtol": 5e-2,
        "atol": 5e-3,
    },
    "muon-exact": {
        "kind": "muon",
        "sharded_mode": "exact",
        "specs": [
            ("muon_w0", (32, 48)),
            ("muon_empty_rank1", (1, 8)),
            ("muon_w2", (17, 24)),
        ],
        "dtype": torch.bfloat16,
        "empty_rank1_name": "muon_empty_rank1",
        "rtol": 5e-2,
        "atol": 5e-3,
    },
    "muon-distributed": {
        "kind": "muon",
        "sharded_mode": "distributed",
        "specs": [
            ("muon_w0", (32, 48)),
            ("muon_empty_owner", (1, 8)),
            ("muon_w2", (24, 32)),
        ],
        "dtype": torch.bfloat16,
        "empty_rank1_name": "muon_empty_owner",
        "rtol": 5e-2,
        "atol": 5e-3,
    },
}

DEFAULT_CASES = ("gefen-backup", "muon-approx", "muon-exact", "muon-distributed")


def _get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


def _make_full_tensors(specs, *, seed, device, dtype, scale):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [
        (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(
            device=device, dtype=dtype
        )
        for _, shape in specs
    ]


def _grad_sequence(specs, *, steps, seed, device, dtype):
    return [
        _make_full_tensors(
            specs, seed=seed + step, device=device, dtype=dtype, scale=0.01
        )
        for step in range(steps)
    ]


def _local_shard(full, rank, world):
    size = full.size(0)
    chunk = (size + world - 1) // world
    start = chunk * rank
    length = max(0, min(size, start + chunk) - start)
    return full.narrow(0, min(start, size), length).contiguous()


def _check_expected_empty_rank1(case, full_inits, rank, world):
    expected_name = case.get("empty_rank1_name")
    if rank != 1 or expected_name is None:
        return True, ""
    for (name, _shape), full in zip(case["specs"], full_inits):
        if name == expected_name:
            local = _local_shard(full, rank, world)
            if local.numel() == 0:
                return True, ""
            return False, "{} rank1 shard is not empty: {}".format(
                expected_name, tuple(local.shape)
            )
    return False, "{} not found in case specs".format(expected_name)


def _fill_static_grads(static_grads, full_grads, rank, world):
    for static_grad, full_grad in zip(static_grads, full_grads):
        static_grad.to_local().copy_(_local_shard(full_grad, rank, world))


def _build_params(specs, full_inits, mesh):
    from torch.distributed.tensor import Shard, distribute_tensor

    params = []
    for (name, _shape), init in zip(specs, full_inits):
        params.append(
            (name, nn.Parameter(distribute_tensor(init.clone(), mesh, [Shard(0)])))
        )
    return params


def _attach_static_grads(params, full_grads0, mesh):
    from torch.distributed.tensor import Shard, distribute_tensor

    static_grads = []
    for (_, p), full_grad in zip(params, full_grads0):
        static_grad = distribute_tensor(torch.zeros_like(full_grad), mesh, [Shard(0)])
        p.grad = static_grad
        static_grads.append(static_grad)
    return static_grads


def _build_optimizer(case, params, lr):
    if case["kind"] == "gefen":
        from gefen.gefen import Gefen

        return Gefen(
            params,
            lr=lr,
            weight_decay=WD,
            fused=True,
            capturable=True,
        )

    from gefen.gefen_muon import GefenMuon

    return GefenMuon(
        params,
        lr=lr,
        weight_decay=WD,
        fused=True,
        sharded_mode=case["sharded_mode"],
        adjust_lr_fn="match_rms_adamw",
        normuon=True,
        capturable=True,
    )


def _assert_counters(opt, steps):
    for pstate in opt.state.values():
        for key in COUNTER_KEYS:
            counter = pstate.get(key)
            if counter is None:
                continue
            if not torch.is_tensor(counter) or counter.dim() != 0 or not counter.is_cuda:
                return False, "{} is not a 0-dim CUDA tensor: {!r}".format(key, counter)
            if counter.item() != steps:
                return False, "{} counter {} != {}".format(key, counter.item(), steps)
    return True, ""


def _gather_params(params):
    return [p.detach().full_tensor().detach().clone() for _, p in params]


def _run_eager(case, mesh, rank, world, full_inits, grads, lr_values):
    params = _build_params(case["specs"], full_inits, mesh)
    lr = torch.tensor(lr_values[0], device=torch.device("cuda", rank))
    opt = _build_optimizer(case, params, lr)
    static_grads = _attach_static_grads(params, grads[0], mesh)
    for step, full_grads in enumerate(grads):
        _fill_static_grads(static_grads, full_grads, rank, world)
        lr.fill_(lr_values[step])
        opt.step()
    torch.cuda.synchronize(rank)
    counter_ok, counter_msg = _assert_counters(opt, len(lr_values))
    return _gather_params(params), counter_ok, counter_msg


def _run_captured(case, mesh, rank, world, full_inits, grads, lr_values):
    import torch.distributed as dist

    params = _build_params(case["specs"], full_inits, mesh)
    lr = torch.tensor(lr_values[0], device=torch.device("cuda", rank))
    opt = _build_optimizer(case, params, lr)
    static_grads = _attach_static_grads(params, grads[0], mesh)

    side = torch.cuda.Stream(device=rank)
    side.wait_stream(torch.cuda.current_stream(rank))
    with torch.cuda.stream(side):
        for step in range(WARMUP):
            _fill_static_grads(static_grads, grads[step], rank, world)
            lr.fill_(lr_values[step])
            opt.step()
    torch.cuda.current_stream(rank).wait_stream(side)
    torch.cuda.synchronize(rank)

    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opt.step()
    torch.cuda.synchronize(rank)
    dist.barrier()

    for step in range(WARMUP, len(lr_values)):
        _fill_static_grads(static_grads, grads[step], rank, world)
        lr.fill_(lr_values[step])
        dist.barrier()
        graph.replay()
    torch.cuda.synchronize(rank)
    dist.barrier()

    counter_ok, counter_msg = _assert_counters(opt, len(lr_values))
    return _gather_params(params), counter_ok, counter_msg


def _worker(rank, world, case_name, port, q):
    import torch.distributed as dist
    from torch.distributed.tensor import init_device_mesh

    case = CASES[case_name]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, timeout=timedelta(seconds=180)
    )
    try:
        mesh = init_device_mesh("cuda", (world,))
        steps = WARMUP + REPLAYS
        lr_values = [LR_START * (LR_DECAY ** i) for i in range(steps)]
        frozen_replay_lr_values = (
            lr_values[:WARMUP] + [lr_values[WARMUP - 1]] * REPLAYS
        )
        full_inits = _make_full_tensors(
            case["specs"],
            seed=1234,
            device=device,
            dtype=case["dtype"],
            scale=0.1,
        )
        grads = _grad_sequence(
            case["specs"],
            steps=steps,
            seed=2000,
            device=device,
            dtype=case["dtype"],
        )
        empty_ok, empty_msg = _check_expected_empty_rank1(
            case, full_inits, rank, world
        )
        if not empty_ok:
            q.put(("error", rank, empty_msg))
            return

        eager, eager_counter_ok, eager_counter_msg = _run_eager(
            case, mesh, rank, world, full_inits, grads, lr_values
        )
        captured, captured_counter_ok, captured_counter_msg = _run_captured(
            case, mesh, rank, world, full_inits, grads, lr_values
        )
        frozen_replay_lr, _, _ = _run_captured(
            case, mesh, rank, world, full_inits, grads, frozen_replay_lr_values
        )

        if rank == 0:
            max_abs = max(
                (got.float() - want.float()).abs().max().item()
                for got, want in zip(captured, eager)
            )
            allclose = all(
                torch.allclose(
                    got.float(),
                    want.float(),
                    rtol=case["rtol"],
                    atol=case["atol"],
                )
                for got, want in zip(captured, eager)
            )
            replay_lr_effect = max(
                (sched.float() - frozen.float()).abs().max().item()
                for sched, frozen in zip(captured, frozen_replay_lr)
            )
            q.put(("result", case_name, max_abs, allclose, replay_lr_effect))
        else:
            q.put(("rank", case_name, rank))

        if not eager_counter_ok:
            q.put(("error", rank, "eager counter check failed: " + eager_counter_msg))
        if not captured_counter_ok:
            q.put(
                (
                    "error",
                    rank,
                    "captured counter check failed: " + captured_counter_msg,
                )
            )
    except Exception:
        q.put(("error", rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_case(case_name):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _get_free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, WORLD, case_name, port, q))
        for rank in range(WORLD)
    ]
    for proc in procs:
        proc.start()

    deadline = time.monotonic() + 240
    result = None
    ranks = set()
    errors = []
    while time.monotonic() < deadline:
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            item = None
        if item is not None:
            if item[0] == "result":
                result = item[1:]
            elif item[0] == "rank":
                ranks.add(item[2])
            elif item[0] == "error":
                errors.append(item[1:])
        if not any(proc.is_alive() for proc in procs):
            break
    for proc in procs:
        proc.join(timeout=0)
    for proc in procs:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=10)
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            break
        if item[0] == "result":
            result = item[1:]
        elif item[0] == "rank":
            ranks.add(item[2])
        elif item[0] == "error":
            errors.append(item[1:])

    for proc in procs:
        assert proc.exitcode == 0, "[{}] worker exited with {}".format(
            case_name, proc.exitcode
        )
    assert not errors, "[{}] worker errors:\n{}".format(case_name, errors)
    assert result is not None, "[{}] no rank0 result".format(case_name)
    assert 1 in ranks, "[{}] rank1 did not report completion".format(case_name)

    _case, max_abs, allclose, replay_lr_effect = result
    ok = allclose and replay_lr_effect > 1e-6
    print(
        "[{}] max|captured-eager|={:.3e} replay_lr_effect={:.3e} {}".format(
            case_name, max_abs, replay_lr_effect, "PASS" if ok else "FAIL"
        )
    )
    return ok


def _selected_cases():
    raw = os.environ.get("GEFEN_CAPTURABLE_FSDP2_CASES")
    if not raw:
        return list(DEFAULT_CASES)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = sorted(set(names) - set(CASES))
    if unknown:
        raise ValueError("unknown capturable FSDP2 case(s): {}".format(unknown))
    return names


def run():
    results = {}
    if (
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
        or torch.cuda.device_count() < WORLD
    ):
        print("capturable FSDP2 capture SKIP: needs >=2 CUDA GPUs and NCCL")
        return None
    for case_name in _selected_cases():
        results[case_name] = _run_case(case_name)
    return all(results.values())


if pytest is not None:

    @pytest.mark.skipif(
        not torch.cuda.is_available()
        or not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
        or torch.cuda.device_count() < WORLD,
        reason="capturable FSDP2 capture needs >=2 CUDA GPUs and NCCL",
    )
    @pytest.mark.parametrize("case_name", list(DEFAULT_CASES))
    def test_capturable_fsdp2_capture(case_name):
        assert _run_case(case_name)


if __name__ == "__main__":
    ok = run()
    if ok is None:
        print("\nCAPTURABLE FSDP2 CAPTURE OVERALL: SKIP")
        raise SystemExit(77)
    print("\nCAPTURABLE FSDP2 CAPTURE OVERALL:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
