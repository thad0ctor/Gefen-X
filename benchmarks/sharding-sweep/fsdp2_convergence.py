"""FSDP2 GefenMuon convergence cell: exact vs approx sharded Newton-Schulz.

One cell of the `sharding-sweep` benchmark (analogous to optimizer-sweep's
`sweep_cell.py`). `run.sh` launches this once per (world_size x sharded_mode)
via `torchrun --nproc_per_node=<world>`; `plot_sharding.py` then turns the
results.jsonl + per-cell logs into the two README charts.

Runs ONE real fine-tune of a causal LM under FSDP2 (fully_shard / DTensor) with
GefenMuonHybrid, in the SAME single-GPU regime as optimizer-sweep (Alpaca,
packed BLOCK-token blocks, bf16 master weights, gradient checkpointing,
micro-batch 1, constant LR, fixed seed), and reports final eval loss. The ONLY
knob under test is --sharded-mode {exact, approx}:

  exact  : every rank all-gathers the full gradient, runs Newton-Schulz on the
           full matrix (bit-for-bit single-GPU parity), slices the update back.
  approx : each rank runs the whole pipeline on its LOCAL row-shard only (no
           all-gather, NS on a smaller matrix) -- NON-PARITY, but ~2.2x faster.

Data is REPLICATED across ranks (every rank sees the identical packed block each
step), so exact-mode FSDP2 reproduces the single-GPU trajectory exactly and any
divergence in approx mode is purely the Newton-Schulz approximation -- not a
data-parallel batch-size change. This isolates the loss impact of the approx NS.

Run under torchrun (launcher pins the GPUs):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,7 \
  TORCH_CUDA_ARCH_LIST=8.6 CUDA_HOME=/usr/local/cuda-13.3 \
  PYTHONPATH=<symlink-dir> \
  torchrun --nproc_per_node=2 fsdp2_convergence.py \
      --model <path> --lr 5e-5 --steps 2000 --sharded-mode approx \
      --tag qwen0p6b --out results_fsdp2.jsonl
"""
import argparse
import json
import os
import random
import sys
import time


def _strip_shadowing_gefen():
    # A flat `gefen.py` on sys.path shadows the installed `gefen/` package.
    sys.path[:] = [
        p for p in sys.path
        if not os.path.isfile(os.path.join(p or ".", "gefen.py"))
    ]


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=positive_int, default=2000)
ap.add_argument("--seq", type=positive_int, default=2048)
ap.add_argument("--sharded-mode", default="exact", choices=["exact", "approx"])
ap.add_argument("--tag", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--dataset", default="tatsu-lab/alpaca")
ap.add_argument("--max-eval", type=positive_int, default=32)
ap.add_argument("--eval-every", type=positive_int, default=50)
ap.add_argument("--muon-adjust", default="match_rms_adamw")
args = ap.parse_args()

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

_strip_shadowing_gefen()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import datasets  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local_rank)
dev = torch.device("cuda", local_rank)
dist.init_process_group("nccl")


def log(*a):
    if rank == 0:
        print(*a, flush=True)


dev_name = torch.cuda.get_device_name(local_rank)
if "RTX PRO 6000" in dev_name:
    sys.exit(f"ABORT: refusing to run on forbidden GPU {dev_name}")
log(f"[gpu] world={world} local0->{dev_name} sharded_mode={args.sharded_mode}")

BLOCK = args.seq
torch.manual_seed(args.seed)

ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)
ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def load_examples():
    if os.path.isdir(args.dataset):
        d = datasets.load_from_disk(args.dataset)
        if "input_ids" in d.column_names and "labels" in d.column_names:
            return d
        raise SystemExit("local dataset lacks input_ids/labels")
    raw = datasets.load_dataset(args.dataset, split="train")
    tok = AutoTokenizer.from_pretrained(args.model)
    eos = tok.eos_token or ""

    def render(ex):
        inp = (ex.get("input") or "").strip()
        if inp:
            prompt = ALPACA_PROMPT_WITH_INPUT.format(
                instruction=ex["instruction"].strip(), input=inp)
        else:
            prompt = ALPACA_PROMPT_NO_INPUT.format(
                instruction=ex["instruction"].strip())
        response = ex["output"].strip() + eos
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tok(response, add_special_tokens=False)["input_ids"]
        ids = (p_ids + r_ids)[:BLOCK]
        labels = ([-100] * len(p_ids) + r_ids)[:BLOCK]
        return {"input_ids": ids, "labels": labels}

    rendered = [render(ex) for ex in raw]
    return [e for e in rendered if any(t != -100 for t in e["labels"])]


d = load_examples()
n_examples = len(d)
idx = list(range(n_examples))
random.Random(args.seed).shuffle(idx)
eval_idx = idx[:args.max_eval]
pack_pool = idx[args.max_eval:]


def _ex(i):
    return d[i] if not hasattr(d, "select") else d[int(i)]


def make_blocks(n_blocks):
    blocks, buf_ids, buf_lab = [], [], []
    for i in pack_pool:
        ex = _ex(i)
        buf_ids.extend(ex["input_ids"])
        buf_lab.extend(ex["labels"])
        while len(buf_ids) >= BLOCK:
            ids = torch.tensor(buf_ids[:BLOCK], dtype=torch.long)[None]
            lab = torch.tensor(buf_lab[:BLOCK], dtype=torch.long)[None]
            blocks.append((ids, lab))
            buf_ids, buf_lab = buf_ids[BLOCK:], buf_lab[BLOCK:]
            if len(blocks) >= n_blocks:
                return blocks
    return blocks


blocks = make_blocks(args.steps)
assert len(blocks) == args.steps, f"only packed {len(blocks)}/{args.steps} blocks"


def eval_batch(i):
    ex = _ex(i)
    ids = torch.tensor(ex["input_ids"][:BLOCK], dtype=torch.long)[None].to(dev)
    lab = torch.tensor(ex["labels"][:BLOCK], dtype=torch.long)[None].to(dev)
    return ids, lab


# ---- model + FSDP2 ----
m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
m.config.use_cache = False
m.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
m.train()
n_params = sum(p.numel() for p in m.parameters())

from torch.distributed.device_mesh import init_device_mesh  # noqa: E402
from torch.distributed.fsdp import fully_shard  # noqa: E402

mesh = init_device_mesh("cuda", (world,))
# Shard each decoder layer, then the root (embeddings / final norm / lm_head).
layers = m.model.layers
for layer in layers:
    fully_shard(layer, mesh=mesh)
fully_shard(m, mesh=mesh)


def split_params_for_muon(model):
    muon, backup = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_matrix = (
            p.ndim == 2
            and not any(k in name.lower() for k in ("embed", "wte", "lm_head"))
        )
        (muon if is_hidden_matrix else backup).append((name, p))
    return muon, backup


from gefen import GefenMuonHybrid  # noqa: E402

_adj = None if args.muon_adjust in ("none", "None") else args.muon_adjust
muon_p, backup_p = split_params_for_muon(m)
opt = GefenMuonHybrid(muon_p, backup_p, lr=args.lr, adjust_lr_fn=_adj,
                      fused=True, sharded_mode=args.sharded_mode)
log(f"[gefen_muon] adjust_lr_fn={_adj!r} sharded_mode={args.sharded_mode} "
    f"muon_params={len(muon_p)} backup_params={len(backup_p)}")


@torch.no_grad()
def evaluate():
    m.eval()
    tot, nb = 0.0, 0
    for i in eval_idx:
        ids, lab = eval_batch(i)
        tot += m(ids, labels=lab).loss.item()
        nb += 1
    m.train()
    return tot / nb


log(f"=== {args.tag} gefen_muon[{args.sharded_mode}] x{world} lr={args.lr:.3e} "
    f"seed={args.seed} steps={args.steps} seq={BLOCK} "
    f"params={n_params/1e9:.3f}B ===")
eval0 = evaluate()
log(f"step 0  eval_loss {eval0:.4f}")

torch.cuda.reset_peak_memory_stats()
ema = None
# Steady-state s/step: time each train step (fwd+bwd+opt, eval excluded) and
# average over the post-warmup steps. Persisted into RESULT so plot_sharding.py
# need not infer it from the last eval log line (wrong when steps % eval_every).
warmup_steps = min(10, max(0, args.steps - 1))
step_times = []
t_start = time.time()
for s in range(1, args.steps + 1):
    t_step = time.time()
    ids, lab = blocks[s - 1]
    ids, lab = ids.to(dev), lab.to(dev)
    loss = m(ids, labels=lab).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    loss_val = loss.item()  # also syncs the step before we read the wall clock
    if s > warmup_steps:
        step_times.append(time.time() - t_step)
    ema = loss_val if ema is None else 0.95 * ema + 0.05 * loss_val
    if s % args.eval_every == 0:
        log(f"step {s}  train_ema {ema:.4f}  eval {evaluate():.4f}  "
            f"({(time.time()-t_start)/s:.2f}s/step)")

final_eval = evaluate()
peak = torch.cuda.max_memory_allocated() / 1024**3
# peak is rank-local; the row is treated as cell-wide, so publish the max peak
# across ranks (collective -- every rank must call before rank 0 writes).
peak_t = torch.tensor(peak, device=dev)
dist.all_reduce(peak_t, op=dist.ReduceOp.MAX)
peak = peak_t.item()
s_per_step = sum(step_times) / len(step_times) if step_times else None

if rank == 0:
    res = {
        "tag": args.tag, "opt": "gefen_muon", "sharded_mode": args.sharded_mode,
        "world_size": world, "lr": args.lr, "seed": args.seed,
        "muon_adjust": args.muon_adjust, "steps": args.steps, "seq": BLOCK,
        "params_B": round(n_params / 1e9, 4), "gpu": dev_name,
        "final_train_ema": round(ema, 4), "eval0": round(eval0, 4),
        "final_eval": round(final_eval, 4), "peak_vram_gib": round(peak, 2),
        "s_per_step": round(s_per_step, 4) if s_per_step is not None else None,
    }
    print("RESULT " + json.dumps(res), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")

dist.barrier()
dist.destroy_process_group()
