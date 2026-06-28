"""One optimizer-comparison cell for the Gefen fair-LR optimizer sweep.

Runs a single (model, optimizer, lr, seed) point under a fixed, fully reproducible
regime and appends one JSON result line to --out. The whole comparison is driven
by repeated invocations of this script (see run.sh).

Regime (identical across every cell): full fine-tune, bf16 master weights,
gradient_checkpointing ON, micro-batch 1, packed seq BLOCK tokens/step, constant
LR, fixed seed + identical packed data order across optimizers within a seed.
The dataset is loaded by HuggingFace name (default tatsu-lab/alpaca), tokenized
with the model's own tokenizer using the standard Alpaca prompt template, and the
prompt tokens are masked out of the loss. This makes the suite self-contained and
reproducible from a clean clone with no pre-prepared local dataset.

seed=0 gives a fixed data order; a non-zero seed reshuffles the packed-data order
identically across all optimizers, giving a noise estimate while keeping
cross-optimizer comparability fixed within a seed.

usage:
  python sweep_cell.py --opt gefen_fused --model <path-or-hf-id> --lr 2e-5 \
      --seed 0 --steps 2000 --seq 2048 --gpu 0 --arch 8.6 --tag qwen1p7b \
      --dataset tatsu-lab/alpaca \
      --out results.jsonl

opt in: adamw_bf16, adamw8bit, adamw4bit, gefen_fused, gefen_nonfused, gefen_muon
"""
import argparse
import json
import os
import random
import sys
import time

def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--opt", required=True,
                help="adamw_bf16|adamw8bit|adamw4bit|gefen_fused|gefen_nonfused|gefen_muon")
ap.add_argument("--model", required=True, help="local path OR HuggingFace model id")
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=positive_int, default=2000)
ap.add_argument("--seq", type=positive_int, default=2048)
ap.add_argument("--gpu", type=int, required=True, help="CUDA device index to pin")
ap.add_argument("--arch", default="8.6",
                help="TORCH_CUDA_ARCH_LIST value, e.g. 8.6 (Ampere), 12.0 (Blackwell)")
ap.add_argument("--tag", required=True, help="short model tag used to group results, e.g. qwen1p7b")
ap.add_argument("--out", required=True, help="JSONL file to append the result line to")
ap.add_argument("--dataset", default="tatsu-lab/alpaca",
                help="HuggingFace dataset id (Alpaca-format) OR a local load_from_disk path "
                     "that already has input_ids/labels columns")
ap.add_argument("--max-eval", type=positive_int, default=32, help="held-out eval examples")
ap.add_argument("--warmup", type=nonnegative_int, default=15, help="steps excluded from steady-state tok/s")
ap.add_argument("--eval-every", type=positive_int, default=50,
                help="intermediate eval-logging cadence only; does not affect recorded "
                     "final_eval/EMA/tok_s (raise to cut eval overhead on long runs)")
ap.add_argument("--muon-adjust", default="match_rms_adamw",
                help="adjust_lr_fn for gefen_muon (GefenMuonHybrid): match_rms_adamw "
                     "(AdamW-scale LR) or original/none (Muon-scale LR)")
args = ap.parse_args()

# GPU pin BEFORE importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402
import datasets  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# --- GPU POLICY GUARD: never run on an RTX PRO 6000 (reserved) ---
dev_name = torch.cuda.get_device_name(0)
print(f"[gpu] CUDA_VISIBLE_DEVICES={args.gpu} -> {dev_name}", flush=True)
if "RTX PRO 6000" in dev_name:
    sys.exit(f"ABORT: refusing to run on forbidden GPU {dev_name}")

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
    """Return a list of {"input_ids", "labels"} dicts.

    If --dataset is a local load_from_disk directory that already carries
    input_ids/labels, use it as-is. Otherwise load an Alpaca-format HF dataset,
    render the standard Alpaca prompt with the model tokenizer, and mask the
    prompt tokens out of the loss.
    """
    if os.path.isdir(args.dataset):
        d = datasets.load_from_disk(args.dataset)
        cols = d.column_names
        if "input_ids" in cols and "labels" in cols:
            return d
        raise SystemExit(
            f"local dataset {args.dataset} lacks input_ids/labels columns: {cols}"
        )

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
    # drop examples whose response was fully truncated away: an all -100 labels
    # row supervises zero tokens and yields a 0/0 NaN if it lands in eval.
    return [e for e in rendered if any(t != -100 for t in e["labels"])]


# ---- deterministic data: identical for every optimizer within a seed ----
d = load_examples()
n_examples = len(d)
idx = list(range(n_examples))
random.Random(args.seed).shuffle(idx)
eval_idx = idx[:args.max_eval]
pack_pool = idx[args.max_eval:]


def _ex(i):
    return d[i] if not hasattr(d, "select") else d[int(i)]


def make_blocks(n_blocks):
    """Greedily concatenate examples into exact BLOCK-token (ids,labels) chunks."""
    blocks = []
    buf_ids, buf_lab = [], []
    for i in pack_pool:
        ex = _ex(i)
        buf_ids.extend(ex["input_ids"])
        buf_lab.extend(ex["labels"])
        while len(buf_ids) >= BLOCK:
            ids = torch.tensor(buf_ids[:BLOCK], dtype=torch.long)[None]
            lab = torch.tensor(buf_lab[:BLOCK], dtype=torch.long)[None]
            blocks.append((ids, lab))
            buf_ids = buf_ids[BLOCK:]
            buf_lab = buf_lab[BLOCK:]
            if len(blocks) >= n_blocks:
                return blocks
    return blocks


blocks = make_blocks(args.steps)
assert len(blocks) == args.steps, (
    f"only packed {len(blocks)}/{args.steps} blocks from {n_examples} examples; "
    f"use a larger dataset or fewer --steps"
)


def eval_batch(i):
    ex = _ex(i)
    ids = torch.tensor(ex["input_ids"][:BLOCK], dtype=torch.long)[None].cuda()
    lab = torch.tensor(ex["labels"][:BLOCK], dtype=torch.long)[None].cuda()
    return ids, lab


# ---- model: bf16 master weights, grad checkpointing ON ----
m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda()
m.config.use_cache = False
m.gradient_checkpointing_enable()
m.train()
n_params = sum(p.numel() for p in m.parameters())


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


lr = args.lr
if args.opt == "adamw_bf16":
    opt = torch.optim.AdamW(m.parameters(), lr=lr, fused=True)
elif args.opt == "adamw8bit":
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(m.parameters(), lr=lr)
elif args.opt == "adamw4bit":
    from torchao.optim import AdamW4bit
    opt = AdamW4bit(m.parameters(), lr=lr)
elif args.opt == "gefen_fused":
    from gefen import Gefen
    opt = Gefen(m.parameters(), lr=lr, fused=True)
elif args.opt == "gefen_nonfused":
    from gefen import Gefen
    opt = Gefen(m.parameters(), lr=lr, fused=False)
elif args.opt == "gefen_muon":
    from gefen import GefenMuonHybrid
    _adj = None if args.muon_adjust in ("none", "None") else args.muon_adjust
    opt = GefenMuonHybrid(*split_params_for_muon(m), lr=lr,
                          adjust_lr_fn=_adj, fused=True)
    print(f"[gefen_muon] adjust_lr_fn={_adj!r}", flush=True)
else:
    sys.exit(f"unknown opt {args.opt}")


@torch.no_grad()
def evaluate():
    m.eval()
    tot = 0.0
    nb = 0
    for i in eval_idx:
        ids, lab = eval_batch(i)
        tot += m(ids, labels=lab).loss.item()
        nb += 1
    m.train()
    return tot / nb


print(f"=== {args.tag} {args.opt} lr={lr:.3e} seed={args.seed} steps={args.steps} "
      f"seq={BLOCK} params={n_params/1e9:.3f}B ===", flush=True)
eval0 = evaluate()
print(f"step 0  eval_loss {eval0:.4f}", flush=True)

torch.cuda.reset_peak_memory_stats()
ema = None
step_tokens = BLOCK
steady_tokens = 0
steady_time = 0.0
# never exclude every step: short runs (steps <= warmup) still report a tok/s.
warmup_eff = min(args.warmup, args.steps - 1) if args.steps > 1 else 0
for s in range(1, args.steps + 1):
    ids, lab = blocks[s - 1]
    ids = ids.cuda()
    lab = lab.cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    loss = m(ids, labels=lab).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    if s > warmup_eff:
        steady_tokens += step_tokens
        steady_time += dt
    l = loss.item()
    ema = l if ema is None else 0.95 * ema + 0.05 * l
    if s % args.eval_every == 0:
        print(f"step {s}  train_ema {ema:.4f}  eval {evaluate():.4f}  "
              f"{step_tokens/dt:.0f} tok/s", flush=True)

final_eval = evaluate()
# allocated = real tensor bytes (the apples-to-apples optimizer-memory metric);
# reserved = caching-allocator pool incl. slack/fragmentation (closer to nvidia-smi).
peak = torch.cuda.max_memory_allocated() / 1024**3
peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
tok_s = steady_tokens / steady_time if steady_time else float("nan")


def nbytes(v):
    # recurse into tensor subclasses (e.g. torchao OptimState4bit packs codes/scale/qmap);
    # logical element_size() over-counts packed 4-bit state ~4x.
    if hasattr(v, "__tensor_flatten__"):
        try:
            names, _ = v.__tensor_flatten__()
            return sum(nbytes(getattr(v, n)) for n in names)
        except Exception:
            pass
    try:
        return v.numel() * v.element_size()
    except Exception:
        return 0


state_bytes = sum(
    nbytes(v)
    for s_ in opt.state.values()
    for v in (s_.values() if hasattr(s_, "values") else [])
    if torch.is_tensor(v)
)
state_bpp = state_bytes / n_params

res = {
    "tag": args.tag, "opt": args.opt, "lr": lr, "seed": args.seed,
    "muon_adjust": (args.muon_adjust if args.opt == "gefen_muon" else None),
    "steps": args.steps, "seq": BLOCK, "params_B": round(n_params / 1e9, 4),
    "gpu": dev_name, "final_train_ema": round(ema, 4),
    "eval0": round(eval0, 4), "final_eval": round(final_eval, 4),
    "tok_per_s": round(tok_s, 1), "peak_vram_gib": round(peak, 2),
    "peak_vram_reserved_gib": round(peak_reserved, 2),
    "opt_state_bpp": round(state_bpp, 3),
    "opt_state_gib": round(state_bytes / 1024**3, 3),
}
print("RESULT " + json.dumps(res), flush=True)
with open(args.out, "a") as f:
    f.write(json.dumps(res) + "\n")
