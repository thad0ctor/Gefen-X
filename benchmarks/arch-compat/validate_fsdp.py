"""Gefen architecture smoke validation under FSDP2 (torch fully_shard).

For models whose full-parameter fine-tune does not fit one 24 GB card. Each
transformer block is wrapped with fully_shard across the visible GPUs; Gefen
steps the resulting DTensor params directly (no unshard). Memorization test as
in the single-GPU harnesses; conditioning is fixed random of the correct shape.

Two model families:
  --kind causal-lm    HuggingFace AutoModelForCausalLM (shards decoder layers)
  --kind diffusion    diffusers denoiser (shards transformer blocks) with a
                      per-arch random batch builder shared with validate_diffusion

Run (example, 5 GPUs):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,4,5,7 torchrun \
      --nproc_per_node=5 validate_fsdp.py --kind causal-lm --model <id> [--steps 40]
"""

import argparse, glob, json, os, sys, time, traceback

import torch
import torch.distributed as dist


TEXTS = [
    "The quick brown fox jumps over the lazy dog while the astronomer catalogs "
    "seventeen binary star systems in the northern hemisphere sky.",
    "Optimizer state quantization reduces fine-tuning memory by storing second "
    "moments in shared codebooks indexed per block of parameters.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["causal-lm", "diffusion"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arch", default=None, help="diffusion arch key (sd3|zimage|anima|ltx)")
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "adamw"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    world = dist.get_world_size()

    rec = {
        "tag": args.tag or os.path.basename(args.model.rstrip("/")),
        "model": args.model,
        "kind": args.kind,
        "optimizer": args.optimizer,
        "method": f"full-param FSDP2 x{world}",
        "steps": args.steps,
        "lr": args.lr,
        "status": "FAIL",
        "error": None,
    }
    try:
        run(args, rec, rank, world)
    except Exception:
        rec["error"] = traceback.format_exc()[-2500:]
        print(f"[rank {rank}]", rec["error"], file=sys.stderr)
    finally:
        if rank == 0:
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec, indent=2))
        dist.destroy_process_group()
    sys.exit(0 if rec["status"] == "PASS" else 1)


def _shard_blocks(model, mesh):
    from torch.distributed.fsdp import fully_shard

    # wrap each block of every top-level ModuleList (>1 block); skip ModuleLists
    # nested inside an already-wrapped block so no param is sharded twice
    # (double-wrapping trips FSDP2's overlapping-mesh / tensor-parallel path).
    wrapped = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 1:
            if any(name == w or name.startswith(w + ".") for w in wrapped):
                continue
            for i, blk in enumerate(module):
                fully_shard(blk, mesh=mesh)
                wrapped.append(f"{name}.{i}")
    fully_shard(model, mesh=mesh)
    return len(wrapped)


def _load_full_state(model, load_dir, rank, subfolder=None):
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    src = os.path.join(load_dir, subfolder) if subfolder else load_dir
    full_sd = None
    if rank == 0:
        full_sd = {}
        shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
        for shard in shards:
            full_sd.update(load_file(shard))
        full_sd = {k: v.to(torch.bfloat16) for k, v in full_sd.items()}
    set_model_state_dict(
        model,
        full_sd if full_sd is not None else {},
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
    )
    del full_sd


def build_causal_lm(args, rank, mesh):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn,
        )
    model = model.to(torch.bfloat16)
    nblk = _shard_blocks(model, mesh)
    model.to_empty(device="cuda")
    # resolve the on-disk snapshot dir for state-dict load
    from huggingface_hub import snapshot_download

    load_dir = args.model if os.path.isdir(args.model) else snapshot_download(args.model)
    _load_full_state(model, load_dir, rank)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(TEXTS, return_tensors="pt", padding=True)
    enc = {k: v.cuda() for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels

    model.train()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, nblk, (lambda: model(**enc).loss)


def build_diffusion(args, rank, mesh):
    # reuse validate_diffusion's arch input builders on an empty-init sharded model
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "vdiff", os.path.join(os.path.dirname(__file__), "validate_diffusion.py")
    )
    vdiff = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vdiff)
    # the single-GPU setup loads + moves to cuda; for FSDP we instead build the
    # denoiser class the same way but shard. Simplest robust path: load full on
    # rank, shard, then broadcast — done by loading normally then wrapping.
    setup = vdiff.ARCHES[args.arch]
    denoiser, loss_fn = setup(args.model, "cuda", torch.bfloat16, args.res, {})
    nblk = _shard_blocks(denoiser, mesh)
    denoiser.train()
    if hasattr(denoiser, "enable_gradient_checkpointing"):
        denoiser.enable_gradient_checkpointing()
    return denoiser, nblk, loss_fn


def run(args, rec, rank, world):
    from torch.distributed.device_mesh import init_device_mesh

    torch.manual_seed(0)
    # name the mesh dim: torch 2.12 fully_shard trips not_none(mesh_dim_names)
    # on an unnamed 1-D mesh when it takes the sharding-spec path
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))
    t0 = time.time()
    if args.kind == "causal-lm":
        model, nblk, loss_fn = build_causal_lm(args, rank, mesh)
    else:
        model, nblk, loss_fn = build_diffusion(args, rank, mesh)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["blocks_sharded"] = nblk
    rec["params_total_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6)

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    rec["param_ndims"] = sorted({p.ndim for _, p in named})

    import gefen

    if args.optimizer == "gefen":
        opt = gefen.Gefen(named, lr=args.lr)
    else:
        opt = torch.optim.AdamW([p for _, p in named], lr=args.lr)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        loss = loss_fn()
        if not torch.isfinite(loss):
            rec["losses"] = losses
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 5))
        if rank == 0 and step % 5 == 0:
            print(f"step {step}: {losses[-1]}", flush=True)
    rec["train_s"] = round(time.time() - t0, 1)
    rec["losses"] = losses[:3] + ["..."] + losses[-3:] if len(losses) > 8 else losses
    rec["loss_first"] = losses[0]
    rec["loss_last"] = losses[-1]
    rec["peak_vram_GiB_per_gpu"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-9)
    rec["loss_drop_pct"] = round(100 * drop, 1)
    rec["status"] = "PASS" if drop > 0.3 else "FAIL"


if __name__ == "__main__":
    main()
