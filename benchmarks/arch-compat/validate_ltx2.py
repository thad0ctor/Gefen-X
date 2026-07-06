"""Gefen architecture smoke validation for LTX-2 (19B dual-stream video+audio DiT).

LTX-2's `LTX2VideoTransformer3DModel` denoises a joint video + audio latent stream
(both modalities are required inputs). At 19B it does not fit one 24 GB card, so the
base is sharded across GPUs with `device_map="auto"` and adapted with LoRA (Gefen
steps the adapters) -- the fallback tier, same rationale as the large MoE LLMs.
Conditioning is fixed random of the correct shape (optimizer/arch check, not fidelity).

Run:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2,4,5 python validate_ltx2.py \
      --model Lightricks/LTX-2 --steps 60 --lr 2e-4
"""

import argparse, json, os, sys, time, traceback

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Lightricks/LTX-2")
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "adamw"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-gpu-mem", default="22GiB")
    ap.add_argument("--full-param", action="store_true",
                    help="train every parameter on a single GPU (needs a big card, "
                         "e.g. a 96 GB RTX PRO 6000); no device_map, no LoRA")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--tag", default="LTX-2-19B")
    args = ap.parse_args()

    rec = {
        "tag": args.tag, "model": args.model, "arch": "ltx2",
        "optimizer": args.optimizer, "method": "full-param" if args.full_param else "lora",
        "steps": args.steps, "lr": args.lr, "status": "FAIL", "error": None,
    }
    try:
        run(args, rec)
    except Exception:
        rec["error"] = traceback.format_exc()[-2500:]
        print(rec["error"], file=sys.stderr)
    finally:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=2))
    sys.exit(0 if rec["status"] == "PASS" else 1)


def run(args, rec):
    from diffusers import LTX2VideoTransformer3DModel
    from peft import LoraConfig, get_peft_model

    # diffusers ships this class without `_no_split_modules`, which blocks
    # device_map sharding; declare the transformer block as the atomic unit.
    if not getattr(LTX2VideoTransformer3DModel, "_no_split_modules", None):
        LTX2VideoTransformer3DModel._no_split_modules = ["LTX2VideoTransformerBlock"]

    dtype = torch.bfloat16
    torch.manual_seed(0)
    n_gpu = torch.cuda.device_count()
    caps = {i: args.max_gpu_mem for i in range(n_gpu)}

    t0 = time.time()
    if args.full_param:
        # single big GPU: whole model on one device -> no cross-device coord issue
        denoiser = LTX2VideoTransformer3DModel.from_pretrained(
            args.model, subfolder="transformer", torch_dtype=dtype,
        ).to("cuda")
    else:
        denoiser = LTX2VideoTransformer3DModel.from_pretrained(
            args.model, subfolder="transformer", torch_dtype=dtype,
            device_map="auto", max_memory=caps,
        )
    rec["load_s"] = round(time.time() - t0, 1)
    rec["architecture"] = type(denoiser).__name__
    rec["params_total_M"] = round(sum(p.numel() for p in denoiser.parameters()) / 1e6)
    per_dev = {}
    for _, p in denoiser.named_parameters():
        per_dev[str(p.device)] = per_dev.get(str(p.device), 0) + p.numel() * p.element_size()
    rec["weights_by_device_GiB"] = {k: round(v / 2**30, 2) for k, v in per_dev.items()}
    if any(d in ("meta", "cpu", "disk") for d in per_dev):
        raise RuntimeError(f"base offloaded off-GPU ({per_dev}); raise --max-gpu-mem or add GPUs")

    c = denoiser.config
    dev = next(denoiser.parameters()).device

    if not args.full_param:
        if hasattr(denoiser, "enable_input_require_grads"):
            denoiser.enable_input_require_grads()
        # attention q/k/v across the video, audio, and cross-modal streams; explicit
        # names avoid matching numeric ModuleList indices (which would hit whole blocks)
        denoiser = get_peft_model(
            denoiser,
            LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
                       bias="none", target_modules=["to_q", "to_k", "to_v"]),
        )
    denoiser.train()
    if hasattr(denoiser, "gradient_checkpointing_enable"):
        denoiser.gradient_checkpointing_enable()

    # fixed synthetic dual-stream batch (single frame, small grid + short audio)
    base = denoiser.get_base_model() if hasattr(denoiser, "get_base_model") else denoiser
    F, H, W, A = 1, 16, 16, 32
    g = torch.Generator(dev).manual_seed(0)

    def rnd(*s):
        return torch.randn(*s, generator=g, device=dev, dtype=dtype)

    v0, vn = rnd(1, F * H * W, c.in_channels), rnd(1, F * H * W, c.in_channels)
    a0, an = rnd(1, A, c.audio_in_channels), rnd(1, A, c.audio_in_channels)
    ehs = rnd(1, 32, c.caption_channels)
    aehs = rnd(1, 32, c.caption_channels)
    tf = 0.5
    v_t, a_t = (1 - tf) * v0 + tf * vn, (1 - tf) * a0 + tf * an
    v_target, a_target = vn - v0, an - a0
    ts = torch.full((1,), tf, device=dev, dtype=dtype)
    video_coords = base.rope.prepare_video_coords(1, F, H, W, dev, fps=25.0)
    audio_coords = base.audio_rope.prepare_audio_coords(1, A, dev)

    def loss_fn():
        vp, apr = denoiser(
            hidden_states=v_t, audio_hidden_states=a_t,
            encoder_hidden_states=ehs, audio_encoder_hidden_states=aehs,
            timestep=ts, audio_timestep=ts,
            num_frames=F, height=H, width=W, fps=25.0, audio_num_frames=A,
            video_coords=video_coords, audio_coords=audio_coords, return_dict=False,
        )
        return torch.nn.functional.mse_loss(vp.float(), v_target.float()) \
            + torch.nn.functional.mse_loss(apr.float(), a_target.float())

    named = [(n, p) for n, p in denoiser.named_parameters() if p.requires_grad]
    rec["params_trained_M"] = round(sum(p.numel() for _, p in named) / 1e6)
    rec["param_ndims"] = sorted({p.ndim for _, p in named})

    import gefen

    opt = gefen.Gefen(named, lr=args.lr) if args.optimizer == "gefen" \
        else torch.optim.AdamW([p for _, p in named], lr=args.lr)

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
    rec["train_s"] = round(time.time() - t0, 1)
    rec["losses"] = losses[:3] + ["..."] + losses[-3:] if len(losses) > 8 else losses
    rec["loss_first"], rec["loss_last"] = losses[0], losses[-1]
    rec["peak_vram_GiB"] = round(
        sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 2**30, 2)
    drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-9)
    rec["loss_drop_pct"] = round(100 * drop, 1)
    rec["status"] = "PASS" if drop > 0.3 else "FAIL"


if __name__ == "__main__":
    main()
