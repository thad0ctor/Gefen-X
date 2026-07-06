"""Gefen architecture smoke validation for diffusion / flow-matching denoisers.

Memorization test on the denoiser only (transformer or UNet): a fixed synthetic
batch (latents, noise, timestep, conditioning) is trained for N optimizer steps
and the MSE loss must fall sharply (>30%) with no NaN/Inf. Text encoders / VAEs
are never trained; conditioning is fixed random of the correct shape, so this
validates the optimizer over the denoiser's parameter tensors, not text fidelity
(same rationale as the FLUX.1-dev entry in COMPATIBILITY.md).

Each architecture has a small `setup_<arch>` that returns (denoiser, loss_fn).
Add a new arch by writing one and registering it in ARCHES.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_diffusion.py \
      --arch sd3|zimage|anima|ltx|ltx2|sdxl|flux2klein --model <path-or-id> \
      --optimizer gefen [--steps 60] [--lr 2e-5] [--res 512]
"""

import argparse, json, os, sys, time, traceback

import torch

PROMPT = "a red square next to a blue circle on a white background"


def _mse(pred, target):
    return torch.nn.functional.mse_loss(pred.float(), target.float())


# --------------------------------------------------------------------------- #
# arch adapters: each returns (denoiser, loss_fn); loss_fn() -> scalar tensor  #
# --------------------------------------------------------------------------- #
def setup_sdxl(model, device, dtype, res, rec):
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    add_time_ids = pipe._get_add_time_ids(
        (res, res), (0, 0), (res, res), dtype,
        text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim,
    ).to(device)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    denoiser = pipe.unet.to(device)
    lat = res // 8
    g = torch.Generator(device).manual_seed(0)
    x0 = torch.randn((1, 4, lat, lat), generator=g, device=device, dtype=dtype)
    noise = torch.randn((1, 4, lat, lat), generator=g, device=device, dtype=dtype)
    t = torch.tensor([500], device=device)
    noisy = pipe.scheduler.add_noise(x0, noise, t)

    def loss_fn():
        pred = denoiser(
            noisy, t, encoder_hidden_states=pe,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        ).sample
        return _mse(pred, noise)

    return denoiser, loss_fn


def setup_sd3(model, device, dtype, res, rec):
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=dtype)
    for enc in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if getattr(pipe, enc, None) is not None:
            getattr(pipe, enc).to(device)
    with torch.no_grad():
        pe, _, pooled, _ = pipe.encode_prompt(
            prompt=PROMPT, prompt_2=None, prompt_3=None, device=device,
            num_images_per_prompt=1, do_classifier_free_guidance=False,
        )
    for enc in ("text_encoder", "text_encoder_2", "text_encoder_3"):
        if getattr(pipe, enc, None) is not None:
            getattr(pipe, enc).to("cpu")
    denoiser = pipe.transformer.to(device)
    ch = denoiser.config.in_channels
    lat = res // 8
    g = torch.Generator(device).manual_seed(0)
    x0 = torch.randn((1, ch, lat, lat), generator=g, device=device, dtype=dtype)
    noise = torch.randn((1, ch, lat, lat), generator=g, device=device, dtype=dtype)
    t_frac = 0.5
    x_t = (1 - t_frac) * x0 + t_frac * noise
    target = noise - x0
    timestep = torch.full((1,), 1000 * t_frac, device=device, dtype=dtype)

    def loss_fn():
        pred = denoiser(
            hidden_states=x_t, timestep=timestep,
            encoder_hidden_states=pe, pooled_projections=pooled,
            return_dict=False,
        )[0]
        return _mse(pred, target)

    return denoiser, loss_fn


def setup_zimage(model, device, dtype, res, rec, denoiser=None):
    # Z-Image: Scalable Single-Stream DiT (S3-DiT), Qwen3 text encoder + SigLIP.
    # forward(x, t, cap_feats, ...): x is a [B,C,H,W] latent (patchified inside),
    # cap_feats are the Qwen3 caption embeddings. SigLIP feats are optional.
    from diffusers import ZImageTransformer2DModel

    if denoiser is None:  # FSDP passes an already empty-init'd + sharded model
        denoiser = ZImageTransformer2DModel.from_pretrained(
            model, subfolder="transformer", torch_dtype=dtype
        ).to(device)
    cfg = denoiser.config
    ch = cfg.in_channels
    lat = res // 8  # Flux VAE x8; the DiT patchifies (patch_size=2) internally
    g = torch.Generator(device).manual_seed(0)

    def rnd(*s):
        return torch.randn(*s, generator=g, device=device, dtype=dtype)

    x0 = rnd(1, ch, 1, lat, lat)  # [B, C, F, H, W], single frame
    noise = rnd(1, ch, 1, lat, lat)
    cap = rnd(1, 64, cfg.cap_feat_dim)
    t_frac = 0.5
    x_t = (1 - t_frac) * x0 + t_frac * noise
    target = (noise - x0)[0]  # model returns a per-sample list of [C,F,H,W]
    timestep = torch.full((1,), t_frac, device=device, dtype=dtype)

    def loss_fn():
        pred = denoiser(x=x_t, t=timestep, cap_feats=cap, return_dict=False)[0][0]
        return _mse(pred, target)

    return denoiser, loss_fn


def setup_anima(model, device, dtype, res, rec, denoiser=None):
    # Anima reuses CosmosTransformer3DModel (video DiT) for still images.
    from diffusers import CosmosTransformer3DModel

    if denoiser is None:
        denoiser = CosmosTransformer3DModel.from_pretrained(
            model, subfolder="transformer", torch_dtype=dtype
        ).to(device)
    cfg = denoiser.config
    ch = cfg.in_channels
    lat = res // 8  # Cosmos patchifies internally; feed the raw VAE latent grid
    g = torch.Generator(device).manual_seed(0)

    def rnd(*s):
        return torch.randn(*s, generator=g, device=device, dtype=dtype)

    # video latents [B, C, T, H, W] with T=1 frame
    x0 = rnd(1, ch, 1, lat, lat)
    noise = rnd(1, ch, 1, lat, lat)
    txt_dim = getattr(cfg, "text_embed_dim", getattr(cfg, "caption_channels", 2048))
    txt = rnd(1, 64, txt_dim)
    t_frac = 0.5
    x_t = (1 - t_frac) * x0 + t_frac * noise
    target = noise - x0
    timestep = torch.full((1,), 1000 * t_frac, device=device, dtype=dtype)
    # concat_padding_mask=True -> the model needs a [B,1,H,W] mask to resize+concat
    padding_mask = torch.zeros(1, 1, lat, lat, device=device, dtype=dtype)

    def loss_fn():
        pred = denoiser(
            hidden_states=x_t, timestep=timestep,
            encoder_hidden_states=txt, padding_mask=padding_mask,
            return_dict=False,
        )[0]
        return _mse(pred, target)

    return denoiser, loss_fn


def setup_ltx(model, device, dtype, res, rec, denoiser=None):
    from diffusers import LTXVideoTransformer3DModel

    if denoiser is None:
        denoiser = LTXVideoTransformer3DModel.from_pretrained(
            model, subfolder="transformer", torch_dtype=dtype
        ).to(device)
    cfg = denoiser.config
    ch = cfg.in_channels
    lat = res // 32
    seq = lat * lat
    g = torch.Generator(device).manual_seed(0)

    def rnd(*s):
        return torch.randn(*s, generator=g, device=device, dtype=dtype)

    x0 = rnd(1, seq, ch)
    noise = rnd(1, seq, ch)
    txt = rnd(1, 64, cfg.caption_channels)
    t_frac = 0.5
    x_t = (1 - t_frac) * x0 + t_frac * noise
    target = noise - x0
    timestep = torch.full((1,), t_frac, device=device, dtype=dtype)
    # LTX needs per-token video coords
    coords = torch.zeros(1, 3, seq, device=device)
    coords[:, 1] = torch.arange(lat, device=device).repeat_interleave(lat)
    coords[:, 2] = torch.arange(lat, device=device).repeat(lat)

    def loss_fn():
        pred = denoiser(
            hidden_states=x_t, encoder_hidden_states=txt, timestep=timestep,
            video_coords=coords, return_dict=False,
        )[0]
        return _mse(pred, target)

    return denoiser, loss_fn


ARCHES = {
    "sdxl": setup_sdxl,
    "sd3": setup_sd3,
    "zimage": setup_zimage,
    "anima": setup_anima,
    "ltx": setup_ltx,
    "ltx2": setup_ltx,  # LTX-2 uses the same transformer family; overridden if it diverges
}


def fsdp_denoiser_class(arch):
    """(diffusers model class, subfolder) for archs that can be empty-init'd and
    sharded — the transformer-subfolder denoisers with random conditioning.
    sdxl/sd3 load a full pipeline for conditioning and stay single-GPU."""
    from diffusers import (
        CosmosTransformer3DModel,
        LTXVideoTransformer3DModel,
        ZImageTransformer2DModel,
    )

    table = {
        "zimage": (ZImageTransformer2DModel, "transformer"),
        "anima": (CosmosTransformer3DModel, "transformer"),
        "ltx": (LTXVideoTransformer3DModel, "transformer"),
        "ltx2": (LTXVideoTransformer3DModel, "transformer"),
    }
    if arch not in table:
        raise ValueError(f"--kind diffusion --arch {arch} is single-GPU only; "
                         f"FSDP2 supports {sorted(table)}")
    return table[arch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "hybrid", "adamw"])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    rec = {
        "tag": args.tag or f"{args.arch}-{args.optimizer}",
        "model": args.model,
        "arch": args.arch,
        "optimizer": args.optimizer,
        "method": "full-param",
        "steps": args.steps,
        "lr": args.lr,
        "status": "FAIL",
        "error": None,
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
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    t0 = time.time()
    denoiser, loss_fn = ARCHES[args.arch](args.model, device, dtype, args.res, rec)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["architecture"] = type(denoiser).__name__
    denoiser.train()
    if hasattr(denoiser, "enable_gradient_checkpointing"):
        denoiser.enable_gradient_checkpointing()
    rec["params_total_M"] = round(sum(p.numel() for p in denoiser.parameters()) / 1e6)
    named = [(n, p) for n, p in denoiser.named_parameters() if p.requires_grad]
    rec["param_ndims"] = sorted({p.ndim for _, p in named})

    import gefen

    if args.optimizer == "gefen":
        opt = gefen.Gefen(named, lr=args.lr)
    elif args.optimizer == "hybrid":
        muon_named, backup_named = gefen.split_params_for_muon(denoiser)
        opt = gefen.GefenMuonHybrid(muon_named, backup_named, lr=args.lr)
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
    rec["train_s"] = round(time.time() - t0, 1)
    rec["losses"] = losses[:3] + ["..."] + losses[-3:] if len(losses) > 8 else losses
    rec["loss_first"] = losses[0]
    rec["loss_last"] = losses[-1]
    rec["peak_vram_GiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-9)
    rec["loss_drop_pct"] = round(100 * drop, 1)
    rec["status"] = "PASS" if drop > 0.3 else "FAIL"


if __name__ == "__main__":
    main()
