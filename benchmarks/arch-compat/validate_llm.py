"""Gefen architecture smoke validation for LLMs / VLMs.

Memorization test: a fixed small batch is trained for N optimizer steps and the
loss must fall sharply (>30%) with no NaN/Inf. A probe backward pass discovers
the parameters that actually receive gradients (so modality towers that get no
input are excluded), then the requested optimizer is built over the named
params and the batch is memorized.

Three coverage tiers (see benchmarks/arch-compat/README.md):
  * full-parameter, single GPU              (default)
  * full-parameter, sharded via device_map  (--device-map auto --max-gpu-mem)
  * QLoRA adapter train                      (--qlora)  -- see README caveat

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx> python validate_llm.py \
      --model <path-or-hf-id> --optimizer gefen [--vl] [--steps 40] [--lr 1e-4] \
      [--trust-remote-code] [--qlora] [--device-map auto --max-gpu-mem 22GiB] \
      [--out results.jsonl]
"""

import argparse, json, os, sys, time, traceback

import torch


TEXTS = [
    "The quick brown fox jumps over the lazy dog while the astronomer catalogs "
    "seventeen binary star systems in the northern hemisphere sky.",
    "Optimizer state quantization reduces fine-tuning memory by storing second "
    "moments in shared codebooks indexed per block of parameters.",
]

VL_CAPTION = "Describe the image."
VL_ANSWER = "A red square sits left of a blue circle on a white background."


IMG_SIZE = 384


def make_image():
    from PIL import Image, ImageDraw

    s = IMG_SIZE / 384.0
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([40 * s, 140 * s, 150 * s, 250 * s], fill="red")
    d.ellipse([220 * s, 140 * s, 330 * s, 250 * s], fill="blue")
    return img


def build_text_batch(tok, device):
    enc = tok(TEXTS, return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


def build_vl_batch(processor, device):
    img = make_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": VL_CAPTION},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": VL_ANSWER}]},
    ]
    enc = processor.apply_chat_template(
        [messages],
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    labels = enc["input_ids"].clone()
    if "attention_mask" in enc:
        labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "hybrid", "adamw"])
    ap.add_argument("--vl", action="store_true")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip-grad-norm", type=float, default=None,
                    help="clip grad norm before opt.step (needed for stable SSM/Mamba training)")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--config-override", default=None,
                    help="comma-separated key=val config overrides for buggy configs, "
                         "e.g. num_hidden_layers=48 (ints/floats parsed, else string)")
    ap.add_argument("--device-map", default=None, help="e.g. 'auto' to shard across visible GPUs")
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--max-gpu-mem", default=None, help="per-GPU cap for device_map, e.g. 22GiB")
    ap.add_argument("--cpu-offload", default=None,
                    help="CPU RAM budget for device_map offload, e.g. 200GiB (frees GPU for "
                         "activations; pair with --lora so only GPU-resident adapters train)")
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base + LoRA adapters (fallback tier)")
    ap.add_argument("--lora", action="store_true", help="bf16 base + LoRA adapters (fallback tier; pairs with --device-map)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", default=None,
                    help="comma-separated module names to adapt (default: all linears; "
                         "set explicitly for MoE to skip huge expert-weight adapters)")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    global IMG_SIZE
    IMG_SIZE = args.img_size

    rec = {
        "tag": args.tag or os.path.basename(args.model.rstrip("/")),
        "model": args.model,
        "optimizer": args.optimizer,
        "vl": args.vl,
        "method": "qlora" if args.qlora else ("lora" if args.lora else ("device_map" if args.device_map else "full-param")),
        "steps": args.steps,
        "lr": args.lr,
        "status": "FAIL",
        "error": None,
    }
    try:
        run(args, rec)
    except Exception:
        rec["error"] = traceback.format_exc()[-2000:]
        print(rec["error"], file=sys.stderr)
    finally:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, indent=2))
    sys.exit(0 if rec["status"] == "PASS" else 1)


def run(args, rec):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    device = "cuda"
    torch.manual_seed(0)
    overrides = {}
    if args.config_override:
        for kv in args.config_override.split(","):
            k, v = kv.split("=", 1)
            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    pass
            overrides[k.strip()] = v
    cfg = AutoConfig.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, **overrides
    )
    arch = (cfg.architectures or ["?"])[0]
    rec["architecture"] = arch
    rec["model_type"] = getattr(cfg, "model_type", "?")

    load_kw = dict(
        dtype=torch.bfloat16,
        attn_implementation=args.attn,
        trust_remote_code=args.trust_remote_code,
        **overrides,
    )
    if args.qlora:
        from transformers import BitsAndBytesConfig

        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        # shard the (partially-quantized) base across GPUs when asked; LoRA has
        # no full-parameter grads, so naive model-parallel placement is enough
        load_kw["device_map"] = args.device_map if args.device_map else {"": 0}
        if args.device_map and args.max_gpu_mem:
            caps = args.max_gpu_mem.split(",")
            if len(caps) == 1:
                caps = caps * torch.cuda.device_count()
            load_kw["max_memory"] = {i: c for i, c in enumerate(caps)}
    elif args.device_map:
        load_kw["device_map"] = args.device_map
        max_memory = {}
        if args.max_gpu_mem:
            caps = args.max_gpu_mem.split(",")
            if len(caps) == 1:
                caps = caps * torch.cuda.device_count()
            max_memory = {i: c for i, c in enumerate(caps)}
        if args.cpu_offload:
            max_memory["cpu"] = args.cpu_offload
        if max_memory:
            load_kw["max_memory"] = max_memory

    t0 = time.time()
    if args.vl:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kw)
        if not (args.device_map or args.qlora):
            model = model.to(device)
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        batch = build_vl_batch(processor, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
        if not (args.device_map or args.qlora):
            model = model.to(device)
        tok = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        batch = build_text_batch(tok, device)
    rec["load_s"] = round(time.time() - t0, 1)
    rec["params_total_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6)

    if args.qlora or args.lora:
        from peft import LoraConfig, get_peft_model

        if args.qlora:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            # bf16 base stays frozen; let grads flow through gradient checkpointing
            model.enable_input_require_grads()
        # discover the linear module names to adapt (robust across peft versions
        # where the "all-linear" sentinel is mis-parsed into a char set). For MoE
        # models pass --lora-targets q_proj,k_proj,v_proj,o_proj to avoid adapting
        # the fused expert weights (their LoRA delta materializes at full size).
        if args.lora_targets and args.lora_targets.startswith("re:"):
            # raw regex (PEFT fullmatch on full module paths); use when the same
            # projection suffix (e.g. self_attn.q_proj) appears in both a trainable
            # tower and a frozen/unsupported one (custom Linear wrappers), so a plain
            # suffix list would match both. e.g. re:.*language_model.*\.(q|k|v|o)_proj
            linear_names = args.lora_targets[3:]
        elif args.lora_targets:
            linear_names = [s.strip() for s in args.lora_targets.split(",") if s.strip()]
        else:
            linear_names = sorted({
                n.split(".")[-1]
                for n, m in model.named_modules()
                if "Linear" in type(m).__name__ and n and "lm_head" not in n
            })
        lcfg = LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            bias="none", target_modules=linear_names,
        )
        model = get_peft_model(model, lcfg)

    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    if args.device_map and hasattr(model, "hf_device_map"):
        per_dev = {}
        for n, p in model.named_parameters():
            per_dev[str(p.device)] = per_dev.get(str(p.device), 0) + p.numel() * p.element_size()
        rec["weights_by_device_GiB"] = {k: round(v / 2**30, 2) for k, v in per_dev.items()}
        if not args.cpu_offload and any(d in ("meta", "cpu", "disk") for d in per_dev):
            raise RuntimeError(
                f"device_map offloaded weights off-GPU ({per_dev}); "
                "full-GPU fine-tune validation is impossible with this cap"
            )

    # probe pass: find params that actually receive grads
    out = model(**batch)
    out.loss.backward()
    if args.device_map:
        rec["alloc_after_probe_GiB"] = [
            round(torch.cuda.memory_allocated(i) / 2**30, 2)
            for i in range(torch.cuda.device_count())
        ]
    named = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    for n, p in model.named_parameters():
        if p.grad is None:
            p.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    rec["params_trained_M"] = round(sum(p.numel() for _, p in named) / 1e6)
    rec["param_ndims"] = sorted({p.ndim for _, p in named})

    import gefen

    if args.optimizer == "gefen":
        opt = gefen.Gefen(named, lr=args.lr)
    elif args.optimizer == "hybrid":
        muon_named, backup_named = gefen.split_params_for_muon(model)
        opt = gefen.GefenMuonHybrid(muon_named, backup_named, lr=args.lr)
    else:
        opt = torch.optim.AdamW([p for _, p in named], lr=args.lr)

    torch.cuda.reset_peak_memory_stats()
    losses = []
    t0 = time.time()
    for step in range(args.steps):
        out = model(**batch)
        loss = out.loss
        if not torch.isfinite(loss):
            rec["losses"] = losses
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        if args.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_([p for _, p in named], args.clip_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 4))
    rec["train_s"] = round(time.time() - t0, 1)
    rec["losses"] = losses[:3] + ["..."] + losses[-3:] if len(losses) > 8 else losses
    rec["loss_first"] = losses[0]
    rec["loss_last"] = losses[-1]
    rec["peak_vram_GiB"] = round(
        sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count()))
        / 2**30,
        2,
    )
    drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-9)
    rec["loss_drop_pct"] = round(100 * drop, 1)
    rec["status"] = "PASS" if drop > 0.3 else "FAIL"


if __name__ == "__main__":
    main()
