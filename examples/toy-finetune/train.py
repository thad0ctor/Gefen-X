"""Toy full fine-tune: teach a small chat model the [GEFEN_REPLY] format.

A plain-PyTorch training loop (no trainer framework) meant to be read and
reused for your own fine-tuning runs. Loss is computed only on the assistant
reply (prompt tokens are masked to -100), exactly matching inference. Swap
`--optimizer` between `gefen`, `hybrid`, and `adamw` to compare peak VRAM and
optimizer-state size on an otherwise identical run.

Usage:
  python train.py [--model Qwen/Qwen3-0.6B] [--optimizer gefen|hybrid|adamw]
                  [--data data/train.jsonl] [--out-dir runs/gefen]
                  [--epochs 1] [--batch-size 8] [--lr <per-optimizer default>]
                  [--lora] [--grad-checkpoint]
"""

import argparse
import json
import math
import os
import random
import time

import torch

# Gefen's published sweeps run AdamW at 5e-5; Gefen's fair LR is 0.6x AdamW
# (factored_v_2d default), and the Muon hybrid reuses the AdamW LR via its
# match_rms_adamw default.
DEFAULT_LR = {"adamw": 5e-5, "gefen": 3e-5, "hybrid": 5e-5}


def apply_template(tok, messages, add_generation_prompt):
    """Chat-template to a flat list of token ids; enable_thinking=False where supported (Qwen3)."""
    try:
        out = tok.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt,
            tokenize=True, enable_thinking=False,
        )
    except TypeError:
        out = tok.apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, tokenize=True,
        )
    if not isinstance(out, list):  # transformers v5 returns a BatchEncoding
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return out


def load_examples(path, tok, seq_len):
    """Tokenize each record into (input_ids, labels); prompt tokens are masked."""
    examples, skipped = [], 0
    with open(path) as f:
        for line in f:
            messages = json.loads(line)["messages"]
            prompt_ids = apply_template(tok, messages[:-1], add_generation_prompt=True)
            full_ids = apply_template(tok, messages, add_generation_prompt=False)
            if full_ids[: len(prompt_ids)] == prompt_ids and len(full_ids) > len(prompt_ids):
                # reply as the template itself encodes it, terminator included
                reply_ids = full_ids[len(prompt_ids):]
            else:
                # template renders history differently from the generation prompt:
                # fall back to raw reply text + EOS
                reply_ids = tok(messages[-1]["content"], add_special_tokens=False).input_ids
                reply_ids = [*reply_ids, tok.eos_token_id]
            if len(prompt_ids) + len(reply_ids) > seq_len:
                skipped += 1
                continue
            input_ids = prompt_ids + reply_ids
            labels = [-100] * len(prompt_ids) + reply_ids
            examples.append((input_ids, labels))
    if skipped:
        print(f"skipped {skipped} examples longer than --seq-len {seq_len}")
    return examples


def make_batches(examples, batch_size, rng):
    """Length-sorted batches (less padding), then shuffled batch order."""
    order = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(examples, idxs, pad_id, device):
    rows = [examples[i] for i in idxs]
    width = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for r, (ids, labs) in enumerate(rows):
        input_ids[r, : len(ids)] = torch.tensor(ids)
        labels[r, : len(labs)] = torch.tensor(labs)
        attention_mask[r, : len(ids)] = 1
    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "attention_mask": attention_mask.to(device),
    }


def optimizer_state_bytes(opt):
    seen, total = set(), 0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v) and v.data_ptr() not in seen:
                seen.add(v.data_ptr())
                total += v.numel() * v.element_size()
    return total


def lr_lambda(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--optimizer", default="gefen", choices=["gefen", "hybrid", "adamw"])
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out-dir", default=None, help="default: runs/<optimizer>")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=None, help=f"default per optimizer: {DEFAULT_LR}")
    ap.add_argument("--seq-len", type=int, default=768)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--clip-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora", action="store_true", help="train LoRA adapters instead of all weights")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lr = args.lr if args.lr is not None else DEFAULT_LR[args.optimizer]
    out_dir = args.out_dir or os.path.join("runs", args.optimizer + ("-lora" if args.lora else ""))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device)

    if args.lora:
        from peft import LoraConfig, get_peft_model

        linear_names = sorted({
            n.split(".")[-1]
            for n, m in model.named_modules()
            if "Linear" in type(m).__name__ and n and "lm_head" not in n
        })
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0,
            bias="none", target_modules=linear_names,
        ))

    model.train()
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    examples = load_examples(args.data, tok, args.seq_len)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in named]
    print(f"{len(examples)} examples | training {sum(p.numel() for p in params) / 1e6:.0f}M params "
          f"with {args.optimizer} @ lr {lr:g}")

    import gefen

    if args.optimizer == "gefen":
        opt = gefen.Gefen(named, lr=lr)
    elif args.optimizer == "hybrid":
        # recommended config: backup (1D/embedding) half at 0.5x the Muon LR
        muon_named, backup_named = gefen.split_params_for_muon(model)
        opt = gefen.GefenMuonHybrid(muon_named, backup_named, lr=lr, backup_lr=0.5 * lr)
    else:
        opt = torch.optim.AdamW(params, lr=lr)
    base_lrs = [group["lr"] for group in opt.param_groups]

    # per-example counts, computed once: loss-bearing (reply) tokens and total tokens
    label_counts = [sum(1 for t in labs if t != -100) for _, labs in examples]

    steps_per_epoch = math.ceil(len(examples) / args.batch_size / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step, tokens_seen, t0 = 0, 0, time.time()
    recent = []
    for epoch in range(args.epochs):
        batches = make_batches(examples, args.batch_size, rng)
        for i in range(0, len(batches), args.grad_accum):
            micro = batches[i : i + args.grad_accum]
            # weight each micro-batch by its share of the group's reply tokens so
            # the accumulated gradient equals one large batch (a plain 1/len(micro)
            # would over-weight micro-batches with fewer target tokens)
            group_counts = [sum(label_counts[j] for j in idxs) for idxs in micro]
            group_total = sum(group_counts)
            for idxs, n_labels in zip(micro, group_counts):
                batch = collate(examples, idxs, tok.pad_token_id, device)
                loss = model(**batch).loss
                (loss * (n_labels / group_total)).backward()
                tokens_seen += sum(len(examples[j][0]) for j in idxs)
                recent.append(loss.detach())  # .item() here would sync the GPU every micro-batch
            if args.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad_norm)
            scale = lr_lambda(step, args.warmup_steps, total_steps)
            for group, base in zip(opt.param_groups, base_lrs):
                group["lr"] = base * scale
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0 or step == total_steps:
                recent = recent[-25 * args.grad_accum:]
                print(f"epoch {epoch + 1} step {step}/{total_steps} "
                      f"loss {torch.stack(recent).mean().item():.4f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} "
                      f"tok/s {tokens_seen / (time.time() - t0):.0f}")

    summary = {
        "model": args.model,
        "optimizer": args.optimizer,
        "lora": args.lora,
        "lr": lr,
        "epochs": args.epochs,
        "steps": step,
        "final_loss": round(torch.stack(recent).mean().item(), 4),
        "train_time_s": round(time.time() - t0, 1),
        "tokens_per_s": round(tokens_seen / (time.time() - t0)),
        "peak_vram_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device == "cuda" else 0.0,
        "optimizer_state_GiB": round(optimizer_state_bytes(opt) / 2**30, 3),
    }
    print(json.dumps(summary, indent=2))

    if args.lora:
        model = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved fine-tuned model -> {out_dir}")


if __name__ == "__main__":
    main()
