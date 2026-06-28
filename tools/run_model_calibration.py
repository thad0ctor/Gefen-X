"""Run LR calibration against real HuggingFace models (hardened).

Two reference modes:

  --reference adamw  (default, gold standard)
      Maintains a shadow AdamW fed the SAME gradients as Gefen and reports, per
      parameter, ratio = adamw_rms / gefen_rms -- the LR multiplier that makes
      each Gefen/Muon update match AdamW. Covers every path:
        * backup / plain-Gefen params: ratio ~= 1 iff Gefen is a faithful AdamW
          drop-in at the same LR.
        * muon params: empirical replacement for the analytic match_rms_adamw
          constant (reported as k, where the matching scale = k*sqrt(max(m,n));
          match_rms_adamw hardcodes k = 0.2).

  --reference self
      Cheaper relative calibration: equalize all groups' applied update RMS to a
      chosen group (no AdamW shadow). See calibrate_relative.

Hardening vs the first pass: varied real batches (a small built-in corpus,
cycled), more steps, and per-parameter-type aggregation so the result is a
per-type multiplier table, not just a single number.

Model paths are CLI args (no hardcoded machine paths). fused=False: bit-identical
to the fused CUDA path for the update math, and needs no kernel compile.
"""
from __future__ import annotations

import argparse
import itertools
import math
import statistics
from collections import defaultdict

import torch

from gefen import GefenMuonHybrid
from gefen.tools.lr_calibration import calibrate_relative, calibrate_vs_adamw


def classify_param_type(name):
    """Classify a parameter by its name into a coarse type (q_proj/.../norm/...).
    Self-contained here so the tooling has no dependency on optimizer internals."""
    n = (name or "").lower()
    for t in (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "qkv_proj", "gate_up_proj",
    ):
        if t in n:
            return t
    if "embed" in n or "wte" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    if "norm" in n or "layernorm" in n or ".ln" in n:
        return "norm"
    if "bias" in n:
        return "bias"
    return "other"


# Varied snippets (different domains/lengths) so the measured gradients are not a
# single memorized batch. Cycled across steps.
CORPUS = [
    "The Gefen optimizer reduces optimizer-state memory by quantizing momentum to "
    "eight bits using a Hessian-block-diagonal partitioning while keeping the AdamW recipe.",
    "In a distant orbit beyond Neptune, the probe unfurled its solar sail and began "
    "the long deceleration toward the icy moon, its instruments humming in the cold.",
    "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= hi:\n"
    "        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            return mid\n",
    "The plaintiff argues that the contract's indemnification clause survives "
    "termination, whereas the defendant contends it was superseded by the later amendment.",
    "Mix the flour, sugar, and baking soda, then fold in the melted butter and two "
    "eggs; bake at 180 degrees for twenty-five minutes until the top is golden brown.",
    "Mitochondria generate ATP through oxidative phosphorylation, coupling electron "
    "transport across the inner membrane to a proton gradient that drives ATP synthase.",
    "Quarterly revenue rose nine percent year over year, driven by cloud subscriptions, "
    "while operating margin contracted as the company invested heavily in data centers.",
    "She tightened the last bolt, wiped the grease from her hands, and listened as the "
    "old engine caught, sputtered, and finally settled into a steady, confident idle.",
]


def split_params_for_muon(model):
    muon, backup = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_hidden_matrix = (
            p.ndim == 2
            and not any(
                k in name.lower()
                for k in (
                    "embed", "wte", "lm_head", "patch", "vision", "visual",
                    "multi_modal", "mm_", "projector",
                )
            )
        )
        (muon if is_hidden_matrix else backup).append((name, p))
    return muon, backup


# Thin alias for readability at the call sites below.
param_type = classify_param_type


def load_model(path, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, low_cpu_mem_usage=True
        )
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            path, dtype=dtype, low_cpu_mem_usage=True
        )
    model.to(device)
    model.train()
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return tok, model


def make_corpus_closure(model, tok, device, seq):
    """A closure that cycles through CORPUS, a different snippet per step."""
    batches = []
    for text in CORPUS:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=seq)
        ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        attn = attn.to(device) if attn is not None else None
        batches.append((ids, attn))
    cyc = itertools.cycle(batches)

    def closure():
        ids, attn = next(cyc)
        out = model(input_ids=ids, attention_mask=attn, labels=ids)
        out.loss.backward()
        return out.loss.detach()

    return closure


def _summ(xs):
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return None
    return (statistics.median(xs), min(xs), max(xs), len(xs))


def run_one(path, device, dtype, steps, warmup, seq, reference):
    print(f"\n{'='*82}\nMODEL: {path}   (reference={reference})\n{'='*82}")
    dev = torch.device(device)
    tok, model = load_model(path, device, dtype)
    device_label = torch.cuda.get_device_name(dev) if dev.type == "cuda" else str(dev)
    print(f"  loaded {model.__class__.__name__} on {device_label}")
    muon, backup = split_params_for_muon(model)
    print(f"  muon matrices: {len(muon)}   backup params: {len(backup)}")
    opt = GefenMuonHybrid(
        muon, backup, lr=1e-3, weight_decay=0.1, fused=False, adjust_lr_fn=None
    )
    closure = make_corpus_closure(model, tok, device, seq)

    result = {"model": path.rstrip("/").split("/")[-1]}

    if reference == "adamw":
        recs = calibrate_vs_adamw(opt, closure, num_steps=steps, warmup=warmup)
        # Per-type aggregation.
        by_type_k = defaultdict(list)      # muon: empirical k vs 0.2
        by_type_ratio = defaultdict(list)  # all: adamw_rms/gefen_rms multiplier
        muon_k, backup_ratio = [], []
        for r in recs:
            t = param_type(r.name)
            by_type_ratio[t].append(r.ratio)
            if r.is_muon and r.sqrt_max:
                k = (r.active_ratio * r.ratio) / r.sqrt_max
                by_type_k[t].append(k)
                muon_k.append(k)
            elif not r.is_muon:
                backup_ratio.append(r.ratio)

        print("\n  PER-TYPE  (ratio = AdamW/Gefen LR multiplier to match AdamW)")
        print(f"    {'type':<14}{'kind':<7}{'n':>4}{'ratio med':>11}{'k_eff med':>11}")
        for t in sorted(by_type_ratio):
            s = _summ(by_type_ratio[t])
            ks = _summ(by_type_k[t]) if by_type_k[t] else None
            kind = "muon" if by_type_k[t] else "backup"
            kcol = f"{ks[0]:.3f}" if ks else "-"
            print(f"    {t:<14}{kind:<7}{s[3]:>4}{s[0]:>11.3f}{kcol:>11}")

        mk = _summ(muon_k)
        br = _summ(backup_ratio)
        if mk:
            print(
                f"\n  MUON     empirical k (match_rms_adamw assumes 0.2): "
                f"median={mk[0]:.3f}  range=[{mk[1]:.3f},{mk[2]:.3f}]  n={mk[3]}"
            )
        if br:
            print(
                f"  BACKUP   Gefen-vs-AdamW ratio (1.0 == faithful AdamW drop-in): "
                f"median={br[0]:.3f}  range=[{br[1]:.3f},{br[2]:.3f}]  n={br[3]}"
            )
        result["muon_k"] = mk
        result["backup_ratio"] = br
    else:
        cal = calibrate_relative(opt, closure, reference="gefen",
                                 num_steps=steps, warmup=warmup)
        print("\n  " + cal.summary())

    del opt, model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--reference", choices=["adamw", "self"], default="adamw")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    results = []
    for path in args.models:
        try:
            results.append(
                run_one(path, args.device, dtype, args.steps, args.warmup,
                        args.seq, args.reference)
            )
        except Exception as e:
            import traceback
            print(f"\n  !! FAILED {path}: {e}")
            traceback.print_exc()

    if args.reference == "adamw":
        print(f"\n{'#'*82}\nCROSS-MODEL SUMMARY\n{'#'*82}")
        print(f"  {'model':<24}{'muon k med':>12}{'backup ratio med':>18}")
        for r in results:
            mk = r.get("muon_k")
            br = r.get("backup_ratio")
            mkc = f"{mk[0]:.3f}" if mk else "-"
            brc = f"{br[0]:.3f}" if br else "-"
            print(f"  {r['model']:<24}{mkc:>12}{brc:>18}")
        print("\n  (muon k: match_rms_adamw assumes 0.2 | backup ratio: 1.0 == AdamW-faithful)")


if __name__ == "__main__":
    main()
