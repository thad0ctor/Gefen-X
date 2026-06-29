"""Two "faster Newton-Schulz" charts for the README (RTX 5090, Qwen3-0.6B):

  * muon_ns_loss.png       -- real-training eval loss vs step, AdamW + every
                              Muon NS variant (does any lever hurt loss?)
  * muon_ns_throughput.png -- full-step training throughput (tok/s) for every
                              Muon NS variant, with speedup vs default annotated

Variants are discovered from the committed stability logs
(docs/benchmarks/logs/ns_stability_*.log); each log's RESULT line carries the
ns_schedule / fp8 / steady-state s_per_step / final_eval. AdamW loss comes from
the committed optimizer-sweep log, sliced to the stability step budget.

All inputs are committed, so this is part of the normal plot flow:
  python plot_ns_levers.py
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STEP_RE = re.compile(r"step\s+(\d+)\s+train_ema\s+[0-9.]+\s+eval\s+([0-9.]+)")
EVAL0_RE = re.compile(r"step\s+0\s+eval_loss\s+([0-9.]+)")
RESULT_RE = re.compile(r"RESULT\s+(\{.*\})")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOGS = os.path.join(_REPO, "docs", "benchmarks", "logs")
_DOCS = os.path.join(_REPO, "docs", "benchmarks")

# (ns_schedule, fp8) -> short key / display label / color, in display order.
VARIANTS = [
    (("standard", False), "standard",   "Muon (default)", "#3f5e8c"),
    (("tuned3",   False), "tuned3",     "tuned3",         "#8c3f7a"),
    (("tuned4",   False), "tuned4",     "tuned4",         "#2e8b57"),
    (("standard", True),  "fp8",        "fp8",            "#c25b1f"),
    (("tuned3",   True),  "tuned3_fp8", "tuned3 + fp8",   "#b22222"),
    (("tuned4",   True),  "tuned4_fp8", "tuned4 + fp8",   "#1f8f8f"),
]
KEY = {sf: k for sf, k, _, _ in VARIANTS}
LABEL = {k: lab for _, k, lab, _ in VARIANTS}
COLOR = {k: c for _, k, _, c in VARIANTS}
ORDER = [k for _, k, _, _ in VARIANTS]
ADAMW_COLOR = "#9aa7b5"


def parse_log(path, maxstep=None):
    steps, ev, res = [], [], None
    e0 = None
    txt = open(path).read()
    for line in txt.splitlines():
        m0 = EVAL0_RE.search(line)
        if m0:
            e0 = float(m0.group(1))
        m = STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            ev.append(float(m.group(2)))
    rm = RESULT_RE.search(txt)
    if rm:
        try:
            res = json.loads(rm.group(1))
        except json.JSONDecodeError:
            pass
    if e0 is not None:
        steps, ev = [0] + steps, [e0] + ev
    if maxstep is not None:
        keep = [(s, e) for s, e in zip(steps, ev) if s <= maxstep]
        steps, ev = [s for s, _ in keep], [e for _, e in keep]
    return steps, ev, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=_LOGS)
    ap.add_argument("--out-dir", default=_DOCS)
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    # discover Muon variants from the stability logs (keyed by ns_schedule, fp8)
    variants = {}
    for path in glob.glob(os.path.join(args.logs_dir, "ns_stability_*.log")):
        s, ev, res = parse_log(path)
        if not res:
            continue
        key = KEY.get((res.get("ns_schedule", "standard"), bool(res.get("fp8"))))
        if key is None:
            continue
        tps = (res.get("seq", 2048) / res["s_per_step"]) if res.get("s_per_step") else None
        variants[key] = {"steps": s, "ev": ev, "final_eval": res.get("final_eval"),
                         "tps": tps}
    present = [k for k in ORDER if k in variants]

    # AdamW loss reference (sliced to the stability budget; GPU-independent)
    adamw = None
    ap_log = os.path.join(args.logs_dir, "qwen0p6b_adamw_bf16.log")
    if os.path.exists(ap_log):
        s, ev, _ = parse_log(ap_log, maxstep=args.steps)
        if s:
            adamw = {"steps": s, "ev": ev}

    dev = "RTX 5090"
    sub = f"Qwen3-0.6B · {dev} · {args.steps} steps · lr 5e-5"

    # ---------------- chart 1: loss ----------------
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    if adamw:
        ax.plot(adamw["steps"], adamw["ev"], ":^", ms=4, lw=1.8, color=ADAMW_COLOR,
                label=f"AdamW (bf16) -> {adamw['ev'][-1]:.3f}")
    for k in present:
        v = variants[k]
        if not v["steps"]:
            continue
        ls = "-o" if k == "standard" else "--s"
        ax.plot(v["steps"], v["ev"], ls, ms=4, lw=2.2, color=COLOR[k], alpha=0.9,
                label=f"{LABEL[k]} -> {v['ev'][-1]:.3f}")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out eval loss  ->  lower is better")
    ax.set_title("Gefen-Muon faster Newton-Schulz — real-training loss\n" + sub +
                 "  ·  do the opt-in NS levers hurt loss?",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, loc="upper right", title="color = optimizer  (final eval)",
              title_fontsize=9)
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "muon_ns_loss.png")
    fig.savefig(p1, dpi=140)
    print("wrote", p1)
    plt.close(fig)

    # ---------------- chart 2: throughput ----------------
    bars = [k for k in present if variants[k]["tps"]]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    base = variants["standard"]["tps"] if "standard" in variants else None
    xs = range(len(bars))
    ax.bar(xs, [variants[k]["tps"] for k in bars],
           color=[COLOR[k] for k in bars], edgecolor="black", linewidth=0.6, width=0.7)
    vmax = max(variants[k]["tps"] for k in bars)
    for i, k in enumerate(bars):
        t = variants[k]["tps"]
        lab = f"{t:.0f} tok/s"
        if base:
            lab += f"\n{t/base:.2f}× vs default"
        ax.text(i, t + vmax * 0.012, lab, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
    if base:
        ax.axhline(base, color="#888", lw=1.0, ls="--", zorder=0)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[k] for k in bars], fontsize=10)
    ax.set_ylabel("full-step training throughput (tokens/sec)  ->  higher is better")
    ax.set_ylim(0, vmax * 1.16)
    ax.set_title("Gefen-Muon faster Newton-Schulz — end-to-end throughput\n" + sub +
                 "  ·  full-step tok/s (NS + model fwd/bwd)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "muon_ns_throughput.png")
    fig.savefig(p2, dpi=140)
    print("wrote", p2)
    plt.close(fig)


if __name__ == "__main__":
    main()
