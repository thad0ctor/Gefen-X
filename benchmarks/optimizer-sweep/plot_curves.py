"""Per-model loss curves + a throughput-vs-VRAM efficiency scatter.

  1. loss curve: held-out eval loss vs step (solid) + train-EMA (dashed), per optimizer
  2. throughput-vs-VRAM scatter: tok/s vs peak VRAM (upper-left = best)

Reads the per-cell logs (<tag>_<opt>.log) + comparison_table.csv. Data-driven:
models/optimizers are discovered from the CSV; logs are matched by the `tag` column.

usage:
  python plot_curves.py --csv <out>/comparison_table.csv --logs <out>/logs --out-dir <out>
"""
import argparse
import csv
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ORDER = ["adamw_bf16", "adamw8bit", "adamw4bit", "gefen_fused", "gefen_nonfused", "gefen_muon"]
NAME = {"adamw_bf16": "AdamW (bf16)", "adamw8bit": "AdamW 8-bit", "adamw4bit": "AdamW 4-bit",
        "gefen_fused": "Gefen (fused)", "gefen_nonfused": "Gefen (non-fused)",
        "gefen_muon": "Gefen Muon"}
COLOR = {"adamw_bf16": "#9aa7b5", "adamw8bit": "#6b8cae", "adamw4bit": "#3f5e8c",
         "gefen_fused": "#e07b39", "gefen_nonfused": "#d68a4e", "gefen_muon": "#c25b1f"}

CONFIG = ("Full fine-tune . bf16 master . grad-checkpointing . seq 2048 . mbs 1 . "
          "Alpaca (packed) . constant LR . each optimizer at its own fair LR")


def parse_log(path):
    steps, ev, ema = [], [], []
    for line in open(path):
        m = re.match(r"step\s+(\d+)\s+.*?eval(?:_loss)?\s+([0-9.]+)", line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        ev.append(float(m.group(2)))
        me = re.search(r"train_ema\s+([0-9.]+)", line)
        ema.append(float(me.group(1)) if me else None)
    return steps, ev, ema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--logs", required=True, help="dir with <tag>_<opt>.log files")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subtitle", default="",
                    help="optional extra footer line (e.g. hardware / date / LRs)")
    args = ap.parse_args()
    config_text = CONFIG + ("\n" + args.subtitle if args.subtitle else "")

    rows = list(csv.DictReader(open(args.csv)))
    # key off the stable tag, not the display label (labels can collide)
    by_tag_opt = {(r["tag"], r["optimizer"]): r for r in rows}
    disp_of = {r["tag"]: r["model"] for r in rows}
    lr_of = {(r["tag"], r["optimizer"]): r["LR"] for r in rows}
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    os.makedirs(args.out_dir, exist_ok=True)

    for tag in tags:
        disp = disp_of[tag]
        opts = [o for o in ORDER if (tag, o) in by_tag_opt]
        slug = tag.lower().replace("-", "_").replace(".", "p").replace("/", "_")

        # ---- 1) loss curves ----
        fig, ax = plt.subplots(figsize=(9.5, 6.2))
        drew = False
        for o in opts:
            p = os.path.join(args.logs, f"{tag}_{o}.log")
            if not os.path.exists(p):
                continue
            s, ev, ema = parse_log(p)
            if not s:
                continue
            drew = True
            lw = 2.6 if o.startswith("gefen") else 1.8
            ax.plot(s, ev, "-o", color=COLOR.get(o, "#888"), lw=lw, ms=3.5,
                    label=f"{NAME.get(o, o)}  (lr {lr_of[(tag, o)]})  -> {ev[-1]:.3f}")
            xe = [a for a, b in zip(s, ema) if b is not None]
            ye = [b for b in ema if b is not None]
            ax.plot(xe, ye, "--", color=COLOR.get(o, "#888"), lw=1.0, alpha=0.45)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.set_title(f"{disp} - eval-loss curves (solid = held-out eval, dashed = train EMA)\n"
                     "lower is better . each optimizer at its own fair LR",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        if drew:
            leg1 = ax.legend(fontsize=9, title="color = optimizer  (final eval)",
                             title_fontsize=9, loc="upper right")
            ax.add_artist(leg1)
            style_handles = [
                Line2D([0], [0], color="#444", lw=2.2, ls="-", marker="o", ms=4,
                       label="solid = held-out eval loss"),
                Line2D([0], [0], color="#444", lw=1.2, ls="--", alpha=0.7,
                       label="dashed = train-loss EMA"),
            ]
            ax.legend(handles=style_handles, fontsize=8.5, title="line style",
                      title_fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.66))
        fig.text(0.5, 0.012, config_text, ha="center", fontsize=7.6, color="#333")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out = os.path.join(args.out_dir, f"loss_curve_{slug}.png")
        fig.savefig(out, dpi=140)
        print("wrote", out)
        plt.close(fig)

        # ---- 2) throughput vs VRAM ----
        fig, ax = plt.subplots(figsize=(9.0, 6.4))
        xs, ys = [], []
        for o in opts:
            r = by_tag_opt.get((tag, o))
            if not r:
                continue
            x = float(r["peak_vram_gib"])
            y = float(r["tok_per_s"])
            xs.append(x)
            ys.append(y)
            big = o.startswith("gefen")
            ax.scatter(x, y, s=320 if big else 200, color=COLOR.get(o, "#888"),
                       edgecolor="black", linewidth=1.4, zorder=3,
                       marker="*" if big else "o")
            ax.annotate(f"{NAME.get(o, o)}\n{y:.0f} tok/s . {x:.1f} GiB . "
                        f"{r['opt_state_bpp']} B/param",
                        (x, y), textcoords="offset points", xytext=(9, 6), fontsize=8.5,
                        fontweight="bold" if big else "normal")
        ax.set_xlabel("peak VRAM (GiB)  ->  less is better")
        ax.set_ylabel("throughput (tokens/sec)  ->  more is better")
        ax.set_title(f"{disp} - throughput vs peak VRAM  (upper-left = faster + less memory)",
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        fxs = [v for v in xs if math.isfinite(v)]
        fys = [v for v in ys if math.isfinite(v)]
        if fxs:
            ax.set_xlim(min(fxs) - 1.2, max(fxs) + 2.8)
        if fys:
            ax.set_ylim(min(fys) * 0.94, max(fys) * 1.06)
        fig.text(0.5, 0.012, config_text, ha="center", fontsize=7.6, color="#333")
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out = os.path.join(args.out_dir, f"tput_vram_{slug}.png")
        fig.savefig(out, dpi=140)
        print("wrote", out)
        plt.close(fig)

    print("done")


if __name__ == "__main__":
    main()
