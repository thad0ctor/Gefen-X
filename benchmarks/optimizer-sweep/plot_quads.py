"""Per-model 2x2 quad charts: eval loss / throughput / peak VRAM / optimizer-state.
Gefen highlighted in orange; the best bar in each panel is outlined green + starred.
Data-driven: models and optimizers are discovered from the comparison_table.csv.

usage:
  python plot_quads.py --csv <out>/comparison_table.csv --out-dir <out>
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ORDER = ["adamw_bf16", "adamw8bit", "adamw4bit", "gefen_fused", "gefen_nonfused", "gefen_muon"]
LABEL = {"adamw_bf16": "AdamW\n(bf16)", "adamw8bit": "AdamW\n8-bit",
         "adamw4bit": "AdamW\n4-bit", "gefen_fused": "Gefen\n(fused)",
         "gefen_nonfused": "Gefen\n(non-fused)", "gefen_muon": "Gefen\nMuon"}
COLOR = {"adamw_bf16": "#9aa7b5", "adamw8bit": "#6b8cae", "adamw4bit": "#4a6fa5",
         "gefen_fused": "#e07b39", "gefen_nonfused": "#d68a4e", "gefen_muon": "#c25b1f"}

PANELS = [
    ("final_eval_loss", "Final eval loss", "loss", "lower"),
    ("tok_per_s", "Throughput", "tokens / sec", "higher"),
    ("peak_vram_gib", "Peak VRAM", "GiB", "lower"),
    ("opt_state_bpp", "Optimizer state", "bytes / param", "lower"),
]

CONFIG = ("Full fine-tune . bf16 master weights . grad-checkpointing . seq 2048 . "
          "micro-batch 1 . Alpaca (packed) . constant LR\n"
          "Each optimizer at its own fair LR . green outline / star = best in panel")


def best_idx(vals, direction):
    return (min if direction == "lower" else max)(range(len(vals)), key=lambda i: vals[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subtitle", default="",
                    help="optional extra footer line (e.g. hardware / date / LRs)")
    args = ap.parse_args()
    config_text = CONFIG + ("\n" + args.subtitle if args.subtitle else "")

    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        for k in ("final_eval_loss", "final_train_ema", "tok_per_s",
                  "peak_vram_gib", "opt_state_bpp"):
            r[k] = float(r[k])

    # group by the stable tag, not the display label (labels can collide)
    disp_of = {r["tag"]: r["model"] for r in rows}
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    os.makedirs(args.out_dir, exist_ok=True)

    for tag in tags:
        disp = disp_of[tag]
        mrows = {r["optimizer"]: r for r in rows if r["tag"] == tag}
        opts = [o for o in ORDER if o in mrows]
        lr = {o: mrows[o]["LR"] for o in opts}

        fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5))
        fig.suptitle(f"{disp} - optimizer comparison  (loss . speed . memory)",
                     fontsize=17, fontweight="bold", y=0.985)

        for ax, (key, title, unit, direction) in zip(axes.flat, PANELS):
            vals = [mrows[o][key] for o in opts]
            colors = [COLOR.get(o, "#888888") for o in opts]
            bi = best_idx(vals, direction)
            bars = ax.bar(range(len(opts)), vals, color=colors,
                          edgecolor="black", linewidth=0.6, width=0.66)
            bars[bi].set_edgecolor("#1a7a1a")
            bars[bi].set_linewidth(2.4)
            ax.set_xticks(range(len(opts)))
            ax.set_xticklabels([LABEL.get(o, o) for o in opts], fontsize=9.5)
            ax.set_ylabel(unit, fontsize=10)
            arrow = "lower is better" if direction == "lower" else "higher is better"
            ax.set_title(f"{title}   ({arrow})", fontsize=12, fontweight="bold")
            finite = [v for v in vals if math.isfinite(v)]
            vmax = max(finite) if finite else 1.0
            ax.set_ylim(0, vmax * 1.18)
            for i, (b, v) in enumerate(zip(bars, vals)):
                txt = (f"{v:.3f}" if key == "final_eval_loss"
                       else (f"{v:.0f}" if key == "tok_per_s" else f"{v:.2f}"))
                star = "  *" if i == bi else ""
                ax.text(b.get_x() + b.get_width() / 2, v + vmax * 0.015, txt + star,
                        ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold" if i == bi else "normal")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)

        handles = [Patch(facecolor=COLOR.get(o, "#888888"), edgecolor="black",
                         label=f"{LABEL.get(o, o).replace(chr(10), ' ')}  (lr {lr[o]})")
                   for o in opts]
        fig.legend(handles=handles, loc="lower center", ncol=len(opts), fontsize=9.5,
                   frameon=False, bbox_to_anchor=(0.5, 0.04))
        fig.text(0.5, 0.005, config_text, ha="center", va="bottom", fontsize=8.0, color="#333333")

        fig.tight_layout(rect=[0, 0.11, 1, 0.965])
        slug = tag.lower().replace("-", "_").replace(".", "p").replace("/", "_")
        out = os.path.join(args.out_dir, f"quad_{slug}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

    print("done")


if __name__ == "__main__":
    main()
