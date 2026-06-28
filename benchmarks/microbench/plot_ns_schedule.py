"""Render the tuned Newton-Schulz schedule microbench (bench_ns_schedule.py) into
a committed PNG: per-shape NS time on two GPUs + per-shape orthogonality.

  * speed panels (one per GPU): CUDA-event ms per NS call for std-5 / tuned-3 /
    tuned-4, grouped by Muon weight shape, annotated with the tuned-vs-std speedup.
  * quality panel (device-independent): orthogonality error ||G-I||_F per shape
    on a log axis (tuned-4 is orders of magnitude tighter on the non-square
    shapes), with std-5 as the reference.

usage:
  python plot_ns_schedule.py --json-3090 ns_schedule_3090.json \
      --json-blackwell ns_schedule_blackwell.json --out muon_ns_schedule.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SCHEDULES = ["std-5", "tuned-3", "tuned-4"]
COLORS = {"std-5": "#3f5e8c", "tuned-3": "#c25b1f", "tuned-4": "#2e8b57"}
SHORT = {"q/k/v/o 2048x2048": "q/k/v/o\n2048²",
         "kv 512x2048": "kv\n512×2048",
         "gate/up 6144x2048": "gate/up\n6144×2048",
         "down 2048x6144": "down\n2048×6144"}


def load(path):
    with open(path) as f:
        return json.load(f)


def speed_panel(ax, data, title):
    shapes = list(data["shapes"].keys())
    x = np.arange(len(shapes))
    w = 0.26
    for i, s in enumerate(SCHEDULES):
        vals = [data["shapes"][sh]["speed_ms"][s] for sh in shapes]
        ax.bar(x + (i - 1) * w, vals, w, label=s, color=COLORS[s])
    # speedup annotations above each tuned bar
    for j, sh in enumerate(shapes):
        b = data["shapes"][sh]["speed_ms"]["std-5"]
        for i, s in enumerate(("tuned-3", "tuned-4")):
            v = data["shapes"][sh]["speed_ms"][s]
            ax.text(x[j] + (SCHEDULES.index(s) - 1) * w, v,
                    f"{b / v:.2f}×", ha="center", va="bottom", fontsize=8,
                    rotation=90, color=COLORS[s])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(sh, sh) for sh in shapes], fontsize=8.5)
    ax.set_ylabel("ms per NS call  →  lower is better")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)


def quality_panel(ax, data, devname):
    shapes = list(data["shapes"].keys())
    x = np.arange(len(shapes))
    w = 0.26
    for i, s in enumerate(SCHEDULES):
        vals = [data["shapes"][sh]["quality"][s]["ortho_err_mean"] for sh in shapes]
        ax.bar(x + (i - 1) * w, vals, w, label=s, color=COLORS[s])
        for j, sh in enumerate(shapes):
            ax.text(x[j] + (i - 1) * w, vals[j], f"{vals[j]:.2g}",
                    ha="center", va="bottom", fontsize=7.5, rotation=90,
                    color=COLORS[s])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(sh, sh) for sh in shapes], fontsize=8.5)
    ax.set_ylabel("‖XᵀX−I‖_F  (log)  →  lower is better")
    ax.set_title(f"orthogonality quality (fp32 metric, {devname})",
                 fontsize=11, fontweight="bold")
    ax.grid(alpha=0.25, axis="y", which="both")
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-3090", required=True)
    ap.add_argument("--json-blackwell", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d3090 = load(args.json_3090)
    dbw = load(args.json_blackwell)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    speed_panel(axes[0], d3090, f"NS step time · {d3090['device']}")
    speed_panel(axes[1], dbw, f"NS step time · {dbw['device']}")
    quality_panel(axes[2], dbw, "GPU-independent")
    fig.suptitle("Gefen-Muon tuned Newton-Schulz schedules: tuned-3 (aggressive) "
                 "and tuned-4 (equal-quality) vs the standard 5-step quintic",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
