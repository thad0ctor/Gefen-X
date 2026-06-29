"""Render the fp8 Newton-Schulz microbench (bench_ns_fp8.py) into a committed PNG.

  * speed panel: CUDA-event ms per NS call, bf16 (eager production path) vs fp8
    (torch.compile-fused), grouped by Muon weight shape, with the fp8 speedup
    annotated. Shapes the size-gate sends back to bf16 (min-dim < FP8_MIN_DIM)
    are hatched and labelled "gated→bf16".
  * quality panel: orthogonality error ||G-I||_F (fp32 metric) for the fp32
    reference, bf16, and fp8 outputs per shape, with the fp8-vs-bf16 relative
    output deviation annotated.

usage:
  python plot_ns_fp8.py --json ns_fp8_5090.json --out muon_ns_fp8.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SHORT = {"q/k/v/o 2048x2048": "q/k/v/o\n2048²",
         "kv 512x2048": "kv\n512×2048",
         "gate/up 6144x2048": "gate/up\n6144×2048",
         "down 2048x6144": "down\n2048×6144"}
C_BF16 = "#3f5e8c"
C_FP8 = "#c25b1f"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.json) as f:
        d = json.load(f)
    shapes = list(d["shapes"].keys())
    dev = d["device"]
    x = np.arange(len(shapes))
    w = 0.38

    fig, (axs, axq) = plt.subplots(1, 2, figsize=(14, 5.4))

    # ---- speed ----
    bf = [d["shapes"][s]["speed_ms"]["bf16"] for s in shapes]
    fp = [d["shapes"][s]["speed_ms"]["fp8"] for s in shapes]
    gated = [d["shapes"][s]["gated_off"] for s in shapes]
    axs.bar(x - w / 2, bf, w, label="bf16 (eager)", color=C_BF16)
    bars = axs.bar(x + w / 2, fp, w, label="fp8 (compiled)", color=C_FP8)
    for j, s in enumerate(shapes):
        if gated[j]:
            bars[j].set_hatch("//")
            bars[j].set_alpha(0.55)
        spd = bf[j] / fp[j]
        lab = "gated→bf16" if gated[j] else f"{spd:.2f}×"
        axs.text(x[j] + w / 2, fp[j], lab, ha="center", va="bottom",
                 fontsize=9, fontweight="bold",
                 color="#888" if gated[j] else "#1a6b1a" if spd >= 1 else "#a33")
    axs.set_xticks(x)
    axs.set_xticklabels([SHORT.get(s, s) for s in shapes], fontsize=8.5)
    axs.set_ylabel("ms per NS call  →  lower is better")
    axs.set_title(f"NS step time: fp8 vs bf16 · {dev}", fontsize=11,
                  fontweight="bold")
    axs.grid(alpha=0.25, axis="y")
    axs.set_axisbelow(True)
    axs.legend(fontsize=9, loc="upper left")

    # ---- quality ----
    ww = 0.26
    for i, (k, col) in enumerate((("fp32", "#777"), ("bf16", C_BF16),
                                  ("fp8", C_FP8))):
        vals = [d["shapes"][s]["ortho_err"][k] for s in shapes]
        axq.bar(x + (i - 1) * ww, vals, ww, label=k, color=col)
    for j, s in enumerate(shapes):
        rel = d["shapes"][s]["rel_fp8_bf16"] * 100
        top = max(d["shapes"][s]["ortho_err"].values())
        note = "—" if gated[j] else f"rel {rel:.1f}%"
        axq.text(x[j], top, note, ha="center", va="bottom", fontsize=8.5,
                 fontweight="bold", color="#444")
    axq.set_xticks(x)
    axq.set_xticklabels([SHORT.get(s, s) for s in shapes], fontsize=8.5)
    axq.set_ylabel("‖XᵀX−I‖_F  →  lower is better")
    axq.set_title("orthogonality: fp32 ref vs bf16 vs fp8\n"
                  "(rel = ‖O_fp8−O_bf16‖/‖O_bf16‖)", fontsize=11,
                  fontweight="bold")
    axq.grid(alpha=0.25, axis="y")
    axq.set_axisbelow(True)
    axq.legend(fontsize=9)

    min_dim = d.get("fp8_min_dim", "?")
    fig.suptitle("Gefen-Muon fp8 (e4m3) Newton-Schulz: opt-in, arch+size-gated "
                 f"(sm_89+, min-dim ≥ {min_dim})",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=140)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
