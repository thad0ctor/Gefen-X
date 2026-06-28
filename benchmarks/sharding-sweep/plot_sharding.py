"""Two plots for the FSDP2 sharded-Newton-Schulz benchmark: GefenMuon exact vs
approx sharded NS, across shard counts (world sizes).

  1. muon_shard_loss.png : held-out eval loss vs step, exact (solid) vs approx
     (dashed), colored by world size, with each run's final eval annotated.
  2. muon_shard_perf.png : avg training-step time (s/step) + peak VRAM, exact vs
     approx per world size, with the approx speedup and loss-penalty annotated.

Data-driven: reads <out>/results_fsdp2.jsonl (one RESULT line per cell, written
by fsdp2_convergence.py) for the authoritative final_eval / peak_vram, and the
per-cell logs in <out>/logs/ for the per-step eval curve and avg s/step. Each
(world_size, sharded_mode) cell is matched to its log either by the RESULT json
embedded in the log or by the log filename (both the run.sh scheme
`*_w<world>_<mode>.log` and the older `full_<N>gpu_<mode>.log` are understood).

usage:
  python plot_sharding.py --results <out>/results_fsdp2.jsonl \
                          --logs <out>/logs --out-dir <out>
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Per-world-size colors, assigned in ascending world order. The first two match
# the colors used for x2 / x4 in the original committed charts.
PALETTE = ["#3f5e8c", "#c25b1f", "#2e8b57", "#8c3f7a", "#7a6a3f", "#3f7a8c"]
LS = {"exact": "-", "approx": "--"}

STEP_RE = re.compile(
    r"step\s+(\d+)\s+train_ema\s+[0-9.]+\s+eval\s+([0-9.]+)\s+\(([0-9.]+)s/step\)")
RESULT_RE = re.compile(r"RESULT\s+(\{.*\})\s*$")
# Tolerant filename parser: `..._w4_approx.log` or `full_2gpu_exact.log`.
NAME_RE = re.compile(r"(?:_w|_)(\d+)(?:gpu)?_(exact|approx)\.log$")


def parse_log(path):
    """Return (steps, evals, last_sstep, embedded_result_or_None)."""
    steps, ev, sstep, res = [], [], None, None
    with open(path) as f:
        for line in f:
            m = STEP_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                ev.append(float(m.group(2)))
                sstep = float(m.group(3))
            rm = RESULT_RE.search(line)
            if rm:
                try:
                    res = json.loads(rm.group(1))
                except json.JSONDecodeError:
                    pass
    return steps, ev, sstep, res


def world_mode_of_log(path, res):
    if res and "world_size" in res and "sharded_mode" in res:
        return int(res["world_size"]), res["sharded_mode"]
    m = NAME_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1)), m.group(2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="results_fsdp2.jsonl (one RESULT line per cell)")
    ap.add_argument("--logs", required=True,
                    help="dir with per-cell convergence logs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default=None,
                    help="only plot rows with this tag (default: all)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Cells are keyed by (world_size, sharded_mode) only, so a results/logs dir
    # that mixes tags would collide. If --tag wasn't given and >1 tag is present,
    # default to one (and warn) so cells never silently overwrite each other.
    if args.tag is None:
        tags = set()
        with open(args.results) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    tags.add(json.loads(line).get("tag"))
                except json.JSONDecodeError:
                    pass
        tags.discard(None)
        if len(tags) > 1:
            args.tag = sorted(tags)[0]
            print(f"WARN: results mix tags {sorted(tags)}; plotting only "
                  f"{args.tag!r} (pass --tag to choose another)")

    # ---- authoritative per-cell results keyed by (world, mode) ----
    results = {}
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if args.tag and r.get("tag") != args.tag:
                continue
            results[(int(r["world_size"]), r["sharded_mode"])] = r

    # ---- per-cell curves + s/step from the logs, keyed by (world, mode) ----
    curves = {}
    for path in sorted(glob.glob(os.path.join(args.logs, "*.log"))):
        steps, ev, sstep, res = parse_log(path)
        if not steps:
            continue
        if args.tag and res and res.get("tag") != args.tag:
            continue
        key = world_mode_of_log(path, res)
        if key is None:
            continue
        curves[key] = {"steps": steps, "ev": ev, "sstep": sstep, "res": res}
        # logs carry a RESULT line too; use it to backfill missing results.jsonl
        if key not in results and res is not None:
            results[key] = res

    if not results and not curves:
        raise SystemExit("no data found in --results or --logs")

    worlds = sorted({w for (w, _) in set(results) | set(curves)})
    color_of = {w: PALETTE[i % len(PALETTE)] for i, w in enumerate(worlds)}

    any_row = next(iter(results.values()), None)
    tag = (any_row or {}).get("tag", "")
    steps_n = (any_row or {}).get("steps", "")
    lr = (any_row or {}).get("lr", "")
    model = {"qwen0p6b": "Qwen3-0.6B", "qwen1p7b": "Qwen3-1.7B"}.get(tag, tag)
    regime = (f"{model} · Alpaca · {steps_n} steps · lr {lr} "
              "· data replicated (only the NS approx differs)")

    def final_eval(world, mode):
        r = results.get((world, mode))
        if r is not None:
            return r["final_eval"]
        c = curves.get((world, mode))
        return c["ev"][-1] if c and c["ev"] else None

    def peak_vram(world, mode):
        r = results.get((world, mode))
        return r.get("peak_vram_gib") if r else None

    def sstep(world, mode):
        # Prefer the authoritative steady-state s/step from RESULT; the last eval
        # log line is only an approximation (and stale when steps % eval_every).
        r = results.get((world, mode))
        if r is not None and r.get("s_per_step") is not None:
            return r["s_per_step"]
        c = curves.get((world, mode))
        return c["sstep"] if c else None

    # ---------- 1) loss curves ----------
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    drew = False
    for world in worlds:
        for mode in ("exact", "approx"):
            c = curves.get((world, mode))
            if not c or not c["steps"]:
                continue
            drew = True
            color = color_of[world]
            solid = mode == "exact"
            fe = final_eval(world, mode)
            ax.plot(c["steps"], c["ev"], LS[mode], color=color,
                    lw=2.4 if solid else 2.0,
                    marker="o" if solid else None, ms=3,
                    alpha=1.0 if solid else 0.85,
                    label=f"{mode} x{world}"
                          + (f"  -> {fe:.3f}" if fe is not None else ""))
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out eval loss  ->  lower is better")
    ax.set_title("Gefen-Muon under FSDP2: exact vs approx sharded Newton-Schulz\n"
                 + regime, fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    if drew:
        ax.legend(title="solid = exact (parity)   dashed = approx (non-parity)"
                        "   -> final eval",
                  fontsize=9.5, title_fontsize=9.5, loc="upper right")
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "muon_shard_loss.png")
    fig.savefig(p1, dpi=140)
    print("wrote", p1)
    plt.close(fig)

    # ---------- 2) performance (step-time + VRAM bars per world) ----------
    groups = [w for w in worlds
              if sstep(w, "exact") is not None and sstep(w, "approx") is not None]
    if groups:
        ex = [sstep(w, "exact") for w in groups]
        ap = [sstep(w, "approx") for w in groups]
        ex_v = [peak_vram(w, "exact") for w in groups]
        ap_v = [peak_vram(w, "approx") for w in groups]
        x = range(len(groups))
        w = 0.36
        fig, ax = plt.subplots(figsize=(9.0, 6.0))
        ax.bar([i - w / 2 for i in x], ex, w, label="exact (parity)",
               color="#3f5e8c")
        ax.bar([i + w / 2 for i in x], ap, w, label="approx (non-parity)",
               color="#c25b1f")
        for i, world in enumerate(groups):
            ev = f"\n{ex_v[i]:.2f} GiB" if ex_v[i] is not None else ""
            av = f"\n{ap_v[i]:.2f} GiB" if ap_v[i] is not None else ""
            ax.text(i - w / 2, ex[i] + 0.02, f"{ex[i]:.2f}s{ev}",
                    ha="center", va="bottom", fontsize=9)
            ax.text(i + w / 2, ap[i] + 0.02, f"{ap[i]:.2f}s{av}",
                    ha="center", va="bottom", fontsize=9)
            spd = ex[i] / ap[i]
            fe_ex, fe_ap = final_eval(world, "exact"), final_eval(world, "approx")
            ann = f"approx: {spd:.2f}x faster"
            if fe_ex is not None and fe_ap is not None:
                ann += f"\n{fe_ap - fe_ex:+.3f} eval loss"
            ax.text(i, max(ex[i], ap[i]) + 0.20, ann, ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#444")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{g} GPUs" for g in groups])
        ax.set_ylabel("avg training-step time (s)  ->  lower is better")
        ax.set_ylim(0, max(ex + ap) + 0.5)
        ax.set_title("Gefen-Muon under FSDP2: the sharded_mode speed / quality trade\n"
                     + regime, fontsize=11.5, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)
        ax.legend(fontsize=10, loc="upper left")
        fig.tight_layout()
        p2 = os.path.join(args.out_dir, "muon_shard_perf.png")
        fig.savefig(p2, dpi=140)
        print("wrote", p2)
        plt.close(fig)
    else:
        print("skip muon_shard_perf.png: need both exact+approx s/step per world")
    print("done")


if __name__ == "__main__":
    main()
