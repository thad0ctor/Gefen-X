"""Aggregate a results.jsonl into a Markdown + CSV comparison table.

Fully data-driven: model groups (tags) and optimizers are discovered from the
results file, so it works for any set of models/optimizers the sweep produced.
Last write wins if a (tag, opt) cell was re-run.

usage:
  python build_table.py --results <out>/results.jsonl --out-dir <out>
  python build_table.py --results r.jsonl --out-dir . --label qwen1p7b=Qwen3-1.7B
"""
import argparse
import csv
import json
import os
from collections import OrderedDict

OPT_ORDER = ["adamw_bf16", "adamw8bit", "adamw4bit",
             "gefen_fused", "gefen_nonfused", "gefen_muon"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results.jsonl from the sweep")
    ap.add_argument("--out-dir", required=True, help="dir to write comparison_table.{csv,md}")
    ap.add_argument("--label", action="append", default=[],
                    help="tag=Display name mapping, repeatable (default: use tag verbatim)")
    args = ap.parse_args()

    label_map = {}
    for kv in args.label:
        k, _, v = kv.partition("=")
        label_map[k] = v

    rows = []
    with open(args.results) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # discover tag order (first appearance) and present optimizers
    tag_order = list(OrderedDict((r["tag"], None) for r in rows))
    by = OrderedDict()
    for r in rows:
        by[(r["tag"], r["opt"])] = r

    def label(tag):
        return label_map.get(tag, tag)

    cols = ["model", "tag", "optimizer", "LR", "final_eval_loss", "final_train_ema",
            "tok_per_s", "peak_vram_gib", "peak_vram_reserved_gib", "opt_state_bpp"]

    csv_rows = []
    for tag in tag_order:
        for opt in OPT_ORDER:
            r = by.get((tag, opt))
            if not r:
                continue
            csv_rows.append([
                label(tag), tag, opt, f"{r['lr']:.0e}",
                r["final_eval"], r["final_train_ema"],
                r["tok_per_s"], r["peak_vram_gib"],
                r.get("peak_vram_reserved_gib", ""), r["opt_state_bpp"],
            ])

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(csv_rows)

    md = [
        "# Gefen fair-LR optimizer bench\n",
        "Each optimizer at its own fair LR. Full fine-tune, bf16 master weights, "
        "grad-checkpointing, seq 2048, micro-batch 1, Alpaca packed to 2048-tok blocks, "
        "constant LR, fixed seed, identical data order within a model.\n",
    ]
    hdr = ("| Model | Optimizer | LR | Final eval loss | Final train EMA | "
           "tok/s | Peak VRAM alloc (GiB) | Peak VRAM reserved (GiB) | Opt-state B/param |")
    md.append(hdr)
    md.append("|" + "---|" * 9)
    for tag in tag_order:
        for opt in OPT_ORDER:
            r = by.get((tag, opt))
            if not r:
                continue
            rsv = r.get("peak_vram_reserved_gib")
            rsv_s = f"{rsv:.2f}" if isinstance(rsv, (int, float)) else "-"
            md.append(f"| {label(tag)} | {opt} | {r['lr']:.0e} | "
                      f"{r['final_eval']:.4f} | {r['final_train_ema']:.4f} | "
                      f"{r['tok_per_s']:.0f} | {r['peak_vram_gib']:.2f} | "
                      f"{rsv_s} | {r['opt_state_bpp']:.3f} |")

    # headline: gefen_fused vs AdamW family per model
    md.append("\n## Headline\n")
    for tag in tag_order:
        gef = by.get((tag, "gefen_fused"))
        if not gef:
            continue
        adamws = [(o, by[(tag, o)]) for o in ("adamw_bf16", "adamw8bit", "adamw4bit")
                  if (tag, o) in by]
        best_adam = min(adamws, key=lambda kv: kv[1]["final_eval"]) if adamws else None
        md.append(f"\n### {label(tag)}\n")
        if best_adam:
            bo, br = best_adam
            gap = gef["final_eval"] - br["final_eval"]  # positive => Gefen worse
            md.append(f"- Loss: gefen_fused eval {gef['final_eval']:.4f} vs best AdamW "
                      f"({bo}) {br['final_eval']:.4f} -> gap {gap:+.4f} "
                      f"({'Gefen worse' if gap > 0 else 'Gefen better'}).")
        ref = by.get((tag, "adamw_bf16"))
        if ref:
            md.append(f"- Throughput: gefen_fused {gef['tok_per_s']:.0f} tok/s vs "
                      f"adamw_bf16 {ref['tok_per_s']:.0f} tok/s "
                      f"({gef['tok_per_s']/ref['tok_per_s']:.2f}x).")
            md.append(f"- Peak VRAM (allocated): gefen_fused {gef['peak_vram_gib']:.2f} GiB vs "
                      f"adamw_bf16 {ref['peak_vram_gib']:.2f} GiB "
                      f"(saves {ref['peak_vram_gib']-gef['peak_vram_gib']:.2f} GiB).")
        present = [o for o in ('adamw_bf16', 'adamw8bit', 'adamw4bit') if (tag, o) in by]
        md.append(f"- Opt-state: gefen_fused {gef['opt_state_bpp']:.2f} B/param vs "
                  + ", ".join(f"{o} {by[(tag, o)]['opt_state_bpp']:.2f}" for o in present)
                  + ".")

    md.append("\n## Caveats\n")
    md.append("- Single seed unless you swept several (n=1 does not estimate seed noise).")
    md.append("- LRs are each optimizer's own fair-sweep optimum at the sweep step budget; "
              "the long-run optimum may differ.")
    md.append("- tok/s and peak VRAM are LR-independent; compare within a model on one GPU type.")
    md.append("- opt-state B/param for adamw4bit counts the real packed codes/scale/qmap "
              "(torchao OptimState4bit), not the logical bf16 view.")
    md.append("- gefen_muon needs --muon-adjust match_rms_adamw to share the AdamW LR scale; "
              "Muon's Newton-Schulz orthogonalization makes it slower per step.")

    md_path = os.path.join(args.out_dir, "comparison_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")

    print("wrote", csv_path)
    print("wrote", md_path)
    print("\n".join(md))


if __name__ == "__main__":
    main()
