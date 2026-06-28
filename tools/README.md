# Gefen LR tooling — finder + diagnostics

Standalone, non-invasive tools for setting and understanding the learning rate for the Gefen optimizer family (`Gefen`, `GefenMuon`, `GefenMuonHybrid`). Nothing here changes the optimizer's training math — each entry point either *measures* a short dry run or *recommends* a number you choose to apply. Safe to run against live setups; weights are snapshotted and restored.

## `find_lr` — the recommended way to set Gefen's LR

Instead of guessing a factor versus your AdamW LR, find Gefen's optimal LR empirically:

```python
from gefen import Gefen
from gefen.tools.find_lr import find_lr

res = find_lr(model, train_blocks, optimizer="gefen",
              eval_blocks=eval_blocks, method="sweep")   # res.lr is the recommendation
opt = Gefen(named_params, lr=res.lr, fused=True)
```

CLI: `python -m gefen.tools.find_lr --model <path> --optimizer gefen|gefen_muon|hybrid|adamw --method sweep`

Two modes:
- **`method="sweep"` (recommended, trustworthy):** runs short real fine-tunes over an LR grid and picks the lowest **held-out eval** loss. This is the gold standard.
- **`method="range_test"` (fast, rough):** a Leslie-Smith LR ramp. Cheap, but **only a bracketing aid** — across models it was run-to-run noisy and sometimes several-× off for the Gefen family. Use it to center the sweep grid, not as a final number.

### What the finder shows about Gefen LRs (RTX 5090 / Alpaca, 3 models)
- **Plain `Gefen`** wants a **low, roughly model-independent LR ≈ 1e-5**, i.e. **~0.1–0.4× your AdamW LR** (lower). The static "≈0.6–0.8×" head-dim heuristic overestimates this and varies by model — prefer the finder.
- **`GefenMuonHybrid`** roughly **reuses your AdamW LR** (~1×), with `adjust_lr_fn="match_rms_adamw"`.

## Diagnostic probes

| entry point | what it measures |
|---|---|
| `lr_range_test` | LR-vs-loss ramp; suggested LR at steepest descent (also the basis of `find_lr` range_test) |
| `calibrate_relative` | per-group LR multipliers that equalize *applied update RMS* to a reference group |
| `calibrate_vs_adamw` | per-parameter Gefen-vs-AdamW update-RMS ratio, via a shadow AdamW fed identical gradients |
| `run_model_calibration.py` | runner that loads a HF model and reports the above per parameter-type (paths via CLI) |

These are diagnostics for *understanding* update magnitudes — useful, but see the finding below before treating any magnitude ratio as an LR prescription.

## Honest finding: matching AdamW *magnitude* is the wrong objective for Muon

`calibrate_vs_adamw` shows that `match_rms_adamw`'s flat `k=0.2` produces Muon updates **below** AdamW's magnitude (q/o_proj ~2.2× under, median ~1.3×). It is tempting to "fix" this by scaling the Muon updates up to match AdamW. We tried that (a per-type prior and an auto-calibration window) and validated it on real-dataset fine-tuning (tatsu-lab/alpaca, held-out eval): **it did not help and slightly hurt** — the original `match_rms_adamw` (k=0.2) tracked/beat AdamW, while the magnitude-matched variants were worse on train and eval.

Conclusion: the *measurement* is correct, but Muon's benefit is the orthogonalized update **direction**, not magnitude parity. So **`match_rms_adamw` stays the recommended default** for the hybrid, and the magnitude-matching machinery is intentionally **not shipped** (only the diagnostic probes that revealed this remain here). It may matter for heavier/from-scratch training — untested.

## Tests
CPU smoke (no CUDA needed): `tests/test_find_lr.py` (range_test across all four families + non-destructive restore) and `tests/test_lr_calibration.py` (the probes). The real-model finder validation runs on GPU.
