"""Learning-rate calibration probes for Gefen / GefenMuon / GefenMuonHybrid.

Why this exists
---------------
The Gefen family applies *different per-update scaling* depending on the path a
parameter takes:

  * plain ``Gefen`` (AdamW-like): update = m_hat / (sqrt(vmean_hat) + eps), where
    ``vmean`` is a **block-mean** of grad^2 (Gefen's partitioning), not AdamW's
    per-element second moment.
  * ``GefenMuon``: the update is Newton-Schulz orthogonalized (singular values
    ~1), then multiplied by ``_adjust_lr``'s shape ratio:
        - None / "original":   sqrt(max(1, rows/cols))     (Muon-native)
        - "match_rms_adamw":   0.2 * sqrt(max(rows, cols))  (analytic AdamW-RMS match)

``GefenMuonHybrid`` hands a **single** base LR to both halves. So whether one LR
is correct for the whole model depends entirely on whether the per-path update
*magnitudes* line up. The analytic ``match_rms_adamw`` constant (0.2 = assumed
AdamW update RMS) ignores both the block-mean denominator and the 8-bit momentum
quantization, so it is an approximation of unknown tightness on this codebase.

This module answers the two *separable* questions empirically:

  1. RELATIVE calibration (``calibrate_relative``): given one base LR, what
     per-group multiplier makes every group's applied update RMS match a chosen
     reference group? This is the empirical analogue of ``match_rms_adamw`` --
     it measures the real update RMS (block-mean + quantization included) instead
     of assuming a constant. Use it to decide whether the analytic flag is good
     enough or whether a residual correction is needed.

  2. ABSOLUTE LR (``lr_range_test``): an LR-range test (Leslie Smith / fastai
     "lr_find"). There is no way to read the right base LR off the model alone;
     this ramps the LR over a few hundred steps and reports the loss-vs-LR curve
     plus a suggested base LR at the steepest-descent point.

Everything here is non-invasive: ``measure_update_rms`` snapshots params around
``optimizer.step()`` and restores any settings it touches; it does not change
optimizer state. It therefore composes with checkpoint/resume and (best-effort)
FSDP2/DTensor, since it reads ``.to_local()`` shards.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch


def _shadow_adamw_unit_update(m, v, grad, t, beta1, beta2, eps):
    """One step of a SHADOW AdamW kept alongside the real optimizer.

    Maintains its own (m, v), fed the SAME gradient the real optimizer sees, but
    its update is never applied -- it only provides the canonical AdamW update
    magnitude on the very gradients the model produces. Mutates m/v in place and
    returns the bias-corrected update direction at *unit* LR; multiply by the
    group LR to put it on the same applied-delta basis as a real step.
    """
    m.mul_(beta1).add_(grad, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    mhat = m / (1 - beta1 ** t)
    vhat = v / (1 - beta2 ** t)
    return mhat / (vhat.sqrt() + eps)


# --------------------------------------------------------------------------- #
# Group identification / introspection
# --------------------------------------------------------------------------- #

def _is_muon_group(group: dict) -> bool:
    """A GefenMuon param group carries Newton-Schulz coefficients; a plain Gefen
    (AdamW-like) group does not. This is the stable structural marker exposed
    through ``GefenMuonHybrid.param_groups``."""
    return "ns_coefficients" in group or "adjust_lr_fn" in group


def _group_label(idx: int, group: dict) -> str:
    name = group.get("name")
    kind = "muon" if _is_muon_group(group) else "gefen"
    if name is not None and name != "none":
        return f"{idx}:{kind}:{name}"
    return f"{idx}:{kind}"


def _local(t: torch.Tensor) -> torch.Tensor:
    """Return the rank-local storage for DTensor/FSDP2 params, else the tensor.

    Calibration is a magnitude measurement, so reading the local shard is fine:
    the per-element update RMS of a shard estimates the full tensor's RMS.
    """
    if hasattr(t, "to_local") and hasattr(t, "placements"):
        return t.to_local()
    return t


def _rms(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(t.detach().float().pow(2).mean().sqrt().item())


# --------------------------------------------------------------------------- #
# Non-destructive probing: snapshot + restore params AND optimizer state
# --------------------------------------------------------------------------- #
# These probes call ``optimizer.step()`` to MEASURE the real update, which would
# otherwise advance the model's weights and the optimizer's state (step counters,
# 8-bit momentum codebooks, vmean, the learned codebook, the global step). To keep
# the documented "safe to run against a live training job" contract, we snapshot
# the params (to CPU, to bound GPU memory) and a deep copy of the optimizer's
# state_dict before the probe and restore both in a ``finally``.
def _snapshot_for_probe(optimizer):
    seen = set()
    param_backup = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if id(p) in seen:
                continue
            seen.add(id(p))
            param_backup.append((p, _local(p).detach().to("cpu", copy=True)))
    # deepcopy because Optimizer.state_dict() returns references to the live state
    # tensors, not copies -- without this the "snapshot" would track the mutation.
    opt_state = copy.deepcopy(optimizer.state_dict())
    return param_backup, opt_state


def _restore_from_probe(optimizer, snapshot) -> None:
    param_backup, opt_state = snapshot
    with torch.no_grad():
        for p, cpu in param_backup:
            local = _local(p)
            local.copy_(cpu.to(local.device))
    optimizer.load_state_dict(opt_state)


# --------------------------------------------------------------------------- #
# 1. Update-magnitude measurement (the shared primitive)
# --------------------------------------------------------------------------- #

@dataclass
class GroupUpdateStats:
    """Per-group applied-update statistics measured over a dry run."""

    label: str
    is_muon: bool
    num_params: int
    # Median over measured steps of the parameter-change RMS, averaged across the
    # group's params (numel-weighted). This is the *applied* update RMS, i.e. it
    # already includes the current lr and any adjust_lr_fn scaling.
    applied_rms: float
    per_step_rms: List[float] = field(default_factory=list)
    # Analytic adjust_lr_fn ratios for the largest param in the group, for
    # comparison against the empirical numbers (muon groups only).
    analytic_original_ratio: Optional[float] = None
    analytic_match_rms_ratio: Optional[float] = None
    # The adjust_lr_fn ratio actually IN EFFECT during the measurement (depends
    # on the group's adjust_lr_fn). Needed to convert the measured (relative)
    # multiplier into an absolute base-LR scaling comparable to the analytic
    # ratios above. 1.0 for non-muon groups.
    active_ratio: float = 1.0


def _analytic_ratios(group: dict) -> Dict[str, Optional[float]]:
    """Reproduce ``gefen_muon._adjust_lr`` ratios for the group's largest 2D
    param, so the empirical multiplier can be sanity-checked against the formula."""
    if not _is_muon_group(group):
        return {"original": None, "match_rms_adamw": None, "active": 1.0}
    best = None
    for p in group["params"]:
        if p.ndim >= 2:
            rows, cols = p.shape[:2]
            size = rows * cols
            if best is None or size > best[0]:
                best = (size, rows, cols)
    if best is None:
        return {"original": None, "match_rms_adamw": None, "active": 1.0}
    _, rows, cols = best
    original = math.sqrt(max(1, rows / cols))
    match = 0.2 * math.sqrt(max(rows, cols))
    fn = group.get("adjust_lr_fn")
    if fn == "match_rms_adamw":
        active = match
    else:  # None / "original"
        active = original
    return {
        "original": original,
        "match_rms_adamw": match,
        "active": active,
    }


def measure_update_rms(
    optimizer,
    closure: Callable[[], torch.Tensor],
    *,
    num_steps: int = 20,
    warmup: int = 3,
    zero_weight_decay: bool = True,
) -> List[GroupUpdateStats]:
    """Measure the per-group *applied* update RMS over a short dry run.

    Parameters
    ----------
    optimizer
        A constructed Gefen / GefenMuon / GefenMuonHybrid (or any torch optimizer
        exposing ``param_groups``).
    closure
        Callable that runs forward + ``loss.backward()`` and returns the loss.
        It must populate ``.grad`` on the params (i.e. call ``backward``). The
        probe handles ``zero_grad`` and ``step`` itself.
    num_steps
        Total optimizer steps to run.
    warmup
        Leading steps to discard. Gefen learns its codebook / period and the
        bias-correction transient dominates the first few steps, so these are
        not representative of steady-state update magnitude.
    zero_weight_decay
        Temporarily force ``weight_decay = 0`` in every group during the probe so
        the measured parameter delta is the pure optimizer update (decoupled
        weight decay would otherwise add a ``-lr*wd*p`` term and bias the RMS).
        Restored on exit.

    Returns
    -------
    list[GroupUpdateStats], one per param group, in ``optimizer.param_groups``
    order.
    """
    if num_steps <= warmup:
        raise ValueError(f"num_steps ({num_steps}) must exceed warmup ({warmup})")

    groups = optimizer.param_groups

    # Non-destructive: snapshot params + optimizer state so the probe leaves the
    # live training job exactly as it found it (restored in the finally below).
    probe_snapshot = _snapshot_for_probe(optimizer)

    # Snapshot and optionally neutralize weight decay.
    saved_wd = [g.get("weight_decay", 0.0) for g in groups]
    if zero_weight_decay:
        for g in groups:
            if "weight_decay" in g:
                g["weight_decay"] = 0.0

    # Accumulate numel-weighted RMS per group per measured step.
    per_step: List[List[float]] = [[] for _ in groups]

    try:
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            if loss is None or not torch.isfinite(torch.as_tensor(loss)).all():
                raise RuntimeError(
                    f"closure returned non-finite loss at step {step}: {loss}"
                )

            # Snapshot params (local shards) before the step.
            before: List[List[torch.Tensor]] = []
            for g in groups:
                snaps = []
                for p in g["params"]:
                    if p.grad is None:
                        snaps.append(None)
                    else:
                        snaps.append(_local(p).detach().clone())
                before.append(snaps)

            optimizer.step()

            if step < warmup:
                continue

            for gi, g in enumerate(groups):
                sq_sum = 0.0
                n = 0
                for p, prev in zip(g["params"], before[gi]):
                    if prev is None:
                        continue
                    cur = _local(p).detach()
                    delta = (cur - prev).float()
                    sq_sum += float(delta.pow(2).sum().item())
                    n += delta.numel()
                if n > 0:
                    per_step[gi].append(math.sqrt(sq_sum / n))
    finally:
        for g, wd in zip(groups, saved_wd):
            if "weight_decay" in g:
                g["weight_decay"] = wd
        # Revert all weight + optimizer-state mutation from the probe steps.
        _restore_from_probe(optimizer, probe_snapshot)

    stats: List[GroupUpdateStats] = []
    for gi, g in enumerate(groups):
        samples = per_step[gi]
        applied = _median(samples) if samples else float("nan")
        ar = _analytic_ratios(g)
        stats.append(
            GroupUpdateStats(
                label=_group_label(gi, g),
                is_muon=_is_muon_group(g),
                num_params=len(g["params"]),
                applied_rms=applied,
                per_step_rms=samples,
                analytic_original_ratio=ar["original"],
                analytic_match_rms_ratio=ar["match_rms_adamw"],
                active_ratio=ar["active"],
            )
        )
    return stats


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


# --------------------------------------------------------------------------- #
# 2. Relative calibration (cross-path consistency)
# --------------------------------------------------------------------------- #

@dataclass
class RelativeCalibration:
    """Result of ``calibrate_relative``.

    ``multipliers[label]`` is the factor to apply to each group's LR so that all
    groups produce an applied update RMS equal to the reference group's. With
    these multipliers, a single base LR is consistent across every path.
    """

    reference_label: str
    stats: List[GroupUpdateStats]
    multipliers: Dict[str, float]
    # For the muon group(s): empirical multiplier vs. what the analytic flags
    # would have produced, so the caller can judge whether 'match_rms_adamw'
    # alone suffices.
    analytic_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Relative LR calibration (target = reference group's applied update RMS)",
            f"  reference: {self.reference_label}",
            "",
            f"  {'group':<28}{'applied_rms':>14}{'lr_multiplier':>16}",
        ]
        for s in self.stats:
            m = self.multipliers.get(s.label, float("nan"))
            lines.append(f"  {s.label:<28}{s.applied_rms:>14.4e}{m:>16.4f}")
        if self.analytic_comparison:
            lines += ["", "  muon: empirical vs analytic adjust_lr_fn ratios"]
            for label, c in self.analytic_comparison.items():
                lines.append(
                    f"    {label}: empirical_abs={c['total_empirical']:.4f} "
                    f"(active={c['active_ratio']:.4f} x {c['empirical_vs_ref']:.4f})  "
                    f"original={c['original']:.4f}  "
                    f"match_rms_adamw={c['match_rms_adamw']:.4f}  "
                    f"(match_rms error={c['match_rms_rel_error']*100:.1f}%)"
                )
        return "\n".join(lines)


def calibrate_relative(
    optimizer,
    closure: Callable[[], torch.Tensor],
    *,
    reference: str = "gefen",
    num_steps: int = 20,
    warmup: int = 3,
) -> RelativeCalibration:
    """Compute per-group LR multipliers that equalize applied update RMS.

    ``reference`` selects the group whose update RMS becomes the target:
      * "gefen"  -> the (first) non-Muon, AdamW-like group. This is the natural
                    choice for a hybrid: it makes the Muon half match the
                    AdamW-scale half, i.e. an *empirical* match_rms_adamw.
      * "muon"   -> the (first) Muon group.
      * a label  -> match that specific group (use ``GroupUpdateStats.label``).

    The multiplier for group g is ``ref_applied_rms / g_applied_rms`` (the applied
    RMS is linear in LR, so the ratio is LR-independent and stable). Apply it as
    ``group["lr"] *= multiplier`` or fold it into ``adjust_lr_fn`` choice.
    """
    stats = measure_update_rms(
        optimizer, closure, num_steps=num_steps, warmup=warmup
    )

    ref_stat = _pick_reference(stats, reference)
    ref_rms = ref_stat.applied_rms
    if not math.isfinite(ref_rms) or ref_rms == 0.0:
        raise RuntimeError(
            f"reference group '{ref_stat.label}' has unusable applied RMS "
            f"{ref_rms}; cannot calibrate"
        )

    multipliers: Dict[str, float] = {}
    analytic_comparison: Dict[str, Dict[str, float]] = {}
    for s in stats:
        if math.isfinite(s.applied_rms) and s.applied_rms > 0:
            multipliers[s.label] = ref_rms / s.applied_rms
        else:
            multipliers[s.label] = float("nan")

        if s.is_muon and s.analytic_match_rms_ratio is not None:
            # `emp` is the multiplier needed ON TOP OF the scaling that was
            # already in effect during the measurement (s.active_ratio). To
            # compare against the analytic ratios -- which are ABSOLUTE scalings
            # of the base LR relative to a no-adjust (ratio=1) baseline -- fold
            # the active ratio back in:
            #     total_empirical = active_ratio * emp
            # This makes the comparison valid no matter which adjust_lr_fn the
            # calibration run used.
            emp = multipliers[s.label]
            total_empirical = s.active_ratio * emp
            match = s.analytic_match_rms_ratio
            analytic_comparison[s.label] = {
                "empirical_vs_ref": emp,
                "active_ratio": s.active_ratio,
                "total_empirical": total_empirical,
                "original": s.analytic_original_ratio,
                "match_rms_adamw": match,
                # How far the analytic match_rms_adamw is from the absolute
                # scaling that actually equalizes the muon group with the ref.
                "match_rms_rel_error": (
                    abs(match - total_empirical) / total_empirical
                    if total_empirical
                    else float("nan")
                ),
            }

    return RelativeCalibration(
        reference_label=ref_stat.label,
        stats=stats,
        multipliers=multipliers,
        analytic_comparison=analytic_comparison,
    )


def _pick_reference(stats: List[GroupUpdateStats], reference: str) -> GroupUpdateStats:
    if reference == "gefen":
        for s in stats:
            if not s.is_muon and math.isfinite(s.applied_rms):
                return s
        raise ValueError("no usable non-Muon (gefen) group found for reference")
    if reference == "muon":
        for s in stats:
            if s.is_muon and math.isfinite(s.applied_rms):
                return s
        raise ValueError("no usable Muon group found for reference")
    for s in stats:
        if s.label == reference:
            return s
    raise ValueError(f"reference group '{reference}' not found among {[s.label for s in stats]}")


# --------------------------------------------------------------------------- #
# 2b. Shadow-AdamW reference (identical-gradient calibration)
# --------------------------------------------------------------------------- #

@dataclass
class ParamVsAdamw:
    """Per-parameter comparison of Gefen's update RMS vs a shadow AdamW fed the
    SAME gradient sequence."""

    name: str
    is_muon: bool
    rows: int
    cols: int
    gefen_rms: float
    adamw_rms: float
    # multiplier on this param's LR that makes Gefen's update RMS match AdamW's.
    ratio: float
    active_ratio: float  # adjust_lr_fn ratio in effect (muon); 1.0 otherwise
    sqrt_max: Optional[float]  # sqrt(max(rows,cols)) for muon, else None


def calibrate_vs_adamw(
    optimizer,
    closure: Callable[[], torch.Tensor],
    *,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
    num_steps: int = 40,
    warmup: int = 5,
) -> List[ParamVsAdamw]:
    """Per-parameter: Gefen update RMS vs AdamW update RMS on identical gradients.

    A *shadow* AdamW state (m, v) is maintained alongside the real optimizer and
    fed the same gradient each step, but never applied -- the real optimizer
    (Gefen / GefenMuon / hybrid) drives the trajectory. This removes the
    arbitrary-reference problem of ``calibrate_relative``: the target is the
    canonical AdamW update magnitude, on the very gradients the model actually
    produces. ``ratio = adamw_rms / gefen_rms`` is the LR multiplier that makes
    each parameter's Gefen update match AdamW.

    Works for every path:
      * backup / plain-Gefen params -> ratio ~= 1 iff Gefen is a faithful AdamW
        drop-in at the same LR; deviation quantifies the block-mean second-moment
        effect.
      * muon params -> ratio is the empirical replacement for the analytic
        ``match_rms_adamw`` constant.
    """
    b1, b2 = betas
    groups = optimizer.param_groups
    # Non-destructive: snapshot params + optimizer state, restored in the finally.
    probe_snapshot = _snapshot_for_probe(optimizer)
    saved_wd = [g.get("weight_decay", 0.0) for g in groups]
    for g in groups:
        if "weight_decay" in g:
            g["weight_decay"] = 0.0

    metas = []  # (p, group, analytic_ratios, group_lr, param_name)
    for g in groups:
        ar = _analytic_ratios(g)
        lr = g["lr"]
        lr = float(lr.item() if isinstance(lr, torch.Tensor) else lr)
        g_params = g["params"]
        base = g.get("name", "?")
        # Key each record by the PARAM, not just the group: the Gefen family puts
        # one param per group (so base == the named_parameters() name), but a
        # generic optimizer may group many tensors -- disambiguate those with an
        # index so distinct tensors don't collapse into one per-type bucket.
        for idx, p in enumerate(g_params):
            pname = base if len(g_params) == 1 else f"{base}.{idx}"
            metas.append((p, g, ar, lr, pname))

    m: Dict[int, torch.Tensor] = {}
    v: Dict[int, torch.Tensor] = {}
    g_acc: Dict[int, List[float]] = {}
    a_acc: Dict[int, List[float]] = {}
    t = 0
    try:
        for step in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            if loss is None or not torch.isfinite(torch.as_tensor(loss)).all():
                raise RuntimeError(f"non-finite loss at step {step}: {loss}")
            t += 1

            before: Dict[int, torch.Tensor] = {}
            for p, g, ar, lr, _pname in metas:
                if p.grad is None:
                    continue
                grad = _local(p.grad).detach().float()
                key = id(p)
                if key not in m:
                    m[key] = torch.zeros_like(grad)
                    v[key] = torch.zeros_like(grad)
                mm, vv = m[key], v[key]
                # Shared shadow-AdamW primitive (same code path the in-optimizer
                # auto-calibration window uses). Scale by the group LR so the
                # shadow update is on the same *applied-delta* basis as Gefen's
                # measured delta (which already includes lr and any adjust_lr_fn
                # scaling). ratio is then the lr-independent multiplier to match.
                unit = _shadow_adamw_unit_update(mm, vv, grad, t, b1, b2, eps)
                upd = unit * lr
                before[key] = _local(p).detach().clone()
                if step >= warmup:
                    a_acc.setdefault(key, []).append(_rms(upd))

            optimizer.step()
            if step < warmup:
                continue
            for p, g, ar, lr, _pname in metas:
                key = id(p)
                if key not in before:
                    continue
                delta = (_local(p).detach() - before[key]).float()
                g_acc.setdefault(key, []).append(_rms(delta))
    finally:
        for g, wd in zip(groups, saved_wd):
            if "weight_decay" in g:
                g["weight_decay"] = wd
        # Revert all weight + optimizer-state mutation from the probe steps.
        _restore_from_probe(optimizer, probe_snapshot)

    out: List[ParamVsAdamw] = []
    for p, g, ar, lr, pname in metas:
        key = id(p)
        if key not in g_acc or key not in a_acc:
            continue
        grms = _median(g_acc[key])
        arms = _median(a_acc[key])
        sqrt_max = (ar["match_rms_adamw"] / 0.2) if ar["match_rms_adamw"] else None
        out.append(
            ParamVsAdamw(
                name=pname,
                is_muon=_is_muon_group(g),
                rows=p.shape[0] if p.ndim >= 1 else 1,
                cols=p.shape[1] if p.ndim >= 2 else 1,
                gefen_rms=grms,
                adamw_rms=arms,
                ratio=(arms / grms) if grms > 0 else float("nan"),
                active_ratio=ar["active"],
                sqrt_max=sqrt_max,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# 3. Absolute LR-range test (Leslie Smith / fastai lr_find)
# --------------------------------------------------------------------------- #

@dataclass
class LRRangeResult:
    lrs: List[float]
    losses: List[float]
    smoothed: List[float]
    suggested_lr: float
    min_loss_lr: float
    diverged_at: Optional[float]

    def summary(self) -> str:
        lines = [
            "LR-range test",
            f"  scanned {len(self.lrs)} points "
            f"[{self.lrs[0]:.2e} .. {self.lrs[-1]:.2e}]",
            f"  steepest-descent LR (suggested base): {self.suggested_lr:.3e}",
            f"  min-loss LR (upper bound, too hot):    {self.min_loss_lr:.3e}",
        ]
        if self.diverged_at is not None:
            lines.append(f"  diverged at LR ~ {self.diverged_at:.3e}")
        return "\n".join(lines)


def lr_range_test(
    optimizer,
    closure: Callable[[], torch.Tensor],
    *,
    num_iter: int = 100,
    start_lr: float = 1e-7,
    end_lr: float = 1.0,
    smooth_beta: float = 0.9,
    diverge_factor: float = 4.0,
) -> LRRangeResult:
    """Ramp the LR exponentially and record the loss to locate a good base LR.

    Each iteration: set every group's LR to the current ramp value, run
    ``closure`` (forward + backward), ``optimizer.step()``, record the loss, and
    multiply the LR by a fixed gamma. The suggested base LR is the point of
    steepest loss descent on the EMA-smoothed curve -- the classic lr_find heuristic.

    NOTE: this mutates the model/optimizer (it actually trains for ``num_iter``
    steps). Run it on a throwaway copy if you need to preserve weights. It does
    restore the original per-group LRs on exit.
    """
    groups = optimizer.param_groups
    saved_lrs = [g["lr"] for g in groups]
    gamma = (end_lr / start_lr) ** (1.0 / max(1, num_iter - 1))

    lrs: List[float] = []
    losses: List[float] = []
    smoothed: List[float] = []
    avg = 0.0
    best = float("inf")
    diverged_at: Optional[float] = None

    try:
        lr = start_lr
        for it in range(num_iter):
            for g in groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = closure()
            loss_val = float(loss)
            optimizer.step()

            avg = smooth_beta * avg + (1 - smooth_beta) * loss_val
            sm = avg / (1 - smooth_beta ** (it + 1))  # bias-corrected EMA

            lrs.append(lr)
            losses.append(loss_val)
            smoothed.append(sm)

            if not math.isfinite(sm):
                diverged_at = lr
                break
            if sm < best:
                best = sm
            if sm > diverge_factor * best:
                diverged_at = lr
                break

            lr *= gamma
    finally:
        for g, olr in zip(groups, saved_lrs):
            g["lr"] = olr

    suggested, min_loss_lr = _suggest_from_curve(lrs, smoothed)
    return LRRangeResult(
        lrs=lrs,
        losses=losses,
        smoothed=smoothed,
        suggested_lr=suggested,
        min_loss_lr=min_loss_lr,
        diverged_at=diverged_at,
    )


def _suggest_from_curve(
    lrs: List[float], smoothed: List[float], window_factor: float = 3.0
):
    """Locate a working base LR on the smoothed loss-vs-LR curve.

    ``min_loss_lr`` is the LR at the smoothed-loss minimum (the "too hot" upper
    bound -- loss starts rising past it). The suggested base LR is the point of
    steepest descent w.r.t. ``log(lr)``, but the slope is measured over a WINDOW
    spanning a factor ``window_factor`` in LR rather than between adjacent points.
    Adjacent-point slopes are dominated by single-batch noise in the flat low-LR
    head of a fine-tune curve, which made the old heuristic latch onto a spurious
    ~1e-6 suggestion; the windowed slope averages that out so the suggestion lands
    in the genuinely-descending mid-range. The search is restricted to LRs at or
    below ``min_loss_lr`` (never suggest into the rising/diverging region).
    """
    if len(lrs) < 3:
        return (lrs[0] if lrs else float("nan"), lrs[-1] if lrs else float("nan"))
    min_idx = min(range(len(smoothed)), key=lambda i: smoothed[i])
    min_loss_lr = lrs[min_idx]

    best_slope = 0.0
    best_idx = None
    for i in range(min_idx + 1):
        # Window end j: first point at least window_factor higher in LR, capped at
        # the loss minimum.
        j = i
        while (
            j + 1 <= min_idx
            and lrs[j] < lrs[i] * window_factor
        ):
            j += 1
        if j <= i:
            continue
        dlog = math.log(lrs[j]) - math.log(lrs[i])
        if dlog <= 0:
            continue
        slope = (smoothed[j] - smoothed[i]) / dlog
        if slope < best_slope:
            best_slope = slope
            # Geometric centre of the window -- where the descent is steepest.
            best_idx = (i + j) // 2

    if best_idx is None:
        # No descent found (e.g. immediate divergence): fall back to the classic
        # conservative choice an order of magnitude below the loss minimum.
        return min_loss_lr / 10.0, min_loss_lr
    return lrs[best_idx], min_loss_lr
