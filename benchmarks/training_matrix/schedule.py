"""LR schedules that preserve distinct Muon and auxiliary base LRs."""

from __future__ import annotations

import math
from typing import Any


class GroupLRSchedule:
    """Apply one multiplier to every live parameter-group base LR.

    Capturing each group's base LR is important for the hybrid recipes: an
    absolute scheduler assignment would silently erase the recommended 2:1
    Muon-to-backup ratio.
    """

    def __init__(
        self,
        optimizer: Any,
        *,
        schedule: str,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ):
        if schedule not in {"constant", "warmup_cosine"}:
            raise ValueError("schedule must be constant or warmup_cosine")
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if warmup_steps < 0 or warmup_steps >= total_steps:
            if not (schedule == "constant" and warmup_steps == 0):
                raise ValueError("warmup_steps must be in [0, total_steps)")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.schedule = schedule
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if not self.base_lrs:
            raise ValueError("optimizer has no parameter groups")

    def multiplier(self, step: int) -> float:
        if not 0 <= step < self.total_steps:
            raise ValueError(f"step must be in [0, {self.total_steps}), got {step}")
        if self.schedule == "constant":
            return 1.0
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step + 1 - self.warmup_steps) / decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def apply(self, step: int) -> list[float]:
        multiplier = self.multiplier(step)
        current = []
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            lr = base_lr * multiplier
            group["lr"] = lr
            current.append(lr)
        return current

    def metadata(self) -> dict:
        return {
            "name": self.schedule,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }
