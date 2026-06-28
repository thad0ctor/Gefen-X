"""Standalone, non-invasive tools for working with Gefen optimizers.

Nothing here mutates optimizer state or changes the training math by itself;
each entry point either *measures* the optimizer in a short dry run or *reports*
a recommendation the caller is free to apply. This keeps the tools safe to run
against live training setups (including FSDP2/DTensor) without perturbing
checkpoint/resume semantics.
"""

from .lr_calibration import (
    GroupUpdateStats,
    RelativeCalibration,
    ParamVsAdamw,
    LRRangeResult,
    measure_update_rms,
    calibrate_relative,
    calibrate_vs_adamw,
    lr_range_test,
)

__all__ = [
    "GroupUpdateStats",
    "RelativeCalibration",
    "ParamVsAdamw",
    "LRRangeResult",
    "measure_update_rms",
    "calibrate_relative",
    "calibrate_vs_adamw",
    "lr_range_test",
]
