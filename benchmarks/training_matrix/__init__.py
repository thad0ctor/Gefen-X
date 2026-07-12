"""Reproducible optimizer-training matrix helpers.

The package intentionally lives under ``benchmarks``: it is a development
harness, not part of the installed ``gefen`` runtime API.
"""

from .cells import (
    ALL_CELLS,
    BATCHED_NS_DEFAULT_WORKSPACE_BYTES,
    CELL_RECIPES,
    CORE_CELLS,
    ISOLATION_CELLS,
    CellBuildConfig,
    OptimizerPair,
    build_optimizer,
    resolve_cell,
)

__all__ = [
    "ALL_CELLS",
    "BATCHED_NS_DEFAULT_WORKSPACE_BYTES",
    "CELL_RECIPES",
    "CORE_CELLS",
    "ISOLATION_CELLS",
    "CellBuildConfig",
    "OptimizerPair",
    "build_optimizer",
    "resolve_cell",
]
