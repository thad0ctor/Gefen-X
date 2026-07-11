"""Dependency-free labels shared by the optimizer-sweep plotters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def optimizer_label(
    opt: str,
    row: Mapping[str, Any],
    *,
    base_labels: Mapping[str, str] | None = None,
    multiline: bool = False,
) -> str:
    """Return the display label for an optimizer result row."""
    if opt != "gefen_muon":
        return (base_labels or {}).get(opt, opt)
    backup = str(row.get("muon_backup_optimizer", "")).strip().lower()
    suffix = {"adamw": "AdamW", "gefen": "Gefen"}.get(backup, "hybrid")
    separator = "\n+ " if multiline else " + "
    return f"Gefen Muon{separator}{suffix}"


def muon_recipe_text(row: Mapping[str, Any]) -> str:
    """Summarize the recorded Muon recipe for a result row."""
    variant = row.get("variant", "unspecified") or "unspecified"
    schedule = row.get("muon_ns_schedule", "unspecified") or "unspecified"
    normuon = str(row.get("muon_normuon", "")).strip().lower()
    normuon_text = {
        "true": "NorMuon",
        "false": "NorMuon off",
    }.get(normuon, "NorMuon unspecified")
    backup = str(row.get("muon_backup_optimizer", "")).strip().lower()
    backup_label = {
        "adamw": "AdamW",
        "gefen": "Gefen",
    }.get(backup, "unspecified")
    return (
        f"Muon row = {variant}: {schedule} + {normuon_text} + "
        f"{backup_label} backup"
    )
