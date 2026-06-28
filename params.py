"""Parameter splitting + validation helpers for the GefenMuon hybrid.

``GefenMuonHybrid`` takes two ``(name, param)`` lists: the 2D hidden weight
matrices that go to Muon, and everything else (embeddings, LM head, norms,
biases) that goes to the plain-Gefen backup. Getting that split wrong is a quiet
correctness footgun -- a param in *both* lists is stepped twice per optimizer
step, and a trainable param in *neither* is silently never optimized. This module
centralizes the split so callers stop hand-rolling it (the benchmark harness had
its own copy) and adds validation the hybrid runs at construction.
"""
from typing import Iterable, List, Tuple

import torch.nn as nn

# Substrings (matched case-insensitively against the parameter name) that force a
# 2D weight onto the backup-Gefen side instead of Muon. Muon orthogonalizes a 2D
# *hidden* weight matrix; the token embedding and the (often tied) LM head are
# vocabulary projections, not hidden matrices, and Muon has no special handling
# for them, so they go to the backup like every non-2D tensor.
DEFAULT_BACKUP_SUBSTRINGS: Tuple[str, ...] = ("embed", "wte", "lm_head")


def is_muon_param(name: str, param, backup_substrings=DEFAULT_BACKUP_SUBSTRINGS) -> bool:
    """A parameter is Muon-routed iff it is a 2D non-embedding/-head weight."""
    if param.ndim != 2:
        return False
    lname = name.lower()
    return not any(sub in lname for sub in backup_substrings)


def split_params_for_muon(
    model: nn.Module,
    backup_substrings: Iterable[str] = DEFAULT_BACKUP_SUBSTRINGS,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """Split a model's trainable params into (muon_named, backup_named).

    2D hidden weight matrices -> Muon; embeddings / LM head / 1D (norm, bias) and
    any other shapes -> backup Gefen. Returns ``(name, param)`` pairs so each
    sub-optimizer keys its codebook cache on a unique name (see hybrid docstring).
    Parameters with ``requires_grad=False`` are skipped.
    """
    backup_substrings = tuple(backup_substrings)
    muon: List[Tuple[str, nn.Parameter]] = []
    backup: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (muon if is_muon_param(name, p, backup_substrings) else backup).append((name, p))
    return muon, backup


def validate_split(muon_named_params, backup_named_params, model: nn.Module = None):
    """Raise on a split that would double-step or silently drop a parameter.

    Checks, by parameter *identity* (id), that no param appears in both lists and
    that no param appears twice in either list; checks names are unique (the
    codebook cache key). If ``model`` is given, also raises when a trainable
    model parameter is in neither list (silently un-optimized).

    Returns the two lists unchanged (so it can wrap a split inline).
    """
    muon_named_params = list(muon_named_params)
    backup_named_params = list(backup_named_params)

    muon_ids = {id(p) for _, p in muon_named_params}
    backup_ids = {id(p) for _, p in backup_named_params}

    if len(muon_ids) != len(muon_named_params):
        raise ValueError("Duplicate parameter object in the muon list")
    if len(backup_ids) != len(backup_named_params):
        raise ValueError("Duplicate parameter object in the backup list")

    overlap = muon_ids & backup_ids
    if overlap:
        dupes = [n for n, p in muon_named_params if id(p) in overlap]
        raise ValueError(
            "Parameter(s) {} appear in BOTH the muon and backup lists; they would "
            "be stepped twice per optimizer step".format(dupes)
        )

    names = [n for n, _ in muon_named_params] + [n for n, _ in backup_named_params]
    if len(set(names)) != len(names):
        seen, dupes = set(), set()
        for n in names:
            (dupes if n in seen else seen).add(n)
        raise ValueError(
            "Duplicate parameter name(s) {} across the muon/backup lists; the "
            "per-parameter codebook cache keys on the name, so duplicates collide".format(
                sorted(dupes)
            )
        )

    if model is not None:
        covered = muon_ids | backup_ids
        missing = [
            name
            for name, p in model.named_parameters()
            if p.requires_grad and id(p) not in covered
        ]
        if missing:
            raise ValueError(
                "Trainable parameter(s) {} are in neither the muon nor backup "
                "list; they would never be optimized".format(missing)
            )

    return muon_named_params, backup_named_params
