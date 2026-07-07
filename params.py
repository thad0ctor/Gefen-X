"""Parameter splitting + validation helpers for the GefenMuon hybrid.

``GefenMuonHybrid`` takes two ``(name, param)`` lists: the 2D hidden weight
matrices that go to Muon, and everything else (embeddings, LM head, norms,
biases) that goes to the plain-Gefen backup. Getting that split wrong is a quiet
correctness footgun -- a param in *both* lists is stepped twice per optimizer
step, and a trainable param in *neither* is silently never optimized. This module
centralizes the split so callers stop hand-rolling it (the benchmark harness had
its own copy) and adds validation the hybrid runs at construction.
"""
import logging
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Substrings (matched case-insensitively against the parameter name) that force a
# 2D weight onto the backup-Gefen side instead of Muon. Muon orthogonalizes a 2D
# *hidden* weight matrix; token/position embeddings, the (often tied) LM head and
# classifier heads are vocabulary/position/class projections, not hidden
# matrices, and Muon has no special handling for them, so they go to the backup
# like every non-2D tensor. Deliberately conservative: broad tokens like plain
# "head"/"output" would also swallow hidden matrices (e.g. "head_dim"-named
# projections, "output.dense"), so they are NOT listed -- ``split_params_for_muon``
# additionally routes by MODULE TYPE (any ``nn.Embedding`` weight, tied or not),
# which is what catches embeddings these name patterns miss (e.g. T5's
# ``shared``).
DEFAULT_BACKUP_SUBSTRINGS: Tuple[str, ...] = (
    "embed",
    "wte",
    "wpe",
    "lm_head",
    "classifier",
    "score",
)


def is_muon_param(name: str, param, backup_substrings=DEFAULT_BACKUP_SUBSTRINGS) -> bool:
    """A parameter is Muon-routed iff it is a 2D non-embedding/-head weight."""
    if param.ndim != 2:
        return False
    lname = name.lower()
    return not any(sub.lower() in lname for sub in backup_substrings)


def _embedding_param_ids(model: nn.Module) -> set:
    """ids of every ``nn.Embedding``/``nn.EmbeddingBag`` (or subclass) weight.

    Keyed by tensor identity so TIED weights are caught regardless of which
    name ``named_parameters`` surfaces them under: a Linear LM head whose
    ``.weight`` IS the token embedding's weight object shares the id and is
    routed to the backup with it.
    """
    ids = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            weight = getattr(module, "weight", None)
            if weight is not None:
                ids.add(id(weight))
    return ids


def split_params_for_muon(
    model: nn.Module,
    backup_substrings: Iterable[str] = DEFAULT_BACKUP_SUBSTRINGS,
) -> Tuple[List[Tuple[str, nn.Parameter]], List[Tuple[str, nn.Parameter]]]:
    """Split a model's trainable params into (muon_named, backup_named).

    2D hidden weight matrices -> Muon; embeddings / LM head / 1D (norm, bias) and
    any other shapes -> backup Gefen. Embedding routing is module-type-aware:
    any parameter that is the ``.weight`` of an ``nn.Embedding``/``nn.EmbeddingBag``
    (or subclass) goes to the backup by tensor identity -- so a tied LM head and
    embeddings whose names match none of ``backup_substrings`` (e.g. T5's
    ``shared``) are still routed correctly. The name substrings remain as a
    second net for untied heads (``lm_head``, ``classifier``, ``score``, ...).

    Returns ``(name, param)`` pairs so each sub-optimizer keys its codebook cache
    on a unique name (see hybrid docstring). Parameters with
    ``requires_grad=False`` are skipped. Logs a one-line summary of the split
    (INFO) so misrouting is visible.
    """
    backup_substrings = tuple(backup_substrings)
    embedding_ids = _embedding_param_ids(model)
    muon: List[Tuple[str, nn.Parameter]] = []
    backup: List[Tuple[str, nn.Parameter]] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) not in embedding_ids and is_muon_param(name, p, backup_substrings):
            muon.append((name, p))
        else:
            backup.append((name, p))
    logger.info(
        "split_params_for_muon: %d muon / %d backup params; backup = %s",
        len(muon),
        len(backup),
        [n for n, _ in backup],
    )
    return muon, backup


def _check_named_pairs(named_params, which: str):
    """TypeError unless every item is a ``(str name, tensor param)`` pair.

    Bare tensors are a footgun here: a 2-row tensor even tuple-unpacks into two
    rows that then "validate" as garbage. Fail loudly instead and point at the
    supported forms.
    """
    for item in named_params:
        if (
            isinstance(item, (tuple, list))
            and len(item) == 2
            and isinstance(item[0], str)
            and isinstance(item[1], torch.Tensor)
        ):
            continue
        raise TypeError(
            "validate_split expects the {} list to contain (name, param) pairs, "
            "got {!r}. Pass named parameters (e.g. model.named_parameters() or "
            "the lists from split_params_for_muon), or build the optimizer via "
            "GefenMuonHybrid.from_model(model, ...)".format(which, type(item))
        )


def validate_split(muon_named_params, backup_named_params, model: nn.Module = None):
    """Raise on a split that would double-step or silently drop a parameter.

    Checks each list holds ``(name, param)`` pairs (TypeError otherwise); checks,
    by parameter *identity* (id), that no param appears in both lists and
    that no param appears twice in either list; checks names are unique (the
    codebook cache key). If ``model`` is given, also raises when a trainable
    model parameter is in neither list (silently un-optimized).

    Returns the two lists unchanged (so it can wrap a split inline).
    """
    muon_named_params = list(muon_named_params)
    backup_named_params = list(backup_named_params)

    _check_named_pairs(muon_named_params, "muon")
    _check_named_pairs(backup_named_params, "backup")

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

    names = [str(n).lower() for n, _ in muon_named_params] + [
        str(n).lower() for n, _ in backup_named_params
    ]
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
