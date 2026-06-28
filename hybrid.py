"""Hybrid optimizer: GefenMuon on 2D hidden weights, Gefen on everything else.

GefenMuon is a Muon-style optimizer: it only accepts 2D parameters and has no
fallback path for embeddings, the LM head, or 1D (norm/bias) parameters. This
composite routes the 2D hidden weight matrices to GefenMuon and routes the rest
(embeddings, LM head, norms, biases) to plain Gefen, exposing a single
``torch.optim.Optimizer``-compatible interface so it drops straight into the
Hugging Face Trainer / accelerate single-GPU path.

Both sub-optimizers key their per-parameter codebook cache on ``group["name"]``,
so parameters MUST be supplied as ``(name, param)`` pairs with unique names --
passing bare tensors collapses every name to ``"none"`` and corrupts the cache.

Scope: single-GPU / DDP and FSDP2 (fully_shard / DTensor). Under FSDP2 each rank
holds only a shard of every parameter, so the backup (plain Gefen) params update
per-shard directly, while the Muon 2D matrices all-gather their gradient to the
full matrix per step (GefenMuon._step_automatic), run the quantized-momentum +
Newton-Schulz pipeline on the full matrix so the numerics match the single-GPU
reference, then slice the orthogonalized update back to the local shard. Tensor
parallelism / multi-dim (HSDP x TP) meshes are not validated.
"""
from collections import OrderedDict

import torch

from gefen.gefen import Gefen
from gefen.gefen_muon import GefenMuon
from gefen.params import split_params_for_muon, validate_split


class GefenMuonHybrid(torch.optim.Optimizer):
    def __init__(
        self,
        muon_named_params,
        backup_named_params,
        *,
        lr,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
        sharded_mode="exact",
        verbose=False,
    ):
        muon_named_params = list(muon_named_params)
        backup_named_params = list(backup_named_params)
        if not muon_named_params and not backup_named_params:
            raise ValueError("GefenMuonHybrid received no parameters to optimize")
        # Catch the silent footguns: a param routed to both halves (stepped
        # twice) or a duplicate name (codebook cache key collision). Completeness
        # (a trainable param in neither list) needs the model, so it is checked in
        # split_params_for_muon's callers, not here.
        validate_split(muon_named_params, backup_named_params)

        self.muon = (
            GefenMuon(
                muon_named_params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                adjust_lr_fn=adjust_lr_fn,
                fused=fused,
                sharded_mode=sharded_mode,
                verbose=verbose,
            )
            if muon_named_params
            else None
        )
        self.backup = (
            Gefen(
                backup_named_params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                fused=fused,
                verbose=verbose,
            )
            if backup_named_params
            else None
        )
        self._subopts = [o for o in (self.muon, self.backup) if o is not None]

        # Deliberately do NOT call super().__init__(): we expose each
        # sub-optimizer's real param_groups/state via properties (shared dict
        # refs), so the LR scheduler's in-place ``group["lr"] = ...`` updates
        # reach the children. We still set the hook registries torch.optim and
        # accelerate may introspect, so registration never crashes.
        self.defaults = self._subopts[0].defaults
        self._optimizer_step_pre_hooks = OrderedDict()
        self._optimizer_step_post_hooks = OrderedDict()
        self._optimizer_state_dict_pre_hooks = OrderedDict()
        self._optimizer_state_dict_post_hooks = OrderedDict()
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()

    @property
    def param_groups(self):
        groups = []
        for o in self._subopts:
            groups.extend(o.param_groups)
        return groups

    @property
    def state(self):
        merged = {}
        for o in self._subopts:
            merged.update(o.state)
        return merged

    def zero_grad(self, set_to_none: bool = True):
        for o in self._subopts:
            o.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for o in self._subopts:
            o.step()
        return loss

    def state_dict(self):
        return {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "backup": self.backup.state_dict() if self.backup is not None else None,
        }

    def load_state_dict(self, state_dict):
        if self.muon is not None and state_dict.get("muon") is not None:
            self.muon.load_state_dict(state_dict["muon"])
        if self.backup is not None and state_dict.get("backup") is not None:
            self.backup.load_state_dict(state_dict["backup"])

    @classmethod
    def from_model(cls, model, *, backup_substrings=None, **kwargs):
        """Build the hybrid straight from a model, splitting + validating params.

        Routes 2D hidden weights to Muon and everything else to the backup (see
        ``gefen.split_params_for_muon``), then validates that the split covers
        every trainable parameter exactly once before constructing. All other
        keyword arguments are forwarded to ``__init__``.
        """
        split_kwargs = {} if backup_substrings is None else {"backup_substrings": backup_substrings}
        muon_named, backup_named = split_params_for_muon(model, **split_kwargs)
        validate_split(muon_named, backup_named, model=model)
        return cls(muon_named, backup_named, **kwargs)

    def add_param_group(self, param_group):
        raise NotImplementedError(
            "GefenMuonHybrid splits params at construction; add_param_group is unsupported"
        )

    def __repr__(self):
        nm = len(self.muon.param_groups) if self.muon is not None else 0
        nb = len(self.backup.param_groups) if self.backup is not None else 0
        return f"GefenMuonHybrid(muon_params={nm}, backup_params={nb})"
