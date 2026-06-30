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

Recommended configuration (minimizes divergence vs AdamW at matched LR -- see the
loss-recovery sweep; Qwen3-0.6B/1.7B): keep the Muon half at AdamW scale and lower
only the backup half, plus per-element 2nd moment on the 1D backup tensors. Both
are free on throughput; ``ns_schedule`` defaults to ``"tuned3"`` (loss-neutral,
~+28% step throughput)::

    opt = GefenMuonHybrid(
        *split_params_for_muon(model),
        lr=ADAMW_LR,                  # Muon (2D) half stays at AdamW scale
        backup_lr=0.5 * ADAMW_LR,     # backup half ~0.4-0.6x (tune); the main LR lever
        adjust_lr_fn="match_rms_adamw",
        backup_1d_period_one=True,    # AdamW-like per-element 2nd moment on norms/biases
    )

This beats AdamW at 0.6B and trails it only slightly at 1.7B while keeping ~1.0
B/param optimizer state. To CLOSE the residual 1.7B gap to AdamW, add
``backup_2d_period_one=True`` -- per-element 2nd moment on the embedding/LM-head
too -- which matches AdamW's loss at ~2.45 B/param (still 0.6x AdamW), a
loss/memory trade. ``stochastic_round=True`` is a free (throughput-neutral,
loss-neutral) opt-in that debiases the 8-bit momentum quantization. The
constructor defaults stay parity-preserving (period-1 off, shared lr,
stochastic_round off) except ``ns_schedule="tuned3"``; set the rest explicitly to
opt in.
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
        muon_lr=None,
        backup_lr=None,
        weight_decay=0.0,
        muon_weight_decay=None,
        backup_weight_decay=None,
        no_decay_substrings=(),
        backup_1d_period_one=False,
        backup_2d_period_one=False,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        ns_schedule="tuned3",
        adjust_lr_fn="match_rms_adamw",
        sharded_mode="exact",
        fp8_ns=False,
        fp8_ns_compile=True,
        stochastic_round=False,
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

        # adjust_lr_fn guard (idea 5). The default is "match_rms_adamw", which
        # rescales each Muon update to AdamW-equivalent RMS so a single AdamW-scale
        # lr is correct for both halves. The legacy None/"original" scaling leaves
        # the Muon matrices on Muon-native footing; feeding them an AdamW-scale lr
        # then under-trains them, and no single shared lr suits both halves. Make
        # that loud rather than silent.
        if adjust_lr_fn in (None, "original"):
            import warnings

            warnings.warn(
                "GefenMuonHybrid: adjust_lr_fn={!r} uses Muon-native LR scaling, so "
                "an AdamW-scale lr will mis-scale the 2D Muon matrices relative to "
                "the backup half. Use adjust_lr_fn='match_rms_adamw' (the default) "
                "to share one AdamW-scale lr, or set muon_lr explicitly.".format(
                    adjust_lr_fn
                ),
                stacklevel=2,
            )

        # Per-half learning rates. The shared ``lr`` keeps the documented
        # single-LR behavior; ``muon_lr`` / ``backup_lr`` override it per half so
        # the Muon matrices (with ``adjust_lr_fn`` rescaling them to AdamW-RMS)
        # and the backup-Gefen norms/embeddings/head can be tuned independently.
        # NOTE: an LR scheduler that overwrites every group to one absolute value
        # collapses the split; a multiplicative scheduler (the common case)
        # preserves the muon/backup ratio because each sub-optimizer's groups
        # carry their own lr.
        muon_lr = lr if muon_lr is None else muon_lr
        backup_lr = lr if backup_lr is None else backup_lr

        # Per-half weight decay. Decoupled (AdamW-style) decay is `lr*wd` per
        # step; keeping muon/backup decay separate lets a recipe decay the 2D
        # matrices while leaving the backup (norms/biases/embeddings/head) on a
        # different schedule. Both default to the shared weight_decay.
        muon_weight_decay = weight_decay if muon_weight_decay is None else muon_weight_decay
        backup_weight_decay = (
            weight_decay if backup_weight_decay is None else backup_weight_decay
        )

        self.muon = (
            GefenMuon(
                muon_named_params,
                lr=muon_lr,
                weight_decay=muon_weight_decay,
                momentum=momentum,
                nesterov=nesterov,
                ns_steps=ns_steps,
                ns_schedule=ns_schedule,
                adjust_lr_fn=adjust_lr_fn,
                fused=fused,
                sharded_mode=sharded_mode,
                fp8_ns=fp8_ns,
                fp8_ns_compile=fp8_ns_compile,
                stochastic_round=stochastic_round,
                verbose=verbose,
            )
            if muon_named_params
            else None
        )
        # AdamW-style group semantics: names matching no_decay_substrings (e.g.
        # "norm", "bias") form a separate backup group with weight_decay=0, the
        # rest keep backup_weight_decay. With the default empty no_decay_substrings
        # the backup is a single flat group at backup_weight_decay -- identical to
        # the previous single-group construction.
        no_decay_substrings = tuple(s.lower() for s in no_decay_substrings)

        def _is_no_decay(name):
            lname = name.lower()
            return any(sub in lname for sub in no_decay_substrings)

        if backup_named_params and no_decay_substrings:
            decay_group = [(n, p) for n, p in backup_named_params if not _is_no_decay(n)]
            no_decay_group = [(n, p) for n, p in backup_named_params if _is_no_decay(n)]
            backup_groups = [
                g
                for g in (
                    {
                        "params": decay_group,
                        "lr": backup_lr,
                        "betas": betas,
                        "eps": eps,
                        "weight_decay": backup_weight_decay,
                    }
                    if decay_group
                    else None,
                    {
                        "params": no_decay_group,
                        "lr": backup_lr,
                        "betas": betas,
                        "eps": eps,
                        "weight_decay": 0.0,
                    }
                    if no_decay_group
                    else None,
                )
                if g is not None
            ]
            self.backup = Gefen(backup_groups, lr=backup_lr, betas=betas, eps=eps,
                                 weight_decay=backup_weight_decay, fused=fused,
                                 force_1d_period_one=backup_1d_period_one,
                                 force_2d_period_one=backup_2d_period_one,
                                 stochastic_round=stochastic_round,
                                 verbose=verbose)
        else:
            self.backup = (
                Gefen(
                    backup_named_params,
                    lr=backup_lr,
                    betas=betas,
                    eps=eps,
                    weight_decay=backup_weight_decay,
                    fused=fused,
                    force_1d_period_one=backup_1d_period_one,
                    force_2d_period_one=backup_2d_period_one,
                    stochastic_round=stochastic_round,
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
