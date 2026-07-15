"""Hybrid optimizer: GefenMuon on 2D hidden weights, with a configurable backup.

GefenMuon is a Muon-style optimizer: it only accepts 2D parameters and has no
fallback path for embeddings, the LM head, or 1D (norm/bias) parameters. This
composite routes the 2D hidden weight matrices to GefenMuon and routes the rest
(embeddings, LM head, norms, biases) to plain Gefen by default, or to AdamW when
``backup_optimizer="adamw"`` is selected. It exposes a single
``torch.optim.Optimizer``-compatible interface so it drops straight into the
Hugging Face Trainer / accelerate single-GPU path.

Both backup choices mirror per-parameter names in each public group's
``param_names`` list (Gefen also keeps them in its per-parameter state), so
parameters SHOULD be supplied as ``(name, param)`` pairs with unique names. Bare
tensors get positional auto-names ("param_0", ...), so the split validation loses
its real names and checkpoint state can only be re-matched by construction order,
which silently breaks on any reordering.

Scope: single-GPU / DDP and FSDP2 (fully_shard / DTensor). Under FSDP2 each rank
holds only a shard of every parameter, so the selected backup optimizer updates
its params per-shard directly, while the Muon 2D matrices all-gather their
gradient to the full matrix per step (GefenMuon._step_automatic), run the
quantized-momentum + Newton-Schulz pipeline on the full matrix so the numerics
match the single-GPU reference, then slice the orthogonalized update back to the
local shard. Tensor parallelism / multi-dim (HSDP x TP) meshes are not validated.

The retained validation is task-dependent. For balanced SFT, use tuned3 plus
NorMuon and a full-LR AdamW backup::

    opt = GefenMuonHybrid(
        model,
        lr=MUON_LR,
        backup_optimizer="adamw",
        backup_lr=MUON_LR,
        ns_schedule="tuned3",
        normuon=True,
        adjust_lr_fn="match_rms_adamw",
    )

For quality-first pretraining, keep the AdamW backup but select
``ns_schedule="standard"`` / ``ns_steps=5`` and ``normuon=False``. For the
smallest state, retain ``backup_optimizer="gefen"`` (the backward-compatible
selector default), use a half-LR backup, and enable ``backup_1d_period_one``.
AdamW backup state is conventional rather than quantized, so it spends more
memory on the routed parameters. ``fused`` is independent of this choice: it
selects each child's fused implementation, not the backup algorithm.

``normuon`` is ON by default in this hybrid and essentially free in throughput
and memory; raw ``GefenMuon`` keeps it off. Resuming a checkpoint saved without
NorMuon state is safe because the per-row state initializes lazily with bias
correction. ``stochastic_round=True`` remains a throughput-neutral opt-in;
``cautious`` lost clearly in validation and should remain off. The remaining
constructor defaults stay backward-compatible (Gefen backup, shared LR,
period-one off, stochastic rounding off), with ``ns_schedule="tuned3"`` and
``normuon=True`` as the hybrid-specific defaults.
"""
import logging
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn

from gefen.gefen import (
    Gefen,
    _amp_native_scaling_required,
    _amp_prepare_optimizer_step,
    _assert_optimizer_gradients_structurally_valid,
)
from gefen.gefen_muon import GefenMuon
from gefen.params import (
    DEFAULT_BACKUP_SUBSTRINGS,
    is_muon_param,
    split_params_for_muon,
    validate_split,
)

logger = logging.getLogger(__name__)

_UNKNOWN_STATE_KEY_MSG = (
    "GefenMuonHybrid.state was accessed with a key that is not a parameter of "
    "either sub-optimizer (the Muon half or the backup half). A torch optimizer "
    "would silently auto-create state here, but the hybrid cannot know which "
    "half should own an unknown parameter. If this access comes from DeepSpeed "
    "ZeRO (it reads optimizer.state[<flattened 1-D fp32 partition>] during the "
    "first backward), note that GefenMuonHybrid cannot be used as the DeepSpeed "
    "ZeRO client optimizer: ZeRO steps flattened 1-D fp32 partitions, and the "
    "Muon half's 2D Newton-Schulz orthogonalization cannot be applied to a flat "
    "partition. Use plain Gefen as the client optimizer under DeepSpeed ZeRO, "
    "or train the hybrid under FSDP2 / DDP / single-GPU instead."
)


class _HybridMergedState(dict):
    """Merged view of the children's per-param state that ROUTES writes.

    The hybrid's ``.state`` property historically returned a plain merged dict
    rebuilt per access, so a top-level write (``opt.state[p] = {...}``, or a
    read of a not-yet-initialized param relying on torch's defaultdict
    auto-create) landed in a throwaway copy and was silently discarded --
    DeepSpeed ZeRO's ``optimizer.state[flat_partition]`` access then surfaced
    as a bare ``KeyError`` deep inside its first ``backward()``. This subclass
    keeps the merged READ view (values are the children's live per-param dicts,
    so ``opt.state[p][k] = v`` always persisted and still does) and adds:

    * ``opt.state[p]`` for a known param with no entry yet auto-creates the
      entry in the OWNING child (torch defaultdict semantics) and persists.
    * ``opt.state[p] = value`` for a known param routes to the owning child
      and persists.
    * Either access with an UNKNOWN key (e.g. DeepSpeed ZeRO's flattened 1-D
      fp32 partition) raises a KeyError that explains the ZeRO incompatibility
      instead of a bare tensor-keyed KeyError.

    All other mutators route too: ``setdefault``/``update``/``pop``/``del``
    reach the owning child (unknown keys raise the same explanatory
    ``KeyError``), and ``clear`` clears both children. ``popitem`` is
    unsupported (ambiguous on a merged two-child view).
    """

    def __init__(self, subopts, param_owner):
        super().__init__()
        self._subopts = subopts
        # Ownership is FROZEN at hybrid construction (the hybrid's param set is
        # fixed by design: add_param_group raises). Scanning the LIVE group
        # dicts here instead would defeat the guard: DeepSpeed ZeRO mutates the
        # shared group dicts to point at its flat partitions, which would then
        # masquerade as known params and push the failure to a later, more
        # cryptic point.
        self._param_owner = param_owner
        for o in subopts:
            dict.update(self, o.state)

    def _owner_of(self, key):
        # getattr: a pickled/deepcopied husk loses instance attributes; degrade
        # to "unknown key" instead of an AttributeError mislabeled as ZeRO.
        entry = getattr(self, "_param_owner", {}).get(id(key))
        if entry is not None and entry[0] is key:
            return entry[1]
        return None

    def __missing__(self, key):
        owner = self._owner_of(key)
        if owner is None:
            raise KeyError(_UNKNOWN_STATE_KEY_MSG)
        # torch's Optimizer.state is a defaultdict(dict): reading a known
        # param auto-creates its entry. Route that to the owning child so the
        # created entry is the child's live dict, then cache it in this view.
        value = owner.state[key]
        dict.__setitem__(self, key, value)
        return value

    def __setitem__(self, key, value):
        owner = self._owner_of(key)
        if owner is None:
            raise KeyError(_UNKNOWN_STATE_KEY_MSG)
        owner.state[key] = value
        dict.__setitem__(self, key, value)

    def setdefault(self, key, default=None):
        owner = self._owner_of(key)
        if owner is None:
            raise KeyError(_UNKNOWN_STATE_KEY_MSG)
        if key in owner.state:
            value = owner.state[key]
        else:
            owner.state[key] = default
            value = default
        dict.__setitem__(self, key, value)
        return value

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def pop(self, key, *default):
        owner = self._owner_of(key)
        if owner is None or key not in owner.state:
            if default:
                dict.pop(self, key, None)
                return default[0]
            raise KeyError(_UNKNOWN_STATE_KEY_MSG if owner is None else key)
        dict.pop(self, key, None)
        return owner.state.pop(key)

    def __delitem__(self, key):
        owner = self._owner_of(key)
        if owner is None:
            raise KeyError(_UNKNOWN_STATE_KEY_MSG)
        del owner.state[key]
        dict.pop(self, key, None)

    def clear(self):
        for subopt in self._subopts:
            subopt.state.clear()
        dict.clear(self)

    def popitem(self):
        raise NotImplementedError(
            "GefenMuonHybrid.state.popitem() is ambiguous on a merged "
            "two-child view; use pop(param) or del state[param], which "
            "route to the owning child."
        )


class GefenMuonHybrid(torch.optim.Optimizer):
    """Composite optimizer: GefenMuon on 2D hidden weights, a backup on the rest.

    See the module docstring for the recommended configuration. Two composite
    quirks to be aware of:

    * Hook semantics: INSTANCE hooks registered on the hybrid
      (``register_step_pre_hook`` / ``register_step_post_hook`` and the
      state-dict / load-state-dict pre/post hooks) fire once per hybrid
      ``step()`` / ``state_dict()`` / ``load_state_dict()``, mirroring
      ``torch.optim.Optimizer``. GLOBAL optimizer hooks
      (``torch.optim.optimizer.register_optimizer_step_pre_hook`` etc.) instead
      fire on each CHILD sub-optimizer's step -- i.e. up to twice per hybrid
      step, with the child (not the hybrid) as the ``optimizer`` argument. That
      is inherent to the composite design: the children are real
      ``torch.optim.Optimizer`` instances and their steps are what torch wraps.

    * ``state_dict()`` schema: NOT the standard ``{"state", "param_groups"}``
      layout, but ``{"muon": <GefenMuon state_dict or None>, "backup":
      <backup state_dict or None>, "backup_optimizer": "gefen" | "adamw"}``.
      ``load_state_dict`` only accepts that nested schema; checkpoints
      consolidated/converted to the flat torch layout (e.g. by FSDP/DeepSpeed
      tooling) are rejected rather than silently ignored. Legacy nested
      checkpoints without ``backup_optimizer`` are treated as Gefen-backed.
    """

    def __init__(
        self,
        muon_named_params,
        backup_named_params=None,
        *,
        lr,
        backup_substrings=None,
        muon_lr=None,
        backup_lr=None,
        backup_optimizer="gefen",
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
        batched_ns=False,
        batched_ns_workspace_bytes=256 << 20,
        stochastic_round=False,
        deterministic=False,
        normuon=True,
        normuon_beta2=0.95,
        normuon_eps=1e-8,
        cautious=False,
        capturable=False,
        verbose=False,
    ):
        """Build the hybrid from pre-split lists, a single named-param iterable, or a model.

        Three call forms:

        * Two lists (primary): ``GefenMuonHybrid(muon_named, backup_named, lr=...)``
          with the params already split into ``(name, param)`` pairs.
        * Single argument (convenience, for platform factories that call
          ``cls(params, **kwargs)``): pass an ``nn.Module`` OR a single iterable
          of ``(name, param)`` pairs (e.g. ``model.named_parameters()``) as the
          first argument and omit ``backup_named_params``; the split runs
          internally. A model enables module-type embedding detection (tied
          heads, T5 ``shared``); a bare named-param iterable falls back to
          name-substring routing. Bare tensors are rejected (they can't be
          routed). ``GefenMuonHybrid.from_model(model, ...)`` is the equivalent
          explicit classmethod.

        Args:
            muon_named_params: ``(name, param)`` pairs for the 2D hidden weight
                matrices (GefenMuon half); may be empty (backup-only hybrid). In
                the single-argument form, this is instead the ``nn.Module`` or
                the named-param iterable to split.
            backup_named_params: ``(name, param)`` pairs for everything else --
                embeddings, LM/classifier heads, norms, biases (backup half).
                May be empty (muon-only hybrid). Leave as ``None`` to use the
                single-argument form.
            lr: shared learning rate for both halves (AdamW scale under the
                default ``adjust_lr_fn="match_rms_adamw"``).
            backup_substrings: single-argument form only -- override the
                substrings that force a 2D weight to the backup half (defaults to
                ``gefen.params.DEFAULT_BACKUP_SUBSTRINGS``). Passing it with the
                two-list form is an error.
            muon_lr / backup_lr: per-half overrides of ``lr``. The low-memory
                Gefen-backup recipe lowers ``backup_lr`` to ~0.5x the shared /
                Muon rate; the quality-oriented AdamW-backup recipe keeps a
                full-rate ``backup_lr`` equal to that shared / Muon rate.
            backup_optimizer: backup algorithm for embeddings, heads, norms,
                and biases: ``"gefen"`` (default, quantized state) or
                ``"adamw"`` (``torch.optim.AdamW`` with conventional state).
            weight_decay: shared decoupled (AdamW-style) weight decay, default
                0.0; ``muon_weight_decay`` / ``backup_weight_decay`` override it
                per half.
            no_decay_substrings: names matching any substring (case-insensitive)
                form a separate backup group with weight_decay=0 (AdamW "no
                decay on norms/biases" semantics). Default: off.
            backup_1d_period_one / backup_2d_period_one: per-element 2nd moment
                on the backup's 1D / 2D tensors (loss levers; memory trade for
                the 2D one). AdamW already keeps a per-element second moment, so
                these remain accepted but are no-ops with
                ``backup_optimizer="adamw"``.
            betas, eps: Adam betas / epsilon for the backup half.
            fused: use the fused implementation in both halves (the Gefen CUDA
                kernel or ``torch.optim.AdamW(fused=True)``; default True).
            momentum, nesterov, ns_steps, ns_schedule, adjust_lr_fn,
            sharded_mode, fp8_ns, fp8_ns_compile, batched_ns,
            batched_ns_workspace_bytes, normuon, normuon_beta2, normuon_eps,
            cautious: forwarded to GefenMuon (see its docstring).
                Note ``normuon=True`` and ``ns_schedule="tuned3"`` are the
                hybrid's defaults, unlike raw GefenMuon.
            capturable: forwarded to both halves. ``stochastic_round`` and
                ``deterministic`` are forwarded to Gefen children; with an
                AdamW backup they apply only to the Muon half. ``verbose`` is
                likewise forwarded to Gefen children. Every captured parameter
                must live on the current CUDA capture device.
        """
        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a bool")
        self._deterministic = deterministic
        self.capturable = capturable
        if backup_named_params is None:
            # Single-argument convenience form: the first arg is a model or a
            # named-param iterable to split internally.
            muon_named_params, backup_named_params = self._auto_split(
                muon_named_params, backup_substrings
            )
        elif backup_substrings is not None:
            raise TypeError(
                "backup_substrings is only valid with the single-argument "
                "(model / named-params) form; you also passed backup_named_params."
            )
        muon_named_params = list(muon_named_params)
        backup_named_params = list(backup_named_params)
        if not muon_named_params and not backup_named_params:
            raise ValueError("GefenMuonHybrid received no parameters to optimize")
        # DeepSpeed ZeRO stage 3 (zero.Init) partitions parameters BEFORE the
        # optimizer is constructed: every param the hybrid then sees is a
        # 0-size 1-D placeholder, so the 2D split finds no Muon candidates and
        # would silently build a backup-only hybrid -- plain Gefen/AdamW at
        # backup_lr, no Muon ever, no warning. Real models never hand the
        # optimizer zero-numel weights, so zero Muon candidates plus any
        # zero-numel input is the ZeRO-3 signature; fail loudly. A genuinely
        # 1D-only model (all params with real storage) still constructs a
        # backup-only hybrid exactly as before.
        if not muon_named_params and backup_named_params:
            zero_numel = [
                name for name, p in backup_named_params if p.numel() == 0
            ]
            # Only the ALL-placeholder set is the ZeRO-3 signature; a mixed
            # set (e.g. an intentionally empty slot among real 1-D weights)
            # still constructs a backup-only hybrid as before.
            if len(zero_numel) == len(backup_named_params):
                raise ValueError(
                    "GefenMuonHybrid found no 2D Muon-eligible parameters, and "
                    "{} of the {} supplied parameters are zero-numel "
                    "placeholders (e.g. {!r}). This is the signature of "
                    "DeepSpeed ZeRO stage 3 (zero.Init partitions parameters "
                    "to 0-size 1-D placeholders before the optimizer is "
                    "constructed), and the hybrid would silently degrade to a "
                    "backup-only optimizer at backup_lr with no Muon half. The "
                    "Muon family cannot run under DeepSpeed ZeRO (ZeRO steps "
                    "flattened 1-D partitions; Muon's 2D orthogonalization "
                    "cannot apply) -- use plain Gefen as the DeepSpeed client "
                    "optimizer, or construct the hybrid without ZeRO "
                    "partitioning (FSDP2 / DDP / single-GPU).".format(
                        len(zero_numel), len(backup_named_params), zero_numel[0]
                    )
                )
        if backup_optimizer not in ("gefen", "adamw"):
            raise ValueError(
                "backup_optimizer must be 'gefen' or 'adamw' but is: {!r}".format(
                    backup_optimizer
                )
            )
        self.backup_optimizer = backup_optimizer
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
        # and the backup norms/embeddings/head can be tuned independently.
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
                batched_ns=batched_ns,
                batched_ns_workspace_bytes=batched_ns_workspace_bytes,
                stochastic_round=stochastic_round,
                deterministic=deterministic,
                normuon=normuon,
                normuon_beta2=normuon_beta2,
                normuon_eps=normuon_eps,
                cautious=cautious,
                capturable=capturable,
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
        else:
            backup_groups = backup_named_params

        if not backup_named_params:
            self.backup = None
        elif backup_optimizer == "gefen":
            self.backup = Gefen(
                backup_groups,
                lr=backup_lr,
                betas=betas,
                eps=eps,
                weight_decay=backup_weight_decay,
                fused=fused,
                force_1d_period_one=backup_1d_period_one,
                force_2d_period_one=backup_2d_period_one,
                # Pin plain Gefen's factored-v default off for the backup: the
                # hybrid recipe was measured with block-vmean on the
                # embedding/head, not the untested factored-v combination.
                factored_v_2d=False,
                stochastic_round=stochastic_round,
                deterministic=deterministic,
                capturable=capturable,
                verbose=verbose,
            )
        else:
            # Keep names in conventional torch param groups. Passing tensors
            # plus a parallel ``param_names`` list works across the supported
            # torch range and makes AdamW checkpoints retain the same routing
            # metadata as Gefen checkpoints.
            if no_decay_substrings:
                adamw_groups = []
                for group in backup_groups:
                    named = group["params"]
                    adamw_groups.append(
                        {
                            **{k: v for k, v in group.items() if k != "params"},
                            "params": [param for _, param in named],
                            "param_names": [name for name, _ in named],
                        }
                    )
            else:
                adamw_groups = [
                    {
                        "params": [param for _, param in backup_groups],
                        "param_names": [name for name, _ in backup_groups],
                    }
                ]
            self.backup = torch.optim.AdamW(
                adamw_groups,
                lr=backup_lr,
                betas=betas,
                eps=eps,
                weight_decay=backup_weight_decay,
                fused=fused,
                capturable=capturable,
            )
        self._subopts = [o for o in (self.muon, self.backup) if o is not None]
        # Frozen param->child ownership for the .state routing view. Keyed by
        # id() with a strong param ref (so ids can never be recycled); built
        # once here because the hybrid's param set is fixed at construction.
        self._state_param_owner = {}
        for o in self._subopts:
            for group in o.param_groups:
                for p in group["params"]:
                    self._state_param_owner[id(p)] = (p, o)

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
    def _step_supports_amp_scaling(self) -> bool:
        # Inspect the union of both halves so GradScaler chooses one protocol
        # and one overflow decision for the whole composite. FP32-master AMP
        # stays on the ordinary externally-skipped path; true FP16 uses the
        # native path because generic unscale rejects FP16 tensors. A hybrid
        # combining any true-FP16 storage with multi-rank DTensors selects that
        # protocol statically after a union-wide collective presence preflight
        # (FSDP1 FlatParameters require ShardedGradScaler).
        return _amp_native_scaling_required(self)

    @param_groups.setter
    def param_groups(self, value):
        # A composite cannot accept a wholesale param_groups assignment: the
        # groups belong to the children and an arbitrary replacement cannot be
        # split back. Without this setter the failure is a bare AttributeError
        # ("property 'param_groups' ... has no setter") -- DeepSpeed ZeRO hits
        # exactly that inside engine.step() (its _optimizer_step assigns
        # optimizer.param_groups per group), so explain the real cause.
        raise TypeError(
            "GefenMuonHybrid.param_groups cannot be assigned: the groups "
            "belong to the Muon/backup sub-optimizers. If this assignment "
            "comes from DeepSpeed ZeRO (its optimizer step re-assigns "
            "optimizer.param_groups), note that GefenMuonHybrid cannot be "
            "used as the DeepSpeed ZeRO client optimizer: ZeRO steps "
            "flattened 1-D fp32 partitions, and the Muon half's 2D "
            "orthogonalization cannot be applied to a flat partition. Use "
            "plain Gefen as the client optimizer under DeepSpeed ZeRO, or "
            "train the hybrid under FSDP2 / DDP / single-GPU instead."
        )

    @property
    def state(self):
        # A routing merged view (see _HybridMergedState): reads/writes for
        # params owned by a child reach that child's live state; access with an
        # unknown key (DeepSpeed ZeRO's flattened 1-D fp32 partitions) fails
        # fast with a self-explanatory KeyError instead of silently landing in
        # a rebuilt-per-access throwaway dict.
        return _HybridMergedState(self._subopts, self._state_param_owner)

    def zero_grad(self, set_to_none: bool = True):
        for o in self._subopts:
            o.zero_grad(set_to_none=set_to_none)

    def _assert_capturable_devices_if_capturing(self) -> None:
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if not capturing:
            return
        if not self.capturable:
            raise RuntimeError(
                "Attempting CUDA graph capture of GefenMuonHybrid.step() but "
                "capturable=False. Construct the optimizer with capturable=True "
                "to make step() graph-safe."
            )
        devices = {
            param.device
            for optimizer in self._subopts
            for group in optimizer.param_groups
            for param in group["params"]
        }
        capture_device = torch.device("cuda", torch.cuda.current_device())
        if devices != {capture_device}:
            raise RuntimeError(
                "CUDA graph capture requires every GefenMuonHybrid parameter on "
                "the current capture device {}; found {}".format(
                    capture_device, sorted(map(str, devices))
                )
            )

    @torch._dynamo.disable
    def _step_failure_process_groups(self):
        """Failure-sync scope for the composite preflight: the UNION of both
        children's sharded-mesh (process_group, device) pairs.

        The composite preflight (closure + structural gradient validation + AMP
        controls) covers BOTH children, so the pre-collective failure sync must
        span every sharded mesh EITHER child owns -- not only the Muon child's.
        Deriving the scope from the Muon child alone missed any mesh owned only
        by the backup child (a sharded backup weight whose Muon half is
        non-sharded or absent): a one-rank preflight failure on that backup-only
        mesh then raised on the failing rank while its mesh peers proceeded to
        step and mutate their shard, diverging cross-rank state. Both children's
        param_groups are folded through ONE deduped, sorted scan
        (``GefenMuon._collect_sharded_failure_groups``), so the standard
        fully_shard case -- both halves sharded on the SAME mesh -- collapses to
        exactly the Muon-only scope with no extra collective, while a
        backup-only mesh is included in one deterministic cross-rank order.
        """
        if not GefenMuon._dist_available():
            return ()
        param_groups = [
            group
            for optimizer in self._subopts
            for group in optimizer.param_groups
        ]
        params = [param for group in param_groups for param in group["params"]]
        # The protocol returns host-readable flags and cannot be captured; a
        # captured step already requires an eager warmup with fixed control flow
        # (mirrors GefenMuon._step_failure_process_groups).
        if any(param.device.type == "cuda" for param in params) and (
            torch.cuda.is_current_stream_capturing()
        ):
            return ()
        return GefenMuon._collect_sharded_failure_groups(param_groups)

    def step(self, closure=None):
        process_groups = self._step_failure_process_groups()
        # Dispatch the INSTANCE step hooks around the composite step, mirroring
        # torch.optim.Optimizer.profile_hook_step exactly: hooks receive
        # (optimizer, args, kwargs) where args are the raw step() call args
        # (self first), and a pre-hook may return replacement (args, kwargs).
        # The hybrid skips Optimizer.__init__ (so its step is never wrapped by
        # _patch_step_function); without this, registered hooks would silently
        # never fire. GLOBAL step hooks are NOT dispatched here -- they fire on
        # each child sub-optimizer's (wrapped) step; see the class docstring.
        args = (self, closure) if closure is not None else (self,)
        kwargs = {}
        loss = None
        try:
            self._assert_capturable_devices_if_capturing()
            for pre_hook in self._optimizer_step_pre_hooks.values():
                result = pre_hook(self, args, kwargs)
                if result is not None:
                    if isinstance(result, tuple) and len(result) == 2:
                        args, kwargs = result
                    else:
                        raise RuntimeError(
                            f"{self.__class__.__name__}.step pre hook must return None "
                            f"or a tuple of (new_args, new_kwargs), but got {result}."
                        )
            # Re-read the closure from the (possibly hook-rewritten) call args,
            # as torch's wrapper would by calling step(*args, **kwargs).
            closure = args[1] if len(args) > 1 else kwargs.get("closure")
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for child in self._subopts:
                _assert_optimizer_gradients_structurally_valid(
                    child, require_2d_params=child is self.muon
                )
            local_preflight_error = None
        except Exception as exc:
            loss = None
            local_preflight_error = exc
        if self.muon is not None:
            self.muon._synchronize_sharded_step_error(
                local_preflight_error, "hybrid step preflight", process_groups
            )
        elif local_preflight_error is not None:
            raise local_preflight_error

        # A non-finite gradient in either half skips BOTH children before their
        # codebooks, states, counters, or parameters can move. Explicit
        # scaler.unscale_(hybrid) is detected by grad_scale=None and is not
        # repeated; automatic unscale covers every child parameter exactly once.
        if self.muon is not None:
            should_step = self.muon._prepare_synchronized_amp_step(
                self, process_groups
            )
        elif hasattr(self, "found_inf") or hasattr(self, "grad_scale"):
            should_step = _amp_prepare_optimizer_step(self)
        else:
            should_step = True
        if not should_step:
            for post_hook in self._optimizer_step_post_hooks.values():
                post_hook(self, args, kwargs)
            return loss

        with torch.no_grad():
            for o in self._subopts:
                if o is not self.muon:
                    o.step()
                    continue
                marker = object()
                previous = getattr(
                    o, "_gefen_hybrid_precollective_preflight", marker
                )
                o._gefen_hybrid_precollective_preflight = True
                try:
                    o.step()
                finally:
                    if previous is marker:
                        del o._gefen_hybrid_precollective_preflight
                    else:
                        o._gefen_hybrid_precollective_preflight = previous

        for post_hook in self._optimizer_step_post_hooks.values():
            post_hook(self, args, kwargs)
        return loss

    def state_dict(self):
        # Instance state-dict pre/post hooks, mirroring Optimizer.state_dict:
        # pre-hooks take (optimizer) and return nothing; a post-hook may return
        # a replacement state_dict.
        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)
        state_dict = {
            "muon": self.muon.state_dict() if self.muon is not None else None,
            "backup": self.backup.state_dict() if self.backup is not None else None,
            "backup_optimizer": self.backup_optimizer,
        }
        for post_hook in self._optimizer_state_dict_post_hooks.values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        return state_dict

    def load_state_dict(self, state_dict):
        # Instance load pre-hooks first (a pre-hook may return a replacement
        # dict -- e.g. one that converts a foreign schema), mirroring
        # Optimizer.load_state_dict's shallow copy + hook pass.
        state_dict = state_dict.copy()
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Schema guard: this used to silently skip loading whenever the keys
        # were absent, so resuming from a standard {"state", "param_groups"}
        # checkpoint (e.g. one consolidated/converted by FSDP or DeepSpeed
        # tooling) ran on with ZEROED momentum -- a quiet correctness bug.
        if "muon" not in state_dict and "backup" not in state_dict:
            raise ValueError(
                "GefenMuonHybrid.load_state_dict expects the hybrid's own nested "
                'schema {{"muon": <GefenMuon state dict or None>, "backup": '
                '<backup state dict or None>, "backup_optimizer": "gefen" | '
                '"adamw"}} (what GefenMuonHybrid.state_dict() '
                "saves), but got keys {}. Consolidated/converted checkpoints in "
                'the standard {{"state", "param_groups"}} layout are not '
                "supported; resume from the hybrid-saved checkpoint "
                "instead.".format(sorted(map(str, state_dict.keys())))
            )
        # A half present here but missing/None in the checkpoint (or vice
        # versa) would silently resume that half from scratch (zeroed
        # momentum), so reject those presence mismatches before loading either
        # child. The backup-backend check below runs before child loads too.
        for attr in ("muon", "backup"):
            sub = getattr(self, attr)
            sub_state = state_dict.get(attr)
            if sub is None and sub_state is not None:
                raise ValueError(
                    "GefenMuonHybrid.load_state_dict: the checkpoint carries a "
                    '"{}" half but this optimizer has none (was it constructed '
                    "with a different parameter split?)".format(attr)
                )
            if sub is not None and sub_state is None:
                raise ValueError(
                    "GefenMuonHybrid.load_state_dict: this optimizer has a "
                    '"{}" half but the checkpoint carries none; loading would '
                    "silently reset its momentum/state".format(attr)
                )
        # Old hybrid checkpoints predate the selector and necessarily contain a
        # Gefen backup. Guard the backend before loading either child: torch's
        # optimizer loader otherwise accepts the other backend's group/state
        # layout and fails only on a later step (or, worse, misinterprets state).
        if self.backup is not None:
            checkpoint_backup_optimizer = state_dict.get(
                "backup_optimizer", "gefen"
            )
            if checkpoint_backup_optimizer != self.backup_optimizer:
                raise ValueError(
                    "GefenMuonHybrid.load_state_dict: checkpoint backup_optimizer "
                    "is {!r}, but this optimizer uses {!r}; construct the hybrid "
                    "with the checkpoint's backup_optimizer instead.".format(
                        checkpoint_backup_optimizer, self.backup_optimizer
                    )
                )
        # Load the two children with two-phase composite semantics: validate and
        # stage BOTH halves before publishing EITHER, so a rejection on either
        # half (a different param count, a foreign layout, a corrupted/truncated
        # child state) leaves both live children byte-for-byte untouched. The
        # previous code committed the muon child first and only then loaded the
        # backup, recovering via a second full ``muon.load_state_dict(snapshot)``
        # reload -- a rollback that could itself raise (e.g. CUDA OOM re-staging
        # every muon tensor), masking the real backup error and leaving a
        # half-loaded hybrid (new muon, old backup).
        #
        # Each Gefen child already loads atomically via a stage-then-swap
        # primitive (``_prepare_load_state_dict`` returns an isolated shadow;
        # ``_publish_load_state_dict`` publishes it through non-throwing dict
        # swaps). Staging is process-group-safe: ``_stage_load_state_dict``
        # shallow-copies the child's ``__dict__`` and swaps in fresh state
        # containers -- it never deepcopies the live optimizer, so a codebook
        # process-group handle (or any live ProcessGroup) is shared by reference,
        # never duplicated. We deliberately avoid ``copy.deepcopy`` of a child or
        # its process groups for exactly that reason.
        # Phase one -- stage every child (validate, mutating nothing live). Phase
        # two -- commit every child's raw state through non-throwing swaps. Phase
        # three -- only then dispatch any child's load post-hooks. Separating the
        # raw commit (phase two) from the post-hooks (phase three) matters: a
        # post-hook can raise, and if it fired while a sibling child were still
        # uncommitted it would strand the hybrid half-loaded (e.g. muon step N /
        # backup step N-1). The foreign backup gets the same isolated staging as
        # the Gefen children -- torch ``AdamW.load_state_dict`` is NOT itself
        # fail-before-mutation (``__setstate__`` installs the new state via
        # ``super().__setstate__`` and only then reads ``state[0]["step"]``, so a
        # checkpoint whose per-parameter state omits ``step`` raises
        # ``KeyError("step")`` after the live backup has already been mutated).
        muon_staged = None
        if self.muon is not None:
            muon_staged = self.muon._prepare_load_state_dict(state_dict["muon"])

        if self.backup is None:
            # muon-only hybrid.
            if muon_staged is not None:
                self.muon._commit_staged_load_state_dict(muon_staged)
                self.muon._run_load_state_dict_post_hooks()
        elif isinstance(self.backup, Gefen):
            backup_staged = self.backup._prepare_load_state_dict(state_dict["backup"])
            if muon_staged is not None:
                self.muon._commit_staged_load_state_dict(muon_staged)
            self.backup._commit_staged_load_state_dict(backup_staged)
            if muon_staged is not None:
                self.muon._run_load_state_dict_post_hooks()
            self.backup._run_load_state_dict_post_hooks()
        else:
            # Foreign backup (torch ``AdamW``): stage it on an isolated shadow
            # too so a mid-load raise leaves the live backup byte-for-byte
            # untouched, exactly like the Gefen children.
            backup_shadow = self._stage_foreign_backup_load(
                self.backup, state_dict["backup"]
            )
            if muon_staged is not None:
                self.muon._commit_staged_load_state_dict(muon_staged)
            self._commit_foreign_backup_load(self.backup, backup_shadow)
            if muon_staged is not None:
                self.muon._run_load_state_dict_post_hooks()
            self._run_foreign_backup_post_hooks(self.backup)

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @staticmethod
    def _stage_foreign_backup_load(backup, state_dict):
        """Stage a foreign (non-Gefen) backup load on an isolated shadow.

        ``torch.optim.Optimizer.load_state_dict`` is not fail-before-mutation for
        every optimizer: torch ``AdamW``'s ``__setstate__`` installs the new
        state via ``super().__setstate__`` and only then reads
        ``state_values[0]["step"]``, so a checkpoint whose per-parameter state
        omits ``step`` raises ``KeyError("step")`` after the live backup has
        already been mutated. Build the full restore on a shallow shadow first
        (never a ``copy.deepcopy`` of the live state or any process group), so a
        rejection leaves the live backup byte-for-byte untouched. Returns the
        shadow; publish it with ``_commit_foreign_backup_load``.
        """

        # Mirror Optimizer.load_state_dict's pre-hook + shallow-copy pass, then
        # run the base load on the shadow with its own hook maps emptied so the
        # staging build is side-effect free (post-hooks are deferred to the
        # publish step); the raise, if any, happens here on the shadow.
        state_dict = state_dict.copy()
        for pre_hook in backup._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(backup, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        shadow = object.__new__(type(backup))
        shadow.__dict__ = backup.__dict__.copy()
        shadow.defaults = backup.defaults.copy()
        shadow.state = defaultdict(dict)
        shadow._optimizer_load_state_dict_pre_hooks = OrderedDict()
        shadow._optimizer_load_state_dict_post_hooks = OrderedDict()
        torch.optim.Optimizer.load_state_dict(shadow, state_dict)
        return shadow

    @staticmethod
    def _commit_foreign_backup_load(backup, shadow):
        """Publish a staged foreign-backup restore through non-throwing swaps."""

        live_defaults = backup.defaults
        live_defaults.update(shadow.defaults)
        backup.state = shadow.state
        backup.param_groups = shadow.param_groups

    @staticmethod
    def _run_foreign_backup_post_hooks(backup):
        for post_hook in backup._optimizer_load_state_dict_post_hooks.values():
            post_hook(backup)

    @staticmethod
    def _auto_split(params_or_model, backup_substrings):
        """Split a model / named-param iterable into (muon_named, backup_named).

        A model routes by module type (embeddings, incl. tied heads) plus name
        substrings and shape; a named-param iterable routes by name+shape only
        (no module context -- an embedding whose name matches no substring, e.g.
        T5 ``shared``, would slip to Muon, so a model is preferred). Bare tensors
        raise, since they carry neither names nor module type to route on.
        """
        subs = (
            DEFAULT_BACKUP_SUBSTRINGS
            if backup_substrings is None
            else tuple(backup_substrings)
        )
        if isinstance(params_or_model, nn.Module):
            muon_named, backup_named = split_params_for_muon(
                params_or_model, backup_substrings=subs
            )
            validate_split(muon_named, backup_named, model=params_or_model)
            return muon_named, backup_named
        items = list(params_or_model)
        for item in items:
            if not (
                isinstance(item, (tuple, list))
                and len(item) == 2
                and isinstance(item[0], str)
                and isinstance(item[1], torch.Tensor)
            ):
                raise TypeError(
                    "GefenMuonHybrid's single-argument form needs an nn.Module or "
                    "an iterable of (name, param) pairs (e.g. model.named_parameters()); "
                    "got a bare {}. Bare tensors can't be routed (embeddings/heads "
                    "would go to Muon) -- pass the model or use "
                    "GefenMuonHybrid.from_model(model, ...).".format(
                        type(item).__name__
                    )
                )
        muon_named, backup_named = [], []
        for name, param in items:
            if not param.requires_grad:
                continue
            target = muon_named if is_muon_param(name, param, subs) else backup_named
            target.append((name, param))
        logger.info(
            "GefenMuonHybrid auto-split (name-based -- pass an nn.Module for "
            "module-type embedding detection): %d muon / %d backup",
            len(muon_named),
            len(backup_named),
        )
        validate_split(muon_named, backup_named)
        return muon_named, backup_named

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
        nm = (
            sum(len(group["params"]) for group in self.muon.param_groups)
            if self.muon is not None
            else 0
        )
        nb = (
            sum(len(group["params"]) for group in self.backup.param_groups)
            if self.backup is not None
            else 0
        )
        return (
            f"GefenMuonHybrid(muon_params={nm}, backup_params={nb}, "
            f"backup_optimizer={self.backup_optimizer!r})"
        )
