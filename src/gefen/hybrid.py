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

import copy
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from gefen.codebook import CodebookProcessGroupBinding
from gefen.contracts import (
    LogicalSlice,
    OptimizerContract,
    ParameterIdentity,
    ParameterLayout,
    ShardIdentity,
    ShardingManifest,
    _hybrid_contract,
)
from gefen.gefen import (
    Gefen,
    _amp_native_scaling_required,
    _assert_optimizer_gradients_structurally_valid,
)
from gefen.gefen_muon import GefenMuon
from gefen.params import (
    DEFAULT_BACKUP_SUBSTRINGS,
    is_muon_param,
    split_params_for_muon,
    validate_split,
)
from gefen.rebinding import ParameterRebinding

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
            muon_named_params, backup_named_params = self._auto_split(muon_named_params, backup_substrings)
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
            zero_numel = [name for name, p in backup_named_params if p.numel() == 0]
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
            raise ValueError("backup_optimizer must be 'gefen' or 'adamw' but is: {!r}".format(backup_optimizer))
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
                "to share one AdamW-scale lr, or set muon_lr explicitly.".format(adjust_lr_fn),
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
        backup_weight_decay = weight_decay if backup_weight_decay is None else backup_weight_decay

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

        # Composite stable identity is installed only after every present Gefen
        # child has staged the same complete post-sharding transaction. These
        # fields remain separate from either child's rank-local metadata so a
        # failed second-child preparation cannot leave the Hybrid half-bound.
        self._hybrid_post_sharding_finalized = False
        self._hybrid_sharding_manifest = None
        self._hybrid_local_shard_bindings = ()
        self._hybrid_fqn_roles = ()
        self._hybrid_codebook_process_group = None
        self._hybrid_finalized_slots = ()

        # Composite finalized-layout forensics cache, mirroring the O(local
        # params) scheme Gefen/GefenMuon use for their own step guards. The
        # version counter is bumped by the only hybrid API that reassigns any
        # of the finalized composite fields (post_sharding, which every rebind
        # helper routes through); the cached verdict is one identity-token
        # snapshot of everything the composite forensic rebuild reads.
        self._hybrid_layout_version = 0
        self._hybrid_layout_forensics_verdict = None

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
        self._assert_finalized_binding_layout()
        return _HybridMergedState(self._subopts, self._state_param_owner)

    def _gefen_rebinding_children(self):
        children = []
        if self.muon is not None:
            if type(self.muon) is not GefenMuon:
                raise TypeError("GefenMuonHybrid post_sharding requires an exact GefenMuon child")
            children.append(("muon", self.muon))
        if self.backup is not None:
            if self.backup_optimizer != "gefen" or type(self.backup) is not Gefen:
                raise NotImplementedError(
                    "GefenMuonHybrid post_sharding does not yet support an AdamW "
                    "backup; use backup_optimizer='gefen' until AdamW has stable "
                    "rebinding identity and atomic staged state I/O"
                )
            children.append(("backup", self.backup))
        if not children:
            raise RuntimeError("GefenMuonHybrid has no child optimizer to rebind")
        return tuple(children)

    @staticmethod
    def _same_local_binding(left, right) -> bool:
        return left[0] is right[0] and left[1] == right[1]

    @staticmethod
    def _reject_rebinding_method_shadows(value) -> None:
        if type(value.__dict__) is not dict:
            raise TypeError("GefenMuonHybrid rebinding requires exact attribute dictionaries")
        for name in value.__dict__:
            descriptor = None
            for owner in type(value).__mro__:
                if name in owner.__dict__:
                    descriptor = owner.__dict__[name]
                    break
            if isinstance(descriptor, (staticmethod, classmethod)):
                descriptor = descriptor.__func__
            if callable(descriptor):
                raise TypeError("GefenMuonHybrid rebinding rejects instance-level method shadows")

    def _hybrid_identity_metadata_empty(self) -> bool:
        return (
            self._hybrid_sharding_manifest is None
            and self._hybrid_local_shard_bindings == ()
            and self._hybrid_fqn_roles == ()
            and self._hybrid_codebook_process_group is None
            and self._hybrid_finalized_slots == ()
        )

    def _assert_composite_rebinding_pristine(self, children) -> None:
        if self._hybrid_post_sharding_finalized:
            raise RuntimeError("GefenMuonHybrid post-sharding identity is already finalized")
        if not self._hybrid_identity_metadata_empty():
            raise RuntimeError("GefenMuonHybrid parameter rebinding found an incomplete prior identity plan")
        if type(self._subopts) is not list or len(self._subopts) != len(children):
            raise RuntimeError("GefenMuonHybrid child optimizer order changed before post_sharding")
        if any(live is not expected for live, (_role, expected) in zip(self._subopts, children)):
            raise RuntimeError("GefenMuonHybrid child optimizer order changed before post_sharding")
        if type(self._state_param_owner) is not dict:
            raise TypeError("GefenMuonHybrid parameter ownership must use an exact dictionary")
        child_set = {child for _role, child in children}
        owner_counts = {child: 0 for child in child_set}
        for parameter_id, live in self._state_param_owner.items():
            if (
                type(live) is not tuple
                or len(live) != 2
                or not isinstance(live[0], torch.Tensor)
                or parameter_id != id(live[0])
                or live[1] not in child_set
            ):
                raise RuntimeError("GefenMuonHybrid parameter ownership changed before post_sharding")
            owner_counts[live[1]] += 1
        for _role, child in children:
            live_slot_count = sum(len(group["params"]) for group in child.param_groups)
            if owner_counts[child] != live_slot_count:
                raise RuntimeError("GefenMuonHybrid child slot counts changed before post_sharding")

    def _stage_post_sharding(
        self,
        rebindings,
        manifest: ShardingManifest,
        codebook_process_group=None,
    ):
        GefenMuonHybrid._reject_rebinding_method_shadows(self)
        children = GefenMuonHybrid._gefen_rebinding_children(self)
        for _role, child in children:
            GefenMuonHybrid._reject_rebinding_method_shadows(child)
        GefenMuonHybrid._assert_composite_rebinding_pristine(self, children)
        if type(manifest) is not ShardingManifest:
            raise TypeError("manifest must be an exact ShardingManifest")
        validated_manifest = ShardingManifest(
            manifest.shards,
            schema_version=manifest.schema_version,
        )
        if validated_manifest != manifest:
            raise ValueError("GefenMuonHybrid requires a canonical manifest")
        if type(rebindings) is not tuple or not rebindings:
            raise TypeError("rebindings must be a non-empty tuple of ParameterRebinding values")
        if any(type(item) is not ParameterRebinding for item in rebindings):
            raise TypeError("rebindings must contain exact ParameterRebinding values")
        if codebook_process_group is not None and type(codebook_process_group) is not CodebookProcessGroupBinding:
            raise TypeError("codebook_process_group must be an exact CodebookProcessGroupBinding")

        source_entries = {}
        source_roles = {}
        child_by_role = dict(children)
        for parameter_id, (parameter, child) in self._state_param_owner.items():
            if parameter_id != id(parameter):
                raise RuntimeError("GefenMuonHybrid parameter ownership contains a stale key")
            role = "muon" if child is self.muon else "backup"
            if role not in child_by_role or child_by_role[role] is not child:
                raise RuntimeError("GefenMuonHybrid parameter ownership names a foreign child")
            source_entries[parameter_id] = parameter
            source_roles[parameter_id] = role
        if len(rebindings) != len(source_entries):
            raise ValueError(
                "GefenMuonHybrid post_sharding requires exactly one rebinding for every original child slot"
            )

        seen_sources = set()
        seen_targets = set()
        seen_fqns = set()
        targets = []
        by_role = {role: [] for role, _child in children}
        fqn_roles = {}
        for rebinding in rebindings:
            source = rebinding.old_parameter
            source_id = id(source)
            if source_id not in source_entries or source_entries[source_id] is not source:
                raise ValueError("GefenMuonHybrid rebinding source is not an original child slot")
            if source_id in seen_sources:
                raise ValueError("GefenMuonHybrid rebinding source tensors must be unique")
            seen_sources.add(source_id)
            target = rebinding.new_parameter
            if target is not None:
                if not isinstance(target, torch.Tensor):
                    raise TypeError("GefenMuonHybrid rebinding target must be a Tensor or None")
                target_id = id(target)
                if target_id in seen_targets:
                    raise ValueError("GefenMuonHybrid rebound target tensors must be unique")
                seen_targets.add(target_id)
                original_target = source_entries.get(target_id)
                if original_target is not None and original_target is not source:
                    raise ValueError("GefenMuonHybrid rebound targets cannot steal another original child slot")
                targets.append(target)
            fqn = rebinding.shard.parameter.fqn
            if fqn in seen_fqns:
                raise ValueError("GefenMuonHybrid local canonical parameter FQNs must be unique")
            seen_fqns.add(fqn)
            role = source_roles[source_id]
            fqn_roles[fqn] = role
            by_role[role].append(rebinding)

        if seen_sources != set(source_entries):
            raise ValueError("GefenMuonHybrid post_sharding did not bind every original child slot")
        Gefen._assert_rebound_storage_disjoint(targets)

        manifest_fqns = {shard.parameter.fqn for shard in manifest.shards}
        if seen_fqns != manifest_fqns:
            raise ValueError("GefenMuonHybrid manifest FQNs must exactly match all child slots")
        if codebook_process_group is not None:
            if any(shard.process_group != codebook_process_group.identity for shard in manifest.shards):
                raise ValueError("every Hybrid manifest shard must use the shared codebook process-group identity")
            if any(item.shard.local_member != codebook_process_group.local_member for item in rebindings):
                raise ValueError("every Hybrid local shard must match the shared codebook member")

        staged_children = []
        for role, child in children:
            child_rebindings = tuple(by_role[role])
            if not child_rebindings:
                raise ValueError("GefenMuonHybrid child {!r} has no routed rebinding".format(role))
            child_fqns = {item.shard.parameter.fqn for item in child_rebindings}
            child_manifest = ShardingManifest(
                tuple(shard for shard in manifest.shards if shard.parameter.fqn in child_fqns),
                schema_version=manifest.schema_version,
            )
            staged = child._stage_post_sharding(
                child_rebindings,
                child_manifest,
                codebook_process_group,
            )
            if (
                type(child.__dict__) is not dict
                or type(staged.__dict__) is not dict
                or staged._gefen_codebook_process_group is not codebook_process_group
                or not staged._finalized_binding_layout_matches()
            ):
                raise TypeError("GefenMuonHybrid child staging produced an unsafe finalized optimizer")
            staged_children.append((role, child, staged, child_manifest))

        local_bindings = []
        local_fqns = set()
        new_state_param_owner = {}
        finalized_slots = []
        for role, child, staged, _child_manifest in staged_children:
            finalized_slots.append(
                (
                    role,
                    tuple(tuple(group["params"]) for group in staged.param_groups),
                )
            )
            for parameter, shard in staged._gefen_local_shard_bindings:
                fqn = shard.parameter.fqn
                if fqn in local_fqns or fqn_roles.get(fqn) != role:
                    raise ValueError("GefenMuonHybrid staged child identities overlap or changed routing")
                local_fqns.add(fqn)
                local_bindings.append((parameter, shard))
            for group in staged.param_groups:
                for parameter in group["params"]:
                    parameter_id = id(parameter)
                    if parameter_id in new_state_param_owner:
                        raise ValueError("GefenMuonHybrid staged children share a live parameter")
                    new_state_param_owner[parameter_id] = (parameter, child)
        if local_fqns != seen_fqns:
            raise ValueError("GefenMuonHybrid staged children do not cover the full manifest")
        local_bindings.sort(key=lambda item: item[1].sort_key)
        if type(self.__dict__) is not dict:
            raise TypeError("GefenMuonHybrid publication requires an exact attribute dictionary")
        return {
            "children": tuple(staged_children),
            "state_param_owner": new_state_param_owner,
            "manifest": manifest,
            "local_bindings": tuple(local_bindings),
            "fqn_roles": tuple(sorted(fqn_roles.items())),
            "codebook_process_group": codebook_process_group,
            "finalized_slots": tuple(finalized_slots),
        }

    def post_sharding(
        self,
        rebindings,
        *,
        manifest: ShardingManifest,
        codebook_process_group=None,
    ) -> None:
        """Atomically finalize every present Gefen child after sharding."""

        if isinstance(rebindings, (str, bytes)):
            raise TypeError("rebindings must be a sequence")
        try:
            rebindings = tuple(rebindings)
        except TypeError as exc:
            raise TypeError("rebindings must be a sequence") from exc
        staged = GefenMuonHybrid._stage_post_sharding(
            self,
            rebindings,
            manifest,
            codebook_process_group,
        )
        for _role, child, staged_child, _child_manifest in staged["children"]:
            dict.update(child.__dict__, staged_child.__dict__)
        dict.update(
            self.__dict__,
            {
                "_state_param_owner": staged["state_param_owner"],
                "_hybrid_post_sharding_finalized": True,
                "_hybrid_sharding_manifest": staged["manifest"],
                "_hybrid_local_shard_bindings": staged["local_bindings"],
                "_hybrid_fqn_roles": staged["fqn_roles"],
                "_hybrid_codebook_process_group": staged["codebook_process_group"],
                "_hybrid_finalized_slots": staged["finalized_slots"],
                # Reassigning the composite fields invalidates any warm verdict
                # (there is none from a pristine hybrid, but bumping keeps the
                # counter honest and forces the next guard through a full
                # rebuild before it caches a fresh verdict).
                "_hybrid_layout_version": getattr(self, "_hybrid_layout_version", 0)
                + 1,
                "_hybrid_layout_forensics_verdict": None,
            },
        )

    def rebind_shard(
        self,
        old_parameter,
        new_parameter,
        *,
        shard: ShardIdentity,
        manifest: ShardingManifest,
    ) -> None:
        """Apply a complete one-slot Hybrid shard plan."""

        self.post_sharding(
            (ParameterRebinding(old_parameter, new_parameter, shard),),
            manifest=manifest,
        )

    def rebind_parameter(
        self,
        old_parameter,
        new_parameter,
        *,
        identity: ParameterIdentity,
    ) -> None:
        """Bind one complete replicated parameter on a one-slot Hybrid."""

        if type(identity) is not ParameterIdentity:
            raise TypeError("identity must be an exact ParameterIdentity")
        shard = ShardIdentity(
            identity,
            ParameterLayout.REPLICATED,
            LogicalSlice.full(identity),
        )
        self.rebind_shard(
            old_parameter,
            new_parameter,
            shard=shard,
            manifest=ShardingManifest((shard,)),
        )

    @staticmethod
    def _hybrid_child_param_group_tokens(child, tokens):
        # Append the child's live group containers, their ``params`` list
        # objects, and every parameter in them so an in-place slot swap in a
        # child's public param_groups is visible to the composite fast path.
        tokens.append(child)
        tokens.append(getattr(child, "_gefen_sharding_manifest", None))
        tokens.append(getattr(child, "_gefen_codebook_process_group", None))
        tokens.append(getattr(child, "_gefen_local_shard_bindings", None))
        tokens.append(getattr(child, "defaults", None))
        groups = child.param_groups
        tokens.append(groups)
        tokens.append(len(groups))
        for group in groups:
            params = group.get("params") if type(group) is dict else None
            tokens.append(group)
            tokens.append(params)
            if isinstance(params, (list, tuple)):
                tokens.append(len(params))
                tokens.extend(params)

    def _hybrid_layout_forensics_fast_tokens(self):
        # O(local params) identity snapshot of everything the composite
        # forensic rebuild reads by identity, plus each child's own O(local)
        # cached fast-path verdict. Calling the children's fast path means any
        # child-level change they can detect (an in-place ``group['params']``
        # slot swap, a replaced private child registry, a mutated name cache)
        # flips a bool in this token and forces the composite through a full
        # rebuild; the composite-owned fields are captured directly so a
        # replaced manifest/roles/local-bindings/slots/owner/binding container
        # is caught even if no child noticed. Legitimate mutating APIs replace
        # these containers (never mutate them in place) and bump the version
        # counter, so an unchanged attribute is the same object.
        live = [
            getattr(self, "_hybrid_layout_version", 0),
            self._hybrid_post_sharding_finalized,
            self._hybrid_sharding_manifest,
            self._hybrid_local_shard_bindings,
            self._hybrid_fqn_roles,
            self._hybrid_finalized_slots,
            self._hybrid_codebook_process_group,
            self._state_param_owner,
            len(self._state_param_owner),
            self.defaults,
            self._subopts,
            len(self._subopts),
        ]
        for child in self._subopts:
            live.append(child._finalized_binding_layout_matches())
            GefenMuonHybrid._hybrid_child_param_group_tokens(child, live)
        return tuple(live)

    def _finalized_binding_layout_matches(self, *, full: bool = False) -> bool:
        # Steady-state guard: reuse one cached verdict validated by an
        # O(local params) identity-token check. Any real finalized-layout
        # change either bumps the hybrid version, replaces a composite
        # container, or flips a child's own fast-path verdict -- all captured
        # by the token -- so a stale hit is impossible; a full boundary check
        # (``full=True`` from base contract-readiness call sites) always
        # rebuilds. Detection stays before any mutation.
        verdict = getattr(self, "_hybrid_layout_forensics_verdict", None)
        if not full and verdict is not None:
            try:
                fast_tokens = self._hybrid_layout_forensics_fast_tokens()
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                fast_tokens = None
            if fast_tokens is not None and Gefen._forensics_tokens_match(
                verdict, fast_tokens
            ):
                return True
        self._hybrid_layout_forensics_verdict = None
        matches = self._finalized_binding_layout_matches_full()
        if matches:
            try:
                self._hybrid_layout_forensics_verdict = (
                    self._hybrid_layout_forensics_fast_tokens()
                )
            except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
                self._hybrid_layout_forensics_verdict = None
        return matches

    def _finalized_binding_layout_matches_full(self) -> bool:
        try:
            if (
                not self._hybrid_post_sharding_finalized
                or type(self._hybrid_sharding_manifest) is not ShardingManifest
                or type(self._hybrid_local_shard_bindings) is not tuple
                or type(self._hybrid_fqn_roles) is not tuple
                or type(self._hybrid_finalized_slots) is not tuple
                or type(self._state_param_owner) is not dict
            ):
                return False
            children = self._gefen_rebinding_children()
            if (
                type(self._subopts) is not list
                or len(self._subopts) != len(children)
                or any(live is not child for live, (_role, child) in zip(self._subopts, children))
                or self.defaults is not children[0][1].defaults
            ):
                return False
            role_by_fqn = dict(self._hybrid_fqn_roles)
            if (
                len(role_by_fqn) != len(self._hybrid_fqn_roles)
                or tuple(sorted(role_by_fqn.items())) != self._hybrid_fqn_roles
                or any(role not in {"muon", "backup"} for role in role_by_fqn.values())
            ):
                return False
            manifest_fqns = {shard.parameter.fqn for shard in self._hybrid_sharding_manifest.shards}
            if manifest_fqns != set(role_by_fqn):
                return False
            binding = self._hybrid_codebook_process_group
            if binding is not None:
                if type(binding) is not CodebookProcessGroupBinding:
                    return False
                if any(shard.process_group != binding.identity for shard in self._hybrid_sharding_manifest.shards):
                    return False

            expected_local = []
            expected_owner = {}
            expected_slots = []
            for role, child in children:
                if (
                    not child._finalized_binding_layout_matches()
                    or child._gefen_codebook_process_group is not self._hybrid_codebook_process_group
                ):
                    return False
                expected_child_shards = tuple(
                    shard
                    for shard in self._hybrid_sharding_manifest.shards
                    if role_by_fqn.get(shard.parameter.fqn) == role
                )
                if child._gefen_sharding_manifest.shards != expected_child_shards:
                    return False
                expected_slots.append(
                    (
                        role,
                        tuple(tuple(group["params"]) for group in child.param_groups),
                    )
                )
                for parameter, shard in child._gefen_local_shard_bindings:
                    if role_by_fqn.get(shard.parameter.fqn) != role:
                        return False
                    if binding is not None and (
                        shard.process_group != binding.identity or shard.local_member != binding.local_member
                    ):
                        return False
                    expected_local.append((parameter, shard))
                for group in child.param_groups:
                    for parameter in group["params"]:
                        parameter_id = id(parameter)
                        if parameter_id in expected_owner:
                            return False
                        expected_owner[parameter_id] = (parameter, child)
            expected_local.sort(key=lambda item: item[1].sort_key)
            if len(expected_local) != len(self._hybrid_local_shard_bindings):
                return False
            if any(
                not self._same_local_binding(live, expected)
                for live, expected in zip(
                    self._hybrid_local_shard_bindings,
                    expected_local,
                )
            ):
                return False
            if tuple(expected_slots) != self._hybrid_finalized_slots:
                return False
            if set(expected_owner) != set(self._state_param_owner):
                return False
            for parameter_id, (parameter, child) in expected_owner.items():
                live = self._state_param_owner[parameter_id]
                if type(live) is not tuple or len(live) != 2 or live[0] is not parameter or live[1] is not child:
                    return False
            return True
        except (
            AttributeError,
            KeyError,
            NotImplementedError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            return False

    def _canonical_identity_ready(self) -> bool:
        # Contract readiness is an honest external claim, so it never trusts the
        # per-step fast-path verdict and always runs the full forensic rebuild.
        return self._hybrid_post_sharding_finalized and self._finalized_binding_layout_matches(
            full=True
        )

    def _codebook_scope_ready(self) -> bool:
        return self._hybrid_codebook_process_group is not None and self._canonical_identity_ready()

    def _assert_finalized_binding_layout(self, *, full: bool = False) -> None:
        if self._hybrid_post_sharding_finalized:
            if not self._finalized_binding_layout_matches(full=full):
                raise RuntimeError("GefenMuonHybrid finalized parameter layout changed outside post_sharding")
        elif not self._hybrid_identity_metadata_empty():
            raise RuntimeError("GefenMuonHybrid found an incomplete post_sharding identity plan")

    def parameter_identity(self, parameter) -> ParameterIdentity:
        """Return the canonical identity bound to one live Hybrid parameter."""

        return self.shard_identity(parameter).parameter

    def shard_identity(self, parameter) -> ShardIdentity:
        """Return the stable shard identity bound to one live parameter."""

        self._assert_finalized_binding_layout()
        entry = self._state_param_owner.get(id(parameter))
        if entry is None or entry[0] is not parameter:
            raise KeyError("parameter has no finalized GefenMuonHybrid shard identity")
        return entry[1].shard_identity(parameter)

    def shard_bindings(self):
        """Return all local tensor/identity pairs in canonical order."""

        self._assert_finalized_binding_layout()
        return self._hybrid_local_shard_bindings

    def parameter_routing(self):
        """Return the immutable canonical FQN-to-child-role routing."""

        self._assert_finalized_binding_layout()
        if not self._canonical_identity_ready():
            raise RuntimeError("GefenMuonHybrid parameter routing is not finalized")
        return self._hybrid_fqn_roles

    def sharding_manifest(self):
        """Return the complete composite manifest, or ``None``."""

        self._assert_finalized_binding_layout()
        return self._hybrid_sharding_manifest

    def codebook_process_group_binding(self):
        """Return the one binding shared by every present Gefen child."""

        self._assert_finalized_binding_layout()
        return self._hybrid_codebook_process_group

    @property
    def _gefen_codebook_process_group(self):
        # Read-only alias: the scoped-step protocol borrowed from Gefen
        # (_prepare_scoped_amp_optimizer_step and the failure synchronization
        # it drives) looks the binding up under the child attribute name. The
        # composite's one shared binding lives in
        # _hybrid_codebook_process_group; nothing may store under this name.
        return self._hybrid_codebook_process_group

    def _synchronize_codebook_scope_failure(self, error, phase: str) -> None:
        # Composite-level preflight/AMP failures synchronize on the one
        # binding shared by every child, so every scope member raises
        # together instead of stranding peers inside a child's scoped step
        # collectives. The finalized layout guarantees the first child holds
        # the identical binding; without a binding, the local error is raised
        # unchanged, exactly as the children do.
        binding = self._hybrid_codebook_process_group
        if binding is None:
            if error is not None:
                raise error
            return
        self._subopts[0]._synchronize_codebook_scope_failure(error, phase)

    def _prepare_scoped_amp_optimizer_step(self) -> bool:
        # GradScaler attaches found_inf/grad_scale to the composite, never to
        # the children, so the children's own scoped AMP gates cannot see an
        # overflow. Run the base scoped protocol once with the composite as
        # the optimizer: under a multi-member codebook binding it validates
        # found_inf/grad_scale agreement collectively and makes an overflow
        # skip a group-wide decision, and on a finite step it unscales the
        # union of both children's gradients exactly once. Without a binding
        # this is exactly _amp_prepare_optimizer_step(self).
        return Gefen._prepare_scoped_amp_optimizer_step(self)

    def zero_grad(self, set_to_none: bool = True):
        self._assert_finalized_binding_layout()
        for o in self._subopts:
            o.zero_grad(set_to_none=set_to_none)

    def _assert_capturable_devices_if_capturing(self) -> None:
        capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        if not capturing:
            return
        if not self.capturable:
            raise RuntimeError(
                "Attempting CUDA graph capture of GefenMuonHybrid.step() but "
                "capturable=False. Construct the optimizer with capturable=True "
                "to make step() graph-safe."
            )
        devices = {
            param.device for optimizer in self._subopts for group in optimizer.param_groups for param in group["params"]
        }
        capture_device = torch.device("cuda", torch.cuda.current_device())
        if devices != {capture_device}:
            raise RuntimeError(
                "CUDA graph capture requires every GefenMuonHybrid parameter on "
                "the current capture device {}; found {}".format(capture_device, sorted(map(str, devices)))
            )

    def step(self, closure=None):
        self._assert_finalized_binding_layout()
        self._assert_capturable_devices_if_capturing()
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
        # Re-read the closure from the (possibly hook-rewritten) call args, as
        # torch's wrapper would by calling step(*args, **kwargs).
        closure = args[1] if len(args) > 1 else kwargs.get("closure")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._assert_finalized_binding_layout()
        # Composite structural preflight, atomically over BOTH children before
        # either child steps. Under a shared codebook scope the failure is
        # synchronized on the binding first (mirroring Gefen.step and
        # GefenMuon.step), so a rank-local structural error raises on every
        # scope member together instead of stranding peers inside a child's
        # scoped step collectives.
        try:
            for child in self._subopts:
                _assert_optimizer_gradients_structurally_valid(child, require_2d_params=child is self.muon)
            local_preflight_error = None
        except Exception as exc:
            local_preflight_error = exc
        if self._hybrid_codebook_process_group is not None:
            self._synchronize_codebook_scope_failure(
                local_preflight_error, "gradient preflight"
            )
        elif local_preflight_error is not None:
            raise local_preflight_error
        # A non-finite gradient in either half skips BOTH children before their
        # codebooks, states, counters, or parameters can move. GradScaler
        # attaches found_inf/grad_scale to the composite, so under a shared
        # multi-member codebook scope the overflow skip is the children's
        # scoped AMP protocol run here -- collective found_inf/grad_scale
        # agreement, then a group-wide skip -- before any child enters its
        # scoped step collectives. Explicit scaler.unscale_(hybrid) is
        # detected by grad_scale=None and is not repeated; automatic unscale
        # covers every child parameter exactly once.
        if (hasattr(self, "found_inf") or hasattr(self, "grad_scale")) and not self._prepare_scoped_amp_optimizer_step():
            for post_hook in self._optimizer_step_post_hooks.values():
                post_hook(self, args, kwargs)
            return loss
        with torch.no_grad():
            for o in self._subopts:
                o.step()

        for post_hook in self._optimizer_step_post_hooks.values():
            post_hook(self, args, kwargs)
        return loss

    def state_dict(self):
        self._assert_finalized_binding_layout()
        # Instance state-dict pre/post hooks, mirroring Optimizer.state_dict:
        # pre-hooks take (optimizer) and return nothing; a post-hook may return
        # a replacement state_dict.
        for pre_hook in self._optimizer_state_dict_pre_hooks.values():
            pre_hook(self)
        self._assert_finalized_binding_layout()
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

    def optimizer_contract(self) -> OptimizerContract:
        """Return the composite contract without flattening either child schema."""

        try:
            GefenMuonHybrid._reject_rebinding_method_shadows(self)
            rebinding_ready = bool(GefenMuonHybrid._gefen_rebinding_children(self))
            identity_ready = GefenMuonHybrid._canonical_identity_ready(self)
        except Exception:
            rebinding_ready = False
            identity_ready = False
        try:
            from gefen.portable_hybrid import _hybrid_portable_contract_support

            (
                canonical_global_same_topology,
                canonical_global_topology_changing,
                canonical_global_topology_change_kinds,
            ) = _hybrid_portable_contract_support(self)
        except Exception:
            canonical_global_same_topology = frozenset()
            canonical_global_topology_changing = frozenset()
            canonical_global_topology_change_kinds = frozenset()
        muon_contract = self.muon.optimizer_contract() if self.muon is not None else None
        backup_contract = (
            self.backup.optimizer_contract()
            if self.backup is not None and hasattr(self.backup, "optimizer_contract")
            else None
        )
        backup_implementation = ""
        if self.backup is not None:
            backup_implementation = (
                backup_contract.implementation
                if backup_contract is not None
                else type(self.backup).__module__ + "." + type(self.backup).__qualname__
            )
        return _hybrid_contract(
            muon=muon_contract,
            backup=backup_contract,
            backup_implementation=backup_implementation,
            canonical_parameter_fqns=identity_ready,
            stable_shard_identity=identity_ready,
            explicit_process_group_codebook_scope=rebinding_ready,
            shard_rebinding=rebinding_ready,
            post_sharding=rebinding_ready,
            canonical_global_same_topology=canonical_global_same_topology,
            canonical_global_topology_changing=canonical_global_topology_changing,
            canonical_global_topology_change_kinds=canonical_global_topology_change_kinds,
        )

    def export_portable_state(
        self,
        *,
        checkpoint_process_group,
        transaction_id,
        limits,
    ):
        """Collectively export one complete Gefen-backed composite document."""

        from gefen.portable_hybrid import _export_hybrid_portable_state

        return _export_hybrid_portable_state(
            self,
            checkpoint_process_group=checkpoint_process_group,
            transaction_id=transaction_id,
            limits=limits,
        )

    def import_portable_state(
        self,
        state,
        *,
        checkpoint_process_group,
        transaction_id,
        limits,
    ) -> None:
        """Collectively stage and atomically publish composite portable state."""

        from gefen.portable_hybrid import _import_hybrid_portable_state

        _import_hybrid_portable_state(
            self,
            state,
            checkpoint_process_group=checkpoint_process_group,
            transaction_id=transaction_id,
            limits=limits,
        )

    def load_state_dict(self, state_dict):
        self._assert_finalized_binding_layout()
        # Instance load pre-hooks first (a pre-hook may return a replacement
        # dict -- e.g. one that converts a foreign schema), mirroring
        # Optimizer.load_state_dict's shallow copy + hook pass.
        state_dict = state_dict.copy()
        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        self._assert_finalized_binding_layout()

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
            checkpoint_backup_optimizer = state_dict.get("backup_optimizer", "gefen")
            if checkpoint_backup_optimizer != self.backup_optimizer:
                raise ValueError(
                    "GefenMuonHybrid.load_state_dict: checkpoint backup_optimizer "
                    "is {!r}, but this optimizer uses {!r}; construct the hybrid "
                    "with the checkpoint's backup_optimizer instead.".format(
                        checkpoint_backup_optimizer, self.backup_optimizer
                    )
                )
        # Load the two children atomically: torch's optimizer loader can still
        # reject a child on its internal layout (e.g. a different param count),
        # and a backup failure after the muon child loaded would leave the
        # hybrid half-loaded (new muon, old backup). Snapshot the muon child's
        # state first and roll it back if the backup load raises.
        muon_snapshot = None
        if self.muon is not None:
            if self.backup is not None:
                muon_snapshot = copy.deepcopy(self.muon.state_dict())
            self.muon.load_state_dict(state_dict["muon"])
        if self.backup is not None:
            try:
                self.backup.load_state_dict(state_dict["backup"])
            except Exception:
                if muon_snapshot is not None:
                    self.muon.load_state_dict(muon_snapshot)
                raise

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @staticmethod
    def _auto_split(params_or_model, backup_substrings):
        """Split a model / named-param iterable into (muon_named, backup_named).

        A model routes by module type (embeddings, incl. tied heads) plus name
        substrings and shape; a named-param iterable routes by name+shape only
        (no module context -- an embedding whose name matches no substring, e.g.
        T5 ``shared``, would slip to Muon, so a model is preferred). Bare tensors
        raise, since they carry neither names nor module type to route on.
        """
        subs = DEFAULT_BACKUP_SUBSTRINGS if backup_substrings is None else tuple(backup_substrings)
        if isinstance(params_or_model, nn.Module):
            muon_named, backup_named = split_params_for_muon(params_or_model, backup_substrings=subs)
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
                    "GefenMuonHybrid.from_model(model, ...).".format(type(item).__name__)
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
        raise NotImplementedError("GefenMuonHybrid splits params at construction; add_param_group is unsupported")

    def __repr__(self):
        nm = sum(len(group["params"]) for group in self.muon.param_groups) if self.muon is not None else 0
        nb = sum(len(group["params"]) for group in self.backup.param_groups) if self.backup is not None else 0
        return f"GefenMuonHybrid(muon_params={nm}, backup_params={nb}, backup_optimizer={self.backup_optimizer!r})"
