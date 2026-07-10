"""Exact optimizer recipes used by the training matrix.

The recipe registry is deliberately data, not a collection of ad-hoc CLI
branches.  A result row therefore carries the complete resolved recipe and can
distinguish current constructor defaults from the recommended configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch

from gefen import GefenMuon, GefenMuonHybrid, split_params_for_muon


BATCHED_NS_DEFAULT_WORKSPACE_BYTES = 256 << 20


@dataclass(frozen=True)
class CellRecipe:
    name: str
    label: str
    recipe_class: str
    primary: str
    auxiliary: str | None
    backup_lr_multiplier: float | None
    ns_schedule: str | None
    ns_steps: int | None
    adjust_lr_fn: str | None
    normuon: bool | None
    backup_1d_period_one: bool
    backup_2d_period_one: bool
    description: str


CELL_RECIPES = {
    recipe.name: recipe
    for recipe in (
        CellRecipe(
            name="adamw",
            label="torch.optim.AdamW with native parameter-dtype states",
            recipe_class="control",
            primary="torch.optim.AdamW",
            auxiliary=None,
            backup_lr_multiplier=None,
            ns_schedule=None,
            ns_steps=None,
            adjust_lr_fn=None,
            normuon=None,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Real torch.optim.AdamW over every trainable parameter; moments use the parameter dtype.",
        ),
        CellRecipe(
            name="torch_muon_adamw",
            label="stock torch.optim.Muon + native-dtype-state AdamW auxiliary",
            recipe_class="control",
            primary="torch.optim.Muon",
            auxiliary="torch.optim.AdamW",
            backup_lr_multiplier=1.0,
            ns_schedule="standard",
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
            normuon=False,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Stock PyTorch Muon on hidden matrices at AdamW-scale LR; AdamW on embeddings, head, norms, and biases.",
        ),
        CellRecipe(
            name="gefen_muon_classic_adamw",
            label="GefenMuon classic NS5/NorMuon-off + native-dtype-state AdamW auxiliary",
            recipe_class="classic",
            primary="gefen.GefenMuon",
            auxiliary="torch.optim.AdamW",
            backup_lr_multiplier=1.0,
            ns_schedule="standard",
            ns_steps=5,
            adjust_lr_fn="match_rms_adamw",
            normuon=False,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Quantized Muon with classic five-step Newton-Schulz, no NorMuon normalization, and AdamW-scale LR.",
        ),
        CellRecipe(
            name="gefen_muon_tuned3_adamw",
            label="GefenMuon tuned3/NorMuon-off + native-dtype-state AdamW auxiliary",
            recipe_class="isolation_ns_schedule",
            primary="gefen.GefenMuon",
            auxiliary="torch.optim.AdamW",
            backup_lr_multiplier=1.0,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=False,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Changes only the hidden Muon NS schedule relative to the classic GefenMuon+AdamW cell.",
        ),
        CellRecipe(
            name="gefen_muon_normuon_adamw",
            label="GefenMuon tuned3/NorMuon-on + native-dtype-state AdamW auxiliary",
            recipe_class="isolation_normuon",
            primary="gefen.GefenMuon",
            auxiliary="torch.optim.AdamW",
            backup_lr_multiplier=1.0,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Changes only NorMuon relative to the tuned3/NorMuon-off GefenMuon+AdamW cell.",
        ),
        CellRecipe(
            name="gefen_muon_recommended_adamw",
            label="GefenMuon recommended hidden + native-dtype-state AdamW auxiliary",
            recipe_class="isolation_backup_lr_real_adamw",
            primary="gefen.GefenMuon",
            auxiliary="torch.optim.AdamW",
            backup_lr_multiplier=0.5,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Changes only real-AdamW auxiliary LR after tuned3 and NorMuon, forming the recommended-hidden auxiliary control.",
        ),
        CellRecipe(
            name="gefen_hybrid_period1_all",
            label="Gefen hybrid, 1D+2D backup period-one",
            recipe_class="ablation",
            primary="gefen.GefenMuonHybrid",
            auxiliary="gefen.Gefen",
            backup_lr_multiplier=1.0,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=True,
            backup_2d_period_one=True,
            description="Literal hybrid LR split with per-element second moment on every backup tensor.",
        ),
        CellRecipe(
            name="gefen_hybrid_split_lr_only",
            label="Gefen hybrid split-LR only",
            recipe_class="isolation_backup_lr",
            primary="gefen.GefenMuonHybrid",
            auxiliary="gefen.Gefen",
            backup_lr_multiplier=0.5,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Changes only backup LR relative to literal hybrid defaults when weight decay is zero.",
        ),
        CellRecipe(
            name="gefen_hybrid_period1_only",
            label="Gefen hybrid 1D period-one only",
            recipe_class="isolation_backup_1d_period_one",
            primary="gefen.GefenMuonHybrid",
            auxiliary="gefen.Gefen",
            backup_lr_multiplier=1.0,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=True,
            backup_2d_period_one=False,
            description="Changes only 1D backup period-one relative to literal hybrid defaults.",
        ),
        CellRecipe(
            name="gefen_hybrid_recommended",
            label="Gefen hybrid recommended",
            recipe_class="recommended",
            primary="gefen.GefenMuonHybrid",
            auxiliary="gefen.Gefen",
            backup_lr_multiplier=0.5,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=True,
            backup_2d_period_one=False,
            description="Recommended split backup LR and 1D period-one recipe.",
        ),
        CellRecipe(
            name="gefen_hybrid_literal",
            label="Gefen hybrid literal constructor defaults",
            recipe_class="literal_constructor_defaults",
            primary="gefen.GefenMuonHybrid",
            auxiliary="gefen.Gefen",
            backup_lr_multiplier=1.0,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=False,
            backup_2d_period_one=False,
            description="Current literal constructor behavior: shared LR, tuned3, NorMuon on, no period-one backup overrides.",
        ),
        CellRecipe(
            name="gefen_hybrid_recommended_2d",
            label="Gefen hybrid recommended + 2D period-one",
            recipe_class="recommended_plus_2d_period_one",
            primary="gefen.GefenMuonHybrid",
            auxiliary="gefen.Gefen",
            backup_lr_multiplier=0.5,
            ns_schedule="tuned3",
            ns_steps=3,
            adjust_lr_fn="match_rms_adamw",
            normuon=True,
            backup_1d_period_one=True,
            backup_2d_period_one=True,
            description="Recommended recipe plus AdamW-like per-element second moment on 2D backup weights.",
        ),
    )
}

# Keep the user's requested seven-cell comparison stable. Diagnostic controls
# are opt-in so adding methodology checks never silently expands a costly run.
CORE_CELLS = (
    "adamw",
    "torch_muon_adamw",
    "gefen_muon_classic_adamw",
    "gefen_hybrid_period1_all",
    "gefen_hybrid_recommended",
    "gefen_hybrid_literal",
    "gefen_hybrid_recommended_2d",
)
ISOLATION_CELLS = (
    "gefen_muon_tuned3_adamw",
    "gefen_muon_normuon_adamw",
    "gefen_muon_recommended_adamw",
    "gefen_hybrid_split_lr_only",
    "gefen_hybrid_period1_only",
)
ALL_CELLS = tuple(CELL_RECIPES)


@dataclass(frozen=True)
class CellBuildConfig:
    """Shared, explicit optimizer knobs.

    Weight decay defaults to zero for every cell so the out-of-box matrix is
    matched.  Per-half overrides stay available for controlled ablations.
    ``backup_lr=None`` selects each recipe's documented multiplier.
    """

    lr: float
    muon_lr: float | None = None
    backup_lr: float | None = None
    weight_decay: float = 0.0
    muon_weight_decay: float | None = None
    backup_weight_decay: float | None = None
    betas: tuple[float, float] = (0.9, 0.999)
    muon_eps: float = 1e-7
    backup_eps: float = 1e-8
    momentum: float = 0.95
    nesterov: bool = True
    fused: bool = False
    batched_ns: bool = False
    batched_ns_workspace_bytes: int = BATCHED_NS_DEFAULT_WORKSPACE_BYTES
    adjust_lr_fn: str | None = None


class OptimizerPair:
    """One logical optimizer backed by a primary optimizer and real AdamW.

    This intentionally exposes the child group dictionaries by reference, so
    multiplicative scheduling preserves the Muon/auxiliary LR ratio.  Its
    namespaced state dict avoids pretending the two optimizers have one flat
    torch schema.
    """

    def __init__(self, primary: torch.optim.Optimizer, auxiliary: torch.optim.AdamW):
        self.primary = primary
        self.auxiliary = auxiliary

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return list(self.primary.param_groups) + list(self.auxiliary.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.primary.zero_grad(set_to_none=set_to_none)
        self.auxiliary.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.primary.step()
        self.auxiliary.step()
        return loss

    def state_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.state_dict(),
            "auxiliary": self.auxiliary.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) != {"primary", "auxiliary"}:
            raise ValueError("OptimizerPair checkpoint must contain primary and auxiliary states")
        self.primary.load_state_dict(state_dict["primary"])
        self.auxiliary.load_state_dict(state_dict["auxiliary"])


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _check_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


def resolve_cell(name: str, config: CellBuildConfig) -> dict[str, Any]:
    """Return the fully resolved, JSON-serializable recipe for one cell."""

    try:
        recipe = CELL_RECIPES[name]
    except KeyError as exc:
        raise ValueError(f"unknown cell {name!r}; choose from {sorted(CELL_RECIPES)}") from exc

    _check_positive("lr", config.lr)
    _check_positive("muon_eps", config.muon_eps)
    _check_positive("backup_eps", config.backup_eps)
    if not isinstance(config.batched_ns, bool):
        raise TypeError("batched_ns must be a boolean")
    if not isinstance(config.batched_ns_workspace_bytes, int) or isinstance(
        config.batched_ns_workspace_bytes, bool
    ):
        raise TypeError("batched_ns_workspace_bytes must be an integer")
    _check_positive(
        "batched_ns_workspace_bytes", config.batched_ns_workspace_bytes
    )
    primary_lr = (
        config.lr
        if recipe.ns_steps is None or config.muon_lr is None
        else config.muon_lr
    )
    _check_positive("muon_lr", primary_lr)

    if recipe.backup_lr_multiplier is None:
        backup_lr = None
    elif config.backup_lr is not None:
        backup_lr = config.backup_lr
    else:
        backup_lr = config.lr * recipe.backup_lr_multiplier
    if backup_lr is not None:
        _check_positive("backup_lr", backup_lr)
    if recipe.primary == "gefen.GefenMuonHybrid" and config.muon_eps != 1e-7:
        raise ValueError(
            "GefenMuonHybrid does not expose the Muon-half epsilon; "
            "--muon-eps must remain its actual 1e-7 default for hybrid cells"
        )

    muon_wd = config.weight_decay if config.muon_weight_decay is None else config.muon_weight_decay
    backup_wd = config.weight_decay if config.backup_weight_decay is None else config.backup_weight_decay
    for key, value in (
        ("weight_decay", config.weight_decay),
        ("muon_weight_decay", muon_wd),
        ("backup_weight_decay", backup_wd),
    ):
        _check_nonnegative(key, value)

    batched_ns_supported = recipe.primary in {
        "gefen.GefenMuon",
        "gefen.GefenMuonHybrid",
    }
    resolved = asdict(recipe)
    if recipe.ns_steps is not None and config.adjust_lr_fn is not None:
        if config.adjust_lr_fn not in {"original", "match_rms_adamw"}:
            raise ValueError(
                "adjust_lr_fn must be 'original' or 'match_rms_adamw', "
                f"got {config.adjust_lr_fn!r}"
            )
        resolved["adjust_lr_fn"] = config.adjust_lr_fn
    resolved.update(
        {
            "lr": config.lr,
            "primary_lr": primary_lr,
            "backup_lr": backup_lr,
            "weight_decay": config.weight_decay,
            "muon_weight_decay": muon_wd,
            "backup_weight_decay": backup_wd,
            "betas": list(config.betas),
            "muon_eps": config.muon_eps,
            "backup_eps": config.backup_eps,
            "momentum": config.momentum,
            "nesterov": config.nesterov,
            "fused_requested": config.fused,
            "primary_fused": (
                None if recipe.primary == "torch.optim.Muon" else config.fused
            ),
            "auxiliary_fused": config.fused if recipe.auxiliary is not None else None,
            "batched_ns_requested": config.batched_ns,
            "batched_ns_supported": batched_ns_supported,
            "batched_ns": config.batched_ns if batched_ns_supported else False,
            "batched_ns_workspace_bytes_requested": (
                config.batched_ns_workspace_bytes
            ),
            "batched_ns_workspace_bytes": (
                config.batched_ns_workspace_bytes if batched_ns_supported else None
            ),
        }
    )
    overrides = []
    if config.muon_lr is not None and recipe.ns_steps is not None:
        overrides.append("muon_lr")
    if config.backup_lr is not None and recipe.backup_lr_multiplier is not None:
        overrides.append("backup_lr")
    if config.weight_decay != 0.0:
        overrides.append("weight_decay")
    if config.muon_weight_decay is not None:
        overrides.append("muon_weight_decay")
    if config.backup_weight_decay is not None:
        overrides.append("backup_weight_decay")
    if config.adjust_lr_fn is not None and recipe.ns_steps is not None:
        overrides.append("muon_adjust")
    if config.betas != (0.9, 0.999):
        overrides.append("betas")
    if config.muon_eps != 1e-7:
        overrides.append("muon_eps")
    if config.backup_eps != 1e-8:
        overrides.append("backup_eps")
    if config.momentum != 0.95:
        overrides.append("momentum")
    if config.nesterov is not True:
        overrides.append("nesterov")
    if config.batched_ns and batched_ns_supported:
        overrides.append("batched_ns")
    if (
        config.batched_ns_workspace_bytes != BATCHED_NS_DEFAULT_WORKSPACE_BYTES
        and batched_ns_supported
    ):
        overrides.append("batched_ns_workspace_bytes")
    resolved["unsupported_requests"] = (
        ["batched_ns"] if config.batched_ns and not batched_ns_supported else []
    )
    resolved["recipe_overrides"] = overrides
    resolved["effective_label"] = (
        recipe.label
        if not overrides
        else f"{recipe.label} (overridden: {','.join(overrides)})"
    )
    return resolved


def _params(named: Iterable[tuple[str, torch.nn.Parameter]]) -> list[torch.nn.Parameter]:
    return [param for _, param in named]


def build_optimizer(model: torch.nn.Module, name: str, config: CellBuildConfig):
    """Construct one exact cell and return ``(optimizer, resolved_recipe)``."""

    resolved = resolve_cell(name, config)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    if name == "adamw":
        optimizer = torch.optim.AdamW(
            _params(named),
            lr=resolved["lr"],
            betas=config.betas,
            eps=config.backup_eps,
            weight_decay=resolved["weight_decay"],
            fused=config.fused,
        )
        return optimizer, resolved

    muon_named, backup_named = split_params_for_muon(model)
    if not muon_named or not backup_named:
        raise ValueError(
            f"cell {name} requires both hidden 2D and auxiliary parameters; "
            f"split produced {len(muon_named)} Muon / {len(backup_named)} auxiliary"
        )

    if name == "torch_muon_adamw":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("torch.optim.Muon is unavailable; use PyTorch 2.9 or newer")
        primary = torch.optim.Muon(
            _params(muon_named),
            lr=resolved["primary_lr"],
            weight_decay=resolved["muon_weight_decay"],
            momentum=config.momentum,
            nesterov=config.nesterov,
            eps=config.muon_eps,
            ns_steps=resolved["ns_steps"],
            adjust_lr_fn=resolved["adjust_lr_fn"],
        )
        auxiliary = torch.optim.AdamW(
            _params(backup_named),
            lr=resolved["backup_lr"],
            betas=config.betas,
            eps=config.backup_eps,
            weight_decay=resolved["backup_weight_decay"],
            fused=config.fused,
        )
        return OptimizerPair(primary, auxiliary), resolved

    if name in {
        "gefen_muon_classic_adamw",
        "gefen_muon_tuned3_adamw",
        "gefen_muon_normuon_adamw",
        "gefen_muon_recommended_adamw",
    }:
        primary = GefenMuon(
            muon_named,
            lr=resolved["primary_lr"],
            weight_decay=resolved["muon_weight_decay"],
            momentum=config.momentum,
            nesterov=config.nesterov,
            eps=config.muon_eps,
            ns_steps=resolved["ns_steps"],
            ns_schedule=resolved["ns_schedule"],
            adjust_lr_fn=resolved["adjust_lr_fn"],
            normuon=resolved["normuon"],
            fused=config.fused,
            batched_ns=resolved["batched_ns"],
            batched_ns_workspace_bytes=resolved["batched_ns_workspace_bytes"],
        )
        auxiliary = torch.optim.AdamW(
            _params(backup_named),
            lr=resolved["backup_lr"],
            betas=config.betas,
            eps=config.backup_eps,
            weight_decay=resolved["backup_weight_decay"],
            fused=config.fused,
        )
        return OptimizerPair(primary, auxiliary), resolved

    optimizer = GefenMuonHybrid(
        muon_named,
        backup_named,
        lr=resolved["lr"],
        muon_lr=resolved["primary_lr"],
        backup_lr=resolved["backup_lr"],
        weight_decay=resolved["weight_decay"],
        muon_weight_decay=resolved["muon_weight_decay"],
        backup_weight_decay=resolved["backup_weight_decay"],
        betas=config.betas,
        eps=config.backup_eps,
        fused=config.fused,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_steps=resolved["ns_steps"],
        ns_schedule=resolved["ns_schedule"],
        adjust_lr_fn=resolved["adjust_lr_fn"],
        normuon=resolved["normuon"],
        batched_ns=resolved["batched_ns"],
        batched_ns_workspace_bytes=resolved["batched_ns_workspace_bytes"],
        backup_1d_period_one=resolved["backup_1d_period_one"],
        backup_2d_period_one=resolved["backup_2d_period_one"],
    )
    return optimizer, resolved
