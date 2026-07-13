"""Platform-agnostic optimizer capability and state-layout contracts.

The descriptors in this module are read-only declarations. They describe the
optimizer state Gefen owns and the integration surfaces implemented today
without importing DCP, DTensor, Megatron, DeepSpeed, or another adapter.
"""

from dataclasses import dataclass
from enum import Enum
from typing import (
    AbstractSet,
    FrozenSet,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)


CONTRACT_SCHEMA_VERSION = 1


class StateScope(str, Enum):
    """Ownership category for one optimizer-state field."""

    OPTIMIZER_COMMON = "optimizer_common"
    PARAMETER = "parameter"
    DERIVED = "derived"


class StateGeometry(str, Enum):
    """Logical geometry kind, independent of local/global extent."""

    SCALAR = "scalar"
    CODEBOOK = "codebook"
    PARAMETER = "parameter"
    BLOCK = "block"
    ROW = "row"
    COLUMN = "column"
    OPAQUE = "opaque"


class StateKeyMatch(str, Enum):
    """How a field declaration matches live or serialized dictionary keys."""

    EXACT = "exact"
    PREFIX = "prefix"


class StateExtent(str, Enum):
    """Logical extent represented by a per-parameter state variant."""

    METADATA_ONLY = "metadata_only"
    LOCAL_STORAGE = "local_storage"
    GLOBAL_PARAMETER = "global_parameter"
    OWNER_PARAMETER = "owner_parameter"


class ParameterStateRole(str, Enum):
    """Ownership role on which a per-parameter variant is present."""

    ANY = "any"
    OWNER = "owner"
    NON_OWNER = "non_owner"


class ParameterLayout(str, Enum):
    """Concrete parameter layouts exposed by the current implementation."""

    REPLICATED = "replicated"
    FLATTENED_ELEMENT_SHARD = "flattened_element_shard"
    WHOLE_PARAMETER_OWNER = "whole_parameter_owner"
    DTENSOR_1D_DEFAULT_WORLD = "dtensor_1d_default_world"


class ProcessGroupScope(str, Enum):
    """How an implemented path obtains its collective process group."""

    NONE = "none"
    DEFAULT_WORLD = "default_world"
    INFERRED_DEVICE_MESH = "inferred_device_mesh"
    ADAPTER_DEFINED = "adapter_defined"


class CheckpointTransport(str, Enum):
    """Checkpoint transport with independently scoped topology support."""

    NATIVE_OPTIMIZER = "native_optimizer"
    PYTORCH_RANK_LOCAL = "pytorch_rank_local"
    COMPOSITE_NATIVE = "composite_native"


class TopologyChange(str, Enum):
    """Specific topology mutation supported by a checkpoint transport."""

    WORLD_SIZE_OWNER_REDISTRIBUTION = "world_size_owner_redistribution"
    PLACEMENT_RESHARD = "placement_reshard"


class Precision(str, Enum):
    """Parameter/gradient storage precision accepted by validated core paths."""

    FLOAT32 = "float32"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"
    FLOAT64 = "float64"


def _tuple(value):
    return tuple(value)


def _frozenset(value):
    return frozenset(value)


def _validate_dimensions(name, values, *, positive):
    if len(set(values)) != len(values):
        raise ValueError("{} must be unique".format(name))
    minimum = 1 if positive else 0
    if any(type(value) is not int or value < minimum for value in values):
        relation = "positive" if positive else "nonnegative"
        raise ValueError("{} must contain {} integers".format(name, relation))


@dataclass(frozen=True)
class StateField:
    """One named authoritative field, runtime cache, or transport field."""

    name: str
    scope: StateScope
    geometry: StateGeometry
    checkpointed: bool
    key_match: StateKeyMatch = StateKeyMatch.EXACT
    applicable_sharded_modes: AbstractSet[str] = frozenset()
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applicable_sharded_modes",
            _frozenset(self.applicable_sharded_modes),
        )
        if not self.name:
            raise ValueError("StateField.name must be non-empty")
        if not isinstance(self.scope, StateScope):
            raise TypeError("StateField.scope must be a StateScope")
        if not isinstance(self.geometry, StateGeometry):
            raise TypeError("StateField.geometry must be a StateGeometry")
        if not isinstance(self.key_match, StateKeyMatch):
            raise TypeError("StateField.key_match must be a StateKeyMatch")

    @property
    def authoritative(self) -> bool:
        """Whether the field carries optimizer meaning rather than a cache."""

        return self.scope is not StateScope.DERIVED

    def matches(self, key: str) -> bool:
        """Return whether a live or serialized key matches this declaration."""

        if self.key_match is StateKeyMatch.PREFIX:
            return isinstance(key, str) and key.startswith(self.name)
        return key == self.name


@dataclass(frozen=True)
class StateVariant:
    """A machine-selectable valid per-parameter state combination."""

    name: str
    fields: Sequence[str]
    layouts: AbstractSet[ParameterLayout]
    extent: StateExtent
    role: ParameterStateRole = ParameterStateRole.ANY
    initialized: bool = True
    parameter_ranks: Optional[Sequence[int]] = None
    excluded_parameter_ranks: Sequence[int] = ()
    sharded_mode: Optional[str] = None
    inactive_fields: Sequence[str] = ()
    migration_only: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _tuple(self.fields))
        object.__setattr__(self, "layouts", _frozenset(self.layouts))
        object.__setattr__(self, "inactive_fields", _tuple(self.inactive_fields))
        object.__setattr__(
            self, "excluded_parameter_ranks", _tuple(self.excluded_parameter_ranks)
        )
        if self.parameter_ranks is not None:
            object.__setattr__(self, "parameter_ranks", _tuple(self.parameter_ranks))
        if not self.name:
            raise ValueError("StateVariant.name must be non-empty")
        if not self.fields:
            raise ValueError("StateVariant.fields must be non-empty")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("StateVariant.fields must not contain duplicates")
        if not self.layouts:
            raise ValueError("StateVariant.layouts must be non-empty")
        if not isinstance(self.extent, StateExtent):
            raise TypeError("StateVariant.extent must be a StateExtent")
        if not isinstance(self.role, ParameterStateRole):
            raise TypeError("StateVariant.role must be a ParameterStateRole")
        if not set(self.inactive_fields).issubset(self.fields):
            raise ValueError("StateVariant.inactive_fields must be present in fields")
        if self.parameter_ranks is not None and set(
            self.parameter_ranks
        ) & set(self.excluded_parameter_ranks):
            raise ValueError("included and excluded parameter ranks must be disjoint")
        if self.parameter_ranks is not None:
            _validate_dimensions("parameter_ranks", self.parameter_ranks, positive=False)
        _validate_dimensions(
            "excluded_parameter_ranks",
            self.excluded_parameter_ranks,
            positive=False,
        )


@dataclass(frozen=True)
class OptimizerStateLayout:
    """Declared fields, per-parameter variants, and composite namespaces."""

    fields: Sequence[StateField]
    parameter_variants: Sequence[StateVariant]
    composite_namespaces: Sequence[str] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _tuple(self.fields))
        object.__setattr__(self, "parameter_variants", _tuple(self.parameter_variants))
        object.__setattr__(self, "composite_namespaces", _tuple(self.composite_namespaces))
        names = tuple(field.name for field in self.fields)
        if len(set(names)) != len(names):
            raise ValueError("OptimizerStateLayout fields must have unique names")
        if len(set(self.composite_namespaces)) != len(self.composite_namespaces):
            raise ValueError("composite namespaces must be unique")
        variant_names = tuple(variant.name for variant in self.parameter_variants)
        if len(set(variant_names)) != len(variant_names):
            raise ValueError("StateVariant names must be unique")
        parameter_fields = {
            field.name for field in self.fields if field.scope is StateScope.PARAMETER
        }
        for variant in self.parameter_variants:
            unknown = set(variant.fields) - parameter_fields
            if unknown:
                raise ValueError(
                    "StateVariant {!r} references undeclared parameter fields: {}".format(
                        variant.name, sorted(unknown)
                    )
                )

    def fields_for_scope(self, scope: StateScope) -> Tuple[StateField, ...]:
        """Return fields in declaration order for one ownership scope."""

        return tuple(field for field in self.fields if field.scope is scope)

    def field(self, name: str) -> StateField:
        """Return one declared field by name."""

        for field in self.fields:
            if field.matches(name):
                return field
        raise KeyError(name)


@dataclass(frozen=True)
class TrainingSupport:
    """One qualified training layout capability."""

    layout: ParameterLayout
    process_group_scope: ProcessGroupScope
    mesh_dimensions: Optional[Sequence[int]] = None
    sharded_mode: Optional[str] = None
    requires_complete_parameter_storage: bool = False
    requires_complete_logical_matrix: bool = False

    def __post_init__(self) -> None:
        if self.mesh_dimensions is not None:
            object.__setattr__(self, "mesh_dimensions", _tuple(self.mesh_dimensions))
            _validate_dimensions(
                "mesh_dimensions", self.mesh_dimensions, positive=True
            )
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("TrainingSupport.layout must be a ParameterLayout")
        if not isinstance(self.process_group_scope, ProcessGroupScope):
            raise TypeError(
                "TrainingSupport.process_group_scope must be a ProcessGroupScope"
            )


@dataclass(frozen=True)
class CheckpointSupport:
    """One transport's separately qualified checkpoint capability."""

    transport: CheckpointTransport
    same_topology: AbstractSet[ParameterLayout]
    topology_changing: AbstractSet[ParameterLayout]
    process_group_scope: ProcessGroupScope
    topology_change_kinds: AbstractSet[TopologyChange] = frozenset()
    mesh_dimensions: Optional[Sequence[int]] = None
    required_sharded_modes: AbstractSet[str] = frozenset()
    requires_collective: bool = False
    atomic_load: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "same_topology", _frozenset(self.same_topology))
        object.__setattr__(self, "topology_changing", _frozenset(self.topology_changing))
        object.__setattr__(
            self, "topology_change_kinds", _frozenset(self.topology_change_kinds)
        )
        object.__setattr__(
            self,
            "required_sharded_modes",
            _frozenset(self.required_sharded_modes),
        )
        if self.mesh_dimensions is not None:
            object.__setattr__(self, "mesh_dimensions", _tuple(self.mesh_dimensions))
            _validate_dimensions(
                "mesh_dimensions", self.mesh_dimensions, positive=True
            )
        if not isinstance(self.transport, CheckpointTransport):
            raise TypeError("CheckpointSupport.transport must be a CheckpointTransport")
        if not isinstance(self.process_group_scope, ProcessGroupScope):
            raise TypeError(
                "CheckpointSupport.process_group_scope must be a ProcessGroupScope"
            )
        if bool(self.topology_changing) != bool(self.topology_change_kinds):
            raise ValueError(
                "topology-changing layouts and change kinds must be declared together"
            )


@dataclass(frozen=True)
class OptimizerCapabilities:
    """Implemented integration capabilities, including explicit negative claims."""

    training: Sequence[TrainingSupport]
    checkpoints: Sequence[CheckpointSupport]
    precisions: AbstractSet[Precision]
    supported_parameter_ranks: Optional[Sequence[int]]
    accepts_semantic_parameter_names: bool
    canonical_parameter_fqns: bool
    stable_shard_identity: bool
    explicit_process_group_codebook_scope: bool
    shard_rebinding: bool
    post_sharding: bool
    canonical_state_io: bool
    atomic_state_movement: bool
    state_offload: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "training", _tuple(self.training))
        object.__setattr__(self, "checkpoints", _tuple(self.checkpoints))
        object.__setattr__(self, "precisions", _frozenset(self.precisions))
        if self.supported_parameter_ranks is not None:
            object.__setattr__(
                self,
                "supported_parameter_ranks",
                _tuple(self.supported_parameter_ranks),
            )
            _validate_dimensions(
                "supported_parameter_ranks",
                self.supported_parameter_ranks,
                positive=False,
            )


@dataclass(frozen=True)
class OptimizerChildContract:
    """One named child of a composite optimizer contract."""

    role: str
    implementation: str
    contract: Optional["OptimizerContract"]

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("OptimizerChildContract.role must be non-empty")
        if not self.implementation:
            raise ValueError("OptimizerChildContract.implementation must be non-empty")


@dataclass(frozen=True)
class OptimizerContract:
    """Complete read-only state and capability contract for one optimizer."""

    implementation: str
    state_layout: OptimizerStateLayout
    capabilities: OptimizerCapabilities
    children: Sequence[OptimizerChildContract] = ()
    schema_version: int = CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", _tuple(self.children))
        if not self.implementation:
            raise ValueError("OptimizerContract.implementation must be non-empty")
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported optimizer contract schema version: {}".format(
                    self.schema_version
                )
            )
        roles = tuple(child.role for child in self.children)
        if len(set(roles)) != len(roles):
            raise ValueError("OptimizerContract child roles must be unique")


@runtime_checkable
class OptimizerContractProvider(Protocol):
    """Structural protocol implemented by optimizers that expose a contract."""

    def optimizer_contract(self) -> OptimizerContract:
        """Return the optimizer's immutable state and capability declaration."""


_ALL_PRECISIONS = frozenset(
    {Precision.FLOAT32, Precision.BFLOAT16, Precision.FLOAT16, Precision.FLOAT64}
)
_DTENSOR_LAYOUT = ParameterLayout.DTENSOR_1D_DEFAULT_WORLD
_MUON_MODES = frozenset({"exact", "approx", "distributed"})


def _common_fields() -> Tuple[StateField, ...]:
    return (
        StateField(
            "gefen_global_step",
            StateScope.OPTIMIZER_COMMON,
            StateGeometry.SCALAR,
            True,
            description="Canonical optimizer-wide update counter.",
        ),
        StateField(
            "gefen_codebook",
            StateScope.OPTIMIZER_COMMON,
            StateGeometry.CODEBOOK,
            True,
            description="Canonical learned 256-entry momentum codebook.",
        ),
        StateField(
            "gefen_deterministic",
            StateScope.OPTIMIZER_COMMON,
            StateGeometry.SCALAR,
            True,
            description="Checkpoint-bound deterministic execution policy.",
        ),
    )


def _base_parameter_fields() -> Tuple[StateField, ...]:
    return (
        StateField("name", StateScope.PARAMETER, StateGeometry.OPAQUE, True),
        StateField("automatic_period", StateScope.PARAMETER, StateGeometry.SCALAR, True),
        StateField("step", StateScope.PARAMETER, StateGeometry.SCALAR, True),
        StateField("m_codebook", StateScope.PARAMETER, StateGeometry.PARAMETER, True),
        StateField("m_magnitude", StateScope.PARAMETER, StateGeometry.BLOCK, True),
    )


def _derived_fields() -> Tuple[StateField, ...]:
    return (
        StateField("stepsize", StateScope.DERIVED, StateGeometry.BLOCK, False),
        StateField("_h_buf", StateScope.DERIVED, StateGeometry.BLOCK, False),
        StateField("_capt_scalars", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField("_capt_consts", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField("_capt_consts_key", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField("_capt_stack", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField("_capt_row", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField("_param_names", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField("_lr_scalar_cache", StateScope.DERIVED, StateGeometry.SCALAR, False),
        StateField("_fused_build_ok", StateScope.DERIVED, StateGeometry.SCALAR, False),
        StateField("_static_mark_sig", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField(
            "m_codebook_shape",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            True,
            description="Inert legacy checkpoint metadata retained for compatibility.",
        ),
        StateField(
            "_gefen_codebook_by_device",
            StateScope.DERIVED,
            StateGeometry.CODEBOOK,
            False,
        ),
        StateField(
            "_gefen_codebook_lut_by_device",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            False,
        ),
        StateField("_sr_seed_by_device", StateScope.DERIVED, StateGeometry.SCALAR, False),
        StateField(
            "_gefen_global_step_by_device",
            StateScope.DERIVED,
            StateGeometry.SCALAR,
            False,
        ),
        StateField("_capt_stacks", StateScope.DERIVED, StateGeometry.OPAQUE, False),
        StateField(
            "_gefen_rank_local_payload_",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            True,
            key_match=StateKeyMatch.PREFIX,
            description="PyTorch rank-local checkpoint transport.",
        ),
        StateField(
            "_gefen_rank_local_member",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            True,
            description="PyTorch rank-local checkpoint transport marker.",
        ),
        StateField(
            "_gefen_checkpoint_metadata",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            True,
            description="PyTorch transport mirror of optimizer-common state.",
        ),
    )


def _base_training() -> Tuple[TrainingSupport, ...]:
    return (
        TrainingSupport(ParameterLayout.REPLICATED, ProcessGroupScope.NONE),
        TrainingSupport(
            _DTENSOR_LAYOUT,
            ProcessGroupScope.INFERRED_DEVICE_MESH,
            mesh_dimensions=(1,),
        ),
    )


def _negative_capabilities(
    *,
    training: Tuple[TrainingSupport, ...],
    checkpoints: Tuple[CheckpointSupport, ...],
    supported_parameter_ranks: Optional[Tuple[int, ...]],
) -> OptimizerCapabilities:
    return OptimizerCapabilities(
        training=training,
        checkpoints=checkpoints,
        precisions=_ALL_PRECISIONS,
        supported_parameter_ranks=supported_parameter_ranks,
        accepts_semantic_parameter_names=True,
        canonical_parameter_fqns=False,
        stable_shard_identity=False,
        explicit_process_group_codebook_scope=False,
        shard_rebinding=False,
        post_sharding=False,
        canonical_state_io=False,
        atomic_state_movement=False,
        state_offload=False,
    )


def _gefen_contract(*, factored_v_2d: bool) -> OptimizerContract:
    block_fields = (
        StateField("vmean", StateScope.PARAMETER, StateGeometry.BLOCK, True),
        StateField("vmean_step", StateScope.PARAMETER, StateGeometry.SCALAR, True),
    )
    factored_fields = (
        StateField("v_row", StateScope.PARAMETER, StateGeometry.ROW, True),
        StateField("v_col", StateScope.PARAMETER, StateGeometry.COLUMN, True),
        StateField("factored_step", StateScope.PARAMETER, StateGeometry.SCALAR, True),
    )
    layouts = frozenset(
        {
            ParameterLayout.REPLICATED,
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            _DTENSOR_LAYOUT,
        }
    )
    base_names = tuple(field.name for field in _base_parameter_fields())
    variants = [
        StateVariant(
            "name_only",
            ("name",),
            layouts,
            StateExtent.METADATA_ONLY,
            initialized=False,
        ),
    ]
    factored_names = tuple(field.name for field in factored_fields)
    block_names = tuple(field.name for field in block_fields)
    legacy_block_names = ("vmean",)
    if factored_v_2d:
        variants.extend(
            (
                StateVariant(
                    "block_second_moment_non_2d",
                    base_names + block_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.LOCAL_STORAGE,
                    excluded_parameter_ranks=(2,),
                ),
                StateVariant(
                    "block_second_moment_sharded",
                    base_names + block_names,
                    frozenset(
                        {
                            ParameterLayout.FLATTENED_ELEMENT_SHARD,
                            _DTENSOR_LAYOUT,
                        }
                    ),
                    StateExtent.LOCAL_STORAGE,
                ),
                StateVariant(
                    "factored_second_moment",
                    base_names + factored_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.GLOBAL_PARAMETER,
                    parameter_ranks=(2,),
                ),
                StateVariant(
                    "block_state_pending_factored_initialization",
                    base_names + block_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.GLOBAL_PARAMETER,
                    parameter_ranks=(2,),
                    migration_only=True,
                ),
                StateVariant(
                    "legacy_block_state_pending_factored_initialization",
                    base_names + legacy_block_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.GLOBAL_PARAMETER,
                    parameter_ranks=(2,),
                    migration_only=True,
                ),
                StateVariant(
                    "factored_with_retained_block_state",
                    base_names + factored_names + block_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.GLOBAL_PARAMETER,
                    parameter_ranks=(2,),
                    inactive_fields=block_names,
                    migration_only=True,
                    description="Factored-v state after loading a block-v checkpoint.",
                ),
                StateVariant(
                    "factored_with_retained_legacy_block_state",
                    base_names + factored_names + legacy_block_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.GLOBAL_PARAMETER,
                    parameter_ranks=(2,),
                    inactive_fields=legacy_block_names,
                    migration_only=True,
                    description="Factored-v state after loading a pre-vmean-counter checkpoint.",
                ),
            )
        )
    else:
        variants.extend(
            (
                StateVariant(
                    "block_second_moment",
                    base_names + block_names,
                    layouts,
                    StateExtent.LOCAL_STORAGE,
                ),
                StateVariant(
                    "factored_state_pending_block_initialization",
                    base_names + factored_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.GLOBAL_PARAMETER,
                    parameter_ranks=(2,),
                    migration_only=True,
                ),
                StateVariant(
                    "legacy_block_without_vmean_counter",
                    base_names + legacy_block_names,
                    layouts,
                    StateExtent.LOCAL_STORAGE,
                    migration_only=True,
                ),
                StateVariant(
                    "block_with_retained_factored_state",
                    base_names + block_names + factored_names,
                    frozenset({ParameterLayout.REPLICATED}),
                    StateExtent.LOCAL_STORAGE,
                    parameter_ranks=(2,),
                    inactive_fields=factored_names,
                    migration_only=True,
                    description="Block-v state after loading a factored-v checkpoint.",
                ),
            )
        )
    fields = (
        _common_fields()
        + _base_parameter_fields()
        + block_fields
        + factored_fields
    )
    fields += _derived_fields()
    training = _base_training() + (
        TrainingSupport(
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ProcessGroupScope.NONE,
        ),
    )
    checkpoints = (
        CheckpointSupport(
            CheckpointTransport.NATIVE_OPTIMIZER,
            frozenset(
                {
                    ParameterLayout.REPLICATED,
                    ParameterLayout.FLATTENED_ELEMENT_SHARD,
                }
            ),
            frozenset(),
            ProcessGroupScope.NONE,
        ),
        CheckpointSupport(
            CheckpointTransport.PYTORCH_RANK_LOCAL,
            frozenset({_DTENSOR_LAYOUT}),
            frozenset(),
            ProcessGroupScope.DEFAULT_WORLD,
            mesh_dimensions=(1,),
            requires_collective=True,
            atomic_load=True,
        ),
    )
    return OptimizerContract(
        implementation="gefen.Gefen",
        state_layout=OptimizerStateLayout(fields, tuple(variants)),
        capabilities=_negative_capabilities(
            training=training,
            checkpoints=checkpoints,
            supported_parameter_ranks=None,
        ),
    )


def _muon_initialized_variant(
    *,
    name: str,
    field_names: Tuple[str, ...],
    layout: ParameterLayout,
    extent: StateExtent,
    mode: str,
    role: ParameterStateRole = ParameterStateRole.ANY,
) -> StateVariant:
    return StateVariant(
        name,
        field_names,
        frozenset({layout}),
        extent,
        role=role,
        parameter_ranks=(2,),
        sharded_mode=mode,
    )


def _gefen_muon_contract(
    *,
    sharded_modes: FrozenSet[str],
    normuon_modes: FrozenSet[str],
    non_normuon_modes: FrozenSet[str],
) -> OptimizerContract:
    sharded_modes = _frozenset(sharded_modes)
    normuon_modes = _frozenset(normuon_modes)
    non_normuon_modes = _frozenset(non_normuon_modes)
    normuon_fields = (
        StateField("normuon_v", StateScope.PARAMETER, StateGeometry.ROW, True),
        StateField("normuon_step", StateScope.PARAMETER, StateGeometry.SCALAR, True),
    )
    fields = _common_fields() + _base_parameter_fields()
    if "distributed" in sharded_modes:
        fields += (
            StateField(
                "gefen_muon_distributed",
                StateScope.DERIVED,
                StateGeometry.OPAQUE,
                True,
                applicable_sharded_modes=frozenset({"distributed"}),
                description="Native Parallel-Muon ownership manifest.",
            ),
        )
    if normuon_modes:
        fields += normuon_fields
    fields += _derived_fields()

    base_names = tuple(field.name for field in _base_parameter_fields())
    normuon_names = base_names + tuple(field.name for field in normuon_fields)
    variants = []
    for mode in sorted(sharded_modes):
        variants.append(
            StateVariant(
                "name_only_replicated_" + mode,
                ("name",),
                frozenset({ParameterLayout.REPLICATED}),
                StateExtent.METADATA_ONLY,
                initialized=False,
                parameter_ranks=(2,),
                sharded_mode=mode,
            )
        )
        if mode == "distributed":
            variants.append(
                StateVariant(
                    "distributed_owner_name_only",
                    ("name",),
                    frozenset({_DTENSOR_LAYOUT}),
                    StateExtent.METADATA_ONLY,
                    role=ParameterStateRole.OWNER,
                    initialized=False,
                    parameter_ranks=(2,),
                    sharded_mode=mode,
                )
            )
        else:
            variants.append(
                StateVariant(
                    "name_only_dtensor_" + mode,
                    ("name",),
                    frozenset({_DTENSOR_LAYOUT}),
                    StateExtent.METADATA_ONLY,
                    initialized=False,
                    parameter_ranks=(2,),
                    sharded_mode=mode,
                )
            )
    for mode_set, field_names, prefix in (
        (non_normuon_modes, base_names, "quantized_muon"),
        (normuon_modes, normuon_names, "quantized_normuon"),
    ):
        if not mode_set:
            continue
        for mode in sorted(mode_set):
            variants.append(
                _muon_initialized_variant(
                    name=prefix + "_replicated_" + mode,
                    field_names=field_names,
                    layout=ParameterLayout.REPLICATED,
                    extent=StateExtent.GLOBAL_PARAMETER,
                    mode=mode,
                )
            )
        if "approx" in mode_set:
            variants.append(
                _muon_initialized_variant(
                    name=prefix + "_local",
                    field_names=field_names,
                    layout=_DTENSOR_LAYOUT,
                    extent=StateExtent.LOCAL_STORAGE,
                    mode="approx",
                )
            )
        if "exact" in mode_set:
            variants.append(
                _muon_initialized_variant(
                    name=prefix + "_global",
                    field_names=field_names,
                    layout=_DTENSOR_LAYOUT,
                    extent=StateExtent.GLOBAL_PARAMETER,
                    mode="exact",
                )
            )
        if "distributed" in mode_set:
            variants.append(
                _muon_initialized_variant(
                    name=prefix + "_owner",
                    field_names=field_names,
                    layout=_DTENSOR_LAYOUT,
                    extent=StateExtent.OWNER_PARAMETER,
                    mode="distributed",
                    role=ParameterStateRole.OWNER,
                )
            )
    if "distributed" in sharded_modes:
        variants.extend(
            (
                StateVariant(
                    "distributed_non_owner_name_only",
                    ("name",),
                    frozenset({_DTENSOR_LAYOUT}),
                    StateExtent.METADATA_ONLY,
                    role=ParameterStateRole.NON_OWNER,
                    initialized=False,
                    parameter_ranks=(2,),
                    sharded_mode="distributed",
                ),
                StateVariant(
                    "distributed_non_owner",
                    ("name", "automatic_period"),
                    frozenset({_DTENSOR_LAYOUT}),
                    StateExtent.METADATA_ONLY,
                    role=ParameterStateRole.NON_OWNER,
                    initialized=False,
                    parameter_ranks=(2,),
                    sharded_mode="distributed",
                ),
            )
        )

    training = tuple(
        support
        for mode in sorted(sharded_modes)
        for support in (
            TrainingSupport(
                ParameterLayout.REPLICATED,
                ProcessGroupScope.NONE,
                sharded_mode=mode,
                requires_complete_logical_matrix=True,
            ),
            TrainingSupport(
                _DTENSOR_LAYOUT,
                ProcessGroupScope.INFERRED_DEVICE_MESH,
                mesh_dimensions=(1,),
                sharded_mode=mode,
                requires_complete_logical_matrix=mode in {"exact", "distributed"},
            ),
        )
    )
    checkpoints = [
        CheckpointSupport(
            CheckpointTransport.NATIVE_OPTIMIZER,
            frozenset({ParameterLayout.REPLICATED}),
            frozenset(),
            ProcessGroupScope.NONE,
            required_sharded_modes=sharded_modes,
        )
    ]
    if (
        "approx" in sharded_modes
        and sharded_modes.issubset({"approx", "distributed"})
    ):
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.PYTORCH_RANK_LOCAL,
                frozenset({_DTENSOR_LAYOUT}),
                frozenset(),
                ProcessGroupScope.DEFAULT_WORLD,
                mesh_dimensions=(1,),
                required_sharded_modes=sharded_modes,
                requires_collective=True,
                atomic_load=True,
            )
        )
    if sharded_modes == frozenset({"distributed"}):
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.NATIVE_OPTIMIZER,
                frozenset({_DTENSOR_LAYOUT}),
                frozenset({_DTENSOR_LAYOUT}),
                ProcessGroupScope.INFERRED_DEVICE_MESH,
                topology_change_kinds=frozenset(
                    {TopologyChange.WORLD_SIZE_OWNER_REDISTRIBUTION}
                ),
                mesh_dimensions=(1,),
                required_sharded_modes=frozenset({"distributed"}),
                requires_collective=True,
                atomic_load=True,
            )
        )
    return OptimizerContract(
        implementation="gefen.GefenMuon",
        state_layout=OptimizerStateLayout(fields, tuple(variants)),
        capabilities=_negative_capabilities(
            training=training,
            checkpoints=tuple(checkpoints),
            supported_parameter_ranks=(2,),
        ),
    )


def _hybrid_contract(
    *,
    muon: Optional[OptimizerContract],
    backup: Optional[OptimizerContract],
    backup_implementation: str,
) -> OptimizerContract:
    fields = (
        StateField(
            "backup_optimizer",
            StateScope.OPTIMIZER_COMMON,
            StateGeometry.OPAQUE,
            True,
        ),
    )
    children = []
    if muon is not None:
        children.append(OptimizerChildContract("muon", muon.implementation, muon))
    if backup_implementation:
        children.append(
            OptimizerChildContract("backup", backup_implementation, backup)
        )
    if muon is None:
        training = _base_training()
    else:
        training = muon.capabilities.training
    checkpoints = (
        CheckpointSupport(
            CheckpointTransport.COMPOSITE_NATIVE,
            frozenset({ParameterLayout.REPLICATED}),
            frozenset(),
            ProcessGroupScope.NONE,
        ),
    )
    return OptimizerContract(
        implementation="gefen.GefenMuonHybrid",
        state_layout=OptimizerStateLayout(
            fields,
            (),
            composite_namespaces=("muon", "backup"),
        ),
        capabilities=_negative_capabilities(
            training=training,
            checkpoints=checkpoints,
            supported_parameter_ranks=None,
        ),
        children=tuple(children),
    )


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "CheckpointSupport",
    "CheckpointTransport",
    "OptimizerCapabilities",
    "OptimizerChildContract",
    "OptimizerContract",
    "OptimizerContractProvider",
    "OptimizerStateLayout",
    "ParameterLayout",
    "ParameterStateRole",
    "Precision",
    "ProcessGroupScope",
    "StateExtent",
    "StateField",
    "StateGeometry",
    "StateKeyMatch",
    "StateScope",
    "StateVariant",
    "TrainingSupport",
    "TopologyChange",
]
