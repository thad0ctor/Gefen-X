"""Platform-agnostic optimizer capability and state-layout contracts.

The descriptors in this module are read-only declarations. They describe the
optimizer state Gefen owns and the integration surfaces implemented today
without importing DCP, DTensor, Megatron, DeepSpeed, or another adapter.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import (
    AbstractSet,
    FrozenSet,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)


CONTRACT_SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1


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


class PlacementKind(str, Enum):
    """Framework-neutral placement carried by a stable shard identity."""

    REPLICATE = "replicate"
    FLAT_SHARD = "flat_shard"
    DIMENSION_SHARD = "dimension_shard"
    WHOLE_PARAMETER_OWNER = "whole_parameter_owner"


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
    CANONICAL_LOCAL = "canonical_local"
    CANONICAL_GLOBAL = "canonical_global"


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


def _validate_identity_name(name, value):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "{} must be a non-empty string without outer whitespace".format(name)
        )
    if "\x00" in value:
        raise ValueError("{} must not contain NUL".format(name))


def _validate_identity_schema_version(name, value):
    if type(value) is not int or value != IDENTITY_SCHEMA_VERSION:
        raise ValueError("unsupported {} schema version".format(name))


@dataclass(frozen=True)
class ParameterIdentity:
    """Canonical logical parameter identity, independent of tensor storage."""

    fqn: str
    global_shape: Sequence[int]
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_identity_name("ParameterIdentity.fqn", self.fqn)
        if self.fqn.startswith(".") or self.fqn.endswith(".") or ".." in self.fqn:
            raise ValueError(
                "ParameterIdentity.fqn must contain non-empty dot-separated components"
            )
        if isinstance(self.global_shape, (str, bytes, bytearray)):
            raise TypeError(
                "ParameterIdentity.global_shape must be a sequence of dimensions"
            )
        object.__setattr__(self, "global_shape", _tuple(self.global_shape))
        if any(type(dim) is not int or dim < 0 for dim in self.global_shape):
            raise ValueError(
                "ParameterIdentity.global_shape must contain nonnegative integers"
            )
        _validate_identity_schema_version("parameter identity", self.schema_version)

    @property
    def numel(self) -> int:
        """Return the canonical logical element count."""

        return math.prod(self.global_shape)


@dataclass(frozen=True)
class ProcessGroupIdentity:
    """Stable semantic process-group identity supplied by an adapter."""

    semantic_name: str
    ordered_members: Sequence[str]
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _validate_identity_name(
            "ProcessGroupIdentity.semantic_name", self.semantic_name
        )
        if isinstance(self.ordered_members, (str, bytes)):
            raise TypeError(
                "ProcessGroupIdentity.ordered_members must be a sequence of member IDs"
            )
        object.__setattr__(self, "ordered_members", _tuple(self.ordered_members))
        if not self.ordered_members:
            raise ValueError("ProcessGroupIdentity.ordered_members must be non-empty")
        for member in self.ordered_members:
            _validate_identity_name("process-group member", member)
        if len(set(self.ordered_members)) != len(self.ordered_members):
            raise ValueError("ProcessGroupIdentity.ordered_members must be unique")
        _validate_identity_schema_version("process-group identity", self.schema_version)


@dataclass(frozen=True)
class ShardPlacement:
    """One explicit mesh-axis placement for a logical parameter shard."""

    mesh_axis: str
    kind: PlacementKind
    coordinate: int
    parts: int
    parameter_dimension: Optional[int] = None

    def __post_init__(self) -> None:
        _validate_identity_name("ShardPlacement.mesh_axis", self.mesh_axis)
        if not isinstance(self.kind, PlacementKind):
            raise TypeError("ShardPlacement.kind must be a PlacementKind")
        if type(self.parts) is not int or self.parts <= 0:
            raise ValueError("ShardPlacement.parts must be a positive integer")
        if (
            type(self.coordinate) is not int
            or self.coordinate < 0
            or self.coordinate >= self.parts
        ):
            raise ValueError("ShardPlacement.coordinate must be within parts")
        if self.kind is PlacementKind.DIMENSION_SHARD:
            if (
                type(self.parameter_dimension) is not int
                or self.parameter_dimension < 0
            ):
                raise ValueError(
                    "dimension-shard placement requires a nonnegative parameter dimension"
                )
        elif self.parameter_dimension is not None:
            raise ValueError(
                "only a dimension-shard placement may name a parameter dimension"
            )


@dataclass(frozen=True)
class LogicalSlice:
    """Contiguous range in canonical row-major flattened parameter order."""

    flat_offset: int
    length: int

    def __post_init__(self) -> None:
        if type(self.flat_offset) is not int or self.flat_offset < 0:
            raise ValueError("LogicalSlice.flat_offset must be a nonnegative integer")
        if type(self.length) is not int or self.length < 0:
            raise ValueError("LogicalSlice.length must be a nonnegative integer")

    @classmethod
    def full(cls, parameter: ParameterIdentity) -> "LogicalSlice":
        """Return the complete logical range for ``parameter``."""

        if not isinstance(parameter, ParameterIdentity):
            raise TypeError("parameter must be a ParameterIdentity")
        return cls(0, parameter.numel)


@dataclass(frozen=True)
class LogicalRegion:
    """One axis-aligned region in canonical logical parameter coordinates."""

    offsets: Sequence[int]
    lengths: Sequence[int]

    def __post_init__(self) -> None:
        for name, values in (
            ("LogicalRegion.offsets", self.offsets),
            ("LogicalRegion.lengths", self.lengths),
        ):
            if isinstance(values, (str, bytes, bytearray)):
                raise TypeError("{} must be a sequence of dimensions".format(name))
            try:
                normalized = tuple(values)
            except TypeError as exc:
                raise TypeError(
                    "{} must be a sequence of dimensions".format(name)
                ) from exc
            if any(type(value) is not int or value < 0 for value in normalized):
                raise ValueError("{} must contain nonnegative integers".format(name))
            object.__setattr__(self, name.rsplit(".", 1)[1], normalized)
        if len(self.offsets) != len(self.lengths):
            raise ValueError(
                "LogicalRegion offsets and lengths must have the same rank"
            )

    @property
    def rank(self) -> int:
        """Return the logical tensor rank described by this region."""

        return len(self.offsets)

    @property
    def numel(self) -> int:
        """Return the number of logical elements in this region."""

        return math.prod(self.lengths)

    @classmethod
    def full(cls, parameter: ParameterIdentity) -> "LogicalRegion":
        """Return the complete axis-aligned region for ``parameter``."""

        if not isinstance(parameter, ParameterIdentity):
            raise TypeError("parameter must be a ParameterIdentity")
        return cls((0,) * len(parameter.global_shape), parameter.global_shape)

    def validate_bounds(self, parameter: ParameterIdentity) -> None:
        """Raise when this region is not contained by ``parameter``."""

        if not isinstance(parameter, ParameterIdentity):
            raise TypeError("parameter must be a ParameterIdentity")
        if self.rank != len(parameter.global_shape):
            raise ValueError("LogicalRegion rank must match the global parameter rank")
        if any(
            offset + length > dimension
            for offset, length, dimension in zip(
                self.offsets, self.lengths, parameter.global_shape
            )
        ):
            raise ValueError("LogicalRegion exceeds the global parameter")

    def intersection(self, other: "LogicalRegion") -> "LogicalRegion":
        """Return the axis-aligned intersection with another same-rank region."""

        if not isinstance(other, LogicalRegion):
            raise TypeError("other must be a LogicalRegion")
        if self.rank != other.rank:
            raise ValueError("LogicalRegion intersection requires equal ranks")
        offsets = tuple(
            max(left, right) for left, right in zip(self.offsets, other.offsets)
        )
        ends = tuple(
            min(left_offset + left_length, right_offset + right_length)
            for left_offset, left_length, right_offset, right_length in zip(
                self.offsets,
                self.lengths,
                other.offsets,
                other.lengths,
            )
        )
        return LogicalRegion(
            offsets,
            tuple(max(0, end - offset) for offset, end in zip(offsets, ends)),
        )

    def overlaps(self, other: "LogicalRegion") -> bool:
        """Return whether two regions share at least one logical element."""

        return self.intersection(other).numel > 0

    @staticmethod
    def validate_exact_coverage(
        parameter: ParameterIdentity, regions: Sequence["LogicalRegion"]
    ) -> None:
        """Validate that bounded regions cover a parameter exactly once."""

        if not isinstance(parameter, ParameterIdentity):
            raise TypeError("parameter must be a ParameterIdentity")
        if isinstance(regions, (str, bytes, bytearray)):
            raise TypeError("regions must be a sequence of LogicalRegion values")
        try:
            regions = tuple(regions)
        except TypeError as exc:
            raise TypeError(
                "regions must be a sequence of LogicalRegion values"
            ) from exc
        if any(not isinstance(region, LogicalRegion) for region in regions):
            raise TypeError("regions must contain LogicalRegion values")
        for region in regions:
            region.validate_bounds(parameter)
        for index, region in enumerate(regions):
            if any(region.overlaps(other) for other in regions[index + 1 :]):
                raise ValueError("LogicalRegions must not overlap")
        if sum(region.numel for region in regions) != parameter.numel:
            raise ValueError("LogicalRegions must exactly cover the global parameter")


@dataclass(frozen=True)
class ShardIdentity:
    """Stable identity of one process-group member's logical parameter shard."""

    parameter: ParameterIdentity
    layout: ParameterLayout
    logical_slice: Union[LogicalSlice, LogicalRegion]
    placements: Sequence[ShardPlacement] = ()
    process_group: Optional[ProcessGroupIdentity] = None
    local_member: Optional[str] = None
    owner: Optional[str] = None
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.parameter, ParameterIdentity):
            raise TypeError("ShardIdentity.parameter must be a ParameterIdentity")
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("ShardIdentity.layout must be a ParameterLayout")
        uses_logical_region = isinstance(self.logical_slice, LogicalRegion)
        if self.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
            if not uses_logical_region:
                raise ValueError(
                    "stable DTensor identity requires a logical-region descriptor (LogicalRegion)"
                )
            self.logical_slice.validate_bounds(self.parameter)
        elif not isinstance(self.logical_slice, LogicalSlice):
            raise TypeError(
                "non-DTensor ShardIdentity.logical_slice must be a LogicalSlice"
            )
        placements = _tuple(self.placements)
        if any(not isinstance(item, ShardPlacement) for item in placements):
            raise TypeError(
                "ShardIdentity.placements must contain ShardPlacement values"
            )
        axes = tuple(item.mesh_axis for item in placements)
        if len(set(axes)) != len(axes):
            raise ValueError("ShardIdentity placement mesh axes must be unique")
        object.__setattr__(
            self,
            "placements",
            tuple(sorted(placements, key=lambda item: item.mesh_axis)),
        )
        if (
            not uses_logical_region
            and self.logical_slice.flat_offset + self.logical_slice.length
            > self.parameter.numel
        ):
            raise ValueError("ShardIdentity.logical_slice exceeds the global parameter")
        if self.process_group is None:
            if self.local_member is not None or self.owner is not None:
                raise ValueError(
                    "ShardIdentity members and owners require a process-group identity"
                )
        else:
            if not isinstance(self.process_group, ProcessGroupIdentity):
                raise TypeError(
                    "ShardIdentity.process_group must be a ProcessGroupIdentity"
                )
            if self.local_member not in self.process_group.ordered_members:
                raise ValueError(
                    "ShardIdentity.local_member must belong to the process group"
                )
            if (
                self.owner is not None
                and self.owner not in self.process_group.ordered_members
            ):
                raise ValueError("ShardIdentity.owner must belong to the process group")
        if (
            self.layout is not ParameterLayout.WHOLE_PARAMETER_OWNER
            and self.owner is not None
        ):
            raise ValueError(
                "ShardIdentity.owner is valid only for whole-parameter ownership"
            )

        full = (
            self.logical_slice == LogicalRegion.full(self.parameter)
            if uses_logical_region
            else self.logical_slice == LogicalSlice.full(self.parameter)
        )
        kinds = tuple(item.kind for item in self.placements)
        if self.process_group is not None:
            member_index = self.process_group.ordered_members.index(self.local_member)
            for placement in self.placements:
                if (
                    placement.parts != len(self.process_group.ordered_members)
                    or placement.coordinate != member_index
                ):
                    raise ValueError(
                        "ShardIdentity placement coordinates must match the ordered process-group members"
                    )
        for placement in self.placements:
            if (
                placement.parameter_dimension is not None
                and placement.parameter_dimension >= len(self.parameter.global_shape)
            ):
                raise ValueError(
                    "ShardIdentity placement dimension exceeds parameter rank"
                )
        if self.layout is ParameterLayout.REPLICATED:
            if not full or self.owner is not None:
                raise ValueError("replicated identity must cover the full parameter")
            if any(kind is not PlacementKind.REPLICATE for kind in kinds):
                raise ValueError("replicated identity has a non-replicated placement")
            if self.process_group is None and self.placements:
                raise ValueError("an ungrouped replicated identity has no placements")
            if self.process_group is not None and len(self.placements) != 1:
                raise ValueError(
                    "a process-group replicated identity requires one placement"
                )
        elif self.layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
            if self.process_group is None or self.owner is not None:
                raise ValueError(
                    "flattened element shards require a process group and no owner"
                )
            if len(kinds) != 1 or kinds[0] is not PlacementKind.FLAT_SHARD:
                raise ValueError(
                    "flattened element shards require one flat-shard placement"
                )
        elif self.layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
            if self.process_group is None or self.owner is None:
                raise ValueError(
                    "whole-parameter ownership requires a process group and owner"
                )
            owns_parameter = self.local_member == self.owner
            if owns_parameter and not full:
                raise ValueError(
                    "the owner must carry the full whole-parameter logical slice"
                )
            if not owns_parameter and self.logical_slice != LogicalSlice(0, 0):
                raise ValueError("a non-owner whole-parameter slice must be empty")
            if len(kinds) != 1 or kinds[0] is not PlacementKind.WHOLE_PARAMETER_OWNER:
                raise ValueError(
                    "whole-parameter ownership requires one owner placement"
                )
        elif self.layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
            if self.process_group is None or self.owner is not None:
                raise ValueError(
                    "DTensor identities require a process group and no owner"
                )
            if len(self.placements) != 1 or kinds[0] not in {
                PlacementKind.DIMENSION_SHARD,
                PlacementKind.REPLICATE,
            }:
                raise ValueError(
                    "one-dimensional DTensor identities require one dimension-shard or replicate placement"
                )
            placement = self.placements[0]
            if placement.kind is PlacementKind.REPLICATE:
                if not full:
                    raise ValueError(
                        "replicated DTensor identities must cover the full parameter"
                    )
            else:
                shard_dimension = placement.parameter_dimension
                for dimension, (offset, length, global_length) in enumerate(
                    zip(
                        self.logical_slice.offsets,
                        self.logical_slice.lengths,
                        self.parameter.global_shape,
                    )
                ):
                    if dimension != shard_dimension and (
                        offset != 0 or length != global_length
                    ):
                        raise ValueError(
                            "a dimension-sharded DTensor region must cover every unsharded parameter dimension"
                        )
        _validate_identity_schema_version("shard identity", self.schema_version)

    @property
    def logical_region(self) -> Optional[LogicalRegion]:
        """Return the axis-aligned logical region, when this identity has one."""

        if isinstance(self.logical_slice, LogicalRegion):
            return self.logical_slice
        return None

    @property
    def sort_key(self):
        """Return a deterministic structural ordering key."""

        member_index = -1
        group_name = ""
        if self.process_group is not None:
            group_name = self.process_group.semantic_name
            member_index = self.process_group.ordered_members.index(self.local_member)
        owner_index = -1
        if self.process_group is not None and self.owner is not None:
            owner_index = self.process_group.ordered_members.index(self.owner)
        placement_key = tuple(
            (
                item.mesh_axis,
                item.kind.value,
                item.coordinate,
                item.parts,
                -1 if item.parameter_dimension is None else item.parameter_dimension,
            )
            for item in self.placements
        )
        if isinstance(self.logical_slice, LogicalSlice):
            # Preserve the released contiguous-slice key exactly. Besides
            # retaining its public structural shape, this leaves all existing
            # manifest ordering and freshness tokens byte-for-byte stable.
            return (
                self.parameter.fqn,
                self.logical_slice.flat_offset,
                self.logical_slice.length,
                self.layout.value,
                group_name,
                member_index,
                owner_index,
                placement_key,
            )

        flat_offset = 0
        stride = 1
        for offset, dimension in reversed(
            tuple(zip(self.logical_slice.offsets, self.parameter.global_shape))
        ):
            flat_offset += offset * stride
            stride *= dimension
        return (
            self.parameter.fqn,
            flat_offset,
            self.logical_slice.numel,
            self.layout.value,
            group_name,
            member_index,
            owner_index,
            placement_key,
            self.logical_slice.offsets,
            self.logical_slice.lengths,
        )


@dataclass(frozen=True)
class ShardingManifest:
    """Complete deterministic shard identity set supplied by one adapter."""

    shards: Sequence[ShardIdentity]
    schema_version: int = IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        shards = _tuple(self.shards)
        if not shards:
            raise ValueError("ShardingManifest.shards must be non-empty")
        if any(not isinstance(item, ShardIdentity) for item in shards):
            raise TypeError("ShardingManifest.shards must contain ShardIdentity values")
        ordered = tuple(sorted(shards, key=lambda item: item.sort_key))
        if len(set(ordered)) != len(ordered):
            raise ValueError("ShardingManifest.shards must be unique")
        object.__setattr__(self, "shards", ordered)
        _validate_identity_schema_version("sharding manifest", self.schema_version)

        by_fqn = {}
        for shard in ordered:
            by_fqn.setdefault(shard.parameter.fqn, []).append(shard)
        for fqn, parameter_shards in by_fqn.items():
            parameter = parameter_shards[0].parameter
            if any(item.parameter != parameter for item in parameter_shards[1:]):
                raise ValueError(
                    "manifest shards for {!r} disagree on parameter identity".format(
                        fqn
                    )
                )
            layouts = {item.layout for item in parameter_shards}
            groups = {item.process_group for item in parameter_shards}
            if len(layouts) != 1 or len(groups) != 1:
                raise ValueError(
                    "manifest shards for {!r} disagree on layout or process group".format(
                        fqn
                    )
                )
            layout = parameter_shards[0].layout
            group = parameter_shards[0].process_group
            members = tuple(item.local_member for item in parameter_shards)
            if group is None:
                if len(parameter_shards) != 1:
                    raise ValueError(
                        "an ungrouped parameter must have exactly one manifest shard"
                    )
            elif set(members) != set(group.ordered_members) or len(members) != len(
                group.ordered_members
            ):
                raise ValueError(
                    "manifest shards must contain each process-group member exactly once"
                )
            if group is not None:
                placement_shapes = {
                    tuple(
                        (
                            placement.mesh_axis,
                            placement.kind,
                            placement.parts,
                            placement.parameter_dimension,
                        )
                        for placement in item.placements
                    )
                    for item in parameter_shards
                }
                if len(placement_shapes) != 1:
                    raise ValueError(
                        "manifest member placements must agree on one topology"
                    )

            if layout is ParameterLayout.FLATTENED_ELEMENT_SHARD:
                cursor = 0
                boundaries = {0}
                empty_offsets = []
                for item in sorted(
                    parameter_shards,
                    key=lambda shard: (
                        shard.logical_slice.flat_offset,
                        group.ordered_members.index(shard.local_member),
                    ),
                ):
                    if item.logical_slice.length == 0:
                        empty_offsets.append(item.logical_slice.flat_offset)
                        continue
                    if item.logical_slice.flat_offset != cursor:
                        raise ValueError(
                            "flattened manifest slices must be gapless and non-overlapping"
                        )
                    cursor += item.logical_slice.length
                    boundaries.add(cursor)
                if cursor != parameter.numel:
                    raise ValueError(
                        "flattened manifest slices must cover the global parameter"
                    )
                if any(offset not in boundaries for offset in empty_offsets):
                    raise ValueError(
                        "empty flattened manifest slices must use a partition boundary"
                    )
            elif layout is ParameterLayout.REPLICATED:
                if any(
                    item.logical_slice != LogicalSlice.full(parameter)
                    for item in parameter_shards
                ):
                    raise ValueError("replicated manifest shards must all be complete")
            elif layout is ParameterLayout.WHOLE_PARAMETER_OWNER:
                owners = {item.owner for item in parameter_shards}
                if len(owners) != 1:
                    raise ValueError(
                        "whole-parameter manifest shards must agree on one owner"
                    )
            elif layout is ParameterLayout.DTENSOR_1D_DEFAULT_WORLD:
                placements = tuple(item.placements[0] for item in parameter_shards)
                placement_kind = placements[0].kind
                if placement_kind is PlacementKind.REPLICATE:
                    if any(
                        item.logical_region != LogicalRegion.full(parameter)
                        for item in parameter_shards
                    ):
                        raise ValueError(
                            "replicated DTensor manifest regions must all be complete"
                        )
                    continue

                shard_dimension = placements[0].parameter_dimension
                by_coordinate = sorted(
                    parameter_shards,
                    key=lambda item: item.placements[0].coordinate,
                )
                cursor = 0
                for item in by_coordinate:
                    region = item.logical_region
                    if region.offsets[shard_dimension] != cursor:
                        raise ValueError(
                            "dimension-sharded DTensor manifest regions must be "
                            "gapless and non-overlapping in coordinate order"
                        )
                    cursor += region.lengths[shard_dimension]
                if cursor != parameter.global_shape[shard_dimension]:
                    raise ValueError(
                        "dimension-sharded DTensor manifest regions must cover the global parameter"
                    )

    def for_parameter(self, fqn: str) -> Tuple[ShardIdentity, ...]:
        """Return one canonical parameter's shards in deterministic order."""

        return tuple(item for item in self.shards if item.parameter.fqn == fqn)


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
    optional: bool = False

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
        if type(self.optional) is not bool:
            raise TypeError("StateField.optional must be a bool")

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
        if self.parameter_ranks is not None and set(self.parameter_ranks) & set(
            self.excluded_parameter_ranks
        ):
            raise ValueError("included and excluded parameter ranks must be disjoint")
        if self.parameter_ranks is not None:
            _validate_dimensions(
                "parameter_ranks", self.parameter_ranks, positive=False
            )
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
        object.__setattr__(
            self, "composite_namespaces", _tuple(self.composite_namespaces)
        )
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
    requires_post_step_parameter_sync: bool = False

    def __post_init__(self) -> None:
        if self.mesh_dimensions is not None:
            object.__setattr__(self, "mesh_dimensions", _tuple(self.mesh_dimensions))
            _validate_dimensions("mesh_dimensions", self.mesh_dimensions, positive=True)
        if not isinstance(self.layout, ParameterLayout):
            raise TypeError("TrainingSupport.layout must be a ParameterLayout")
        if not isinstance(self.process_group_scope, ProcessGroupScope):
            raise TypeError(
                "TrainingSupport.process_group_scope must be a ProcessGroupScope"
            )
        for name in (
            "requires_complete_parameter_storage",
            "requires_complete_logical_matrix",
            "requires_post_step_parameter_sync",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError("TrainingSupport.{} must be a bool".format(name))


@dataclass(frozen=True)
class CheckpointSupport:
    """One transport's separately qualified checkpoint capability.

    ``atomic_load`` means each participating optimizer instance validates and
    prepares its core restore before local mutation. It does not claim a
    coordinated all-rank commit after failures outside the declared process
    group or arbitrary user hook side effects.
    """

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
        object.__setattr__(
            self, "topology_changing", _frozenset(self.topology_changing)
        )
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
            _validate_dimensions("mesh_dimensions", self.mesh_dimensions, positive=True)
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
        if any(not isinstance(item, TrainingSupport) for item in self.training):
            raise TypeError(
                "OptimizerCapabilities.training must contain TrainingSupport values"
            )
        if any(not isinstance(item, CheckpointSupport) for item in self.checkpoints):
            raise TypeError(
                "OptimizerCapabilities.checkpoints must contain CheckpointSupport values"
            )
        if any(not isinstance(item, Precision) for item in self.precisions):
            raise TypeError(
                "OptimizerCapabilities.precisions must contain Precision values"
            )
        for name in (
            "accepts_semantic_parameter_names",
            "canonical_parameter_fqns",
            "stable_shard_identity",
            "explicit_process_group_codebook_scope",
            "shard_rebinding",
            "post_sharding",
            "canonical_state_io",
            "atomic_state_movement",
            "state_offload",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(
                    "OptimizerCapabilities.{} must be a bool".format(name)
                )
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
        if self.contract is not None and not isinstance(
            self.contract, OptimizerContract
        ):
            raise TypeError(
                "OptimizerChildContract.contract must be an OptimizerContract or None"
            )


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


@runtime_checkable
class CanonicalStateProvider(Protocol):
    """Structural protocol for exact-binding canonical local state."""

    def export_canonical_state(self):
        """Return a versioned local canonical state fragment."""

    def prepare_canonical_state_import(self, state):
        """Validate and stage a canonical fragment without live mutation."""

    def commit_canonical_state_import(self, prepared) -> None:
        """Commit one still-current prepared import."""

    def import_canonical_state(self, state) -> None:
        """Prepare and commit a canonical fragment atomically."""


@runtime_checkable
class PortableStateProvider(Protocol):
    """Structural protocol for collective topology-neutral state I/O."""

    def export_portable_state(
        self, *, checkpoint_process_group, transaction_id, limits
    ):
        """Collectively return one complete portable global-state document."""

    def import_portable_state(
        self,
        state,
        *,
        checkpoint_process_group,
        transaction_id,
        limits,
    ) -> None:
        """Collectively stage and atomically publish portable global state."""


@runtime_checkable
class StateMovementProvider(Protocol):
    """Structural protocol for quiescent atomic optimizer-state movement."""

    def move_state_(self, device=None) -> None:
        """Co-locate authoritative state with the optimizer's live parameters."""


@runtime_checkable
class StateOffloadProvider(Protocol):
    """Structural protocol for persistent live optimizer-state offload."""

    @property
    def state_offload_device(self):
        """Return the active offload device, or ``None`` while resident."""

    def offload_state_(self, device="cpu") -> None:
        """Atomically park supported state and enable transparent step paging."""

    def restore_state_(self) -> None:
        """Atomically co-locate state with parameters and disable offload."""


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
        StateField(
            "gefen_codebook_scope",
            StateScope.OPTIMIZER_COMMON,
            StateGeometry.OPAQUE,
            True,
            description="Stable adapter-defined learned-codebook process-group scope.",
            optional=True,
        ),
    )


def _base_parameter_fields() -> Tuple[StateField, ...]:
    return (
        StateField("name", StateScope.PARAMETER, StateGeometry.OPAQUE, True),
        StateField(
            "automatic_period", StateScope.PARAMETER, StateGeometry.SCALAR, True
        ),
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
        StateField(
            "_gefen_codebook_process_group",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            False,
            description="Live runtime handle for the adapter-defined codebook scope.",
        ),
        StateField(
            "_gefen_codebook_scope_validated",
            StateScope.DERIVED,
            StateGeometry.SCALAR,
            False,
            description="Rebuildable cross-member scope-agreement cache.",
        ),
        StateField(
            "_sr_seed_by_device", StateScope.DERIVED, StateGeometry.SCALAR, False
        ),
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
        StateField(
            "gefen_native_local_shards",
            StateScope.DERIVED,
            StateGeometry.OPAQUE,
            True,
            description="Native rank-local identity guard for scoped physical shards.",
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
    canonical_parameter_fqns: bool = False,
    stable_shard_identity: bool = False,
    explicit_process_group_codebook_scope: bool = False,
    shard_rebinding: bool = False,
    post_sharding: bool = False,
    canonical_state_io: bool = False,
    atomic_state_movement: bool = False,
    state_offload: bool = False,
) -> OptimizerCapabilities:
    return OptimizerCapabilities(
        training=training,
        checkpoints=checkpoints,
        precisions=_ALL_PRECISIONS,
        supported_parameter_ranks=supported_parameter_ranks,
        accepts_semantic_parameter_names=True,
        canonical_parameter_fqns=canonical_parameter_fqns,
        stable_shard_identity=stable_shard_identity,
        explicit_process_group_codebook_scope=explicit_process_group_codebook_scope,
        shard_rebinding=shard_rebinding,
        post_sharding=post_sharding,
        canonical_state_io=canonical_state_io,
        atomic_state_movement=atomic_state_movement,
        state_offload=state_offload,
    )


def _gefen_contract(
    *,
    factored_v_2d: bool,
    canonical_parameter_fqns: bool = False,
    stable_shard_identity: bool = False,
    explicit_process_group_codebook_scope: bool = False,
    native_flattened_checkpoint: bool = False,
    canonical_state_layouts: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_same_topology: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_topology_changing: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_topology_change_kinds: AbstractSet[TopologyChange] = frozenset(),
    atomic_state_movement: bool = False,
    state_offload: bool = False,
) -> OptimizerContract:
    canonical_state_layouts = _frozenset(canonical_state_layouts)
    canonical_global_same_topology = _frozenset(canonical_global_same_topology)
    canonical_global_topology_changing = _frozenset(canonical_global_topology_changing)
    canonical_global_topology_change_kinds = _frozenset(
        canonical_global_topology_change_kinds
    )
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
        StateVariant(
            "period_selected",
            ("name", "automatic_period"),
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
        _common_fields() + _base_parameter_fields() + block_fields + factored_fields
    )
    fields += _derived_fields()
    training = _base_training() + (
        TrainingSupport(
            ParameterLayout.FLATTENED_ELEMENT_SHARD,
            ProcessGroupScope.NONE,
        ),
    )
    checkpoints = [
        CheckpointSupport(
            CheckpointTransport.NATIVE_OPTIMIZER,
            frozenset({ParameterLayout.REPLICATED})
            | (
                frozenset({ParameterLayout.FLATTENED_ELEMENT_SHARD})
                if native_flattened_checkpoint
                else frozenset()
            ),
            frozenset(),
            ProcessGroupScope.NONE,
            atomic_load=True,
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
    ]
    if canonical_state_layouts:
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.CANONICAL_LOCAL,
                canonical_state_layouts,
                frozenset(),
                ProcessGroupScope.NONE,
                atomic_load=True,
            )
        )
    if canonical_global_same_topology or canonical_global_topology_changing:
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.CANONICAL_GLOBAL,
                canonical_global_same_topology,
                canonical_global_topology_changing,
                ProcessGroupScope.ADAPTER_DEFINED,
                topology_change_kinds=canonical_global_topology_change_kinds,
                requires_collective=True,
                atomic_load=True,
            )
        )
    return OptimizerContract(
        implementation="gefen.Gefen",
        state_layout=OptimizerStateLayout(fields, tuple(variants)),
        capabilities=_negative_capabilities(
            training=training,
            checkpoints=tuple(checkpoints),
            supported_parameter_ranks=None,
            canonical_parameter_fqns=canonical_parameter_fqns,
            stable_shard_identity=stable_shard_identity,
            explicit_process_group_codebook_scope=explicit_process_group_codebook_scope,
            shard_rebinding=True,
            post_sharding=True,
            canonical_state_io=bool(
                canonical_state_layouts
                or canonical_global_same_topology
                or canonical_global_topology_changing
            ),
            atomic_state_movement=atomic_state_movement,
            state_offload=state_offload,
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
    canonical_parameter_fqns: bool = False,
    stable_shard_identity: bool = False,
    explicit_process_group_codebook_scope: bool = False,
    whole_parameter_owner: bool = False,
    canonical_state_layouts: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_same_topology: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_topology_changing: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_topology_change_kinds: AbstractSet[TopologyChange] = frozenset(),
    atomic_state_movement: bool = False,
) -> OptimizerContract:
    canonical_state_layouts = _frozenset(canonical_state_layouts)
    canonical_global_same_topology = _frozenset(canonical_global_same_topology)
    canonical_global_topology_changing = _frozenset(canonical_global_topology_changing)
    canonical_global_topology_change_kinds = _frozenset(
        canonical_global_topology_change_kinds
    )
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
        variants.append(
            StateVariant(
                "period_selected_replicated_" + mode,
                ("name", "automatic_period"),
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
            variants.append(
                StateVariant(
                    "distributed_owner_period_selected",
                    ("name", "automatic_period"),
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
            variants.append(
                StateVariant(
                    "period_selected_dtensor_" + mode,
                    ("name", "automatic_period"),
                    frozenset({_DTENSOR_LAYOUT}),
                    StateExtent.METADATA_ONLY,
                    initialized=False,
                    parameter_ranks=(2,),
                    sharded_mode=mode,
                )
            )
        if whole_parameter_owner:
            variants.append(
                StateVariant(
                    "name_only_whole_owner_" + mode,
                    ("name",),
                    frozenset({ParameterLayout.WHOLE_PARAMETER_OWNER}),
                    StateExtent.METADATA_ONLY,
                    role=ParameterStateRole.OWNER,
                    initialized=False,
                    parameter_ranks=(2,),
                    sharded_mode=mode,
                )
            )
            variants.append(
                StateVariant(
                    "period_selected_whole_owner_" + mode,
                    ("name", "automatic_period"),
                    frozenset({ParameterLayout.WHOLE_PARAMETER_OWNER}),
                    StateExtent.METADATA_ONLY,
                    role=ParameterStateRole.OWNER,
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
            if whole_parameter_owner:
                variants.append(
                    _muon_initialized_variant(
                        name=prefix + "_whole_owner_" + mode,
                        field_names=field_names,
                        layout=ParameterLayout.WHOLE_PARAMETER_OWNER,
                        extent=StateExtent.OWNER_PARAMETER,
                        mode=mode,
                        role=ParameterStateRole.OWNER,
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
    if whole_parameter_owner:
        training += tuple(
            TrainingSupport(
                ParameterLayout.WHOLE_PARAMETER_OWNER,
                ProcessGroupScope.ADAPTER_DEFINED,
                sharded_mode=mode,
                requires_complete_parameter_storage=True,
                requires_complete_logical_matrix=True,
                requires_post_step_parameter_sync=True,
            )
            for mode in sorted(sharded_modes)
        )
    checkpoints = [
        CheckpointSupport(
            CheckpointTransport.NATIVE_OPTIMIZER,
            frozenset({ParameterLayout.REPLICATED}),
            frozenset(),
            ProcessGroupScope.NONE,
            required_sharded_modes=sharded_modes,
            atomic_load=True,
        )
    ]
    if "approx" in sharded_modes and sharded_modes.issubset({"approx", "distributed"}):
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
    if canonical_state_layouts:
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.CANONICAL_LOCAL,
                canonical_state_layouts,
                frozenset(),
                ProcessGroupScope.NONE,
                required_sharded_modes=sharded_modes,
                atomic_load=True,
            )
        )
    if canonical_global_same_topology or canonical_global_topology_changing:
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.CANONICAL_GLOBAL,
                canonical_global_same_topology,
                canonical_global_topology_changing,
                ProcessGroupScope.ADAPTER_DEFINED,
                topology_change_kinds=canonical_global_topology_change_kinds,
                required_sharded_modes=sharded_modes,
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
            canonical_parameter_fqns=canonical_parameter_fqns,
            stable_shard_identity=stable_shard_identity,
            explicit_process_group_codebook_scope=explicit_process_group_codebook_scope,
            shard_rebinding=True,
            post_sharding=True,
            canonical_state_io=bool(
                canonical_state_layouts
                or canonical_global_same_topology
                or canonical_global_topology_changing
            ),
            atomic_state_movement=atomic_state_movement,
        ),
    )


def _hybrid_contract(
    *,
    muon: Optional[OptimizerContract],
    backup: Optional[OptimizerContract],
    backup_implementation: str,
    canonical_parameter_fqns: bool = False,
    stable_shard_identity: bool = False,
    explicit_process_group_codebook_scope: bool = False,
    shard_rebinding: bool = False,
    post_sharding: bool = False,
    canonical_global_same_topology: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_topology_changing: AbstractSet[ParameterLayout] = frozenset(),
    canonical_global_topology_change_kinds: AbstractSet[TopologyChange] = frozenset(),
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
        children.append(OptimizerChildContract("backup", backup_implementation, backup))
    if muon is None:
        training = _base_training()
    else:
        training = muon.capabilities.training
    canonical_global_same_topology = _frozenset(canonical_global_same_topology)
    canonical_global_topology_changing = _frozenset(canonical_global_topology_changing)
    canonical_global_topology_change_kinds = _frozenset(
        canonical_global_topology_change_kinds
    )
    checkpoints = [
        CheckpointSupport(
            CheckpointTransport.COMPOSITE_NATIVE,
            frozenset({ParameterLayout.REPLICATED}),
            frozenset(),
            ProcessGroupScope.NONE,
        ),
    ]
    if canonical_global_same_topology or canonical_global_topology_changing:
        checkpoints.append(
            CheckpointSupport(
                CheckpointTransport.CANONICAL_GLOBAL,
                canonical_global_same_topology,
                canonical_global_topology_changing,
                ProcessGroupScope.ADAPTER_DEFINED,
                topology_change_kinds=canonical_global_topology_change_kinds,
                requires_collective=True,
                atomic_load=True,
            )
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
            checkpoints=tuple(checkpoints),
            supported_parameter_ranks=None,
            canonical_parameter_fqns=canonical_parameter_fqns,
            stable_shard_identity=stable_shard_identity,
            explicit_process_group_codebook_scope=explicit_process_group_codebook_scope,
            shard_rebinding=shard_rebinding,
            post_sharding=post_sharding,
            canonical_state_io=bool(
                canonical_global_same_topology or canonical_global_topology_changing
            ),
        ),
        children=tuple(children),
    )


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "CheckpointSupport",
    "CheckpointTransport",
    "CanonicalStateProvider",
    "OptimizerCapabilities",
    "OptimizerChildContract",
    "OptimizerContract",
    "OptimizerContractProvider",
    "OptimizerStateLayout",
    "LogicalRegion",
    "LogicalSlice",
    "ParameterLayout",
    "ParameterIdentity",
    "ParameterStateRole",
    "PortableStateProvider",
    "PlacementKind",
    "Precision",
    "ProcessGroupScope",
    "ProcessGroupIdentity",
    "ShardIdentity",
    "ShardPlacement",
    "ShardingManifest",
    "StateExtent",
    "StateField",
    "StateGeometry",
    "StateKeyMatch",
    "StateMovementProvider",
    "StateOffloadProvider",
    "StateScope",
    "StateVariant",
    "TrainingSupport",
    "TopologyChange",
]
