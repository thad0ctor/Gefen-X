"""Strict primitive wire codecs for portable optimizer identities."""

from gefen.contracts import (
    LogicalRegion,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)


_PARAMETER_IDENTITY_KEYS = frozenset({"schema_version", "fqn", "global_shape"})
_PROCESS_GROUP_IDENTITY_KEYS = frozenset({"schema_version", "semantic_name", "ordered_members"})
_SHARD_IDENTITY_KEYS = frozenset(
    {
        "schema_version",
        "parameter",
        "layout",
        "logical_extent",
        "placements",
        "process_group",
        "local_member",
        "owner",
    }
)
_LOGICAL_SLICE_KEYS = frozenset({"kind", "flat_offset", "length"})
_LOGICAL_REGION_KEYS = frozenset({"kind", "offsets", "lengths"})
_SHARD_PLACEMENT_KEYS = frozenset(
    {
        "mesh_axis",
        "kind",
        "coordinate",
        "parts",
        "parameter_dimension",
    }
)
_SHARDING_MANIFEST_KEYS = frozenset({"schema_version", "shards"})


def _require_exact_record(value, keys, *, name):
    if type(value) is not dict or set(value) != keys:
        raise ValueError("{} has an invalid schema".format(name))
    return value


def _require_exact_list(value, *, name):
    if type(value) is not list:
        raise ValueError("{} must be a list".format(name))
    return value


def _require_optional_string(value, *, name):
    if value is not None and type(value) is not str:
        raise ValueError("{} must be a string or None".format(name))
    return value


def _serialize_parameter_identity(identity):
    if not isinstance(identity, ParameterIdentity):
        raise TypeError("identity must be a ParameterIdentity")
    return {
        "schema_version": identity.schema_version,
        "fqn": identity.fqn,
        "global_shape": list(identity.global_shape),
    }


def _parse_parameter_identity(record):
    record = _require_exact_record(record, _PARAMETER_IDENTITY_KEYS, name="parameter identity")
    shape = _require_exact_list(record["global_shape"], name="parameter identity global_shape")
    if type(record["schema_version"]) is not int:
        raise ValueError("parameter identity schema_version must be an int")
    if type(record["fqn"]) is not str:
        raise ValueError("parameter identity fqn must be a string")
    if any(type(dimension) is not int for dimension in shape):
        raise ValueError("parameter identity global_shape must contain only integers")
    try:
        return ParameterIdentity(
            fqn=record["fqn"],
            global_shape=tuple(shape),
            schema_version=record["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("parameter identity is invalid") from exc


def _normalize_parameter_identity(record):
    return _serialize_parameter_identity(_parse_parameter_identity(record))


def _serialize_process_group_identity(identity):
    if not isinstance(identity, ProcessGroupIdentity):
        raise TypeError("identity must be a ProcessGroupIdentity")
    return {
        "schema_version": identity.schema_version,
        "semantic_name": identity.semantic_name,
        "ordered_members": list(identity.ordered_members),
    }


def _parse_process_group_identity(record):
    record = _require_exact_record(
        record,
        _PROCESS_GROUP_IDENTITY_KEYS,
        name="process-group identity",
    )
    members = _require_exact_list(
        record["ordered_members"],
        name="process-group identity ordered_members",
    )
    if type(record["schema_version"]) is not int:
        raise ValueError("process-group identity schema_version must be an int")
    if type(record["semantic_name"]) is not str:
        raise ValueError("process-group identity semantic_name must be a string")
    if any(type(member) is not str for member in members):
        raise ValueError("process-group identity ordered_members must contain only strings")
    try:
        return ProcessGroupIdentity(
            semantic_name=record["semantic_name"],
            ordered_members=tuple(members),
            schema_version=record["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("process-group identity is invalid") from exc


def _normalize_process_group_identity(record):
    return _serialize_process_group_identity(_parse_process_group_identity(record))


def _serialize_logical_extent(logical_extent):
    if isinstance(logical_extent, LogicalSlice):
        return {
            "kind": "logical_slice",
            "flat_offset": logical_extent.flat_offset,
            "length": logical_extent.length,
        }
    if isinstance(logical_extent, LogicalRegion):
        return {
            "kind": "logical_region",
            "offsets": list(logical_extent.offsets),
            "lengths": list(logical_extent.lengths),
        }
    raise TypeError("logical extent must be a LogicalSlice or LogicalRegion")


def _parse_logical_extent(record):
    if type(record) is not dict or type(record.get("kind")) is not str:
        raise ValueError("shard logical_extent has an invalid schema")
    kind = record["kind"]
    if kind == "logical_slice":
        _require_exact_record(record, _LOGICAL_SLICE_KEYS, name="shard logical_slice")
        if type(record["flat_offset"]) is not int or type(record["length"]) is not int:
            raise ValueError("shard logical_slice offsets and lengths must be integers")
        try:
            return LogicalSlice(record["flat_offset"], record["length"])
        except (TypeError, ValueError) as exc:
            raise ValueError("shard logical_slice is invalid") from exc
    if kind == "logical_region":
        _require_exact_record(record, _LOGICAL_REGION_KEYS, name="shard logical_region")
        offsets = _require_exact_list(record["offsets"], name="shard logical_region offsets")
        lengths = _require_exact_list(record["lengths"], name="shard logical_region lengths")
        if any(type(value) is not int for value in offsets + lengths):
            raise ValueError("shard logical_region offsets and lengths must contain only integers")
        try:
            return LogicalRegion(tuple(offsets), tuple(lengths))
        except (TypeError, ValueError) as exc:
            raise ValueError("shard logical_region is invalid") from exc
    raise ValueError("unsupported shard logical_extent kind")


def _serialize_shard_placement(placement):
    if not isinstance(placement, ShardPlacement):
        raise TypeError("placement must be a ShardPlacement")
    return {
        "mesh_axis": placement.mesh_axis,
        "kind": placement.kind.value,
        "coordinate": placement.coordinate,
        "parts": placement.parts,
        "parameter_dimension": placement.parameter_dimension,
    }


def _parse_shard_placement(record):
    record = _require_exact_record(record, _SHARD_PLACEMENT_KEYS, name="shard placement")
    if type(record["mesh_axis"]) is not str:
        raise ValueError("shard placement mesh_axis must be a string")
    if type(record["kind"]) is not str:
        raise ValueError("shard placement kind must be a string")
    if type(record["coordinate"]) is not int:
        raise ValueError("shard placement coordinate must be an int")
    if type(record["parts"]) is not int:
        raise ValueError("shard placement parts must be an int")
    if record["parameter_dimension"] is not None and type(record["parameter_dimension"]) is not int:
        raise ValueError("shard placement parameter_dimension must be an int or None")
    try:
        kind = PlacementKind(record["kind"])
        return ShardPlacement(
            mesh_axis=record["mesh_axis"],
            kind=kind,
            coordinate=record["coordinate"],
            parts=record["parts"],
            parameter_dimension=record["parameter_dimension"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("shard placement is invalid") from exc


def _serialize_shard_identity(identity):
    if not isinstance(identity, ShardIdentity):
        raise TypeError("identity must be a ShardIdentity")
    return {
        "schema_version": identity.schema_version,
        "parameter": _serialize_parameter_identity(identity.parameter),
        "layout": identity.layout.value,
        "logical_extent": _serialize_logical_extent(identity.logical_slice),
        "placements": [_serialize_shard_placement(placement) for placement in identity.placements],
        "process_group": (
            None if identity.process_group is None else _serialize_process_group_identity(identity.process_group)
        ),
        "local_member": identity.local_member,
        "owner": identity.owner,
    }


def _parse_shard_identity(record):
    record = _require_exact_record(record, _SHARD_IDENTITY_KEYS, name="shard identity")
    if type(record["schema_version"]) is not int:
        raise ValueError("shard identity schema_version must be an int")
    if type(record["layout"]) is not str:
        raise ValueError("shard identity layout must be a string")
    placements = _require_exact_list(record["placements"], name="shard identity placements")
    _require_optional_string(record["local_member"], name="shard identity local_member")
    _require_optional_string(record["owner"], name="shard identity owner")
    try:
        parameter = _parse_parameter_identity(record["parameter"])
        logical_extent = _parse_logical_extent(record["logical_extent"])
        parsed_placements = tuple(_parse_shard_placement(placement) for placement in placements)
        process_group = (
            None if record["process_group"] is None else _parse_process_group_identity(record["process_group"])
        )
        layout = ParameterLayout(record["layout"])
        return ShardIdentity(
            parameter=parameter,
            layout=layout,
            logical_slice=logical_extent,
            placements=parsed_placements,
            process_group=process_group,
            local_member=record["local_member"],
            owner=record["owner"],
            schema_version=record["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("shard identity is invalid") from exc


def _normalize_shard_identity(record):
    return _serialize_shard_identity(_parse_shard_identity(record))


def _serialize_sharding_manifest(manifest):
    if not isinstance(manifest, ShardingManifest):
        raise TypeError("manifest must be a ShardingManifest")
    return {
        "schema_version": manifest.schema_version,
        "shards": [_serialize_shard_identity(shard) for shard in manifest.shards],
    }


def _parse_sharding_manifest(record):
    record = _require_exact_record(record, _SHARDING_MANIFEST_KEYS, name="sharding manifest")
    if type(record["schema_version"]) is not int:
        raise ValueError("sharding manifest schema_version must be an int")
    shards = _require_exact_list(record["shards"], name="sharding manifest shards")
    try:
        return ShardingManifest(
            shards=tuple(_parse_shard_identity(shard) for shard in shards),
            schema_version=record["schema_version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("sharding manifest is invalid") from exc


def _normalize_sharding_manifest(record):
    return _serialize_sharding_manifest(_parse_sharding_manifest(record))
