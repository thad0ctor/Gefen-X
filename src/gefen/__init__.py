from importlib.metadata import PackageNotFoundError, version

try:
    # Single-sourced from the installed distribution metadata (name: gefen-x).
    __version__ = version("gefen-x")
except PackageNotFoundError:
    # Importing straight from a source checkout that was never `pip install`ed.
    __version__ = "0.0.0+unknown"

__all__ = [
    "Gefen",
    "GefenMuon",
    "GefenMuonHybrid",
    "CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "CANONICAL_STATE_FORMAT_VERSION",
    "PORTABLE_STATE_FORMAT_VERSION",
    "IDENTITY_SCHEMA_VERSION",
    "CheckpointSupport",
    "CheckpointTransport",
    "CheckpointProjectionQualifier",
    "CheckpointStateRepresentation",
    "CheckpointStateTransition",
    "CheckpointStateTransitionKind",
    "CanonicalStateProvider",
    "CheckpointProcessGroupBinding",
    "CodebookProcessGroupBinding",
    "PreparedCanonicalStateImport",
    "PortableStateLimits",
    "PortableStateProvider",
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
    "ParameterRebinding",
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
    "TopologyChange",
    "TrainingSupport",
    "build_portable_state_document",
    "normalize_portable_state_document",
    "portable_state_digest",
    "load_portable_dcp",
    "save_portable_dcp",
    "split_params_for_muon",
    "validate_split",
    "kernels",
    "__version__",
]


def __getattr__(name):
    if name == "Gefen":
        from .gefen import Gefen

        return Gefen
    if name == "GefenMuon":
        from .gefen_muon import GefenMuon

        return GefenMuon
    if name == "GefenMuonHybrid":
        from .hybrid import GefenMuonHybrid

        return GefenMuonHybrid
    if name in ("split_params_for_muon", "validate_split"):
        from . import params

        return getattr(params, name)
    if name == "ParameterRebinding":
        from .rebinding import ParameterRebinding

        return ParameterRebinding
    if name == "CodebookProcessGroupBinding":
        from .codebook import CodebookProcessGroupBinding

        return CodebookProcessGroupBinding
    if name == "CheckpointProcessGroupBinding":
        from .checkpoint import CheckpointProcessGroupBinding

        return CheckpointProcessGroupBinding
    if name in (
        "CANONICAL_STATE_FORMAT_VERSION",
        "PreparedCanonicalStateImport",
    ):
        from . import canonical

        return getattr(canonical, name)
    if name in (
        "PORTABLE_STATE_FORMAT_VERSION",
        "build_portable_state_document",
        "normalize_portable_state_document",
        "portable_state_digest",
    ):
        from . import portable_schema

        return getattr(portable_schema, name)
    if name == "PortableStateLimits":
        from .portable_state import PortableStateLimits

        return PortableStateLimits
    if name in ("load_portable_dcp", "save_portable_dcp"):
        from . import portable_dcp

        return getattr(portable_dcp, name)
    if name in (
        "CHECKPOINT_STATE_TRANSITION_SCHEMA_VERSION",
        "CONTRACT_SCHEMA_VERSION",
        "IDENTITY_SCHEMA_VERSION",
        "CheckpointSupport",
        "CheckpointTransport",
        "CheckpointProjectionQualifier",
        "CheckpointStateRepresentation",
        "CheckpointStateTransition",
        "CheckpointStateTransitionKind",
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
        "PlacementKind",
        "PortableStateProvider",
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
        "TopologyChange",
        "TrainingSupport",
    ):
        from . import contracts

        return getattr(contracts, name)
    if name == "kernels":
        # NOT `from . import kernels`: its fromlist handling re-enters this
        # __getattr__ before the submodule import runs, recursing forever.
        import importlib

        return importlib.import_module(".kernels", __name__)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
