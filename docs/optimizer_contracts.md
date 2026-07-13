# Optimizer integration contracts

Gefen exposes immutable, versioned optimizer contracts for platform adapters that need to inspect state ownership and validated distributed capabilities without depending on private attributes. Calling `optimizer.optimizer_contract()` is read-only and does not change parameters, optimizer state, checkpoint schemas, or step behavior.

```python
from gefen import CheckpointTransport, Gefen, ParameterLayout

optimizer = Gefen(model.named_parameters(), lr=3e-5)
contract = optimizer.optimizer_contract()

assert contract.schema_version == 1
rank_local_dcp = next(
    support
    for support in contract.capabilities.checkpoints
    if support.transport is CheckpointTransport.PYTORCH_RANK_LOCAL
)
assert ParameterLayout.DTENSOR_1D_DEFAULT_WORLD in rank_local_dcp.same_topology
```

## Contract boundaries

- `OptimizerStateLayout` separates optimizer-common authoritative state, per-parameter authoritative state, derived caches, checkpoint transport fields, and composite child namespaces.
- `StateVariant` identifies valid lazy, initialized, local-shard, global-parameter, owner, non-owner, and migrated state combinations using structured layout, mode, rank, extent, ownership, and inactive-field declarations.
- `TrainingSupport` qualifies each validated parameter layout by process-group source, mesh dimensionality, sharded mode, and whether the update needs complete parameter storage or a transient complete logical matrix.
- `CheckpointSupport` reports same-topology, topology-changing, and fail-before-mutation load support separately for native, PyTorch rank-local, and composite checkpoint transports.
- Precision, canonical parameter identity, stable shard identity, explicit process-group-scoped codebooks, shard rebinding, post-sharding, canonical state I/O, state movement, and offload are independent capability fields. A false field is an explicit unsupported contract, not an invitation for an adapter to infer support from internal state.

The current DTensor declaration is deliberately narrow: `DTENSOR_1D_DEFAULT_WORLD` means one shared one-dimensional mesh spanning the default world. Multidimensional meshes, subgroups, and placement-changing loads are not implied by that declaration.

Plain Gefen declares replicated, flattened element-shard, and the narrow DTensor training layouts. Its PyTorch rank-local checkpoint transport is same-topology only. GefenMuon declares replicated and narrow DTensor training, with mode-specific state extents: `approx` state is local, `exact` state is logically global, and `distributed` momentum is held by the parameter owner while non-owners retain metadata only. Native Parallel-Muon checkpoints separately declare world-size owner redistribution, not placement-changing resharding. `GefenMuonHybrid` retains its nested child namespaces and does not flatten AdamW or Gefen child state into a fabricated common schema.

## Adapter requirements

An adapter should match the exact `TrainingSupport` or `CheckpointSupport` entry it intends to use, including transport, layout, process-group scope, mesh dimensions, and sharded mode. It should not treat successful training as checkpoint support, same-topology checkpointing as resharding support, or accepted caller names as canonical fully qualified parameter identity.

The contract types do not import a distributed platform and do not perform collectives. Platform adapters remain responsible for topology discovery, deterministic scheduling, and lifecycle orchestration; future core mutation APIs can consume these declarations without changing their meaning.
