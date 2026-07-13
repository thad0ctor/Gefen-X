"""Strict composite envelope for Gefen-backed Hybrid portable state."""

import hashlib
import hmac

from gefen.portable_schema import (
    PORTABLE_STATE_DIGEST_ALGORITHM,
    normalize_portable_state_document,
    portable_state_digest,
)


HYBRID_PORTABLE_STATE_FORMAT = "gefen.portable_composite_state"
HYBRID_PORTABLE_STATE_FORMAT_VERSION = 1
HYBRID_PORTABLE_STATE_COVERAGE = "global_logical_composite_optimizer"
HYBRID_PORTABLE_STATE_IMPLEMENTATION = "gefen.GefenMuonHybrid"

_HYBRID_PORTABLE_ROLES = ("muon", "backup")
_HYBRID_PORTABLE_CHILD_IMPLEMENTATIONS = {
    "muon": "gefen.GefenMuon",
    "backup": "gefen.Gefen",
}
_HYBRID_PORTABLE_TOP_LEVEL_KEYS = frozenset(
    {
        "format",
        "format_version",
        "coverage",
        "implementation",
        "backup_optimizer",
        "routing",
        "children",
        "completion",
    }
)
_HYBRID_PORTABLE_COMPLETION_KEYS = frozenset({"status", "digest_algorithm", "digest"})
_HYBRID_EXPORT_PREFLIGHT_TRANSACTION = "gefen-hybrid-portable-export-preflight-v1"
_HYBRID_IMPORT_PREFLIGHT_TRANSACTION = "gefen-hybrid-portable-import-preflight-v1"


def _hybrid_portable_payload(document):
    return {key: document[key] for key in sorted(_HYBRID_PORTABLE_TOP_LEVEL_KEYS - {"completion"})}


def _normalize_routing(routing):
    if type(routing) is not dict:
        raise ValueError("portable Hybrid routing must be an FQN dictionary")
    if any(type(fqn) is not str for fqn in routing):
        raise ValueError("portable Hybrid routing keys must be strings")
    normalized = {}
    for fqn in sorted(routing):
        role = routing[fqn]
        if not fqn or fqn != fqn.strip():
            raise ValueError("portable Hybrid routing keys must be non-empty trimmed FQNs")
        if type(role) is not str or role not in _HYBRID_PORTABLE_ROLES:
            raise ValueError("portable Hybrid routing values must name exact child roles")
        normalized[fqn] = role
    return normalized


def _normalize_children(children):
    if type(children) is not dict or set(children) != set(_HYBRID_PORTABLE_ROLES):
        raise ValueError("portable Hybrid children must contain exact muon and backup roles")
    normalized = {}
    for role in _HYBRID_PORTABLE_ROLES:
        child = children[role]
        normalized[role] = (
            None
            if child is None
            else normalize_portable_state_document(
                child,
                expected_implementation=_HYBRID_PORTABLE_CHILD_IMPLEMENTATIONS[role],
            )
        )
    if all(child is None for child in normalized.values()):
        raise ValueError("portable Hybrid state must contain at least one child")
    return normalized


def _validate_child_routing(children, routing) -> None:
    expected_routing = {}
    common_step = None
    deterministic = None
    for role in _HYBRID_PORTABLE_ROLES:
        child = children[role]
        if child is None:
            continue
        for fqn in child["parameters"]:
            if fqn in expected_routing:
                raise ValueError("portable Hybrid child parameter FQNs must be disjoint")
            expected_routing[fqn] = role
        child_common = child["common"]
        if type(child_common) is not dict:
            raise ValueError("portable Hybrid child common state must be a dictionary")
        child_step = child_common.get("gefen_global_step")
        child_deterministic = child_common.get("gefen_deterministic")
        if type(child_step) is not int or child_step < 0:
            raise ValueError("portable Hybrid children require nonnegative exact global steps")
        if type(child_deterministic) is not bool:
            raise ValueError("portable Hybrid children require exact deterministic policies")
        if common_step is None:
            common_step = child_step
            deterministic = child_deterministic
        elif child_step != common_step:
            raise ValueError("portable Hybrid child global steps must agree")
        elif child_deterministic is not deterministic:
            raise ValueError("portable Hybrid child deterministic policies must agree")
    if routing != {fqn: expected_routing[fqn] for fqn in sorted(expected_routing)}:
        raise ValueError("portable Hybrid routing does not exactly match child parameters")


def _normalize_hybrid_portable_payload(state):
    if state["format"] != HYBRID_PORTABLE_STATE_FORMAT:
        raise ValueError("unsupported portable Hybrid state format")
    if type(state["format_version"]) is not int or state["format_version"] != HYBRID_PORTABLE_STATE_FORMAT_VERSION:
        raise ValueError("unsupported portable Hybrid state format_version: {}".format(state["format_version"]))
    if state["coverage"] != HYBRID_PORTABLE_STATE_COVERAGE:
        raise ValueError("unsupported portable Hybrid state coverage")
    if state["implementation"] != HYBRID_PORTABLE_STATE_IMPLEMENTATION:
        raise ValueError("portable Hybrid state implementation does not match the target")
    if state["backup_optimizer"] != "gefen":
        raise ValueError("portable Hybrid state requires a Gefen backup policy")
    routing = _normalize_routing(state["routing"])
    children = _normalize_children(state["children"])
    _validate_child_routing(children, routing)
    return {
        "format": HYBRID_PORTABLE_STATE_FORMAT,
        "format_version": HYBRID_PORTABLE_STATE_FORMAT_VERSION,
        "coverage": HYBRID_PORTABLE_STATE_COVERAGE,
        "implementation": HYBRID_PORTABLE_STATE_IMPLEMENTATION,
        "backup_optimizer": "gefen",
        "routing": routing,
        "children": children,
    }


def _validate_hybrid_portable_completion(state, payload) -> None:
    completion = state["completion"]
    if type(completion) is not dict or set(completion) != _HYBRID_PORTABLE_COMPLETION_KEYS:
        raise ValueError("portable Hybrid completion marker has an invalid schema")
    if completion["status"] != "complete":
        raise ValueError("portable Hybrid state is not marked complete")
    if completion["digest_algorithm"] != PORTABLE_STATE_DIGEST_ALGORITHM:
        raise ValueError("unsupported portable Hybrid state digest algorithm")
    digest = completion["digest"]
    if (
        type(digest) is not str
        or len(digest) != hashlib.sha256().digest_size * 2
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("portable Hybrid completion digest is invalid")
    if not hmac.compare_digest(digest, portable_state_digest(payload)):
        raise ValueError("portable Hybrid completion digest does not match its payload")


def build_hybrid_portable_state_document(*, backup_optimizer, routing, children):
    """Build one complete composite wrapper around unchanged portable v3 children."""

    state = {
        "format": HYBRID_PORTABLE_STATE_FORMAT,
        "format_version": HYBRID_PORTABLE_STATE_FORMAT_VERSION,
        "coverage": HYBRID_PORTABLE_STATE_COVERAGE,
        "implementation": HYBRID_PORTABLE_STATE_IMPLEMENTATION,
        "backup_optimizer": backup_optimizer,
        "routing": routing,
        "children": children,
    }
    payload = _normalize_hybrid_portable_payload(state)
    payload["completion"] = {
        "status": "complete",
        "digest_algorithm": PORTABLE_STATE_DIGEST_ALGORITHM,
        "digest": portable_state_digest(payload),
    }
    return payload


def normalize_hybrid_portable_state_document(state):
    """Clone and strictly validate one complete portable Hybrid wrapper."""

    if type(state) is not dict or set(state) != _HYBRID_PORTABLE_TOP_LEVEL_KEYS:
        raise ValueError("portable Hybrid state has an invalid top-level schema")
    payload = _normalize_hybrid_portable_payload(state)
    _validate_hybrid_portable_completion(state, payload)
    payload["completion"] = dict(state["completion"])
    return payload


def _validate_hybrid_portable_limits(state, limits) -> None:
    from gefen.portable_wire import _prepare_canonical_wire_value

    _prepare_canonical_wire_value(state, limits._wire_limits(collective=True))


def _hybrid_children(optimizer):
    from gefen.gefen import Gefen
    from gefen.gefen_muon import GefenMuon
    from gefen.hybrid import GefenMuonHybrid

    if type(optimizer) is not GefenMuonHybrid:
        raise TypeError("portable Hybrid state requires an exact GefenMuonHybrid")
    if optimizer.backup_optimizer != "gefen":
        raise NotImplementedError("portable Hybrid state does not support an AdamW backup")
    children = []
    if optimizer.muon is not None:
        if type(optimizer.muon) is not GefenMuon:
            raise TypeError("portable Hybrid state requires an exact GefenMuon child")
        children.append(("muon", optimizer.muon))
    if optimizer.backup is not None:
        if type(optimizer.backup) is not Gefen:
            raise TypeError("portable Hybrid state requires an exact Gefen backup child")
        children.append(("backup", optimizer.backup))
    if not children:
        raise RuntimeError("portable Hybrid state requires at least one child")
    return tuple(children)


def _hybrid_transport_binding(optimizer, supplied):
    from gefen import portable_runtime as runtime

    # The caller-provided exact binding is the only safe first-status
    # transport: child presence/type may differ across faulty ranks and must be
    # voted before any rank tries to enter a child collective.
    del optimizer
    return runtime._require_binding(supplied)


def _hybrid_portable_live_token(optimizer):
    from gefen import portable_runtime as runtime

    children = _hybrid_children(optimizer)
    return (
        optimizer.backup_optimizer,
        optimizer._deterministic,
        optimizer._hybrid_post_sharding_finalized,
        id(optimizer._hybrid_sharding_manifest),
        id(optimizer._hybrid_codebook_process_group),
        id(optimizer._state_param_owner),
        runtime._portable_value_token(optimizer._hybrid_fqn_roles),
        runtime._portable_value_token(optimizer._hybrid_finalized_slots),
        tuple(
            (
                None if parameter is None else id(parameter),
                shard.sort_key,
            )
            for parameter, shard in optimizer._hybrid_local_shard_bindings
        ),
        tuple((role, id(child), runtime._portable_live_token(child)) for role, child in children),
    )


def _validate_hybrid_portable_readiness(optimizer, binding):
    from gefen import portable_runtime as runtime
    from gefen.checkpoint import CheckpointProcessGroupBinding
    from gefen.hybrid import GefenMuonHybrid

    if type(binding) is not CheckpointProcessGroupBinding:
        raise TypeError("portable Hybrid readiness requires an exact checkpoint binding")
    GefenMuonHybrid._reject_rebinding_method_shadows(optimizer)
    children = _hybrid_children(optimizer)
    if not optimizer._canonical_identity_ready():
        raise RuntimeError("portable Hybrid state requires a finalized composite binding")
    if optimizer._hybrid_codebook_process_group is None:
        raise RuntimeError("portable Hybrid state requires one shared codebook scope")
    routing = dict(optimizer._hybrid_fqn_roles)
    if (
        type(optimizer._hybrid_fqn_roles) is not tuple
        or len(routing) != len(optimizer._hybrid_fqn_roles)
        or tuple(sorted(routing.items())) != optimizer._hybrid_fqn_roles
    ):
        raise RuntimeError("portable Hybrid routing metadata is not canonical")
    expected_routing = {}
    layouts = {}
    common_step = None
    deterministic = None
    for role, child in children:
        if child._gefen_codebook_process_group is not optimizer._hybrid_codebook_process_group:
            raise RuntimeError("portable Hybrid children must share one exact codebook binding")
        child_binding = runtime._preflight_transport_binding(child, binding)
        runtime._validate_supplied_binding(binding, child_binding)
        layouts[role] = runtime._validate_live_readiness(
            child,
            runtime._optimizer_implementation(child),
            binding,
        )
        for slot in child._gefen_logical_slots:
            fqn = slot.shard.parameter.fqn
            if fqn in expected_routing:
                raise RuntimeError("portable Hybrid child FQNs must be disjoint")
            expected_routing[fqn] = role
        child_step = child._canonical_common_global_step()
        child_deterministic = child._deterministic
        if common_step is None:
            common_step = child_step
            deterministic = child_deterministic
        elif child_step != common_step:
            raise RuntimeError("portable Hybrid child global steps must agree")
        elif child_deterministic is not deterministic:
            raise RuntimeError("portable Hybrid child deterministic policies must agree")
    if routing != {fqn: expected_routing[fqn] for fqn in sorted(expected_routing)}:
        raise RuntimeError("portable Hybrid routing does not match its child slots")
    if optimizer._deterministic is not deterministic:
        raise RuntimeError("portable Hybrid deterministic policy disagrees with its children")
    return children, routing, layouts


def _hybrid_child_transaction_id(parent: str, role: str, operation: str) -> str:
    if role not in _HYBRID_PORTABLE_ROLES or operation not in {"export", "import"}:
        raise ValueError("invalid portable Hybrid child transaction domain")
    digest = hashlib.sha256(
        b"gefen.portable_hybrid.child.v1\0"
        + operation.encode("ascii")
        + b"\0"
        + role.encode("ascii")
        + b"\0"
        + parent.encode("utf-8")
    ).hexdigest()
    return "hybrid-{}-{}-{}".format(operation, role, digest)


def _preflight_hybrid_portable_local(
    optimizer,
    *,
    checkpoint_process_group,
    transaction_id,
    limits,
):
    from gefen import portable_runtime as runtime

    binding = _hybrid_transport_binding(optimizer, checkpoint_process_group)
    runtime._validate_supplied_binding(checkpoint_process_group, binding)
    normalized_limits = runtime._require_limits(limits)
    normalized_transaction = runtime._require_transaction_id(transaction_id)
    runtime._validate_context_identity_bounds(binding, normalized_limits)
    children, routing, layouts = _validate_hybrid_portable_readiness(
        optimizer,
        binding,
    )
    context = {
        **runtime._base_context(binding, HYBRID_PORTABLE_STATE_IMPLEMENTATION),
        "composite_format": HYBRID_PORTABLE_STATE_FORMAT,
        "composite_format_version": HYBRID_PORTABLE_STATE_FORMAT_VERSION,
        "backup_optimizer": optimizer.backup_optimizer,
        "roles": tuple(role for role, _child in children),
        "routing": routing,
        "layouts": {
            role: tuple(sorted(layout.value for layout in role_layouts)) for role, role_layouts in layouts.items()
        },
        "transaction_id": normalized_transaction,
    }
    runtime._preflight_portable_value(context, normalized_limits)
    return (
        binding,
        normalized_transaction,
        normalized_limits,
        children,
        routing,
        context,
        runtime._context_digest(context),
        _hybrid_portable_live_token(optimizer),
    )


def _hybrid_portable_contract_support(optimizer):
    """Return dynamic composite CANONICAL_GLOBAL support without collectives."""

    try:
        from gefen.checkpoint import CheckpointProcessGroupBinding
        from gefen.codebook import CodebookProcessGroupBinding
        from gefen.contracts import CheckpointTransport

        scope = optimizer._hybrid_codebook_process_group
        if type(scope) is not CodebookProcessGroupBinding:
            return frozenset(), frozenset(), frozenset()
        binding = CheckpointProcessGroupBinding(
            scope.identity,
            scope.local_member,
            scope.process_group,
            scope.collective_device,
        )
        children, _routing, _layouts = _validate_hybrid_portable_readiness(
            optimizer,
            binding,
        )
        # A composite canonical-global save/load processes EVERY child, so the
        # Hybrid may only advertise a checkpoint guarantee that every child
        # supports: intersect (not union) the children's guarantee sets. (This
        # is the opposite of the per-routed-parameter TRAINING claims, which are
        # legitimately unioned.) None marks "no child seen yet" so the first
        # child seeds each set and later children narrow it.
        same_topology = None
        topology_changing = None
        topology_change_kinds = None
        for _role, child in children:
            supports = tuple(
                support
                for support in child.optimizer_contract().capabilities.checkpoints
                if support.transport is CheckpointTransport.CANONICAL_GLOBAL
            )
            if len(supports) != 1:
                return frozenset(), frozenset(), frozenset()
            support = supports[0]
            child_same = set(support.same_topology)
            child_changing = set(support.topology_changing)
            child_kinds = set(support.topology_change_kinds)
            same_topology = (
                child_same if same_topology is None else same_topology & child_same
            )
            topology_changing = (
                child_changing
                if topology_changing is None
                else topology_changing & child_changing
            )
            topology_change_kinds = (
                child_kinds
                if topology_change_kinds is None
                else topology_change_kinds & child_kinds
            )
        return (
            frozenset(same_topology or ()),
            frozenset(topology_changing or ()),
            frozenset(topology_change_kinds or ()),
        )
    except Exception:
        return frozenset(), frozenset(), frozenset()


def _export_hybrid_portable_state(
    optimizer,
    *,
    checkpoint_process_group,
    transaction_id,
    limits,
):
    from gefen import portable_runtime as runtime
    from gefen.portable_collective import _collective_unanimous_status

    transport = _hybrid_transport_binding(optimizer, checkpoint_process_group)
    wire_limits = runtime._STATUS_FALLBACK_LIMITS
    binding = None
    normalized_transaction = None
    normalized_limits = None
    children = None
    routing = None
    context_digest = bytes(32)
    live_token = None
    error = None
    try:
        (
            binding,
            normalized_transaction,
            normalized_limits,
            children,
            routing,
            _context,
            context_digest,
            live_token,
        ) = _preflight_hybrid_portable_local(
            optimizer,
            checkpoint_process_group=checkpoint_process_group,
            transaction_id=transaction_id,
            limits=limits,
        )
        wire_limits = normalized_limits._wire_limits()
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        transport,
        error,
        operation="hybrid_portable_export_preflight",
        transaction_id=_HYBRID_EXPORT_PREFLIGHT_TRANSACTION,
        context_digest=context_digest,
        limits=wire_limits,
    )
    assert (
        binding is not None
        and normalized_transaction is not None
        and normalized_limits is not None
        and children is not None
        and routing is not None
        and live_token is not None
    )

    child_documents = {"muon": None, "backup": None}
    for role, child in children:
        child_documents[role] = child.export_portable_state(
            checkpoint_process_group=binding,
            transaction_id=_hybrid_child_transaction_id(
                normalized_transaction,
                role,
                "export",
            ),
            limits=normalized_limits,
        )

    document = None
    error = None
    try:
        document = build_hybrid_portable_state_document(
            backup_optimizer=optimizer.backup_optimizer,
            routing=routing,
            children=child_documents,
        )
        _validate_hybrid_portable_limits(document, normalized_limits)
        _validate_hybrid_portable_readiness(optimizer, binding)
        if live_token != _hybrid_portable_live_token(optimizer):
            raise RuntimeError("live Hybrid state changed during portable export")
        context_digest = bytes.fromhex(document["completion"]["digest"])
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="hybrid_portable_export_finalize",
        transaction_id=normalized_transaction,
        context_digest=context_digest,
        limits=wire_limits,
    )
    assert document is not None
    return document


def _import_hybrid_portable_state(
    optimizer,
    state,
    *,
    checkpoint_process_group,
    transaction_id,
    limits,
) -> None:
    from gefen import portable_runtime as runtime
    from gefen.portable_collective import _collective_unanimous_status

    transport = _hybrid_transport_binding(optimizer, checkpoint_process_group)
    wire_limits = runtime._STATUS_FALLBACK_LIMITS
    binding = None
    normalized_transaction = None
    normalized_limits = None
    children = None
    routing = None
    base_context = None
    context_digest = bytes(32)
    composite_live_token = None
    document = None
    error = None
    try:
        (
            binding,
            normalized_transaction,
            normalized_limits,
            children,
            routing,
            base_context,
            context_digest,
            composite_live_token,
        ) = _preflight_hybrid_portable_local(
            optimizer,
            checkpoint_process_group=checkpoint_process_group,
            transaction_id=transaction_id,
            limits=limits,
        )
        wire_limits = normalized_limits._wire_limits()
        from gefen.portable_state import (
            _bounded_clone,
            _normalize_gefen_portable_state_document,
        )

        document = normalize_hybrid_portable_state_document(_bounded_clone(state, normalized_limits, collective=True))
        for role in _HYBRID_PORTABLE_ROLES:
            child_document = document["children"][role]
            if child_document is not None:
                document["children"][role] = _normalize_gefen_portable_state_document(
                    child_document,
                    limits=normalized_limits,
                    expected_implementation=_HYBRID_PORTABLE_CHILD_IMPLEMENTATIONS[role],
                )
        if document["routing"] != routing:
            raise ValueError("portable Hybrid document routing does not match the target")
        expected_presence = {
            role: child is not None for role, child in (("muon", optimizer.muon), ("backup", optimizer.backup))
        }
        if any(
            (document["children"][role] is not None) is not expected_presence[role] for role in _HYBRID_PORTABLE_ROLES
        ):
            raise ValueError("portable Hybrid child presence does not match the target")
        context_digest = bytes.fromhex(document["completion"]["digest"])
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        transport,
        error,
        operation="hybrid_portable_import_document",
        transaction_id=_HYBRID_IMPORT_PREFLIGHT_TRANSACTION,
        context_digest=context_digest,
        limits=wire_limits,
    )
    assert (
        binding is not None
        and normalized_transaction is not None
        and normalized_limits is not None
        and children is not None
        and routing is not None
        and base_context is not None
        and composite_live_token is not None
        and document is not None
    )

    staged_children = []
    target_children = {}
    error = None
    try:
        for role, child in children:
            implementation = runtime._optimizer_implementation(child)
            staged, live_token, target_fragment = runtime._stage_portable_import(
                child,
                implementation,
                binding,
                normalized_limits,
                document["children"][role],
            )
            staged_children.append((role, child, implementation, staged, live_token))
            target_children[role] = runtime._target_context(
                runtime._base_context(binding, implementation),
                target_fragment,
            )
        target_context = {
            **base_context,
            "document_digest": document["completion"]["digest"],
            "target_children": target_children,
        }
        runtime._preflight_portable_value(target_context, normalized_limits)
        context_digest = runtime._context_digest(target_context)
    except Exception as exc:
        error = exc
        context_digest = runtime._context_digest(
            {
                **base_context,
                "document_digest": document["completion"]["digest"],
            }
        )
    _collective_unanimous_status(
        binding,
        error,
        operation="hybrid_portable_import_prepare",
        transaction_id=normalized_transaction,
        context_digest=context_digest,
        limits=wire_limits,
    )
    assert len(staged_children) == len(children)

    target_deterministic = next(
        document["children"][role]["common"]["gefen_deterministic"] for role, _child in children
    )
    from gefen.gefen import Gefen

    commit_staged = Gefen._commit_staged_load_state_dict

    error = None
    try:
        _validate_hybrid_portable_readiness(optimizer, binding)
        if composite_live_token != _hybrid_portable_live_token(optimizer):
            raise RuntimeError("live Hybrid state changed after portable import preparation")
        for _role, child, _implementation, _staged, live_token in staged_children:
            if live_token != runtime._portable_live_token(child):
                raise RuntimeError("live Hybrid child changed after portable import preparation")
    except Exception as exc:
        error = exc
    _collective_unanimous_status(
        binding,
        error,
        operation="hybrid_portable_import_freshness",
        transaction_id=normalized_transaction,
        context_digest=context_digest,
        limits=wire_limits,
    )

    for _role, child, _implementation, staged, _live_token in staged_children:
        commit_staged(child, staged)
    dict.__setitem__(
        optimizer.__dict__,
        "_deterministic",
        target_deterministic,
    )


__all__ = [
    "HYBRID_PORTABLE_STATE_COVERAGE",
    "HYBRID_PORTABLE_STATE_FORMAT",
    "HYBRID_PORTABLE_STATE_FORMAT_VERSION",
    "HYBRID_PORTABLE_STATE_IMPLEMENTATION",
    "build_hybrid_portable_state_document",
    "normalize_hybrid_portable_state_document",
]
