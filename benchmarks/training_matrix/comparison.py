"""Canonical comparison fingerprints for matrix result rows."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def hash_context(context: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(context).encode()).hexdigest()


SOURCE_FINGERPRINT_ENV = "GEFEN_MATRIX_SOURCE_FINGERPRINT"


def source_fingerprint(root: str | Path) -> dict[str, Any]:
    """Hash commit + tracked diff + untracked non-ignored source content."""

    root = Path(root)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            ".",
            ":(exclude)benchmarks/training_matrix/out/**",
        ],
        cwd=root,
        stderr=subprocess.DEVNULL,
    )
    untracked_raw = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=root,
        stderr=subprocess.DEVNULL,
    )
    untracked = sorted(path for path in untracked_raw.decode().split("\0") if path)
    digest = hashlib.sha256()
    digest.update(diff)
    for relative in untracked:
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "commit": commit,
        "diff_sha256": digest.hexdigest(),
        "dirty": bool(diff or untracked),
    }


def immutable_source_fingerprint(root: str | Path) -> dict[str, Any]:
    """Use a launcher-captured fingerprint, or capture once for a direct run."""

    encoded = os.environ.get(SOURCE_FINGERPRINT_ENV)
    if encoded:
        fingerprint = json.loads(encoded)
        if not isinstance(fingerprint, dict):
            raise ValueError(f"{SOURCE_FINGERPRINT_ENV} must encode a JSON object")
        return fingerprint
    return source_fingerprint(root)


def require_unchanged_source(
    root: str | Path, captured: dict[str, Any]
) -> None:
    """Abort a sequential matrix before it spans two source revisions."""

    current = source_fingerprint(root)
    if current != captured:
        raise RuntimeError(
            "source tree changed after the matrix fingerprint was captured; "
            "refusing to launch the next cell\n"
            f"captured={canonical_json(captured)}\n"
            f"current={canonical_json(current)}"
        )


def attach_comparison(result: dict[str, Any], context: dict[str, Any]) -> None:
    """Attach the auditable context and its content hash in place."""

    result["comparison_context"] = context
    result["comparison_id"] = hash_context(context)


def strict_fallback_context(row: dict[str, Any]) -> dict[str, Any]:
    """Conservative fingerprint for legacy/manual rows without an explicit ID."""

    optimizer = row.get("optimizer", {})
    runtime = row.get("runtime", {})
    return {
        "format": row.get("format"),
        "phase": row.get("phase"),
        "seed": row.get("seed"),
        "model": row.get("model"),
        "data": row.get("data"),
        "schedule": row.get("schedule"),
        "training_batch": row.get("training_batch"),
        "measurement_policy": row.get("measurement_policy"),
        "shared_optimizer_knobs": {
            key: optimizer.get(key)
            for key in (
                "lr",
                "weight_decay",
                "betas",
                "eps",
                "muon_eps",
                "backup_eps",
                "momentum",
                "nesterov",
                "fused",
                "batched_ns_requested",
                "batched_ns_workspace_bytes_requested",
            )
        },
        "initialization": row.get("initialization"),
        "runtime": {
            "device_name": runtime.get("device_name"),
            "cuda_visible_devices": runtime.get("cuda_visible_devices"),
            "torch_version": runtime.get("torch_version"),
            "torch_cuda_version": runtime.get("torch_cuda_version"),
            "python_version": runtime.get("python_version"),
            "transformers_version": runtime.get("transformers_version"),
            "datasets_version": runtime.get("datasets_version"),
            "nvidia_driver_version": runtime.get("nvidia_driver_version"),
            "deterministic_kernels": runtime.get("deterministic_kernels"),
            "git": runtime.get("git"),
        },
    }


def comparison_key(row: dict[str, Any]) -> tuple[str, str]:
    explicit_id = row.get("comparison_id")
    context = row.get("comparison_context")
    if explicit_id is not None:
        if not isinstance(context, dict):
            raise ValueError("result has comparison_id but no comparison_context")
        actual = hash_context(context)
        if actual != explicit_id:
            raise ValueError(
                f"result comparison_id is stale/tampered: recorded {explicit_id}, derived {actual}"
            )
        return "explicit", explicit_id
    return "fallback", hash_context(strict_fallback_context(row))
