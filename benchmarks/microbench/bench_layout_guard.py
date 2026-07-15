"""Benchmark the per-step layout-forensics guard cost on a synthetic manifest.

Builds a finalized plain Gefen over a large synthetic flattened-shard
``ShardingManifest`` (defaults: 512 process-group members x 300 parameters =
153,600 global ``ShardIdentity`` records) with no real process group, then
compares:

  * old per-step guard cost — the pre-fix step() sequence re-ran the complete
    O(params x world) forensic rebuild on every guard call (2 passes per
    unscoped step, up to 7 under an explicit multi-member codebook scope) and
    recomputed the manifest sha256 fingerprint inside every scoped operation
    header (2 per step);
  * new per-step guard cost — the exact warm step() guard sequence, which
    reuses one cached forensic verdict through O(local params) identity
    tokens and the manifest digest computed once at post_sharding
    finalization.

The headline is the scoped worst case (7 forensic passes + 2 digest
computes); the unscoped floor (2 passes, no digests) is printed alongside.
Exits nonzero if the scoped per-step reduction is below the required 100x.

Run from the repo root (CPU only, no distributed init needed):

    PYTHONPATH=src python benchmarks/microbench/bench_layout_guard.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from gefen import (
    Gefen,
    LogicalSlice,
    ParameterIdentity,
    ParameterLayout,
    ParameterRebinding,
    PlacementKind,
    ProcessGroupIdentity,
    ShardIdentity,
    ShardPlacement,
    ShardingManifest,
)


def build_finalized_optimizer(members_count: int, params_count: int, local_length: int):
    members = tuple("member{:04d}".format(index) for index in range(members_count))
    group = ProcessGroupIdentity("data_parallel", members)
    local_member = members[0]
    manifest_shards = []
    local_shards = []
    for index in range(params_count):
        identity = ParameterIdentity(
            "Model.Block{}.Weight".format(index), (members_count * local_length,)
        )
        offset = 0
        for coordinate, member in enumerate(members):
            shard = ShardIdentity(
                identity,
                ParameterLayout.FLATTENED_ELEMENT_SHARD,
                LogicalSlice(offset, local_length),
                placements=(
                    ShardPlacement(
                        "data_parallel",
                        PlacementKind.FLAT_SHARD,
                        coordinate,
                        members_count,
                    ),
                ),
                process_group=group,
                local_member=member,
            )
            manifest_shards.append(shard)
            if member == local_member:
                local_shards.append(shard)
            offset += local_length

    start = time.perf_counter()
    manifest = ShardingManifest(tuple(manifest_shards))
    manifest_seconds = time.perf_counter() - start

    parameters = [
        torch.nn.Parameter(torch.randn(local_length)) for _ in range(params_count)
    ]
    optimizer = Gefen(
        [
            ("model.block{}.weight".format(index), parameters[index])
            for index in range(params_count)
        ],
        fused=False,
        factored_v_2d=False,
    )
    start = time.perf_counter()
    optimizer.post_sharding(
        tuple(
            ParameterRebinding(parameters[index], parameters[index], local_shards[index])
            for index in range(params_count)
        ),
        manifest=manifest,
    )
    finalize_seconds = time.perf_counter() - start
    return optimizer, parameters, manifest, manifest_seconds, finalize_seconds


def timed(callable_, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        callable_()
    return (time.perf_counter() - start) / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=int, default=512)
    parser.add_argument("--params", type=int, default=300)
    parser.add_argument("--local-length", type=int, default=4)
    parser.add_argument("--full-repeats", type=int, default=3)
    parser.add_argument("--fast-repeats", type=int, default=200)
    args = parser.parse_args()

    optimizer, parameters, manifest, manifest_seconds, finalize_seconds = (
        build_finalized_optimizer(args.members, args.params, args.local_length)
    )
    print(
        "manifest: {} shards ({} members x {} params), built in {:.2f}s; "
        "post_sharding finalized in {:.2f}s".format(
            len(manifest.shards),
            args.members,
            args.params,
            manifest_seconds,
            finalize_seconds,
        )
    )

    # Warm the runtime and the forensic verdict exactly the way training does.
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 0.5)
    optimizer.step()

    full_pass = timed(
        lambda: optimizer._finalized_binding_layout_matches(full=True),
        args.full_repeats,
    )
    digest_compute = timed(
        lambda: optimizer._compute_codebook_manifest_fingerprint(manifest),
        args.full_repeats,
    )
    cached_digest = timed(
        optimizer._codebook_manifest_fingerprint, args.fast_repeats
    )

    def warm_step_guards():
        # The finalized-layout/process-group step guard sequence (both the
        # pre-closure and post-closure blocks), on a warm verdict.
        optimizer._assert_finalized_binding_layout()
        optimizer._assert_runtime_codebook_process_group()
        optimizer._assert_finalized_binding_layout()
        optimizer._assert_runtime_codebook_process_group()

    warm_guards = timed(warm_step_guards, args.fast_repeats)

    # Pre-fix per-step guard cost. Unscoped step(): 2 complete forensic
    # passes. Scoped step(): up to 7 complete passes (step entry/re-entry,
    # scope asserts, operation headers, failure synchronization, scope
    # agreement) plus 2 manifest fingerprint recomputes in the exchanged
    # "step" and "periodic_step" headers.
    old_unscoped = 2 * full_pass
    old_scoped = 7 * full_pass + 2 * digest_compute
    new_unscoped = warm_guards
    new_scoped = warm_guards + 2 * cached_digest

    print("one full forensic pass:        {:>12.6f}s".format(full_pass))
    print("one manifest digest compute:   {:>12.6f}s".format(digest_compute))
    print("one cached digest fetch:       {:>12.6f}s".format(cached_digest))
    print("warm step guard sequence:      {:>12.6f}s".format(warm_guards))
    print(
        "old per-step guards (unscoped): {:>11.6f}s -> new: {:.6f}s ({:.0f}x)".format(
            old_unscoped, new_unscoped, old_unscoped / new_unscoped
        )
    )
    scoped_ratio = old_scoped / new_scoped
    print(
        "old per-step guards (scoped):   {:>11.6f}s -> new: {:.6f}s ({:.0f}x)".format(
            old_scoped, new_scoped, scoped_ratio
        )
    )
    if scoped_ratio < 100.0:
        print("FAIL: scoped per-step guard reduction is below 100x")
        return 1
    print("PASS: scoped per-step guard reduction is >= 100x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
