"""CPU unit tests for the distributed-mode stable-ownership mapping (issue #45).

`sharded_mode="distributed"` fixes both #45 gaps by keying each matrix's owner
rank on its STABLE position in the full parameter set via
``_stable_distributed_owner`` -- so ownership (and thus a matrix's momentum home
rank) is invariant to which grads are present this step, and a checkpoint saved
at one world size re-derives ownership at another. That end-to-end behavior is
covered only by GPU + NCCL-gated FSDP2 tests, which never run in CPU CI. This
pins the pure ownership contract on CPU so a regression is caught without a GPU.
"""
import pytest

from gefen.gefen_muon import _stable_distributed_owner


def test_round_robin_mapping():
    # Owner is idx % world for every index/world.
    for world in (1, 2, 3, 4):
        for idx in range(7):
            assert _stable_distributed_owner(idx, world) == idx % world
    # Explicit round-robin spot checks.
    assert [_stable_distributed_owner(i, 2) for i in range(6)] == [0, 1, 0, 1, 0, 1]
    assert [_stable_distributed_owner(i, 3) for i in range(6)] == [0, 1, 2, 0, 1, 2]
    assert [_stable_distributed_owner(i, 4) for i in range(6)] == [0, 1, 2, 3, 0, 1]


def test_ownership_is_pure_and_world_stable():
    # Same (index, world) always yields the same owner: ownership does not depend
    # on which other grads are active this step (the gap-1 fix), and changing the
    # world re-derives ownership deterministically (the checkpoint-restore path).
    assert _stable_distributed_owner(5, 4) == _stable_distributed_owner(5, 4)
    assert _stable_distributed_owner(5, 2) == 1
    assert _stable_distributed_owner(5, 4) == 1


def test_rejects_nonpositive_world():
    with pytest.raises(ValueError, match="world must be positive"):
        _stable_distributed_owner(0, 0)
    with pytest.raises(ValueError, match="world must be positive"):
        _stable_distributed_owner(3, -2)


def test_rejects_negative_index():
    with pytest.raises(ValueError, match="stable_index must be non-negative"):
        _stable_distributed_owner(-1, 4)
