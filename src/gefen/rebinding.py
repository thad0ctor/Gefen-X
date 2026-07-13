"""Runtime requests for atomic parameter/shard rebinding."""

from dataclasses import dataclass
from typing import Optional

from gefen.contracts import ShardIdentity


@dataclass(frozen=True, eq=False)
class ParameterRebinding:
    """Bind one optimizer slot to a local tensor or prune it as a non-owner."""

    old_parameter: object
    new_parameter: Optional[object]
    shard: ShardIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.shard, ShardIdentity):
            raise TypeError("ParameterRebinding.shard must be a ShardIdentity")


__all__ = ["ParameterRebinding"]
