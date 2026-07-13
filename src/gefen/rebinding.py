"""Runtime requests for atomic parameter/shard rebinding."""

from dataclasses import dataclass
from typing import Optional

from gefen.contracts import ShardIdentity


@dataclass(frozen=True, slots=True)
class LogicalSlotBinding:
    """Tensor-free identity for one original optimizer parameter slot."""

    group_index: int
    original_slot_index: int
    compatibility_name: str
    shard: ShardIdentity

    def __post_init__(self) -> None:
        if type(self.group_index) is not int:
            raise TypeError("LogicalSlotBinding.group_index must be an integer")
        if self.group_index < 0:
            raise ValueError("LogicalSlotBinding.group_index must be nonnegative")
        if type(self.original_slot_index) is not int:
            raise TypeError(
                "LogicalSlotBinding.original_slot_index must be an integer"
            )
        if self.original_slot_index < 0:
            raise ValueError(
                "LogicalSlotBinding.original_slot_index must be nonnegative"
            )
        if type(self.compatibility_name) is not str:
            raise TypeError("LogicalSlotBinding.compatibility_name must be a string")
        if self.compatibility_name != self.compatibility_name.lower():
            raise ValueError(
                "LogicalSlotBinding.compatibility_name must be lowercase"
            )
        if not isinstance(self.shard, ShardIdentity):
            raise TypeError("LogicalSlotBinding.shard must be a ShardIdentity")


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
