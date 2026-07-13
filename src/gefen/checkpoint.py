"""Runtime process-group binding for checkpoint adapters.

The stable identity contains adapter-defined semantic member IDs. Runtime
global ranks are inspected only while validating the live PyTorch process
group; they are deliberately not part of this binding's durable identity.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from gefen.contracts import ProcessGroupIdentity


def _is_process_group(value: object) -> bool:
    if not torch.distributed.is_available():
        return False
    process_group_type = getattr(torch.distributed, "ProcessGroup", None)
    return process_group_type is not None and isinstance(value, process_group_type)


@dataclass(frozen=True, eq=False)
class CheckpointProcessGroupBinding:
    """Bind a stable checkpoint scope to one live PyTorch process group.

    Multi-member scopes always carry an explicit runtime handle, including
    ``torch.distributed.group.WORLD`` when the default world is the intended
    scope. A one-member scope is local and therefore uses ``None``. The runtime
    handle and runtime global ranks are not canonical checkpoint metadata.
    """

    identity: ProcessGroupIdentity
    local_member: str
    process_group: Optional[object]
    collective_device: torch.device

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ProcessGroupIdentity):
            raise TypeError("CheckpointProcessGroupBinding.identity must be a ProcessGroupIdentity")
        if not isinstance(self.local_member, str):
            raise TypeError("CheckpointProcessGroupBinding.local_member must be a string")
        if self.local_member not in self.identity.ordered_members:
            raise ValueError("CheckpointProcessGroupBinding.local_member must belong to the identity")
        if not isinstance(self.collective_device, torch.device):
            raise TypeError("CheckpointProcessGroupBinding.collective_device must be a torch.device")
        if self.collective_device.type not in {"cpu", "cuda"}:
            raise ValueError("checkpoint collective device must be CPU or CUDA")

        member_count = len(self.identity.ordered_members)
        if member_count == 1:
            if self.process_group is not None:
                raise ValueError("a one-member checkpoint scope must use process_group=None")
        else:
            if self.process_group is None:
                raise ValueError(
                    "a multi-member checkpoint scope requires an explicit runtime process group"
                )
            if not _is_process_group(self.process_group):
                raise TypeError(
                    "CheckpointProcessGroupBinding.process_group must be a torch.distributed.ProcessGroup"
                )

    @property
    def sort_key(self):
        """Return the rank-invariant key used to order checkpoint scopes."""

        return (self.identity.semantic_name, self.identity.ordered_members)

    def validate_runtime(self) -> None:
        """Fail unless this process belongs at the declared semantic coordinate.

        This performs only local process-group metadata queries. It does not
        execute a collective or retain the runtime group's global-rank list.
        """

        members = self.identity.ordered_members
        if len(members) == 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "a multi-member checkpoint scope requires initialized torch.distributed"
            )

        import torch.distributed as dist

        try:
            world_size = dist.get_world_size(self.process_group)
            runtime_global_members = tuple(dist.get_process_group_ranks(self.process_group))
            current_global_rank = dist.get_rank()
            group_rank = dist.get_group_rank(self.process_group, current_global_rank)
            roundtrip_global_rank = dist.get_global_rank(self.process_group, group_rank)
            backend = dist.get_backend(self.process_group)
        except Exception as exc:
            raise ValueError(
                "the current rank must belong to the explicit checkpoint process group"
            ) from exc

        if world_size != len(members) or len(runtime_global_members) != world_size:
            raise ValueError(
                "checkpoint process-group world size does not match its stable identity"
            )
        if (
            group_rank < 0
            or group_rank >= world_size
            or runtime_global_members[group_rank] != current_global_rank
            or roundtrip_global_rank != current_global_rank
        ):
            raise ValueError(
                "the current global rank is not a consistent member of the checkpoint process group"
            )
        if members[group_rank] != self.local_member:
            raise ValueError(
                "runtime checkpoint group order does not match ordered semantic members"
            )
        self._validate_collective_device(backend)

    def _validate_collective_device(self, backend: object) -> None:
        backend_name = str(backend).lower()
        if "nccl" in backend_name:
            if self.collective_device.type != "cuda":
                raise ValueError(
                    "checkpoint collective device is incompatible with the runtime backend"
                )
        elif "gloo" in backend_name or "mpi" in backend_name:
            if self.collective_device.type != "cpu":
                raise ValueError(
                    "checkpoint collective device is incompatible with the runtime backend"
                )

        if self.collective_device.type == "cuda":
            index = self.collective_device.index
            if (
                not torch.cuda.is_available()
                or index is None
                or index < 0
                or index >= torch.cuda.device_count()
            ):
                raise ValueError("checkpoint collective CUDA device is unavailable")


__all__ = ["CheckpointProcessGroupBinding"]
