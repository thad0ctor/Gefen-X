"""Runtime binding for one optimizer-wide learned-codebook process group."""

from dataclasses import dataclass
from typing import Optional

import torch

from gefen.contracts import ProcessGroupIdentity


@dataclass(frozen=True, eq=False)
class CodebookProcessGroupBinding:
    """Bind a stable semantic group to one framework runtime group handle.

    ``process_group`` is deliberately opaque to the descriptor. Gefen validates
    and consumes it through PyTorch's public distributed APIs when the binding
    is installed by ``post_sharding``. A one-member scope uses ``None`` rather
    than implicitly selecting the default world.
    """

    identity: ProcessGroupIdentity
    local_member: str
    process_group: Optional[object]
    collective_device: torch.device

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ProcessGroupIdentity):
            raise TypeError("CodebookProcessGroupBinding.identity must be a ProcessGroupIdentity")
        if self.local_member not in self.identity.ordered_members:
            raise ValueError("CodebookProcessGroupBinding.local_member must belong to the identity")
        try:
            device = torch.device(self.collective_device)
        except (TypeError, RuntimeError) as exc:
            raise TypeError("CodebookProcessGroupBinding.collective_device must be a torch device") from exc
        if device.type == "meta":
            raise ValueError("codebook collectives require a materialized device")
        object.__setattr__(self, "collective_device", device)

    @property
    def sort_key(self):
        """Return the stable adapter scheduling key for this collective scope."""

        return (self.identity.semantic_name, self.identity.ordered_members)


__all__ = ["CodebookProcessGroupBinding"]
