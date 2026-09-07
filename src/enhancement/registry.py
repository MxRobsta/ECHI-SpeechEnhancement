from typing import Dict, Protocol, Optional, Any
import torch


class Enhancement(Protocol):
    def process_session(
        self,
        device_audio: torch.Tensor,
        device_fs: int,
        spkid_audio: torch.Tensor,
        spkid_fs: int,
        kwargs: Optional[Dict] = None,
    ) -> torch.Tensor: ...


enhancement_options: Dict[str, Enhancement] = {}


def register_enhancement(name: str):

    def decorator(enhance: Any):
        enhancement_options[name] = enhance
        return enhance

    return decorator
