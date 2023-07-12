from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class CommonDepthConfig:
    loss: Optional[Literal["l2", "l1", "huber"]] = "l2"
    use_logvar: bool = False
    train_uncertainty_after: Optional[int] = None
    huber_delta: Optional[float] = 1
