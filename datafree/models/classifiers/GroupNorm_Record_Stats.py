import torch
import torch.nn as nn
from torch import Tensor, Size

class GroupNorm_Record_Stats(nn.BatchNorm2d):
    def __init__(
        self,
        num_groups: int, 
        num_channels: int,
        # num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super(GroupNorm_Record_Stats, self).__init__(
            num_features = num_channels,
            eps = eps,
            momentum = momentum,
            affine = affine,
            track_running_stats = track_running_stats,
            device=device,
            dtype=dtype
        )
        self.group_norm = nn.GroupNorm(num_groups = num_groups, num_channels = num_channels)

    def forward(self, input: Tensor) -> Tensor:
        super().forward(input)
        return self.group_norm.forward(input) 
