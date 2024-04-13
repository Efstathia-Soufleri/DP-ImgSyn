import torch.nn as nn
from torch import Tensor
import torch

class GroupNorm_Record_Stats(nn.BatchNorm2d):
    def __init__(
        self,
        num_groups: int, 
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        device=None,
        dtype=None,
        max_norm=1.0,
        noise_multiplier=1.0,
        add_noise_to_bn=True,
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
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.add_noise_to_bn = add_noise_to_bn
        self.max_norm = max_norm
        self.noise_multiplier = noise_multiplier
        self.std = self.noise_multiplier * self.max_norm

    @classmethod
    def clone(cls, module:nn.GroupNorm, max_norm:float, noise_multiplier:float):
        gn_stats = GroupNorm_Record_Stats(
            num_groups=module.num_groups, 
            num_channels=module.num_channels,
            device=module.weight.data.device,
            max_norm=max_norm,
            noise_multiplier=noise_multiplier,
            add_noise_to_bn=True,
        )

        gn_stats.group_norm = module
        return gn_stats

    def forward(self, input: Tensor) -> Tensor:
        if self.add_noise_to_bn:
            # clamped_in = input.clamp(-self.max_norm, self.max_norm)
            # noise = torch.normal(
            #     mean=0,
            #     std=self.std,
            #     size=input.shape,
            #     device=input.device,
            #     generator=None,
            # )
            super().forward(input)
        else:
            # Means we are not adding noise when synthesizing data
            # should not happen but let's throw an error any way to be safe
            raise ValueError("Privacy Violation")
        return self.group_norm.forward(input) 