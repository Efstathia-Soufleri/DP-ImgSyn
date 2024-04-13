import logging
from typing import List
from synthesizer.syn_fix import SynthesisModuleValidator
import torch.nn as nn
from synthesizer.GN_Record_Stats import GroupNorm_Record_Stats

from opacus.validators.errors import (
    ShouldReplaceModuleError,
    UnsupportedModuleError,
)
from opacus.validators.utils import register_module_fixer, register_module_validator


logger = logging.getLogger(__name__)

GROUPNORM = nn.GroupNorm

@register_module_validator(
    [nn.GroupNorm],
    validator_class=SynthesisModuleValidator
)
def validate(module: GROUPNORM) -> List[UnsupportedModuleError]:
    return [
        ShouldReplaceModuleError(
            "BatchNorm cannot support training with differential privacy. "
            "The reason for it is that BatchNorm makes each sample's normalized value "
            "depend on its peers in a batch, ie the same sample x will get normalized to "
            "a different value depending on who else is on its batch. "
            "Privacy-wise, this means that we would have to put a privacy mechanism there too. "
            "While it can in principle be done, there are now multiple normalization layers that "
            "do not have this issue: LayerNorm, InstanceNorm and their generalization GroupNorm "
            "are all privacy-safe since they don't have this property."
            "We offer utilities to automatically replace BatchNorms to GroupNorms and we will "
            "release pretrained models to help transition, such as GN-ResNet ie a ResNet using "
            "GroupNorm, pretrained on ImageNet"
        )
    ]


@register_module_fixer(
    [nn.GroupNorm],
    validator_class=SynthesisModuleValidator
)
def fix(module: GROUPNORM, **kwargs) -> GroupNorm_Record_Stats:
    logger.info(
        "Replace GroupNorm with GroupNorm_Record_Stats"
    )

    return (
        GroupNorm_Record_Stats.clone(module, kwargs['max_norm'], kwargs['noise_multiplier'])
    )

