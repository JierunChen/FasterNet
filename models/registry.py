import torch.nn as nn
from timm.models.registry import register_model
from .fasternet import FasterNet


@register_model
def fasternet(**kwargs):
    model = FasterNet(**kwargs)
    return model