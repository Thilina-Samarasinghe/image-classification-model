"""Pretrained backbones with a replaceable classification head."""
import torch.nn as nn
from torchvision import models


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


def _head(model: nn.Module, arch: str) -> nn.Module:
    return model.fc if arch == "resnet18" else model.classifier


def _last_block(model: nn.Module, arch: str) -> nn.Module:
    if arch == "resnet18":
        return model.layer4
    return model.features[-3:]  # last few feature blocks for mobilenet/efficientnet


def freeze_backbone(model: nn.Module, arch: str) -> None:
    """Stage 1: only the classification head is trainable."""
    for p in model.parameters():
        p.requires_grad = False
    for p in _head(model, arch).parameters():
        p.requires_grad = True


def unfreeze_last_block(model: nn.Module, arch: str) -> None:
    """Stage 2: also fine-tune the last backbone block."""
    for p in _last_block(model, arch).parameters():
        p.requires_grad = True


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
