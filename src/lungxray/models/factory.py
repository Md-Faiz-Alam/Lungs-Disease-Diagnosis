"""Model zoo: the original custom CNN baseline plus ImageNet-pretrained backbones."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm


def _block(cin: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=(k, k), padding=0, bias=True), nn.ReLU(), nn.BatchNorm2d(cout)
    )


class LegacyNet(nn.Module):
    """The project's original ~11.7K-parameter CNN, kept verbatim (same layer names) so the
    v0 weights still load and it can serve as an honest baseline."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.convolution_block1 = _block(3, 8, 3)
        self.pooling11 = nn.MaxPool2d(2, 2)
        self.convolution_block2 = _block(8, 20, 3)
        self.pooling22 = nn.MaxPool2d(2, 2)
        self.convolution_block3 = _block(20, 10, 1)
        self.pooling33 = nn.MaxPool2d(2, 2)
        self.convolution_block4 = _block(10, 20, 3)
        self.convolution_block5 = _block(20, 32, 1)
        self.convolution_block6 = _block(32, 10, 3)
        self.convolution_block7 = _block(10, 10, 1)
        self.convolution_block8 = _block(10, 14, 3)
        self.convolution_block9 = _block(14, 16, 3)
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=4))
        self.convolution_block_out = nn.Sequential(nn.Conv2d(16, num_classes, kernel_size=(4, 4)))

    def forward(self, x):
        x = self.pooling11(self.convolution_block1(x))
        x = self.pooling22(self.convolution_block2(x))
        x = self.pooling33(self.convolution_block3(x))
        for i in range(4, 10):
            x = getattr(self, f"convolution_block{i}")(x)
        x = self.convolution_block_out(self.gap(x))
        # log-probabilities are valid logits: softmax(log_softmax(z)) == softmax(z)
        return F.log_softmax(x.view(-1, self.num_classes), dim=-1)


def build_model(name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    if name == "custom_cnn":
        return LegacyNet(num_classes)
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if name == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier.in_features, num_classes))
        return m
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.classifier[1].in_features, num_classes))
        return m
    if name == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")


def head_module(model: nn.Module, name: str) -> nn.Module:
    """The freshly initialised classification head (trained alone during warm-up)."""
    if name == "custom_cnn":
        return model
    if name == "resnet18":
        return model.fc
    return model.classifier


def set_backbone_trainable(model: nn.Module, name: str, trainable: bool) -> None:
    head = set(head_module(model, name).parameters())
    for p in model.parameters():
        p.requires_grad = trainable or p in head


def gradcam_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last spatial feature map, where Grad-CAM is computed."""
    return {
        "custom_cnn": lambda: model.convolution_block9,
        "resnet18": lambda: model.layer4,
        "densenet121": lambda: model.features,
        "efficientnet_b0": lambda: model.features[-1],
        "convnext_tiny": lambda: model.features[-1],
    }[name]()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
