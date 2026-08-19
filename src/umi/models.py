from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, p_drop: float, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, cout), cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, cout), cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallCNN(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        width: int = 32,
        p_drop: float = 0.3,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        chans = [in_channels] + [width * (2**i) for i in range(n_blocks)]
        self.features = nn.Sequential(
            *[ConvBlock(chans[i], chans[i + 1], p_drop) for i in range(n_blocks)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(chans[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(128, n_classes),
        )
        self.p_drop = p_drop
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)))

    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class ResNet18Small(nn.Module):

    def __init__(
        self, in_channels: int = 1, n_classes: int = 2, p_drop: float = 0.3
    ) -> None:
        super().__init__()
        from torchvision.models import resnet18  # noqa: PLC0415

        net = resnet18(weights=None)
        net.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Identity()
        self.backbone = net
        self.head = nn.Sequential(nn.Dropout(p_drop), nn.Linear(512, n_classes))
        # Inject dropout between residual stages so MC sampling perturbs depth-wise.
        for layer in (net.layer2, net.layer3, net.layer4):
            layer.add_module("mc_dropout", nn.Dropout2d(p_drop))
        self.p_drop = p_drop
        self.n_classes = n_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(
    arch: str = "smallcnn",
    in_channels: int = 1,
    n_classes: int = 2,
    p_drop: float = 0.3,
    width: int = 32,
) -> nn.Module:
    if arch == "smallcnn":
        return SmallCNN(in_channels, n_classes, width=width, p_drop=p_drop)
    if arch == "resnet18":
        return ResNet18Small(in_channels, n_classes, p_drop=p_drop)


def enable_mc_dropout(model: nn.Module) -> int:

    model.eval()
    n = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            if getattr(module, "p", 0.0) > 0.0:
                module.train()
                n += 1

    return n
